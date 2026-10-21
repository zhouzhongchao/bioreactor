# -*- coding: utf-8 -*-
"""Fit the real-hardware forward model used by MPC.

The training source is exclusively ``logs/pure_mpc_data/*.csv``.  Files and
excitation cases are kept as independent time series, then resampled to the
MPC period (0.05 s).  No rigid-body or PAM model is imported by this script.
"""
from __future__ import print_function

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy.signal import savgol_filter
import torch
from torch import nn


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from project_paths import CONTROL_LOG_DIR
DATA_GLOB = str(CONTROL_LOG_DIR / "pure_mpc_data" / "pure_mpc_*.csv")
CONTROL_GLOB = str(CONTROL_LOG_DIR / "mpc_*.csv")
MODEL_PATH = os.path.join(HERE, "pure_mpc_model.pt")
META_PATH = os.path.join(HERE, "pure_mpc_model.json")
DT = 0.05
NCH = 6
SEED = 20260906


def _smooth_q(q):
    """About 0.5 s Savitzky-Golay smoothing at the configured sample time."""
    wanted = max(5, int(round(0.5 / DT)) | 1)
    window = min(wanted, len(q) if len(q) % 2 else len(q) - 1)
    return savgol_filter(q, window, 2, axis=0, mode="interp") if window >= 5 else q


def _read_blocks(paths):
    blocks = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for row in rows:
            if row.get("phase") != "run":
                continue
            groups.setdefault(row.get("case_name", "unknown"), []).append(row)
        for case_name, group in groups.items():
            try:
                group.sort(key=lambda r: float(r["t"]))
                t = np.asarray([float(r["t"]) for r in group], dtype=np.float64)
                q = np.asarray([[float(r["q%d_rad" % i]) for i in (1, 2, 3)]
                                for r in group], dtype=np.float64)
                p = np.asarray([[float(r["P%d_kpa" % i]) for i in range(NCH)]
                                for r in group], dtype=np.float64)
                u = np.asarray([[float(r["u_actual%d" % i]) for i in range(NCH)]
                                for r in group], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            finite = (np.isfinite(t) & np.all(np.isfinite(q), axis=1) &
                      np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(u), axis=1))
            t, q, p, u = t[finite], q[finite], p[finite], u[finite]
            if len(t) < 6:
                continue
            # Duplicate frames carry no new dynamics and break interpolation.
            keep = np.r_[True, np.diff(t) > 1e-6]
            t, q, p, u = t[keep], q[keep], p[keep], u[keep]
            if len(t) < 6 or t[-1] - t[0] < 0.4:
                continue
            tg = np.arange(t[0], t[-1] + 1e-9, DT)
            if len(tg) < 5:
                continue
            qg = np.column_stack([np.interp(tg, t, q[:, j]) for j in range(3)])
            pg = np.column_stack([np.interp(tg, t, p[:, j]) for j in range(NCH)])
            # Voltage commands are zero-order held, never linearly interpolated.
            ui = np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 1)
            ug = u[ui]
            # IMU quantisation makes raw finite differences too noisy for a
            # velocity state.  A 0.5 s local filter keeps the commanded motion.
            qg = _smooth_q(qg)
            blocks.append({"file": os.path.basename(path), "case": case_name,
                           "t": tg, "q": qg, "p": pg, "u": ug})
    return blocks


def _read_control_blocks(paths):
    """Read completed pure-MPC runs as low-volume closed-loop supplements."""
    blocks = []
    for path in paths:
        if path.endswith("_laps.csv"):
            continue
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            required = (["t", "q1", "q2", "q3"] +
                        ["P%d" % i for i in range(NCH)] +
                        ["uact%d" % i for i in range(NCH)])
            if len(rows) < 5 or not all(key in rows[0] for key in required):
                continue
            t = np.asarray([float(r["t"]) for r in rows], dtype=np.float64)
            q = np.radians(np.asarray([[float(r["q%d" % i]) for i in (1, 2, 3)]
                                       for r in rows], dtype=np.float64))
            p = np.asarray([[float(r["P%d" % i]) for i in range(NCH)]
                            for r in rows], dtype=np.float64)
            u = np.asarray([[float(r["uact%d" % i]) for i in range(NCH)]
                            for r in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        finite = (np.isfinite(t) & np.all(np.isfinite(q), axis=1) &
                  np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(u), axis=1))
        t, q, p, u = t[finite], q[finite], p[finite], u[finite]
        if len(t) < 5:
            continue
        # Only supplement with logs recorded at the current MPC period; old
        # 10 Hz runs must not be treated as 20 Hz transitions.
        dt_log = float(np.median(np.diff(t)))
        if abs(dt_log - DT) > 0.0125:
            continue
        cuts = list(np.flatnonzero((np.diff(t) <= 0.0) | (np.diff(t) > 1.7*DT)) + 1)
        for ia, ib in zip([0] + cuts, cuts + [len(t)]):
            if ib - ia < 5:
                continue
            qs = q[ia:ib].copy()
            qs = _smooth_q(qs)
            blocks.append({"file": os.path.basename(path), "case": "closed_loop",
                           "t": t[ia:ib], "q": qs, "p": p[ia:ib], "u": u[ia:ib]})
    return blocks


def _samples(blocks):
    z, target = [], []
    for b in blocks:
        q, p, u = b["q"], b["p"], b["u"]
        dq = np.diff(q, axis=0)
        for k in range(1, len(q) - 1):
            z.append(np.r_[q[k], dq[k - 1], p[k], u[k]])
            # Predict acceleration (change of per-tick displacement), not the
            # whole next velocity.  The explicit velocity residual greatly
            # reduces long-horizon drift.
            target.append(np.r_[dq[k] - dq[k - 1], p[k + 1] - p[k]])
    return np.asarray(z, np.float32), np.asarray(target, np.float32)


def _zero_pressure(z):
    quiet = ((np.max(np.abs(z[:, 12:18]), axis=1) < 1.0) &
             (np.max(z[:, 6:12], axis=1) < 12.0))
    if np.count_nonzero(quiet) < 20:
        return np.percentile(z[:, 6:12], 5, axis=0).astype(np.float32)
    return np.median(z[quiet, 6:12], axis=0).astype(np.float32)


class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
            nn.Linear(96, 64), nn.SiLU(),
            nn.Linear(64, 9))

    def forward(self, z):
        return self.net(z)


class RuntimeDynamics(nn.Module):
    """TorchScript artifact: physical x,u in, physical next state out."""
    def __init__(self, core, z_mean, z_std, y_mean, y_std, p_zero):
        super(RuntimeDynamics, self).__init__()
        self.core = core
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.register_buffer("p_zero", torch.as_tensor(p_zero, dtype=torch.float32))

    def forward(self, x, u):
        z = torch.cat((x, u), dim=-1)
        y = self.core((z - self.z_mean) / self.z_std) * self.y_std + self.y_mean
        # Make the measured resting state an exact equilibrium.  Otherwise a
        # tiny learned bias is amplified by the MPC's 20-step rollout.
        z0 = torch.cat((torch.zeros_like(x[..., :6]),
                        self.p_zero.expand_as(x[..., 6:]), torch.zeros_like(u)), dim=-1)
        y0 = self.core((z0 - self.z_mean) / self.z_std) * self.y_std + self.y_mean
        y = y - y0
        dq_next = torch.clamp(x[..., 3:6] + y[..., :3], -0.30, 0.30)
        p_next = torch.clamp(x[..., 6:] + y[..., 3:], 0.0, 900.0)
        return torch.cat((x[..., :3] + dq_next, dq_next, p_next), dim=-1)


def _fit_core(z, y, zv, yv, max_epochs=1200):
    z_mean, z_std = z.mean(0), z.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    z_std = np.maximum(z_std, np.asarray([0.02] * 3 + [0.005] * 3 +
                                         [5.0] * 6 + [100.0] * 6, np.float32))
    y_std = np.maximum(y_std, np.asarray([0.002] * 3 + [1.0] * 6, np.float32))
    zt = torch.from_numpy((z - z_mean) / z_std)
    yt = torch.from_numpy((y - y_mean) / y_std)
    zvt = torch.from_numpy((zv - z_mean) / z_std)
    yvt = torch.from_numpy((yv - y_mean) / y_std)
    core = Core()
    opt = torch.optim.AdamW(core.parameters(), lr=1e-3, weight_decay=2e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    batch = min(512, len(zt))
    for epoch in range(1, max_epochs + 1):
        core.train()
        order = torch.randperm(len(zt))
        for start in range(0, len(order), batch):
            ix = order[start:start + batch]
            loss = loss_fn(core(zt[ix]), yt[ix])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 5.0)
            opt.step()
        core.eval()
        with torch.no_grad():
            vl = float(torch.mean((core(zvt) - yvt) ** 2).item())
        if vl < best - 1e-6:
            best, best_epoch, stale = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
        else:
            stale += 1
        if epoch % 100 == 0:
            print("# epoch %d  validation normalized-MSE %.5f" % (epoch, vl), flush=True)
        if epoch >= 250 and stale >= 180:
            break
    core.load_state_dict(best_state)
    return core, z_mean, z_std, y_mean, y_std, best_epoch


def _metrics(runtime, blocks):
    angle_one, pressure_one, angle_roll, pressure_roll = [], [], [], []
    runtime.eval()
    with torch.no_grad():
        for b in blocks:
            q, p, u = b["q"], b["p"], b["u"]
            if len(q) < 4:
                continue
            dq = np.diff(q, axis=0)
            x = np.r_[q[1], dq[0], p[1]].astype(np.float32)
            for k in range(1, len(q) - 1):
                true_x = np.r_[q[k + 1], dq[k], p[k + 1]].astype(np.float32)
                teacher_x = np.r_[q[k], dq[k - 1], p[k]].astype(np.float32)
                one = runtime(torch.from_numpy(teacher_x[None]),
                              torch.from_numpy(u[k:k + 1].astype(np.float32)))[0].numpy()
                angle_one.append(np.sum((one[:3] - true_x[:3]) ** 2))
                pressure_one.append(np.mean((one[6:] - true_x[6:]) ** 2))
                x = runtime(torch.from_numpy(x[None]),
                            torch.from_numpy(u[k:k + 1].astype(np.float32)))[0].numpy()
                angle_roll.append(np.sum((x[:3] - true_x[:3]) ** 2))
                pressure_roll.append(np.mean((x[6:] - true_x[6:]) ** 2))
    return {
        "one_step_angle_rmse_deg": math.degrees(math.sqrt(float(np.mean(angle_one)))),
        "one_step_pressure_rmse_kpa": math.sqrt(float(np.mean(pressure_one))),
        "block_rollout_angle_rmse_deg": math.degrees(math.sqrt(float(np.mean(angle_roll)))),
        "block_rollout_pressure_rmse_kpa": math.sqrt(float(np.mean(pressure_roll))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1200)
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    paths = []
    for path in sorted(glob.glob(DATA_GLOB)):
        meta_path = os.path.splitext(path)[0] + ".json"
        try:
            with open(meta_path, encoding="utf-8") as f:
                collection_meta = json.load(f)
            if bool(collection_meta.get("partial", False)):
                continue
        except Exception:
            # Old complete files may predate metadata; their CSV integrity is
            # still checked while parsing.
            pass
        paths.append(path)
    if len(paths) < 3:
        raise SystemExit("至少需要3个完整 pure_mpc CSV 文件")
    blocks = _read_blocks(paths)
    # Only the latest three completed pure-MPC trials are added.  They remain
    # a small supplement to deliberate excitation data and cannot dominate it.
    control_paths = sorted(
        [p for p in glob.glob(CONTROL_GLOB) if not p.endswith("_laps.csv")],
        key=os.path.getmtime)[-3:]
    control_blocks = _read_control_blocks(control_paths)
    # Hold out complete collection files; adjacent frames never leak across sets.
    val_files = set(os.path.basename(p) for p in paths[-max(2, len(paths) // 5):])
    train_blocks = [b for b in blocks if b["file"] not in val_files]
    val_blocks = [b for b in blocks if b["file"] in val_files]
    # Keep the newest control trial honest during model selection; refit on all
    # three only after selecting the epoch count.
    if control_blocks:
        train_blocks.extend(control_blocks[:-1])
        val_blocks.append(control_blocks[-1])
    z, y = _samples(train_blocks); zv, yv = _samples(val_blocks)
    if len(z) < 500 or len(zv) < 100:
        raise SystemExit("有效连续样本不足，不能可靠拟合")
    print("# 纯数据模型：%d文件，训练%d拍/%d段，验证%d拍/%d段" %
          (len(paths), len(z), len(train_blocks), len(zv), len(val_blocks)), flush=True)
    core, zm, zs, ym, ys, best_epoch = _fit_core(z, y, zv, yv, args.epochs)
    p_zero = _zero_pressure(z)
    validation_runtime = RuntimeDynamics(core, zm, zs, ym, ys, p_zero)
    metrics = _metrics(validation_runtime, val_blocks)
    print("# 留出验证：1拍角度 %.2f°，压力 %.2fkPa；工况自由滚动角度 %.2f°，压力 %.2fkPa" %
          (metrics["one_step_angle_rmse_deg"], metrics["one_step_pressure_rmse_kpa"],
           metrics["block_rollout_angle_rmse_deg"], metrics["block_rollout_pressure_rmse_kpa"]),
          flush=True)

    # After honest hold-out validation, refit the deployed artifact on every run
    # for the selected epoch count so none of the expensive hardware data is lost.
    all_blocks = blocks + control_blocks
    za, ya = _samples(all_blocks)
    zma, zsa = za.mean(0), za.std(0); yma, ysa = ya.mean(0), ya.std(0)
    zsa = np.maximum(zsa, np.asarray([0.02] * 3 + [0.005] * 3 +
                                     [5.0] * 6 + [100.0] * 6, np.float32))
    ysa = np.maximum(ysa, np.asarray([0.002] * 3 + [1.0] * 6, np.float32))
    zat = torch.from_numpy((za - zma) / zsa); yat = torch.from_numpy((ya - yma) / ysa)
    final_core = Core(); opt = torch.optim.AdamW(final_core.parameters(), lr=1e-3, weight_decay=2e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5); batch = min(512, len(zat))
    for epoch in range(max(1, best_epoch)):
        order = torch.randperm(len(zat)); final_core.train()
        for start in range(0, len(order), batch):
            ix = order[start:start + batch]
            loss = loss_fn(final_core(zat[ix]), yat[ix])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(final_core.parameters(), 5.0); opt.step()
    p_zero_all = _zero_pressure(za)
    runtime = RuntimeDynamics(final_core.eval(), zma, zsa, yma, ysa, p_zero_all).eval()
    example_x = torch.zeros((1, 12), dtype=torch.float32)
    example_u = torch.zeros((1, 6), dtype=torch.float32)
    traced = torch.jit.trace(runtime, (example_x, example_u))
    tmp_model = MODEL_PATH + ".tmp"
    torch.jit.save(traced, tmp_model); os.replace(tmp_model, MODEL_PATH)
    digest = hashlib.sha256()
    for path in paths + control_paths:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk: break
                digest.update(chunk)
    meta = {
        "schema": 1, "kind": "pure_data_neural_state_space", "sample_dt": DT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source_glob": DATA_GLOB,
        "source_files": [os.path.basename(p) for p in paths], "source_sha256": digest.hexdigest(),
        "source_file_count": len(paths), "source_block_count": len(blocks),
        "closed_loop_supplement_files": [os.path.basename(p) for p in control_paths],
        "closed_loop_supplement_blocks": len(control_blocks),
        "training_samples": int(len(za)), "validation_files": sorted(val_files),
        "validation_metrics": metrics, "selected_epoch": int(best_epoch),
        "input_state": ["q1_rad", "q2_rad", "q3_rad", "dq1_per_tick", "dq2_per_tick",
                        "dq3_per_tick"] + ["P%d_kpa" % i for i in range(6)],
        "input_control": ["u_actual%d_mv" % i for i in range(6)],
        "network": "18-96-96-64-9 SiLU acceleration/pressure residual dynamics",
        "rest_pressure_kpa": [float(v) for v in p_zero_all],
        "uses_physics_model": False,
    }
    tmp_meta = META_PATH + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, META_PATH)
    print("# 已生成：%s" % MODEL_PATH)
    print("# 元数据：%s" % META_PATH)


if __name__ == "__main__":
    main()

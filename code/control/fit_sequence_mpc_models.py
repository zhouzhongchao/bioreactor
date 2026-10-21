# -*- coding: utf-8 -*-
"""Fit two sequence-aware residual forward models from pure-MPC hardware data.

Both artifacts predict a correction to the existing, frozen one-step MLP.  The
runtime contract is::

    correction = model(x_history[B, 10, 12], u_history[B, 10, 6])

The first three correction outputs modify next-tick displacement; the last six
modify next pressure.  Keeping the trusted base MLP in the runtime adapter
makes the sequence models safe fallbacks: deleting either artifact restores the
original model without changing any other controller file.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fit_pure_mpc_model import DATA_GLOB, DT, NCH, _read_blocks


HERE = Path(__file__).resolve().parent
BASE_MODEL_PATH = HERE / "pure_mpc_model.pt"
HISTORY = 10
ROLLOUT = 24
SEED = 20260908

ARTIFACTS = {
    "history_mlp": (HERE / "pure_mpc_history_mlp.pt",
                    HERE / "pure_mpc_history_mlp.json"),
    "gru": (HERE / "pure_mpc_gru.pt", HERE / "pure_mpc_gru.json"),
}


class HistoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HISTORY * 18, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 9),
        )

    def forward(self, z):
        return self.net(z.flatten(1))


class HistoryGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(18, 64, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 9))

    def forward(self, z):
        sequence, _ = self.gru(z)
        return self.head(sequence[:, -1])


class CorrectionRuntime(nn.Module):
    """TorchScript wrapper: raw physical histories in, physical correction out."""

    def __init__(self, core, z_mean, z_std, y_mean, y_std):
        super().__init__()
        self.core = core
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

    def forward(self, x_history, u_history):
        z = torch.cat((x_history, u_history), dim=-1)
        normalized = (z - self.z_mean) / self.z_std
        return self.core(normalized) * self.y_std + self.y_mean


class ZeroCorrection(nn.Module):
    """Evaluate the frozen single-step MLP on exactly the same 24-step blocks."""

    def forward(self, x_history, u_history):
        return torch.zeros((x_history.shape[0], 9), dtype=x_history.dtype,
                           device=x_history.device)


def _states_for_block(block):
    q, p, u = block["q"], block["p"], block["u"]
    dq = np.diff(q, axis=0)
    # state[k] corresponds to the physical sample k+1.
    states = np.concatenate((q[1:], dq, p[1:]), axis=1).astype(np.float32)
    controls = u[1:].astype(np.float32)
    return states, controls


def _sequence_samples(blocks):
    histories_x, histories_u, targets = [], [], []
    for block in blocks:
        states, controls = _states_for_block(block)
        for end in range(HISTORY - 1, len(states) - 1):
            start = end - HISTORY + 1
            histories_x.append(states[start:end + 1])
            histories_u.append(controls[start:end + 1])
            targets.append(states[end + 1])
    return (np.asarray(histories_x, np.float32),
            np.asarray(histories_u, np.float32),
            np.asarray(targets, np.float32))


def _base_predict(base, x, u, batch=2048):
    answers = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            stop = start + batch
            answers.append(base(torch.from_numpy(x[start:stop]),
                                torch.from_numpy(u[start:stop])).cpu().numpy())
    return np.concatenate(answers).astype(np.float32)


def _correction_targets(base, hx, hu, target):
    base_next = _base_predict(base, hx[:, -1], hu[:, -1])
    return np.concatenate((target[:, 3:6] - base_next[:, 3:6],
                           target[:, 6:] - base_next[:, 6:]), axis=1).astype(np.float32)


def _fit(architecture, hx, hu, y, hxv, huv, yv, max_epochs):
    z = np.concatenate((hx, hu), axis=2)
    zv = np.concatenate((hxv, huv), axis=2)
    z_mean = z.reshape(-1, 18).mean(0).astype(np.float32)
    z_std = z.reshape(-1, 18).std(0).astype(np.float32)
    z_std = np.maximum(z_std, np.asarray(
        [0.02] * 3 + [0.003] * 3 + [5.0] * 6 + [100.0] * 6,
        dtype=np.float32))
    y_mean, y_std = y.mean(0).astype(np.float32), y.std(0).astype(np.float32)
    y_std = np.maximum(y_std, np.asarray([0.001] * 3 + [0.5] * 6, np.float32))

    zt = torch.from_numpy((z - z_mean) / z_std)
    yt = torch.from_numpy((y - y_mean) / y_std)
    zvt = torch.from_numpy((zv - z_mean) / z_std)
    yvt = torch.from_numpy((yv - y_mean) / y_std)
    core = HistoryMLP() if architecture == "history_mlp" else HistoryGRU()
    optimizer = torch.optim.AdamW(core.parameters(), lr=5.0e-4, weight_decay=1.0e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    generator = torch.Generator().manual_seed(SEED + (0 if architecture == "history_mlp" else 1))
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    batch_size = min(256, len(zt))

    for epoch in range(1, max_epochs + 1):
        core.train()
        order = torch.randperm(len(zt), generator=generator)
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            loss = loss_fn(core(zt[chosen]), yt[chosen])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 2.0)
            optimizer.step()
        core.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((core(zvt) - yvt) ** 2))
        if val_loss < best_loss - 1.0e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(core.state_dict())
        else:
            stale += 1
        if epoch % 50 == 0:
            print(f"# {architecture} epoch {epoch}: validation normalized-MSE={val_loss:.6f}",
                  flush=True)
        if epoch >= 100 and stale >= 80:
            break

    if best_state is None:
        raise RuntimeError(f"{architecture} did not produce a finite checkpoint")
    core.load_state_dict(best_state)
    core.eval()
    runtime = CorrectionRuntime(core, z_mean, z_std, y_mean, y_std).eval()
    return runtime, best_epoch, best_loss


def _predict(runtime, base, hx, hu):
    with torch.no_grad():
        correction = runtime(torch.from_numpy(hx[None].astype(np.float32)),
                             torch.from_numpy(hu[None].astype(np.float32)))[0]
        base_next = base(torch.from_numpy(hx[-1:]), torch.from_numpy(hu[-1:]))[0]
        dq = torch.clamp(base_next[3:6] + correction[:3], -0.30, 0.30)
        pressure = torch.clamp(base_next[6:] + correction[3:], 0.0, 900.0)
        answer = torch.cat((torch.from_numpy(hx[-1, :3]) + dq, dq, pressure))
    return answer.numpy().astype(np.float32)


def _metrics(runtime, base, blocks):
    one_angle, one_pressure, moving_angle = [], [], []
    roll_angle, roll_pressure = [], []
    for block in blocks:
        states, controls = _states_for_block(block)
        if len(states) <= HISTORY:
            continue
        for end in range(HISTORY - 1, len(states) - 1):
            hx = states[end - HISTORY + 1:end + 1]
            hu = controls[end - HISTORY + 1:end + 1]
            prediction = _predict(runtime, base, hx, hu)
            target = states[end + 1]
            angle_sq = float(np.sum((prediction[:3] - target[:3]) ** 2))
            one_angle.append(angle_sq)
            one_pressure.append(float(np.mean((prediction[6:] - target[6:]) ** 2)))
            actual_step = math.degrees(float(np.linalg.norm(target[:3] - states[end, :3])))
            if actual_step > 0.05:
                moving_angle.append(angle_sq)

        for start in range(HISTORY - 1, len(states) - 1, ROLLOUT):
            hx = states[start - HISTORY + 1:start + 1].copy()
            hu = controls[start - HISTORY + 1:start + 1].copy()
            stop = min(len(states) - 1, start + ROLLOUT)
            for end in range(start, stop):
                hu[-1] = controls[end]
                prediction = _predict(runtime, base, hx, hu)
                target = states[end + 1]
                roll_angle.append(float(np.sum((prediction[:3] - target[:3]) ** 2)))
                roll_pressure.append(float(np.mean((prediction[6:] - target[6:]) ** 2)))
                hx = np.concatenate((hx[1:], prediction[None]), axis=0)
                next_control = controls[min(end + 1, len(controls) - 1)]
                hu = np.concatenate((hu[1:], next_control[None]), axis=0)

    def angle_rmse(values):
        return math.degrees(math.sqrt(float(np.mean(values)))) if values else math.inf

    return {
        "one_step_angle_rmse_deg": angle_rmse(one_angle),
        "moving_one_step_angle_rmse_deg": angle_rmse(moving_angle),
        "one_step_pressure_rmse_kpa": math.sqrt(float(np.mean(one_pressure))),
        "rollout_24_angle_rmse_deg": angle_rmse(roll_angle),
        "rollout_24_pressure_rmse_kpa": math.sqrt(float(np.mean(roll_pressure))),
        "one_step_samples": len(one_angle),
        "moving_samples": len(moving_angle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--architecture", choices=("all", "history_mlp", "gru"), default="all")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    paths = sorted(glob.glob(DATA_GLOB))
    if len(paths) < 4:
        raise SystemExit(f"need at least four pure-MPC files, found {len(paths)}")
    validation_paths = paths[-2:]
    training_paths = paths[:-2]
    train_blocks = _read_blocks(training_paths)
    validation_blocks = _read_blocks(validation_paths)
    hx, hu, target = _sequence_samples(train_blocks)
    hxv, huv, targetv = _sequence_samples(validation_blocks)
    if len(hx) < 100 or len(hxv) < 30:
        raise SystemExit(f"not enough sequence samples: train={len(hx)}, validation={len(hxv)}")

    base = torch.jit.load(str(BASE_MODEL_PATH), map_location="cpu").eval()
    base_metrics = _metrics(ZeroCorrection(), base, validation_blocks)
    y = _correction_targets(base, hx, hu, target)
    yv = _correction_targets(base, hxv, huv, targetv)
    selected = ARTIFACTS if args.architecture == "all" else {
        args.architecture: ARTIFACTS[args.architecture]}
    print(f"# sequence data: train={len(hx)}, validation={len(hxv)}, history={HISTORY}")
    print(f"# same-data base MLP: one-step={base_metrics['moving_one_step_angle_rmse_deg']:.3f}deg, "
          f"24-step={base_metrics['rollout_24_angle_rmse_deg']:.3f}deg")

    for architecture, (model_path, meta_path) in selected.items():
        started = time.time()
        runtime, best_epoch, best_loss = _fit(
            architecture, hx, hu, y, hxv, huv, yv, args.epochs)
        metrics = _metrics(runtime, base, validation_blocks)
        scripted = torch.jit.script(runtime)
        tmp = model_path.with_suffix(model_path.suffix + ".tmp")
        torch.jit.save(scripted, str(tmp))
        os.replace(tmp, model_path)
        meta = {
            "schema": 1,
            "kind": "history_residual_forward_model",
            "architecture": architecture,
            "sample_dt": DT,
            "history_steps": HISTORY,
            "history_seconds": HISTORY * DT,
            "base_model": BASE_MODEL_PATH.name,
            "training_files": [os.path.basename(p) for p in training_paths],
            "validation_files": [os.path.basename(p) for p in validation_paths],
            "training_samples": len(hx),
            "validation_samples": len(hxv),
            "selected_epoch": best_epoch,
            "validation_normalized_mse": best_loss,
            "base_validation_metrics": base_metrics,
            "validation_metrics": metrics,
        }
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with tmp_meta.open("w", encoding="utf-8") as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, meta_path)
        print(f"# {architecture} saved: {model_path.name}; "
              f"one-step={metrics['moving_one_step_angle_rmse_deg']:.3f}deg, "
              f"24-step={metrics['rollout_24_angle_rmse_deg']:.3f}deg, "
              f"elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

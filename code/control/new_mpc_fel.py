# -*- coding: utf-8 -*-
"""Safe active/shadow forward-model learning for the ``new_mpc_fel`` mode.

The public contract is deliberately explicit::

    x_hat_next = model(x_t, u_actual_t)

Both the active model and the shadow model return the complete 12-dimensional
next state ``[q_next(3), dq_next_per_tick(3), pressure_next(6)]``.  The active
model is read-only while a trajectory round is running.  Only the shadow model
is optimized, and it can replace the active model only at a round boundary
after a validation win.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE_MODEL_PATH = HERE / "pure_mpc_model.pt"
BASE_META_PATH = HERE / "pure_mpc_model.json"
ACTIVE_MODEL_PATH = HERE / "new_mpc_fel_active.pt"
SHADOW_MODEL_PATH = HERE / "new_mpc_fel_shadow.pt"
PREVIOUS_MODEL_PATH = HERE / "new_mpc_fel_previous.pt"
STATE_PATH = HERE / "new_mpc_fel_state.json"

SCHEMA = 1
STATE_DIM = 12
CONTROL_DIM = 6
DT = 0.05
MIN_ROUND_SAMPLES = 40
# Accumulate complete hardware laps without touching the shadow weights.
# Every three valid laps, train once on the pooled data.
LEARNING_RATE = 1.0e-4
REPLAY_ROUNDS = 3
REPLAY_EPOCHS = 500
REPLAY_BATCH_SIZE = 64
REPLAY_PATIENCE = 50
WEIGHT_DECAY = 1.0e-6
ANCHOR_WEIGHT = 2.0e-4
MIN_ANGLE_IMPROVEMENT = 0.08
REQUIRED_CONSECUTIVE_WINS = 1
ROLLOUT_HORIZON = 24
# Tracking error does not replace the physical x(t+1) target.  It only makes
# transitions collected while the hardware is tracking poorly more important.
TRACK_WEIGHT_SCALE_DEG = 5.0
TRACK_WEIGHT_MAX = 4.0
# A newly promoted model is used for one complete hardware lap first.  Keep it
# only when its raw-reference MSE is no more than 5% above the old-model
# baseline; otherwise restore the previous active artifact automatically.
CANARY_MAX_MSE_RATIO = 1.05
TRACKING_BASELINE_ROUNDS = 3
# 顺模型能否用于控制，必须看真实运动段；静止/全零段只能用于检查漂移，
# 不能再把运动段的角度误差平均小。0.05°/拍用于排除 IMU 量化抖动。
ANGLE_MOTION_THRESHOLD_DEG = 0.05
PRESSURE_ACTIVITY_THRESHOLD_KPA = 0.5
CONTROL_ACTIVITY_THRESHOLD_MV = 50.0

# All loss components are dimensionless after division by these physical
# scales.  This prevents six pressure outputs from overwhelming the angles.
LOSS_SCALE = np.asarray(
    [0.02] * 3 +       # q: rad (~1.15 deg)
    [0.0025] * 3 +     # dq per 0.05 s tick
    [5.0] * 6,         # pressure: kPa
    dtype=np.float32,
)

# The ten-tick models output only the correction [dq_next(3), P_next(6)]
# applied on top of the frozen fitted one-step model.
SEQUENCE_LOSS_SCALE = np.asarray([0.0025] * 3 + [5.0] * 6, dtype=np.float32)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class ShadowForwardLearner:
    """Own the independent active and shadow forward models for one mode."""

    def __init__(
        self,
        base_model_path: os.PathLike | str = BASE_MODEL_PATH,
        active_model_path: os.PathLike | str = ACTIVE_MODEL_PATH,
        shadow_model_path: os.PathLike | str = SHADOW_MODEL_PATH,
        previous_model_path: os.PathLike | str = PREVIOUS_MODEL_PATH,
        state_path: os.PathLike | str = STATE_PATH,
    ):
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.base_path = Path(base_model_path)
        self.active_path = Path(active_model_path)
        self.shadow_path = Path(shadow_model_path)
        self.previous_path = Path(previous_model_path)
        self.state_path = Path(state_path)
        self.replay_path = self.state_path.with_name(
            self.state_path.stem + "_replay.npz")
        if not self.base_path.exists():
            raise FileNotFoundError(f"基础顺模型不存在：{self.base_path}")

        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        # Adaptive artifacts learned at another control period are not valid:
        # dq is stored per tick and x_hat_next represents exactly one tick.
        # Reset them to the selected base model when the time base changes.
        reset_for_timebase = False
        try:
            with self.state_path.open(encoding="utf-8") as stream:
                saved_state = json.load(stream)
            reset_for_timebase = abs(float(saved_state.get("sample_dt", -1.0)) - DT) > 1e-9
        except Exception:
            reset_for_timebase = self.state_path.exists()
        if reset_for_timebase:
            shutil.copy2(self.base_path, self.active_path)
            shutil.copy2(self.base_path, self.shadow_path)
            shutil.copy2(self.base_path, self.previous_path)
            try:
                self.replay_path.unlink()
            except FileNotFoundError:
                pass
        if not self.active_path.exists():
            shutil.copy2(self.base_path, self.active_path)
        if not self.shadow_path.exists():
            shutil.copy2(self.active_path, self.shadow_path)

        try:
            self.active = torch.jit.load(str(self.active_path), map_location="cpu").eval()
            self.shadow = torch.jit.load(str(self.shadow_path), map_location="cpu").train()
            self._assert_contract(self.active)
            self._assert_contract(self.shadow)
        except Exception:
            # A partially written/corrupt adaptive artifact must never prevent
            # control.  Restore both models from the validated base artifact.
            shutil.copy2(self.base_path, self.active_path)
            shutil.copy2(self.base_path, self.shadow_path)
            self.active = torch.jit.load(str(self.active_path), map_location="cpu").eval()
            self.shadow = torch.jit.load(str(self.shadow_path), map_location="cpu").train()
            self._assert_contract(self.active)
            self._assert_contract(self.shadow)

        self.state = self._load_state()
        self.round_samples: list[tuple] = []
        self.replay_rounds = self._load_replay()
        self.rounds_seen = int(self.state.get("rounds_seen", 0))
        self.state["rounds_seen"] = self.rounds_seen
        _atomic_json(self.state_path, self.state)
        self.pending: dict | None = None
        self.run_rows: list[dict] = []
        self.warning_streak = 0
        self.failure_streak = 0
        self.last_observation: dict | None = None

    def _assert_contract(self, model) -> None:
        torch = self.torch
        with torch.no_grad():
            output = model(torch.zeros((1, STATE_DIM)), torch.zeros((1, CONTROL_DIM)))
        if tuple(output.shape) != (1, STATE_DIM) or not bool(torch.isfinite(output).all()):
            raise ValueError("顺模型接口必须是 (x[12], uact[6]) -> x_hat_next[12]")

    def _load_state(self) -> dict:
        default = {
            "schema": SCHEMA,
            "sample_dt": DT,
            "generation": 0,
            "consecutive_wins": 0,
            "consecutive_failures": 0,
            "rounds_seen": 0,
            "tracking_mse_history_deg2": [],
            "canary": {"active": False},
            "history": [],
        }
        try:
            with self.state_path.open(encoding="utf-8") as stream:
                loaded = json.load(stream)
            if (loaded.get("schema") != SCHEMA or
                    abs(float(loaded.get("sample_dt", -1.0)) - DT) > 1e-9):
                return default
            default.update(loaded)
        except Exception:
            pass
        return default

    def _load_replay(self) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Restore unfinished three-round accumulation across Play sessions."""
        if not self.replay_path.exists():
            return []
        try:
            with np.load(self.replay_path, allow_pickle=False) as blob:
                lengths = np.asarray(blob["lengths"], dtype=np.int64)
                states = np.asarray(blob["states"], dtype=np.float32)
                controls = np.asarray(blob["controls"], dtype=np.float32)
                targets = np.asarray(blob["targets"], dtype=np.float32)
                tracking_errors = (np.asarray(blob["tracking_errors_deg"], dtype=np.float32)
                                   if "tracking_errors_deg" in blob.files else
                                   np.zeros(len(states), dtype=np.float32))
            if (lengths.ndim != 1 or len(lengths) >= REPLAY_ROUNDS or
                    np.any(lengths <= 0) or int(np.sum(lengths)) != len(states) or
                    states.shape != (len(states), STATE_DIM) or
                    controls.shape != (len(states), CONTROL_DIM) or
                    targets.shape != (len(states), STATE_DIM)):
                raise ValueError("invalid forward replay shape")
            rounds, offset = [], 0
            for count in lengths:
                stop = offset + int(count)
                rounds.append([(states[i].copy(), controls[i].copy(), targets[i].copy(),
                                float(tracking_errors[i]))
                               for i in range(offset, stop)])
                offset = stop
            return rounds
        except Exception:
            try:
                self.replay_path.unlink()
            except FileNotFoundError:
                pass
            return []

    def _save_replay(self) -> None:
        if not self.replay_rounds:
            try:
                self.replay_path.unlink()
            except FileNotFoundError:
                pass
            return
        lengths = np.asarray([len(one_round) for one_round in self.replay_rounds], np.int64)
        pooled = [item for one_round in self.replay_rounds for item in one_round]
        states = np.asarray([item[0] for item in pooled], np.float32)
        controls = np.asarray([item[1] for item in pooled], np.float32)
        targets = np.asarray([item[2] for item in pooled], np.float32)
        tracking_errors = np.asarray(
            [item[3] if len(item) > 3 else 0.0 for item in pooled], np.float32)
        tmp = self.replay_path.with_suffix(self.replay_path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, lengths=lengths, states=states,
                                controls=controls, targets=targets,
                                tracking_errors_deg=tracking_errors)
        os.replace(tmp, self.replay_path)

    @property
    def generation(self) -> int:
        return int(self.state.get("generation", 0))

    def _predict(self, model, state, control) -> np.ndarray:
        torch = self.torch
        x = torch.as_tensor(np.asarray(state, np.float32)[None])
        u = torch.as_tensor(np.asarray(control, np.float32)[None])
        with torch.no_grad():
            answer = model(x, u).cpu().numpy()[0].astype(float)
        if answer.shape != (STATE_DIM,) or not np.all(np.isfinite(answer)):
            raise FloatingPointError("顺模型产生非有限的 x_hat_next")
        return answer

    def predict_pair(self, state, control) -> tuple[np.ndarray, np.ndarray]:
        """Return complete active/shadow ``x_hat_next`` predictions."""
        return (self._predict(self.active, state, control),
                self._predict(self.shadow, state, control))

    def observe_next(self, next_state, time_s: float, lap: int) -> dict | None:
        """Pair the previous ``(x_t,uact_t)`` with measured ``x_(t+1)``."""
        if self.pending is None:
            return None
        pending, self.pending = self.pending, None
        measured = np.asarray(next_state, float)
        if (measured.shape != (STATE_DIM,) or not np.all(np.isfinite(measured)) or
                pending["lap"] != int(lap)):
            return None

        active_error = measured - pending["active_hat"]
        shadow_error = measured - pending["shadow_hat"]
        active_angle_norm = float(np.linalg.norm(np.degrees(active_error[:3])))
        shadow_angle_norm = float(np.linalg.norm(np.degrees(shadow_error[:3])))
        self.round_samples.append((pending["state"], pending["control"], measured.copy(),
                                   pending.get("tracking_error_deg", 0.0)))

        if active_angle_norm >= 4.0:
            self.failure_streak += 1
            self.warning_streak += 1
        elif active_angle_norm >= 2.0:
            self.warning_streak += 1
            self.failure_streak = 0
        else:
            self.warning_streak = 0
            self.failure_streak = 0
        status = ("失效" if self.failure_streak >= 3 else
                  "警告" if self.warning_streak >= 3 else "正常")

        row = {
            "lap": int(lap) + 1,
            "t_source": pending["time_s"],
            "t_measured": float(time_s),
            **{f"x{i}": float(v) for i, v in enumerate(pending["state"])},
            **{f"uact{i}": float(v) for i, v in enumerate(pending["control"])},
            **{f"xhat_active{i}": float(v) for i, v in enumerate(pending["active_hat"])},
            **{f"xhat_shadow{i}": float(v) for i, v in enumerate(pending["shadow_hat"])},
            **{f"xnext{i}": float(v) for i, v in enumerate(measured)},
            **{f"e_active{i}": float(v) for i, v in enumerate(active_error)},
            **{f"e_shadow{i}": float(v) for i, v in enumerate(shadow_error)},
            "active_angle_norm_deg": active_angle_norm,
            "shadow_angle_norm_deg": shadow_angle_norm,
            "status": status,
            "generation": self.generation,
        }
        self.run_rows.append(row)
        self.last_observation = {
            "active_error_deg": np.degrees(active_error[:3]).tolist(),
            "shadow_error_deg": np.degrees(shadow_error[:3]).tolist(),
            "active_error_norm_deg": active_angle_norm,
            "shadow_error_norm_deg": shadow_angle_norm,
            "status": status,
            "generation": self.generation,
        }
        return dict(self.last_observation)

    def begin_transition(self, state, actual_control, time_s: float, lap: int,
                         reference=None) -> dict:
        """Store ``x_t,uact_t`` and both complete predictions for the next tick."""
        x = np.asarray(state, float)
        u = np.asarray(actual_control, float)
        if (x.shape != (STATE_DIM,) or u.shape != (CONTROL_DIM,) or
                not np.all(np.isfinite(x)) or not np.all(np.isfinite(u))):
            self.pending = None
            raise ValueError("影子顺模型收到无效的状态或 uact")
        active_hat, shadow_hat = self.predict_pair(x, u)
        tracking_error_deg = 0.0
        if reference is not None:
            ref = np.asarray(reference, float)
            if ref.shape == (3,) and np.all(np.isfinite(ref)):
                tracking_error_deg = float(np.linalg.norm(np.degrees(ref-x[:3])))
        self.pending = {
            "state": x.copy(), "control": u.copy(),
            "active_hat": active_hat, "shadow_hat": shadow_hat,
            "time_s": float(time_s), "lap": int(lap),
            "tracking_error_deg": tracking_error_deg,
        }
        return {"active_xhat_next": active_hat.copy(),
                "shadow_xhat_next": shadow_hat.copy()}

    def discard_pending(self) -> None:
        """Never bridge a round boundary, vent window, stop, or missing frame."""
        self.pending = None
        self.warning_streak = 0
        self.failure_streak = 0

    @staticmethod
    def _activity_masks(states: np.ndarray, controls: np.ndarray,
                        targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle_step_deg = np.degrees(np.linalg.norm(
            targets[:, :3] - states[:, :3], axis=1))
        angle_moving = angle_step_deg > ANGLE_MOTION_THRESHOLD_DEG
        pressure_changing = np.max(
            np.abs(targets[:, 6:] - states[:, 6:]), axis=1) > PRESSURE_ACTIVITY_THRESHOLD_KPA
        actuated = np.max(np.abs(controls), axis=1) > CONTROL_ACTIVITY_THRESHOLD_MV
        pressure_active = pressure_changing | actuated
        return angle_moving, pressure_active, angle_moving | pressure_active

    @classmethod
    def _metrics_from_arrays(cls, pred: np.ndarray, target: np.ndarray,
                             states: np.ndarray, controls: np.ndarray) -> dict:
        error = pred - target
        angle_moving, pressure_active, _ = cls._activity_masks(
            states, controls, target)
        all_angle = float(np.degrees(np.sqrt(np.mean(error[:, :3] ** 2))))
        angle_mse = (float(np.mean(np.degrees(error[angle_moving, :3]) ** 2))
                     if np.any(angle_moving) else math.inf)
        angle_rmse = (float(np.degrees(np.sqrt(np.mean(error[angle_moving, :3] ** 2))))
                      if np.any(angle_moving) else math.inf)
        velocity_rmse = (float(np.degrees(
            np.sqrt(np.mean(error[angle_moving, 3:6] ** 2))) / DT)
            if np.any(angle_moving) else math.inf)
        pressure_rmse = (float(np.sqrt(np.mean(error[pressure_active, 6:] ** 2)))
                         if np.any(pressure_active) else math.inf)
        return {
            "angle_mse_deg2": angle_mse,
            "angle_rmse_deg": angle_rmse,
            "velocity_rmse_deg_s": velocity_rmse,
            "pressure_rmse_kpa": pressure_rmse,
            "all_angle_rmse_deg": all_angle,
            "angle_samples": int(np.count_nonzero(angle_moving)),
            "pressure_samples": int(np.count_nonzero(pressure_active)),
        }

    def _evaluate(self, model, states: np.ndarray, controls: np.ndarray,
                  targets: np.ndarray) -> dict:
        torch = self.torch
        with torch.no_grad():
            pred = model(torch.from_numpy(states), torch.from_numpy(controls)).cpu().numpy()
        if pred.shape != targets.shape or not np.all(np.isfinite(pred)):
            return {"angle_mse_deg2": math.inf,
                    "angle_rmse_deg": math.inf, "velocity_rmse_deg_s": math.inf,
                    "pressure_rmse_kpa": math.inf, "all_angle_rmse_deg": math.inf,
                    "angle_samples": 0, "pressure_samples": 0}
        return self._metrics_from_arrays(pred, targets, states, controls)

    def _rollout_angle_rmse(self, model, samples) -> float:
        if not samples:
            return math.inf
        angle_errors = []
        # Non-overlapping 1.2 s blocks are enough to reject unstable candidates
        # without making round-end training noticeably slow.
        for start in range(0, len(samples), ROLLOUT_HORIZON):
            state = samples[start][0].copy()
            stop = min(len(samples), start + ROLLOUT_HORIZON)
            for index in range(start, stop):
                state = self._predict(model, state, samples[index][1])
                actual_step_deg = math.degrees(float(np.linalg.norm(
                    samples[index][2][:3] - samples[index][0][:3])))
                if actual_step_deg > ANGLE_MOTION_THRESHOLD_DEG:
                    angle_errors.append(state[:3] - samples[index][2][:3])
        error = np.asarray(angle_errors, float)
        return float(np.degrees(np.sqrt(np.mean(error**2)))) if len(error) else math.inf

    def _rollout_rounds_angle_rmse(self, model, rounds) -> float:
        values = [self._rollout_angle_rmse(model, samples)
                  for samples in rounds if samples]
        values = [value for value in values if math.isfinite(value)]
        return float(np.sqrt(np.mean(np.square(values)))) if values else math.inf

    def _validation_loss(self, model, states, controls, targets) -> float:
        torch = self.torch
        _, _, informative = self._activity_masks(states, controls, targets)
        if not np.any(informative):
            return math.inf
        states = states[informative]
        controls = controls[informative]
        targets = targets[informative]
        scale = torch.from_numpy(LOSS_SCALE)
        with torch.no_grad():
            prediction = model(torch.from_numpy(states), torch.from_numpy(controls))
            residual = (prediction - torch.from_numpy(targets)) / scale
            loss = torch.nn.functional.smooth_l1_loss(
                residual, torch.zeros_like(residual), beta=0.5)
        return float(loss)

    @staticmethod
    def _tracking_weights(tracking_errors: np.ndarray) -> np.ndarray:
        errors = np.nan_to_num(np.asarray(tracking_errors, np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)
        weights = 1.0 + np.clip(errors / TRACK_WEIGHT_SCALE_DEG,
                                0.0, TRACK_WEIGHT_MAX-1.0)
        return weights.astype(np.float32)

    def _optimize(self, states, controls, targets, train_mask, val_mask,
                  epochs, batch_size, patience=None,
                  sample_weights=None) -> tuple[float, int]:
        """Optimize the shadow while anchoring it to the frozen active model."""
        torch = self.torch
        self.shadow.train()
        optimizer = torch.optim.AdamW(self.shadow.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        x_train = torch.from_numpy(states[train_mask])
        u_train = torch.from_numpy(controls[train_mask])
        y_train = torch.from_numpy(targets[train_mask])
        if sample_weights is None:
            sample_weights = np.ones(len(states), np.float32)
        weight_train_np = np.asarray(sample_weights, np.float32)[train_mask]
        weight_train_np /= max(1.0e-6, float(np.mean(weight_train_np)))
        weight_train = torch.from_numpy(weight_train_np)
        scale = torch.from_numpy(LOSS_SCALE)
        active_parameters = [parameter.detach().clone()
                             for parameter in self.active.parameters()]
        batch_size = max(1, min(int(batch_size), len(x_train)))
        generator = torch.Generator().manual_seed(
            20260906 + self.generation + self.rounds_seen)
        final_loss = math.inf
        best_loss = math.inf
        best_state = None
        stale = 0
        epochs_run = 0
        for epoch in range(int(epochs)):
            order = torch.randperm(len(x_train), generator=generator)
            for start in range(0, len(order), batch_size):
                chosen = order[start:start + batch_size]
                prediction = self.shadow(x_train[chosen], u_train[chosen])
                residual = (prediction - y_train[chosen]) / scale
                loss_each = torch.nn.functional.smooth_l1_loss(
                    residual, torch.zeros_like(residual), beta=0.5,
                    reduction="none").mean(dim=1)
                chosen_weights = weight_train[chosen]
                loss_data = torch.sum(loss_each*chosen_weights) / torch.sum(chosen_weights)
                anchor = torch.zeros((), dtype=torch.float32)
                for parameter, reference in zip(self.shadow.parameters(), active_parameters):
                    anchor = anchor + torch.mean((parameter-reference) ** 2)
                loss = loss_data + ANCHOR_WEIGHT*anchor
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shadow.parameters(), 1.0)
                optimizer.step()
                final_loss = float(loss.detach())
            epochs_run = epoch + 1
            if patience is not None:
                self.shadow.eval()
                current = self._validation_loss(
                    self.shadow, states[val_mask], controls[val_mask], targets[val_mask])
                if current < best_loss - 1.0e-6:
                    best_loss = current
                    best_state = {key: value.detach().clone()
                                  for key, value in self.shadow.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                self.shadow.train()
                if stale >= int(patience):
                    break
        if best_state is not None:
            self.shadow.load_state_dict(best_state)
        self.shadow.eval()
        return final_loss, epochs_run

    def _save_shadow(self) -> None:
        tmp = self.shadow_path.with_suffix(self.shadow_path.suffix + ".tmp")
        self.torch.jit.save(self.shadow.eval(), str(tmp))
        os.replace(tmp, self.shadow_path)
        self.shadow.train()

    def _promote(self) -> None:
        if self.active_path.exists():
            shutil.copy2(self.active_path, self.previous_path)
        tmp = self.active_path.with_suffix(self.active_path.suffix + ".tmp")
        self.torch.jit.save(self.shadow.eval(), str(tmp))
        os.replace(tmp, self.active_path)
        shutil.copy2(self.active_path, self.shadow_path)
        self.active = self.torch.jit.load(str(self.active_path), map_location="cpu").eval()
        self.shadow = self.torch.jit.load(str(self.shadow_path), map_location="cpu").train()
        self._assert_contract(self.active)
        self._assert_contract(self.shadow)

    def _rollback(self) -> None:
        """Restore the model saved immediately before provisional promotion."""
        if not self.previous_path.exists():
            raise FileNotFoundError("previous forward model is unavailable")
        shutil.copy2(self.previous_path, self.active_path)
        shutil.copy2(self.previous_path, self.shadow_path)
        self.active = self.torch.jit.load(str(self.active_path), map_location="cpu").eval()
        self.shadow = self.torch.jit.load(str(self.shadow_path), map_location="cpu").train()
        self._assert_contract(self.active)
        self._assert_contract(self.shadow)

    def _append_tracking_mse(self, value) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        history = list(self.state.get("tracking_mse_history_deg2", []))
        history.append(float(value))
        self.state["tracking_mse_history_deg2"] = history[-12:]

    def _assess_canary(self, tracking_mse_deg2) -> dict | None:
        canary = dict(self.state.get("canary") or {})
        if not canary.get("active"):
            return None
        if tracking_mse_deg2 is None or not math.isfinite(float(tracking_mse_deg2)):
            return {"checked": False, "reason": "缺少本轮真实跟踪MSE，暂不审核候选模型"}
        baseline = float(canary.get("baseline_mse_deg2", math.inf))
        candidate = float(tracking_mse_deg2)
        accepted = math.isfinite(baseline) and candidate <= CANARY_MAX_MSE_RATIO*baseline
        info = {
            "checked": True, "accepted": bool(accepted),
            "baseline_mse_deg2": baseline,
            "candidate_mse_deg2": candidate,
            "ratio": candidate/max(baseline, 1.0e-12),
        }
        if accepted:
            self.state["canary"] = {"active": False}
            self._append_tracking_mse(candidate)
        else:
            self._rollback()
            self.state["generation"] = int(canary.get(
                "generation_before", max(0, self.generation-1)))
            self.state["canary"] = {"active": False}
            self.state["consecutive_wins"] = 0
            self.state["consecutive_failures"] = 0
        return info

    def train_round(self, tracking_mse_deg2=None) -> dict:
        """Collect one lap; every three valid laps train and audit the shadow model."""
        self.discard_pending()
        samples, self.round_samples = self.round_samples, []
        canary_info = self._assess_canary(tracking_mse_deg2)
        if canary_info and canary_info.get("checked") and not canary_info.get("accepted"):
            result = {
                "samples": len(samples), "trained": False, "promoted": False,
                "rolled_back": True, "canary": canary_info,
                "reason": "候选模型实机跟踪未通过，已自动恢复上一代",
                "generation_after": self.generation,
            }
            self.replay_rounds = []
            self._save_replay()
            self._record_history(result)
            return result
        if not canary_info:
            self._append_tracking_mse(tracking_mse_deg2)
        if len(samples) < MIN_ROUND_SAMPLES:
            result = {"samples": len(samples), "trained": False, "promoted": False,
                      "reason": f"有效样本不足{MIN_ROUND_SAMPLES}"}
            self._record_history(result)
            return result

        self.rounds_seen += 1
        self.state["rounds_seen"] = self.rounds_seen
        self.replay_rounds.append(samples)
        self.replay_rounds = self.replay_rounds[-REPLAY_ROUNDS:]
        self._save_replay()
        # The first two laps only collect data.  In particular, do not let one
        # lap move the shadow weights before the three-lap dataset is available.
        train_now = len(self.replay_rounds) == REPLAY_ROUNDS
        if not train_now:
            result = {
                "samples": len(samples), "trained": False, "promoted": False,
                "rounds_seen": self.rounds_seen,
                "replay_rounds": len(self.replay_rounds),
                "reason": f"已累计{len(self.replay_rounds)}/{REPLAY_ROUNDS}轮，等待三轮合并训练",
            }
            self._record_history(result)
            return result

        pooled = [item for one_round in self.replay_rounds for item in one_round]
        states = np.asarray([item[0] for item in pooled], np.float32)
        controls = np.asarray([item[1] for item in pooled], np.float32)
        targets = np.asarray([item[2] for item in pooled], np.float32)
        tracking_errors = np.asarray(
            [item[3] if len(item) > 3 else 0.0 for item in pooled], np.float32)
        sample_weights = self._tracking_weights(tracking_errors)
        # Preserve the same deterministic holdout inside every lap so every
        # trajectory stage is represented, and validation rows never train.
        val_mask = np.concatenate([
            np.arange(len(one_round)) % 5 == 0
            for one_round in self.replay_rounds])
        train_mask = ~val_mask
        if np.count_nonzero(val_mask) < 8 or np.count_nonzero(train_mask) < 30:
            result = {"samples": len(pooled), "trained": False, "promoted": False,
                      "rounds_seen": self.rounds_seen,
                      "replay_rounds": len(self.replay_rounds),
                      "reason": "三轮合并后的训练/验证样本不足"}
            self._record_history(result)
            return result

        active_metrics = self._evaluate(
            self.active, states[val_mask], controls[val_mask], targets[val_mask])
        shadow_before = self._evaluate(
            self.shadow, states[val_mask], controls[val_mask], targets[val_mask])
        final_loss, replay_epochs_run = self._optimize(
            states, controls, targets, train_mask, val_mask,
            REPLAY_EPOCHS, REPLAY_BATCH_SIZE, REPLAY_PATIENCE,
            sample_weights=sample_weights)
        epochs_run = replay_epochs_run
        consolidated = True
        eval_rounds = list(self.replay_rounds)

        shadow_metrics = self._evaluate(
            self.shadow, states[val_mask], controls[val_mask], targets[val_mask])
        active_rollout = self._rollout_rounds_angle_rmse(self.active, eval_rounds)
        shadow_rollout = self._rollout_rounds_angle_rmse(self.shadow, eval_rounds)
        finite = all(math.isfinite(value) for value in (
            *active_metrics.values(), *shadow_metrics.values(), active_rollout, shadow_rollout))
        angle_win = (shadow_metrics["angle_rmse_deg"] <=
                     (1.0-MIN_ANGLE_IMPROVEMENT)*active_metrics["angle_rmse_deg"])
        velocity_safe = (shadow_metrics["velocity_rmse_deg_s"] <= max(
            1.15*active_metrics["velocity_rmse_deg_s"],
            active_metrics["velocity_rmse_deg_s"] + 0.5))
        pressure_safe = (shadow_metrics["pressure_rmse_kpa"] <= max(
            1.15*active_metrics["pressure_rmse_kpa"],
            active_metrics["pressure_rmse_kpa"] + 0.5))
        rollout_safe = shadow_rollout <= max(1.02*active_rollout, active_rollout + 0.2)
        passed = bool(finite and angle_win and velocity_safe and pressure_safe and rollout_safe)

        wins = int(self.state.get("consecutive_wins", 0)) + 1 if passed else 0
        failures = 0 if passed else int(self.state.get("consecutive_failures", 0)) + 1
        self.state["consecutive_wins"] = wins
        self.state["consecutive_failures"] = failures
        promoted = passed and wins >= REQUIRED_CONSECUTIVE_WINS
        result = {
            "samples": len(states), "round_samples": len(samples),
            "train_samples": int(np.count_nonzero(train_mask)),
            "validation_samples": int(np.count_nonzero(val_mask)), "trained": True,
            "loss": final_loss, "active": active_metrics,
            "shadow_before": shadow_before, "shadow": shadow_metrics,
            "epochs": epochs_run, "consolidated": consolidated,
            "replay_rounds": len(self.replay_rounds),
            "replay_epochs": replay_epochs_run,
            "active_rollout_angle_rmse_deg": active_rollout,
            "shadow_rollout_angle_rmse_deg": shadow_rollout,
            "passed": passed, "consecutive_wins": wins, "promoted": bool(promoted),
            "generation_before": self.generation,
            "tracking_weight_mean": float(np.mean(sample_weights[train_mask])),
            "tracking_weight_max": float(np.max(sample_weights[train_mask])),
        }
        if canary_info:
            result["canary"] = canary_info

        if promoted:
            generation_before = self.generation
            tracking_history = list(self.state.get("tracking_mse_history_deg2", []))
            baseline_values = tracking_history[-TRACKING_BASELINE_ROUNDS:]
            baseline_mse = (float(np.mean(baseline_values))
                            if baseline_values else math.inf)
            self._promote()
            self.state["generation"] = generation_before + 1
            self.state["canary"] = {
                "active": True,
                "generation_before": generation_before,
                "candidate_generation": generation_before + 1,
                "baseline_mse_deg2": baseline_mse,
            }
            self.state["consecutive_wins"] = 0
            self.state["consecutive_failures"] = 0
            result["provisional"] = True
            result["tracking_baseline_mse_deg2"] = baseline_mse
        else:
            self._save_shadow()
            # If adaptation diverges badly for three rounds, restore a clean
            # shadow copy instead of allowing error to accumulate indefinitely.
            if failures >= 3 and shadow_metrics["angle_rmse_deg"] > max(
                    1.5*active_metrics["angle_rmse_deg"],
                    active_metrics["angle_rmse_deg"] + 1.0):
                shutil.copy2(self.active_path, self.shadow_path)
                self.shadow = self.torch.jit.load(
                    str(self.shadow_path), map_location="cpu").train()
                self.state["consecutive_failures"] = 0
                result["shadow_reset"] = True

        result["generation_after"] = self.generation
        self._record_history(result)
        # Use non-overlapping three-lap datasets.  The next lap starts a fresh
        # accumulation window rather than reusing two already-trained laps.
        self.replay_rounds = []
        self._save_replay()
        return result

    def abandon_round(self) -> None:
        """Drop an incomplete/unsafe round without changing either model."""
        self.discard_pending()
        self.round_samples = []

    def _record_history(self, result: dict) -> None:
        history = list(self.state.get("history", []))
        history.append(result)
        self.state["history"] = history[-100:]
        _atomic_json(self.state_path, self.state)

    def save_run_csv(self, path: os.PathLike | str) -> str | None:
        if not self.run_rows:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(self.run_rows[0].keys())
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.run_rows)
        return str(path)


class SequenceShadowForwardLearner(ShadowForwardLearner):
    """Active/shadow learning for a ten-tick history MLP or GRU correction.

    The deployed TorchScript model has the contract
    ``(x_history[B,10,12], u_history[B,10,6]) -> correction[B,9]``.
    A frozen fitted one-step model supplies the nominal next state.  Both the
    active and shadow sequence networks correct the same nominal prediction,
    and only the shadow network is optimized between complete three-lap sets.
    """

    def __init__(self, architecture: str, sequence_base_path,
                 physical_base_path=BASE_MODEL_PATH, **paths):
        architecture = str(architecture).strip().lower()
        if architecture not in ("history_mlp", "gru"):
            raise ValueError(f"未知时序影子模型：{architecture}")
        self.architecture = architecture
        self.history_steps = 10
        self.physical_base_path = Path(physical_base_path)
        self.actual_history: list[tuple[np.ndarray, np.ndarray]] = []
        super().__init__(base_model_path=sequence_base_path, **paths)
        if not self.physical_base_path.exists():
            raise FileNotFoundError(f"基础单拍顺模型不存在：{self.physical_base_path}")
        self.base_net = self.torch.jit.load(
            str(self.physical_base_path), map_location="cpu").eval()
        with self.torch.no_grad():
            answer = self.base_net(
                self.torch.zeros((1, STATE_DIM)),
                self.torch.zeros((1, CONTROL_DIM)))
        if tuple(answer.shape) != (1, STATE_DIM):
            raise ValueError("基础单拍顺模型接口必须是 (x[12],uact[6])->x_next[12]")

    def _assert_contract(self, model) -> None:
        torch = self.torch
        with torch.no_grad():
            output = model(
                torch.zeros((1, self.history_steps, STATE_DIM)),
                torch.zeros((1, self.history_steps, CONTROL_DIM)))
        if tuple(output.shape) != (1, 9) or not bool(torch.isfinite(output).all()):
            raise ValueError(
                "时序影子模型接口必须是 ([B,10,12],[B,10,6])->correction[B,9]")

    def _load_replay(self):
        if not self.replay_path.exists():
            return []
        try:
            with np.load(self.replay_path, allow_pickle=False) as blob:
                lengths = np.asarray(blob["lengths"], dtype=np.int64)
                histories = np.asarray(blob["histories"], dtype=np.float32)
                controls = np.asarray(blob["controls"], dtype=np.float32)
                targets = np.asarray(blob["targets"], dtype=np.float32)
                tracking_errors = (np.asarray(blob["tracking_errors_deg"], dtype=np.float32)
                                   if "tracking_errors_deg" in blob.files else
                                   np.zeros(len(histories), dtype=np.float32))
            total = int(np.sum(lengths))
            if (lengths.ndim != 1 or len(lengths) >= REPLAY_ROUNDS or
                    np.any(lengths <= 0) or total != len(histories) or
                    histories.shape != (total, self.history_steps, STATE_DIM) or
                    controls.shape != (total, self.history_steps, CONTROL_DIM) or
                    targets.shape != (total, STATE_DIM)):
                raise ValueError("invalid sequence replay shape")
            rounds, offset = [], 0
            for count in lengths:
                stop = offset + int(count)
                rounds.append([(histories[i].copy(), controls[i].copy(), targets[i].copy(),
                                float(tracking_errors[i]))
                               for i in range(offset, stop)])
                offset = stop
            return rounds
        except Exception:
            try:
                self.replay_path.unlink()
            except FileNotFoundError:
                pass
            return []

    def _save_replay(self) -> None:
        if not self.replay_rounds:
            try:
                self.replay_path.unlink()
            except FileNotFoundError:
                pass
            return
        lengths = np.asarray([len(one_round) for one_round in self.replay_rounds], np.int64)
        pooled = [sample for one_round in self.replay_rounds for sample in one_round]
        histories = np.asarray([sample[0] for sample in pooled], np.float32)
        controls = np.asarray([sample[1] for sample in pooled], np.float32)
        targets = np.asarray([sample[2] for sample in pooled], np.float32)
        tracking_errors = np.asarray(
            [sample[3] if len(sample) > 3 else 0.0 for sample in pooled], np.float32)
        tmp = self.replay_path.with_suffix(self.replay_path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, lengths=lengths, histories=histories,
                                controls=controls, targets=targets,
                                tracking_errors_deg=tracking_errors)
        os.replace(tmp, self.replay_path)

    def _predict(self, model, state_history, control_history) -> np.ndarray:
        torch = self.torch
        hx = np.asarray(state_history, np.float32)
        hu = np.asarray(control_history, np.float32)
        if (hx.shape != (self.history_steps, STATE_DIM) or
                hu.shape != (self.history_steps, CONTROL_DIM)):
            raise ValueError("时序顺模型历史必须是10拍")
        with torch.no_grad():
            xt = torch.from_numpy(hx[None])
            ut = torch.from_numpy(hu[None])
            correction = model(xt, ut)[0]
            nominal = self.base_net(xt[:, -1], ut[:, -1])[0]
            dq = torch.clamp(nominal[3:6] + correction[:3], -0.30, 0.30)
            pressure = torch.clamp(nominal[6:] + correction[3:], 0.0, 900.0)
            answer = torch.cat((xt[0, -1, :3] + dq, dq, pressure))
        result = answer.cpu().numpy().astype(float)
        if result.shape != (STATE_DIM,) or not np.all(np.isfinite(result)):
            raise FloatingPointError("时序顺模型产生非有限的 x_hat_next")
        return result

    def predict_pair(self, state_history, control_history):
        return (self._predict(self.active, state_history, control_history),
                self._predict(self.shadow, state_history, control_history))

    def begin_transition(self, state, actual_control, time_s: float, lap: int,
                         reference=None) -> dict:
        x = np.asarray(state, float)
        u = np.asarray(actual_control, float)
        if (x.shape != (STATE_DIM,) or u.shape != (CONTROL_DIM,) or
                not np.all(np.isfinite(x)) or not np.all(np.isfinite(u))):
            self.discard_pending()
            raise ValueError("时序影子顺模型收到无效的状态或 uact")
        # Do not train on artificial zero padding.  The first valid transition
        # is created only after nine real preceding samples are available.
        if len(self.actual_history) < self.history_steps - 1:
            self.actual_history.append((x.copy(), u.copy()))
            self.pending = None
            return {}
        prefix = self.actual_history[-(self.history_steps - 1):]
        hx = np.asarray([item[0] for item in prefix] + [x.copy()], np.float32)
        hu = np.asarray([item[1] for item in prefix] + [u.copy()], np.float32)
        active_hat, shadow_hat = self.predict_pair(hx, hu)
        tracking_error_deg = 0.0
        if reference is not None:
            ref = np.asarray(reference, float)
            if ref.shape == (3,) and np.all(np.isfinite(ref)):
                tracking_error_deg = float(np.linalg.norm(np.degrees(ref-x[:3])))
        self.pending = {
            "state_history": hx, "control_history": hu,
            "state": x.copy(), "control": u.copy(),
            "active_hat": active_hat, "shadow_hat": shadow_hat,
            "time_s": float(time_s), "lap": int(lap),
            "tracking_error_deg": tracking_error_deg,
        }
        self.actual_history.append((x.copy(), u.copy()))
        self.actual_history = self.actual_history[-(self.history_steps - 1):]
        return {"active_xhat_next": active_hat.copy(),
                "shadow_xhat_next": shadow_hat.copy()}

    def observe_next(self, next_state, time_s: float, lap: int) -> dict | None:
        if self.pending is None:
            return None
        pending, self.pending = self.pending, None
        measured = np.asarray(next_state, float)
        if (measured.shape != (STATE_DIM,) or not np.all(np.isfinite(measured)) or
                pending["lap"] != int(lap)):
            return None
        active_error = measured - pending["active_hat"]
        shadow_error = measured - pending["shadow_hat"]
        active_angle_norm = float(np.linalg.norm(np.degrees(active_error[:3])))
        shadow_angle_norm = float(np.linalg.norm(np.degrees(shadow_error[:3])))
        self.round_samples.append((pending["state_history"],
                                   pending["control_history"], measured.copy(),
                                   pending.get("tracking_error_deg", 0.0)))
        if active_angle_norm >= 4.0:
            self.failure_streak += 1; self.warning_streak += 1
        elif active_angle_norm >= 2.0:
            self.warning_streak += 1; self.failure_streak = 0
        else:
            self.warning_streak = 0; self.failure_streak = 0
        status = ("失效" if self.failure_streak >= 3 else
                  "警告" if self.warning_streak >= 3 else "正常")
        row = {
            "lap": int(lap) + 1, "t_source": pending["time_s"],
            "t_measured": float(time_s), "architecture": self.architecture,
            **{f"x{i}": float(v) for i, v in enumerate(pending["state"])},
            **{f"uact{i}": float(v) for i, v in enumerate(pending["control"])},
            **{f"xhat_active{i}": float(v) for i, v in enumerate(pending["active_hat"])},
            **{f"xhat_shadow{i}": float(v) for i, v in enumerate(pending["shadow_hat"])},
            **{f"xnext{i}": float(v) for i, v in enumerate(measured)},
            **{f"e_active{i}": float(v) for i, v in enumerate(active_error)},
            **{f"e_shadow{i}": float(v) for i, v in enumerate(shadow_error)},
            "active_angle_norm_deg": active_angle_norm,
            "shadow_angle_norm_deg": shadow_angle_norm,
            "status": status, "generation": self.generation,
        }
        self.run_rows.append(row)
        self.last_observation = {
            "active_error_deg": np.degrees(active_error[:3]).tolist(),
            "shadow_error_deg": np.degrees(shadow_error[:3]).tolist(),
            "active_error_norm_deg": active_angle_norm,
            "shadow_error_norm_deg": shadow_angle_norm,
            "status": status, "generation": self.generation,
        }
        return dict(self.last_observation)

    def discard_pending(self) -> None:
        super().discard_pending()
        self.actual_history = []

    @staticmethod
    def _activity_masks(histories, controls, targets):
        current = histories[:, -1]
        current_u = controls[:, -1]
        angle_step_deg = np.degrees(np.linalg.norm(
            targets[:, :3] - current[:, :3], axis=1))
        angle_moving = angle_step_deg > ANGLE_MOTION_THRESHOLD_DEG
        pressure_changing = np.max(
            np.abs(targets[:, 6:] - current[:, 6:]), axis=1) > PRESSURE_ACTIVITY_THRESHOLD_KPA
        actuated = np.max(np.abs(current_u), axis=1) > CONTROL_ACTIVITY_THRESHOLD_MV
        pressure_active = pressure_changing | actuated
        return angle_moving, pressure_active, angle_moving | pressure_active

    def _predict_arrays(self, model, histories, controls):
        torch = self.torch
        with torch.no_grad():
            hx = torch.from_numpy(np.asarray(histories, np.float32))
            hu = torch.from_numpy(np.asarray(controls, np.float32))
            correction = model(hx, hu)
            nominal = self.base_net(hx[:, -1], hu[:, -1])
            dq = torch.clamp(nominal[:, 3:6] + correction[:, :3], -0.30, 0.30)
            pressure = torch.clamp(nominal[:, 6:] + correction[:, 3:], 0.0, 900.0)
            prediction = torch.cat((hx[:, -1, :3] + dq, dq, pressure), dim=1)
        return prediction.cpu().numpy()

    def _evaluate(self, model, histories, controls, targets) -> dict:
        prediction = self._predict_arrays(model, histories, controls)
        error = prediction - targets
        angle_moving, pressure_active, _ = self._activity_masks(
            histories, controls, targets)
        all_angle = float(np.degrees(np.sqrt(np.mean(error[:, :3] ** 2))))
        angle_mse = (float(np.mean(np.degrees(error[angle_moving, :3]) ** 2))
                     if np.any(angle_moving) else math.inf)
        angle_rmse = (float(np.degrees(np.sqrt(np.mean(error[angle_moving, :3] ** 2))))
                      if np.any(angle_moving) else math.inf)
        velocity_rmse = (float(np.degrees(
            np.sqrt(np.mean(error[angle_moving, 3:6] ** 2))) / DT)
            if np.any(angle_moving) else math.inf)
        pressure_rmse = (float(np.sqrt(np.mean(error[pressure_active, 6:] ** 2)))
                         if np.any(pressure_active) else math.inf)
        return {
            "angle_mse_deg2": angle_mse, "angle_rmse_deg": angle_rmse,
            "velocity_rmse_deg_s": velocity_rmse,
            "pressure_rmse_kpa": pressure_rmse,
            "all_angle_rmse_deg": all_angle,
            "angle_samples": int(np.count_nonzero(angle_moving)),
            "pressure_samples": int(np.count_nonzero(pressure_active)),
        }

    def _rollout_angle_rmse(self, model, samples) -> float:
        if not samples:
            return math.inf
        angle_errors = []
        for start in range(0, len(samples), ROLLOUT_HORIZON):
            hx = samples[start][0].copy()
            hu = samples[start][1].copy()
            stop = min(len(samples), start + ROLLOUT_HORIZON)
            for index in range(start, stop):
                hu[-1] = samples[index][1][-1]
                prediction = self._predict(model, hx, hu)
                target = samples[index][2]
                actual_step_deg = math.degrees(float(np.linalg.norm(
                    target[:3] - samples[index][0][-1, :3])))
                if actual_step_deg > ANGLE_MOTION_THRESHOLD_DEG:
                    angle_errors.append(prediction[:3] - target[:3])
                hx = np.concatenate((hx[1:], prediction[None]), axis=0)
                next_u = (samples[index + 1][1][-1]
                          if index + 1 < stop else samples[index][1][-1])
                hu = np.concatenate((hu[1:], next_u[None]), axis=0)
        error = np.asarray(angle_errors, float)
        return float(np.degrees(np.sqrt(np.mean(error ** 2)))) if len(error) else math.inf

    def _validation_loss(self, model, histories, controls, targets) -> float:
        torch = self.torch
        _, _, informative = self._activity_masks(histories, controls, targets)
        if not np.any(informative):
            return math.inf
        hx = torch.from_numpy(histories[informative])
        hu = torch.from_numpy(controls[informative])
        target = torch.from_numpy(targets[informative])
        scale = torch.from_numpy(SEQUENCE_LOSS_SCALE)
        with torch.no_grad():
            nominal = self.base_net(hx[:, -1], hu[:, -1])
            wanted = torch.cat((target[:, 3:6] - nominal[:, 3:6],
                                target[:, 6:] - nominal[:, 6:]), dim=1)
            residual = (model(hx, hu) - wanted) / scale
            loss = torch.nn.functional.smooth_l1_loss(
                residual, torch.zeros_like(residual), beta=0.5)
        return float(loss)

    def _optimize(self, histories, controls, targets, train_mask, val_mask,
                  epochs, batch_size, patience=None, sample_weights=None):
        torch = self.torch
        self.shadow.train()
        optimizer = torch.optim.AdamW(self.shadow.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        hx_train = torch.from_numpy(histories[train_mask])
        hu_train = torch.from_numpy(controls[train_mask])
        target_train = torch.from_numpy(targets[train_mask])
        if sample_weights is None:
            sample_weights = np.ones(len(histories), np.float32)
        weight_train_np = np.asarray(sample_weights, np.float32)[train_mask]
        weight_train_np /= max(1.0e-6, float(np.mean(weight_train_np)))
        weight_train = torch.from_numpy(weight_train_np)
        with torch.no_grad():
            nominal = self.base_net(hx_train[:, -1], hu_train[:, -1])
            wanted = torch.cat((target_train[:, 3:6] - nominal[:, 3:6],
                                target_train[:, 6:] - nominal[:, 6:]), dim=1)
        scale = torch.from_numpy(SEQUENCE_LOSS_SCALE)
        active_parameters = [p.detach().clone() for p in self.active.parameters()]
        batch_size = max(1, min(int(batch_size), len(hx_train)))
        generator = torch.Generator().manual_seed(
            20260908 + self.generation + self.rounds_seen)
        final_loss = math.inf; best_loss = math.inf; best_state = None; stale = 0
        epochs_run = 0
        for epoch in range(int(epochs)):
            order = torch.randperm(len(hx_train), generator=generator)
            for start in range(0, len(order), batch_size):
                chosen = order[start:start + batch_size]
                correction = self.shadow(hx_train[chosen], hu_train[chosen])
                residual = (correction-wanted[chosen]) / scale
                loss_each = torch.nn.functional.smooth_l1_loss(
                    residual, torch.zeros_like(residual), beta=0.5,
                    reduction="none").mean(dim=1)
                chosen_weights = weight_train[chosen]
                data_loss = torch.sum(loss_each*chosen_weights) / torch.sum(chosen_weights)
                anchor = torch.zeros((), dtype=torch.float32)
                for parameter, reference in zip(self.shadow.parameters(), active_parameters):
                    anchor = anchor + torch.mean((parameter - reference) ** 2)
                loss = data_loss + ANCHOR_WEIGHT * anchor
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shadow.parameters(), 1.0)
                optimizer.step(); final_loss = float(loss.detach())
            epochs_run = epoch + 1
            if patience is not None:
                self.shadow.eval()
                current = self._validation_loss(
                    self.shadow, histories[val_mask], controls[val_mask], targets[val_mask])
                if current < best_loss - 1.0e-6:
                    best_loss = current
                    best_state = {key: value.detach().clone()
                                  for key, value in self.shadow.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                self.shadow.train()
                if stale >= int(patience):
                    break
        if best_state is not None:
            self.shadow.load_state_dict(best_state)
        self.shadow.eval()
        return final_loss, epochs_run

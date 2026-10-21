# -*- coding: utf-8 -*-
"""Constrained, task-indexed iterative learning feedforward for MPC.

The learner never differentiates through a plant model.  It remembers the
repeatable part of the MPC feedback effort from one completed hardware lap and
uses a bounded, smoothed fraction as the nominal input of the next lap.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional

import numpy as np


NCH = 6
ANTAGONIST_PAIRS = ((0, 1), (2, 3), (4, 5))


class ConstrainedILC:
    """Persistent ILC curve with one-lap real-hardware canary and rollback."""

    SCHEMA = 1

    def __init__(self, state_path: str, reference: np.ndarray, plot_reference: np.ndarray,
                 dt: float, u_max: float, learning_rate: float = 0.15,
                 max_update_mv: float = 100.0, smoothing_ticks: int = 5,
                 max_slew_mv_s: float = 2500.0, accept_ratio: float = 1.0):
        self.state_path = os.path.abspath(state_path)
        self.reference = np.asarray(reference, dtype=float)
        self.plot_reference = np.asarray(plot_reference, dtype=float)
        if self.reference.ndim != 2 or self.reference.shape[1] != 3:
            raise ValueError("ILC reference must have shape T x 3")
        if self.plot_reference.shape != self.reference.shape:
            raise ValueError("ILC plot reference must match control reference")
        self.length = len(self.reference)
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.learning_rate = float(np.clip(learning_rate, 0.0, 1.0))
        self.max_update_mv = max(0.0, float(max_update_mv))
        self.smoothing_ticks = max(1, int(smoothing_ticks))
        if self.smoothing_ticks % 2 == 0:
            self.smoothing_ticks += 1
        self.max_slew_mv_s = max(0.0, float(max_slew_mv_s))
        self.accept_ratio = max(0.0, float(accept_ratio))
        self.task_hash = self._fingerprint()
        self.generation = 0
        self.ff = np.zeros((self.length, NCH), dtype=float)
        self.previous_ff = self.ff.copy()
        self.canary_active = False
        self.baseline_mse = float("inf")
        self.history_mse = []
        self._load()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([self.dt], dtype=np.float64).tobytes())
        digest.update(np.round(self.reference, 9).astype(np.float64).tobytes())
        digest.update(np.round(self.plot_reference, 9).astype(np.float64).tobytes())
        return digest.hexdigest()

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            self._save()
            return
        try:
            with np.load(self.state_path, allow_pickle=False) as blob:
                schema = int(np.asarray(blob["schema"]).reshape(-1)[0])
                task_hash = str(np.asarray(blob["task_hash"]).reshape(-1)[0])
                ff = np.asarray(blob["ff"], dtype=float)
                previous = np.asarray(blob["previous_ff"], dtype=float)
                if (schema != self.SCHEMA or task_hash != self.task_hash or
                        ff.shape != (self.length, NCH) or previous.shape != ff.shape):
                    self._save()
                    return
                self.ff = self._sanitize(ff)
                self.previous_ff = self._sanitize(previous)
                self.generation = int(np.asarray(blob["generation"]).reshape(-1)[0])
                self.canary_active = bool(int(np.asarray(blob["canary_active"]).reshape(-1)[0]))
                self.baseline_mse = float(np.asarray(blob["baseline_mse"]).reshape(-1)[0])
                self.history_mse = [float(v) for v in np.asarray(
                    blob.get("history_mse", np.empty(0)), dtype=float).reshape(-1)]
        except Exception:
            # A partial/corrupt learning file must never stop the base MPC.
            self.generation = 0
            self.ff[:] = 0.0
            self.previous_ff[:] = 0.0
            self.canary_active = False
            self.baseline_mse = float("inf")
            self.history_mse = []
            self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray([self.SCHEMA], dtype=np.int32),
                task_hash=np.asarray([self.task_hash]),
                generation=np.asarray([self.generation], dtype=np.int32),
                canary_active=np.asarray([int(self.canary_active)], dtype=np.int8),
                baseline_mse=np.asarray([self.baseline_mse], dtype=float),
                ff=self.ff.astype(np.float32),
                previous_ff=self.previous_ff.astype(np.float32),
                history_mse=np.asarray(self.history_mse[-30:], dtype=float),
            )
        os.replace(tmp, self.state_path)

    @classmethod
    def reset_file(cls, state_path: str) -> None:
        path = os.path.abspath(state_path)
        if os.path.isfile(path):
            os.remove(path)

    @staticmethod
    def _antagonist_project(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        for left, right in ANTAGONIST_PAIRS:
            both = (out[:, left] > 0.0) & (out[:, right] > 0.0)
            keep_left = out[:, left] >= out[:, right]
            out[both & keep_left, right] = 0.0
            out[both & ~keep_left, left] = 0.0
        return out

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        if self.smoothing_ticks <= 1 or len(values) <= 2:
            return values.copy()
        half = self.smoothing_ticks // 2
        # Triangular zero-phase filter: planned feedforward may use future phase,
        # but the short window avoids moving a correction into another segment.
        ramp = np.arange(1, half + 2, dtype=float)
        kernel = np.r_[ramp, ramp[-2::-1]]
        kernel /= np.sum(kernel)
        out = np.empty_like(values, dtype=float)
        for channel in range(NCH):
            padded = np.pad(values[:, channel], (half, half), mode="edge")
            out[:, channel] = np.convolve(padded, kernel, mode="valid")
        return out

    def _sanitize(self, values: np.ndarray) -> np.ndarray:
        out = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0,
                            posinf=self.u_max, neginf=0.0)
        out = np.clip(out, 0.0, self.u_max)
        if len(out):
            out[0] = 0.0
        if self.max_slew_mv_s > 0.0:
            du = self.max_slew_mv_s * self.dt
            for tick in range(1, len(out)):
                out[tick] = np.clip(out[tick], out[tick - 1] - du,
                                    out[tick - 1] + du)
        return self._antagonist_project(np.clip(out, 0.0, self.u_max))

    def horizon(self, tick: int, horizon: int) -> np.ndarray:
        """Nominal inputs u(t)..u(t+N-1), aligned with the MPC action horizon."""
        if self.length == 0:
            return np.zeros((int(horizon), NCH), dtype=float)
        index = np.minimum(np.arange(int(horizon)) + int(tick), self.length - 1)
        return self.ff[index].copy()

    def _propose(self, feedback: np.ndarray, valid_mask: np.ndarray,
                 baseline_mse: float) -> bool:
        feedback = np.asarray(feedback, dtype=float)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if feedback.shape != self.ff.shape or valid_mask.shape != self.ff.shape:
            return False
        learned = np.where(valid_mask, feedback, 0.0)
        learned = self._smooth(learned)
        delta = np.clip(self.learning_rate * learned,
                        -self.max_update_mv, self.max_update_mv)
        candidate = self._sanitize(self.ff + delta)
        if not np.any(np.abs(candidate - self.ff) >= 0.5):
            return False
        self.previous_ff = self.ff.copy()
        self.ff = candidate
        self.generation += 1
        self.canary_active = True
        self.baseline_mse = float(baseline_mse)
        return True

    def finish_round(self, total_u: np.ndarray, used_ff: np.ndarray,
                     valid_mask: np.ndarray, tracking_mse_deg2: float) -> Dict[str, object]:
        """Audit the current curve, then form at most one next-lap candidate."""
        total_u = np.asarray(total_u, dtype=float)
        used_ff = np.asarray(used_ff, dtype=float)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        score = float(tracking_mse_deg2)
        result: Dict[str, object] = {
            "tracking_mse_deg2": score,
            "generation_before": self.generation,
            "canary_checked": bool(self.canary_active),
            "accepted": None,
            "rolled_back": False,
            "proposed": False,
        }
        if (not np.isfinite(score) or total_u.shape != self.ff.shape or
                used_ff.shape != self.ff.shape or valid_mask.shape != self.ff.shape):
            result["reason"] = "incomplete or invalid ILC round"
            return result

        feedback = total_u - used_ff
        if self.canary_active:
            accepted = bool(score <= self.accept_ratio * self.baseline_mse)
            result.update({"accepted": accepted, "baseline_mse_deg2": self.baseline_mse,
                           "ratio": score / max(self.baseline_mse, 1.0e-12)})
            if accepted:
                self.canary_active = False
                self.previous_ff = self.ff.copy()
            else:
                self.ff = self.previous_ff.copy()
                self.generation = max(0, self.generation - 1)
                self.canary_active = False
                result["rolled_back"] = True

        self.history_mse.append(score)
        # Do not learn from a rejected candidate lap.  Run the restored stable
        # curve once before attempting a new bounded update.
        if not result["rolled_back"]:
            result["proposed"] = self._propose(feedback, valid_mask, score)
        result["generation_after"] = self.generation
        result["ff_peak_mv"] = float(np.max(self.ff)) if self.ff.size else 0.0
        result["ff_mean_mv"] = float(np.mean(self.ff)) if self.ff.size else 0.0
        self._save()
        return result

    def abort_round(self) -> Dict[str, object]:
        """An interrupted hardware canary is ungraded and therefore rolled back."""
        rolled_back = bool(self.canary_active)
        if rolled_back:
            self.ff = self.previous_ff.copy()
            self.generation = max(0, self.generation - 1)
            self.canary_active = False
            self._save()
        return {"rolled_back": rolled_back, "generation": self.generation}

    def status(self) -> Dict[str, object]:
        return {
            "generation": int(self.generation),
            "canary_active": bool(self.canary_active),
            "baseline_mse_deg2": (float(self.baseline_mse)
                                   if np.isfinite(self.baseline_mse) else None),
            "ff_peak_mv": float(np.max(self.ff)) if self.ff.size else 0.0,
            "ff_mean_mv": float(np.mean(self.ff)) if self.ff.size else 0.0,
        }

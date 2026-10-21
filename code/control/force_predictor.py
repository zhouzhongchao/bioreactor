from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODEL_VERSION = "force_estimator_v1_vib_xyz"
DEPLOYMENT_MODE_ALIASES = {
    "fusion": "vib_multihead_xyz",
    "vib_multihead_xyz": "vib_multihead_xyz",
    "separate": "hybrid_best_per_axis",
    "hybrid_best_per_axis": "hybrid_best_per_axis",
}
MODEL_VERSION_BY_MODE = {
    "vib_multihead_xyz": MODEL_VERSION,
    "hybrid_best_per_axis": "force_estimator_v1_separate_axes",
}
SEEDS = (2025, 2026, 2027)
FEATURE_TOKENS = (
    "P_kPa",
    "L_mm",
    "d_active_mm",
    "d_passive_mm",
    "dP_dt_kPa_s",
    "dL_dt_mm_s",
)


class LocalEncoder(nn.Module):
    def __init__(self, latent_dim: int = 38, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HAW36SingleForceRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_encoders = nn.ModuleList([LocalEncoder() for _ in range(6)])
        self.fusion = nn.Sequential(
            nn.Linear(6 * 38, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = [
            encoder(x[:, index * 6:(index + 1) * 6])
            for index, encoder in enumerate(self.local_encoders)
        ]
        return self.fusion(torch.cat(latents, dim=1)).squeeze(1)


class VariationalLocalEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, latent_dim: int = 38) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-10.0, 6.0)


class HAW36VIBMultiHeadRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(
            [VariationalLocalEncoder(64, 38) for _ in range(6)]
        )
        self.shared_fusion = nn.Sequential(
            nn.Linear(6 * 38, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = []
        for index, encoder in enumerate(self.encoders):
            mean, _ = encoder(x[:, index * 6:(index + 1) * 6])
            latents.append(mean)
        shared = self.shared_fusion(torch.cat(latents, dim=1))
        return torch.cat([head(shared) for head in self.task_heads], dim=1)


class StepEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(6, 32), nn.ReLU()) for _ in range(6)
        ])
        self.fusion = nn.Sequential(nn.Linear(6 * 32, 128), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [
            encoder(x[:, :, index * 6:(index + 1) * 6])
            for index, encoder in enumerate(self.local_encoders)
        ]
        return self.fusion(torch.cat(encoded, dim=2))


class TemporalGRUFxyzRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.step_encoder = StepEncoder()
        self.temporal = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.temporal(self.step_encoder(x))
        representation = output[:, -1]
        return torch.cat([head(representation) for head in self.heads], dim=1)


class ForceEstimator:
    channel_count = 6
    feature_count = 36
    window_size = 16

    def __init__(self, model_dir: str | Path, deployment_mode: str | None = None) -> None:
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Force model directory not found: {self.model_dir}")
        torch.set_num_threads(1)
        self.feature_names = tuple(
            f"ch{channel}_{token}"
            for channel in range(1, self.channel_count + 1)
            for token in FEATURE_TOKENS
        )
        self.feature_mean, self.feature_std = self._load_scaler()
        manifest = describe_model(self.model_dir)
        requested_mode = (manifest.get("deployment_mode", "hybrid_best_per_axis")
                          if deployment_mode is None else deployment_mode)
        self.deployment_mode = DEPLOYMENT_MODE_ALIASES.get(
            str(requested_mode).strip().lower(), "")
        if not self.deployment_mode:
            raise ValueError(f"Unsupported deployment_mode: {requested_mode}")
        self.model_version = MODEL_VERSION_BY_MODE[self.deployment_mode]
        if self.deployment_mode == "vib_multihead_xyz":
            self.model_version = manifest.get("model_version", self.model_version)
            self.fusion_seeds = tuple(manifest.get("fusion_seeds", SEEDS))
            (self.vib_models, self.vib_target_mean,
             self.vib_target_std) = self._load_multi_target_family(
                manifest.get("fusion_family", "fy"), HAW36VIBMultiHeadRegressor
            )
            self.fx_models = self.fy_models = self.fz_models = ()
            self.window_size = 1
        elif self.deployment_mode == "hybrid_best_per_axis":
            self.fx_models, self.fx_target = self._load_family(
                "fx", HAW36SingleForceRegressor, target_index=0
            )
            self.fy_models, self.fy_target = self._load_family(
                "fy", HAW36VIBMultiHeadRegressor, target_index=1
            )
            self.fz_models, self.fz_target = self._load_family(
                "fz", TemporalGRUFxyzRegressor, target_index=2
            )
            self.vib_models = ()
        self._history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._previous_pressure: np.ndarray | None = None
        self._previous_length: np.ndarray | None = None
        self._previous_time: float | None = None

    def _load_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        path = self.model_dir / "train_scaler_stats.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = {row["feature"]: row for row in csv.DictReader(stream)}
        missing = [name for name in self.feature_names if name not in rows]
        if missing:
            raise ValueError(f"Missing force-model scaler entries: {missing}")
        mean = np.asarray(
            [float(rows[name]["train_mean"]) for name in self.feature_names],
            dtype=np.float32,
        )
        std = np.asarray(
            [float(rows[name]["train_std"]) for name in self.feature_names],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Force-model scaler contains invalid values")
        return mean, std

    def _load_family(self, family, model_type, target_index):
        models = []
        target_mean = None
        target_std = None
        for seed in SEEDS:
            path = self.model_dir / family / f"model_seed{seed}.pt"
            checkpoint = torch.load(str(path), map_location="cpu")
            if tuple(checkpoint.get("feature_columns", ())) != tuple(
                name + "_z" for name in self.feature_names
            ):
                raise ValueError(f"Feature order mismatch in {path.name}")
            model = model_type()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            models.append(model)
            means = checkpoint.get("target_means_n")
            stds = checkpoint.get("target_stds_n")
            mean = float(means[target_index]) if means is not None else float(
                checkpoint["target_mean_n"]
            )
            std = float(stds[target_index]) if stds is not None else float(
                checkpoint["target_std_n"]
            )
            if target_mean is None:
                target_mean, target_std = mean, std
            elif not np.allclose([target_mean, target_std], [mean, std], rtol=0, atol=1e-10):
                raise ValueError(f"Target scaler mismatch in {path.name}")
        return tuple(models), (float(target_mean), float(target_std))

    def _load_multi_target_family(self, family, model_type):
        models = []
        target_mean = None
        target_std = None
        for seed in self.fusion_seeds:
            path = self.model_dir / family / f"model_seed{seed}.pt"
            checkpoint = torch.load(str(path), map_location="cpu")
            if tuple(checkpoint.get("feature_columns", ())) != tuple(
                name + "_z" for name in self.feature_names
            ):
                raise ValueError(f"Feature order mismatch in {path.name}")
            model = model_type()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            models.append(model)
            mean = np.asarray(checkpoint.get("target_means_n"), dtype=np.float64)
            std = np.asarray(checkpoint.get("target_stds_n"), dtype=np.float64)
            if (mean.shape != (3,) or std.shape != (3,) or
                    not np.all(np.isfinite(mean)) or
                    not np.all(np.isfinite(std)) or np.any(std <= 0)):
                raise ValueError(f"Invalid three-axis target scaler in {path.name}")
            if target_mean is None:
                target_mean, target_std = mean, std
            elif not np.allclose(
                    np.concatenate([target_mean, target_std]),
                    np.concatenate([mean, std]), rtol=0, atol=1e-10):
                raise ValueError(f"Target scaler mismatch in {path.name}")
        return tuple(models), target_mean, target_std

    @staticmethod
    def _six_finite(values, name) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain six finite values")
        return array

    def reset_state(self) -> None:
        self._history.clear()
        self._previous_pressure = None
        self._previous_length = None
        self._previous_time = None

    def build_standardized_feature(
        self,
        pressure_kpa,
        length_mm,
        active_mm,
        passive_mm,
        timestamp_s,
    ) -> np.ndarray:
        pressure = self._six_finite(pressure_kpa, "pressure_kpa")
        length = self._six_finite(length_mm, "length_mm")
        active = self._six_finite(active_mm, "active_mm")
        passive = self._six_finite(passive_mm, "passive_mm")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        dt = timestamp - self._previous_time if self._previous_time is not None else 0.0
        if (self._previous_pressure is None or self._previous_length is None or
                not np.isfinite(dt) or dt <= 1e-4 or dt > 1.0):
            pressure_rate = np.zeros(6, dtype=np.float32)
            length_rate = np.zeros(6, dtype=np.float32)
        else:
            pressure_rate = (pressure - self._previous_pressure) / dt
            length_rate = (length - self._previous_length) / dt
        self._previous_pressure = pressure.copy()
        self._previous_length = length.copy()
        self._previous_time = timestamp
        raw = np.asarray([
            value
            for channel in range(6)
            for value in (
                pressure[channel], length[channel], active[channel], passive[channel],
                pressure_rate[channel], length_rate[channel],
            )
        ], dtype=np.float32)
        return (raw - self.feature_mean) / self.feature_std

    def predict_standardized(
        self,
        current_feature,
        history=None,
    ) -> np.ndarray:
        feature = np.asarray(current_feature, dtype=np.float32).reshape(-1)
        if feature.shape != (self.feature_count,) or not np.all(np.isfinite(feature)):
            raise ValueError("current_feature must contain 36 finite values")
        if history is None:
            sequence = np.repeat(feature[None, :], self.window_size, axis=0)
        else:
            sequence = np.asarray(history, dtype=np.float32)
            if sequence.ndim != 2 or sequence.shape[1] != self.feature_count:
                raise ValueError("history must have shape [T,36]")
            if len(sequence) > self.window_size:
                sequence = sequence[-self.window_size:]
            if len(sequence) < self.window_size:
                sequence = np.concatenate([
                    np.repeat(sequence[:1], self.window_size - len(sequence), axis=0),
                    sequence,
                ], axis=0)
        x = torch.from_numpy(feature[None, :])
        if self.deployment_mode == "vib_multihead_xyz":
            with torch.inference_mode():
                prediction_z = np.mean([
                    model(x)[0].cpu().numpy() for model in self.vib_models
                ], axis=0)
            return prediction_z * self.vib_target_std + self.vib_target_mean
        seq = torch.from_numpy(sequence[None, :, :])
        with torch.inference_mode():
            fx_z = np.mean([float(model(x).item()) for model in self.fx_models])
            fy_z = np.mean([float(model(x)[0, 1].item()) for model in self.fy_models])
            fz_z = np.mean([float(model(seq)[0, 2].item()) for model in self.fz_models])
        return np.asarray([
            fx_z * self.fx_target[1] + self.fx_target[0],
            fy_z * self.fy_target[1] + self.fy_target[0],
            fz_z * self.fz_target[1] + self.fz_target[0],
        ], dtype=np.float64)

    def predict(
        self,
        pressure_kpa,
        length_mm,
        active_mm,
        passive_mm,
        timestamp_s,
    ) -> dict:
        started = time.perf_counter()
        feature = self.build_standardized_feature(
            pressure_kpa, length_mm, active_mm, passive_mm, timestamp_s
        )
        self._history.append(feature.copy())
        force = self.predict_standardized(feature, np.asarray(self._history))
        return {
            "state": "ready",
            "model_version": self.model_version,
            "prediction_mode": (
                "fusion" if self.deployment_mode == "vib_multihead_xyz" else "separate"),
            "force_xyz_n": force.tolist(),
            "history_samples": len(self._history),
            "window_size": self.window_size,
            "feature_max_abs_z": float(np.max(np.abs(feature))),
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
            "timestamp_s": float(timestamp_s),
        }


def describe_model(model_dir: str | Path) -> dict:
    path = Path(model_dir) / "manifest.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

# -*- coding: utf-8 -*-
"""Gymnasium environment for MPC + bounded residual PPO simulation.

The environment never opens a serial port.  The existing data-driven forward
models are used as the simulated plant, while the existing DAMPC supplies the
base action.  PPO only learns a small correction on top of that action.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# NumPy/Conda MKL and the PyTorch 1.12 Windows wheel both load Intel OpenMP.
# Initialising NumPy's LAPACK first gives them one consistent runtime and avoids
# the unsafe KMP_DUPLICATE_LIB_OK workaround.
_NUMPY_LINALG_PRIME = float(np.linalg.norm(np.eye(2), 2))

import gymnasium as gym
from gymnasium import spaces

try:
    from . import mpc_station as ms
except ImportError:  # direct execution/import from the control directory
    import mpc_station as ms


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "rl_sim_config.json"
DEFAULT_GAINS_PATH = HERE / "gains.json"


def load_rl_config(path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_path"] = str(config_path)
    return config


def _resolve_relative(path: str, base_file: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(base_file).resolve().parent / candidate
    return candidate.resolve()


def _load_model(architecture: str):
    architecture = str(architecture).strip().lower()
    if architecture == "mlp":
        return ms.PureDataModel()
    if architecture in ("history_mlp", "gru"):
        return ms.SequencePureDataModel(architecture)
    raise ValueError(f"未知顺模型 {architecture!r}；只能是 mlp/history_mlp/gru")


def _shortest_angle_deg(value: np.ndarray) -> np.ndarray:
    answer = np.asarray(value, dtype=float).copy()
    answer[..., 2] = (answer[..., 2] + 180.0) % 360.0 - 180.0
    return answer


def build_reference(config: Dict[str, Any], rng: Optional[np.random.Generator] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return model-coordinate and GUI-coordinate references at 20 Hz."""
    trajectory_path = _resolve_relative(
        str(config.get("trajectory_path", "important_task_trajectory.json")),
        str(config["_config_path"]))
    with trajectory_path.open(encoding="utf-8") as stream:
        blob = json.load(stream)
    points = blob.get("points", [])
    if not points:
        raise ValueError(f"轨迹文件没有示教点：{trajectory_path}")

    scale_bounds = config.get("trajectory_axis_scale_range", [1.0, 1.0])
    low, high = float(scale_bounds[0]), float(scale_bounds[1])
    axis_scale = (np.ones(3) if rng is None or abs(high-low) < 1e-12 else
                  rng.uniform(low, high, size=3))
    use_control = bool(config.get("use_q_control", True))
    dt = float(ms.DT)
    gui_rows = [np.zeros(3, dtype=float)]
    current = np.zeros(3, dtype=float)
    for point in points:
        key = "q_control" if use_control and isinstance(point.get("q_control"), list) else "q"
        target = np.asarray((list(point[key]) + [0.0, 0.0, 0.0])[:3], dtype=float)
        target *= axis_scale
        duration = max(dt, float(point.get("t", 2.0)))
        count = max(1, int(round(duration/dt)))
        start = current.copy()
        for tick in range(1, count+1):
            gui_rows.append(start + (target-start)*ms._s_curve5(tick/count))
        current = target

    if bool(blob.get("auto_hidden_return_to_zero", True)):
        duration = max(dt, float(blob.get("hidden_return_duration_s", 2.0)))
        count = max(1, int(round(duration/dt)))
        start = current.copy()
        for tick in range(1, count+1):
            gui_rows.append(start*(1.0-ms._s_curve5(tick/count)))

    gui_ref = np.asarray(gui_rows, dtype=float)
    model_ref = np.asarray([ms.q_model_from_st(*row) for row in gui_ref], dtype=float)
    return model_ref, gui_ref


class ResidualPneumaticEnv(gym.Env):
    """Task-specific 20 Hz simulator for bounded residual control.

    Action: six values in [-1, 1], mapped to bounded residual valve voltages.
    State transition: selected learned forward model, with modest episode-level
    actuator/dynamics randomisation and sensor noise.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: Optional[os.PathLike] = None,
                 *, training: bool = True, residual_enabled: bool = True,
                 config_overrides: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = copy.deepcopy(load_rl_config(config_path))
        if config_overrides:
            self.config.update(config_overrides)
        self.training = bool(training)
        self.residual_enabled = bool(residual_enabled)

        with DEFAULT_GAINS_PATH.open(encoding="utf-8") as stream:
            self.gains = json.load(stream)
        self.u_max = float(np.clip(float(self.gains.get("u_max", 2500.0)),
                                   100.0, ms.HARD_CLAMP_MV))
        ms.CLAMP_MV = self.u_max
        requested_controller_arch = str(
            self.config.get("controller_forward_model", "gains")).strip().lower()
        self.controller_arch = (str(self.gains.get("mpc_forward_model", "mlp")).lower()
                                if requested_controller_arch == "gains"
                                else requested_controller_arch)
        self.plant_arch = str(self.config.get("plant_forward_model", "gru")).lower()
        self.controller_model = _load_model(self.controller_arch)
        self.plant_model = _load_model(self.plant_arch)

        solver = str(self.config.get("mpc_solver", "ltv")).strip().lower()
        if solver == "gains":
            solver = str(self.gains.get("mpc_solver", "ltv")).strip().lower()
        self.mpc = ms.DAMPC(
            self.controller_model,
            N=int(self.gains.get("mpc_N", 20)),
            w_q=tuple(float(v) for v in self.gains.get("mpc_w_q", [5000.0]*3)),
            w_v=tuple(float(v) for v in self.gains.get("mpc_w_v", [200.0]*3)),
            rd=float(self.gains.get("mpc_rd", 0.008)),
            rho=float(self.gains.get("mpc_rho", 0.0)),
            l1=float(self.gains.get("mpc_l1", 0.0)),
            x_comp_ratio=float(self.gains.get("mpc_x_comp_ratio", 0.5)),
            lambda_c=float(self.gains.get("mpc_lambda_c", 0.0)),
            lock_lead_ms=float(self.gains.get("mpc_lock_lead_ms", 100.0)),
            solver=solver,
            nl_iters=int(self.gains.get("mpc_nonlinear_iterations", 20)),
            nl_lr_mv=float(self.gains.get("mpc_nonlinear_lr_mv", 180.0)),
            nl_budget_ms=float(self.gains.get("mpc_nonlinear_budget_ms", 40.0)),
            mppi_samples=int(self.gains.get("mpc_mppi_samples", 192)),
            mppi_iterations=int(self.gains.get("mpc_mppi_iterations", 2)),
            mppi_sigma_mv=float(self.gains.get("mpc_mppi_sigma_mv", 700.0)),
            mppi_temperature=float(self.gains.get("mpc_mppi_temperature", 1.0)),
            mppi_knots=int(self.gains.get("mpc_mppi_knots", 5)))

        self.max_residual_mv = max(0.0, float(self.config.get("max_residual_mv", 250.0)))
        self.residual_slew_mv_s = max(
            0.0, float(self.config.get("residual_slew_mv_s", 1200.0)))
        self.release_pressure = max(
            0.0, float(self.config.get("lock_release_pressure_kpa", 30.0)))
        self.future_steps = max(1, int(self.config.get("future_observation_steps", 10)))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ms.NCH,), dtype=np.float32)
        # All fields are normalised; generous bounds retain fault visibility.
        self.observation_space = spaces.Box(
            low=-20.0, high=20.0, shape=(40,), dtype=np.float32)

        self.reference = np.zeros((1, 3), dtype=float)
        self.gui_reference = np.zeros((1, 3), dtype=float)
        self.state = np.zeros(12, dtype=float)
        self.index = 0
        self.base_u = np.zeros(ms.NCH, dtype=float)
        self.total_u = np.zeros(ms.NCH, dtype=float)
        self.residual_u = np.zeros(ms.NCH, dtype=float)
        self.lock_states = ["o"] * ms.NCH
        self.lock_ticks = np.zeros(ms.NCH, dtype=int)
        self.actuator_gain = np.ones(ms.NCH, dtype=float)
        self.dynamics_speed = 1.0
        self.error_sq_sum = 0.0
        self.axis_error_sq_sum = np.zeros(3, dtype=float)
        self.steps_taken = 0
        self.solve_ms_sum = 0.0
        self.saturation_count = 0
        self._last_info: Dict[str, Any] = {}

    def _sample_uniform(self, key: str, default: Tuple[float, float]) -> float:
        bounds = self.config.get(key, list(default))
        low, high = float(bounds[0]), float(bounds[1])
        return float(self.np_random.uniform(low, high)) if self.training else 0.5*(low+high)

    def _reset_model_history(self) -> None:
        for model in (self.controller_model, self.plant_model):
            if hasattr(model, "reset_history"):
                model.reset_history()

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        randomise = self.training and not bool((options or {}).get("deterministic", False))
        ref_rng = self.np_random if randomise else None
        self.reference, self.gui_reference = build_reference(self.config, ref_rng)

        rest = np.asarray(self.plant_model.meta.get(
            "rest_pressure_kpa", [2.0]*ms.NCH), dtype=float)
        self.state = np.r_[np.zeros(6), rest].astype(float)
        if randomise:
            angle_std = math.radians(float(self.config.get("initial_angle_std_deg", 0.25)))
            self.state[:3] += self.np_random.normal(0.0, angle_std, 3)
            self.state[6:] += self.np_random.normal(
                0.0, float(self.config.get("initial_pressure_std_kpa", 0.5)), ms.NCH)
            gain_bounds = self.config.get("actuator_gain_range", [0.9, 1.1])
            self.actuator_gain = self.np_random.uniform(
                float(gain_bounds[0]), float(gain_bounds[1]), ms.NCH)
            self.dynamics_speed = self._sample_uniform(
                "dynamics_speed_range", (0.92, 1.08))
        else:
            self.actuator_gain[:] = 1.0
            self.dynamics_speed = 1.0
        self.state[6:] = np.clip(self.state[6:], 0.0, ms.P_MAX_KPA)

        self.index = 0
        self.base_u[:] = 0.0
        self.total_u[:] = 0.0
        self.residual_u[:] = 0.0
        self.lock_states = ["o"] * ms.NCH
        self.lock_ticks[:] = 0
        self.error_sq_sum = 0.0
        self.axis_error_sq_sum[:] = 0.0
        self.steps_taken = 0
        self.solve_ms_sum = 0.0
        self.saturation_count = 0
        self._last_info = {}
        self.mpc.reset_runtime()
        self._reset_model_history()
        return self._observation(), self._info()

    def _reference_horizon(self):
        n = int(self.mpc.N)
        return [self.reference[min(self.index+j, len(self.reference)-1)]
                for j in range(1, n+1)]

    def _update_locks(self, command: np.ndarray, pressure: np.ndarray) -> None:
        lead = max(1, int(self.mpc.lock_lead_steps))
        for channel in range(ms.NCH):
            request = bool(command[channel])
            state = self.lock_states[channel]
            if state == "o":
                if request:
                    self.lock_states[channel] = "w"
                    self.lock_ticks[channel] = 1
            elif state == "w":
                if not request:
                    self.lock_states[channel] = "o"
                    self.lock_ticks[channel] = 0
                else:
                    self.lock_ticks[channel] += 1
                    if self.lock_ticks[channel] >= lead:
                        self.lock_states[channel] = "L"
                        self.lock_ticks[channel] = lead
            elif not request and float(pressure[channel]) <= self.release_pressure:
                self.lock_states[channel] = "o"
                self.lock_ticks[channel] = 0

    @staticmethod
    def _residual_antagonist_project(candidate: np.ndarray,
                                     base: np.ndarray) -> np.ndarray:
        """Enforce antagonist pairs without allowing a small residual to erase base MPC."""
        answer = np.asarray(candidate, dtype=float).copy()
        for left, right in ms.ANTAGONIST_PAIRS:
            if base[left] > 0.5 and base[right] <= 0.5:
                answer[right] = 0.0
            elif base[right] > 0.5 and base[left] <= 0.5:
                answer[left] = 0.0
            elif answer[left] > 0.0 and answer[right] > 0.0:
                if answer[left] >= answer[right]:
                    answer[right] = 0.0
                else:
                    answer[left] = 0.0
        return answer

    def _apply_residual(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        desired = (action*self.max_residual_mv if self.residual_enabled
                   else np.zeros(ms.NCH, dtype=float))
        locked = np.asarray([state == "L" for state in self.lock_states], dtype=bool)
        desired[~locked] = 0.0
        du = self.residual_slew_mv_s*ms.DT
        residual = np.clip(desired, self.residual_u-du, self.residual_u+du)
        candidate = np.clip(self.base_u+residual, 0.0, self.u_max)
        total = self._residual_antagonist_project(candidate, self.base_u)
        total[~locked] = 0.0
        self.residual_u = total-self.base_u
        return total

    def _plant_step(self, control: np.ndarray) -> np.ndarray:
        old_state = self.state.copy()
        actual_control = np.clip(control*self.actuator_gain, 0.0, self.u_max)
        if hasattr(self.plant_model, "begin_rollout"):
            self.plant_model.begin_rollout(old_state)
        nominal = self.plant_model._predict_batch(
            old_state[None, :], actual_control[None, :])[0]
        next_state = old_state + self.dynamics_speed*(np.asarray(nominal, float)-old_state)

        if self.training:
            q_noise = math.radians(float(self.config.get("angle_noise_std_deg", 0.025)))
            next_state[:3] += self.np_random.normal(0.0, q_noise, 3)
            next_state[6:] += self.np_random.normal(
                0.0, float(self.config.get("pressure_noise_std_kpa", 0.35)), ms.NCH)
        next_state[3:6] = next_state[:3]-old_state[:3]
        next_state[3:6] = np.clip(next_state[3:6], -0.30, 0.30)
        next_state[6:] = np.clip(next_state[6:], 0.0, ms.P_MAX_KPA)
        if hasattr(self.plant_model, "observe_actual"):
            self.plant_model.observe_actual(old_state, actual_control)
        if hasattr(self.controller_model, "observe_actual"):
            self.controller_model.observe_actual(old_state, control)
        return next_state

    def _gui_velocity_deg_s(self) -> np.ndarray:
        gui_now = np.asarray(ms.st3_from_q_model(self.state[:3]), dtype=float)
        gui_previous = np.asarray(
            ms.st3_from_q_model(self.state[:3]-self.state[3:6]), dtype=float)
        return _shortest_angle_deg(gui_now-gui_previous)/ms.DT

    def step(self, action):
        old_residual = self.residual_u.copy()
        self.base_u, _ = self.mpc.step(
            self.state.copy(), self._reference_horizon(),
            lock_states=self.lock_states)
        self.base_u = np.asarray(self.base_u, dtype=float)
        lock_command = np.asarray(self.mpc.lock_cmd, dtype=int)
        self.total_u = self._apply_residual(action)
        # DAMPC's delta-u term must start from the voltage actually applied to
        # the simulated plant, not from the pre-residual base command.
        self.mpc.u_prev[:] = self.total_u
        self.state = self._plant_step(self.total_u)
        # The lock command issued in this tick becomes visible to the next
        # 20 Hz tick.  Therefore two 50 ms pre-lock ticks cannot accidentally
        # receive a PPO voltage on the tick that merely changes w -> L.
        self._update_locks(lock_command, self.state[6:])
        self.index = min(self.index+1, len(self.reference)-1)

        gui_actual = np.asarray(ms.st3_from_q_model(self.state[:3]), dtype=float)
        error = _shortest_angle_deg(gui_actual-self.gui_reference[self.index])
        tracking_sq = float(np.dot(error, error))
        self.error_sq_sum += tracking_sq
        self.axis_error_sq_sum += error*error
        self.steps_taken += 1
        self.solve_ms_sum += float(getattr(self.mpc, "last_solve_ms", 0.0))
        saturated = int(np.count_nonzero(self.total_u >= self.u_max-1.0))
        self.saturation_count += saturated

        previous_ref = self.gui_reference[max(0, self.index-1)]
        reference_velocity = _shortest_angle_deg(
            self.gui_reference[self.index]-previous_ref)/ms.DT
        actual_velocity = self._gui_velocity_deg_s()
        reward_cfg = self.config.get("reward", {})
        reward = -tracking_sq/max(1e-9, float(
            reward_cfg.get("tracking_scale_deg2", 400.0)))
        velocity_error_sq = float(np.dot(actual_velocity-reference_velocity,
                                             actual_velocity-reference_velocity))
        reward -= float(reward_cfg.get("velocity_weight", 0.01))*velocity_error_sq/max(
            1e-9, float(reward_cfg.get("velocity_scale_deg_s2", 400.0)))
        residual_normalised = self.residual_u/max(1.0, self.max_residual_mv)
        delta_normalised = (self.residual_u-old_residual)/max(1.0, self.max_residual_mv)
        reward -= float(reward_cfg.get("residual_l2_weight", 0.015))*float(
            np.dot(residual_normalised, residual_normalised))
        reward -= float(reward_cfg.get("residual_delta_weight", 0.03))*float(
            np.dot(delta_normalised, delta_normalised))
        reward -= float(reward_cfg.get("saturation_weight", 0.02))*saturated

        finite = bool(np.all(np.isfinite(self.state)))
        excessive = bool(np.any(np.abs(gui_actual) > 120.0) or
                         np.any(self.state[6:] > ms.P_MAX_KPA-1e-6))
        terminated = not finite or excessive
        truncated = self.index >= len(self.reference)-1
        if terminated:
            reward -= float(reward_cfg.get("unsafe_penalty", 100.0))
        info = self._info(error=error, tracking_sq=tracking_sq)
        self._last_info = info
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        gui_actual = np.asarray(ms.st3_from_q_model(self.state[:3]), dtype=float)
        ref = self.gui_reference[min(self.index, len(self.gui_reference)-1)]
        future = self.gui_reference[min(self.index+self.future_steps,
                                        len(self.gui_reference)-1)]
        error = _shortest_angle_deg(ref-gui_actual)
        lock_values = np.asarray([
            0.0 if value == "o" else 0.5 if value == "w" else 1.0
            for value in self.lock_states], dtype=float)
        fields = [
            gui_actual/60.0,
            self._gui_velocity_deg_s()/60.0,
            self.state[6:]/200.0,
            ref/60.0,
            future/60.0,
            error/60.0,
            self.base_u/max(1.0, self.u_max),
            self.total_u/max(1.0, self.u_max),
            lock_values,
            np.asarray([self.index/max(1, len(self.reference)-1)], dtype=float)]
        observation = np.concatenate(fields).astype(np.float32)
        return np.clip(observation, -20.0, 20.0)

    def _info(self, error: Optional[np.ndarray] = None,
              tracking_sq: Optional[float] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "reference_index": int(self.index),
            "episode_steps": int(self.steps_taken),
            "base_u_mv": self.base_u.copy(),
            "residual_u_mv": self.residual_u.copy(),
            "total_u_mv": self.total_u.copy(),
            "lock_states": "".join(self.lock_states),
            "solver_ms": float(getattr(self.mpc, "last_solve_ms", 0.0))}
        if error is not None:
            info["tracking_error_gui_deg"] = np.asarray(error, dtype=float).copy()
        if tracking_sq is not None:
            info["tracking_sq_deg2"] = float(tracking_sq)
        if self.steps_taken:
            mse = self.error_sq_sum/self.steps_taken
            axis_mse = self.axis_error_sq_sum/self.steps_taken
            info.update({
                "tracking_mse_3axis_deg2": float(mse),
                "tracking_rmse_vector_deg": float(math.sqrt(mse)),
                "axis_rmse_deg": np.sqrt(axis_mse).copy(),
                "mean_solver_ms": float(self.solve_ms_sum/self.steps_taken),
                "saturation_count": int(self.saturation_count)})
        return info


def make_env(config_path: Optional[os.PathLike] = None, *, training: bool = True,
             residual_enabled: bool = True,
             config_overrides: Optional[Dict[str, Any]] = None):
    return ResidualPneumaticEnv(
        config_path=config_path, training=training,
        residual_enabled=residual_enabled,
        config_overrides=config_overrides)

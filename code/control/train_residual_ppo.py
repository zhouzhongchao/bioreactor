# -*- coding: utf-8 -*-
"""Train/evaluate bounded residual PPO on the learned pneumatic simulator."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

# Importing rl_sim_env first intentionally primes NumPy LAPACK before PyTorch.
try:
    from .rl_sim_env import DEFAULT_CONFIG_PATH, load_rl_config, make_env
except ImportError:
    from rl_sim_env import DEFAULT_CONFIG_PATH, load_rl_config, make_env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise SystemExit(
        "缺少 Gymnasium/Stable-Baselines3。请先激活 bioreactor 环境后运行：\n"
        "python -m pip install gymnasium==0.28.1 stable-baselines3==2.0.0") from exc


HERE = Path(__file__).resolve().parent
from project_paths import CONTROL_LOG_DIR
MODEL_DIR = HERE / "rl_models"
LOG_DIR = CONTROL_LOG_DIR / "rl_sim"
MODEL_PATH = MODEL_DIR / "residual_ppo"


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def run_episode(model, config_path: Path, seed: int, residual_enabled: bool):
    # Keep simulation randomisation enabled during evaluation.  Reusing the
    # same seed for MPC and MPC+PPO creates a paired, fair disturbance test.
    env = make_env(config_path, training=True,
                   residual_enabled=residual_enabled)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    reward_sum = 0.0
    trace = []
    while not (terminated or truncated):
        if residual_enabled and model is not None:
            action, _ = model.predict(observation, deterministic=True)
        else:
            action = env.action_space.sample()*0.0
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        trace.append({
            "t_s": env.index*0.05,
            "reference_x_deg": env.gui_reference[env.index, 0],
            "reference_y_deg": env.gui_reference[env.index, 1],
            "reference_phi_deg": env.gui_reference[env.index, 2],
            "actual_x_deg": info.get("tracking_error_gui_deg", [0, 0, 0])[0]
                            + env.gui_reference[env.index, 0],
            "actual_y_deg": info.get("tracking_error_gui_deg", [0, 0, 0])[1]
                            + env.gui_reference[env.index, 1],
            "actual_phi_deg": info.get("tracking_error_gui_deg", [0, 0, 0])[2]
                              + env.gui_reference[env.index, 2],
            "tracking_sq_deg2": info.get("tracking_sq_deg2", float("nan")),
            **{f"base_u{i}_mv": env.base_u[i] for i in range(6)},
            **{f"residual_u{i}_mv": env.residual_u[i] for i in range(6)},
            **{f"total_u{i}_mv": env.total_u[i] for i in range(6)}})
    summary = dict(info)
    summary["reward_sum"] = float(reward_sum)
    summary["seed"] = int(seed)
    summary["controller"] = "MPC+PPO" if residual_enabled else "MPC"
    env.close()
    return summary, trace


def evaluate(model, config_path: Path, episodes: int, seed: int):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = {}
    for offset in range(episodes):
        episode_seed = seed+offset
        for enabled in (False, True):
            summary, trace = run_episode(
                model, config_path, episode_seed, enabled)
            rows.append(summary)
            traces[(offset, enabled)] = trace

    def average(name, enabled):
        values = [float(row[name]) for row in rows
                  if row["controller"] == ("MPC+PPO" if enabled else "MPC")]
        return float(sum(values)/max(1, len(values)))

    result = {
        "episodes": int(episodes),
        "mpc_tracking_mse_3axis_deg2": average(
            "tracking_mse_3axis_deg2", False),
        "ppo_tracking_mse_3axis_deg2": average(
            "tracking_mse_3axis_deg2", True),
        "mpc_tracking_rmse_vector_deg": average(
            "tracking_rmse_vector_deg", False),
        "ppo_tracking_rmse_vector_deg": average(
            "tracking_rmse_vector_deg", True),
        "mpc_saturation_count": average("saturation_count", False),
        "ppo_saturation_count": average("saturation_count", True)}
    base = result["mpc_tracking_mse_3axis_deg2"]
    candidate = result["ppo_tracking_mse_3axis_deg2"]
    result["mse_improvement_percent"] = (
        100.0*(base-candidate)/base if base > 1e-12 else 0.0)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = LOG_DIR / f"ppo_eval_{stamp}.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump({"summary": result,
                   "episodes_detail": [{k: _jsonable(v) for k, v in row.items()}
                                       for row in rows]},
                  stream, ensure_ascii=False, indent=2)
    trace_path = LOG_DIR / f"ppo_eval_trace_{stamp}.csv"
    fieldnames = list(next(iter(traces.values()))[0].keys())
    with trace_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "controller"]+fieldnames)
        writer.writeheader()
        for (episode, enabled), trace in traces.items():
            for item in trace:
                writer.writerow({"episode": episode,
                                 "controller": "MPC+PPO" if enabled else "MPC",
                                 **item})
    print("\n# 仿真评估（相同随机种子成对比较）")
    print(f"  纯MPC：三轴合计MSE {base:.3f} 平方度，向量RMSE "
          f"{result['mpc_tracking_rmse_vector_deg']:.3f}°")
    print(f"  MPC+PPO：三轴合计MSE {candidate:.3f} 平方度，向量RMSE "
          f"{result['ppo_tracking_rmse_vector_deg']:.3f}°")
    print(f"  MSE变化：{result['mse_improvement_percent']:+.1f}%（正数才是改善）")
    print(f"  报告：{report_path}")
    print(f"  逐拍数据：{trace_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="训练 MPC residual PPO（纯仿真，不接真机）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="只训练256步，用于检查环境和文件链路")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_rl_config(config_path)
    ppo_cfg = config.get("ppo", {})
    seed = int(ppo_cfg.get("seed", 20260909))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_zip = Path(str(MODEL_PATH)+".zip")
    if args.eval_only:
        if not model_zip.exists():
            raise SystemExit(f"找不到已训练策略：{model_zip}")
        model = PPO.load(str(MODEL_PATH), device=str(ppo_cfg.get("device", "cpu")))
        evaluate(model, config_path, args.eval_episodes, seed)
        return

    raw_env = make_env(config_path, training=True, residual_enabled=True)
    check_env(raw_env, warn=True, skip_render_check=True)
    raw_env.close()
    env = Monitor(make_env(config_path, training=True, residual_enabled=True),
                  filename=str(LOG_DIR / "ppo_train_monitor.csv"))

    smoke = bool(args.smoke)
    total_timesteps = (256 if smoke else int(
        args.timesteps or ppo_cfg.get("total_timesteps", 100000)))
    n_steps = 128 if smoke else int(ppo_cfg.get("n_steps", 512))
    batch_size = min(int(ppo_cfg.get("batch_size", 64)), n_steps)
    n_epochs = 1 if smoke else int(ppo_cfg.get("n_epochs", 10))
    layers = [int(v) for v in ppo_cfg.get("policy_layers", [128, 128])]
    model = PPO(
        "MlpPolicy", env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.001)),
        policy_kwargs={"net_arch": dict(pi=layers, vf=layers)},
        seed=seed, device=str(ppo_cfg.get("device", "cpu")), verbose=1)
    checkpoint_every = max(1, int(ppo_cfg.get("checkpoint_every_steps", 25000)))
    callback = CheckpointCallback(
        save_freq=checkpoint_every, save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="residual_ppo")
    print(f"# 开始纯仿真训练：{total_timesteps}步；不会连接串口。")
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=False)
    save_path = (MODEL_DIR / "residual_ppo_smoke") if smoke else MODEL_PATH
    model.save(str(save_path))
    env.close()
    saved_zip = Path(str(save_path)+".zip")
    print(f"# {'冒烟测试' if smoke else '正式'}PPO策略已保存：{saved_zip}")
    evaluate(model, config_path, args.eval_episodes, seed)


if __name__ == "__main__":
    main()

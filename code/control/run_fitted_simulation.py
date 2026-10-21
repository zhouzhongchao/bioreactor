# -*- coding: utf-8 -*-
"""Run the fixed task completely in fitted-model simulation and plot results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

# This import primes NumPy before any PyTorch/SB3 import on Windows.
try:
    from .rl_sim_env import DEFAULT_CONFIG_PATH, make_env
    from . import mpc_station as ms
except ImportError:
    from rl_sim_env import DEFAULT_CONFIG_PATH, make_env
    import mpc_station as ms


HERE = Path(__file__).resolve().parent
from project_paths import CONTROL_LOG_DIR
OUTPUT_DIR = CONTROL_LOG_DIR / "fitted_sim"


def _wrapped_error(actual, reference):
    error = np.asarray(actual, dtype=float)-np.asarray(reference, dtype=float)
    error[2] = (error[2]+180.0) % 360.0-180.0
    return error


def _run(args):
    overrides = {}
    if args.controller:
        overrides["controller_forward_model"] = args.controller
    if args.plant:
        overrides["plant_forward_model"] = args.plant
    use_policy = bool(args.ppo or args.ppo_smoke)
    env = make_env(args.config, training=bool(args.randomized),
                   residual_enabled=use_policy, config_overrides=overrides)
    policy = None
    if use_policy:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise SystemExit("PPO策略仿真需要 stable-baselines3") from exc
        model_name = "residual_ppo_smoke" if args.ppo_smoke else "residual_ppo"
        model_path = HERE / "rl_models" / model_name
        if not Path(str(model_path)+".zip").exists():
            raise SystemExit(f"找不到PPO策略：{model_path}.zip")
        policy = PPO.load(str(model_path), device="cpu")

    observation, _ = env.reset(seed=args.seed)
    rows = []
    errors = []
    terminated = truncated = False
    reward_sum = 0.0
    while not (terminated or truncated):
        if policy is None:
            action = np.zeros(6, dtype=np.float32)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        actual = np.asarray(ms.st3_from_q_model(env.state[:3]), dtype=float)
        reference = env.gui_reference[env.index].copy()
        error = _wrapped_error(actual, reference)
        errors.append(error)
        row = {
            "t_s": env.index*ms.DT,
            **{f"reference_{name}_deg": reference[i]
               for i, name in enumerate(("x", "y", "phi"))},
            **{f"actual_{name}_deg": actual[i]
               for i, name in enumerate(("x", "y", "phi"))},
            **{f"error_{name}_deg": error[i]
               for i, name in enumerate(("x", "y", "phi"))},
            **{f"u{i}_mv": env.total_u[i] for i in range(6)},
            **{f"base_u{i}_mv": env.base_u[i] for i in range(6)},
            **{f"residual_u{i}_mv": env.residual_u[i] for i in range(6)},
            **{f"p{i}_kpa": env.state[6+i] for i in range(6)},
            "locks": "".join(env.lock_states),
            "reward": reward}
        rows.append(row)

    error_array = np.asarray(errors, dtype=float)
    axis_mse = np.mean(error_array*error_array, axis=0)
    mse_sum = float(np.sum(axis_mse))
    summary = {
        "simulation_only_no_serial": True,
        "controller_forward_model": env.controller_arch,
        "simulated_plant_forward_model": env.plant_arch,
        "controller_training_samples": int(getattr(env.controller_model, "n_fit", 0)),
        "plant_training_samples": int(getattr(env.plant_model, "n_fit", 0)),
        "randomized": bool(args.randomized),
        "residual_policy": ("smoke" if args.ppo_smoke else
                            "trained" if args.ppo else "none"),
        "steps": int(len(rows)),
        "duration_s": float(len(rows)*ms.DT),
        "tracking_mse_3axis_deg2": mse_sum,
        "tracking_rmse_vector_deg": float(math.sqrt(mse_sum)),
        "axis_mse_deg2": axis_mse.tolist(),
        "axis_rmse_deg": np.sqrt(axis_mse).tolist(),
        "axis_max_abs_error_deg": np.max(np.abs(error_array), axis=0).tolist(),
        "reward_sum": reward_sum,
        "saturation_count": int(info.get("saturation_count", 0))}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"sim_{env.controller_arch}_to_{env.plant_arch}_{stamp}"
    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    json_path = OUTPUT_DIR / f"{prefix}.json"
    png_path = OUTPUT_DIR / f"{prefix}.png"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray([row["t_s"] for row in rows], dtype=float)
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    names = ("x", "y", "phi")
    for axis, name in zip(axes[:3], names):
        axis.plot(t, [row[f"reference_{name}_deg"] for row in rows],
                  "k--", linewidth=2.0, label="reference")
        axis.plot(t, [row[f"actual_{name}_deg"] for row in rows],
                  linewidth=1.6, label="simulation response")
        axis.set_ylabel(f"{name} (deg)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    for channel in range(6):
        axes[3].plot(t, [row[f"u{channel}_mv"] for row in rows], label=f"V{channel}")
        axes[4].plot(t, [row[f"p{channel}_kpa"] for row in rows], label=f"P{channel}")
    axes[3].set_ylabel("command (mV)")
    axes[4].set_ylabel("pressure (kPa)")
    axes[4].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.3); axes[4].grid(True, alpha=0.3)
    axes[3].legend(ncol=6, loc="upper center")
    axes[4].legend(ncol=6, loc="upper center")
    fig.suptitle(
        f"Fitted-model simulation: MPC {env.controller_arch} -> plant {env.plant_arch}\n"
        f"3-axis MSE={mse_sum:.2f} deg^2, vector RMSE={math.sqrt(mse_sum):.2f} deg")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    env.close()
    return summary, csv_path, json_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="用真机数据拟合的顺模型跑完整固定任务仿真（不接串口）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--controller", choices=("mlp", "history_mlp", "gru"))
    parser.add_argument("--plant", choices=("mlp", "history_mlp", "gru"))
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--ppo", action="store_true", help="加载正式 residual PPO")
    parser.add_argument("--ppo-smoke", action="store_true", help="只加载冒烟测试策略")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.ppo and args.ppo_smoke:
        parser.error("--ppo 和 --ppo-smoke 不能同时使用")
    summary, csv_path, json_path, png_path = _run(args)
    print("# 拟合模型纯仿真完成（未连接串口）")
    print(f"  MPC真正模型：{summary['controller_forward_model']}；"
          f"虚拟真机：{summary['simulated_plant_forward_model']}")
    print(f"  三轴合计MSE：{summary['tracking_mse_3axis_deg2']:.3f} 平方度；"
          f"向量RMSE：{summary['tracking_rmse_vector_deg']:.3f}°")
    print("  各轴RMSE [X,Y,phi]：" + "/".join(
        f"{value:.3f}°" for value in summary["axis_rmse_deg"]))
    print("  各轴最大绝对误差 [X,Y,phi]：" + "/".join(
        f"{value:.3f}°" for value in summary["axis_max_abs_error_deg"]))
    print(f"  曲线图：{png_path}")
    print(f"  逐拍数据：{csv_path}")
    print(f"  指标：{json_path}")


if __name__ == "__main__":
    main()

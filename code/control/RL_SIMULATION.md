# MPC + residual PPO 仿真训练

这个分支只在已有顺模型中训练，不连接串口，也不会修改 `gains.json`、真正模型或影子模型。

控制链为：

`现有 MPC 电压 + PPO 小残差 → 残差限幅/斜率限制 → 拮抗肌/锁状态/总电压投影 → 仿真顺模型`

默认设置：

- 控制器顺模型：读取 `gains.json` 的 `mpc_forward_model`。
- 仿真对象：10 拍 GRU，避免 MPC 和仿真对象完全相同。
- 频率：20 Hz；固定重要任务轨迹；每轮对轨迹幅值、阀增益、动态速度和噪声做小范围随机化。
- PPO 只能修正已锁通道，单路残差最大 250 mV，残差变化率最大 1200 mV/s。

在 VS Code 选择 `bioreactor` 解释器后运行：

```powershell
python control/train_residual_ppo.py --smoke
python control/train_residual_ppo.py --timesteps 100000 --eval-episodes 10
python control/train_residual_ppo.py --eval-only --eval-episodes 20
```

策略保存在 `control/rl_models/residual_ppo.zip`。评估报告和逐拍曲线数据保存在 `control/logs/rl_sim/`；报告会用完全相同的种子成对比较纯 MPC 与 MPC+PPO 的真实三轴合计跟踪 MSE。

所有可调项集中在 `rl_sim_config.json`。只有仿真评估持续优于纯 MPC 后，才应该另做真机只读/小残差接入；这个脚本本身不会驱动真机。

只看现有拟合模型和 MPC 的纯仿真响应（自动生成 PNG、CSV 和 JSON）可运行：

```powershell
# 理想同模型：MPC和虚拟真机都用单拍MLP
python control/run_fitted_simulation.py --controller mlp --plant mlp

# 模型失配：MPC用单拍MLP，虚拟真机用10拍GRU
python control/run_fitted_simulation.py --controller mlp --plant gru
```

结果保存在 `control/logs/fitted_sim/`，两个命令都不会打开串口。

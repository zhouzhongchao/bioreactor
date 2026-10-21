# -*- coding: utf-8 -*-
"""Render the TRO AI supervision architecture as local PNG/SVG figures."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


def _configure_chinese_font():
    preferred = ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei",
                 "Noto Sans CJK SC", "Arial Unicode MS")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def render(output_dir: str | os.PathLike):
    _configure_chinese_font()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="#f5f7fa")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, color="#ffffff", edge="#8a98a8",
            title_color="#183153", body_size=10.0, linewidth=1.4):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.10",
            facecolor=color, edgecolor=edge, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(x + 0.16, y + h - 0.22, title, ha="left", va="top",
                fontsize=13, fontweight="bold", color=title_color)
        ax.text(x + 0.16, y + h - 0.62, body, ha="left", va="top",
                fontsize=body_size, color="#334155", linespacing=1.45)
        return patch

    def arrow(start, end, color="#557086", width=1.8, style="-|>", rad=0.0):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                    connectionstyle=f"arc3,rad={rad}"))

    ax.text(0.7, 9.55, "TRO 气动机器人：AI 接管与监控框架", fontsize=22,
            fontweight="bold", color="#102a43", va="top")
    ax.text(0.72, 9.08,
            "LLM 只做低频诊断和受限补偿器设计；20 Hz 控制与安全否决始终留在本机",
            fontsize=12, color="#526777", va="top")

    # Real-time deterministic path.
    ax.text(0.72, 8.55, "实时确定性通路", fontsize=13, fontweight="bold",
            color="#2463a7")
    box(0.7, 6.65, 2.45, 1.55, "传感器", "IMU 四元数\n6 路压力 / 锁状态\n编码器 / 肌肉长度\n相机帧（仅本机）",
        color="#eef6ff", edge="#5b8fc5", body_size=9.5)
    box(3.75, 6.65, 2.6, 1.55, "状态融合与健康度", "时间同步与滤波\nX / Y / φ、角速度\n视觉置信度\n传感器一致性残差",
        color="#edf8f5", edge="#3c9b7a", body_size=9.5)
    box(6.95, 6.65, 2.7, 1.55, "安全监督 + MPC", "限位 / 超压 / 失联\n100 ms 预锁规划\n拮抗通道硬约束\nMPC 基础电压",
        color="#fff7e8", edge="#d2952f", body_size=9.5)
    box(10.25, 6.65, 2.55, 1.55, "Compensator", "受限 MIMO 残差\n仅作用于已锁通道\n幅值与斜率投影\n候选可自动回滚",
        color="#f5effb", edge="#8a62ae", body_size=9.5)
    box(13.4, 6.65, 1.9, 1.55, "固件 / 真机", "最终电压\n锁阀执行\n硬件急停",
        color="#fcefee", edge="#bb5b56", body_size=9.5)
    arrow((3.15, 7.43), (3.72, 7.43))
    arrow((6.35, 7.43), (6.92, 7.43))
    arrow((9.65, 7.43), (10.22, 7.43))
    arrow((12.80, 7.43), (13.37, 7.43))

    # Offline/event-driven AI path.
    ax.text(0.72, 5.95, "低频 AI 学习通路", fontsize=13, fontweight="bold",
            color="#6c4b8e")
    box(0.7, 4.30, 2.45, 1.25, "本地完整日志", "20 Hz 原始数据全部落盘\n环形缓冲保留故障前后窗口",
        color="#ffffff", edge="#8a98a8", body_size=9.5)
    box(3.75, 4.30, 2.60, 1.25, "事件摘要器", "每轴指标 + 六路汇总\n只选 10–24 个变化点",
        color="#ffffff", edge="#8a98a8", body_size=9.5)
    box(6.95, 4.30, 2.70, 1.25, "LLM 离线诊断", "判断误差阶段与耦合\n只返回候选 JSON",
        color="#f5effb", edge="#8a62ae", body_size=9.5)
    box(10.25, 4.30, 2.55, 1.25, "验证门", "Schema 与硬边界检查\n历史日志离线回放",
        color="#fff7e8", edge="#d2952f", body_size=9.5)
    box(13.40, 4.30, 1.90, 1.25, "晋级 / 回滚", "Shadow\n真机 Canary\n变差即回滚",
        color="#edf8f5", edge="#3c9b7a", body_size=9.2)
    arrow((3.15, 4.92), (3.72, 4.92))
    arrow((6.35, 4.92), (6.92, 4.92))
    arrow((9.65, 4.92), (10.22, 4.92))
    arrow((12.80, 4.92), (13.37, 4.92))
    arrow((14.35, 5.57), (12.50, 6.62), color="#6c4b8e", rad=-0.16)
    ax.text(13.45, 5.83, "仅合格候选", fontsize=9.5, color="#6c4b8e")
    arrow((1.92, 6.62), (1.92, 5.58), color="#557086")

    # Compact data and vision responsibilities.
    box(0.7, 1.75, 7.15, 1.85, "给 LLM 的最小动态包（建议每轮 1–3k token）",
        "• 身份：task / model / controller hash、candidate generation\n"
        "• 性能：X/Y/φ 的 RMSE、超调、滞后、稳态误差\n"
        "• 通道：压力峰值、响应延迟、饱和比例、锁可用率\n"
        "• 健康：IMU–视觉、编码器–V6 残差与故障标志\n"
        "• 事件：10–24 个关键变化点；不发送逐拍完整轨迹",
        color="#ffffff", edge="#8a98a8", body_size=9.4)
    box(8.35, 1.75, 6.95, 1.85, "相机支路（当前需要补齐）",
        "相机原始帧留在本机视觉进程；活动端使用 ArUco / AprilTag。\n"
        "视觉模块只发布：时间戳、X/Y/φ 或四元数、置信度、\n"
        "重投影误差、遮挡标志。常规情况下不给 LLM 图片；\n"
        "数值无法解释故障时，才人工选择 1–3 张降采样图片。",
        color="#eef6ff", edge="#5b8fc5", body_size=9.4)

    safety = FancyBboxPatch(
        (0.7, 0.62), 14.6, 0.70,
        boxstyle="round,pad=0.018,rounding_size=0.10",
        facecolor="#fff0f0", edgecolor="#c44949", linewidth=1.5)
    ax.add_patch(safety)
    ax.text(0.95, 0.97,
            "不可被 AI 绕过：人工急停｜215 kPa 超压｜机械碰撞限位｜传感器超时｜100 ms 预锁｜拮抗互斥｜30 kPa 仅用于解锁/对抗交棒",
            fontsize=10.5, color="#8b2c2c", fontweight="bold", va="center")
    ax.text(15.25, 0.25, "TRO V6 · local architecture figure", fontsize=8.5,
            color="#7b8794", ha="right")

    png = output / "TRO_AI_supervision_architecture.png"
    svg = output / "TRO_AI_supervision_architecture.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(png), str(svg)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    target = here.parent / "output" / "figures"
    for generated in render(target):
        print(generated)

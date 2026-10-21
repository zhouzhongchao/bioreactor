"""
shoulder_pam_model.py
=====================
仿生肩关节机器人系统的几何与动力学模型
—— 基于 8 根气动人工肌肉 (PAM) 驱动的三自由度球关节 (按 fin_Model.pdf 实现)

约定
----
* 固定坐标系 {B}: 原点 O_B, 与固定基座固连.
* 运动坐标系 {M}: 原点 O_M 与球关节中心 O 重合, 随运动臂转动.
* ^B p_BO = O_B O_M = [-131, 0, 0]^T mm                                         (1)
* 广义坐标 q = [q1, q2, q3]^T: q1 抬臂角, q2 抬臂平面角, q3 肱骨轴向旋转角     (2)
* ^B R_M(q) = Rz(q2) Ry(-q1) Rx(q3)                                             (3)
* 几何量内部采用 mm; 进入力矩 / 动力学计算时统一换算为 m (MM2M).
* 单位: 长度 mm 或 m (函数名/文档中注明), 角度 rad, 力 N, 力矩 N·m.

公式编号与 PDF 一致, 便于对照.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 0. 常量几何参数 (表 1)
# ---------------------------------------------------------------------------
MM2M: float = 1e-3          # mm -> m
G_ACC: float = 9.81         # 重力加速度 m/s^2
N_PAM: int = 8

# 球心在固定坐标系中的位置 ^B p_BO (mm)                                   (1)
P_BO_MM = np.array([-131.0, 0.0, 0.0])

# 第 i 根 PAM 固定端连接点 ^B A_i (mm)
A_MM = np.array([
    [187.0,  56.0,  23.0],
    [187.0,  23.0,  56.0],
    [187.0, -23.0,  56.0],
    [187.0, -56.0,  23.0],
    [187.0,  56.0, -23.0],
    [187.0,  23.0, -56.0],
    [187.0, -23.0, -56.0],
    [187.0, -56.0, -23.0],
])

# 第 i 根 PAM 运动端连接点 ^M b_i (mm)
B_MM = np.array([
    [20.0,  20.0,   8.0],
    [20.0,   8.0,  20.0],
    [20.0,  -8.0,  20.0],
    [20.0, -20.0,   8.0],
    [20.0,  20.0,  -8.0],
    [20.0,   8.0, -20.0],
    [20.0,  -8.0, -20.0],
    [20.0, -20.0,  -8.0],
])

# 平移到球心的固定端向量 a_i = ^B A_i - ^B p_BO = (318, A_yi, A_zi) (mm)
A_TILDE_MM = A_MM - P_BO_MM

# ||a_i||^2 + ||b_i||^2, 8 根 PAM 均为 105653 mm^2                           (8a)
NORM2_CONST_MM2 = float(np.sum(A_TILDE_MM[0] ** 2) + np.sum(B_MM[0] ** 2))
assert abs(NORM2_CONST_MM2 - 105653.0) < 1e-9

# 按根的 ||a_i||^2+||b_i||^2 向量。前 8 根同值 (8a)；第 9/10 根不同值,
# 所以 (7) 必须用这个向量而不是常量——见 pam_lengths_mm。
NORM2_VEC_MM2 = np.sum(A_TILDE_MM ** 2, axis=1) + np.sum(B_MM ** 2, axis=1)

# ---------------------------------------------------------------------------
# 0b. v5.1 10PAM 扩展：第 9/10 根（斜向，驱动 q3）
# ---------------------------------------------------------------------------
# 几何来源与 GUI 定版一致（v5.1 表 1）：新 CAD Procrustes 配准（16 对应点，
# rms 2.19 mm），第 10 根 = 第 9 根的 y 镜像。滑环实物确认<<不滑动>>（装配
# 位置固定），故 b_9/b_10 为常量，9/10 与前 8 根完全同构，无滑移自由度。
# 坐标系：与表 1 相同，以 O_M 为原点（若启用 GUI 的 PIVOT_DX=66 转动中心
# 修正，此处与前 8 根须一起平移——本文件与表 1 保持一致，未平移）。
A_TILDE_AUX_MM = np.array([
    [161.54, -111.50, 63.18],     # a_9  （支撑柱 C 侧挂点，O_M 原点系）
    [161.54,  111.50, 63.18],     # a_10 = a_9 的 y 镜像
])
B_AUX_MM = np.array([
    [26.23, -20.49, -110.55],     # b_9  （安装环，固定）
    [26.23,  20.49, -110.55],     # b_10
])

def use_ten_pam() -> None:
    """切换为 10 根建模（默认 8 根，向后兼容；各几何函数形状变 (10,3)）。"""
    global N_PAM, A_TILDE_MM, B_MM, A_MM, NORM2_VEC_MM2
    if N_PAM == 10:
        return
    A_TILDE_MM = np.vstack([A_TILDE_MM, A_TILDE_AUX_MM])
    A_MM = np.vstack([A_MM, A_TILDE_AUX_MM + P_BO_MM])
    B_MM = np.vstack([B_MM, B_AUX_MM])
    NORM2_VEC_MM2 = np.sum(A_TILDE_MM ** 2, axis=1) + np.sum(B_MM ** 2, axis=1)
    N_PAM = 10


def set_pivot_dx(dx_mm: float = 66.0) -> None:
    """实际转动中心修正（作者 2026-08-30 两次确认 Δx=66 mm，GUI 同款）：
    实际球心相对表 1 的 O_M 沿 x_B 偏移 dx。a_i 与 b_i 同时平移 −dx——
    q=0 时 l_i = a_i − b_i 不变（长度不受影响），但力臂 r_i = R b_i 变化
    显著：环绕 8 根的 |b| 从 ~29mm 增到 ~53mm。辨识实测（2026-09-03）：
    不加此修正，模型对 V0..V3 的稳态响应低估 2~10 倍，而 9/10（|b|≈116mm）
    几乎不受影响——与该修正的力学含义完全吻合。"""
    global A_TILDE_MM, B_MM, NORM2_VEC_MM2
    off = np.array([dx_mm, 0.0, 0.0])
    A_TILDE_MM = A_TILDE_MM - off
    B_MM = B_MM - off
    NORM2_VEC_MM2 = np.sum(A_TILDE_MM ** 2, axis=1) + np.sum(B_MM ** 2, axis=1)


# ---------------------------------------------------------------------------
# 1. 运动学基础: 旋转矩阵与角速度映射
# ---------------------------------------------------------------------------
def rot_x(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_M(q: np.ndarray) -> np.ndarray:
    """^B R_M(q) = Rz(q2) Ry(-q1) Rx(q3), 显式形式 (4)."""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c3, s3 = np.cos(q3), np.sin(q3)
    return np.array([
        [c2 * c1, -s2 * c3 - c2 * s1 * s3,  s2 * s3 - c2 * s1 * c3],
        [s2 * c1,  c2 * c3 - s2 * s1 * s3, -c2 * s3 - s2 * s1 * c3],
        [s1,       c1 * s3,                 c1 * c3],
    ])


def E_matrix(q: np.ndarray) -> np.ndarray:
    """角速度映射 ω = E(q) q̇, 固定坐标系表示 (13b). 仅依赖 q1, q2."""
    q1, q2, _ = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    return np.array([
        [ s2, 0.0, c1 * c2],
        [-c2, 0.0, c1 * s2],
        [0.0, 1.0, s1],
    ])


def E_dot(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Ė(q, q̇) (27c)."""
    q1, q2, _ = q
    qd1, qd2, _ = qd
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    return np.array([
        [c2 * qd2, 0.0, -s1 * c2 * qd1 - c1 * s2 * qd2],
        [s2 * qd2, 0.0, -s1 * s2 * qd1 + c1 * c2 * qd2],
        [0.0,      0.0,  c1 * qd1],
    ])


def omega(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """空间角速度 ω = E(q) q̇ (13a)."""
    return E_matrix(q) @ qd


def omega_dot(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    """空间角加速度 ω̇ = E q̈ + Ė q̇ (27b)."""
    return E_matrix(q) @ qdd + E_dot(q, qd) @ qd


# ---------------------------------------------------------------------------
# 2. PAM 几何运动学
# ---------------------------------------------------------------------------
def moment_arm_vectors_mm(q: np.ndarray) -> np.ndarray:
    """球心到运动端连接点 r_i = ^B R_M(q) b_i (mm), 形状 (8,3)          (11)"""
    return B_MM @ R_M(q).T


def pam_points_mm(q: np.ndarray) -> np.ndarray:
    """运动端连接点在 {B} 中的位置 ^B p_i = ^B p_BO + R b_i (mm)        (5)"""
    return P_BO_MM + moment_arm_vectors_mm(q)


def pam_vectors_mm(q: np.ndarray) -> np.ndarray:
    """从运动端指向固定端的几何向量 l_i = a_i - R b_i (mm)               (6)"""
    return A_TILDE_MM - moment_arm_vectors_mm(q)


def pam_S(q: np.ndarray) -> np.ndarray:
    """S_i(q) = a_i^T R(q) b_i (mm^2), 附录 A 展开式的矩阵形式             (8a)"""
    R = R_M(q)
    return np.einsum("ij,jk,ik->i", A_TILDE_MM, R, B_MM)


def pam_lengths_mm(q: np.ndarray) -> np.ndarray:
    """PAM 瞬时几何长度 L_i = sqrt(||a_i||²+||b_i||² - 2 S_i) (mm)     (7)/(8a)
    前 8 根的 ||a||²+||b||² 同为 105653（8a 的常量捷径）；第 9/10 根不同值，
    故用按根向量 NORM2_VEC_MM2——8 根时两者完全等价。"""
    return np.sqrt(NORM2_VEC_MM2 - 2.0 * pam_S(q))


def pam_lengths_m(q: np.ndarray) -> np.ndarray:
    return pam_lengths_mm(q) * MM2M


def pam_unit_dirs(q: np.ndarray) -> np.ndarray:
    """单位拉力方向 u_i = l_i / L_i (无量纲), 形状 (8,3)                   (9)"""
    l = pam_vectors_mm(q)
    return l / np.linalg.norm(l, axis=1, keepdims=True)


def effective_arm_vectors_m(q: np.ndarray) -> np.ndarray:
    """三维有效力臂 h_i = r_i × u_i = (r_i × a_i)/L_i (m), 形状 (8,3)  (19)/(19a)"""
    r = moment_arm_vectors_mm(q)                      # mm
    L = pam_lengths_mm(q)                             # mm
    h_mm = np.cross(r, A_TILDE_MM) / L[:, None]       # mm
    return h_mm * MM2M


def H_matrix(q: np.ndarray) -> np.ndarray:
    """驱动映射矩阵 H(q) = [h_1 ... h_8] ∈ R^{3×8} (m)                    (20)"""
    return effective_arm_vectors_m(q).T


def J_L(q: np.ndarray) -> np.ndarray:
    """长度雅可比 J_L(q) = ∂L/∂q = -H^T E ∈ R^{8×3} (m/rad)            (15b)/(15c)"""
    return -H_matrix(q).T @ E_matrix(q)


def pam_length_rates_m(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """长度速度 L̇ = J_L(q) q̇ (m/s)                                        (15)"""
    return J_L(q) @ qd


def actuation_rank(q: np.ndarray, rtol: float = 1e-9) -> Tuple[int, np.ndarray, float]:
    """验证 rank H(q) = 3 (22). 返回 (rank, 奇异值, 条件数)."""
    sv = np.linalg.svd(H_matrix(q), compute_uv=False)
    rank = int(np.sum(sv > rtol * sv.max()))
    cond = sv.max() / sv.min() if sv.min() > 0 else np.inf
    return rank, sv, cond


# ---------------------------------------------------------------------------
# 3. PAM 拉力与球关节驱动力矩映射
# ---------------------------------------------------------------------------
def pam_forces_spatial(q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """空间力 f_i = F_i u_i (N), 形状 (8,3)                                 (16)"""
    return np.asarray(F)[:, None] * pam_unit_dirs(q)


def pam_torque_spatial(q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """关于球心的总驱动力矩 τ_PAM = H(q) F (N·m)                        (17)-(21)"""
    return H_matrix(q) @ np.asarray(F)


def Q_pam(q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """广义 PAM 驱动力 Q_PAM = -J_L^T F = E^T H F (N·m)                       (23)"""
    return E_matrix(q).T @ pam_torque_spatial(q, F)


# ---------------------------------------------------------------------------
# 4. 运动臂动力学参数 (待由 CAD 获取)
# ---------------------------------------------------------------------------
@dataclass
class ArmParams:
    """绿色运动总成的质量特性 (均在运动坐标系 {M} 下, SI 单位).

    m       : 总质量 kg
    r_G_M   : 质心相对球心的位置 ^M r_G = [xG, yG, zG] (m)
    I_O_M   : 关于球心的惯量矩阵 ^M I_O (kg·m^2), 对称 3×3
    g       : 重力加速度 (m/s^2), z_B 竖直向上, ^B g = [0, 0, -g]

    默认值为占位, 必须替换为 CAD 数据.
    """
    m: float = 1.0
    r_G_M: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 0.0]))
    I_O_M: np.ndarray = field(default_factory=lambda: np.diag([2e-3, 5e-3, 5e-3]))
    g: float = G_ACC

    def __post_init__(self):
        self.r_G_M = np.asarray(self.r_G_M, float).reshape(3)
        self.I_O_M = np.asarray(self.I_O_M, float).reshape(3, 3)
        if not np.allclose(self.I_O_M, self.I_O_M.T):
            raise ValueError("I_O_M 必须对称")


# ---------------------------------------------------------------------------
# 5. 重力、硅胶交互与刚体动力学
# ---------------------------------------------------------------------------
def com_position_B(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """质心相对球心在 {B} 中的位置 ^B r_G = R(q) ^M r_G (m)              (24)"""
    return R_M(q) @ p.r_G_M


def gravity_torque_spatial(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """重力矩 τ_g = ^B r_G × m ^B g (N·m)                                   (25)"""
    g_B = np.array([0.0, 0.0, -p.g])
    return np.cross(com_position_B(q, p), p.m * g_B)


def G_vector(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """广义重力项 G(q) = -E^T τ_g, 显式形式 (25a)."""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c3, s3 = np.cos(q3), np.sin(q3)
    xG, yG, zG = p.r_G_M
    return p.m * p.g * np.array([
        xG * c1 - yG * s1 * s3 - zG * s1 * c3,
        0.0,
        c1 * (yG * c3 - zG * s3),
    ])


def Q_silicone(q: np.ndarray, r_c: np.ndarray, f_s: np.ndarray) -> np.ndarray:
    """硅胶广义交互力矩 Q_s = E^T (r_c × f_s) (N·m)                        (26a)

    r_c: 球心指向等效接触点的位置向量 (m, {B} 表示)
    f_s: 等效接触力 (N, {B} 表示)
    """
    return E_matrix(q).T @ np.cross(np.asarray(r_c), np.asarray(f_s))


def inertia_B(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """^B I_O(q) = R ^M I_O R^T                                             (27a)"""
    R = R_M(q)
    return R @ p.I_O_M @ R.T


def M_matrix(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """广义质量矩阵 M(q) = E^T ^B I_O E                                       (28)"""
    E = E_matrix(q)
    return E.T @ inertia_B(q, p) @ E


def C_times_qd(q: np.ndarray, qd: np.ndarray, p: ArmParams) -> np.ndarray:
    """科氏/离心项 C(q, q̇) q̇ = E^T [ ^B I_O Ė q̇ + (E q̇) × (^B I_O E q̇) ]   (28)"""
    E = E_matrix(q)
    I_B = inertia_B(q, p)
    w = E @ qd
    return E.T @ (I_B @ (E_dot(q, qd) @ qd) + np.cross(w, I_B @ w))


def spatial_rigid_body_residual(q, qd, qdd, F, p: ArmParams,
                                tau_s=None, tau_f=None) -> np.ndarray:
    """空间刚体转动方程 (27) 的残差:
       ^B I_O ω̇ + ω × (^B I_O ω) - (τ_PAM + τ_g + τ_s - τ_f)   → 应为 0
    """
    tau_s = np.zeros(3) if tau_s is None else np.asarray(tau_s)
    tau_f = np.zeros(3) if tau_f is None else np.asarray(tau_f)
    I_B = inertia_B(q, p)
    w = omega(q, qd)
    lhs = I_B @ omega_dot(q, qd, qdd) + np.cross(w, I_B @ w)
    rhs = pam_torque_spatial(q, F) + gravity_torque_spatial(q, p) + tau_s - tau_f
    return lhs - rhs


def inverse_dynamics(q, qd, qdd, p: ArmParams, Q_s=None, Q_f=None) -> np.ndarray:
    """逆动力学: 由 (28) 求实现给定 q̈ 所需的广义 PAM 驱动力
       Q_PAM = -J_L^T F = M q̈ + C q̇ + G + Q_f - Q_s      (N·m, 3 维)
    注意: 由 Q_PAM 反解 8 维 F 需另加冗余分配 (F ≥ 0), 此处不做.
    """
    Q_s = np.zeros(3) if Q_s is None else np.asarray(Q_s)
    Q_f = np.zeros(3) if Q_f is None else np.asarray(Q_f)
    return M_matrix(q, p) @ qdd + C_times_qd(q, qd, p) + G_vector(q, p) + Q_f - Q_s


def forward_dynamics(q, qd, F, p: ArmParams, Q_s=None, Q_f=None) -> np.ndarray:
    """正动力学 q̈ = M^{-1} [ -J_L^T F + Q_s - C q̇ - G - Q_f ]               (30)"""
    Q_s = np.zeros(3) if Q_s is None else np.asarray(Q_s)
    Q_f = np.zeros(3) if Q_f is None else np.asarray(Q_f)
    rhs = -J_L(q).T @ np.asarray(F) + Q_s - C_times_qd(q, qd, p) - G_vector(q, p) - Q_f
    return np.linalg.solve(M_matrix(q, p), rhs)


# ---------------------------------------------------------------------------
# 6. 面向 MPC 的状态空间模型
# ---------------------------------------------------------------------------
def state_derivative(x: np.ndarray, u: np.ndarray, p: ArmParams,
                     Q_s=None, Q_f=None) -> np.ndarray:
    """连续时间非线性状态方程 ẋ = f(x, u, Q_s)                               (31)
    x = [q; q̇] ∈ R^6, u = F ∈ R^8 (PAM 目标拉力, N)
    """
    x = np.asarray(x, float)
    q, qd = x[:3], x[3:]
    return np.concatenate([qd, forward_dynamics(q, qd, u, p, Q_s, Q_f)])


def rk4_step(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
             x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """四阶 Runge–Kutta 离散化 (零阶保持输入)."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize(x0: np.ndarray, u0: np.ndarray, p: ArmParams,
              Q_s=None, Q_f=None, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """在工作点 (x0, u0) 对 (31) 数值线性化: δẋ ≈ A δx + B δu.
    返回 A ∈ R^{6×6}, B ∈ R^{6×8} (中心差分).
    """
    x0 = np.asarray(x0, float)
    u0 = np.asarray(u0, float)
    f = lambda x, u: state_derivative(x, u, p, Q_s, Q_f)
    A = np.zeros((6, 6))
    B = np.zeros((6, N_PAM))
    for j in range(6):
        d = np.zeros(6); d[j] = eps
        A[:, j] = (f(x0 + d, u0) - f(x0 - d, u0)) / (2 * eps)
    for j in range(N_PAM):
        d = np.zeros(N_PAM); d[j] = eps
        B[:, j] = (f(x0, u0 + d) - f(x0, u0 - d)) / (2 * eps)
    return A, B


def discretize_linear(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """零阶保持精确离散化 (用于线性/线性时变 MPC)."""
    from scipy.linalg import expm  # 局部导入, 避免硬依赖
    n, m = B.shape
    Mx = np.zeros((n + m, n + m))
    Mx[:n, :n] = A * dt
    Mx[:n, n:] = B * dt
    Phi = expm(Mx)
    return Phi[:n, :n], Phi[:n, n:]


# ---------------------------------------------------------------------------
# 7. 附录 A 校验用: S_1 的纯 q 展开式
# ---------------------------------------------------------------------------
def S1_expanded(q: np.ndarray) -> float:
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c3, s3 = np.cos(q3), np.sin(q3)
    return (6360 * c1 * c2 + 184 * c1 * c3 + 1120 * c1 * s2 + 460 * c1 * s3
            - 2544 * c2 * c3 * s1 + 1120 * c2 * c3
            - 6360 * c2 * s1 * s3 - 448 * c2 * s3 - 448 * c3 * s1 * s2
            - 6360 * c3 * s2 - 1120 * s1 * s2 * s3 + 460 * s1 + 2544 * s2 * s3)


# ---------------------------------------------------------------------------
# 8. 自检: 与 PDF 中的零位数值对照
# ---------------------------------------------------------------------------
def _self_test(verbose: bool = True) -> None:
    np.set_printoptions(precision=6, suppress=True, linewidth=140)
    q0 = np.zeros(3)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"[{'OK ' if cond else 'FAIL'}] {name}")

    # --- 旋转矩阵 (3)/(4) 一致性 ---
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.2, 1.2, 3)
        R_prod = rot_z(q[1]) @ rot_y(-q[0]) @ rot_x(q[2])
        check("R_M 显式式(4) == Rz(q2)Ry(-q1)Rx(q3)", np.allclose(R_M(q), R_prod))
        check("R_M 正交", np.allclose(R_M(q).T @ R_M(q), np.eye(3)))
    check("R_M(0) = I", np.allclose(R_M(q0), np.eye(3)))

    # --- E 矩阵与数值微分对照 (13a) ---
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, 3); qd = rng.normal(size=3); eps = 1e-6
        Rd = (R_M(q + eps * qd) - R_M(q - eps * qd)) / (2 * eps)
        W = Rd @ R_M(q).T                             # [ω]_x
        w_num = np.array([W[2, 1], W[0, 2], W[1, 0]])
        check("ω = E q̇ 与 Ṙ R^T 一致", np.allclose(w_num, omega(q, qd), atol=1e-6))
        Ed_num = (E_matrix(q + eps * qd) - E_matrix(q - eps * qd)) / (2 * eps)
        check("Ė 式(27c) 与数值微分一致", np.allclose(Ed_num, E_dot(q, qd), atol=1e-6))
    check("E(0)", np.allclose(E_matrix(q0), [[0, 0, 1], [-1, 0, 0], [0, 1, 0]]))

    # --- 零位几何 ---
    p0 = pam_points_mm(q0)
    check("p_i(0) 第1根 = (-111,20,8)", np.allclose(p0[0], [-111, 20, 8]))
    l0 = pam_vectors_mm(q0)
    check("l_i(0) 第1根 = (298,36,15)", np.allclose(l0[0], [298, 36, 15]))
    L0 = pam_lengths_mm(q0)
    check("L_i(0) = 300.541 mm (全部)", np.allclose(L0, 300.541, atol=5e-4))
    check("L_i = ||l_i||  (7) vs (8a)", np.allclose(L0, np.linalg.norm(l0, axis=1)))
    u0 = pam_unit_dirs(q0)
    check("u_1(0) = (0.991545, 0.119784, 0.049910)",
          np.allclose(u0[0], [0.991545, 0.119784, 0.049910], atol=1e-6))

    # --- 附录 A: S_1 展开式 ---
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, 3)
        check("附录 A S_1 展开式 == a_1^T R b_1", np.isclose(S1_expanded(q), pam_S(q)[0]))

    # --- H(0) (23a) 与 J_L(0) (15d) ---
    H0 = H_matrix(q0)
    H0_pdf_col1 = np.array([0.00003993, 0.006934, -0.017435])
    check("H(0) 第1列", np.allclose(H0[:, 0], H0_pdf_col1, atol=1e-6))
    JL0 = J_L(q0)
    JL0_pdf_row1 = np.array([0.006934, 0.017435, -0.00003993])
    check("J_L(0) 第1行", np.allclose(JL0[0], JL0_pdf_row1, atol=1e-6))
    rank, sv, cond = actuation_rank(q0)
    check("rank H(0) = 3", rank == 3)
    check("σ(H(0)) ≈ {0.037527, 0.037527, 0.00011293}",
          np.allclose(sv, [0.037527, 0.037527, 0.00011293], atol=2e-6))
    check("cond H(0) ≈ 332.3", abs(cond - 332.3) < 0.5)

    # --- J_L 与 L(q) 数值微分对照 ---
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, 3); eps = 1e-6
        J_num = np.zeros((N_PAM, 3))
        for j in range(3):
            d = np.zeros(3); d[j] = eps
            J_num[:, j] = (pam_lengths_m(q + d) - pam_lengths_m(q - d)) / (2 * eps)
        check("J_L = -H^T E 与 ∂L/∂q 数值一致", np.allclose(J_num, J_L(q), atol=1e-7))
        # h_i = r_i × u_i  (19) 与 (19a) 一致
        h_a = np.cross(moment_arm_vectors_mm(q), pam_unit_dirs(q)) * MM2M
        check("h_i = r_i × u_i == (r_i × a_i)/L_i", np.allclose(h_a, effective_arm_vectors_m(q)))
        # Q_PAM 两种写法
        F = rng.uniform(0, 100, N_PAM)
        check("Q_PAM = -J_L^T F == E^T H F", np.allclose(-J_L(q).T @ F, Q_pam(q, F)))

    # --- 动力学 ---
    p = ArmParams(m=1.3, r_G_M=[0.08, 0.01, -0.02],
                  I_O_M=[[3e-3, 2e-4, -1e-4], [2e-4, 6e-3, 3e-4], [-1e-4, 3e-4, 5e-3]])
    check("G(0) = m g [xG, 0, yG]", np.allclose(G_vector(q0, p), p.m * p.g * np.array([0.08, 0, 0.01])))
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, 3)
        check("G(q) 式(25a) == -E^T τ_g", np.allclose(G_vector(q, p), -E_matrix(q).T @ gravity_torque_spatial(q, p)))
    Ixx, Iyy, Izz = np.diag(p.I_O_M); Ixy, Ixz, Iyz = p.I_O_M[0, 1], p.I_O_M[0, 2], p.I_O_M[1, 2]
    M0_pdf = np.array([[Iyy, -Iyz, -Ixy], [-Iyz, Izz, Ixz], [-Ixy, Ixz, Ixx]])
    check("M(0) 结构 (28a)", np.allclose(M_matrix(q0, p), M0_pdf))
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, 3); qd = rng.normal(size=3); F = rng.uniform(0, 80, N_PAM)
        qdd = forward_dynamics(q, qd, F, p)
        check("正动力学(30) 满足空间刚体方程(27)",
              np.allclose(spatial_rigid_body_residual(q, qd, qdd, F, p), 0, atol=1e-9))
        check("逆动力学与正动力学互逆",
              np.allclose(inverse_dynamics(q, qd, qdd, p), Q_pam(q, F)))
        M = M_matrix(q, p)
        check("M 对称正定", np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0))

    if verbose:
        print("\nL(0) [mm]:", L0)
        print("\nu(0):\n", u0)
        print("\nH(0) [m]:\n", H0)
        print("\nJ_L(0) [m/rad]:\n", JL0)
        print(f"\nσ(H(0)) = {sv}, cond = {cond:.1f}")
        print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    if not ok:
        raise AssertionError("self test failed")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _self_test()

    # --- 示例: 零位附近线性化 + 短仿真 ---
    p = ArmParams()                       # 占位参数, 需替换为 CAD 值
    x0 = np.zeros(6)
    F0 = np.full(N_PAM, 20.0)
    A, B = linearize(x0, F0, p)
    print("\n线性化 A (6x6):\n", A)
    print("\n线性化 B (6x8):\n", B)

    f = lambda x, u: state_derivative(x, u, p)
    x, dt = x0.copy(), 1e-3
    for _ in range(200):
        x = rk4_step(f, x, F0, dt)
    print("\n0.2 s 后状态 x = [q; q̇]:", x)
    print("对应 PAM 长度 [mm]:", pam_lengths_mm(x[:3]))

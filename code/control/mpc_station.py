# -*- coding: utf-8 -*-
"""
mpc_station.py — PC 端 MPC 控制站：用【作者自建模型 shoulder_pam_model.py】追示教轨迹

原则（作者定版 2026-09-03）：预测模型 = 自建刚体+PAM 几何模型（M/C/G、J_L、正动力学
全部原封调用，不做任何黑箱替换）。fit 模式只辨识模型里【本来就没有的量】：
  τ_up/τ_dn ×6   通道充/放气一阶滞后（regulator+管路，推不出来只能测）
  gamma          压力→指令收缩量增益（mm/kPa，线性假设）
  kf             肌肉刚度（N/mm）：F_i = kf·max(0, Δcmd_i − (L0_i − L_i(q)))，≤FMAX
  c_damp         关节粘性阻尼（经模型的 Q_f 口进入，作者式里本来就留了这个钩子）
几何、惯量、重力、力臂、耦合——全部来自作者模型，fit 不碰。

被控通道：六路全部 = {1,5}{4,8}{2,3}{6,7}{9}{10}（模型经 use_ten_pam 扩到 10 根；
物理肌肉编号 → 模型行号经 PYROW 映射——两边行序不同，别直接用下标！）。

数据链路（跑之前先断开 GUI 的串口连接；固件 = teach_mode）：
  读 10Hz 帧  eul | quat | gyro | linacc | T | CAL | L1..L10
  写 W<mv0..5>（固件 2s 看门狗自动放气）

用法：
  python mpc_station.py ident --port COM5              # 逐路阶跃 → logs/ident_*.csv
  python mpc_station.py fit                            # 辨识 τ/gamma/kf/c → params.json
  python mpc_station.py mpc --traj teach.json          # MPC 追 GUI 导出的示教轨迹
  python mpc_station.py mpc --targets "10,0,3;0,8,3"   # 或内联路点 sx,sy,秒

安全：总摆角>45° 清零退出；Ctrl+C 清零退出；固件看门狗兜底。
依赖：numpy、pyserial；matplotlib 可选。
"""
import argparse, json, math, os, shutil, sys, tempfile, time, csv, threading
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # 上一级目录里是作者模型
from project_paths import CONTROL_LOG_DIR
import shoulder_pam_model as spm                   # ← 预测模型本体
from codex_compensator import normalize_reasoning_effort

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("需要 pyserial：pip install pyserial")

PARAMS_JSON = os.path.join(HERE, "params.json")
LOG_DIR = str(CONTROL_LOG_DIR)

DT = 0.05                      # MPC/真正模型使用20Hz（50ms一拍）
PI_DT = 0.05                   # PI / PI+FEL 同样使用20Hz
NCH = 6
ACTIVE_CH = [0, 1, 2, 3, 4, 5] # 模型已扩到 10 根（spm.use_ten_pam），六路全放开
ANTAGONIST_PAIRS = ((0, 1), (2, 3), (4, 5))      # {1,5}↔{4,8}, {2,3}↔{6,7}, 9↔10
HARD_CLAMP_MV = 3000.0         # 软件/固件最终安全上限；日常上限只由 gains.json:u_max 决定
CLAMP_MV = 2200.0              # 启动默认值；每次播放由 gains.json 覆盖
CO_CONTRACT_MV = 0.0           # 共收缩下限。作者 2026-09-04 试过 300mV 底气后撤回：六个
                               # regulator 同时工作噪音太大。回 0V；闲置通道电压归零。
                               # 固件侧：任何有效驱动前锁阀先动作100ms；压力≤30kPa后才开锁。
KPA_PER_MV = 0.09              # 【仅物理先验用】名义指令换算（vc2 口径，900 kPa 满量程）。
                               # 真机的指令→压力增益逐路不同（实测 0.047~0.122），由 DataModel.g
                               # 从日志学；控制回路用 g，不用这个常数。
P_MAX_KPA = 900.0              # 压力状态上界：AD7606 监测口满量程（1–5V ↔ 0–900 kPa）。
                               # 别再用 CLAMP_MV*KPA_PER_MV(=270) 当上界——g>0.09 的通道
                               # 满指令稳态压能到 366kPa，会被削掉、预测饱和
FMAX = 500.0                   # 单根拉力上限 N（与 GUI 一致）
PRESSURE_EMPTY_KPA = 30.0      # 与固件一致：仅开锁/保持判据使用，不截断遥测或模型压力
PRESSURE_HOLD_KP = 35.0        # 保压内环比例增益，mV/kPa（仅 PI/PI+FEL 的已完成轴）
PRESSURE_HOLD_KI = 18.0        # 保压内环积分增益，mV/(kPa*s)
PRESSURE_HOLD_DEADBAND_KPA = 1.5
PRESSURE_HOLD_I_MAX_MV = 500.0
PI_DU_MAX_MV_PER_S = 3000.0    # PI六路输出斜率限制；20Hz时每拍最多变化150mV
TILT_ABORT_DEG = 45.0

# 肌肉长度 PI 使用 20 路拉线编码器：主动端 ID1..10、被动端 ID11..20。
# 物理阀顺序必须与 W 指令一致，不能直接沿用 GUI 的显示顺序：
#   V0=A1+A5, V1=A4+A8, V2=A2+A3, V3=A6+A7, V4=A9, V5=A10。
MUSCLE_SENSOR_L0_MM = 235.0
MUSCLE_LENGTH_GROUPS = ((0, 4), (3, 7), (1, 2), (5, 6), (8,), (9,))
MUSCLE_GUI_TO_PHYSICAL = np.array([0, 2, 1, 3, 4, 5], dtype=int)
MUSCLE_PI_KP = 120.0           # mV/mm
MUSCLE_PI_KI = 20.0            # mV/(mm*s)
MUSCLE_PI_DEADBAND_MM = 0.5
MUSCLE_PI_I_MAX_MV = 800.0
# 六路长度反馈方向独立列出；+1 表示严格按“实测比目标长才收缩”的规则。
# 后续若确认某只编码器物理方向接反，只改对应元素即可。
MUSCLE_PI_SENSOR_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def _muscle_lengths_from_lens(raw):
    """20 路位移 → 六个物理阀组的真实平均长度(mm)；缺任一传感器即返回 NaN。"""
    if not isinstance(raw, list) or len(raw) < 20:
        return np.full(NCH, np.nan)
    individual = np.full(10, np.nan)
    for i in range(10):
        try:
            active, passive = float(raw[i]), float(raw[i+10])
            if np.isfinite(active) and np.isfinite(passive):
                # 编码器安装方向可能使角度增量为负；这里只使用位移幅值：
                # 当前长度 = 原长 - 主动收缩量 + 被动拉伸量。
                individual[i] = MUSCLE_SENSOR_L0_MM - abs(active) + abs(passive)
        except (TypeError, ValueError):
            pass
    grouped = np.full(NCH, np.nan)
    for ch, ids in enumerate(MUSCLE_LENGTH_GROUPS):
        vals = individual[list(ids)]
        if np.all(np.isfinite(vals)):
            grouped[ch] = float(np.mean(vals))
    return grouped


def _group_signed_displacements_from_lens(raw):
    """Return signed active/passive means for all six physical valve groups."""
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) < 20:
        return np.full((NCH, 2), np.nan)
    grouped = np.full((NCH, 2), np.nan)
    for channel, indexes in enumerate(MUSCLE_LENGTH_GROUPS):
        try:
            active = np.asarray([raw[index] for index in indexes], dtype=float)
            passive = np.asarray([raw[index+10] for index in indexes], dtype=float)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(active)) and np.all(np.isfinite(passive)):
            grouped[channel] = [float(np.mean(active)), float(np.mean(passive))]
    return grouped


def _l15_signed_displacements_from_lens(raw):
    """Backward-compatible A1+A5 view of the six-group displacement array."""
    return _group_signed_displacements_from_lens(raw)[0]

def _antagonist_project(u):
    """拮抗肌硬投影：每对只保留较大一侧，另一侧精确置 0。"""
    a = np.asarray(u, float).copy()
    one = a.ndim == 1
    if one: a = a[None, :]
    for left, right in ANTAGONIST_PAIRS:
        both = (a[:, left] > 0.0) & (a[:, right] > 0.0)
        keep_left = a[:, left] >= a[:, right]
        a[both & keep_left, right] = 0.0
        a[both & ~keep_left, left] = 0.0
    return a[0] if one else a

spm.use_ten_pam()              # v5.1：加入第 9/10 根（滑环固定，b 为常量）
spm.set_pivot_dx(66.0)         # 实际转动中心 Δx=66mm（作者确认；不加则 V0..V3 低估数倍）

# 物理通道 → 模型行下标。⚠ py 模型的行序 ≠ 物理肌肉编号！逐行比对 A/B 坐标
# 得到：物理肌肉 k（1..10）= 模型行 PYROW[k-1]（前 8 根两套表只是排序不同）。
PYROW = [2, 3, 7, 6, 1, 0, 4, 5, 8, 9]
_ch_phys = [[1, 5], [4, 8], [2, 3], [6, 7], [9], [10]]   # V0..V5 的物理肌肉对
CH_MUSCLES = [[PYROW[m-1] for m in ms] for ms in _ch_phys]
MUSCLE_CH = {m: c for c, ms in enumerate(CH_MUSCLES) for m in ms}

# 质量特性：GUI 定版（SolidWorks 运动件整体值，M 系）——py 模型的默认是占位符
_M_KG = 0.2219
_RG_M = np.array([0.017272, -0.0000187, -0.090313])          # 质心，臂垂向 −z_M
_IG_M = np.array([[0.000943919, -4.8e-8,     5.743e-6],      # 关于质心
                  [-4.8e-8,      0.000928455, -8.5e-8],
                  [5.743e-6,    -8.5e-8,      0.000037198]])
# 平行轴：I_O = I_G + m(|r|²E − r rᵀ)（关于球心，M 系）
_IO_M = _IG_M + _M_KG*(np.dot(_RG_M, _RG_M)*np.eye(3) - np.outer(_RG_M, _RG_M))
ARM = spm.ArmParams(m=_M_KG, r_G_M=_RG_M, I_O_M=_IO_M)
Q0 = np.zeros(3)
L0_MM = spm.pam_lengths_mm(Q0)                     # 零位肌肉长度，力法则的基准

# ── 默认未知量（fit 覆盖）────────────────────────────────────────────────────
# 名义值来自 GUI 定版：km=3000 N/m → kf=3 N/mm；γ ≈ ε_max·L0/(Pmax·1000)
#   = 0.30×300.54/600 ≈ 0.15 mm/kPa（9/10 另有 L0/ε，先共用一个 γ，fit 折中）
DEF_P = {"tau_up": [0.4]*NCH, "tau_dn": [1.0]*NCH,
         "gamma": 0.15, "gamma9": 0.3, "kf": 3.0, "c_damp": 0.08}
# gamma9：肌肉 9/10 的压力→收缩增益单列（L0=238mm、ε_max 与 1..8 不同，GUI 亦分开）

# ───────────────────────── 姿态换算 ─────────────────────────────────────────
def rotm_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def q_conj_mul(a, b):                              # conj(a) ⊗ b
    aw, ax, ay, az = a[0], -a[1], -a[2], -a[3]
    return np.array([
        aw*b[0]-ax*b[1]-ay*b[2]-az*b[3],
        aw*b[1]+ax*b[0]+ay*b[3]-az*b[2],
        aw*b[2]-ax*b[3]+ay*b[0]+az*b[1],
        aw*b[3]+ax*b[2]-ay*b[1]+az*b[0]])

MOUNTS = {"ry-": spm.rot_y(-math.pi/2), "ry+": spm.rot_y(math.pi/2), "none": np.eye(3)}

def q_model_from_quat(quat, qref, mount):
    """IMU 四元数 → 作者模型的 q=(q1,q2,q3)。
    相对零位旋转经安装矩阵共轭映射到 B 系，再按 R = Rz(q2)Ry(−q1)Rx(q3) 反解。"""
    M = MOUNTS[mount]
    R = M @ rotm_from_quat(q_conj_mul(qref, quat)) @ M.T
    q1 = math.asin(max(-1.0, min(1.0, R[2, 0])))          # R[2,0] = sin(q1)（b=−q1）
    q2 = math.atan2(R[1, 0], R[0, 0])
    q3 = math.atan2(R[2, 1], R[2, 2])
    return np.array([q1, q2, q3])

def q_model_from_st(sx_deg, sy_deg, psi_deg=0.0, mount="ry-"):
    """GUI 示教导出的 (s_x, s_y, ψ)° → 模型 q（经旋转矩阵走一遍，不近似）。

    ``mount`` 只用于把 IMU 传感器坐标转到机构坐标，不能再次作用于已经
    位于机构坐标中的 GUI 参考；否则会把 x 与 φ 的参考轴互换。
    参数保留仅为兼容已有调用。
    """
    sx, sy, ps = math.radians(sx_deg), math.radians(sy_deg), math.radians(psi_deg)
    th = math.hypot(sx, sy)
    if th < 1e-12:
        Rsw = np.eye(3)
    else:
        k = np.array([sx/th, sy/th, 0.0])
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rsw = np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K @ K)
    R = Rsw @ spm.rot_z(ps)
    q1 = math.asin(max(-1.0, min(1.0, R[2, 0])))
    q2 = math.atan2(R[1, 0], R[0, 0])
    q3 = math.atan2(R[2, 1], R[2, 2])
    return np.array([q1, q2, q3])

def tilt_deg(q):
    """总摆角 = 臂轴（−z_M，质心方向）偏离铅垂的角度。"""
    R = spm.R_M(q)
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

# ───────────────────────── 作者模型 + 未知量 = 组合动力学 ────────────────────
class Plant:
    """x = [q(3), qd(3), P(6 kPa)]，u = 六路 mV。刚体部分全部调作者模型。"""
    KSIL = np.diag([2.0, 2.0, 0.2])   # 硅胶柱回复刚度 N·m/rad（GUI 作者定值 k=[2,2,0.2]）。
                                      # 漏传它时模型在共收缩下会被扭到 q1=90° 参数化奇点炸掉
                                      # （2026-09-03 发现：任何步长都炸=动力学无界，不是数值问题）

    def __init__(self, p):
        self.tu = np.array(p["tau_up"]); self.td = np.array(p["tau_dn"])
        self.gamma = p["gamma"]; self.gamma9 = p.get("gamma9", p["gamma"])
        self.kf = p["kf"]; self.c = p["c_damp"]
        # 静息预紧力矩（2026-09-04）：IMU 的零位是真机【静止悬垂】的姿态，模型的零位
        # 也必须是静止平衡——否则 q=0、P=0 时模型自带 40 rad/s² 的加速度（重力+硅胶
        # 不平衡），控制器一上电就在跟幻影漂移打架，offset-free 把幻影积成 −10° 目标，
        # 电压冲到 3000（01:32 日志）。物理含义 = 硅胶柱的中性姿态就是静息姿态：
        # Q_s = −K q + τ0，τ0 抵消 q=0、P=0 时的重力与静态预紧力。
        q0 = np.zeros(3)
        self.tau0 = np.zeros(3)
        F0 = self.forces(q0, np.zeros(NCH))
        self.tau0 = spm.J_L(q0).T @ F0 + spm.G_vector(q0, ARM)

    SMOOTH_MM = 2.0   # softplus 平滑宽度（mm）：把 max(0,·) 的硬拐点磨圆。
                      # 拐点是数值微分的噩梦——松弛区求出的雅可比比例全错，实机上
                      # 表现为"开 2800mV 修 3° 扭转、把 q2 捎带到 50°"的疯狂分配
                      # （2026-09-03 22:18 日志）。物理上腱也是渐进绷紧的，平滑更真。
    # 压力从 0 kPa 起连续参与出力，不设人为压力门槛（作者定 2026-09-04）。

    def forces(self, q, P_kpa):
        """压力→拉力：F_i = kf·softplus(γ·P_ch − (L0_i − L_i(q)))，C¹ 光滑、单边、限幅。"""
        L = spm.pam_lengths_mm(q)
        F = np.zeros(spm.N_PAM)
        s = Plant.SMOOTH_MM
        for i in range(spm.N_PAM):
            g = self.gamma9 if i >= 8 else self.gamma
            p_eff = max(0.0, float(P_kpa[MUSCLE_CH[i]]))
            e = g * p_eff - (L0_MM[i] - L[i])              # 指令收缩−几何收缩=绷紧量
            z = e / s
            # softplus，含线性渐近（复检修正 2026-09-04）：
            # 原式 min(z,30) 把 z>30（绷紧>60mm，u>~2200mV）的力冻结在 180N、
            # dF/dP 精确为 0——高压通道对 QP 完全隐形，成为"开2800修3°、q2甩到50°"
            # 灾难拍的直接机理。softplus 大 z 渐近就是 z 本身，取 sp=e 数学精确。
            sp = e if z > 30.0 else s*math.log1p(math.exp(z))
            F[i] = min(FMAX, self.kf * sp)
        return F

    def f(self, x, u_mv):
        q, qd, P = x[:3], x[3:6], x[6:]
        Pc = np.clip(u_mv, 0, CLAMP_MV) * KPA_PER_MV
        tau = np.where(Pc >= P, self.tu, self.td)
        dP = (Pc - P) / np.maximum(tau, 0.05)
        F = self.forces(q, P)
        qdd = spm.forward_dynamics(q, qd, F, ARM,
                                   Q_s=self.tau0 - Plant.KSIL @ q,   # 硅胶回复力矩 + 静息预紧（作者 Q_s 口）
                                   Q_f=self.c * qd)                  # 阻尼走作者的 Q_f 口
        return np.concatenate([qd, qdd, dP])

    def step(self, x, u, dt, n=None):
        """半隐式（辛）欧拉：先更新速度、再用新速度推进位置。绳索弹簧把固有
        频率抬到几百 rad/s，前向欧拉对振荡模态无条件不稳定，辛欧拉在 ωh<2
        内稳定。子步 1ms：真实惯量里绕臂轴 I≈1e-4，ω 和显式阻尼项都得压住。"""
        if n is None: n = max(4, int(math.ceil(dt/0.0005)))   # 0.5ms：显式阻尼项 c·h/I 必须 <1
        h = dt / n
        x = x.copy()
        for _ in range(n):
            dx = self.f(x, u)              # [qd, qdd, dP]
            x[3:6] += h * dx[3:6]
            x[0:3] += h * x[3:6]
            x[6:]  += h * dx[6:]
        return x

    def linearize_affine(self, x0, u0, eps_x=1e-5, eps_u=1.0):
        """LTV 用：当地离散化 (Ad, Bd, wd)，wd = ∫e^{As}ds·f(x0,u0)（零阶保持精确）。
        仿射项不做任何非线性积分——积分器在高气压/高刚度下的脆弱性从预测路径根除。"""
        n, m = 12, NCH
        A = np.zeros((n, n)); B = np.zeros((n, m))
        for j in range(n):
            d = np.zeros(n); d[j] = eps_x
            A[:, j] = (self.f(x0+d, u0) - self.f(x0-d, u0)) / (2*eps_x)
        for j in range(m):
            d = np.zeros(m); d[j] = eps_u
            B[:, j] = (self.f(x0, u0+d) - self.f(x0, u0-d)) / (2*eps_u)
        f0 = self.f(x0, u0)
        Ad, Bd = spm.discretize_linear(A, B, DT)
        _, wd = spm.discretize_linear(A, f0.reshape(-1, 1), DT)
        return Ad, Bd, wd.ravel()

    def linearize(self, x0, u0, eps_x=1e-5, eps_u=1.0):
        """数值线性化组合动力学（作者 spm.linearize 的思路，扩到 12 维 + mV 输入）。
        离散化用作者的 discretize_linear（expm 精确 ZOH）——绳索刚度的振荡模态
        几百 rad/s，任何欧拉族的近似离散都会把预测矩阵搞爆。"""
        n, m = 12, NCH
        A = np.zeros((n, n)); B = np.zeros((n, m))
        for j in range(n):
            d = np.zeros(n); d[j] = eps_x
            A[:, j] = (self.f(x0+d, u0) - self.f(x0-d, u0)) / (2*eps_x)
        for j in range(m):
            d = np.zeros(m); d[j] = eps_u
            B[:, j] = (self.f(x0, u0+d) - self.f(x0, u0-d)) / (2*eps_u)
        return spm.discretize_linear(A, B, DT)     # 作者的零阶保持精确离散（expm）

# ───────────────────────── 串口链路 ─────────────────────────────────────────
class Link:
    def __init__(self, port, baud=115200):
        if not port:
            cands = [p.device for p in list_ports.comports()]
            if not cands: sys.exit("找不到串口。--port COMx 指定")
            port = cands[0]
            print(f"# 未指定 --port，用 {port}（候选：{cands}）")
        self.ser = serial.Serial(port, baud, timeout=0.02)
        self.buf = b""; self.quat = None; self.lens = [None]*10

    def poll(self):
        try: data = self.ser.read(4096)
        except serial.SerialException: return
        if data: self.buf += data
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            s = line.decode("ascii", "ignore").strip()
            if not s or s[0] in "[#":
                if s.startswith("[tch]") and ("ABORT" in s or "timeout" in s): print("  fw:", s)
                continue
            parts = [p.strip() for p in s.split("|")]
            if len(parts) < 2: continue
            try: qq = [float(x) for x in parts[1].split()]
            except ValueError: continue
            if len(qq) < 4: continue
            v = np.array(qq[:4]); n = np.linalg.norm(v)
            if n > 1e-6: self.quat = v/n
            if len(parts) >= 7:
                for i, t in enumerate(parts[6].split()[:10]):
                    try: self.lens[i] = float(t)
                    except ValueError: self.lens[i] = None

    def send_mv(self, u):
        self.ser.write(("W" + ",".join(str(int(round(max(0, min(CLAMP_MV, x))))) for x in u) + "\n").encode())

    def all_zero(self):
        for _ in range(3): self.send_mv([0]*NCH); time.sleep(0.05)

    def capture_ref(self, seconds=1.0):
        print("# 对零：保持手臂静止…")
        t0 = time.time()
        while time.time()-t0 < seconds: self.poll(); time.sleep(0.01)
        if self.quat is None: sys.exit("没收到 IMU 帧——固件是 teach_mode 吗？GUI 断开了吗？")
        return self.quat.copy()

# ───────────────────────── ident ────────────────────────────────────────────
def _stillness_check(link, qref, mount, seconds=3.0):
    """对零后静置观察：臂不动时 q 应≈0。若在涨，是安装矩阵/偏航漂移，不是运动。"""
    print(f"# 静置自检 {seconds:.0f}s（臂别动）…")
    t0 = time.time(); worst = np.zeros(3)
    while time.time()-t0 < seconds:
        link.poll(); time.sleep(0.02)
        if link.quat is None: continue
        q = np.abs(q_model_from_quat(link.quat, qref, mount))
        worst = np.maximum(worst, q)
    wd = np.degrees(worst)
    print(f"#   静置漂移峰值 q=({wd[0]:.2f}, {wd[1]:.2f}, {wd[2]:.2f})°")
    if wd[0] > 3 or wd[1] > 3:
        print("#   ⚠ 臂没动但摆角在涨——多半是 --mount 方向不对（换 ry+ 试）或磁力计"
              "没校准导致偏航漂移被映成摆角。别在这种状态下跑辨识。")
    elif wd[2] > 3:
        print("#   （扭转 q3 在漂但摆角干净——偏航漂移已正确落在扭转轴上，无害，可以跑）")

def mode_ident(args):
    link = Link(args.port); qref = link.capture_ref()
    _stillness_check(link, qref, args.mount)
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, time.strftime("ident_%Y%m%d_%H%M%S.csv"))
    lv = [int(x) for x in args.levels.split(",")]
    print(f"# 阶跃辨识：V{ACTIVE_CH} 各 {lv} mV × {args.hold}s → {path}")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "ch", "u_cmd", "q1", "q2", "q3"] + [f"L{i+1}" for i in range(10)])
        t0 = time.time()
        def run(ch, mv, dur):
            u = [0]*NCH
            if ch >= 0: u[ch] = mv
            tend = time.time() + dur
            while time.time() < tend:
                link.send_mv(u)
                tn = time.time() + DT
                while time.time() < tn: link.poll(); time.sleep(0.005)
                if link.quat is None: continue
                # 标定不做角度保护（作者定：气源侧已设最大气压，安全由硬件兜底）。
                # 角度只记录，供 fit 拟合用。
                q = q_model_from_quat(link.quat, qref, args.mount)
                wr.writerow([round(time.time()-t0, 3), ch, mv] + [round(math.degrees(v), 3) for v in q]
                            + [x if x is not None else "" for x in link.lens])
        try:
            for ch in ACTIVE_CH:
                print(f"#  V{ch} …")
                run(-1, 0, 2.0)
                for mv in lv: run(ch, mv, args.hold)
                run(ch, 0, args.hold)
        finally:
            link.all_zero()
    print(f"# 完成 → {path}\n# 下一步：python mpc_station.py fit")

# ───────────────────────── fit：只辨识模型没有的量 ──────────────────────────
def _segments(dat, ch):
    segs, cur = [], None
    for r in dat[dat[:, 1] == ch]:
        if cur is None or r[2] != cur[0]:
            cur = (r[2], [], []); segs.append(cur)
        cur[1].append(r[0]); cur[2].append(np.radians(r[3:6]))
    return segs                                    # (u_mv, t[], q_rad[])

def _t63(t, y):
    y0, y1 = y[0], np.mean(y[-max(1, len(y)//3):])
    th = y0 + 0.632*(y1-y0)
    for i in range(1, len(y)):
        if (y[i]-th)*(y0-th) < 0: return t[i]-t[0]
    return (t[-1]-t[0])*0.5

def _sim_settle(plant, u_mv, seconds=None):
    """作者模型在恒定指令下的平衡姿态：静力学松弛（势能梯度流）。
    不跑动力学——绕臂轴惯量 ~1e-4 kg·m²，显式积分怎么调步长都在悬崖边；
    平衡本来只由 −J_Lᵀ F − G = 0 决定，与惯量/阻尼无关。"""
    P = np.clip(np.asarray(u_mv, float), 0, CLAMP_MV) * KPA_PER_MV
    q = np.zeros(3); lr = 4e-3
    for _ in range(6000):
        F = plant.forces(q, P)
        tau = -spm.J_L(q).T @ F - spm.G_vector(q, ARM) - Plant.KSIL @ q + plant.tau0   # 含硅胶回复+静息预紧
        qn = q + lr * tau
        if not np.all(np.isfinite(qn)): break
        if np.linalg.norm(qn - q) < 1e-8: q = qn; break
        q = qn
    return q

def mode_fit(args):
    files = sorted(f for f in os.listdir(LOG_DIR) if f.startswith("ident_")) if os.path.isdir(LOG_DIR) else []
    if not files: sys.exit("没有 ident 日志，先跑 ident")
    path = os.path.join(LOG_DIR, files[-1])
    print(f"# 灰盒辨识（刚体部分不动，只填未知量）：{path}")
    rows = list(csv.reader(open(path)))[1:]
    dat = np.array([[float(v) for v in r[:6]] for r in rows])
    p = dict(DEF_P)

    # ① τ：63% 时间扣除臂的机械响应（阶跃里量出的都是串联结果，拆分靠假设 T≈0.3s）
    for ch in ACTIVE_CH:
        for i, (u, t, qs) in enumerate(_segments(dat, ch)):
            if i == 0 or len(t) < 4: continue
            mag = [np.linalg.norm(x) for x in qs]
            t63 = _t63(t, mag)
            prev_u = _segments(dat, ch)[i-1][0]
            if u > prev_u:   p["tau_up"][ch] = max(0.05, t63-0.3)
            elif u < prev_u: p["tau_dn"][ch] = max(0.05, t63-0.3)

    # ② gamma, kf：网格搜稳态——作者模型算平衡姿态 vs 实测稳态，最小平方误差。
    # 实测取【相对本通道起始姿态的增量】：放气后臂常回不到零（迟滞/硅胶记忆），
    # 各路基线互相继承漂移，绝对角度会把 γ/kf 拟歪；增量才是本路拉力的贡献。
    meas = []                                      # (ch, u_mv, Δq_ss_meas)
    for ch in ACTIVE_CH:
        segs = _segments(dat, ch)
        if not segs or len(segs[0][2]) < 1: continue
        # 基线 = 本通道首段【之前】的静置段（ch=-1 行）。复检发现旧写法取的是
        # 已通电段的头 3 拍（阶跃后 0.1~0.3s，污染达稳态幅值 ~30%），系统性低估 γ/kf。
        t0seg = segs[0][1][0]
        idle = dat[(dat[:, 1] == -1) & (dat[:, 0] >= t0seg - 1.8) & (dat[:, 0] < t0seg)]
        base = (np.radians(idle[:, 3:6]).mean(axis=0) if len(idle)
                else np.mean(segs[0][2][:3], axis=0))
        for i, (u, t, qs) in enumerate(segs):
            if u <= 0 or len(qs) < 4: continue
            k = max(1, len(qs)//3)
            meas.append((ch, u, np.mean(qs[-k:], axis=0) - base))
    if not meas: sys.exit("日志里没有有效稳态段")
    def _uvec(ch, mv):
        u = np.zeros(NCH); u[ch] = mv; return u
    m03 = [m for m in meas if m[0] <= 3]           # γ/kf 用环绕 8 根的段
    m45 = [m for m in meas if m[0] >= 4]           # γ9 用 9/10 的段
    best = (1e18, p["gamma"], p["kf"])
    for gamma in [0.15, 0.22, 0.3, 0.4, 0.55, 0.75]:   # 有死区后有效压力变小，γ 上界放开（旧网格顶边 0.3）
        for kf in [1.5, 3.0, 6.0, 12.0, 24.0]:
            plant = Plant({**p, "gamma": gamma, "kf": kf})
            err = sum(float(np.sum((_sim_settle(plant, _uvec(c, u)) - qm)**2)) for c, u, qm in m03)
            if err < best[0]: best = (err, gamma, kf)
            print(f"#   γ={gamma:<5} kf={kf:<4} err={math.degrees(math.sqrt(err/max(1,len(m03)))):6.2f}°")
    p["gamma"], p["kf"] = best[1], best[2]
    print(f"# → gamma={p['gamma']} mm/kPa, kf={p['kf']} N/mm（V0..V3 稳态 RMS "
          f"{math.degrees(math.sqrt(best[0]/max(1,len(m03)))):.2f}°）")
    if m45:
        bg = (1e18, p["gamma9"])
        for g9 in [0.15, 0.22, 0.3, 0.4, 0.55, 0.75]:
            plant = Plant({**p, "gamma9": g9})
            err = sum(float(np.sum((_sim_settle(plant, _uvec(c, u)) - qm)**2)) for c, u, qm in m45)
            if err < bg[0]: bg = (err, g9)
        p["gamma9"] = bg[1]
        print(f"# → gamma9={p['gamma9']} mm/kPa（V4/V5 稳态 RMS "
              f"{math.degrees(math.sqrt(bg[0]/len(m45))):.2f}°）")

    # ③ c_damp：用最大上行阶跃的 63% 时间一维搜（τ、γ、kf 已定）
    ch, u_mv, _ = max(meas, key=lambda m: m[1])
    seg = [s for s in _segments(dat, ch) if s[0] == u_mv][0]
    t63_meas = _t63(seg[1], [np.linalg.norm(x) for x in seg[2]])
    best_c = (1e18, p["c_damp"])
    for c in [0.02, 0.05, 0.1, 0.2, 0.4]:
        plant = Plant({**p, "c_damp": c})
        x = np.zeros(12); u = np.zeros(NCH); u[ch] = u_mv
        tt, mag = [0.0], [0.0]
        for k in range(int(4.0/0.01)):
            x = plant.step(x, u, 0.01, n=50)       # h=0.2ms：小惯量轴要压得更狠
            tt.append((k+1)*0.01); mag.append(float(np.linalg.norm(x[:3])))
        e = abs(_t63(tt, mag) - t63_meas)
        if e < best_c[0]: best_c = (e, c)
    p["c_damp"] = best_c[1]
    print(f"# → c_damp={p['c_damp']}（t63 偏差 {best_c[0]:.2f}s）")

    p["src"] = path
    json.dump(p, open(PARAMS_JSON, "w"), indent=1)
    print(f"# → {PARAMS_JSON}")

# ───────────────────────── MPC（模型 = Plant = 作者模型 + 已辨识未知量）──────
class MPC:
    # w_q 单位说明：q 是弧度、u 是 mV，两边数量级差 ~5 个量级，权重必须把
    # rad² 折算到 deg² 一个级别（×57.3²≈3283），否则 ρ‖u‖² 会把跟踪项压死。
    # N=16（1.6s）：实测充/放气 τ 达 0.8~2.5s，时域必须盖住主导动态。
    # ── LTV 版（作者拍板恢复 2026-09-03）────────────────────────────────
    # 每拍在【当前状态】重新线性化。冻结模型在 ±30° 处的"弹簧幻觉"是实机
    # 钉死事故的根子（21:27 日志：V1/V4/V5 恒 3000、q1 卡 +28°）。当时否掉
    # LTV 的那些数值事故（NaN 污染、离散化爆炸）已全部修复并保留防护。
    def __init__(self, plant, N=16, w_q=(3300.0, 3300.0, 330.0),
                 w_v=(200.0, 200.0, 40.0), rd=2e-4, rho=1e-5):
        # rho（电压绝对值惩罚）2e-4→1e-5（2026-09-04）：作者实测臂不动——
        #   ρ 太大让 MPC "省电压"，9.6° 误差只挤出 345mV 推不动。ρ 小=舍得给力。
        # rd（Δu 平滑）1e-3→2e-4 同步松，别再把出力憋回去。
        self.pl = plant; self.N = N
        self.Wy = np.diag(list(w_q) + list(w_v))   # 输出=[q;qd]，qd 目标恒 0（电子阻尼）
        self.rd, self.rho = rd, rho
        self.ki = 0.0                              # 参考偏置积分增益（每拍），见 step 里的 d。
                                                   # 默认关：失配对象仿真里积分把充气滞后当稳态偏差积进去，
                                                   # 0.8× 拉力时峰值误差 6°→11°。稳态偏差交给 FEL 逐轮学；
                                                   # 若纯 MPC 多轮后仍有恒定偏差，可试 0.02
        self.d = np.zeros(3); self.q_pred = None; self.u_prev = np.zeros(NCH)
        m = NCH
        self._Qb = np.kron(np.eye(N), self.Wy)
        Dm = np.eye(m*N)
        for k in range(1, N): Dm[m*k:m*k+m, m*(k-1):m*k] = -np.eye(m)
        self._D = Dm
        self._setup_ref_model()
        self.d = np.zeros(3)                       # 输出偏差估计（offset-free）
        self.q_pred = None
        self.u_prev = np.zeros(NCH)

    def _setup_ref_model(self):
        """固定参考线性化（400mV 共收缩点）。控制不用它——只给 PI 分支当
        正模型的一步预测器（bridge 里 m_.x0/Ad/Bd/u0 引用的就是这套）。"""
        self.u0 = np.full(NCH, 400.0)
        q0 = _sim_settle(self.pl, self.u0)
        self.x0 = np.concatenate([q0, np.zeros(3), self.u0 * KPA_PER_MV])
        self.Ad, self.Bd = self.pl.linearize(self.x0, self.u0)

    def step(self, x_est, r_hor, u_ff_hor=None):
        """一拍。r_hor：未来 N 拍参考；u_ff_hor（可选，N×6 mV）：逆模型给的前馈序列——
        给了就当【已知名义输入】进预测，QP 只解在它之上的反馈修正 v（ρ 罚的是 v，
        前馈学好了反馈自然缩到零——FEL 的本意），总量 u = u_ff + v 仍守 [CO, 3V]。"""
        # 状态消毒：先清 NaN 再 clip（np.clip(NaN)=NaN，光 clip 会让污染驻留）
        x_est = np.nan_to_num(np.asarray(x_est, float), nan=0.0, posinf=0.0, neginf=0.0)
        x_est[3:6] = np.clip(x_est[3:6], -3.0, 3.0)
        x_est[6:]  = np.clip(x_est[6:], 0.0, CLAMP_MV*KPA_PER_MV)
        # 参考偏置积分（2026-09-04 换掉"一步预测残差"式 offset-free）：
        #   旧式 d ← q − q̂ 量的是模型【一拍内没兑现的运动】，不是输出偏移——模型比真机
        #   猛（惯量小、无摩擦）时 q̂ 每拍领先几度，d 被积成 −10°，目标被抬到 +10°，
        #   控制器把 V1/V2/V4 顶到 3000 而臂根本没动（01:32 日志复现）。
        #   现在 d 就是跟踪误差的积分：目标 = r − d = r + ki·Σe，臂到不了就慢慢抬目标。
        self.d -= self.ki * (np.asarray(r_hor[0], float) - x_est[:3])
        self.d = np.clip(self.d, -0.35, 0.35)      # |d| ≤ 20°：抗饱和
        N, m = self.N, NCH

        def _fallback():
            u_ = np.clip(self.u_prev*0.5, CO_CONTRACT_MV, CLAMP_MV)
            self.q_pred = None; self.u_prev = u_
            self.ok = False                        # 本拍异常：学习层据此丢弃样本
            return np.clip(u_, 0, CLAMP_MV), x_est

        # ── LTV 心脏：当地离散化 (Ad,Bd,wd)，仿射项 expm 精确、零积分 ──
        try:
            A, B, w = self.pl.linearize_affine(x_est, self.u_prev)
        except Exception:
            print("  !! LTV 线性化异常，安全退让"); return _fallback()
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(w))):
            print("  !! LTV 线性化非有限，安全退让"); return _fallback()
        # ── 预测卫生（2026-09-03 实机教训，两版演化）────────────────────────
        # v1 教训：局部不稳定状态的 e^{A·0.1} 爆到 1e18，投毒 QP/正模型/offset-free。
        # v2 教训：对 A、B 做【整体范数缩放】矫枉过正——松弛拐点上 B 一根尖峰
        #   会把整个 B 连健康分量一起缩没，控制器瘫痪（实机：臂冻在 −1.3° 任凭
        #   参考扫到 −15°，电压只出几十 mV，"收敛"的是全零行为）。
        # 定版：【逐元素限幅】——只削尖峰，不碰健康项。
        A = np.clip(A, -60.0, 60.0)
        B = np.clip(B, -0.1, 0.1)
        w = np.clip(w, -_W_CLIP, _W_CLIP)
        # ── 解耦校正（2026-09-04）────────────────────────────────────────
        # ① 静态：B 的姿态响应行块左乘实测校正阵 S（模型方向 → 真机方向）。
        #    只旋 B、不旋 w——w[:3]≈qd·DT 是纯运动学积分（无模型不确定性），
        #    旋它会把 qd 的跨轴分量灌进 q_pred，毒化 offset-free d（复检发现 2026-09-04）
        B[:3, :] = _S_CORR @ B[:3, :]
        B[3:6, :] = _S_CORR @ B[3:6, :]
        # 预测（δu 相对 u_prev，δx0=0）：δx⁺ = A δx + B δu + w
        C = np.zeros((6, 12)); C[:3, :3] = np.eye(3); C[3:, 3:6] = np.eye(3)
        Su = np.zeros((6*N, m*N)); Sw = np.zeros(6*N)
        pows = [np.eye(12)]
        for _ in range(N):
            P_ = A @ pows[-1]
            nP = np.linalg.norm(P_)
            if nP > 600.0: P_ = P_ * (600.0/nP)    # 幂增长饱和：只拦天文数字，不掐正常增长
            pows.append(P_)
        for k in range(N):
            accw = np.zeros(12)
            for j in range(k+1):
                Su[6*k:6*k+6, m*j:m*j+m] = C @ pows[k-j] @ B
                accw += pows[k-j] @ w
            Sw[6*k:6*k+6] = C @ accw
        Ybase = np.tile(C @ x_est, N)
        Yref = np.concatenate([np.concatenate([r - self.d, np.zeros(3)]) for r in r_hor])
        # 名义输入序列 ū：无前馈 = u_prev 恒持（δū=0，即原式）；有前馈 = 逆模型的 u_ff 序列
        if u_ff_hor is None:
            ubar = np.tile(self.u_prev, (N, 1))
        else:
            ubar = np.clip(np.asarray(u_ff_hor, float).reshape(N, m), CO_CONTRACT_MV, CLAMP_MV)
        dub = (ubar - self.u_prev).ravel()              # 名义输入相对 u_prev 的增量（已知）
        DtD = self._D.T @ self._D
        R0 = Ybase + Sw + Su @ dub - Yref
        H = Su.T @ self._Qb @ Su + self.rd*DtD + self.rho*np.eye(m*N)
        # ρ 项：无前馈时罚总电压偏离 CO（linear 项 ρ·(u_prev−CO)，闲置通道泄向 0）；
        # 有前馈时罚的是反馈量 v 本身（中心就是 u_ff，linear 项为零）
        g = Su.T @ self._Qb @ R0 + self.rd*(DtD @ dub)
        if u_ff_hor is None: g = g + self.rho*np.tile(self.u_prev - CO_CONTRACT_MV, N)
        if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
            print("  !! LTV QP 非有限，安全退让"); return _fallback()
        lo = (CO_CONTRACT_MV - ubar).ravel()             # v 的界随名义输入走 → 总量 u∈[CO, 3V]
        hi = (CLAMP_MV - ubar).ravel()
        Lip = np.linalg.norm(H, 2)
        def hard_project(v):
            total = _antagonist_project(ubar + v.reshape(N, m))
            return (total-ubar).ravel()
        V = self._fista(H, g, np.zeros(m*N), Lip, lo, hi, project=hard_project)
        u = _antagonist_project(np.clip(ubar[0] + V[:m], CO_CONTRACT_MV, CLAMP_MV))
        for ch in range(NCH):
            if ch not in ACTIVE_CH: u[ch] = 0.0
        xn = x_est + w + B @ (u - self.u_prev)          # 一步预测：仿射项 + 输入增量响应
        if not np.all(np.isfinite(xn)): xn = x_est.copy()
        # 一步预测也过物理界：q̂ 离实测最多 0.1rad（一拍 5.7°=57°/s，真机极限量级）——
        # 否则正模型训练目标被投毒
        xn[:3] = x_est[:3] + np.clip(xn[:3] - x_est[:3], -0.1, 0.1)
        xn[3:6] = np.clip(xn[3:6], -5.0, 5.0)
        xn[6:] = np.clip(xn[6:], 0.0, CLAMP_MV*KPA_PER_MV)
        self.ok = True                                  # 本拍正常（fallback 路径会置 False）
        self.q_pred = xn[:3]
        self.u_prev = u
        return np.clip(u, 0, CLAMP_MV), xn

    @staticmethod
    def _fista(H, g, U0, Lip, lo, hi, iters=120, project=None,
               extra_grad=None):
        U = np.clip(U0, lo, hi); Y = U.copy(); t = 1.0
        for _ in range(iters):
            grad = H @ Y + g
            if extra_grad is not None:
                grad = grad + extra_grad(Y)
            Un = np.clip(Y - grad/max(Lip, 1e-9), lo, hi)
            if project is not None: Un = project(Un)
            tn = 0.5*(1+math.sqrt(1+4*t*t))
            Y = Un + (t-1)/tn*(Un-U)
            U, t = Un, tn
        return U

def _s_curve5(x):
    """Minimum-jerk 五次 S 曲线：两端速度、加速度都为零。x∈[0,1]。"""
    x = float(np.clip(x, 0.0, 1.0))
    return x*x*x*(10.0 + x*(-15.0 + 6.0*x))


# GUI 示教分量 [X, Y, φ] 对应 PI 使用的模型轴 [q3, q1, q2]。
# 选轴必须看 GUI 分量；倾斜后改变 φ 会使 Euler q3 也变，但那不是 X 示教动作。
GUI_AXIS_TO_MODEL = np.array([2, 0, 1], dtype=int)


def _pi_plan_segment(cur_gui, target_gui, plan_delta_gui, requested_s, dt,
                     speed_dirs, fallback_deg_s, step_reference=False):
    """按 GUI 的 X/Y/φ 差值预先选轴和方向，再生成单轴模型 q 参考。"""
    cur_gui = np.asarray(cur_gui, float).copy()
    target_gui = np.asarray(target_gui, float)
    plan_delta_gui = np.asarray(plan_delta_gui, float)
    active_gui = [a for a in range(3)
                  if abs(float(plan_delta_gui[a])) > PI_PLAN_DEADBAND_DEG]
    deadband_gui = [a for a in range(3)
                    if 0.0 < abs(float(plan_delta_gui[a])) <= PI_PLAN_DEADBAND_DEG]
    x_intent_ignored = []
    # PID 的 9/10 号肌肉只服务 X。Y/φ 运动会被动带出不小的 X 偏移，
    # 因此仅当 X 是本示教段三个 GUI 分量中严格最大的变化时，才承认 X 控制意图。
    # X 排第二、第三或与其他轴并列时，一律禁止 V4/V5。
    if 0 in active_gui:
        dx = abs(float(plan_delta_gui[0]))
        if dx <= max(abs(float(plan_delta_gui[1])), abs(float(plan_delta_gui[2]))):
            active_gui.remove(0)
            x_intent_ignored.append(int(GUI_AXIS_TO_MODEL[0]))
    smallest_ignored = []
    if len(active_gui) > 2:
        # 每个示教段最多调整两轴；按作者要求，无论第三轴本身转了多少，
        # 三者中绝对变化最小的轴都不安排动作，并保持它已有的压力设定。
        drop = min(active_gui, key=lambda a: abs(float(plan_delta_gui[a])))
        active_gui.remove(drop)
        smallest_ignored.append(int(GUI_AXIS_TO_MODEL[drop]))
    deadband_ignored = [int(GUI_AXIS_TO_MODEL[a]) for a in deadband_gui]
    if not active_gui:
        duration = min(PI_PLAN_MAX_DURATION_S, max(float(requested_s), dt))
        n = max(1, int(round(duration/dt)))
        q_hold = q_model_from_st(*cur_gui)
        return ([q_hold.copy() for _ in range(n)], [-1]*n, [-1]*n, cur_gui, n*dt, [],
                deadband_ignored, smallest_ignored, x_intent_ignored)

    def direction(gui_axis, d):
        if gui_axis == 0: return "x+" if d > 0.0 else "x-"
        if gui_axis == 1: return "y+" if d > 0.0 else "y-"
        return "φ+" if d > 0.0 else "φ-"

    needs = []
    for gui_axis in active_gui:
        move_deg = float(target_gui[gui_axis] - cur_gui[gui_axis])
        key = direction(gui_axis, move_deg)
        lim = float(speed_dirs.get(key, {}).get("planning_deg_s", 0.0))
        if lim <= 1e-6:
            lim = max(1.0, float(fallback_deg_s))
        need = max(PI_AXIS_MIN_HOLD_S, 1.875*abs(move_deg)/lim)
        needs.append((gui_axis, int(GUI_AXIS_TO_MODEL[gui_axis]), key, move_deg, need))
    needs.sort(key=lambda z: z[4], reverse=True)       # 最费时间的轴先走
    minimum = sum(z[4] for z in needs)
    total = min(PI_PLAN_MAX_DURATION_S, max(float(requested_s), minimum))
    scale = total/minimum

    channel_for_direction = {"y+": 0, "y-": 1, "φ+": 2,
                             "φ-": 3, "x-": 4, "x+": 5}
    refs, axes, channels, desc = [], [], [], []
    for gui_axis, model_axis, key, move_deg, need in needs:
        dur = need*scale
        n = max(1, int(round(dur/dt)))
        start = cur_gui.copy(); end = cur_gui.copy(); end[gui_axis] = target_gui[gui_axis]
        for j in range(1, n+1):
            gui_ref = end if step_reference else start + (end-start)*_s_curve5(j/n)
            refs.append(q_model_from_st(*gui_ref))
            axes.append(model_axis)
            channels.append(channel_for_direction[key])
        cur_gui = end
        desc.append((key, move_deg, n*dt))
    return (refs, axes, channels, cur_gui, len(refs)*dt, desc,
            deadband_ignored, smallest_ignored, x_intent_ignored)


def _muscle_length_refs_for_plan(cur_length, target_length, channels,
                                 step_reference=False):
    """把角度 PI 的顺序通道计划转换成肌肉长度参考。

    每个连续子阶段只改变当前计划通道的长度目标；未被本段选中的通道保持
    上一示教点的目标。因此肌肉 PI 仍由真实长度闭环，但不会六路同时抢调。
    """
    cur = np.asarray(cur_length, float).copy()
    target = np.asarray(target_length, float)
    channels = [int(ch) for ch in channels]
    refs = []
    i = 0
    while i < len(channels):
        ch = channels[i]
        j = i + 1
        while j < len(channels) and channels[j] == ch:
            j += 1
        count = j - i
        if ch < 0:
            refs.extend([cur.copy() for _ in range(count)])
        else:
            start = float(cur[ch])
            for k in range(1, count + 1):
                row = cur.copy()
                row[ch] = (float(target[ch]) if step_reference else
                           start + (float(target[ch])-start)*_s_curve5(k/count))
                refs.append(row)
            cur[ch] = float(target[ch])
        i = j
    return refs, cur


def load_ref(args):
    pts = []
    if args.traj:
        blob = json.load(open(args.traj, encoding="utf-8"))
        raw = blob["points"] if isinstance(blob, dict) else blob
        for pt in raw:
            v3 = [float(v) for v in pt["q"][:3]]
            qm = q_model_from_st(v3[0], v3[1], v3[2], args.mount)
            pts.append((qm, float(pt.get("t", 2.0))))
        print(f"# 轨迹：{args.traj}，{len(pts)} 点（st° → 模型 q）")
    elif args.targets:
        for seg in args.targets.split(";"):
            a = [float(x) for x in seg.split(",")]
            pts.append((q_model_from_st(a[0], a[1], 0.0, args.mount), a[2] if len(a) > 2 else 3.0))
    else:
        sys.exit("--traj teach.json 或 --targets \"sx,sy,秒;...\"")
    ref, cur = [np.zeros(3)], np.zeros(3)
    for qm, dur in pts:
        n = max(1, int(round(dur/DT)))
        for k in range(1, n+1): ref.append(cur + (qm-cur)*_s_curve5(k/n))
        cur = qm
    ref += [cur]*int(2.0/DT)
    return np.array(ref)

def mode_mpc(args):
    if not os.path.exists(PARAMS_JSON): sys.exit("没有 params.json，先 ident + fit")
    p = json.load(open(PARAMS_JSON))
    print(f"# 模型：shoulder_pam_model (N_PAM={spm.N_PAM}) + 未知量 {['%s=%.3g'%(k,p[k]) for k in ('gamma','kf','c_damp')]}")
    plant = Plant(p); mpc = MPC(plant, N=args.N)
    ref = load_ref(args)
    link = Link(args.port); qref = link.capture_ref()
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, time.strftime("mpc_%Y%m%d_%H%M%S.csv"))
    print(f"# MPC：{len(ref)} 拍（{len(ref)*DT:.0f}s），N={args.N} → {path}")
    x = np.zeros(12); q_prev = None
    log = []
    try:
        for k in range(len(ref)):
            tn = time.time() + DT
            while time.time() < tn: link.poll(); time.sleep(0.003)
            if link.quat is None: continue
            q = q_model_from_quat(link.quat, qref, args.mount)
            # 角度急停已取消（作者定：气压侧兜底）
            qd = (q - q_prev)/DT if q_prev is not None else np.zeros(3)
            q_prev = q
            x[:3] = q; x[3:6] = 0.7*x[3:6] + 0.3*qd          # 姿态用实测，速度低通
            hor = [ref[min(k+j, len(ref)-1)] for j in range(1, args.N+1)]
            u, xn = mpc.step(x.copy(), hor)
            x[6:] = xn[6:]                                    # 气压状态：模型递推
            link.send_mv(u)
            log.append([k*DT] + [math.degrees(v) for v in q] + [math.degrees(v) for v in ref[k]]
                       + [round(v) for v in u] + [w if w is not None else "" for w in link.lens])
            if k % 10 == 0:
                e = math.degrees(np.linalg.norm(ref[k]-q))
                print(f"  t={k*DT:5.1f}s q=({math.degrees(q[0]):+6.2f},{math.degrees(q[1]):+6.2f},{math.degrees(q[2]):+6.2f})°"
                      f" |e|={e:5.2f}° u={[round(v) for v in u]}")
    finally:
        link.all_zero()
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "q1", "q2", "q3", "r1", "r2", "r3"]
                        + [f"u{i}" for i in range(NCH)] + [f"L{i+1}" for i in range(10)])
            wr.writerows(log)
        print(f"# 日志 → {path}")
    if log and not args.noplot:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            a = np.array([[float(v) for v in r[:7+NCH]] for r in log])
            fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            for i, nm in enumerate(["q1", "q2", "q3"]):
                ax[0].plot(a[:, 0], a[:, 4+i], "--", lw=1)
                ax[0].plot(a[:, 0], a[:, 1+i], label=nm)
            ax[0].set_ylabel("deg"); ax[0].legend(); ax[0].set_title("MPC (author's model) tracking taught trajectory")
            for i in range(NCH): ax[1].plot(a[:, 0], a[:, 7+i], label=f"u{i}")
            ax[1].set_ylabel("mV"); ax[1].set_xlabel("s"); ax[1].legend(ncol=3, fontsize=8)
            png = path.replace(".csv", ".png"); fig.savefig(png, dpi=130)
            print(f"# 图 → {png}")
        except ImportError:
            print("# （matplotlib 未装，跳过画图）")

# ───────────────────────── serve：GUI 遥控模式 ──────────────────────────────
# GUI 示教面板选「MPC·控制站」时走这里：本脚本占串口，GUI 通过 127.0.0.1:8765
# 下发轨迹/启停、轮询状态画曲线。协议（JSON，CORS 全开）：
#   POST /start {points:[{q:[sx,sy,psi]°, t:秒}, ...]}   开始 MPC 追踪
#   POST /stop                                            清零停止
#   GET  /status → {mode, t, ref:[sx,sy]°, meas:[sx,sy]°, u:[6], lens:[10]}
def st3_from_q_model(q):
    """模型 q → GUI 的 (s_x, s_y, ψ)°，与 q_model_from_st 精确互逆。"""
    R = spm.R_M(q)
    # 摆动-扭转分解：扭转绕 z。R 的 z 列 = 摆动把 z 转到哪，与扭转无关。
    z = R[:, 2]
    c = max(-1.0, min(1.0, z[2]))
    ax, ay = -z[1], z[0]
    n = math.hypot(ax, ay)
    th = math.atan2(n, c)
    if n < 1e-12:
        sx, sy = (math.pi, 0.0) if c < 0.0 else (0.0, 0.0)
    else:
        sx, sy = th*ax/n, th*ay/n

    # R = R_swing Rz(ψ)，用 R_swing 第一列在 R 的前两列上的投影恢复 ψ。
    sth = math.hypot(sx, sy)
    if sth < 1e-12:
        Rsw = np.eye(3)
    else:
        k = np.array([sx/sth, sy/sth, 0.0])
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rsw = np.eye(3) + math.sin(sth)*K + (1.0-math.cos(sth))*(K @ K)
    cp = float(Rsw[:, 0] @ R[:, 0])
    sp = -float(Rsw[:, 0] @ R[:, 1])
    psi = math.atan2(sp, cp)
    return math.degrees(sx), math.degrees(sy), math.degrees(psi)


def st_from_q_model(q):
    """模型 q → GUI 的 (s_x, s_y)°；保留旧页面接口。"""
    return st3_from_q_model(q)[:2]

# 控制响应遥测环形缓冲：bridge 的 /step 每拍写入，qt_gui 的「控制响应」窗口读取绘图
from collections import deque
class ResponseTelemetry(deque):
    """Bound plotting work without discarding the current response for export."""

    def __init__(self, maxlen=6000):
        super().__init__(maxlen=maxlen)
        self._export_rows = []
        self._export_lock = threading.RLock()

    def append(self, record):
        with self._export_lock:
            self._export_rows.append(record)
            super().append(record)

    def clear(self):
        with self._export_lock:
            self._export_rows.clear()
            super().clear()

    def export_snapshot(self):
        with self._export_lock:
            return list(self._export_rows)


TELEM = ResponseTelemetry(maxlen=6000)
# Available before bridge startup; mode_bridge publishes its live state here.
BR = {"mode": "idle", "codex_mpc": False, "codex_auto_status": {}}
HOST_LOCKS = ""                                  # qt_gui 串口层从固件帧直接更新（o/w/L×6）
HOST_MPC_COMMAND = None                          # bridge /step 的一次性原子(lock,u)转发槽
HOST_LLM_API_KEYS = {"claude": ""}              # 仅驻留内存，绝不写入 gains.json
HOST_LLM_API_KEYS_LOCK = threading.Lock()


def set_transient_llm_api_key(provider, api_key):
    """Set/clear a provider key for this process without persisting it."""
    name = str(provider or "").strip().lower()
    if name not in HOST_LLM_API_KEYS:
        return False
    with HOST_LLM_API_KEYS_LOCK:
        HOST_LLM_API_KEYS[name] = str(api_key or "").strip()
    return True


def has_transient_llm_api_key(provider):
    name = str(provider or "").strip().lower()
    with HOST_LLM_API_KEYS_LOCK:
        return bool(HOST_LLM_API_KEYS.get(name, ""))


def get_transient_llm_api_key(provider):
    name = str(provider or "").strip().lower()
    with HOST_LLM_API_KEYS_LOCK:
        return str(HOST_LLM_API_KEYS.get(name, ""))

SRV = {"cmd": None, "pts": None, "mount": "ry-",
       "mode": "idle", "t": 0.0, "ref": [0, 0], "meas": [0, 0],
       "u": [0]*NCH, "lens": [None]*10}
SRV_LK = threading.Lock()

def _http_server(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _hdr(self, code=200):
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")  # Chrome PNA
            self.send_header("Content-Type", "application/json")
            self.end_headers()
        def do_OPTIONS(self): self._hdr(204)
        def do_GET(self):
            with SRV_LK:
                body = json.dumps({k: SRV[k] for k in
                                   ("mode", "t", "ref", "meas", "u", "lens")})
            self._hdr(); self.wfile.write(body.encode())
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            with SRV_LK:
                if self.path == "/start": SRV["cmd"] = "start"; SRV["pts"] = data.get("points", [])
                elif self.path == "/stop": SRV["cmd"] = "stop"
            self._hdr(); self.wfile.write(b"{}")
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"# GUI 遥控端口 http://127.0.0.1:{port}（示教面板选「MPC·控制站」）")

def mode_serve(args):
    if not os.path.exists(PARAMS_JSON): sys.exit("没有 params.json，先 ident + fit")
    p = json.load(open(PARAMS_JSON))
    SRV["mount"] = args.mount
    plant = Plant(p)
    _http_server(args.http)                        # HTTP 先起：串口有问题也能向 GUI 报状态
    with SRV_LK: SRV["mode"] = "no-serial"
    link = None; qref = None
    print("# ⚠ serve 模式已弃用：现 GUI 走 bridge 网关（qt_gui 内嵌），二者串口互斥。")
    print("# 等串口…（GUI 的「连接 IMU」必须断开；Ctrl+C 退出）")
    ref = None; mpc = None; k = 0; x = np.zeros(12); q_prev = None
    try:
        while True:
            if link is None:                       # 反复试开串口，占用中就等着
                try:
                    link = Link(args.port)
                except (Exception, SystemExit):
                    with SRV_LK: SRV["mode"] = "no-serial"
                    time.sleep(2); continue
                print("# 串口已开，等 IMU 帧…")
            if qref is None:                       # 等到第一帧才算就绪；对零在 /start 时做
                link.poll()
                if link.quat is None:
                    with SRV_LK: SRV["mode"] = "no-imu"
                    time.sleep(0.1); continue
                qref = link.quat.copy()
                with SRV_LK: SRV["mode"] = "idle"
                print("# 就绪，待命：等 GUI 下发轨迹…")
            with SRV_LK:
                cmd, pts = SRV["cmd"], SRV["pts"]; SRV["cmd"] = None
            if cmd == "stop" and ref is not None:
                link.all_zero(); ref = None
                with SRV_LK: SRV["mode"] = "idle"; SRV["u"] = [0]*NCH
                print("# GUI 停止，已清零")
            if cmd == "start" and pts:
                for _ in range(20): link.poll(); time.sleep(0.01)
                if link.quat is not None: qref = link.quat.copy()   # 以启动瞬间的静止姿态为零位
                fake = type("A", (), {"traj": "", "targets": "", "mount": args.mount})()
                fake.targets = ";".join(f'{q["q"][0]},{q["q"][1]},{q.get("t",2)}' for q in pts)
                ref = load_ref(fake); mpc = MPC(plant, N=args.N)
                k = 0; x = np.zeros(12); q_prev = None
                with SRV_LK: SRV["mode"] = "mpc"
                print(f"# GUI 启动 MPC：{len(pts)} 点，{len(ref)*DT:.0f}s")
            if ref is None:
                link.poll(); time.sleep(0.03); continue
            tn = time.time() + DT
            while time.time() < tn: link.poll(); time.sleep(0.003)
            if link.quat is None: continue
            q = q_model_from_quat(link.quat, qref, args.mount)
            # 角度急停已取消（作者定：气压侧兜底）
            qd = (q - q_prev)/DT if q_prev is not None else np.zeros(3)
            q_prev = q
            x[:3] = q; x[3:6] = 0.7*x[3:6] + 0.3*qd
            hor = [ref[min(k+j, len(ref)-1)] for j in range(1, args.N+1)]
            u, xn = mpc.step(x.copy(), hor)
            x[6:] = xn[6:]
            link.send_mv(u)
            with SRV_LK:
                SRV["t"] = k*DT
                SRV["ref"] = list(st_from_q_model(ref[min(k, len(ref)-1)]))
                SRV["meas"] = list(st_from_q_model(q))
                SRV["u"] = [int(v) for v in u]
                SRV["lens"] = link.lens[:]
            k += 1
            if k >= len(ref):
                link.all_zero(); ref = None
                with SRV_LK: SRV["mode"] = "idle"
                print("# 轨迹完成，已清零，待命")
    finally:
        if link: link.all_zero()

# ───────────────────────── PI 基线控制器（对照 MPC 用）──────────────────────
# ───────────────────────── 数据锚定 MPC（2026-09-04，作者定："要个大体模型，但别这么依赖它"）──
# 实机结论（PI 日志 1215 拍交叉验证）：作者物理模型的【方向】对、【动力学】完全不是真机——
# 喂实测压预测真机姿态，5 拍误差 21°–52°，1 拍数值爆掉；而只有结构、参数来自数据的离散模型
# 10 拍自由预测 2.3°。所以 MPC 的预测模型换成下面这个，物理模型只当先验（各路方向、
# 初始时间常数），参数由真机日志岭回归拟合，每轮结束再用本轮数据重拟合（PPT"正模型进 MPC"）。
#   q⁺  = q + A_v·Δq − K·q + B_p·P + c           Δq = q − q⁻（每拍位移）, P≥0
#   P⁺  = P + (DT/τ)(0.09·u − P)                  τ 分充/放、逐路，也从数据拟合
# 状态 x = [q(3), Δq(3), P(6)]；没有逐拍线性化、expm、卫生限幅——LTI 预测天然有界。
DM_PATH = os.path.join(HERE, "data_model.json")
PURE_DM_PATH = os.path.join(HERE, "pure_mpc_model.pt")
PURE_DM_META_PATH = os.path.join(HERE, "pure_mpc_model.json")
SPEED_LIMIT_PATH = os.path.join(HERE, "speed_limits.json")
X_COMP_PATH = os.path.join(HERE, "x_compensation.json")
DM_SCALE = np.array([0.25*DT]*3 + [0.3]*3 + [100.0]*6 + [1.0]) # 特征尺度：Δq 随采样周期缩放

class DataModel:
    LAM = 5.0        # 岭回归强度（归一化特征下），拉向先验/上一版参数

    def __init__(self, plant):
        self.plant = plant
        self.Av = 0.3*np.eye(3); self.K = 0.05*np.eye(3); self.c = np.zeros(3)
        G = np.zeros((3, NCH))                     # 物理先验：各路稳态增益 rad/kPa（方向由作者模型定）
        for ch in range(NCH):
            uv = np.zeros(NCH); uv[ch] = 600.0
            G[:, ch] = _sim_settle(plant, uv)/600.0/KPA_PER_MV   # 先验用名义换算即可，量级会被数据改写
        self.Bp = self.K @ G                       # 稳态 q = K⁻¹·B_p·g = G·g
        self.tu = np.array(plant.tu, float); self.td = np.array(plant.td, float)
        self.g = np.full(NCH, KPA_PER_MV)          # 指令→稳态压力增益 kPa/mV（逐路，从数据学）
        self.p_dead = [False]*NCH                  # 压力监测口坏掉的通道（改用指令估算）
        self.n_fit = 0; self.rmse_deg = None; self.src = "physics-prior"

    # ── 参数打包 Θ(3×13) = [A_v | K | B_p | c]，与特征 [Δq, −q, P, 1] 对应 ──
    def theta(self): return np.hstack([self.Av, self.K, self.Bp, self.c[:, None]])
    def set_theta(self, Th):
        Th = np.asarray(Th, float)
        self.Av = np.clip(Th[:, 0:3], -0.95, 0.95)
        K = Th[:, 3:6]; np.fill_diagonal(K, np.clip(np.diag(K), 0.005, 0.5)); self.K = K
        self.Bp = np.clip(Th[:, 6:12], -0.01, 0.01)
        self.c = np.zeros(3)                       # q=0、P=0 必须静止，禁止模型产生虚假漂移

    def resample(self, old_dt):
        """把按 old_dt 辨识的姿态离散模型转换到当前 DT。

        先把每拍位移 Δq 换成速度，再对含常值压力输入的增广离散系统取
        DT/old_dt 次幂，最后换回新采样周期下的 Δq。气路 τ/g 是连续参数，
        仍由 jac() 使用新 DT 离散。
        """
        old_dt = float(old_dt)
        if old_dt <= 0 or DT <= 0: return False
        if abs(old_dt-DT) <= 1e-9: return True
        try:
            from scipy.linalg import fractional_matrix_power
            I = np.eye(3)
            Ay = np.block([[I-self.K, old_dt*self.Av],
                           [-self.K/old_dt, self.Av]])
            Gy = np.vstack([self.Bp, self.Bp/old_dt])
            aug = np.block([[Ay, Gy],
                            [np.zeros((NCH, 6)), np.eye(NCH)]])
            frac = fractional_matrix_power(aug, DT/old_dt)
            if np.max(np.abs(np.imag(frac))) > 1e-7: return False
            frac = np.real(frac)
            Anew, Gnew = frac[:6, :6], frac[:6, 6:]
            Th = self.theta()
            Th[:, 0:3] = Anew[:3, 3:6]/DT
            Th[:, 3:6] = I-Anew[:3, :3]
            Th[:, 6:12] = Gnew[:3, :]
            self.set_theta(Th)
            self.c = np.zeros(3)
            return bool(np.all(np.isfinite(self.theta())))
        except Exception:
            return False

    @staticmethod
    def feats(q, dq, P):
        P = np.asarray(P, float)
        return np.concatenate([dq, -np.asarray(q, float), P, [1.0]])

    def fit(self, X, Y, prior=None, lam=None):
        """岭回归：Θ = argmin ‖XΘᵀ−Y‖² + λ‖Θ−Θ_prior‖²（归一化特征）。返回 1 拍 RMSE(°)。"""
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        if len(X) < 30: return None
        Xs = X / DM_SCALE
        Thp = (self.theta() if prior is None else np.asarray(prior, float)) * DM_SCALE   # 先验换到归一化空间
        lam = self.LAM if lam is None else lam
        A = Xs.T @ Xs + lam*np.eye(Xs.shape[1])
        Ths = np.linalg.solve(A, Xs.T @ Y + lam*Thp.T).T
        self.set_theta(Ths / DM_SCALE)
        res = Y - X @ self.theta().T
        self.rmse_deg = float(np.degrees(np.sqrt(np.mean(np.sum(res**2, axis=1)))))
        self.n_fit = len(X)
        return self.rmse_deg

    # 气路拟合的搜索网格（多步自由预测评分，见 fit_pneu）
    G_GRID = np.array([0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.105, 0.120])
    TAU_GRID = np.array([0.3, 0.6, 1.0, 1.5, 2.2, 3.0, 4.5])
    PNEU_WIN = max(5, int(round(3.0/DT)))  # 评分窗口保持约 3s（10 Hz 时为 30 拍）
    DEAD_SPAN_KPA = 15.0     # 指令扫过全程、实测压变化不到这个数 → 判该路监测口坏

    def fit_pneu(self, blocks, verbose=False):
        """逐路拟合 指令→压力：P⁺ = P + DT(g·u − P)/τ，充/放两个 τ。
        【为什么不用一步回归】：τ≈3s 的慢环节做一步最小二乘是病态的——u 与 P 高度相关，
        回归会把增益推到 0.12（实测平台值只有 0.045~0.089），MPC 于是以为"少给点电压
        压力就到位"，出力过小、臂不动（2026-09-04 03:07 实机）。这里改成【多步自由预测】
        评分：每 3s 窗口只用窗首的实测压起步，其余全靠模型推，选窗内 RMSE 最小的 (g,τ↑,τ↓)。
        同时体检监测口：指令扫过全程而实测压几乎不动 → 标记 dead，该路改用指令估算。"""
        segs = [[] for _ in range(NCH)]
        for U_, P_ in blocks:
            U_ = np.asarray(U_, float); P_ = np.asarray(P_, float)
            n = min(len(U_), len(P_))
            if n < self.PNEU_WIN + 1: continue
            for c in range(NCH):
                for s in range(0, n - self.PNEU_WIN, max(1, self.PNEU_WIN//2)):
                    u = U_[s:s+self.PNEU_WIN, c]; p = P_[s:s+self.PNEU_WIN, c]
                    if u.max() < 150 and p.max() < 5: continue   # 全程没气，无信息
                    segs[c].append((u, p))
        for c in range(NCH):
            if len(segs[c]) < 5: continue
            Us = np.array([s[0] for s in segs[c]])               # (nseg, win)
            Ps = np.array([s[1] for s in segs[c]])
            # 监测口体检：指令动了、压力不动 → dead
            self.p_dead[c] = bool(Us.max() > 1500 and (Ps.max() - Ps.min()) < self.DEAD_SPAN_KPA)
            if self.p_dead[c]:
                if verbose: print(f"#   ⚠ V{c} 压力监测口无响应（指令到 {Us.max():.0f}mV，压力只变 "
                                  f"{Ps.max()-Ps.min():.0f}kPa）→ 该路改用指令估算压力")
                continue
            best = None
            for gg in self.G_GRID:
                Pc = gg*Us                                        # (nseg, win) 各拍的稳态目标
                for tu in self.TAU_GRID:
                    for td in self.TAU_GRID:
                        ph = Ps[:, 0].copy(); err = 0.0
                        for k in range(self.PNEU_WIN - 1):
                            pc = Pc[:, k]
                            tau = np.where(pc >= ph, tu, td)
                            ph = ph + DT*(pc - ph)/tau
                            d = ph - Ps[:, k+1]; err += float(d @ d)
                        if best is None or err < best[0]: best = (err, gg, tu, td)
            rmse = math.sqrt(best[0]/max(1, Us.shape[0]*(self.PNEU_WIN-1)))
            self.g[c] = float(best[1]); self.tu[c] = float(best[2]); self.td[c] = float(best[3])
            if verbose: print(f"#   V{c}: g={self.g[c]:.3f} kPa/mV  τ↑={self.tu[c]:.1f}s τ↓={self.td[c]:.1f}s"
                              f"  （3s 自由预测 RMSE {rmse:.1f} kPa，{Us.shape[0]} 窗）")
        # 监测口坏的通道测不出自己的气路参数 → 借用健康通道的中位数（比留着物理先验靠谱：
        # params.json 里 V5 的 τ 是 0.26s，用它估算会以为气瞬间就到）
        ok = [c for c in range(NCH) if not self.p_dead[c]]
        if ok and len(ok) < NCH:
            gm, tum, tdm = np.median(self.g[ok]), np.median(self.tu[ok]), np.median(self.td[ok])
            for c in range(NCH):
                if self.p_dead[c]:
                    self.g[c], self.tu[c], self.td[c] = float(gm), float(tum), float(tdm)
                    if verbose: print(f"#     V{c} 气路参数借用健康通道中位数：g={gm:.3f} τ↑={tum:.1f}s τ↓={tdm:.1f}s")

    def p_estimate(self, P_meas, u_prev, P_state):
        """监测口坏掉的通道用指令估算压力顶上（其余通道原样用实测）。"""
        P = np.array(P_meas, float)
        for c in range(NCH):
            if self.p_dead[c]:
                tau = self.tu[c] if self.g[c]*u_prev[c] >= P_state[c] else self.td[c]
                P[c] = P_state[c] + DT*(self.g[c]*u_prev[c] - P_state[c])/max(0.05, tau)
        return P

    def jac(self, x, u):
        """(A, B, w)：δx⁺ = A δx + B δu + w，w = f(x,u) − x（与 MPC.step 的 δ 形式一致）。"""
        q, dq, P = x[:3], x[3:6], x[6:]
        Pc = np.clip(u, 0, CLAMP_MV)*self.g        # 指令对应的稳态压力（增益从数据学，非写死 0.09）
        tau = np.where(Pc >= P, self.tu, self.td)
        Dd = np.minimum(1.0, DT/np.maximum(tau, 0.05))
        A = np.zeros((12, 12)); B = np.zeros((12, NCH))
        A[0:3, 0:3] = np.eye(3) - self.K; A[0:3, 3:6] = self.Av; A[0:3, 6:] = self.Bp
        A[3:6, 0:3] = -self.K;            A[3:6, 3:6] = self.Av; A[3:6, 6:] = self.Bp
        A[6:, 6:] = np.diag(1.0 - Dd)
        B[6:, :] = np.diag(self.g*Dd)
        qn = q + self.Av @ dq - self.K @ q + self.Bp @ P + self.c
        Pn = P + Dd*(Pc - P)
        w = np.concatenate([qn - q, (qn - q) - dq, Pn - P])
        return A, B, w

    def gain_table_deg(self):
        """稳态增益：每路 +1000mV 指令的姿态变化（°）——"哪个方向要动多少"一目了然。
        含气路增益 g（指令→压力）与姿态增益 K⁻¹B_p（压力→角度）两段。"""
        P1000 = self.g*1000.0
        G = np.linalg.solve(self.K, self.Bp * P1000[None, :])
        return np.degrees(G)

    def save(self, path=None):
        path = path or DM_PATH                     # 运行时取全局：测试可改路径，不写真机文件
        try:
            tmp = path + ".tmp"
            json.dump({"sample_dt": DT,
                       "theta": self.theta().tolist(), "tu": self.tu.tolist(), "td": self.td.tolist(),
                       "g": self.g.tolist(), "p_dead": list(self.p_dead), "n_fit": self.n_fit,
                       "rmse_deg": self.rmse_deg, "src": self.src}, open(tmp, "w"))
            os.replace(tmp, path)
        except Exception:
            pass

    def load(self, path=None):
        path = path or DM_PATH
        try:
            b = json.load(open(path))
            old_dt = float(b.get("sample_dt", -1.0))
            Th = np.array(b["theta"], float)
            if Th.shape != (3, 13): return False
            self.set_theta(Th); self.tu = np.array(b["tu"], float); self.td = np.array(b["td"], float)
            if "g" in b: self.g = np.clip(np.array(b["g"], float), 0.01, 0.15)
            if "p_dead" in b: self.p_dead = [bool(v) for v in b["p_dead"]][:NCH]
            self.n_fit = int(b.get("n_fit", 0)); self.rmse_deg = b.get("rmse_deg"); self.src = b.get("src", "file")
            if abs(old_dt-DT) > 1e-9:
                if not self.resample(old_dt): return False
                self.src = f"{self.src}:resampled-{old_dt:g}to{DT:g}s"
                self.rmse_deg = None                # 旧周期的一拍 RMSE 不能冒充新周期指标
                self.converted_from_dt = old_dt
                self.save(path)                     # 一次转换后固定保存为当前采样周期
            return True
        except Exception:
            return False

def _dm_samples_from_logs(log_dir=None):
    """从带实测压列的控制日志（pi_*/mpc_*）攒 (X, Y, U, P0, P1)。每份日志逐路扣掉静息偏移
    （5% 分位）——旧固件监测口零点没扣，新固件已扣，同一规则两边都对。"""
    log_dir = log_dir or LOG_DIR
    X, Y, U, Pa, Pb, blocks = [], [], [], [], [], []
    if not os.path.isdir(log_dir): return X, Y, U, Pa, Pb, blocks
    for fn in sorted(os.listdir(log_dir)):
        if not (fn.startswith("pi_") or fn.startswith("mpc_")) or fn.endswith("_laps.csv"): continue
        try:
            rows = list(csv.DictReader(open(os.path.join(log_dir, fn))))
            need = (["t", "q1", "q2", "q3"] + [f"u{i}" for i in range(NCH)]
                    + [f"P{i}" for i in range(NCH)])
            if not rows or not all(k in rows[0] for k in need): continue
            d = []
            # 新日志同时保存控制器请求 u 和锁状态门控后的真实比例阀输出 uact。
            # 重建正模型必须优先用 uact，避免把100ms预锁期间误当成已经施压。
            u_prefix = "uact" if all(f"uact{i}" in rows[0] for i in range(NCH)) else "u"
            for r in rows:
                try:
                    d.append([float(r["t"])] + [float(r[f"q{i}"]) for i in (1, 2, 3)]
                             + [float(r[f"{u_prefix}{i}"]) for i in range(NCH)]
                             + [float(r[f"P{i}"]) for i in range(NCH)])
                except ValueError: continue
            d = np.array(d)
            if len(d) < 40: continue
            # 只使用与当前控制周期相同的日志；5 Hz 模型由 load() 精确转换为 10 Hz，
            # 不对 5 Hz 日志凭空插值制造训练样本。
            dt_log = float(np.median(np.diff(d[:, 0]))) if len(d) > 1 else DT
            if abs(dt_log-DT) > 0.02: continue
            t = d[:, 0]; q = np.radians(d[:, 1:4]); u = d[:, 4:10]; P = d[:, 10:16]
            P = np.maximum(0.0, P - np.percentile(P, 5, axis=0))
            s = 0                                       # 连续段（时间不连续处断开）→ 气路多步拟合要用
            for k in range(1, len(d)):
                brk = abs((t[k] - t[k-1]) - DT) > 0.02
                if brk or k == len(d) - 1:
                    e = k if brk else k + 1
                    if e - s >= DataModel.PNEU_WIN + 1: blocks.append((u[s:e], P[s:e]))
                    s = k
            for k in range(1, len(d) - 1):
                if abs((t[k+1] - t[k]) - DT) > 0.02 or abs((t[k] - t[k-1]) - DT) > 0.02: continue
                X.append(DataModel.feats(q[k], q[k] - q[k-1], P[k])); Y.append(q[k+1] - q[k])
                U.append(u[k]); Pa.append(P[k]); Pb.append(P[k+1])
        except Exception:
            continue
    return X, Y, U, Pa, Pb, blocks


class PureDataModel:
    """Neural state-space model fitted only from pure_mpc hardware logs.

    The TorchScript artifact maps ``(x=[q,dq,P], uact)`` directly to ``x_next``.
    ``jac`` obtains a local linear model from symmetric finite differences so
    the existing constrained QP can be reused without a physical/PAM model.
    """
    def __init__(self, model_path=PURE_DM_PATH, meta_path=PURE_DM_META_PATH):
        import torch
        self.torch = torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self.net = torch.jit.load(model_path, map_location="cpu").eval()
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        if abs(float(self.meta.get("sample_dt", -1.0)) - DT) > 1e-9:
            raise ValueError("纯数据模型采样周期与控制周期不一致")
        self.n_fit = int(self.meta.get("training_samples", 0))
        self.rmse_deg = float(self.meta.get("validation_metrics", {}).get(
            "one_step_angle_rmse_deg", float("nan")))
        self.src = "pure-hardware-NN"
        self.p_dead = [False] * NCH
        self._eps_x = np.array([1e-3] * 3 + [5e-4] * 3 + [0.5] * 6, float)
        self._eps_u = np.full(NCH, 5.0, float)

    def _predict_batch(self, xb, ub):
        with self.torch.no_grad():
            xt = self.torch.as_tensor(np.asarray(xb, np.float32))
            ut = self.torch.as_tensor(np.asarray(ub, np.float32))
            ans = self.net(xt, ut).cpu().numpy().astype(float)
        return ans

    def jac(self, x, u):
        x = np.asarray(x, float); u = np.asarray(u, float)
        xb = [x]; ub = [u]
        for j, eps in enumerate(self._eps_x):
            xp, xm = x.copy(), x.copy(); xp[j] += eps; xm[j] -= eps
            xb.extend((xp, xm)); ub.extend((u, u))
        for j, eps in enumerate(self._eps_u):
            up, um = u.copy(), u.copy(); up[j] += eps; um[j] -= eps
            xb.extend((x, x)); ub.extend((up, um))
        pred = self._predict_batch(np.asarray(xb), np.asarray(ub))
        f0 = pred[0]
        A = np.column_stack([(pred[1 + 2*j] - pred[2 + 2*j]) /
                             (2.0*self._eps_x[j]) for j in range(12)])
        off = 1 + 2*12
        B = np.column_stack([(pred[off + 2*j] - pred[off + 2*j + 1]) /
                             (2.0*self._eps_u[j]) for j in range(NCH)])
        if not (np.all(np.isfinite(f0)) and np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            raise FloatingPointError("纯数据模型局部线性化产生非有限数")
        return A, B, f0 - x

    def p_estimate(self, P_meas, u_prev, P_state):
        return np.asarray(P_meas, float).copy()


SEQUENCE_MODEL_FILES = {
    "history_mlp": (os.path.join(HERE, "pure_mpc_history_mlp.pt"),
                    os.path.join(HERE, "pure_mpc_history_mlp.json")),
    "gru": (os.path.join(HERE, "pure_mpc_gru.pt"),
            os.path.join(HERE, "pure_mpc_gru.json")),
}


class SequencePureDataModel:
    """Ten-tick residual history model used by the LTV MPC.

    The fixed pure-data MLP supplies the safe nominal prediction.  A history
    MLP or GRU adds a learned correction using the previous 0.5 s.  During an
    MPC rollout the history is shifted with predicted states and candidate
    controls, so the finite-difference Jacobian sees the same temporal model
    that generates the nominal 20-tick trajectory.
    """

    def __init__(self, architecture, model_path=None, meta_path=None):
        import torch
        architecture = str(architecture).strip().lower()
        if architecture not in SEQUENCE_MODEL_FILES:
            raise ValueError(f"未知时序顺模型：{architecture}")
        default_model_path, default_meta_path = SEQUENCE_MODEL_FILES[architecture]
        model_path = str(model_path or default_model_path)
        meta_path = str(meta_path or default_meta_path)
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"时序顺模型尚未训练：{model_path}")
        self.torch = torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self.base = PureDataModel(PURE_DM_PATH, PURE_DM_META_PATH)
        self.correction = torch.jit.load(model_path, map_location="cpu").eval()
        with open(meta_path, encoding="utf-8") as stream:
            self.meta = json.load(stream)
        if abs(float(self.meta.get("sample_dt", -1.0)) - DT) > 1e-9:
            raise ValueError("时序顺模型采样周期与MPC控制周期不一致")
        self.history_steps = int(self.meta.get("history_steps", 10))
        if self.history_steps != 10:
            raise ValueError("当前控制器只接受10拍历史模型")
        self.architecture = architecture
        self.n_fit = int(self.meta.get("training_samples", 0))
        vm = self.meta.get("validation_metrics", {})
        self.rmse_deg = float(vm.get("moving_one_step_angle_rmse_deg", float("nan")))
        self.src = f"pure-hardware-{architecture}"
        self.p_dead = [False] * NCH
        self._eps_x = np.array([1e-3] * 3 + [5e-4] * 3 + [0.5] * 6, float)
        self._eps_u = np.full(NCH, 5.0, float)
        self._actual_history = []
        self._roll_history = []
        self._roll_confidence = 0.0
        with torch.no_grad():
            test_x = torch.zeros((1, self.history_steps, 12))
            test_u = torch.zeros((1, self.history_steps, NCH))
            answer = self.correction(test_x, test_u)
        if tuple(answer.shape) != (1, 9) or not bool(torch.isfinite(answer).all()):
            raise ValueError("时序修正模型接口必须是 ([B,10,12],[B,10,6])->[B,9]")

    def reset_history(self):
        self._actual_history = []
        self._roll_history = []
        self._roll_confidence = 0.0

    def observe_actual(self, state, actual_control):
        x = np.asarray(state, float)
        u = np.asarray(actual_control, float)
        if (x.shape != (12,) or u.shape != (NCH,) or
                not np.all(np.isfinite(x)) or not np.all(np.isfinite(u))):
            self.reset_history()
            return
        self._actual_history.append((x.copy(), u.copy()))
        self._actual_history = self._actual_history[-(self.history_steps - 1):]

    def begin_rollout(self, current_state):
        current = np.asarray(current_state, float)
        history = [(x.copy(), u.copy()) for x, u in self._actual_history]
        self._roll_confidence = min(1.0, len(history) / float(self.history_steps - 1))
        if history:
            pad_x = history[0][0].copy()
        else:
            pad_x = current.copy()
        while len(history) < self.history_steps - 1:
            history.insert(0, (pad_x.copy(), np.zeros(NCH)))
        self._roll_history = history[-(self.history_steps - 1):]

    def advance_rollout(self, state, control):
        self._roll_history.append((np.asarray(state, float).copy(),
                                   np.asarray(control, float).copy()))
        self._roll_history = self._roll_history[-(self.history_steps - 1):]
        self._roll_confidence = min(1.0, self._roll_confidence +
                                    1.0 / float(self.history_steps - 1))

    def _predict_batch(self, xb, ub):
        xb = np.asarray(xb, np.float32)
        ub = np.asarray(ub, np.float32)
        if xb.ndim == 1:
            xb = xb[None]
        if ub.ndim == 1:
            ub = ub[None]
        batch = len(xb)
        prefix = self._roll_history[-(self.history_steps - 1):]
        if prefix:
            pad_x = prefix[0][0]
        else:
            pad_x = xb[0]
        prefix = [(x.copy(), u.copy()) for x, u in prefix]
        while len(prefix) < self.history_steps - 1:
            prefix.insert(0, (np.asarray(pad_x, float).copy(), np.zeros(NCH)))
        px = np.asarray([item[0] for item in prefix], np.float32)
        pu = np.asarray([item[1] for item in prefix], np.float32)
        hx = np.concatenate((np.broadcast_to(px, (batch,) + px.shape), xb[:, None]), axis=1)
        hu = np.concatenate((np.broadcast_to(pu, (batch,) + pu.shape), ub[:, None]), axis=1)
        with self.torch.no_grad():
            correction = self.correction(
                self.torch.from_numpy(hx.copy()),
                self.torch.from_numpy(hu.copy())).cpu().numpy().astype(float)
        correction *= float(self._roll_confidence)
        base_next = self.base._predict_batch(xb, ub)
        dq = np.clip(base_next[:, 3:6] + correction[:, :3], -0.30, 0.30)
        pressure = np.clip(base_next[:, 6:] + correction[:, 3:], 0.0, 900.0)
        return np.concatenate((xb[:, :3] + dq, dq, pressure), axis=1)

    def jac(self, x, u):
        x = np.asarray(x, float); u = np.asarray(u, float)
        xb = [x]; ub = [u]
        for j, eps in enumerate(self._eps_x):
            xp, xm = x.copy(), x.copy(); xp[j] += eps; xm[j] -= eps
            xb.extend((xp, xm)); ub.extend((u, u))
        for j, eps in enumerate(self._eps_u):
            up, um = u.copy(), u.copy(); up[j] += eps; um[j] -= eps
            xb.extend((x, x)); ub.extend((up, um))
        prediction = self._predict_batch(np.asarray(xb), np.asarray(ub))
        f0 = prediction[0]
        A = np.column_stack([(prediction[1 + 2*j] - prediction[2 + 2*j]) /
                             (2.0*self._eps_x[j]) for j in range(12)])
        offset = 1 + 2*12
        B = np.column_stack([(prediction[offset + 2*j] - prediction[offset + 2*j + 1]) /
                             (2.0*self._eps_u[j]) for j in range(NCH)])
        if not (np.all(np.isfinite(f0)) and np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            raise FloatingPointError("时序顺模型局部线性化产生非有限数")
        return A, B, f0 - x

    def p_estimate(self, P_meas, u_prev, P_state):
        return np.asarray(P_meas, float).copy()


def _model_file_stamp(architecture="mlp"):
    """Reload the controller when either deployed model artifact changes."""
    paths = [p for p in (PURE_DM_PATH, PURE_DM_META_PATH, DM_PATH) if os.path.exists(p)]
    if architecture in SEQUENCE_MODEL_FILES:
        paths.extend(p for p in SEQUENCE_MODEL_FILES[architecture] if os.path.exists(p))
    return max((os.path.getmtime(p) for p in paths), default=None)

def data_model_ready(plant, architecture="mlp"):
    """/start 用：data_model.json 有就载入；没有就用全部日志拟合一次（再没有就物理先验）。"""
    architecture = str(architecture).strip().lower()
    if architecture in SEQUENCE_MODEL_FILES:
        try:
            dm = SequencePureDataModel(architecture)
            vm = dm.meta.get("validation_metrics", {})
            print(f"# 10拍{architecture}顺模型已载入：{dm.n_fit}样本，"
                  f"运动1拍 {vm.get('moving_one_step_angle_rmse_deg', float('nan')):.2f}°，"
                  f"24拍滚动 {vm.get('rollout_24_angle_rmse_deg', float('nan')):.2f}°；"
                  "时序模型固定使用LTV-MPC")
            return dm
        except Exception as exc:
            print(f"# 10拍{architecture}模型载入失败，回退现有单拍MLP：{exc}")
    if os.path.exists(PURE_DM_PATH) and os.path.exists(PURE_DM_META_PATH):
        try:
            dm = PureDataModel()
            vm = dm.meta.get("validation_metrics", {})
            print(f"# 纯实机数据模型已载入（{PURE_DM_PATH}）：{dm.n_fit} 样本，"
                  f"留出1拍 {vm.get('one_step_angle_rmse_deg', float('nan')):.2f}°，"
                  f"工况滚动 {vm.get('block_rollout_angle_rmse_deg', float('nan')):.2f}°")
            return dm
        except Exception as exc:
            print(f"# 纯数据模型载入失败，回退旧 DataModel：{exc}")
    dm = DataModel(plant)
    if dm.load():
        if hasattr(dm, "converted_from_dt"):
            print(f"# DataModel 已从 {dm.converted_from_dt:g}s/拍转换为 {DT:g}s/拍，作为 20 Hz 初始模型")
        rr = dm.rmse_deg
        print(f"# 数据模型已载入（{DM_PATH}）：{dm.n_fit} 样本拟合，1拍RMSE {rr:.2f}°" if rr is not None
              else f"# 数据模型已载入（{DM_PATH}）：{dm.n_fit} 样本拟合")
        _dm_print(dm)
        return dm
    X, Y, U, Pa, Pb, blocks = _dm_samples_from_logs()
    r = dm.fit(X, Y)
    if r is not None:
        print(f"# 数据模型从 {len(X)} 拍日志拟合：1拍RMSE {r:.2f}°；气路（{len(blocks)} 个连续段）：")
        dm.fit_pneu(blocks, verbose=True); dm.src = "logs"
        dm.save()
    else:
        print("# 没有带实测压的日志——数据模型先用物理先验（各路方向），跑一轮后自动拟合")
    _dm_print(dm)
    return dm

def _dm_print(dm):
    dead = [f"V{c}" for c in range(NCH) if dm.p_dead[c]]
    print(f"#   气路 g(kPa/mV)={np.round(dm.g, 3)}  τ_up={np.round(dm.tu, 2)} τ_dn={np.round(dm.td, 2)}"
          + (f"  ⚠ 压力监测口坏：{','.join(dead)}（改用指令估算）" if dead else ""))
    G = dm.gain_table_deg()
    print("#   稳态增益（每路 +1000mV → Δq°）：" + "  ".join(
        f"V{c}:({G[0,c]:+.0f},{G[1,c]:+.0f},{G[2,c]:+.0f})" for c in range(NCH)))

class DAMPC:
    """数据锚定 MPC：支持 LTV-QP、acados/OSQP SQP-RTI、非线性优化和 MPPI。

    ``nonlinear`` 直接把真正模型展开 N 拍，用 PyTorch autograd 优化电压；
    ``ltv`` 则先滚动名义轨迹，再分别在 ``(xbar[k], ubar[k])`` 求
    ``A_k, B_k``并解QP；``mppi`` 批量采样受约束电压轨迹，不依赖
    ``∂x_hat/∂u``；``sqp`` 用 OSQP 解局部 QP并重复线性化；
    ``acados_sqp_rti`` 使用真正的 acados SQP_RTI + HPIPM 两阶段求解。
    所有方式都只执行第一拍，下一拍用实测状态重算。

    接口与 MPC.step 同：
    step(x_est=[q,Δq,P], r_hor, u_ff_hor=None, lock_states=o/w/L×6) → (u, xn)。
    ``lock_cmd`` 保存同一拍规划出的六路锁动作；未收到固件 L 之前，返回的
    比例阀电压 u 必须为 0，因此合法控制拍始终满足 u == uact。
    """
    def __init__(self, dm, N=20, w_q=(3300.0, 3300.0, 330.0), w_v=(200.0, 200.0, 40.0),
                 rd=2e-4, rho=1e-5, l1=1e-3, x_comp_ratio=0.5,
                 lambda_c=5e-3,
                 lock_lead_ms=100.0,
                 solver="ltv",
                 nl_iters=20, nl_lr_mv=180.0, nl_budget_ms=40.0,
                 mppi_samples=192, mppi_iterations=2, mppi_sigma_mv=700.0,
                 mppi_temperature=1.0, mppi_knots=5,
                 sqp_iterations=2, sqp_max_qp_iterations=100,
                 acados_qp_solver="PARTIAL_CONDENSING_HPIPM",
                 acados_cond_N=5, acados_cmake_generator="",
                 acados_codegen_dir="", acados_source_dir=""):
        self.dm = dm; self.N = N
        self.Wy = np.diag(list(w_q) + list(w_v))
        self.rd, self.rho, self.l1 = rd, rho, max(0.0, float(l1))
        # 实机9+67扫描得到的单向软约束：使用9号时，鼓励67号至少达到
        # 标定比例；67号仍可因负φ需求继续增加。10号当前不参与该约束。
        self.x_comp_ratio = float(np.clip(x_comp_ratio, 0.0, 2.0))
        self.lambda_c = max(0.0, float(lambda_c))
        # 固件在比例阀真正输出前先驱动开关锁阀。MPC 内部把这段时间建模为
        # uact=0，而不是错误地认为请求电压 u 已经作用到气路。
        self.lock_lead_steps = max(1, int(math.ceil(
            max(0.0, float(lock_lead_ms))/1000.0/DT)))
        self.lock_states = ["L"] * NCH
        self.lock_delay_steps = np.zeros(NCH, dtype=int)
        self.lock_cmd = np.ones(NCH, dtype=int)
        self.solver = str(solver).strip().lower()
        if self.solver in ("acados", "sqp_rti", "acados-sqp-rti"):
            self.solver = "acados_sqp_rti"
        self.solver_requested = self.solver
        if self.solver not in ("ltv", "sqp", "acados_sqp_rti", "nonlinear", "mppi"):
            self.solver = "ltv"
        # 非线性、SQP和MPPI都需要 PureDataModel/TorchScript 真正模型。
        if self.solver in ("nonlinear", "sqp", "mppi") and not (
                hasattr(dm, "torch") and hasattr(dm, "net")):
            print(f"# {self.solver}需要PyTorch真正模型，本次自动回退LTV逐拍线性化")
            self.solver = "ltv"
        self.nl_iters = max(1, int(nl_iters))
        self.nl_lr_mv = max(1.0, float(nl_lr_mv))
        self.nl_budget_s = max(0.01, float(nl_budget_ms)/1000.0)
        self.mppi_samples = int(np.clip(int(mppi_samples), 32, 2048))
        self.mppi_iterations = int(np.clip(int(mppi_iterations), 1, 10))
        self.mppi_sigma_mv = float(np.clip(float(mppi_sigma_mv), 10.0, HARD_CLAMP_MV))
        self.mppi_temperature = float(np.clip(float(mppi_temperature), 0.05, 20.0))
        self.mppi_knots = int(np.clip(int(mppi_knots), 2, self.N))
        self.sqp_iterations = int(np.clip(int(sqp_iterations), 1, 5))
        self.sqp_max_qp_iterations = int(np.clip(int(sqp_max_qp_iterations), 20, 1000))
        self.acados_backend = None
        if self.solver == "acados_sqp_rti":
            try:
                try:
                    from acados_sqp_rti import AcadosSQPRTIBackend
                except ImportError:
                    from .acados_sqp_rti import AcadosSQPRTIBackend
                self.acados_backend = AcadosSQPRTIBackend(
                    self.N, DT, CLAMP_MV,
                    qp_solver=acados_qp_solver,
                    cond_N=acados_cond_N,
                    qp_iter_max=self.sqp_max_qp_iterations,
                    codegen_dir=(str(acados_codegen_dir).strip() or None),
                    cmake_generator=acados_cmake_generator,
                    source_dir=acados_source_dir)
            except Exception as exc:
                print(f"# acados SQP-RTI不可用，本次自动回退LTV：{exc}")
                self.solver = "ltv"
        self._mppi_rng = (dm.torch.Generator(device="cpu").manual_seed(20260907)
                          if hasattr(dm, "torch") else None)
        self.u_plan = None
        self.last_solve_ms = 0.0
        self.last_nl_iters = 0
        self._nl_warned = False
        m = NCH
        self._Qb = np.kron(np.eye(N), self.Wy)
        Dm = np.eye(m*N)
        for k in range(1, N): Dm[m*k:m*k+m, m*(k-1):m*k] = -np.eye(m)
        self._D = Dm
        self.u_prev = np.zeros(NCH); self.q_pred = None; self.ok = True
        self.d = np.zeros(3)
        # 把TorchScript反向传播和Adam的首次初始化延迟放在“开始播放”请求里，
        # 避免真正的第一个50ms控制拍被一次性冷启动拖慢。
        if self.solver in ("nonlinear", "sqp", "mppi", "acados_sqp_rti"):
            try:
                if self.solver == "nonlinear":
                    self._step_nonlinear(np.zeros(12), [np.zeros(3)]*self.N)
                elif self.solver == "sqp":
                    self._step_sqp(np.zeros(12), [np.zeros(3)]*self.N)
                elif self.solver == "acados_sqp_rti":
                    self._step_acados_sqp_rti(np.zeros(12), [np.zeros(3)]*self.N)
                else:
                    self._step_mppi(np.zeros(12), [np.zeros(3)]*self.N)
            except Exception as exc:
                print(f"# {self.solver.upper()}-MPC预热失败，本次自动回退LTV：{exc}")
                self.solver = "ltv"
            self.reset_runtime()
        else:
            # TorchScript、有限差分和线性代数第一次调用会有一次性开销；
            # 在播放开始前完成，避免首个50ms实机控制拍超时。
            try:
                p0 = np.asarray(getattr(dm, "meta", {}).get(
                    "rest_pressure_kpa", [0.0]*NCH), float)
                self._step_ltv(np.r_[np.zeros(6), p0], [np.zeros(3)]*self.N)
            except Exception as exc:
                print(f"# LTV-MPC预热提示：{exc}")
            self.reset_runtime()

    def reset_runtime(self):
        """轮间放气后清空控制器瞬态和非线性优化热启动序列。"""
        self.u_prev[:] = 0.0
        self.q_pred = None
        self.u_plan = None
        self.last_solve_ms = 0.0
        self.last_nl_iters = 0
        self.ok = True
        self.lock_states = ["L"] * NCH
        self.lock_delay_steps[:] = 0
        self.lock_cmd[:] = 0
        if hasattr(self.dm, "reset_history"):
            self.dm.reset_history()

    def set_lock_states(self, states):
        """把固件 o/w/L 状态换成预测域内每路 uact 的生效延迟。"""
        if states is None or len(states) != NCH:
            clean = ["o"] * NCH
        else:
            clean = [str(v).strip() if str(v).strip() in ("o", "w", "L") else "o"
                     for v in states]
        self.lock_states = clean
        # 求解失败时的保守后备值：已经在锁合过程中的通道保持锁请求；成功
        # 求解后会被 _lock_action_now() 用本次预测序列覆盖。
        self.lock_cmd = np.asarray([int(v in ("w", "L")) for v in clean], dtype=int)
        for ch, state in enumerate(clean):
            if state == "L":
                delay = 0
            elif state == "w":
                # 收到 w 时预锁通常已经走过至少一个 50ms 控制拍。
                delay = max(1, self.lock_lead_steps - 1)
            else:
                delay = self.lock_lead_steps
            self.lock_delay_steps[ch] = min(delay, self.N - 1)

    def _lock_availability(self):
        """[N,6]：1 表示该预测拍比例阀电压可以真实作用。"""
        return (np.arange(self.N, dtype=int)[:, None] >=
                self.lock_delay_steps[None, :]).astype(float)

    def _lock_shifted_commands(self, desired_plan):
        """预测域内真正允许施加的比例阀电压；预锁拍严格为零。"""
        desired = np.asarray(desired_plan, float).reshape(self.N, NCH)
        return desired*self._lock_availability()

    def _lock_transform(self):
        """线性映射 T：计划电压 -> 受当前锁状态约束的实际比例阀电压。"""
        return np.diag(self._lock_availability().ravel())

    def _lock_shifted_commands_torch(self, desired_plan):
        """Torch 版实际电压序列，支持 [N,6] 和 [batch,N,6]。"""
        availability = desired_plan.new_tensor(self._lock_availability())
        return desired_plan*availability

    def _lock_command_now(self, desired_plan):
        # 比例阀命令只在固件已经回报 L 时下发。o/w 两种状态只发送锁动作，
        # 从协议层消除“控制器请求1500、实际却因预锁执行0”的歧义。
        command = self._lock_shifted_commands(desired_plan)[0]
        command = _antagonist_project(np.clip(command, CO_CONTRACT_MV, CLAMP_MV))
        for ch in range(NCH):
            if ch not in ACTIVE_CH:
                command[ch] = 0.0
        return command

    def _lock_action_now(self, desired_plan):
        """从最优电压序列提取本拍六路锁动作（1=保持/开始锁，0=释放）。

        OPEN/LOCKING 通道查看其锁提前量之后的计划电压；LOCKED 通道查看
        当前拍。这样滚动时序为 ``锁,锁,充气``，而不是提前把充气电压冒充
        锁触发信号。若一对拮抗通道同时请求，优先保留已经锁住的一侧，
        否则保留未来计划电压较大的一侧。
        """
        plan = np.asarray(desired_plan, float).reshape(self.N, NCH)
        cmd = np.zeros(NCH, dtype=int)
        strength = np.zeros(NCH, dtype=float)
        for ch, state in enumerate(self.lock_states):
            if ch not in ACTIVE_CH:
                continue
            look = 0 if state == "L" else min(int(self.lock_delay_steps[ch]), self.N - 1)
            strength[ch] = max(0.0, float(plan[look, ch]))
            cmd[ch] = int(strength[ch] > 0.5)
        for left, right in ANTAGONIST_PAIRS:
            if not (cmd[left] and cmd[right]):
                continue
            left_active = self.lock_states[left] in ("w", "L")
            right_active = self.lock_states[right] in ("w", "L")
            if left_active != right_active:
                cmd[right if left_active else left] = 0
            elif strength[left] >= strength[right]:
                cmd[right] = 0
            else:
                cmd[left] = 0
        return cmd

    def _lock_effective_now(self, command):
        """当前这拍真正进入比例阀/真正模型的 uact。"""
        return np.asarray(command, float)

    def step(self, x_est, r_hor, u_ff_hor=None, lock_states=None):
        """按配置求解，非线性/MPPI失败时安全回退到逐拍 LTV-QP。"""
        if lock_states is not None:
            self.set_lock_states(lock_states)
        if self.solver == "mppi":
            try:
                answer = self._step_mppi(x_est, r_hor, u_ff_hor)
                self._nl_warned = False
                return answer
            except Exception as exc:
                if not self._nl_warned:
                    print(f"  !! MPPI失败，本拍回退LTV：{exc}")
                    self._nl_warned = True
        if self.solver == "acados_sqp_rti":
            try:
                answer = self._step_acados_sqp_rti(x_est, r_hor, u_ff_hor)
                self._nl_warned = False
                return answer
            except Exception as exc:
                if not self._nl_warned:
                    print(f"  !! acados SQP-RTI失败，本拍回退LTV：{exc}")
                    self._nl_warned = True
        if self.solver == "sqp":
            try:
                answer = self._step_sqp(x_est, r_hor, u_ff_hor)
                self._nl_warned = False
                return answer
            except Exception as exc:
                if not self._nl_warned:
                    print(f"  !! SQP-RTI失败，本拍回退LTV：{exc}")
                    self._nl_warned = True
        if self.solver == "nonlinear":
            try:
                answer = self._step_nonlinear(x_est, r_hor, u_ff_hor)
                self._nl_warned = False
                return answer
            except Exception as exc:
                if not self._nl_warned:
                    print(f"  !! 非线性MPC失败，本拍回退LTV：{exc}")
                    self._nl_warned = True
        return self._step_ltv(x_est, r_hor, u_ff_hor)

    def _step_nonlinear(self, x_est, r_hor, u_ff_hor=None):
        """直接展开真正模型 N 拍，用 autograd + Adam 优化六路电压序列。"""
        torch = self.dm.torch
        started = time.perf_counter()
        x0 = np.nan_to_num(np.asarray(x_est, float), nan=0.0, posinf=0.0, neginf=0.0)
        x0[3:6] = np.clip(x0[3:6], -0.3, 0.3)
        x0[6:] = np.clip(x0[6:], 0.0, P_MAX_KPA)
        refs = np.asarray(r_hor, float).reshape(self.N, 3)

        if u_ff_hor is not None:
            initial = np.clip(np.asarray(u_ff_hor, float).reshape(self.N, NCH),
                              CO_CONTRACT_MV, CLAMP_MV)
        elif self.u_plan is not None and np.shape(self.u_plan) == (self.N, NCH):
            initial = np.vstack([self.u_plan[1:], self.u_plan[-1:]])
        else:
            initial = np.tile(self.u_prev, (self.N, 1))
        initial = _antagonist_project(initial)
        for ch in range(NCH):
            if ch not in ACTIVE_CH: initial[:, ch] = 0.0

        dtype = torch.float32
        U = torch.tensor(initial, dtype=dtype, requires_grad=True)
        x_start = torch.tensor(x0[None], dtype=dtype)
        ref_t = torch.tensor(refs, dtype=dtype)
        wq_t = torch.tensor(np.diag(self.Wy)[:3], dtype=dtype)
        wv_t = torch.tensor(np.diag(self.Wy)[3:6], dtype=dtype)
        prev_t = torch.tensor(self.u_prev, dtype=dtype)
        availability_t = torch.tensor(self._lock_availability(), dtype=dtype)
        center_t = torch.tensor(initial if u_ff_hor is not None else
                                np.full((self.N, NCH), CO_CONTRACT_MV), dtype=dtype)
        optimizer = torch.optim.Adam([U], lr=self.nl_lr_mv)
        completed = 0

        for iteration in range(self.nl_iters):
            optimizer.zero_grad()
            u_seq = torch.clamp(U, CO_CONTRACT_MV, CLAMP_MV)
            u_effective = u_seq*availability_t
            u_command = self._lock_shifted_commands_torch(u_seq)
            state = x_start
            tracking = torch.zeros((), dtype=dtype)
            for k in range(self.N):
                state = self.dm.net(state, u_effective[k:k+1])
                # 只限制速度/压力进入模型的范围，角度保持真正模型原始输出。
                state = torch.cat((state[:, :3],
                                   torch.clamp(state[:, 3:6], -0.3, 0.3),
                                   torch.clamp(state[:, 6:], 0.0, P_MAX_KPA)), dim=1)
                eq = state[0, :3] - ref_t[k]
                ev = state[0, 3:6] / DT
                tracking = tracking + torch.sum(wq_t*eq*eq) + torch.sum(wv_t*ev*ev)
            du = torch.cat(((u_command[0]-prev_t)[None],
                            u_command[1:]-u_command[:-1]), dim=0)
            loss = tracking + self.rd*torch.sum(du*du)
            if self.lambda_c:
                ratio = self.x_comp_ratio
                shortfall = torch.relu(ratio*u_effective[:, 4] - u_effective[:, 3])
                loss = loss + 0.5*self.lambda_c*torch.sum(shortfall*shortfall)
            if self.rho:
                loss = loss + self.rho*torch.sum(
                    (u_effective-availability_t*center_t)**2)
            if self.l1:
                loss = loss + self.l1*torch.sum(u_effective)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("非线性目标函数出现NaN/Inf")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                U.clamp_(CO_CONTRACT_MV, CLAMP_MV)
                for left, right in ANTAGONIST_PAIRS:
                    both = (U[:, left] > 0.0) & (U[:, right] > 0.0)
                    keep_left = U[:, left] >= U[:, right]
                    U[both & keep_left, right] = 0.0
                    U[both & ~keep_left, left] = 0.0
                for ch in range(NCH):
                    if ch not in ACTIVE_CH: U[:, ch] = 0.0
            completed = iteration + 1
            if time.perf_counter()-started >= self.nl_budget_s:
                break

        plan = U.detach().cpu().numpy().astype(float)
        if not np.all(np.isfinite(plan)):
            raise FloatingPointError("非线性MPC输出出现NaN/Inf")
        plan = _antagonist_project(np.clip(plan, CO_CONTRACT_MV, CLAMP_MV))
        self.lock_cmd = self._lock_action_now(plan)
        u = self._lock_command_now(plan)
        u_effective_now = self._lock_effective_now(u)
        with torch.no_grad():
            xn = self.dm.net(
                x_start, torch.tensor(u_effective_now[None], dtype=dtype)).cpu().numpy()[0]
        xn = np.nan_to_num(xn.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        xn[3:6] = np.clip(xn[3:6], -0.3, 0.3)
        xn[6:] = np.clip(xn[6:], 0.0, P_MAX_KPA)
        self.u_plan = plan
        self.last_nl_iters = completed
        self.last_solve_ms = 1000.0*(time.perf_counter()-started)
        self.q_pred = xn[:3].copy(); self.u_prev = u; self.ok = True
        return u, xn

    def _mppi_project_torch(self, plan):
        """Project batched ``[..., N, 6]`` plans onto the hardware constraints."""
        torch = self.dm.torch
        out = torch.clamp(plan, CO_CONTRACT_MV, CLAMP_MV).clone()
        for left, right in ANTAGONIST_PAIRS:
            lv = out[..., left].clone()
            rv = out[..., right].clone()
            keep_left = lv >= rv
            out[..., left] = torch.where(keep_left, lv, torch.zeros_like(lv))
            out[..., right] = torch.where(keep_left, torch.zeros_like(rv), rv)
        for ch in range(NCH):
            if ch not in ACTIVE_CH:
                out[..., ch] = 0.0
        return out

    def _step_mppi(self, x_est, r_hor, u_ff_hor=None):
        """Batched, gradient-free MPPI rollout on the true neural model."""
        torch = self.dm.torch
        started = time.perf_counter()
        x0 = np.nan_to_num(np.asarray(x_est, float), nan=0.0, posinf=0.0, neginf=0.0)
        x0[3:6] = np.clip(x0[3:6], -0.3, 0.3)
        x0[6:] = np.clip(x0[6:], 0.0, P_MAX_KPA)
        refs = np.asarray(r_hor, float).reshape(self.N, 3)

        if u_ff_hor is not None:
            initial = np.clip(np.asarray(u_ff_hor, float).reshape(self.N, NCH),
                              CO_CONTRACT_MV, CLAMP_MV)
        elif self.u_plan is not None and np.shape(self.u_plan) == (self.N, NCH):
            initial = np.vstack([self.u_plan[1:], self.u_plan[-1:]])
        else:
            initial = np.tile(self.u_prev, (self.N, 1))
        initial = _antagonist_project(initial)
        for ch in range(NCH):
            if ch not in ACTIVE_CH:
                initial[:, ch] = 0.0

        dtype = torch.float32
        base = torch.as_tensor(initial, dtype=dtype).clone()
        center = base.clone()
        x_start = torch.as_tensor(x0[None], dtype=dtype)
        ref_t = torch.as_tensor(refs, dtype=dtype)
        wq = torch.as_tensor(np.diag(self.Wy)[:3].copy(), dtype=dtype)
        wv = torch.as_tensor(np.diag(self.Wy)[3:6].copy(), dtype=dtype)
        prev = torch.as_tensor(self.u_prev, dtype=dtype)
        availability = torch.as_tensor(self._lock_availability(), dtype=dtype)
        completed = 0

        with torch.no_grad():
            for iteration in range(self.mppi_iterations):
                samples = self.mppi_samples
                half = max(1, (samples-1)//2)
                sigma = self.mppi_sigma_mv*(0.65**iteration)
                eps = torch.randn((half, self.mppi_knots, NCH), dtype=dtype,
                                  generator=self._mppi_rng)*sigma
                knots = torch.cat((torch.zeros((1, self.mppi_knots, NCH), dtype=dtype),
                                   eps, -eps), dim=0)
                if len(knots) < samples:
                    extra = torch.randn((samples-len(knots), self.mppi_knots, NCH),
                                        dtype=dtype, generator=self._mppi_rng)*sigma
                    knots = torch.cat((knots, extra), dim=0)
                knots = knots[:samples]
                # Five low-rate control knots are interpolated to the 20 model
                # ticks.  This greatly reduces random voltage chatter.
                noise = torch.nn.functional.interpolate(
                    knots.permute(0, 2, 1), size=self.N,
                    mode="linear", align_corners=True).permute(0, 2, 1)
                candidates = self._mppi_project_torch(base[None] + noise)
                candidates[0] = base                 # Always retain the nominal plan.
                effective = candidates*availability[None]
                commands = self._lock_shifted_commands_torch(candidates)

                state = x_start.expand(samples, -1).clone()
                cost = torch.zeros(samples, dtype=dtype)
                for k in range(self.N):
                    state = self.dm.net(state, effective[:, k])
                    state = torch.cat((state[:, :3],
                                       torch.clamp(state[:, 3:6], -0.3, 0.3),
                                       torch.clamp(state[:, 6:], 0.0, P_MAX_KPA)), dim=1)
                    eq = state[:, :3] - ref_t[k]
                    ev = state[:, 3:6]/DT
                    cost += torch.sum(wq*eq*eq, dim=1) + torch.sum(wv*ev*ev, dim=1)

                du = torch.cat((commands[:, :1] - prev[None, None],
                                commands[:, 1:] - commands[:, :-1]), dim=1)
                cost += self.rd*torch.sum(du*du, dim=(1, 2))
                if self.rho:
                    cost += self.rho*torch.sum(
                        (effective-availability[None]*center[None])**2, dim=(1, 2))
                if self.l1:
                    cost += self.l1*torch.sum(effective, dim=(1, 2))
                if self.lambda_c:
                    shortfall = torch.relu(
                        self.x_comp_ratio*effective[:, :, 4] - effective[:, :, 3])
                    cost += 0.5*self.lambda_c*torch.sum(shortfall*shortfall, dim=1)

                finite = torch.isfinite(cost)
                if not bool(torch.any(finite)):
                    raise FloatingPointError("MPPI全部候选轨迹出现NaN/Inf")
                finite_cost = cost[finite]
                best = torch.min(finite_cost)
                quartile = torch.quantile(finite_cost, 0.25)
                scale = torch.clamp(quartile-best, min=1.0)
                logits = torch.clamp(
                    -(cost-best)/(self.mppi_temperature*scale), min=-60.0, max=0.0)
                logits = torch.where(finite, logits,
                                     torch.full_like(logits, -60.0))
                weights = torch.softmax(logits, dim=0)
                base = torch.sum(weights[:, None, None]*candidates, dim=0)
                base = self._mppi_project_torch(base)
                completed = iteration + 1
                if time.perf_counter()-started >= self.nl_budget_s:
                    break

            plan = base.cpu().numpy().astype(float)
            if not np.all(np.isfinite(plan)):
                raise FloatingPointError("MPPI输出出现NaN/Inf")
            self.lock_cmd = self._lock_action_now(plan)
            u = self._lock_command_now(plan)
            u_effective_now = self._lock_effective_now(u)
            xn = self.dm.net(
                x_start, torch.as_tensor(u_effective_now[None], dtype=dtype)).cpu().numpy()[0]

        xn = np.nan_to_num(xn.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        xn[3:6] = np.clip(xn[3:6], -0.3, 0.3)
        xn[6:] = np.clip(xn[6:], 0.0, P_MAX_KPA)
        self.u_plan = plan
        self.last_nl_iters = completed
        self.last_solve_ms = 1000.0*(time.perf_counter()-started)
        self.q_pred = xn[:3].copy(); self.u_prev = u; self.ok = True
        return np.clip(u, 0.0, CLAMP_MV), xn

    def _step_acados_sqp_rti(self, x_est, r_hor, u_ff_hor=None):
        """逐拍真正模型线性化，由 acados SQP_RTI + HPIPM 解受约束 OCP。"""
        if self.acados_backend is None:
            raise RuntimeError("acados后端尚未初始化")

        started = time.perf_counter()
        x0 = np.nan_to_num(np.asarray(x_est, float), nan=0.0,
                           posinf=0.0, neginf=0.0)
        x0[3:6] = np.clip(x0[3:6], -0.3, 0.3)
        x0[6:] = np.clip(x0[6:], 0.0, P_MAX_KPA)
        refs = np.asarray(r_hor, float).reshape(self.N, 3)
        N, m = self.N, NCH
        if u_ff_hor is not None:
            nominal = np.clip(np.asarray(u_ff_hor, float).reshape(N, m),
                              CO_CONTRACT_MV, CLAMP_MV)
            rho_center = nominal.copy()
        elif self.u_plan is not None and np.shape(self.u_plan) == (N, m):
            nominal = np.vstack([self.u_plan[1:], self.u_plan[-1:]])
            rho_center = np.full((N, m), CO_CONTRACT_MV)
        else:
            nominal = np.tile(self.u_prev, (N, 1))
            rho_center = np.full((N, m), CO_CONTRACT_MV)
        nominal = _antagonist_project(nominal)
        for ch in range(NCH):
            if ch not in ACTIVE_CH:
                nominal[:, ch] = 0.0

        availability = self._lock_availability()
        A_seq = np.empty((N, 12, 12))
        B_seq = np.empty((N, 12, m))
        c_seq = np.empty((N, 12))
        state_nodes = np.empty((N+1, 12))
        state_nodes[0] = x0
        state = x0.copy()
        try:
            if hasattr(self.dm, "begin_rollout"):
                self.dm.begin_rollout(state)
            for k in range(N):
                actual_nominal = availability[k]*nominal[k]
                A_k, B_k, w_k = self.dm.jac(state, actual_nominal)
                state_next = np.asarray(state + w_k, float)
                if not (np.all(np.isfinite(A_k)) and np.all(np.isfinite(B_k)) and
                        np.all(np.isfinite(state_next))):
                    raise FloatingPointError("线性化产生NaN/Inf")
                state_next[3:6] = np.clip(state_next[3:6], -0.3, 0.3)
                state_next[6:] = np.clip(state_next[6:], 0.0, P_MAX_KPA)
                A_seq[k] = A_k
                B_seq[k] = B_k
                # acados dynamics uses B*(availability*u), so the affine term
                # must be formed around the genuinely applied nominal voltage.
                c_seq[k] = state_next - A_k @ state - B_k @ actual_nominal
                state_nodes[k+1] = state_next
                if hasattr(self.dm, "advance_rollout"):
                    self.dm.advance_rollout(state, actual_nominal)
                state = state_next
        except Exception as exc:
            raise RuntimeError(f"acados逐拍真正模型线性化失败：{exc}") from exc

        lower = np.full((N, m), CO_CONTRACT_MV)
        upper = np.full((N, m), CLAMP_MV)
        C = np.zeros((6, 12))
        C[:3, :3] = np.eye(3)
        C[3:, 3:6] = np.eye(3)/DT
        output_weights = np.diag(self.Wy)

        # “一对只能充一侧”是离散析取约束。和现有OSQP-SQP一样，RTI按
        # 名义计划（全零时按局部下降梯度）选择一侧，并把另一侧上界钉为0。
        for k in range(N):
            residual = C @ state_nodes[k+1] - np.r_[refs[k], np.zeros(3)]
            B_effective = B_seq[k]*availability[k][None, :]
            local_gradient = (B_effective.T @ C.T @ (output_weights*residual)
                              + self.l1*availability[k])
            for left, right in ANTAGONIST_PAIRS:
                if max(nominal[k, left], nominal[k, right]) > 1e-6:
                    keep_left = nominal[k, left] >= nominal[k, right]
                else:
                    keep_left = local_gradient[left] <= local_gradient[right]
                upper[k, right if keep_left else left] = 0.0
            for ch in range(NCH):
                if ch not in ACTIVE_CH:
                    upper[k, ch] = 0.0

        plan = self.acados_backend.solve(
            x0=x0, A=A_seq, B=B_seq, c=c_seq, refs=refs,
            centers=rho_center, availability=availability,
            wq=np.diag(self.Wy)[:3], wv=np.diag(self.Wy)[3:6],
            rd=self.rd, rho=self.rho, l1=self.l1,
            x_comp_ratio=self.x_comp_ratio, lambda_c=self.lambda_c,
            lower=lower, upper=upper, nominal_states=state_nodes,
            nominal_controls=nominal, previous_control=self.u_prev)
        plan = _antagonist_project(np.clip(plan, CO_CONTRACT_MV, CLAMP_MV))
        for ch in range(NCH):
            if ch not in ACTIVE_CH:
                plan[:, ch] = 0.0

        self.lock_cmd = self._lock_action_now(plan)
        u = self._lock_command_now(plan)
        u_effective_now = self._lock_effective_now(u)
        try:
            if hasattr(self.dm, "begin_rollout"):
                self.dm.begin_rollout(x0)
            _A, _B, first_w = self.dm.jac(x0, u_effective_now)
            xn = np.asarray(x0 + first_w, float)
        except Exception:
            # xn is diagnostic; hardware replaces it with the next measured state.
            xn = state_nodes[1].copy()
        xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
        xn[3:6] = np.clip(xn[3:6], -0.3, 0.3)
        xn[6:] = np.clip(xn[6:], 0.0, P_MAX_KPA)
        self.u_plan = plan.copy()
        self.last_nl_iters = 1
        self.last_solve_ms = 1000.0*(time.perf_counter()-started)
        self.q_pred = xn[:3].copy()
        self.u_prev = u
        self.ok = True
        return np.clip(u, CO_CONTRACT_MV, CLAMP_MV), xn

    def _step_sqp(self, x_est, r_hor, u_ff_hor=None):
        """Sequential linearization with an OSQP solve at each RTI iteration."""
        try:
            import osqp
            from scipy import sparse
        except ImportError as exc:
            raise RuntimeError("SQP-RTI需要osqp和scipy") from exc

        started = time.perf_counter()
        x0 = np.nan_to_num(np.asarray(x_est, float), nan=0.0, posinf=0.0, neginf=0.0)
        x0[3:6] = np.clip(x0[3:6], -0.3, 0.3)
        x0[6:] = np.clip(x0[6:], 0.0, P_MAX_KPA)
        refs = np.asarray(r_hor, float).reshape(self.N, 3)
        N, m = self.N, NCH
        if u_ff_hor is not None:
            nominal = np.clip(np.asarray(u_ff_hor, float).reshape(N, m),
                              CO_CONTRACT_MV, CLAMP_MV)
            rho_center = nominal.copy()
        elif self.u_plan is not None and np.shape(self.u_plan) == (N, m):
            nominal = np.vstack([self.u_plan[1:], self.u_plan[-1:]])
            rho_center = np.full((N, m), CO_CONTRACT_MV)
        else:
            nominal = np.tile(self.u_prev, (N, 1))
            rho_center = np.full((N, m), CO_CONTRACT_MV)
        nominal = _antagonist_project(nominal)
        for ch in range(NCH):
            if ch not in ACTIVE_CH:
                nominal[:, ch] = 0.0
        availability = self._lock_availability()
        availability_flat = availability.ravel()
        command_T = self._lock_transform()

        C = np.zeros((6, 12))
        C[:3, :3] = np.eye(3)
        C[3:, 3:6] = np.eye(3)/DT
        Yref = np.concatenate([np.r_[r, np.zeros(3)] for r in refs])
        DtD = self._D.T @ self._D
        completed = 0

        for _iteration in range(self.sqp_iterations):
            Su = np.zeros((6*N, m*N))
            Ynom = np.zeros(6*N)
            Sx = np.zeros((12, m*N))
            state = x0.copy()
            try:
                for k in range(N):
                    effective_nominal = availability[k]*nominal[k]
                    A_k, B_k, w_k = self.dm.jac(state, effective_nominal)
                    B_k = B_k*availability[k][None, :]
                    state_next = np.asarray(state + w_k, float)
                    if not (np.all(np.isfinite(A_k)) and np.all(np.isfinite(B_k)) and
                            np.all(np.isfinite(state_next))):
                        raise FloatingPointError("SQP线性化产生非有限数")
                    state_next[3:6] = np.clip(state_next[3:6], -0.3, 0.3)
                    state_next[6:] = np.clip(state_next[6:], 0.0, P_MAX_KPA)
                    Sx = A_k @ Sx
                    Sx[:, m*k:m*(k+1)] += B_k
                    Su[6*k:6*k+6] = C @ Sx
                    Ynom[6*k:6*k+6] = C @ state_next
                    state = state_next
            except Exception as exc:
                raise RuntimeError(f"SQP逐拍线性化失败：{exc}") from exc

            command_nominal = self._lock_shifted_commands(nominal)
            dub = (command_nominal-self.u_prev).ravel()
            residual = Ynom-Yref
            command_DtD = command_T.T @ DtD @ command_T
            H = (Su.T @ self._Qb @ Su + self.rd*command_DtD +
                 self.rho*np.diag(availability_flat))
            g = (Su.T @ self._Qb @ residual +
                 self.rd*(command_T.T @ (DtD @ dub)))
            g += self.rho*availability_flat*(nominal-rho_center).ravel()
            g += self.l1*availability_flat

            # Sequentially activate the one-sided 9→67 compensation penalty.
            # A violation found after the first solve is therefore corrected by
            # the next SQP relinearization instead of being hidden by projection.
            if self.lambda_c:
                for k in range(N):
                    row = np.zeros(m*N)
                    row[m*k+4] = self.x_comp_ratio*availability[k, 4]
                    row[m*k+3] = -availability[k, 3]
                    violation = float(row @ nominal.ravel())
                    if violation > 0.0:
                        H += self.lambda_c*np.outer(row, row)
                        g += self.lambda_c*violation*row

            H = 0.5*(H+H.T) + 1e-9*np.eye(m*N)
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
                raise FloatingPointError("SQP目标矩阵出现NaN/Inf")
            lo = (CO_CONTRACT_MV-nominal).ravel()
            hi = (CLAMP_MV-nominal).ravel()

            # Antagonistic exclusivity is a disjunction, not a convex QP
            # constraint.  RTI chooses one side from the nominal plan or local
            # descent direction, then fixes the opposite side to exactly zero.
            for k in range(N):
                for left, right in ANTAGONIST_PAIRS:
                    li, ri = m*k+left, m*k+right
                    if max(nominal[k, left], nominal[k, right]) > 1e-6:
                        keep_left = nominal[k, left] >= nominal[k, right]
                    else:
                        keep_left = g[li] <= g[ri]
                    disabled = ri if keep_left else li
                    lo[disabled] = hi[disabled] = -nominal.ravel()[disabled]
                for ch in range(NCH):
                    if ch not in ACTIVE_CH:
                        idx = m*k+ch
                        lo[idx] = hi[idx] = -nominal[k, ch]

            remaining = max(0.002, self.nl_budget_s-(time.perf_counter()-started))
            problem = osqp.OSQP()
            problem.setup(P=sparse.csc_matrix(np.triu(H)), q=g,
                          A=sparse.eye(m*N, format="csc"), l=lo, u=hi,
                          verbose=False, warm_start=True, polish=False,
                          eps_abs=0.1, eps_rel=1e-3,
                          max_iter=self.sqp_max_qp_iterations,
                          time_limit=remaining)
            problem.warm_start(x=np.zeros(m*N))
            result = problem.solve()
            delta = None if result.x is None else np.asarray(result.x, float)
            status = int(getattr(result.info, "status_val", -1))
            if (delta is None or delta.shape != (m*N,) or
                    not np.all(np.isfinite(delta)) or status not in (1, 2)):
                raise RuntimeError(f"OSQP未收敛：{getattr(result.info, 'status', status)}")
            nominal = _antagonist_project(np.clip(
                nominal + delta.reshape(N, m), CO_CONTRACT_MV, CLAMP_MV))
            for ch in range(NCH):
                if ch not in ACTIVE_CH:
                    nominal[:, ch] = 0.0
            completed += 1
            if time.perf_counter()-started >= self.nl_budget_s:
                break

        if completed == 0:
            raise RuntimeError("SQP没有完成任何一次QP更新")
        plan = nominal
        self.lock_cmd = self._lock_action_now(plan)
        u = self._lock_command_now(plan)
        u_effective_now = self._lock_effective_now(u)
        with self.dm.torch.no_grad():
            xt = self.dm.torch.as_tensor(x0[None], dtype=self.dm.torch.float32)
            ut = self.dm.torch.as_tensor(u_effective_now[None], dtype=self.dm.torch.float32)
            xn = self.dm.net(xt, ut).cpu().numpy()[0]
        xn = np.nan_to_num(xn.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        xn[3:6] = np.clip(xn[3:6], -0.3, 0.3)
        xn[6:] = np.clip(xn[6:], 0.0, P_MAX_KPA)
        self.u_plan = plan.copy()
        self.last_nl_iters = completed
        self.last_solve_ms = 1000.0*(time.perf_counter()-started)
        self.q_pred = xn[:3].copy(); self.u_prev = u; self.ok = True
        return np.clip(u, 0.0, CLAMP_MV), xn

    def _step_ltv(self, x_est, r_hor, u_ff_hor=None):
        x_est = np.nan_to_num(np.asarray(x_est, float), nan=0.0, posinf=0.0, neginf=0.0)
        x_est[3:6] = np.clip(x_est[3:6], -0.3, 0.3)
        x_est[6:] = np.clip(x_est[6:], 0.0, P_MAX_KPA)      # 传感器量程，不是 CLAMP_MV*0.09
        N, m = self.N, NCH
        C = np.zeros((6, 12)); C[:3, :3] = np.eye(3); C[3:, 3:6] = np.eye(3)/DT   # 输出 [q; Δq/DT]
        if u_ff_hor is None: ubar = np.tile(self.u_prev, (N, 1))
        else: ubar = np.clip(np.asarray(u_ff_hor, float).reshape(N, m), CO_CONTRACT_MV, CLAMP_MV)
        # 名义轨迹本身也必须满足实机的拮抗肌硬约束，否则 QP 会围绕一条不可能
        # 真正执行的轨迹做线性化。
        ubar = _antagonist_project(ubar)
        availability = self._lock_availability()
        availability_flat = availability.ravel()
        command_T = self._lock_transform()

        # xbar[k+1] = f(xbar[k], ubar[k])。Sx 保存 xbar[k] 对整段 δU 的
        # 灵敏度；每走一步就用本步 A_k 传播，并把本步 B_k 写入对应输入块。
        Su = np.zeros((6*N, m*N))
        Ynom = np.zeros(6*N)
        Sx = np.zeros((12, m*N))
        xbar = x_est.copy()
        first_B = None
        first_next = None
        try:
            if hasattr(self.dm, "begin_rollout"):
                self.dm.begin_rollout(xbar)
            for k in range(N):
                effective_ubar = availability[k]*ubar[k]
                A_k, B_k, w_k = self.dm.jac(xbar, effective_ubar)
                B_k = B_k*availability[k][None, :]
                xnext = np.asarray(xbar + w_k, float)
                if not (np.all(np.isfinite(A_k)) and np.all(np.isfinite(B_k)) and
                        np.all(np.isfinite(xnext))):
                    raise FloatingPointError("逐拍线性化产生非有限数")
                # 防止模型轻微越过物理边界后，后续线性化离开训练数据范围。
                xnext[3:6] = np.clip(xnext[3:6], -0.3, 0.3)
                xnext[6:] = np.clip(xnext[6:], 0.0, P_MAX_KPA)
                Sx = A_k @ Sx
                Sx[:, m*k:m*(k+1)] += B_k
                Su[6*k:6*k+6] = C @ Sx
                Ynom[6*k:6*k+6] = C @ xnext
                if k == 0:
                    first_B = B_k.copy()
                    first_next = xnext.copy()
                if hasattr(self.dm, "advance_rollout"):
                    self.dm.advance_rollout(xbar, effective_ubar)
                xbar = xnext
        except Exception as exc:
            print(f"  !! 逐拍线性化失败，安全退让：{exc}")
            u_ = np.clip(self.u_prev*0.5, CO_CONTRACT_MV, CLAMP_MV)
            self.q_pred = None; self.u_prev = u_; self.ok = False
            return u_, x_est

        Yref = np.concatenate([np.concatenate([r, np.zeros(3)]) for r in r_hor])
        command_ubar = self._lock_shifted_commands(ubar)
        dub = (command_ubar - self.u_prev).ravel()
        DtD = self._D.T @ self._D
        # Ynom 已经是 ubar 作用下的真正模型滚动结果；决策量只剩 δU，
        # 因此这里不能再把名义输入 dub 通过 Su 重复加一次。
        R0 = Ynom - Yref
        command_DtD = command_T.T @ DtD @ command_T
        H = (Su.T @ self._Qb @ Su + self.rd*command_DtD +
             self.rho*np.diag(availability_flat))
        g = (Su.T @ self._Qb @ R0 +
             self.rd*(command_T.T @ (DtD @ dub)))
        if u_ff_hor is None:
            g = g + self.rho*availability_flat*np.tile(
                self.u_prev - CO_CONTRACT_MV, N)
        # u>=0，因此 ||u||₁ = Σu。常数 Σubar 不影响最优解，梯度直接加 λ·1；
        # 与 L2 不同，线性耗气成本会把收益不足的通道推到精确的 0。
        g = g + self.l1*availability_flat
        if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
            print("  !! 数据模型 QP 非有限，安全退让")
            u_ = np.clip(self.u_prev*0.5, CO_CONTRACT_MV, CLAMP_MV)
            self.u_prev = u_; self.ok = False
            return u_, x_est
        lo = (CO_CONTRACT_MV - ubar).ravel(); hi = (CLAMP_MV - ubar).ravel()
        def hard_project(v):
            total = _antagonist_project(ubar + v.reshape(N, m))
            return (total-ubar).ravel()
        extra_grad = None
        lip = np.linalg.norm(H, 2)
        if self.lambda_c:
            ratio = self.x_comp_ratio
            weight = self.lambda_c
            lip += weight*(1.0 + ratio*ratio)

            def coupling_grad(v):
                total = ubar + v.reshape(N, m)
                effective_total = availability*total
                shortfall = np.maximum(
                    ratio*effective_total[:, 4] - effective_total[:, 3], 0.0)
                grad = np.zeros_like(total)
                grad[:, 4] = weight*ratio*shortfall*availability[:, 4]
                grad[:, 3] = -weight*shortfall*availability[:, 3]
                return grad.ravel()
            extra_grad = coupling_grad

        V = MPC._fista(H, g, np.zeros(m*N), lip, lo, hi,
                       project=hard_project, extra_grad=extra_grad)
        plan = _antagonist_project(np.clip(
            ubar + V.reshape(N, m), CO_CONTRACT_MV, CLAMP_MV))
        self.lock_cmd = self._lock_action_now(plan)
        u = self._lock_command_now(plan)
        u_effective_now = self._lock_effective_now(u)
        # 返回本次 LTV 模型对实际第一拍控制的预测；下一控制拍仍以传感器实测
        # x_est 重新建立整条名义轨迹，不把这里的预测状态盲目递推。
        xn = first_next + first_B @ (
            u_effective_now - availability[0]*ubar[0])
        xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
        xn[3:6] = np.clip(xn[3:6], -0.3, 0.3)
        xn[6:] = np.clip(xn[6:], 0.0, P_MAX_KPA)
        self.u_plan = plan.copy()
        self.q_pred = xn[:3].copy(); self.u_prev = u; self.ok = True
        return np.clip(u, 0, CLAMP_MV), xn

PI_KP = 4500.0   # mV/rad（≈80 mV/°，teach_mode 投影 P 的量级）
PI_KI = 1500.0   # mV/(rad·s)
PI_KD = 0.0      # mV·s/rad；默认关闭，GUI 可在线调整
PI_KV = 1500.0   # mV/(rad/s)：速度环比例增益
PI_VMAX_DEG = 30.0  # 外环允许的最大目标角速度（deg/s）
PI_VPOS = 2.0    # 位置误差→目标速度，单位 1/s；5°误差约给10°/s
PID_D_ALPHA = 0.7  # 测量微分的一阶低通系数；越大越平滑
PI_AXIS_MIN_HOLD_S = 0.25   # 单轴至少保持这么久，避免20Hz逐拍抢占
PI_AXIS_HYST_DEG = 1.5      # 新轴必须比当前轴多出该误差才允许抢占
PI_AXIS_DONE_DEG = 0.5      # 当前轴进入此死区后可立即交棒；全轴进死区则零输出
PI_PLAN_DEADBAND_DEG = 5.0  # 相邻示教点的 GUI X/Y/φ 变化不超过5°时不安排该物理轴
PI_PLAN_MAX_DURATION_S = 5.0 # 每个示教段的基础规划时长上限
PI_STAGE_DONE_DEG = 1.0     # 基础时间结束时，超过该误差才使用延长预算
PI_STAGE_EXTENSION_RATIO = 0.30 # 每个GUI段最多延长其基础规划时间的30%
# PID 专用固定选通表（不用于 MPC）：
#   V0={1,5}: y+，V1={4,8}: y-；V2={2,3}: φ+，V3={6,7}: φ-；V4=9: x-，V5=10: x+。
# GUI/机构轴到模型 q：y+=-q1，φ+=+q2，x+=+q3。
# 物理通道：V0/V1=y+/y-，V2/V3=φ+/φ-，V4/V5=x-/x+。
PID_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=int)
PID_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
# PI 的 X 耦合补偿：9号(V4, x-)动作时同步6/7(V3, φ-)；
# 10号(V5, x+)动作时同步2/3(V2, φ+)。补偿电压先按主通道1:1，实机后可调比例。
PI_X_PHI_COMP_CHANNEL = {4: 3, 5: 2}
PI_X_PHI_COMP_RATIO = 1.0
PI_X_PHI_COMP_RATIOS = {4: PI_X_PHI_COMP_RATIO, 5: PI_X_PHI_COMP_RATIO}
try:
    _x_comp_blob = json.load(open(X_COMP_PATH, encoding="utf-8"))
    _x_comp_saved = _x_comp_blob.get("ratios", {})
    PI_X_PHI_COMP_RATIOS[4] = float(_x_comp_saved.get("9_plus_67", PI_X_PHI_COMP_RATIO))
    PI_X_PHI_COMP_RATIOS[5] = float(_x_comp_saved.get("10_plus_23", PI_X_PHI_COMP_RATIO))
except Exception:
    pass


def _x_phi_comp_ratio(primary_channel):
    return float(np.clip(PI_X_PHI_COMP_RATIOS.get(int(primary_channel),
                                                  PI_X_PHI_COMP_RATIO), 0.0, 2.0))

class PIController:
    """固定轴/方向 PID：只控制当前绝对角度误差最大的轴，并只启用其所需的一侧肌肉。
    该映射完全独立于 DataModel，且不改变 MPC 的通道分配。
    D 项对测量值微分并低通滤波，避免参考轨迹变化时产生微分冲击。
    单边输出 + 电压钳位 + 抗饱和。无预测、无气压模型——纯基线。"""
    def __init__(self, plant, kp=PI_KP, ki=PI_KI, kd=PI_KD,
                 kv=PI_KV, vmax_deg=PI_VMAX_DEG, dt=PI_DT):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.kv = float(kv)
        self.vmax = math.radians(float(vmax_deg))
        self.dt = float(dt)
        self.I = np.zeros(NCH)
        self.prev_meas = np.zeros(NCH)
        self.d_filt = np.zeros(NCH)
        self.last_vref = np.zeros(NCH)
        self.last_vmeas = np.zeros(NCH)
        self.active_axis = None
        self.axis_hold_s = 0.0
        self.first = True

    def set_gains(self, kp, ki, kd, kv=PI_KV, vmax_deg=PI_VMAX_DEG):
        """在线换增益；保留积分状态，避免应用参数时输出突然归零。"""
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.kv = float(kv)
        self.vmax = math.radians(float(vmax_deg))

    def set_dt(self, dt):
        self.dt = float(dt)

    def reset(self):
        self.I[:] = 0.0
        self.prev_meas[:] = 0.0
        self.d_filt[:] = 0.0
        self.last_vref[:] = 0.0
        self.last_vmeas[:] = 0.0
        self.active_axis = None
        self.axis_hold_s = 0.0
        self.first = True

    @staticmethod
    def signed_errors(q, r):
        """六路单向误差；>0 才允许该路输出，<=0 时该路必须为零。"""
        e = np.asarray(r, float) - np.asarray(q, float)
        return PID_SIGN * e[PID_AXIS]

    def _select_axis(self, q, r):
        """带保持、滞回和完成死区的单轴调度器；每个控制拍只调用一次。"""
        ae = np.abs(np.asarray(r, float) - np.asarray(q, float))
        done = math.radians(PI_AXIS_DONE_DEG)
        hyst = math.radians(PI_AXIS_HYST_DEG)
        candidate = int(np.argmax(ae))

        if float(ae[candidate]) <= done:
            self.active_axis = None
            self.axis_hold_s = 0.0
            return None
        if self.active_axis is None:
            self.active_axis = candidate
            self.axis_hold_s = 0.0
            return self.active_axis

        self.axis_hold_s += self.dt
        current = int(self.active_axis)
        if candidate != current:
            current_done = float(ae[current]) <= done
            challenger_wins = (self.axis_hold_s >= PI_AXIS_MIN_HOLD_S and
                               float(ae[candidate]) >= float(ae[current]) + hyst)
            if current_done or challenger_wins:
                self.active_axis = candidate
                self.axis_hold_s = 0.0
        return self.active_axis

    def active_mask(self, q, r):
        """返回本拍 step 已选中轴所允许的单侧通道；本函数不再次切换轴。"""
        if self.active_axis is None:
            return np.zeros(NCH, dtype=bool)
        return ((self.signed_errors(q, r) > math.radians(PI_AXIS_DONE_DEG)) &
                (PID_AXIS == int(self.active_axis)))

    def step(self, q, r, rdot=None, forced_axis=None, forced_channel=None):
        """位置 PID 加速度内环。

        外环用 v_ref = rdot + PI_VPOS*位置误差生成期望速度，内环用
        kv*(v_ref-v_meas) 修正电压。误差方向仍由固定六路表决定，速度环不能
        绕过单边选通或拮抗肌硬约束。
        """
        rdot = np.zeros(3) if rdot is None else np.asarray(rdot, float)
        s_all = self.signed_errors(q, r)
        if forced_axis is None:
            active_axis = self._select_axis(q, r)
        else:
            active_axis = int(forced_axis)
            if self.active_axis != active_axis:
                self.active_axis = active_axis
                self.axis_hold_s = 0.0
            else:
                self.axis_hold_s += self.dt
        raw = np.zeros(NCH)
        allowed = np.zeros(NCH, dtype=bool)
        for k in range(NCH):
            s = float(s_all[k])
            meas = float(PID_SIGN[k] * q[PID_AXIS[k]])
            d_meas = 0.0 if self.first else -(meas - self.prev_meas[k])/self.dt
            alpha = PID_D_ALPHA**(self.dt/DT)    # 改采样率时保持相同的实际滤波时间常数
            self.d_filt[k] = alpha*self.d_filt[k] + (1.0-alpha)*d_meas
            self.prev_meas[k] = meas
            v_meas = -self.d_filt[k]
            v_ref = float(np.clip(PID_SIGN[k]*rdot[PID_AXIS[k]] + PI_VPOS*s,
                                  0.0, self.vmax))
            self.last_vref[k] = v_ref
            self.last_vmeas[k] = v_meas
            if (active_axis is None or PID_AXIS[k] != active_axis or
                    (forced_channel is not None and k != int(forced_channel)) or
                    s <= math.radians(PI_AXIS_DONE_DEG)):
                self.I[k] = 0.0                 # 误差换向：旧侧积分立即清除，禁止继续充气
                continue
            allowed[k] = True
            raw[k] = (self.kp*s + self.I[k] + self.kd*self.d_filt[k]
                      + self.kv*(v_ref-v_meas))
        self.first = False
        # 先做上下限及对抗肌硬约束，再决定是否积分；被硬约束压成 0 的一路也不能继续积累。
        u = _antagonist_project(np.clip(raw, 0.0, CLAMP_MV))
        for k in range(NCH):
            s = float(s_all[k])
            if allowed[k] and s > 0.0 and abs(u[k] - raw[k]) < 1e-9:
                self.I[k] += self.ki*s*self.dt
        return u


class MuscleLengthPIController:
    """六阀组独立长度 PI。

    每拍实时使用 e=L实测-L目标；只有肌肉比参考长超过死区时才给该路
    充气使其缩短。需要拉长、已经到位或传感器掉线时停止并清积分。
    最终仍经过三对拮抗肌硬约束。
    """
    def __init__(self, kp=MUSCLE_PI_KP, ki=MUSCLE_PI_KI,
                 deadband=MUSCLE_PI_DEADBAND_MM, dt=PI_DT):
        self.kp = float(kp)
        self.ki = float(ki)
        self.deadband = float(deadband)
        self.dt = float(dt)
        self.I = np.zeros(NCH)

    def set_gains(self, kp, ki, deadband=MUSCLE_PI_DEADBAND_MM):
        self.kp = float(kp)
        self.ki = float(ki)
        self.deadband = float(deadband)

    def set_dt(self, dt):
        self.dt = float(dt)

    def reset(self):
        self.I[:] = 0.0

    def step(self, measured_mm, target_mm, allowed_channels=None):
        measured = np.asarray(measured_mm, float)
        target = np.asarray(target_mm, float)
        valid = np.isfinite(measured) & np.isfinite(target)
        if allowed_channels is not None:
            valid &= np.asarray(allowed_channels, bool)
        err = MUSCLE_PI_SENSOR_SIGN * (measured - target)
        err_eff = np.where(valid & (err > self.deadband), err-self.deadband, 0.0)

        # 目标已经达到、方向反了或传感器掉线时清积分；掉线通道必须严格为0。
        self.I[~(valid & (err_eff > 0.0))] = 0.0
        raw = self.kp*err_eff + self.I
        u = _antagonist_project(np.clip(raw, 0.0, CLAMP_MV))

        # 被饱和或被拮抗硬约束压掉的通道不积分，防止解除约束时突然冲满电压。
        for ch in range(NCH):
            if valid[ch] and err_eff[ch] > 0.0 and abs(u[ch]-raw[ch]) < 1e-9:
                self.I[ch] = float(np.clip(
                    self.I[ch] + self.ki*err_eff[ch]*self.dt,
                    0.0, MUSCLE_PI_I_MAX_MV))
        u[~valid] = 0.0
        return u, err

# ───────────────────────── bridge：GUI 当串口网关，本站纯计算 ────────────────
# 拓扑：GIGA ══串口══ GUI ──HTTP──► 本站。串口永远只归 GUI，一个持有者，无独占冲突。
# GUI POST /step {quat:[w,x,y,z], lens:[10], kpa:[6]|null, locks:"owL..."}，本站算一拍，响应里
# 直接带 {u:[六路mV]}，GUI 转手经 Web Serial 写给固件（W 指令）。
#   POST /start {points}  设定轨迹并开始    POST /stop  清零停止
#   GET  /status          {mode,...}        POST /step  见上
# 注意：控制节拍由浏览器驱动——标签页切到后台会被浏览器降频，固件 2s 看门狗兜底。
FEL_NN_PATH = os.path.join(HERE, "fel_network.pt")
NEW_MPC_FEL_NN_PATH = os.path.join(HERE, "new_mpc_fel_inverse.pt")
NEW_MPC_FEL_NN_SHADOW_PATH = os.path.join(HERE, "new_mpc_fel_inverse_shadow.pt")
NEW_MPC_FEL_NN_PREVIOUS_PATH = os.path.join(HERE, "new_mpc_fel_inverse_previous.pt")
NEW_MPC_FEL_NN_STATE_PATH = os.path.join(HERE, "new_mpc_fel_inverse_state.json")
NEW_MPC_FEL_ACTIVE_PATH = os.path.join(HERE, "new_mpc_fel_active.pt")
MPC_SHADOW_FORWARD_ACTIVE_PATH = os.path.join(HERE, "mpc_shadow_active.pt")
MPC_SHADOW_FORWARD_SHADOW_PATH = os.path.join(HERE, "mpc_shadow_shadow.pt")
MPC_SHADOW_FORWARD_PREVIOUS_PATH = os.path.join(HERE, "mpc_shadow_previous.pt")
MPC_SHADOW_FORWARD_STATE_PATH = os.path.join(HERE, "mpc_shadow_state.json")
MPC_SHADOW_SEQUENCE_PATHS = {
    architecture: {
        "base": SEQUENCE_MODEL_FILES[architecture][0],
        "meta": SEQUENCE_MODEL_FILES[architecture][1],
        "active": os.path.join(HERE, f"mpc_shadow_{architecture}_active.pt"),
        "shadow": os.path.join(HERE, f"mpc_shadow_{architecture}_shadow.pt"),
        "previous": os.path.join(HERE, f"mpc_shadow_{architecture}_previous.pt"),
        "state": os.path.join(HERE, f"mpc_shadow_{architecture}_state.json"),
    }
    for architecture in ("history_mlp", "gru")
}
ORIGINAL_FORWARD_PATH = os.path.join(HERE, "original_pure_mpc_model.pt")
ORIGINAL_FORWARD_META_PATH = os.path.join(HERE, "original_pure_mpc_model.json")
ORIGINAL_INVERSE_PATH = os.path.join(HERE, "original_fel_network.pt")
MPC_ILC_STATE_PATH = os.path.join(HERE, "mpc_ilc_state.npz")
CODEX_COMP_STATE_PATH = os.path.join(HERE, "codex_compensator_state.json")
CODEX_COMP_CANDIDATE_PATH = os.path.join(HERE, "codex_compensator_candidate.json")
CODEX_COMP_REQUEST_PATH = os.path.join(HERE, "codex_compensator_request.json")
CODEX_COMP_SCHEMA_PATH = os.path.join(HERE, "codex_compensator_candidate_schema.json")
CODEX_REPLAY_RUNTIME_DIR = os.path.join(
    tempfile.gettempdir(), f"tro_codex_replay_{os.getpid()}")
CODEX_REPLAY_STATE_PATH = os.path.join(CODEX_REPLAY_RUNTIME_DIR, "state.json")
CODEX_REPLAY_CANDIDATE_PATH = os.path.join(CODEX_REPLAY_RUNTIME_DIR, "candidate.json")
CODEX_REPLAY_REQUEST_PATH = os.path.join(CODEX_REPLAY_RUNTIME_DIR, "request.json")
# “原始初始化”有自己的活动/影子学习分支。原始快照本身永远不覆盖，
# 同时也不会污染“最新已验证”分支，便于比较两种初始化的收敛效果。
ORIGINAL_INIT_FORWARD_ACTIVE_PATH = os.path.join(HERE, "original_init_mpc_fel_active.pt")
ORIGINAL_INIT_FORWARD_SHADOW_PATH = os.path.join(HERE, "original_init_mpc_fel_shadow.pt")
ORIGINAL_INIT_FORWARD_PREVIOUS_PATH = os.path.join(HERE, "original_init_mpc_fel_previous.pt")
ORIGINAL_INIT_FORWARD_STATE_PATH = os.path.join(HERE, "original_init_mpc_fel_state.json")
ORIGINAL_INIT_INVERSE_ACTIVE_PATH = os.path.join(HERE, "original_init_mpc_fel_inverse.pt")
ORIGINAL_INIT_INVERSE_SHADOW_PATH = os.path.join(HERE, "original_init_mpc_fel_inverse_shadow.pt")
ORIGINAL_INIT_INVERSE_PREVIOUS_PATH = os.path.join(HERE, "original_init_mpc_fel_inverse_previous.pt")
ORIGINAL_INIT_INVERSE_STATE_PATH = os.path.join(HERE, "original_init_mpc_fel_inverse_state.json")
SHADOW_REPLAY_ROUNDS = 3                            # 顺/逆影子模型每3个完整轨迹合并训练一次
FEL_SCHEMA = 3                                     # v3：PyTorch MLP 逆模型
FEL_NF = 37                                        # 归一化输入：参考/实测/误差/压力/上拍输出/锁
FEL_H1, FEL_H2 = 32, 16                            # 每路独立的 37→32→16→1 MLP
FEL_EPOCHS = 60                                    # 圈末批量训练，圈内网络冻结
FEL_LR = 3e-3
FEL_WEIGHT_DECAY = 1e-5
FEL_TRACK_WEIGHT = 0.08                            # 混合 loss 中的模型预测角度误差权重
FEL_INVERSE_TRAJECTORY_WEIGHT = 1.0               # NEW MPC+FEL：逆模型经冻结顺模型追reference的主loss权重
FEL_TRACK_HORIZON = 1                              # 逆模型只用冻结真正顺模型做一步预测；每个样本从实测x(t)出发

# B 方向校正阵 S（2026-09-04，ident 实测拟合）：真机 dq = S·模型 dq。
# 泄漏矩阵显示模型方向失配 38%（如"模型纯 q2 动作，真机带 −32% 进扭转"），
# S 把 B 的姿态行拉向实测方向。文件由 fitS 流程生成；缺失时退化为单位阵。
try:
    _S_CORR = np.array(json.load(open(os.path.join(HERE, "b_correction.json"))))
    assert _S_CORR.shape == (3, 3)
    print("# B 方向校正阵 S 已载入（ident 实测拟合）")
except Exception:
    _S_CORR = np.eye(3)
# 预测卫生用：一步仿射项 w 的物理界，按采样周期缩放。
_W_CLIP = np.array([1.5*DT]*3 + [25.0*DT]*3 + [200.0*DT]*6)
FEL_BLEND_PI = 0.2                                 # 每圈只吸收 20% 反馈修正，防止一次坏数据改坏前馈
FEL_BLEND_MPC = 0.2
FEL_FF_MAX = CLAMP_MV                              # 前馈限幅同总上限（作者定：不设额外限制）

# ── PyTorch 逆模型：六路独立 MLP，避免某一路坏样本污染其他通道 ────────
def _fel_phi(r, rd, q=None, qd=None, pressure=None, u_prev=None, locks=None):
    """状态感知逆模型特征。

    静态的 ``目标轨迹→电压`` 无法区分气压、误差和迟滞状态；这里把当前实测状态一起
    送入逆模型。各组先按物理量级归一化，避免 kPa/mV 的数值尺度支配 NN 训练。
    """
    r = np.asarray(r, float); rd = np.asarray(rd, float)
    q = np.zeros(3) if q is None else np.asarray(q, float)
    qd = np.zeros(3) if qd is None else np.asarray(qd, float)
    pressure = np.zeros(NCH) if pressure is None else np.asarray(pressure, float)
    u_prev = np.zeros(NCH) if u_prev is None else np.asarray(u_prev, float)
    locks = np.zeros(NCH) if locks is None else np.asarray(locks, float)
    base = np.concatenate([
        [1.0], r/0.5, rd, 0.3*np.sign(rd), q/0.5, qd, (r-q)/0.5,
        pressure/max(P_MAX_KPA, 1.0), u_prev/max(CLAMP_MV, 1.0), locks
    ])
    return np.nan_to_num(np.clip(base, -3.0, 3.0))

def _fel_torch():
    """PyTorch 只在 MPC/FEL 运行时加载，DataModel 辨识仍可单独导入本文件。"""
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError("当前 Python 环境没有 PyTorch；请用 conda 的 pytorch 环境启动 qt_gui.py") from e

def _fel_new(seed=20260904):
    """创建 6 个小 MLP。末层为零，第一轮前馈严格为 0，不会随机充气。"""
    torch = _fel_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    heads = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.Linear(FEL_NF, FEL_H1), torch.nn.Tanh(),
            torch.nn.Linear(FEL_H1, FEL_H2), torch.nn.Tanh(),
            torch.nn.Linear(FEL_H2, 1),
        ) for _ in range(NCH)
    ])
    for head in heads:
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)
    return heads.cpu()

def _fel_torch_project(y):
    """NN 张量的拮抗对硬投影；每对较小一侧严格为 0。"""
    torch = _fel_torch()
    cols = []
    for left, right in ANTAGONIST_PAIRS:
        yl, yr = y[:, left], y[:, right]
        keep_left = yl >= yr
        cols.extend([torch.where(keep_left, yl, torch.zeros_like(yl)),
                     torch.where(keep_left, torch.zeros_like(yr), yr)])
    return torch.stack(cols, dim=1)

def _fel_predict(nn_model, phi):
    """计算六路 NN 前馈，返回 mV；支持单样本或 (n,37) 批量。"""
    torch = _fel_torch()
    X = np.asarray(phi, np.float32)
    one = X.ndim == 1
    if one: X = X[None, :]
    xt = torch.from_numpy(X)
    nn_model.eval()
    with torch.no_grad():
        y = torch.cat([head(xt) for head in nn_model], dim=1).clamp_(0.0, 1.0)
        y = _fel_torch_project(y)
    out = y.numpy()*FEL_FF_MAX
    return out[0] if one else out

def _fel_train(nn_model, phi, residual, mask, beta, dm=None):
    """混合 loss：学 MPC/PI 反馈修正 + 通过冻结 DataModel 最小化预测角度误差。"""
    torch = _fel_torch()
    if len(phi) < 20: return 0
    Xn = np.asarray(phi, np.float32)
    E = np.asarray(residual, np.float32)
    M = np.asarray(mask, bool)
    if Xn.shape != (len(Xn), FEL_NF) or E.shape != (len(Xn), NCH) or M.shape != E.shape: return 0
    if not np.any(M): return 0
    old = _fel_predict(nn_model, Xn)
    target_np = np.clip((old + beta*E)/max(FEL_FF_MAX, 1.0), 0.0, 1.0).astype(np.float32)
    X = torch.from_numpy(Xn); target = torch.from_numpy(target_np); valid = torch.from_numpy(M)
    # DataModel 参数只做定值可微分仿真，训练时不修改顺模型。
    # The legacy linear DataModel supports a differentiable tracking term.
    # A TorchScript pure-data model is still a valid MPC predictor, but its
    # weights stay encapsulated; FEL then learns from feedback residual only.
    track_dm = dm if dm is not None and all(
        hasattr(dm, key) for key in ("Av", "K", "Bp", "c", "g", "tu", "td")) else None
    if track_dm is not None:
        Av = torch.as_tensor(track_dm.Av, dtype=torch.float32)
        Km = torch.as_tensor(track_dm.K, dtype=torch.float32)
        Bp = torch.as_tensor(track_dm.Bp, dtype=torch.float32)
        cc = torch.as_tensor(track_dm.c, dtype=torch.float32)
        gg = torch.as_tensor(track_dm.g, dtype=torch.float32)
        tu = torch.as_tensor(track_dm.tu, dtype=torch.float32)
        td = torch.as_tensor(track_dm.td, dtype=torch.float32)
    opt = torch.optim.Adam(nn_model.parameters(), lr=FEL_LR, weight_decay=FEL_WEIGHT_DECAY)
    nn_model.train()
    for _ in range(FEL_EPOCHS):
        opt.zero_grad()
        pred_raw = torch.cat([head(X) for head in nn_model], dim=1)
        loss_fb = torch.nn.functional.smooth_l1_loss(pred_raw[valid], target[valid], beta=0.05)
        loss_track = torch.zeros((), dtype=torch.float32)
        if track_dm is not None:
            r = X[:, 1:4]*0.5
            q = X[:, 10:13]*0.5
            dq = X[:, 13:16]*DT
            P = X[:, 19:25]*P_MAX_KPA
            pred_hard = _fel_torch_project(pred_raw.clamp(0.0, 1.0))
            u = pred_hard*FEL_FF_MAX
            for _h in range(FEL_TRACK_HORIZON):
                qn = q + dq @ Av.T - q @ Km.T + P @ Bp.T + cc
                Pc = u*gg
                tau = torch.where(Pc >= P, tu, td)
                Dd = torch.clamp(DT/torch.clamp(tau, min=0.05), max=1.0)
                P = P + Dd*(Pc-P)
                dq, q = qn-q, qn
            loss_track = torch.mean(((q-r)/0.5)**2)
        loss = loss_fb + FEL_TRACK_WEIGHT*loss_track
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
        opt.step()
    nn_model.eval()
    print(f"#   FEL NN loss：反馈 {float(loss_fb):.5f} + "
          f"角度 {float(loss_track):.5f}×{FEL_TRACK_WEIGHT:g}")
    return int(np.sum(M))

def _fel_lock_state(raw):
    """把固件 o/w/L 转成 0/0.5/1；同时返回原始状态与帧是否有效。"""
    if not isinstance(raw, str): return np.zeros(NCH), ["?"]*NCH, False
    st = [c for c in raw if c in "owL"][:NCH]
    if len(st) != NCH: return np.zeros(NCH), ["?"]*NCH, False
    return np.array([0.0 if c == "o" else 0.5 if c == "w" else 1.0 for c in st]), st, True

def _fel_load(path=None, bootstrap_path=None):
    """载入 FEL 的 PyTorch MLP；MPC 的唯一顺模型仍是 DataModel。"""
    path = path or FEL_NN_PATH
    try:
        load_path = path if os.path.exists(path) else bootstrap_path
        if load_path and os.path.exists(load_path):
            torch = _fel_torch()
            blob = torch.load(load_path, map_location="cpu")
            if isinstance(blob, dict) and blob.get("schema") == FEL_SCHEMA:
                model = _fel_new(); model.load_state_dict(blob["state_dict"]); model.eval()
                source = "（从现有FEL初始化）" if load_path != path else ""
                print(f"# FEL PyTorch MLP 已载入（{load_path}）{source}")
                return model
        if os.path.exists(os.path.join(HERE, "fel_weights.json")):
            print("# 旧的岭回归/RBF 权重不再使用，PyTorch MLP 从零学习")
    except Exception as e:
        print(f"# FEL MLP 读取失败，本次从零学习：{e}")
    return None


def _ensure_original_model_pair():
    """首次选择“原始基线”时固化当前基线文件，以后任何学习模式都不覆盖它们。"""
    pairs = ((PURE_DM_PATH, ORIGINAL_FORWARD_PATH),
             (PURE_DM_META_PATH, ORIGINAL_FORWARD_META_PATH),
             (FEL_NN_PATH, ORIGINAL_INVERSE_PATH))
    missing = [src for src, _ in pairs if not os.path.exists(src)]
    if missing:
        raise FileNotFoundError("无法建立原始模型快照，缺少：" + ", ".join(missing))
    created = []
    for src, dst in pairs:
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            created.append(dst)
    if created:
        print("# 已固化原始正/逆模型快照：" + ", ".join(created))

def _fel_weight_mean(nn_model):
    if nn_model is None: return 0.0
    vals = [float(p.detach().abs().mean()) for p in nn_model.parameters()]
    return float(np.mean(vals)) if vals else 0.0


# ───────────────────────── 增益面板：control/gains.json（作者定 2026-09-04）──────
# 每次点【播放】(/start) 重新读盘 → 改完存盘、直接重播即可生效，不用重启控制台。
# 文件不存在就按下面的默认值自动写一份出来。
GAINS_PATH = os.path.join(HERE, "gains.json")
GAINS_DEFAULT = {
    "_说明": "改完存盘 → GUI 重按播放即生效（不用重启）。删掉本文件可恢复默认。",
    "mpc_forward_model": "mlp",
    "_mpc_forward_model": "真正顺模型：mlp=单拍模型；history_mlp=10拍历史残差MLP；gru=10拍GRU。纯MPC及MPC+影子模型均支持，时序模型固定使用LTV求解器",
    "mpc_N": 20,
    "_mpc_N": "预测时域（拍，1拍=0.05s）。当前20拍=1.0s，兼顾气路响应与20Hz计算量",
    "mpc_lock_lead_ms": 100.0,
    "_mpc_lock_lead_ms": "开关锁阀先于比例阀动作的时间；100ms在20Hz下占2拍，MPC会把这2拍按uact=0规划",
    "mpc_solver": "mppi",
    "_mpc_solver": "求解方式：acados_sqp_rti=acados+HPIPM真RTI；sqp=OSQP+重复线性化；mppi=无梯度批量采样；nonlinear=直接滚动+梯度优化；ltv=FISTA线性化QP",
    "mpc_nonlinear_iterations": 20,
    "_mpc_nonlinear_iterations": "每拍最多梯度更新次数；同时受计算时间预算限制",
    "mpc_nonlinear_lr_mv": 180.0,
    "_mpc_nonlinear_lr_mv": "非线性MPC的Adam步长，单位近似mV；过大会跳变，过小会来不及收敛",
    "mpc_nonlinear_budget_ms": 40.0,
    "_mpc_nonlinear_budget_ms": "每个50ms控制拍留给非线性/MPPI优化的最长计算时间",
    "mpc_mppi_samples": 192,
    "_mpc_mppi_samples": "MPPI每轮候选电压轨迹数；越大搜索越充分但计算越慢",
    "mpc_mppi_iterations": 2,
    "_mpc_mppi_iterations": "MPPI每个控制拍的采样更新轮数，同时受40ms预算限制",
    "mpc_mppi_sigma_mv": 700.0,
    "_mpc_mppi_sigma_mv": "MPPI第一轮探索电压标准差mV，后续迭代自动缩小",
    "mpc_mppi_temperature": 1.0,
    "_mpc_mppi_temperature": "MPPI权重温度；小=更相信少数最优样本，大=输出更平均",
    "mpc_mppi_knots": 5,
    "_mpc_mppi_knots": "20拍电压轨迹中的随机控制节点数，节点间线性插值以减少抖动",
    "mpc_sqp_iterations": 2,
    "_mpc_sqp_iterations": "SQP每个控制拍最多重复线性化和求QP的次数",
    "mpc_sqp_max_qp_iterations": 100,
    "_mpc_sqp_max_qp_iterations": "每次OSQP内部最大迭代次数，同时受40ms总预算限制",
    "mpc_acados_qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "_mpc_acados_qp_solver": "acados内部QP求解器；推荐PARTIAL_CONDENSING_HPIPM",
    "mpc_acados_cond_N": 5,
    "_mpc_acados_cond_N": "HPIPM部分凝聚后的区间数，1到mpc_N；5通常适合20拍",
    "mpc_acados_cmake_generator": "",
    "_mpc_acados_cmake_generator": "Windows CMake生成器；空字符串用acados默认值，有新版VS时可写Visual Studio 17 2022",
    "mpc_acados_codegen_dir": "",
    "_mpc_acados_codegen_dir": "生成代码缓存目录；空字符串使用control/acados_generated",
    "mpc_acados_source_dir": "",
    "_mpc_acados_source_dir": "acados源码/编译根目录；若系统已设ACADOS_SOURCE_DIR可留空",
    "mpc_w_q": [5000.0, 5000.0, 5000.0],
    "_mpc_w_q": "跟踪权重 [q1 q2 q3]，三轴同等重要。调大=更用力追该轴",
    "mpc_w_v": [300.0, 300.0, 300.0],
    "_mpc_w_v": "速度阻尼权重（目标速度恒 0）。调大=运动更柔更慢、抑制过冲；调小=更快更冲",
    "mpc_rd": 1e-3,
    "_mpc_rd": "相邻拍电压变化惩罚。调大=电压更平滑不抖，但跟不上快段。1e-4 ~ 2e-3",
    "mpc_rho": 0.0,
    "_mpc_rho": "电压绝对值/L2惩罚。当前关闭，只保留跟踪和电压变化惩罚",
    "mpc_l1": 3e-4,
    "_mpc_l1": "regulator稀疏/L1惩罚 λ·Σu。非负电压下把收益不足的通道推到精确0；当前为弱约束",
    "mpc_x_comp_ratio": 0.5,
    "_mpc_x_comp_ratio": "MPC的9+67补偿比例：希望u67至少达到该比例×u9；实机五档标定当前推荐0.5",
    "mpc_lambda_c": 0.005,
    "_mpc_lambda_c": "9+67单向软约束λc；越大越强，0关闭。只用于MPC，10号暂不参与",
    "ilc_learning_rate": 0.15,
    "_ilc_learning_rate": "MPC+ILC每轮吸收上一轮真实MPC反馈电压的比例；建议0.05~0.25",
    "ilc_max_update_mv": 100.0,
    "_ilc_max_update_mv": "MPC+ILC每个轨迹时刻、每个通道一轮最多改变的前馈电压mV",
    "ilc_smoothing_ticks": 5,
    "_ilc_smoothing_ticks": "MPC+ILC前馈更新的时间平滑窗口，20Hz下5拍=0.25s；偶数自动加1",
    "ilc_max_slew_mv_s": 2500.0,
    "_ilc_max_slew_mv_s": "MPC+ILC前馈曲线最大变化率mV/s；只约束学习曲线，不替代MPC硬限幅",
    "ilc_accept_ratio": 1.0,
    "_ilc_accept_ratio": "候选ILC下一轮真实跟踪MSE必须≤该倍数×基准MSE才保留；1.0表示不能变差",
    "codex_comp_accept_ratio": 1.0,
    "codex_final_fall_slew_mv_s": 1200.0,
    "codex_final_phi_rise_slew_mv_s": 600.0,
    "_codex_comp_accept_ratio": "候选补偿器真实跟踪MSE不超过上一稳定版的该倍数才保留；失败response仍导出给Codex继续设计",
    "_codex_final_fall_slew_mv_s": "仅LLM+MPC最终下发电压的最大下降速度mV/s；20Hz下1200表示每拍最多下降60mV",
    "_codex_final_phi_rise_slew_mv_s": "仅LLM+MPC最终V2/V3电压的最大上升速度mV/s；20Hz下600表示每拍最多上升30mV",
    "codex_reuse_mse_threshold_deg2": 15.0,
    "_codex_reuse_mse_threshold_deg2": "有效完整轮次的真实跟踪MSE不超过该值时沿用当前已验证补偿器并跳过LLM改版；0表示关闭",
    "codex_comp_angle_jump_deg": 10.0,
    "_codex_comp_angle_jump_deg": "单个20Hz控制拍允许的最大传感器角度跳变；超过则本轮标记异常且候选回退",
    "codex_auto_design": True,
    "_codex_auto_design": "LLM+MPC完整轮结束后调用所选GPT/Claude读取response并生成下一版候选；Qt GUI勾选循环时，候选就绪后自动开始下一轮",
    "codex_model": "",
    "_codex_model": "轮间离线设计使用的具体模型；GPT留空时使用Codex默认模型",
    "codex_prompt_profile": "base",
    "_codex_prompt_profile": "补偿器设计版本：base使用矩阵PID；plus使用压力感知分层补偿器；pro由LLM自由选择结构",
    "llm_provider": "gpt",
    "_llm_provider": "轮间离线补偿器使用的LLM提供商：gpt或claude；API Key只驻留内存",
    "llm_auth_method": "account",
    "_llm_auth_method": "所选LLM的认证方式：account或api；不保存任何API Key",
    "codex_reasoning_effort": "medium",
    "_codex_reasoning_effort": "LLM离线设计推理强度：low/medium/high/xhigh（默认medium）",
    "codex_fast_mode": False,
    "_codex_fast_mode": "GPT/Codex快速模式：消耗更多额度换取更低延迟；Claude忽略此项",
    "codex_auto_timeout_s": 300.0,
    "_codex_auto_timeout_s": "轮后LLM离线分析最长等待秒数，限制30~900；分析在后台进行，不进入20Hz控制环",
    "fel_track_weight": 0.08,
    "_fel_track_weight": "NN混合loss中的角度跟踪项权重。0=只学MPC反馈；建议0.03~0.15，太大会过度信任DataModel",
    "fel_inverse_trajectory_weight": 1.0,
    "_fel_inverse_trajectory_weight": "NEW MPC+FEL逆模型的冻结顺模型单步跟踪loss权重；1.0为默认，0会关闭这条直接追reference的训练路线",
    "pi_kp": 4500.0,
    "_pi_kp": "PI 比例增益 mV/rad（4500≈80mV/°）。调大=响应快、易振",
    "pi_ki": 1500.0,
    "_pi_ki": "PI 积分增益 mV/(rad·s)。调大=消稳态误差快、易积分超调",
    "pi_kd": 0.0,
    "_pi_kd": "PID 微分增益 mV·s/rad。对测量值微分并低通滤波；调大可抑制过冲，过大会放大噪声",
    "pi_kv": 1500.0,
    "_pi_kv": "速度环增益 mV/(rad/s)。卡住时补输出、实际速度过快时减输出；建议先在500~2500调",
    "pi_vmax_deg": 30.0,
    "_pi_vmax_deg": "速度环目标角速度上限 deg/s。越小越柔和，越大越敢追；建议15~45",
    "muscle_pi_kp": MUSCLE_PI_KP,
    "_muscle_pi_kp": "肌肉长度 PI 比例增益 mV/mm；只在肌肉长度 PI 模式有效",
    "muscle_pi_ki": MUSCLE_PI_KI,
    "_muscle_pi_ki": "肌肉长度 PI 积分增益 mV/(mm*s)；只在肌肉长度 PI 模式有效",
    "muscle_pi_deadband_mm": MUSCLE_PI_DEADBAND_MM,
    "_muscle_pi_deadband_mm": "肌肉长度误差死区 mm；死区内停止该阀并清积分",
    "pressure_hold_kp": PRESSURE_HOLD_KP,
    "_pressure_hold_kp": "保压内环 Kp，mV/kPa。压力掉下时立即增加的电压；太大会压力抖动",
    "pressure_hold_ki": PRESSURE_HOLD_KI,
    "_pressure_hold_ki": "保压内环 Ki，mV/(kPa*s)。消除长期压差；太大会积分过冲",
    "pressure_hold_deadband_kpa": PRESSURE_HOLD_DEADBAND_KPA,
    "_pressure_hold_deadband_kpa": "保压死区 kPa。压差在此范围内不修正，防止 ADC 噪声造成电压抖动",
    "u_max": 2200.0,
    "_u_max": "六路比例阀电压上限mV；唯一日常调节点，可设100～3000，保存后重新播放生效。",
    "dm_lam": 5.0,
    "_dm_lam": "数据模型岭回归强度：大=新数据改动小（保守、稳），小=更贴最新一轮（激进）。1 ~ 50",
}

def _gains_load():
    """读 gains.json（缺就写默认）。返回补全后的 dict，并同步 u_max 到全局钳位。"""
    g = dict(GAINS_DEFAULT)
    try:
        if os.path.exists(GAINS_PATH):
            blob = json.load(open(GAINS_PATH, encoding="utf-8"))
            for k in GAINS_DEFAULT:
                if not k.startswith("_") and k in blob: g[k] = blob[k]
        else:
            json.dump(GAINS_DEFAULT, open(GAINS_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"# 已生成增益面板 {GAINS_PATH}（改完存盘，重按播放即生效）")
    except Exception as e:
        print(f"# gains.json 读取失败（用默认值）：{e}")
    global CLAMP_MV, FEL_FF_MAX, FEL_TRACK_WEIGHT, FEL_INVERSE_TRAJECTORY_WEIGHT
    try:
        CLAMP_MV = float(np.clip(float(g["u_max"]), 100.0, HARD_CLAMP_MV))
        FEL_FF_MAX = CLAMP_MV
    except Exception:
        pass
    DataModel.LAM = float(g.get("dm_lam", 5.0))
    FEL_TRACK_WEIGHT = float(np.clip(float(g.get("fel_track_weight", 0.08)), 0.0, 1.0))
    FEL_INVERSE_TRAJECTORY_WEIGHT = float(np.clip(
        float(g.get("fel_inverse_trajectory_weight", 1.0)), 0.0, 10.0))
    return g


def mode_bridge(args):
    global BR
    if not os.path.exists(PARAMS_JSON): sys.exit("没有 params.json，先 ident + fit")
    p = json.load(open(PARAMS_JSON))
    plant = Plant(p)
    _G0 = _gains_load()
    try:
        _codex_effort0 = normalize_reasoning_effort(
            _G0.get("codex_reasoning_effort", "medium"))
    except ValueError:
        _codex_effort0 = "medium"
    _codex_prompt_profile0 = str(
        _G0.get("codex_prompt_profile", "base")).strip().lower()
    if _codex_prompt_profile0 not in ("base", "plus", "pro"):
        _codex_prompt_profile0 = "base"
    # 逆模型按播放模式惰性加载；纯MPC和“仅影子正模型”模式完全不加载逆模型。
    _NN0 = None
    BR = {"mode": "idle", "mpc": None, "ref": None, "k": 0, "qref": None,
          "x": np.zeros(12), "q_prev": None, "log": None, "path": "",
          "nn": _NN0, "fel_path": FEL_NN_PATH,
          "new_mpc_fel": False, "shadow_forward_only": False,
          "shadow_forward": False, "forward_learner": None,
          "ilc_mpc": False, "ilc": None,
          "ilc_u": [], "ilc_ff": [], "ilc_mask": [],
          "codex_mpc": False, "codex_comp": None,
          "codex_u_comp": np.zeros(NCH),
          "codex_auto_enabled": bool(_G0.get("codex_auto_design", True)),
          "llm_provider": str(_G0.get("llm_provider", "gpt")).strip().lower(),
          "llm_auth_method": str(_G0.get("llm_auth_method", "account")).strip().lower(),
          "codex_model": str(_G0.get("codex_model", "")).strip(),
          "codex_prompt_profile": _codex_prompt_profile0,
          "codex_reasoning_effort": _codex_effort0,
          "codex_fast_mode": bool(_G0.get("codex_fast_mode", False)),
          "codex_auto_timeout_s": float(_G0.get("codex_auto_timeout_s", 300.0)),
          "codex_reuse_mse_threshold_deg2": float(
              _G0.get("codex_reuse_mse_threshold_deg2", 15.0)),
          "codex_auto_thread": None,
          "codex_feedback_queue": [],
          "codex_feedback_history": [],
          "codex_feedback_event": threading.Event(),
          "codex_feedback_hold": False,
          "codex_feedback_deadline_monotonic": 0.0,
          "codex_auto_status": {
              "state": "idle", "message": "等待MPC+Codex完成一轮后自动分析"},
          "codex_capture_active": False, "codex_capture_rounds": [],
          "codex_replay": None, "codex_replay_active": False,
          "model_source": "latest", "models_frozen": False,
          "forward_arch": "mlp",
          "forward_last": None, "new_round_complete": False,
          "inverse_shadow": None, "inverse_replay": [],
          "inverse_rounds_seen": 0, "inverse_state": None,
          "inverse_active_path": NEW_MPC_FEL_NN_PATH,
          "inverse_shadow_path": NEW_MPC_FEL_NN_SHADOW_PATH,
          "inverse_previous_path": NEW_MPC_FEL_NN_PREVIOUS_PATH,
          "inverse_state_path": NEW_MPC_FEL_NN_STATE_PATH,
          "forward_meta_path": PURE_DM_META_PATH,
          "forward_promoted_this_round": False,
          "mpc_weights": {
              "wq": float(np.mean(_G0["mpc_w_q"])),
              "wv": float(np.mean(_G0["mpc_w_v"]))},
          "pid": {"kp": float(_G0["pi_kp"]), "ki": float(_G0["pi_ki"]),
                  "kd": float(_G0.get("pi_kd", 0.0)),
                  "kv": float(_G0.get("pi_kv", PI_KV)),
                  "vmax": float(_G0.get("pi_vmax_deg", PI_VMAX_DEG))},
          "pressure_pid": {
              "kp": float(_G0.get("pressure_hold_kp", PRESSURE_HOLD_KP)),
              "ki": float(_G0.get("pressure_hold_ki", PRESSURE_HOLD_KI)),
              "deadband": float(_G0.get("pressure_hold_deadband_kpa", PRESSURE_HOLD_DEADBAND_KPA))},
          "muscle_pid": {
              "kp": float(_G0.get("muscle_pi_kp", MUSCLE_PI_KP)),
              "ki": float(_G0.get("muscle_pi_ki", MUSCLE_PI_KI)),
              "deadband": float(_G0.get("muscle_pi_deadband_mm", MUSCLE_PI_DEADBAND_MM))}}
    LK = threading.Lock()

    def _llm_config_status():
        try:
            from codex_compensator import _find_codex_executable
            executable = _find_codex_executable()
        except Exception:
            executable = None
        return {
            "ok": 1,
            "provider": BR.get("llm_provider", "gpt"),
            "auth_method": BR.get("llm_auth_method", "account"),
            "model": BR.get("codex_model", ""),
            "prompt_profile": BR.get("codex_prompt_profile", "base"),
            "reasoning_effort": BR.get("codex_reasoning_effort", "medium"),
            "fast_mode": bool(BR.get("codex_fast_mode", False)),
            "timeout_s": float(BR.get("codex_auto_timeout_s", 300.0)),
            "reuse_mse_threshold_deg2": float(
                BR.get("codex_reuse_mse_threshold_deg2", 15.0)),
            "auto_design": bool(BR.get("codex_auto_enabled", True)),
            "cli_available": bool(executable),
            "claude_api_key_loaded": has_transient_llm_api_key("claude"),
            "cli_path": str(executable or ""),
            "design_running": bool(BR.get("codex_auto_thread") is not None and
                                   BR["codex_auto_thread"].is_alive()),
        }

    def _llm_config_apply(data):
        provider = str(data.get(
            "provider", BR.get("llm_provider", "gpt")) or "gpt").strip().lower()
        auth_method = str(data.get(
            "auth_method", BR.get("llm_auth_method", "account")) or "account").strip().lower()
        if auth_method == "chatgpt":
            auth_method = "account"
        model = str(data.get("model", BR.get("codex_model", "")) or "").strip()
        prompt_profile = str(data.get(
            "prompt_profile", BR.get("codex_prompt_profile", "base")) or
            "base").strip().lower()
        effort_raw = str(data.get(
            "reasoning_effort", BR.get("codex_reasoning_effort", "medium"))).strip().lower()
        try:
            effort = normalize_reasoning_effort(effort_raw)
        except ValueError as exc:
            return {"err": str(exc)}
        try:
            timeout_s = float(data.get(
                "timeout_s", BR.get("codex_auto_timeout_s", 300.0)))
        except (TypeError, ValueError):
            return {"err": "LLM分析超时必须是数字"}
        try:
            reuse_mse_threshold = float(data.get(
                "reuse_mse_threshold_deg2",
                BR.get("codex_reuse_mse_threshold_deg2", 15.0)))
        except (TypeError, ValueError):
            return {"err": "补偿器沿用MSE阈值必须是数字"}
        if provider not in ("gpt", "claude"):
            return {"err": "LLM提供商只能是gpt或claude"}
        if auth_method not in ("account", "api"):
            return {"err": "LLM认证方式只能是account或api"}
        if len(model) > 100 or any(ord(ch) < 32 for ch in model):
            return {"err": "LLM模型名称无效"}
        if prompt_profile not in ("base", "plus", "pro"):
            return {"err": "补偿器设计版本只能是base或plus"}
        if not np.isfinite(timeout_s) or not 30.0 <= timeout_s <= 900.0:
            return {"err": "LLM分析超时必须在30～900秒"}
        if (not np.isfinite(reuse_mse_threshold) or
                not 0.0 <= reuse_mse_threshold <= 10000.0):
            return {"err": "补偿器沿用MSE阈值必须在0～10000 deg²；0表示关闭"}
        auto_design = bool(data.get("auto_design", BR.get("codex_auto_enabled", True)))
        fast_mode = bool(data.get("fast_mode", BR.get("codex_fast_mode", False)))
        BR["llm_provider"] = provider
        BR["llm_auth_method"] = auth_method
        BR["codex_model"] = model
        BR["codex_prompt_profile"] = prompt_profile
        BR["codex_reasoning_effort"] = effort
        BR["codex_fast_mode"] = fast_mode
        BR["codex_auto_timeout_s"] = timeout_s
        BR["codex_reuse_mse_threshold_deg2"] = reuse_mse_threshold
        BR["codex_auto_enabled"] = auto_design
        warning = ("旧版minimal推理强度已自动迁移为low"
                   if effort_raw == "minimal" else None)
        try:
            blob = dict(GAINS_DEFAULT)
            if os.path.exists(GAINS_PATH):
                loaded = json.load(open(GAINS_PATH, encoding="utf-8"))
                if isinstance(loaded, dict):
                    blob.update(loaded)
            blob.update({
                "llm_provider": provider,
                "llm_auth_method": auth_method,
                "codex_model": model,
                "codex_prompt_profile": prompt_profile,
                "codex_reasoning_effort": effort,
                "codex_fast_mode": fast_mode,
                "codex_auto_timeout_s": timeout_s,
                "codex_reuse_mse_threshold_deg2": reuse_mse_threshold,
                "codex_auto_design": auto_design,
            })
            tmp = GAINS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as stream:
                json.dump(blob, stream, ensure_ascii=False, indent=2)
            os.replace(tmp, GAINS_PATH)
        except Exception as exc:
            warning = f"LLM配置已在本次运行生效，但保存失败：{exc}"
        result = _llm_config_status()
        if warning:
            result["warn"] = warning
        return result

    def _codex_replay_status():
        replay = BR.get("codex_replay")
        if not isinstance(replay, dict) or not replay.get("active"):
            return {"ok": 1, "active": False,
                    "captured_rounds": len(BR.get("codex_capture_rounds") or [])}
        rounds = replay.get("rounds") or []
        return {
            "ok": 1,
            "active": True,
            "path": str(replay.get("path", "")),
            "name": str(replay.get("name", "MPC+LLM replay")),
            "round_count": len(rounds),
            "duration_min": replay.get("duration_min", 0),
            "next_round": min(len(rounds), int(replay.get("index", 0))) + 1,
            "completed_rounds": min(len(rounds), int(replay.get("index", 0))),
            "task_hash": str(replay.get("task_hash", "")),
        }

    def _codex_capture_start():
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，不能重置10轮过程记录"}
        BR["codex_capture_active"] = True
        BR["codex_capture_rounds"] = []
        BR["codex_replay"] = None
        BR["codex_replay_active"] = False
        return {"ok": 1, "message": "已开始记录本次MPC+LLM轮次过程"}

    def _codex_replay_save(data):
        if BR.get("mode") == "mpc":
            return {"err": "请在完整轮次结束后保存10轮过程"}
        raw_path = str(data.get("path", "")).strip()
        if not raw_path:
            return {"err": "没有选择保存文件"}
        path = os.path.abspath(raw_path)
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            lap_count = max(1, min(100, int(data.get("lap_count", 10))))
        except (TypeError, ValueError):
            lap_count = 10
        captured = [dict(item) for item in (BR.get("codex_capture_rounds") or [])
                    if isinstance(item, dict) and item.get("path")]
        if len(captured) < lap_count:
            return {"err": (f"本次过程目前只有{len(captured)}个有效完整轮次，"
                            f"需要{lap_count}轮后才能保存")}
        selected = captured[-lap_count:]
        reports = []
        task_hash = ""
        for index, item in enumerate(selected, 1):
            report_path = os.path.abspath(str(item["path"]))
            with open(report_path, encoding="utf-8") as stream:
                report = json.load(stream)
            controller = report.get("tested_compensator")
            current_hash = str(report.get("task_hash", ""))
            if not isinstance(controller, dict):
                return {"err": f"第{index}轮报告缺少实际使用的补偿器"}
            if not task_hash:
                task_hash = current_hash
            if not current_hash or current_hash != task_hash:
                return {"err": "所选10轮不属于同一条参考轨迹，不能组成回放包"}
            reports.append({
                "round": index,
                "request_id": str(report.get("request_id", "")),
                "generation": int((report.get("audit") or {}).get(
                    "generation_before", report.get("active_generation", 0))),
                "tracking_mse_deg2": report.get("tracking_mse_deg2"),
                "controller": controller,
                "report": report,
            })
        points = data.get("points")
        if not isinstance(points, list) or not points:
            return {"err": "当前示教轨迹为空，不能建立可直接执行的回放包"}
        payload = {
            "schema": 1,
            "kind": "tro_mpc_llm_process_replay",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "name": str(data.get("name") or f"MPC+LLM {lap_count}-round replay"),
            "task_hash": task_hash,
            "sample_dt_s": reports[0]["report"].get("sample_dt_s", DT),
            "round_count": len(reports),
            "points": points,
            "auto_hidden_return_to_zero": bool(
                data.get("auto_hidden_return_to_zero", True)),
            "rounds": reports,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
        return {"ok": 1, "path": path, "round_count": len(reports),
                "message": f"已保存{len(reports)}轮MPC+LLM过程"}

    def _codex_replay_load(data):
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，请先停止再载入回放"}
        active_design = BR.get("codex_auto_thread")
        if active_design is not None and active_design.is_alive():
            return {"err": "LLM仍在完成上一轮设计，请等推理结束后再载入回放"}
        raw_path = str(data.get("path", "")).strip()
        if bool(data.get("comparison_preset")):
            path = os.path.join(os.path.dirname(CODEX_COMP_REQUEST_PATH),
                                "codex_compensator_request_a77196a8365c7603.json")
            with open(path, encoding="utf-8") as stream:
                selected_report = json.load(stream)
            payload = {"schema": 1, "kind": "tro_mpc_llm_process_replay",
                       "task_hash": selected_report["task_hash"],
                       "points": data.get("points"), "rounds": [],
                       "auto_hidden_return_to_zero": bool(
                           data.get("auto_hidden_return_to_zero", True))}
        else:
            if not raw_path:
                return {"err": "没有选择回放包"}
            path = os.path.abspath(raw_path)
            if not os.path.isfile(path):
                return {"err": "回放包不存在"}
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
        if (not isinstance(payload, dict) or int(payload.get("schema", -1)) != 1 or
                payload.get("kind") != "tro_mpc_llm_process_replay"):
            return {"err": "不是有效的MPC+LLM过程回放包"}
        points = payload.get("points")
        source_rounds = payload.get("rounds")
        # Operator-selected comparison schedule, explicitly distinct from a
        # chronological replay or an online learning experiment.
        from comparison_replay import build_comparison_rounds
        source_rounds = build_comparison_rounds(
            str(payload.get("task_hash", "")), os.path.dirname(CODEX_COMP_REQUEST_PATH),
            max(10, len(source_rounds) if isinstance(source_rounds, list) else 10),
            variant=int(data.get("comparison_variant", 1)))
        timed_load2 = bool(data.get("load2_30min"))
        if timed_load2 and int(data.get("comparison_variant", 1)) != 2:
            return {"err": "30分钟固定控制器测试仅用于Load 2"}
        if bool(data.get("modified_single_test")) or timed_load2:
            source_rounds = [dict(source_rounds[4], round=1, simulated_wait_after_s=0)]
        if not isinstance(points, list) or not points:
            return {"err": "回放包没有示教轨迹"}
        if not isinstance(source_rounds, list) or not source_rounds:
            return {"err": "回放包没有轮次控制器"}
        rounds = []
        for index, item in enumerate(source_rounds, 1):
            if not isinstance(item, dict):
                return {"err": f"回放包第{index}轮结构无效"}
            controller = item.get("controller")
            kind = str(controller.get("kind", "")) if isinstance(controller, dict) else ""
            if kind not in ("python_class", "legacy_parameters"):
                return {"err": f"回放包第{index}轮补偿器类型无效"}
            rounds.append({
                "round": index,
                "request_id": str(item.get("request_id", "")),
                "generation": max(0, int(item.get("generation", 0))),
                "tracking_mse_deg2": item.get("tracking_mse_deg2"),
                "controller": controller,
                "forbid_v3": bool(item.get("forbid_v3", False)),
                "default1_phi_v3_assist": bool(item.get("default1_phi_v3_assist", False)),
                "comparison_role": item.get("comparison_role", ""),
                "source_report": item.get("source_report", ""),
                "simulated_wait_after_s": item.get("simulated_wait_after_s", 0),
            })
        replay = {
            "active": True,
            "path": path,
            "name": "今日Default 1对比：纯MPC / MSE约100 / 第5轮拼接第4轮φ下降",
            "task_hash": str(payload.get("task_hash", "")),
            "points": points,
            "auto_hidden_return_to_zero": bool(
                payload.get("auto_hidden_return_to_zero", True)),
            "rounds": rounds,
            "index": 0,
            "session_id": f"replay-{time.time_ns()}",
        }
        BR["codex_replay"] = replay
        replay["comparison_variant"] = int(data.get("comparison_variant", 1))
        replay["duration_min"] = 30 if timed_load2 else 0
        replay["completed_laps"] = 0
        replay["name"] += f" / Load {int(data.get('comparison_variant', 1))}"
        if bool(data.get("modified_single_test")):
            replay["name"] = "Default 1修改版单轮测试（第5轮主体＋第4轮φ下降）"
        BR["codex_replay_active"] = False
        BR["codex_capture_active"] = False
        BR["codex_capture_rounds"] = []
        from codex_compensator import CodexCompensator
        CodexCompensator.reset_files(
            CODEX_REPLAY_STATE_PATH, CODEX_REPLAY_CANDIDATE_PATH,
            CODEX_REPLAY_REQUEST_PATH)
        BR["codex_auto_status"] = {
            "state": "replay_loaded",
            "message": f"已载入{len(rounds)}轮预设对比；第1轮禁V3，第2~4轮今日MSE约100，第5轮起使用第5轮主体+第4轮φ下降（待实测）；不调用LLM",
            "round_count": len(rounds),
        }
        if bool(data.get("modified_single_test")):
            BR["codex_auto_status"]["message"] = "已载入修改版单轮测试；保留V3辅助，不调用LLM"
        return {"ok": 1, **_codex_replay_status(), "points": points,
                "auto_hidden_return_to_zero": replay["auto_hidden_return_to_zero"]}

    def _codex_replay_reset():
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，不能重置回放"}
        replay = BR.get("codex_replay")
        if not isinstance(replay, dict) or not replay.get("active"):
            return {"err": "尚未载入MPC+LLM过程回放包"}
        replay["index"] = 0
        replay["completed_laps"] = 0
        replay.pop("deadline_monotonic", None)
        replay["session_id"] = f"replay-{time.time_ns()}"
        BR["codex_replay_active"] = False
        BR["codex_auto_status"] = {
            "state": "replay_loaded",
            "message": f"回放已复位，准备执行{len(replay.get('rounds') or [])}轮",
            "round_count": len(replay.get("rounds") or []),
        }
        return _codex_replay_status()

    def _codex_replay_clear():
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，不能退出回放"}
        BR["codex_replay"] = None
        BR["codex_replay_active"] = False
        BR["codex_auto_status"] = {
            "state": "idle", "message": "已退出过程回放；恢复正常LLM逐轮设计"}
        return {"ok": 1, "active": False}

    def _mpc_weights_apply(data):
        """Apply scalar MPC angle/velocity weights equally to all three axes."""
        if (BR.get("mode") == "mpc" and BR.get("codex_replay_active") and
                (BR.get("codex_replay") or {}).get("comparison_variant") == 2):
            return {"err": "Load 2执行期间请勿改权重：首轮使用普通MPC配置；后续固定wq=1000、wv=200"}
        old = BR["mpc_weights"]
        try:
            vals = {name: float(data.get(name, old[name])) for name in ("wq", "wv")}
        except (TypeError, ValueError):
            return {"err": "MPC Wq/Wv 必须是数字"}
        if (not all(np.isfinite(v) for v in vals.values()) or
                not 0.0 <= vals["wq"] <= 100000.0 or
                not 0.0 <= vals["wv"] <= 100000.0):
            return {"err": "MPC Wq/Wv 必须在 0～100000"}
        BR["mpc_weights"] = vals
        w_q = (vals["wq"],) * 3
        w_v = (vals["wv"],) * 3
        cfg = BR.get("mpc_cfg")
        if isinstance(cfg, dict):
            cfg["w_q"] = w_q
            cfg["w_v"] = w_v
        controller = BR.get("dampc")
        active = (BR.get("mode") == "mpc" and BR.get("ctrl") == "mpc" and
                  controller is not None)
        if active:
            controller.Wy = np.diag(list(w_q) + list(w_v))
            controller._Qb = np.kron(np.eye(controller.N), controller.Wy)
        warning = None
        try:
            blob = dict(GAINS_DEFAULT)
            if os.path.exists(GAINS_PATH):
                loaded = json.load(open(GAINS_PATH, encoding="utf-8"))
                if isinstance(loaded, dict):
                    blob.update(loaded)
            blob.update({"mpc_w_q": list(w_q), "mpc_w_v": list(w_v)})
            tmp = GAINS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            os.replace(tmp, GAINS_PATH)
        except Exception as exc:
            warning = f"参数已在本次运行生效，但保存失败：{exc}"
        return {"ok": 1, **vals, "active": active,
                **({"warn": warning} if warning else {})}

    def _pid_apply(data):
        """在线应用共享 PID 增益，并原子保存到 gains.json。"""
        old = BR["pid"]
        try:
            vals = {name: float(data.get(name, old[name]))
                    for name in ("kp", "ki", "kd", "kv", "vmax")}
        except (TypeError, ValueError):
            return {"err": "PID 参数必须是数字"}
        if any(not np.isfinite(vals[k]) or vals[k] < 0.0 or vals[k] > 100000.0
               for k in ("kp", "ki", "kd", "kv")) or not (1.0 <= vals["vmax"] <= 180.0):
            return {"err": "Kp、Ki、Kd、Kv 必须在 0～100000，Vmax 必须在 1～180°/s"}
        BR["pid"] = vals
        if BR.get("pi") is not None:
            BR["pi"].set_gains(vals["kp"], vals["ki"], vals["kd"], vals["kv"], vals["vmax"])
        warning = None
        try:
            blob = dict(GAINS_DEFAULT)
            if os.path.exists(GAINS_PATH):
                loaded = json.load(open(GAINS_PATH, encoding="utf-8"))
                if isinstance(loaded, dict): blob.update(loaded)
            blob.update({"pi_kp": vals["kp"], "pi_ki": vals["ki"], "pi_kd": vals["kd"],
                         "pi_kv": vals["kv"], "pi_vmax_deg": vals["vmax"]})
            tmp = GAINS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            os.replace(tmp, GAINS_PATH)
        except Exception as exc:
            warning = f"参数已在本次运行生效，但保存失败：{exc}"
        return {"ok": 1, **vals,
                "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "pi",
                **({"warn": warning} if warning else {})}

    def _pressure_pid_apply(data):
        """运行中应用保压内环参数，并保存到 gains.json。"""
        old = BR["pressure_pid"]
        try:
            vals = {name: float(data.get(name, old[name]))
                    for name in ("kp", "ki", "deadband")}
        except (TypeError, ValueError):
            return {"err": "气压环参数必须是数字"}
        if (not 0.0 <= vals["kp"] <= 1000.0 or
                not 0.0 <= vals["ki"] <= 1000.0 or
                not 0.0 <= vals["deadband"] <= 30.0 or
                not all(np.isfinite(v) for v in vals.values())):
            return {"err": "气压 Kp/Ki 必须在0～1000，死区必须在0～30 kPa"}
        BR["pressure_pid"] = vals
        warning = None
        try:
            blob = dict(GAINS_DEFAULT)
            if os.path.exists(GAINS_PATH):
                loaded = json.load(open(GAINS_PATH, encoding="utf-8"))
                if isinstance(loaded, dict): blob.update(loaded)
            blob.update({"pressure_hold_kp": vals["kp"],
                         "pressure_hold_ki": vals["ki"],
                         "pressure_hold_deadband_kpa": vals["deadband"]})
            tmp = GAINS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            os.replace(tmp, GAINS_PATH)
        except Exception as exc:
            warning = f"参数已在本次运行生效，但保存失败：{exc}"
        return {"ok": 1, **vals,
                "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "pi",
                **({"warn": warning} if warning else {})}

    def _muscle_pid_apply(data):
        """运行中应用肌肉长度 PI 参数，并保存到 gains.json。"""
        old = BR["muscle_pid"]
        try:
            vals = {name: float(data.get(name, old[name]))
                    for name in ("kp", "ki", "deadband")}
        except (TypeError, ValueError):
            return {"err": "肌肉长度 PI 参数必须是数字"}
        if (not 0.0 <= vals["kp"] <= 2000.0 or
                not 0.0 <= vals["ki"] <= 2000.0 or
                not 0.0 <= vals["deadband"] <= 20.0 or
                not all(np.isfinite(v) for v in vals.values())):
            return {"err": "长度 Kp/Ki 必须在0～2000，死区必须在0～20 mm"}
        BR["muscle_pid"] = vals
        if BR.get("muscle_pi") is not None:
            BR["muscle_pi"].set_gains(vals["kp"], vals["ki"], vals["deadband"])
        warning = None
        try:
            blob = dict(GAINS_DEFAULT)
            if os.path.exists(GAINS_PATH):
                loaded = json.load(open(GAINS_PATH, encoding="utf-8"))
                if isinstance(loaded, dict): blob.update(loaded)
            blob.update({"muscle_pi_kp": vals["kp"], "muscle_pi_ki": vals["ki"],
                         "muscle_pi_deadband_mm": vals["deadband"]})
            tmp = GAINS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            os.replace(tmp, GAINS_PATH)
        except Exception as exc:
            warning = f"参数已在本次运行生效，但保存失败：{exc}"
        return {"ok": 1, **vals,
                "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "musclepi",
                **({"warn": warning} if warning else {})}

    def _inverse_state_save():
        state = BR.get("inverse_state")
        if not isinstance(state, dict):
            return
        state_path = BR.get("inverse_state_path", NEW_MPC_FEL_NN_STATE_PATH)
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, state_path)

    def _inverse_replay_path():
        state_path = BR.get("inverse_state_path", NEW_MPC_FEL_NN_STATE_PATH)
        return os.path.splitext(state_path)[0] + "_replay.npz"

    def _inverse_replay_save():
        path = _inverse_replay_path()
        replay = BR.get("inverse_replay", [])
        if not replay:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return
        lengths = np.asarray([len(item[0]) for item in replay], np.int64)
        features = np.concatenate([item[0] for item in replay]).astype(np.float32)
        targets = np.concatenate([item[1] for item in replay]).astype(np.float32)
        masks = np.concatenate([item[2] for item in replay]).astype(bool)
        corrections = np.concatenate([
            item[3] if len(item) > 3 else np.zeros_like(item[1])
            for item in replay]).astype(np.float32)
        tmp = path + ".tmp"
        with open(tmp, "wb") as stream:
            np.savez_compressed(stream, lengths=lengths, features=features,
                                targets=targets, masks=masks,
                                corrections=corrections)
        os.replace(tmp, path)

    def _inverse_replay_load():
        path = _inverse_replay_path()
        if not os.path.exists(path):
            return []
        try:
            with np.load(path, allow_pickle=False) as blob:
                lengths = np.asarray(blob["lengths"], np.int64)
                features = np.asarray(blob["features"], np.float32)
                targets = np.asarray(blob["targets"], np.float32)
                masks = np.asarray(blob["masks"], bool)
                if "corrections" not in blob.files:
                    raise ValueError("legacy inverse replay has no MPC correction")
                corrections = np.asarray(blob["corrections"], np.float32)
            if (lengths.ndim != 1 or len(lengths) >= SHADOW_REPLAY_ROUNDS or
                    np.any(lengths <= 0) or int(np.sum(lengths)) != len(features) or
                    features.shape != (len(features), FEL_NF) or
                    targets.shape != (len(features), NCH) or masks.shape != targets.shape or
                    corrections.shape != targets.shape):
                raise ValueError("invalid inverse replay shape")
            replay, offset = [], 0
            for count in lengths:
                stop = offset + int(count)
                replay.append((features[offset:stop].copy(), targets[offset:stop].copy(),
                               masks[offset:stop].copy(), corrections[offset:stop].copy()))
                offset = stop
            return replay
        except Exception:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return []

    def _inverse_model_save(model, path):
        torch = _fel_torch()
        tmp = path + ".tmp"
        torch.save({"schema": FEL_SCHEMA, "state_dict": model.state_dict()}, tmp)
        os.replace(tmp, path)

    def _new_inverse_prepare():
        """Load an inverse shadow isolated from the active FEL network."""
        active_path = BR.get("inverse_active_path", NEW_MPC_FEL_NN_PATH)
        shadow_path = BR.get("inverse_shadow_path", NEW_MPC_FEL_NN_SHADOW_PATH)
        state_path = BR.get("inverse_state_path", NEW_MPC_FEL_NN_STATE_PATH)
        if BR.get("nn") is None:
            BR["nn"] = _fel_new()
        if not os.path.exists(active_path):
            _inverse_model_save(BR["nn"], active_path)
        BR["inverse_shadow"] = _fel_load(
            shadow_path,
            bootstrap_path=active_path)
        if BR["inverse_shadow"] is None:
            BR["inverse_shadow"] = _fel_new()
            BR["inverse_shadow"].load_state_dict(BR["nn"].state_dict())
        default = {"schema": 1, "generation": 0, "consecutive_wins": 0,
                   "rounds_seen": 0, "tracking_mse_history_deg2": [],
                   "canary": {"active": False}, "history": []}
        try:
            loaded = json.load(open(state_path, encoding="utf-8"))
            if isinstance(loaded, dict) and int(loaded.get("schema", 0)) == 1:
                default.update(loaded)
        except Exception:
            pass
        BR["inverse_state"] = default
        BR["inverse_replay"] = _inverse_replay_load()
        BR["inverse_rounds_seen"] = int(default.get("rounds_seen", 0))
        _inverse_state_save()

    def _inverse_validation_rmse(model, features, targets, masks, row_mask):
        chosen = np.asarray(row_mask, bool)
        if not np.any(chosen):
            return float("inf")
        pred = _fel_predict(model, features[chosen])
        valid = masks[chosen]
        if not np.any(valid):
            return float("inf")
        return float(np.sqrt(np.mean((pred[valid]-targets[chosen][valid])**2)))

    def _inverse_tracking_append(value):
        """Keep only real, complete-trajectory tracking scores for inverse audit."""
        if value is None or not np.isfinite(float(value)):
            return
        state = BR["inverse_state"]
        history = list(state.get("tracking_mse_history_deg2", []))
        history.append(float(value))
        state["tracking_mse_history_deg2"] = history[-12:]

    def _inverse_assess_canary(tracking_mse_deg2):
        """Accept/reject the inverse candidate using actual response-reference MSE."""
        state = BR["inverse_state"]
        canary = dict(state.get("canary") or {})
        if not canary.get("active"):
            _inverse_tracking_append(tracking_mse_deg2)
            return None
        if tracking_mse_deg2 is None or not np.isfinite(float(tracking_mse_deg2)):
            return {"checked": False, "reason": "缺少完整轨迹真实跟踪MSE"}
        baseline = float(canary.get("baseline_mse_deg2", float("inf")))
        candidate = float(tracking_mse_deg2)
        accepted = bool(np.isfinite(baseline) and candidate <= baseline)
        info = {"checked": True, "accepted": accepted,
                "baseline_mse_deg2": baseline,
                "candidate_mse_deg2": candidate,
                "ratio": candidate/max(baseline, 1.0e-12)}
        if accepted:
            state["canary"] = {"active": False}
            _inverse_tracking_append(candidate)
            print(f"#   ✓ 影子逆模型实机审核通过：真实跟踪MSE "
                  f"{baseline:.2f}→{candidate:.2f} deg²，保留候选逆模型")
        else:
            previous_path = BR.get("inverse_previous_path", NEW_MPC_FEL_NN_PREVIOUS_PATH)
            active_path = BR.get("inverse_active_path", NEW_MPC_FEL_NN_PATH)
            shadow_path = BR.get("inverse_shadow_path", NEW_MPC_FEL_NN_SHADOW_PATH)
            restored = _fel_load(previous_path)
            if restored is None:
                info["rollback_error"] = "previous inverse model unavailable"
                print("#   ⚠ 影子逆模型实机审核失败，但旧逆模型文件不可用，暂不回滚")
            else:
                BR["nn"].load_state_dict(restored.state_dict())
                BR["inverse_shadow"].load_state_dict(restored.state_dict())
                _inverse_model_save(BR["nn"], active_path)
                _inverse_model_save(BR["inverse_shadow"], shadow_path)
                state["generation"] = int(canary.get(
                    "generation_before", max(0, int(state.get("generation", 1))-1)))
                state["canary"] = {"active": False}
                print(f"#   ⚠ 影子逆模型实机审核失败并回滚：真实跟踪MSE "
                      f"{baseline:.2f}→{candidate:.2f} deg²")
        history = list(state.get("history", []))
        history.append({"event": "real_tracking_canary", **info})
        state["history"] = history[-100:]
        _inverse_state_save()
        return info

    @staticmethod
    def _inverse_rollout_starts(round_lengths, horizon, validation=False):
        """Return contiguous starts without ever crossing a lap boundary."""
        starts, offset = [], 0
        for count in [int(v) for v in round_lengths]:
            # Each stored inverse feature at real tick t already contains r(t+1).
            last = max(0, count-int(horizon)+1)
            for local in range(last):
                if (local % 5 == 0) == bool(validation):
                    starts.append(offset+local)
            offset += count
        return np.asarray(starts, dtype=np.int64)

    def _inverse_model_rollout_loss(model, features, corrections, starts,
                                    horizon, forward_net):
        """Differentiable inverse -> frozen forward-model trajectory loss."""
        torch = _fel_torch()
        if forward_net is None or len(starts) == 0:
            zero = torch.zeros((), dtype=torch.float32)
            return zero, zero, zero
        Xall = torch.as_tensor(np.asarray(features, np.float32))
        Call = torch.as_tensor(np.asarray(corrections, np.float32))
        start = torch.as_tensor(np.asarray(starts, np.int64), dtype=torch.long)
        first = Xall[start]
        # True forward-model state is [q(rad), delta_q_per_tick(rad), P(kPa)].
        state = torch.cat((first[:, 10:13]*0.5,
                           first[:, 13:16]*DT,
                           first[:, 19:25]*P_MAX_KPA), dim=1)
        u_prev = first[:, 25:31]*max(FEL_FF_MAX, 1.0)
        angle_sum = torch.zeros((), dtype=torch.float32)
        velocity_sum = torch.zeros((), dtype=torch.float32)
        smooth_sum = torch.zeros((), dtype=torch.float32)
        for step in range(int(horizon)):
            template = Xall[start+step].clone()
            reference = template[:, 1:4]*0.5
            reference_velocity = template[:, 4:7]
            # During rollout the reference schedule remains recorded, while
            # measured-state fields are replaced by the model-predicted state.
            template[:, 10:13] = state[:, :3]/0.5
            template[:, 13:16] = state[:, 3:6]/DT
            template[:, 16:19] = (reference-state[:, :3])/0.5
            template[:, 19:25] = state[:, 6:]/max(P_MAX_KPA, 1.0)
            template[:, 25:31] = u_prev/max(FEL_FF_MAX, 1.0)
            raw = torch.cat([head(template) for head in model], dim=1)
            u_inverse = _fel_torch_project(raw.clamp(0.0, 1.0))*FEL_FF_MAX
            # The logged MPC feedback is a fixed (stop-gradient) correction.
            # Gradients therefore pass only through the inverse-model output.
            u_total_n = (u_inverse+Call[start+step]).clamp(
                0.0, FEL_FF_MAX)/max(FEL_FF_MAX, 1.0)
            u_total = _fel_torch_project(u_total_n)*FEL_FF_MAX
            locked = (template[:, 31:37] >= 0.99).to(u_total.dtype)
            u_total = u_total*locked
            predicted = forward_net(state, u_total)
            # Keep long rollout numerically inside the fitted physical region.
            state = torch.cat((predicted[:, :3].clamp(-math.pi, math.pi),
                               predicted[:, 3:6].clamp(-0.5, 0.5),
                               predicted[:, 6:].clamp(0.0, P_MAX_KPA)), dim=1)
            angle_sum = angle_sum + torch.mean(torch.sum(
                ((state[:, :3]-reference)/0.5)**2, dim=1))
            velocity_sum = velocity_sum + torch.mean(torch.sum(
                (state[:, 3:6]/DT-reference_velocity)**2, dim=1))
            smooth_sum = smooth_sum + torch.mean(
                ((u_total-u_prev)/max(FEL_FF_MAX, 1.0))**2)
            u_prev = u_total
        scale = 1.0/max(1, int(horizon))
        return angle_sum*scale, velocity_sum*scale, smooth_sum*scale

    def _inverse_rollout_mse_deg2(model, features, corrections, round_lengths,
                                  forward_net, validation=True):
        """Three-axis-sum trajectory MSE used to pre-screen inverse candidates."""
        horizon = min(FEL_TRACK_HORIZON, max(1, min(map(int, round_lengths))))
        starts = _inverse_rollout_starts(round_lengths, horizon, validation)
        if len(starts) == 0:
            return float("inf")
        # Fixed, evenly distributed subset keeps every-lap training responsive.
        if len(starts) > 32:
            starts = starts[np.linspace(0, len(starts)-1, 32, dtype=int)]
        model.eval()
        with _fel_torch().no_grad():
            angle, _, _ = _inverse_model_rollout_loss(
                model, features, corrections, starts, horizon, forward_net)
        # Training angle is sum((error/0.5rad)^2); restore deg^2.
        return float(angle)*(math.degrees(0.5)**2)

    def _inverse_shadow_optimize(features, corrections, round_lengths, forward_net,
                                 epochs, batch_size, patience=None):
        """Train only the inverse shadow; validation rows never enter Adam."""
        torch = _fel_torch()
        model = BR.get("inverse_shadow")
        if model is None:
            return float("inf"), 0
        if forward_net is not None:
            forward_net.eval()
            for parameter in forward_net.parameters():
                parameter.requires_grad_(False)
        horizon = min(FEL_TRACK_HORIZON, max(1, min(map(int, round_lengths))))
        trajectory_starts = _inverse_rollout_starts(
            round_lengths, horizon, validation=False)
        if forward_net is None or len(trajectory_starts) == 0:
            return float("inf"), 0
        # Plain Adam: no implicit weight decay.  The inverse objective must
        # contain only the two explicitly requested tracking terms.
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        batch_size = max(1, min(int(batch_size), len(trajectory_starts)))
        generator = torch.Generator().manual_seed(
            20260906 + int(BR.get("inverse_rounds_seen", 0)))
        best_loss, best_state, stale = float("inf"), None, 0
        final_loss, epochs_run = float("inf"), 0
        model.train()
        for epoch in range(int(epochs)):
            pick = torch.randperm(
                len(trajectory_starts), generator=generator)[:batch_size].numpy()
            selected = trajectory_starts[pick]
            loss_track, loss_velocity, _ = _inverse_model_rollout_loss(
                model, features, corrections, selected, horizon, forward_net)
            # Exactly two terms: one-step angle and velocity tracking.
            loss = (FEL_INVERSE_TRAJECTORY_WEIGHT*loss_track +
                    0.02*FEL_INVERSE_TRAJECTORY_WEIGHT*loss_velocity)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
            epochs_run = epoch + 1
            if patience is not None:
                current = _inverse_rollout_mse_deg2(
                    model, features, corrections, round_lengths,
                    forward_net, validation=True)
                if current < best_loss - 1.0e-4:
                    best_loss = current
                    best_state = {key: value.detach().clone()
                                  for key, value in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= int(patience):
                    break
                model.train()
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return final_loss, epochs_run

    def _new_inverse_round_update(phi, err, mask, tracking_mse_deg2=None):
        """Train inverse compensation through a frozen forward model every lap.

        The real MPC correction recorded in the lap is stop-gradient.  Only the
        inverse network receives gradients from the one-step reference tracking
        loss.  Every third lap additionally performs a longer replay update.
        """
        if BR.get("inverse_shadow") is None or BR.get("nn") is None:
            return {"samples": 0, "trained": False, "promoted": False}
        canary_info = _inverse_assess_canary(tracking_mse_deg2)
        X = np.asarray(phi, np.float32)
        E = np.asarray(err, np.float32)
        M = np.asarray(mask, bool)
        if (len(X) < 20 or X.shape != (len(X), FEL_NF) or
                E.shape != (len(X), NCH) or M.shape != E.shape or not np.any(M)):
            return {"samples": len(X), "trained": False, "promoted": False}
        beta = FEL_BLEND_MPC
        active_output = _fel_predict(BR["nn"], X)
        target = _antagonist_project(
            np.clip(active_output + beta*E, 0.0, FEL_FF_MAX)).astype(np.float32)
        BR["inverse_rounds_seen"] = int(BR.get("inverse_rounds_seen", 0)) + 1
        BR["inverse_state"]["rounds_seen"] = BR["inverse_rounds_seen"]
        replay = BR.setdefault("inverse_replay", [])
        # E is the MPC correction that was actually sent beside the inverse
        # output.  It stays fixed during inverse backpropagation.
        replay.append((X.copy(), target.copy(), M.copy(), E.copy()))
        del replay[:-SHADOW_REPLAY_ROUNDS]
        _inverse_replay_save()

        # 第1/2轮也更新100 epochs；第3轮用三轮总数据做500 epochs consolidation。
        consolidated = len(replay) == SHADOW_REPLAY_ROUNDS
        X = np.concatenate([item[0] for item in replay], axis=0)
        target = np.concatenate([item[1] for item in replay], axis=0)
        M = np.concatenate([item[2] for item in replay], axis=0)
        corrections = np.concatenate([
            item[3] if len(item) > 3 else np.zeros_like(item[1])
            for item in replay], axis=0)
        round_lengths = [len(item[0]) for item in replay]
        # 每轮都按相同相位留出1/5验证数据，验证行不进入Adam。
        val_parts = []
        for item in replay:
            local = np.arange(len(item[0]))
            val_parts.append(local % 5 == 0)
        val_mask = np.concatenate(val_parts)
        train_mask = ~val_mask

        # 若本轮顺模型刚晋级，逆模型必须仍用晋级前的稳定真正模型训练，
        # 不能利用一个尚未经过下一轮实机审核的候选顺模型。
        forward_net = getattr(BR.get("dm"), "net", None)
        if BR.get("forward_promoted_this_round", False):
            learner = BR.get("forward_learner")
            previous_path = getattr(learner, "previous_path", None)
            if previous_path and os.path.exists(previous_path):
                try:
                    forward_net = _fel_torch().jit.load(
                        str(previous_path), map_location="cpu").eval()
                except Exception as exc:
                    print(f"#   影子逆模型：旧真正顺模型读取失败，取消本轮逆模型训练：{exc}")
                    return {"samples": int(len(X)), "trained": False,
                            "promoted": False, "reason": str(exc)}
        if forward_net is None:
            return {"samples": int(len(X)), "trained": False,
                    "promoted": False,
                    "reason": "当前真正顺模型不支持PyTorch梯度"}

        active_rmse = _inverse_validation_rmse(
            BR["nn"], X, target, M, val_mask)
        before_rmse = _inverse_validation_rmse(
            BR["inverse_shadow"], X, target, M, val_mask)
        active_track_mse = _inverse_rollout_mse_deg2(
            BR["nn"], X, corrections, round_lengths, forward_net, validation=True)
        before_track_mse = _inverse_rollout_mse_deg2(
            BR["inverse_shadow"], X, corrections, round_lengths,
            forward_net, validation=True)
        epochs = 500 if consolidated else 100
        batch_size = 64 if consolidated else max(1, int(np.sum(train_mask)))
        patience = 50 if consolidated else 20
        _, replay_epochs = _inverse_shadow_optimize(
            X, corrections, round_lengths, forward_net,
            epochs, batch_size, patience)
        epochs_run = replay_epochs
        shadow_rmse = _inverse_validation_rmse(
            BR["inverse_shadow"], X, target, M, val_mask)
        shadow_track_mse = _inverse_rollout_mse_deg2(
            BR["inverse_shadow"], X, corrections, round_lengths,
            forward_net, validation=True)
        finite = (np.isfinite(active_track_mse) and
                  np.isfinite(shadow_track_mse) and
                  np.isfinite(active_rmse) and np.isfinite(shadow_rmse))
        # 晋级先看冻结真正顺模型中的reference跟踪；电压模仿只是安全护栏。
        task_better = shadow_track_mse <= 0.98*active_track_mse
        voltage_safe = shadow_rmse <= 1.20*max(active_rmse, 1.0)
        passed = bool(finite and task_better and voltage_safe)
        state = BR["inverse_state"]
        wins = int(state.get("consecutive_wins", 0)) + 1 if passed else 0
        state["consecutive_wins"] = wins
        promotion_ready = passed and wins >= 1
        # Do not start forward and inverse hardware canaries on the same lap;
        # otherwise a tracking change cannot be attributed to either model.
        promotion_deferred = bool(promotion_ready and
                                  BR.get("forward_promoted_this_round", False))
        promotion_ready = bool(promotion_ready and not promotion_deferred)
        promoted = bool(promotion_ready)
        generation_before = int(state.get("generation", 0))
        if promoted:
            active_path = BR.get("inverse_active_path", NEW_MPC_FEL_NN_PATH)
            shadow_path = BR.get("inverse_shadow_path", NEW_MPC_FEL_NN_SHADOW_PATH)
            previous_path = BR.get("inverse_previous_path", NEW_MPC_FEL_NN_PREVIOUS_PATH)
            if os.path.exists(active_path):
                shutil.copy2(active_path, previous_path)
            BR["nn"].load_state_dict(BR["inverse_shadow"].state_dict())
            _inverse_model_save(BR["nn"], active_path)
            _inverse_model_save(BR["inverse_shadow"], shadow_path)
            state["generation"] = generation_before + 1
            state["consecutive_wins"] = 0
            track_history = list(state.get("tracking_mse_history_deg2", []))
            baseline_values = track_history[-3:]
            state["canary"] = {
                "active": True,
                "generation_before": generation_before,
                "candidate_generation": generation_before + 1,
                "baseline_mse_deg2": (float(np.mean(baseline_values))
                                      if baseline_values else float("inf")),
            }
        else:
            _inverse_model_save(
                BR["inverse_shadow"],
                BR.get("inverse_shadow_path", NEW_MPC_FEL_NN_SHADOW_PATH))
        result = {
            "samples": int(len(X)), "trained": True, "epochs": epochs_run,
            "replay_rounds": len(replay),
            "active_track_mse_deg2": active_track_mse,
            "shadow_before_track_mse_deg2": before_track_mse,
            "shadow_track_mse_deg2": shadow_track_mse,
            "active_rmse_mv": active_rmse, "shadow_before_rmse_mv": before_rmse,
            "shadow_rmse_mv": shadow_rmse, "passed": passed,
            "consecutive_wins": wins, "promoted": bool(promoted),
            "promotion_deferred": promotion_deferred,
            "consolidated": consolidated, "replay_epochs": replay_epochs,
            "generation_before": generation_before,
            "generation_after": int(state.get("generation", 0)),
        }
        if canary_info:
            result["canary"] = canary_info
        history = list(state.get("history", [])); history.append(result)
        state["history"] = history[-100:]
        _inverse_state_save()
        # 三轮长训练后清空；下一轮重新开始一个新的三轮学习块。
        if consolidated:
            replay.clear()
        _inverse_replay_save()
        print(f"#   影子逆模型：冻结顺模型单步跟踪MSE "
              f"{active_track_mse:.3f}→{shadow_track_mse:.3f} deg²，"
              f"电压偏离诊断(不进loss) {active_rmse:.1f}→{shadow_rmse:.1f}mV；"
              f"{len(round_lengths)}轮数据训练{replay_epochs} epochs，审核 {wins}/1" +
              ("（三轮总数据强化）" if consolidated else "") +
              ("；候选逆模型已临时上线，下一轮按真实跟踪MSE审核" if promoted else
               "；与顺模型审核冲突，本次不同时上线" if promotion_deferred else ""))
        return result

    def _fel_round_update(tracking_mse_deg2=None):
        """圈末更新逆模型；每轮短训、每三轮合并长训，并剔除预锁样本。"""
        phi = BR.pop("felPhi", [])
        err = BR.pop("felErr", [])
        mask = BR.pop("felMask", [])
        BR["felPhi"], BR["felErr"], BR["felMask"] = [], [], []
        nu = 0
        if not BR.get("fel", True): return nu
        if BR.get("models_frozen", False): return nu
        beta = FEL_BLEND_PI if BR.get("ctrl") == "pi" else FEL_BLEND_MPC
        if BR.get("nn") is None: return 0
        if BR.get("new_mpc_fel"):
            result = _new_inverse_round_update(
                phi, err, mask, tracking_mse_deg2=tracking_mse_deg2)
            BR["inverse_last_result"] = result
            return int(result.get("samples", 0)) if result.get("trained") else 0
        return _fel_train(BR["nn"], phi, err, mask, beta, BR.get("dm"))

    def _dm_round_update():
        """MPC+FEL 的 round 边界更新 DataModel；纯 MPC 必须保持冻结。"""
        if not BR.get("fel", False) or BR.get("ctrl") == "pi": return 0
        X, Y = BR.get("dmX", []), BR.get("dmY", [])
        dm = BR.get("dm")
        # The deployed pure-data NN is trained offline from deliberately
        # excited runs.  Do not corrupt it with one closed-loop trajectory.
        if dm is None or not hasattr(dm, "fit") or len(X) < 30: return 0
        r = dm.fit(X, Y)
        blocks = [(np.array(a), np.array(b)) for a, b in BR.get("dmBlk", []) if len(a) >= DataModel.PNEU_WIN + 1]
        if blocks: dm.fit_pneu(blocks)
        dm.src = "round"; dm.save()
        print(f"#   数据模型重拟合：{len(X)} 样本，1拍RMSE {r:.2f}°")
        _dm_print(dm)
        return len(X)

    def _new_forward_rebuild_controller():
        """Reload a promoted active model into MPC; only called between rounds."""
        learner = BR.get("forward_learner")
        cfg = BR.get("mpc_cfg")
        if learner is None or not cfg:
            return
        architecture = str(BR.get("forward_arch", "mlp"))
        if architecture in ("history_mlp", "gru"):
            dm = SequencePureDataModel(
                architecture,
                model_path=str(learner.active_path),
                meta_path=BR.get("forward_meta_path",
                                 SEQUENCE_MODEL_FILES[architecture][1]))
            cfg["solver"] = "ltv"
        else:
            dm = PureDataModel(
                model_path=str(learner.active_path),
                meta_path=BR.get("forward_meta_path", PURE_DM_META_PATH))
        BR["dm"] = dm
        BR["dampc"] = DAMPC(dm, **cfg)
        BR["xd"][:] = 0.0
        BR["last_u_cmd"][:] = 0.0

    def _new_forward_round_update(tracking_mse_deg2=None):
        """Train/validate the shadow forward model and promote only if safe."""
        BR["forward_promoted_this_round"] = False
        if not BR.get("shadow_forward") or BR.get("forward_learner") is None:
            return None
        try:
            result = BR["forward_learner"].train_round(
                tracking_mse_deg2=tracking_mse_deg2)
            canary = result.get("canary") or {}
            if result.get("rolled_back"):
                _new_forward_rebuild_controller()
                print(f"#   ⚠ 候选顺模型实机审核失败并自动回滚：真实跟踪MSE "
                      f"{canary.get('baseline_mse_deg2', float('nan')):.2f}→"
                      f"{canary.get('candidate_mse_deg2', float('nan')):.2f} deg²")
            elif canary.get("checked") and canary.get("accepted"):
                print(f"#   ✓ 候选顺模型实机审核通过：真实跟踪MSE "
                      f"{canary.get('baseline_mse_deg2', float('nan')):.2f}→"
                      f"{canary.get('candidate_mse_deg2', float('nan')):.2f} deg²")
            if not result.get("trained"):
                print(f"#   影子顺模型未训练：{result.get('samples', 0)}样本，"
                      f"{result.get('reason', '条件不足')}")
                return result
            active = result["active"]; shadow = result["shadow"]
            print(f"#   影子顺模型：全程有效动态角度MSE "
                  f"{active['angle_mse_deg2']:.4f}→{shadow['angle_mse_deg2']:.4f} deg²"
                  f"（活动模型→影子模型，三个轴合计，"
                  f"{active.get('angle_samples', 0)}/"
                  f"{result.get('validation_samples', 0)}个验证点）；"
                  f"更新审核 {result.get('consecutive_wins', 0)}/1")
            if result.get("promoted"):
                BR["forward_promoted_this_round"] = True
                _new_forward_rebuild_controller()
                print(f"#   ✓ 影子顺模型预测审核通过，下一轮临时实机验证；"
                      f"旧模型真实跟踪MSE基准="
                      f"{result.get('tracking_baseline_mse_deg2', float('nan')):.2f} deg²，"
                      f"候选 generation={result.get('generation_after', 0)}")
            elif result.get("shadow_reset"):
                print("#   ⚠ 影子顺模型连续恶化，已恢复为活动模型副本")
            return result
        except Exception as exc:
            # Failure of learning must never alter or stop the active controller.
            print(f"#   ⚠ 影子顺模型训练失败，活动顺模型保持不变：{exc}")
            try:
                BR["forward_learner"].abandon_round()
            except Exception:
                pass
            return {"trained": False, "promoted": False, "reason": str(exc)}

    def _save_w():
        """FEL MLP 原子落盘（tmp+replace）。"""
        if (not BR.get("fel", False) or BR.get("nn") is None or
                BR.get("models_frozen", False)): return
        try:
            torch = _fel_torch()
            save_path = BR.get("fel_path", FEL_NN_PATH)
            tmp = save_path + ".tmp"
            torch.save({"schema": FEL_SCHEMA, "state_dict": BR["nn"].state_dict()}, tmp)
            os.replace(tmp, save_path)
        except Exception as e:
            print(f"# FEL MLP 保存失败：{e}")

    def _ilc_buffers_reset():
        BR["ilc_u"], BR["ilc_ff"], BR["ilc_mask"] = [], [], []

    def _ilc_round_update(tracking_mse_deg2):
        """Use one complete hardware lap to audit/update the task-indexed ILC."""
        learner = BR.get("ilc")
        if not BR.get("ilc_mpc") or learner is None:
            return None
        total_u = np.asarray(BR.get("ilc_u", []), dtype=float)
        used_ff = np.asarray(BR.get("ilc_ff", []), dtype=float)
        valid = np.asarray(BR.get("ilc_mask", []), dtype=bool)
        expected = len(BR.get("ref", []))
        if (total_u.shape != (expected, NCH) or used_ff.shape != total_u.shape or
                valid.shape != total_u.shape):
            _ilc_buffers_reset()
            print(f"# 约束ILC：本轮样本不完整({len(total_u)}/{expected})，不学习")
            return {"reason": "incomplete", "samples": len(total_u)}
        result = learner.finish_round(total_u, used_ff, valid, tracking_mse_deg2)
        _ilc_buffers_reset()
        if result.get("rolled_back"):
            print(f"# 约束ILC：候选实机MSE {result.get('baseline_mse_deg2', float('nan')):.2f}"
                  f"→{result.get('tracking_mse_deg2', float('nan')):.2f} deg²，变差，已回滚；"
                  f"generation={result.get('generation_after', 0)}")
        else:
            audit = ""
            if result.get("canary_checked"):
                audit = (f"；实机审核 {result.get('baseline_mse_deg2', float('nan')):.2f}"
                         f"→{result.get('tracking_mse_deg2', float('nan')):.2f} deg²通过")
            proposed = ("；下一轮测试新候选" if result.get("proposed") else
                        "；没有足够的有效反馈更新")
            print(f"# 约束ILC：真实跟踪MSE {result.get('tracking_mse_deg2', float('nan')):.2f} deg²"
                  f"{audit}{proposed}；generation={result.get('generation_after', 0)}，"
                  f"前馈峰值={result.get('ff_peak_mv', 0.0):.0f}mV")
        return result

    def _ilc_abort():
        learner = BR.get("ilc")
        if not BR.get("ilc_mpc") or learner is None:
            return
        answer = learner.abort_round()
        _ilc_buffers_reset()
        if answer.get("rolled_back"):
            print(f"# 约束ILC：候选轮被中止，已回滚到 generation={answer.get('generation', 0)}")

    def _start_codex_auto_design(result):
        """Start a read-only Codex CLI design job after the hardware is stopped."""
        if not BR.get("codex_auto_enabled", True):
            BR["codex_auto_status"] = {
                "state": "disabled", "message": "自动读取已在gains.json中关闭"}
            return
        if not result.get("valid_for_design", False):
            postmortem = result.get("postmortem") or {}
            reason = str(postmortem.get("decision_reason") or
                         "本轮数据审核未通过")
            BR["codex_auto_status"] = {
                "state": "retry",
                "message": (f"{reason}；本轮不更新补偿器，自动重跑稳定版，"
                            "取得有效数据后继续设计"),
                "request_id": result.get("request_id"),
                "generation": int(result.get("generation_after", 0)),
                "invalid_reasons": list(result.get("invalid_reasons") or []),
                "completed_lap": int(result.get("completed_lap_index", 0)),
                "retry_same_generation": True}
            print(f"# Codex补偿器：{reason}；跳过本轮脏数据，自动重跑稳定版后继续设计")
            return
        reuse_threshold = max(0.0, float(
            BR.get("codex_reuse_mse_threshold_deg2", 15.0)))
        tracking_mse = float(result.get("tracking_mse_deg2", float("inf")))
        rolled_back = bool(result.get("rolled_back", False))
        # After a failed canary, finish_round has already restored the previous
        # stable compensator.  Grade that restored stable version with its
        # recorded baseline MSE; never mistake the rejected canary score for
        # the quality of the compensator that will actually be reused.
        stable_mse = float(result.get("baseline_mse_deg2", float("inf"))) \
            if rolled_back else tracking_mse
        if (reuse_threshold > 0.0 and np.isfinite(stable_mse) and
                stable_mse <= reuse_threshold):
            reuse_prefix = "候选已回退；" if rolled_back else ""
            BR["codex_auto_status"] = {
                "state": "reused",
                "message": (f"{reuse_prefix}稳定版MSE {stable_mse:.3f}≤"
                            f"{reuse_threshold:.3f} deg²，"
                            "沿用当前已验证补偿器"),
                "tracking_mse_deg2": tracking_mse,
                "stable_mse_deg2": stable_mse,
                "reuse_mse_threshold_deg2": reuse_threshold,
                "generation": int(result.get("generation_after", 0)),
                "request_id": result.get("request_id")}
            print(f"# Codex补偿器：{reuse_prefix}稳定版真实MSE {stable_mse:.3f}≤"
                  f"{reuse_threshold:.3f} deg²，达到沿用阈值；跳过LLM改版")
            return
        active = BR.get("codex_auto_thread")
        if active is not None and active.is_alive():
            BR["codex_auto_status"] = {
                "state": "error", "message": "上一次Codex自动分析仍在运行"}
            return
        request_path = str(result.get("request_path") or CODEX_COMP_REQUEST_PATH)
        request_id = str(result.get("request_id") or "")
        has_prior_design = False
        try:
            with open(request_path, encoding="utf-8") as stream:
                has_prior_design = bool((json.load(stream).get("last_design") or {}))
        except Exception:
            has_prior_design = bool(int(result.get("generation_after", 0)) > 0)
        design_provider = str(BR.get("llm_provider", "gpt")).strip().lower()
        design_auth_method = str(BR.get("llm_auth_method", "account")).strip().lower()
        design_model = str(BR.get("codex_model", "")).strip()
        design_prompt_profile = str(
            BR.get("codex_prompt_profile", "base")).strip().lower()
        design_effort = str(BR.get("codex_reasoning_effort", "medium")).strip().lower()
        design_fast_mode = bool(BR.get("codex_fast_mode", False))
        design_timeout_s = float(BR.get("codex_auto_timeout_s", 300.0))
        completed_lap = max(1, int(result.get("completed_lap_index", 1)))
        next_lap = completed_lap + 1
        target_generation = int(result.get("generation_after", 0)) + 1
        design_started_monotonic = time.monotonic()
        design_api_key = (get_transient_llm_api_key("claude")
                          if design_provider == "claude" and design_auth_method == "api" else "")
        if (design_provider == "claude" and design_auth_method == "api" and
                not design_api_key):
            BR["codex_auto_status"] = {
                "state": "error", "message": "Claude尚未配置API Key",
                "request_id": request_id}
            return
        feedback_event = threading.Event()
        feedback_deadline = time.monotonic() + 10.0
        BR["codex_auto_status"] = {
            "state": "waiting_feedback",
            "message": (f"第{completed_lap}轮数据已保存；10秒内可输入本轮设计意见，"
                        "否则自动开始LLM推理"),
            "request_id": request_id, "started_at": time.strftime("%H:%M:%S"),
            "completed_lap": completed_lap, "next_lap": next_lap,
            "target_generation": target_generation, "elapsed_s": 0,
            "phase": "operator_input", "feedback_wait_s": 10,
            "provider": design_provider, "model": design_model or "default",
            "prompt_profile": design_prompt_profile,
            "auth_method": design_auth_method, "reasoning_effort": design_effort,
            "fast_mode": bool(design_fast_mode and design_provider == "gpt")}
        BR["codex_feedback_queue"] = []
        BR["codex_feedback_event"] = feedback_event
        BR["codex_feedback_hold"] = False
        BR["codex_feedback_deadline_monotonic"] = feedback_deadline
        print(f"# LLM自动设计（{design_provider}）：等待10秒人工输入窗口"
              f"（request={request_id}）；无输入将自动开始")

        progress_done = threading.Event()

        def progress_worker():
            while not progress_done.wait(10.0):
                elapsed_s = int(max(0.0, time.monotonic() - design_started_monotonic))
                with LK:
                    current = BR.get("codex_auto_status") or {}
                    if (current.get("state") != "running" or
                            current.get("request_id") != request_id):
                        return
                    current = dict(current)
                    reviewing = current.get("phase") == "operator_review"
                    current.update({"elapsed_s": elapsed_s})
                    if reviewing:
                        current["message"] = (
                            f"Codex正在按人工意见复审generation {target_generation}，"
                            f"已等待{elapsed_s}秒；复审完成后才会进入第{next_lap}轮")
                    else:
                        current.update({
                            "phase": "llm_reasoning",
                            "message": (f"第{completed_lap}轮数据已保存；Codex正在推理并生成"
                                        f"generation {target_generation}，已等待{elapsed_s}秒；"
                                        f"完成后自动执行第{next_lap}轮"),
                        })
                    BR["codex_auto_status"] = current
                print(f"# LLM设计进度：第{completed_lap}轮→第{next_lap}轮，"
                      f"generation {target_generation} 推理中，已等待{elapsed_s}秒")

        def worker():
            try:
                while True:
                    if feedback_event.wait(0.1):
                        break
                    with LK:
                        held = bool(BR.get("codex_feedback_hold", False))
                        same_request = ((BR.get("codex_auto_status") or {}).get(
                            "request_id") == request_id)
                    if not same_request:
                        return
                    if not held and time.monotonic() >= feedback_deadline:
                        break

                with LK:
                    initial_notes = list(BR.get("codex_feedback_queue") or [])
                    BR["codex_feedback_queue"] = []
                    BR["codex_feedback_hold"] = False
                    if initial_notes:
                        history = list(BR.get("codex_feedback_history") or [])
                        history.extend({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        "request_id": request_id,
                                        "message": str(note),
                                        "delivery": "initial_prompt"}
                                       for note in initial_notes)
                        BR["codex_feedback_history"] = history[-20:]
                    current = dict(BR.get("codex_auto_status") or {})
                    current.update({
                        "state": "running", "phase": "llm_reasoning",
                        "feedback_applied": len(initial_notes),
                        "feedback_pending": False,
                        "message": (f"正在推理generation {target_generation}；"
                                    + (f"prompt已加入{len(initial_notes)}条人工意见"
                                       if initial_notes else "本轮无人工补充")),
                    })
                    BR["codex_auto_status"] = current
                print(f"# LLM自动设计（{design_provider}）：开始读取本轮response并推理"
                      f"（人工意见{len(initial_notes)}条）……")
                progress_thread = threading.Thread(
                    target=progress_worker, name="codex-compensator-progress", daemon=True)
                progress_thread.start()
                from codex_compensator import design_candidate_with_codex

                def take_operator_feedback():
                    with LK:
                        notes = list(BR.get("codex_feedback_queue") or [])
                        BR["codex_feedback_queue"] = []
                        if notes:
                            history = list(BR.get("codex_feedback_history") or [])
                            history.extend({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            "request_id": request_id,
                                            "message": str(note),
                                            "delivery": "same_session_review"}
                                           for note in notes)
                            BR["codex_feedback_history"] = history[-20:]
                            current = dict(BR.get("codex_auto_status") or {})
                            current.update({
                                "phase": "operator_review",
                                "feedback_pending": False,
                                "feedback_applied": int(
                                    current.get("feedback_applied", 0)) + len(notes),
                                "message": (f"已收到{len(notes)}条人工意见；"
                                            "正在同一会话中复审候选"),
                            })
                            BR["codex_auto_status"] = current
                    return notes

                answer = design_candidate_with_codex(
                    request_path, CODEX_COMP_CANDIDATE_PATH,
                    CODEX_COMP_SCHEMA_PATH, os.path.dirname(HERE),
                    timeout_s=design_timeout_s, model=design_model,
                    reasoning_effort=design_effort, fast_mode=design_fast_mode,
                    provider=design_provider,
                    api_key=design_api_key, auth_method=design_auth_method,
                    operator_feedback="\n".join(map(str, initial_notes)),
                    operator_feedback_count=len(initial_notes),
                    operator_feedback_items=list(map(str, initial_notes)),
                    feedback_provider=(take_operator_feedback
                                       if design_provider == "gpt" else None),
                    prompt_profile=design_prompt_profile)
                answer.update({
                    "completed_lap": completed_lap,
                    "next_lap": next_lap,
                    "target_generation": target_generation,
                    "elapsed_s": int(max(0.0, time.monotonic() - design_started_monotonic)),
                })
                feedback_used = int(answer.get("human_feedback_count", 0))
                if feedback_used:
                    answer["message"] = (f"{answer.get('message', '候选已生成')}；"
                                         f"已应用{feedback_used}条人工意见")
                with LK:
                    BR["codex_feedback_queue"] = []
                    BR["codex_auto_status"] = answer
                print(f"# LLM自动设计：{answer.get('message')}，耗时"
                      f"{answer.get('duration_s', 0):.1f}s")
                if has_prior_design and answer.get("failure_analysis"):
                    print(f"# 1. 上一轮预期与本轮实测差异：{answer['failure_analysis']}")
                if answer.get("design_strategy"):
                    print(f"# 2. 本轮改进方法：{answer['design_strategy']}")
                if answer.get("expected_effect"):
                    print(f"# 3. 下一轮预期效果：{answer['expected_effect']}")
                source_code = str(answer.get("source_code") or "").strip()
                if source_code:
                    print("# 4. 本轮LLM response中的已校验补偿器（下一轮实际候选）：")
                    print(source_code)
                for label, key in (("Kp", "kp_matrix"),
                                   ("Kd", "kd_matrix"),
                                   ("Ki", "ki_matrix"),
                                   ("输出限幅/mV", "output_limit_mv"),
                                   ("输出变化率/mV/s", "output_slew_mv_per_s")):
                    if answer.get(key) is not None:
                        print(f"#    {label} = {answer[key]}")
            except Exception as exc:
                answer = {"state": "error", "message": str(exc),
                          "request_id": request_id,
                          "completed_lap": completed_lap,
                          "next_lap": next_lap,
                          "target_generation": target_generation,
                          "elapsed_s": int(max(
                              0.0, time.monotonic() - design_started_monotonic))}
                with LK:
                    BR["codex_feedback_queue"] = []
                    BR["codex_auto_status"] = answer
                print(f"# ⚠ LLM自动设计失败：{exc}；稳定补偿器保持不变")
            finally:
                progress_done.set()

        thread = threading.Thread(
            target=worker, name="codex-compensator-designer", daemon=True)
        BR["codex_auto_thread"] = thread
        thread.start()

    def _codex_round_update(tracking_mse_deg2):
        """Grade one lap and always export its response for the next Codex design."""
        learner = BR.get("codex_comp")
        if not BR.get("codex_mpc") or learner is None:
            return None
        try:
            result = learner.finish_round(tracking_mse_deg2)
        except Exception as exc:
            print(f"# Codex补偿器：本轮报告导出失败，当前稳定参数不变：{exc}")
            return {"valid_for_design": False, "reason": str(exc)}
        if result.get("rolled_back"):
            print(f"# Codex补偿器：候选真实MSE "
                  f"{result.get('baseline_mse_deg2', float('nan')):.3f}→"
                  f"{result.get('tracking_mse_deg2', float('nan')):.3f} deg²，"
                  f"执行参数已回退；失败response仍保留供下一版设计")
        elif result.get("canary_checked"):
            print(f"# Codex补偿器：候选真实MSE "
                  f"{result.get('baseline_mse_deg2', float('nan')):.3f}→"
                  f"{result.get('tracking_mse_deg2', float('nan')):.3f} deg²，已保留")
        else:
            print(f"# Codex补偿器：基线真实跟踪MSE "
                  f"{result.get('tracking_mse_deg2', float('nan')):.3f} deg²")
        postmortem = result.get("postmortem") or {}
        if postmortem.get("decision_reason"):
            print(f"# 本轮审核原因：{postmortem['decision_reason']}")
        if postmortem.get("next_round_avoid"):
            print(f"# 下一轮经验约束：{postmortem['next_round_avoid']}")
        if result.get("request_path"):
            print(f"# Codex补偿器：新response设计包 → {result['request_path']}")
            if BR.get("codex_replay_active"):
                replay = BR.get("codex_replay") or {}
                replay_rounds = replay.get("rounds") or []
                replay_index = int(BR.get(
                    "codex_replay_round_index", replay.get("index", 0)))
                request_id = str(result.get("request_id") or
                                 f"{replay.get('session_id','replay')}-{replay_index+1}")
                if not result.get("valid_for_design", False):
                    BR["codex_auto_status"] = {
                        "state": "retry",
                        "message": (f"回放第{replay_index+1}轮数据审核无效；"
                                    "保持同一控制器自动重跑"),
                        "request_id": request_id,
                        "replay_round": replay_index + 1,
                        "round_count": len(replay_rounds),
                        "retry_same_generation": True,
                    }
                else:
                    next_index = replay_index + 1
                    replay["completed_laps"] = int(replay.get("completed_laps", 0))+1
                    if replay.get("duration_min") and time.monotonic() < replay.get("deadline_monotonic", 0):
                        next_index = 0  # Repeat the same modified controller, without LLM or delay.
                    replay["index"] = next_index
                    replay_complete = next_index >= len(replay_rounds)
                    BR["codex_auto_status"] = {
                        "state": ("replay_complete" if replay_complete
                                  else "replay_ready"),
                        "message": (f"{len(replay_rounds)}轮过程回放已完成"
                                    if replay_complete else
                                    f"回放第{replay_index+1}轮完成；"
                                    f"准备第{next_index+1}/{len(replay_rounds)}轮"),
                        "request_id": request_id,
                        "replay_round": replay_index + 1,
                        "next_replay_round": next_index + 1,
                        "round_count": len(replay_rounds),
                        "tracking_mse_deg2": result.get("tracking_mse_deg2"),
                    }
                    if replay_complete:
                        BR["codex_replay_active"] = False
                    else:
                        wait_s = float(replay_rounds[replay_index].get("simulated_wait_after_s", 0))
                        if wait_s > 0:
                            BR["codex_auto_status"].update(
                                state="replay_wait",
                                simulated_wait_s=wait_s,
                                wait_deadline_monotonic=time.monotonic()+wait_s,
                                message=f"模拟思考等待 {wait_s:.0f} 秒（不调用LLM）；随后执行第{next_index+1}轮")
                    if replay.get("duration_min"):
                        BR["codex_auto_status"]["message"] = (
                            f"Load 2修改版30分钟任务完成，共{replay['completed_laps']}轮" if replay_complete else
                            f"Load 2修改版已完成{replay['completed_laps']}轮，继续沿用；不调用LLM")
                return result
            if (BR.get("codex_capture_active") and
                    result.get("valid_for_design", False)):
                request_path = os.path.abspath(str(result["request_path"]))
                request_id = str(result.get("request_id") or "")
                archive_path = (os.path.splitext(request_path)[0] + "_" +
                                request_id + ".json")
                if os.path.isfile(archive_path):
                    captured = BR.setdefault("codex_capture_rounds", [])
                    if not any(str(item.get("request_id", "")) == request_id
                               for item in captured if isinstance(item, dict)):
                        captured.append({
                            "request_id": request_id,
                            "path": archive_path,
                            "tracking_mse_deg2": result.get("tracking_mse_deg2"),
                            "generation": result.get("generation_before"),
                        })
            _start_codex_auto_design(result)
        return result

    def _codex_abort():
        learner = BR.get("codex_comp")
        if not BR.get("codex_mpc") or learner is None:
            return
        answer = learner.abort_round()
        if answer.get("rolled_back"):
            print(f"# Codex补偿器：候选轮被中止，已回退到 generation="
                  f"{answer.get('generation', 0)}")

    def _finish(msg):
        # ILC needs the complete reference length while grading the final lap.
        if BR.get("ilc_mpc"):
            if BR.get("new_round_complete", False):
                _ilc_round_update(BR.get("completed_track_mse_deg2"))
            else:
                _ilc_abort()
        if BR.get("codex_mpc"):
            if BR.get("new_round_complete", False):
                _codex_round_update(BR.get("completed_track_mse_deg2"))
            else:
                _codex_abort()
        BR["mode"] = "idle"; BR["ref"] = None; BR["venting"] = 0
        if BR.get("shadow_forward") and BR.get("forward_learner") is not None:
            if BR.get("new_round_complete", False):
                _new_forward_round_update(BR.get("completed_track_mse_deg2"))
            else:
                # Stop/frozen-frame paths are incomplete and must not train or
                # promote a forward model.
                BR["forward_learner"].abandon_round()
        if BR.get("fel", False):                    # 纯 MPC/PI：模型与权重完全冻结
            if BR.get("new_mpc_fel") and not BR.get("new_round_complete", False):
                # 急停/失联得到的是截断轨迹；顺、逆两个影子都不能拿它晋级。
                BR["felPhi"], BR["felErr"], BR["felMask"] = [], [], []
                nu_e = 0
            else:
                nu_e = _fel_round_update(
                    BR.get("completed_track_mse_deg2"))  # 完整单程结束时训练一次
            nd_e = 0 if BR.get("new_mpc_fel") else _dm_round_update()
            if nu_e or nd_e: print(f"# 本程训练：DataModel {nd_e} 样本 / 逆模型 NN {nu_e} 样本")
            _save_w()
        if BR.get("shadow_forward") and BR.get("forward_learner") is not None and BR.get("path"):
            try:
                fp = BR["path"].replace(".csv", "_forward.csv")
                saved = BR["forward_learner"].save_run_csv(fp)
                if saved:
                    print(f"# 活动/影子顺模型 x_hat_next 日志 → {saved}")
            except Exception as exc:
                print(f"# 顺模型预测日志保存失败：{exc}")
        BR["new_round_complete"] = False
        BR["completed_track_mse_deg2"] = None
        if BR.get("lapstats"):                      # 逐圈收敛史 → 论文 rounds 图的原始数据
            lp = BR["path"].replace(".csv", "_laps.csv")
            try:
                with open(lp, "w", newline="") as f:
                    wr = csv.writer(f)
                    third = ("ilc_feedback_rmse_mv" if BR.get("ilc_mpc") else
                             "invmodel_rmse_mv")
                    wr.writerow(["lap", "track_rms_deg", third, "NN_weight_mean"])
                    wr.writerows(BR["lapstats"])
                print(f"# 逐圈收敛史 → {lp}")
            except Exception:
                pass
            BR["lapstats"] = []
        if BR["log"]:
            with open(BR["path"], "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["t", "q1", "q2", "q3", "r1", "r2", "r3",
                             "rraw1", "rraw2", "rraw3",
                             "v1", "v2", "v3", "vref1", "vref2", "vref3"]
                            + [f"u{i}" for i in range(NCH)]
                            + [f"uact{i}" for i in range(NCH)]
                            + [f"kcmd{i}" for i in range(NCH)]
                            + [f"uff{i}" for i in range(NCH)] + [f"ufb{i}" for i in range(NCH)]
                            + [f"ucomp{i}" for i in range(NCH)]
                            + [f"P{i}" for i in range(NCH)] + [f"lock{i}" for i in range(NCH)]
                            + [f"L{i}_mm" for i in range(NCH)]
                            + [f"Lref{i}_mm" for i in range(NCH)]
                            + ["plan_axis", "plan_channel", "plan_hold",
                               "mpc_solver", "mpc_solve_ms", "mpc_iterations"])
                wr.writerows(BR["log"])
            print(f"# {msg}，日志 → {BR['path']}")
            BR["log"] = None
        else:
            print(f"# {msg}")

    def _stage_update(key, text, point=-1):
        """统一更新终端和 Qt 的当前控制阶段。point 是 GUI 示教列表的 0 基下标。"""
        if BR.get("stage_key") != key:
            print(f"----{text}----")
            BR["stage_key"] = key
        BR["stage_text"] = text
        BR["stage_point"] = int(point)
        return {"stage": text, "stage_point": int(point)}

    def _reset_mpc_shadow_forward():
        """Reset all no-inverse shadow architectures to their fitted bases."""
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，请先停止再重置影子模型"}
        if not os.path.exists(PURE_DM_PATH):
            return {"err": f"基础拟合顺模型不存在：{PURE_DM_PATH}"}
        missing_sequence = [paths["base"] for paths in MPC_SHADOW_SEQUENCE_PATHS.values()
                            if not os.path.exists(paths["base"])]
        if missing_sequence:
            return {"err": "缺少时序基础模型：" + "、".join(missing_sequence)}
        try:
            reset_specs = [{
                "architecture": "mlp", "base": PURE_DM_PATH,
                "active": MPC_SHADOW_FORWARD_ACTIVE_PATH,
                "shadow": MPC_SHADOW_FORWARD_SHADOW_PATH,
                "previous": MPC_SHADOW_FORWARD_PREVIOUS_PATH,
                "state": MPC_SHADOW_FORWARD_STATE_PATH,
            }]
            reset_specs.extend({"architecture": architecture, **paths}
                               for architecture, paths in MPC_SHADOW_SEQUENCE_PATHS.items())
            for spec in reset_specs:
                for dst in (spec["active"], spec["shadow"], spec["previous"]):
                    tmp = dst + ".reset.tmp"
                    shutil.copy2(spec["base"], tmp)
                    os.replace(tmp, dst)
                state = {
                    "schema": 1, "sample_dt": DT,
                    "architecture": spec["architecture"], "generation": 0,
                    "consecutive_wins": 0, "consecutive_failures": 0,
                    "rounds_seen": 0, "tracking_mse_history_deg2": [],
                    "canary": {"active": False}, "history": [],
                }
                tmp_state = spec["state"] + ".reset.tmp"
                with open(tmp_state, "w", encoding="utf-8") as stream:
                    json.dump(state, stream, ensure_ascii=False, indent=2)
                os.replace(tmp_state, spec["state"])
                replay_path = os.path.splitext(spec["state"])[0] + "_replay.npz"
                try:
                    os.remove(replay_path)
                except FileNotFoundError:
                    pass
            # 下一次播放重新构造 learner/MPC，重新从三轮累计的第1轮开始。
            BR["forward_learner"] = None
            BR["forward_last"] = None
            BR["dm"] = None
            BR["dampc"] = None
            BR["dm_kind"] = None
            BR["forward_promoted_this_round"] = False
            print("# MPC+影子正模型：MLP/history_mlp/GRU均已恢复各自基础模型，从generation 0重新训练")
            return {"ok": 1, "generation": 0,
                    "message": "三种影子正模型均已恢复为各自基础模型，将从第1/3轮重新累计"}
        except Exception as exc:
            return {"err": f"重置影子正模型失败：{exc}"}

    def _reset_mpc_ilc():
        """Clear only the task-indexed ILC curve; never touch an MPC model."""
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，请先停止再重置ILC"}
        try:
            from mpc_ilc import ConstrainedILC
            ConstrainedILC.reset_file(MPC_ILC_STATE_PATH)
            BR["ilc"] = None
            _ilc_buffers_reset()
            print("# 约束ILC：前馈曲线和审核历史已清零，下一次从generation 0开始")
            return {"ok": 1, "generation": 0,
                    "message": "ILC前馈曲线已清零；真正顺模型和MPC参数没有改动"}
        except Exception as exc:
            return {"err": f"重置ILC失败：{exc}"}

    def _reset_codex_compensator():
        """Clear only Codex-designed residual parameters and their reports."""
        if BR.get("mode") == "mpc":
            return {"err": "控制正在运行，请先停止再重置Codex补偿器"}
        active = BR.get("codex_auto_thread")
        if active is not None and active.is_alive():
            return {"err": "Codex正在自动分析本轮数据，请等分析结束再清零"}
        try:
            from codex_compensator import CodexCompensator
            CodexCompensator.reset_files(
                CODEX_COMP_STATE_PATH, CODEX_COMP_CANDIDATE_PATH,
                CODEX_COMP_REQUEST_PATH)
            BR["codex_comp"] = None
            BR["codex_u_comp"] = np.zeros(NCH)
            BR["codex_auto_status"] = {
                "state": "idle", "message": "已清零，等待新的基线轮"}
            print("# Codex补偿器：参数、候选和最近设计报告已清零，从无补偿基线开始")
            return {"ok": 1, "generation": 0,
                    "message": "Codex补偿器已清零；原始真正模型和MPC参数没有改动"}
        except Exception as exc:
            return {"err": f"重置Codex补偿器失败：{exc}"}

    def _reset_original_new_mpc_fel_branch():
        """Start the original comparison branch at forward gen-0/inverse zero.

        Selecting ``original`` is an experimental reset, not a request to reopen
        a previously adapted branch.  The fitted forward model is copied into
        all active/shadow slots and a freshly-created zero-output inverse MLP is
        installed.  Learning remains enabled after the run starts.
        """
        _ensure_original_model_pair()
        for dst in (ORIGINAL_INIT_FORWARD_ACTIVE_PATH,
                    ORIGINAL_INIT_FORWARD_SHADOW_PATH,
                    ORIGINAL_INIT_FORWARD_PREVIOUS_PATH):
            tmp = dst + ".reset.tmp"
            shutil.copy2(ORIGINAL_FORWARD_PATH, tmp)
            os.replace(tmp, dst)
        forward_state = {
            "schema": 1, "sample_dt": DT, "architecture": "mlp",
            "generation": 0, "consecutive_wins": 0,
            "consecutive_failures": 0, "rounds_seen": 0,
            "tracking_mse_history_deg2": [],
            "canary": {"active": False}, "history": [],
        }
        tmp_state = ORIGINAL_INIT_FORWARD_STATE_PATH + ".reset.tmp"
        with open(tmp_state, "w", encoding="utf-8") as stream:
            json.dump(forward_state, stream, ensure_ascii=False, indent=2)
        os.replace(tmp_state, ORIGINAL_INIT_FORWARD_STATE_PATH)

        zero_inverse = _fel_new()
        for dst in (ORIGINAL_INIT_INVERSE_ACTIVE_PATH,
                    ORIGINAL_INIT_INVERSE_SHADOW_PATH,
                    ORIGINAL_INIT_INVERSE_PREVIOUS_PATH):
            _inverse_model_save(zero_inverse, dst)
        inverse_state = {
            "schema": 1, "generation": 0, "consecutive_wins": 0,
            "rounds_seen": 0, "tracking_mse_history_deg2": [],
            "canary": {"active": False}, "history": [],
        }
        tmp_inverse_state = ORIGINAL_INIT_INVERSE_STATE_PATH + ".reset.tmp"
        with open(tmp_inverse_state, "w", encoding="utf-8") as stream:
            json.dump(inverse_state, stream, ensure_ascii=False, indent=2)
        os.replace(tmp_inverse_state, ORIGINAL_INIT_INVERSE_STATE_PATH)
        for state_path in (ORIGINAL_INIT_FORWARD_STATE_PATH,
                           ORIGINAL_INIT_INVERSE_STATE_PATH):
            replay_path = os.path.splitext(state_path)[0] + "_replay.npz"
            try:
                os.remove(replay_path)
            except FileNotFoundError:
                pass

        # Force this /start request to rebuild every in-memory model from the
        # newly reset artifacts.
        BR["nn"] = None
        BR["fel_path"] = None
        BR["inverse_shadow"] = None
        BR["inverse_replay"] = []
        BR["inverse_rounds_seen"] = 0
        BR["inverse_state"] = None
        BR["forward_learner"] = None
        BR["forward_last"] = None
        BR["dm"] = None
        BR["dampc"] = None
        BR["dm_kind"] = None
        BR["forward_promoted_this_round"] = False
        print("# NEW MPC+FEL 原始对照：真正顺模型 generation 0；逆模型六路零输出")

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def _hdr(self, code=200):
            self.send_response(code)
            for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "*"),
                         ("Access-Control-Allow-Methods", "*"), ("Access-Control-Allow-Private-Network", "true"),
                         ("Content-Type", "application/json")):
                self.send_header(k, v)
            self.end_headers()
        def _out(self, obj):
            # Qt 页面旧代码仍调用 imuSend("W...")。把本次 /step 同时返回的
            # lock+u 暂存为一次性原子命令，由 SerialBridge.write 转成 M 协议；
            # PI 响应没有 lock，会显式清空该槽并继续使用传统 W 协议。
            global HOST_MPC_COMMAND
            if isinstance(obj, dict) and "u" in obj:
                lock = obj.get("lock")
                u = obj.get("u")
                HOST_MPC_COMMAND = ((list(lock), list(u), bool(obj.get("phi_pair_assist", False)))
                                    if isinstance(lock, list) and len(lock) == NCH and
                                    isinstance(u, list) and len(u) == NCH else None)
            self._hdr()
            self.wfile.write(json.dumps(obj).encode())
        def do_OPTIONS(self): self._hdr(204)
        def do_GET(self):
            with LK:
                if self.path == "/pid":
                    self._out({"ok": 1, **BR["pid"],
                               "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "pi"})
                elif self.path == "/mpc_weights":
                    self._out({"ok": 1, **BR["mpc_weights"],
                               "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "mpc"})
                elif self.path == "/llm_config":
                    self._out(_llm_config_status())
                elif self.path == "/codex_replay":
                    self._out(_codex_replay_status())
                elif self.path == "/llm_feedback":
                    auto = dict(BR.get("codex_auto_status") or {})
                    job = BR.get("codex_auto_thread")
                    alive = bool(job is not None and job.is_alive())
                    state = str(auto.get("state", ""))
                    waiting = bool(alive and state == "waiting_feedback")
                    running = bool(alive and state == "running")
                    held = bool(BR.get("codex_feedback_hold", False))
                    remaining = (None if held else max(0, int(math.ceil(
                        float(BR.get("codex_feedback_deadline_monotonic", 0.0)) -
                        time.monotonic())))) if waiting else 0
                    applied_feedback = [dict(item) for item in
                                        (BR.get("codex_feedback_history") or [])[-5:]
                                        if isinstance(item, dict)]
                    queued_feedback = [str(item) for item in
                                       (BR.get("codex_feedback_queue") or [])[-5:]]
                    self._out({
                        "ok": 1,
                        "input_open": waiting or running,
                        "waiting_for_prompt": waiting,
                        "design_running": running,
                        "held": held,
                        "remaining_s": remaining,
                        "provider": str(BR.get("llm_provider", "gpt")),
                        "queued": len(BR.get("codex_feedback_queue") or []),
                        "applied": int(auto.get("feedback_applied", 0)),
                        "queued_feedback": queued_feedback,
                        "applied_feedback": applied_feedback,
                        "phase": str(auto.get("phase", "")),
                        "message": str(auto.get("message", "")),
                    })
                elif self.path == "/pressure_pid":
                    self._out({"ok": 1, **BR["pressure_pid"],
                               "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "pi"})
                elif self.path == "/muscle_pid":
                    self._out({"ok": 1, **BR["muscle_pid"],
                               "active": BR.get("mode") == "mpc" and BR.get("ctrl") == "musclepi"})
                else:
                    auto = BR.get("codex_auto_status") or {}
                    if auto.get("state") == "replay_wait":
                        remaining = max(0, math.ceil(float(auto["wait_deadline_monotonic"])-time.monotonic()))
                        auto["remaining_s"] = remaining
                        auto["message"] = (f"模拟思考等待：剩余{remaining}秒（不调用LLM）" if remaining else
                                           "模拟等待结束，准备下一轮回放")
                        if not remaining:
                            auto["state"] = "replay_ready"
                    self._out({"mode": BR["mode"], "k": BR["k"],
                               "stage": BR.get("stage_text", "待机"),
                               "stage_point": BR.get("stage_point", -1),
                               "codex_auto": dict(BR.get("codex_auto_status") or {})})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            with LK:
                if self.path == "/pid":
                    self._out(_pid_apply(data)); return
                if self.path == "/mpc_weights":
                    self._out(_mpc_weights_apply(data)); return
                if self.path == "/llm_config":
                    self._out(_llm_config_apply(data)); return
                if self.path == "/codex_replay/capture_start":
                    self._out(_codex_capture_start()); return
                if self.path == "/codex_replay/save":
                    self._out(_codex_replay_save(data)); return
                if self.path == "/codex_replay/load":
                    self._out(_codex_replay_load(data)); return
                if self.path == "/codex_replay/reset":
                    self._out(_codex_replay_reset()); return
                if self.path == "/codex_replay/clear":
                    self._out(_codex_replay_clear()); return
                if self.path == "/llm_feedback":
                    text = str(data.get("message", "")).strip()
                    action = str(data.get("action", "submit")).strip().lower()
                    auto = dict(BR.get("codex_auto_status") or {})
                    job = BR.get("codex_auto_thread")
                    alive = bool(job is not None and job.is_alive())
                    state = str(auto.get("state", ""))
                    if not alive or state not in ("waiting_feedback", "running"):
                        self._out({"err": "当前没有可介入的LLM设计任务"}); return
                    if state == "waiting_feedback" and action == "hold":
                        BR["codex_feedback_hold"] = True
                        auto.update({"feedback_hold": True,
                                     "message": "已暂停自动开始；输入意见后提交，或直接开始推理"})
                        BR["codex_auto_status"] = auto
                        self._out({"ok": 1, "held": True,
                                   "message": "已暂停10秒倒计时"}); return
                    if state == "running" and str(BR.get("llm_provider", "gpt")) != "gpt":
                        self._out({"err": "推理开始后的介入暂仅支持GPT/Codex"}); return
                    if action not in ("submit", "skip"):
                        self._out({"err": "未知的人工介入操作"}); return
                    if action == "submit" and not text:
                        self._out({"err": "请输入要追加给LLM的意见"}); return
                    if len(text) > 1200:
                        self._out({"err": "单条意见不能超过1200个字符"}); return
                    queue = list(BR.get("codex_feedback_queue") or [])
                    if len(queue) >= 5:
                        self._out({"err": "本次推理最多排队5条人工意见"}); return
                    if text:
                        queue.append(text)
                    BR["codex_feedback_queue"] = queue
                    if state == "waiting_feedback":
                        event = BR.get("codex_feedback_event")
                        if isinstance(event, threading.Event):
                            event.set()
                        auto.update({"feedback_pending": bool(text),
                                     "feedback_queued": len(queue),
                                     "message": ("意见已加入本轮prompt；正在开始LLM推理"
                                                 if text else "正在立即开始LLM推理")})
                    else:
                        auto.update({"feedback_pending": True,
                                     "feedback_queued": len(queue)})
                    BR["codex_auto_status"] = auto
                    if state == "waiting_feedback":
                        print(f"# LLM人工输入窗口：{'已加入prompt' if text else '无补充立即开始'}")
                    else:
                        print(f"# LLM人工介入：已排队第{len(queue)}条意见，当前推理结束后同会话复审")
                    self._out({"ok": 1, "queued": len(queue),
                               "started": state == "waiting_feedback",
                               "message": ("意见已加入本轮prompt"
                                           if state == "waiting_feedback" and text else
                                           "已立即开始推理" if state == "waiting_feedback" else
                                           "意见已提交；当前推理结束后将在同一会话中复审")})
                    return
                if self.path == "/pressure_pid":
                    self._out(_pressure_pid_apply(data)); return
                if self.path == "/muscle_pid":
                    self._out(_muscle_pid_apply(data)); return
                if self.path == "/reset_mpc_shadow":
                    self._out(_reset_mpc_shadow_forward()); return
                if self.path == "/reset_mpc_ilc":
                    self._out(_reset_mpc_ilc()); return
                if self.path == "/reset_codex_compensator":
                    self._out(_reset_codex_compensator()); return
                if self.path == "/start":
                    mount = str(data.get("mount", args.mount))
                    if mount == "id": mount = "none"          # GUI 选项名 → 控制站内部名
                    if mount not in MOUNTS:
                        self._out({"err": f"未知 IMU 映射 {mount}"}); return
                    args.mount = mount
                    SRV["mount"] = mount
                    pts = data.get("points", [])
                    auto_hidden_return_to_zero = bool(
                        data.get("auto_hidden_return_to_zero", True))
                    ctrl = data.get("ctrl", "mpc")             # 'mpc' | 'pi' | 'musclepi'
                    if ctrl not in ("mpc", "pi", "musclepi"):
                        self._out({"err": f"未知控制模式 {ctrl}"}); return
                    new_mpc_fel = bool(data.get("new_mpc_fel", False))
                    shadow_forward_only = bool(data.get("shadow_forward_only", False))
                    ilc_mpc = bool(data.get("ilc_mpc", False))
                    codex_mpc = bool(data.get("codex_compensator", False))
                    fixed_task_id = str(data.get("fixed_task_id", "")).strip()
                    try:
                        execution_round = max(1, int(float(data.get("execution_round", 1))))
                    except (TypeError, ValueError):
                        execution_round = 1
                    # Explicit Default-1 hardware exception requested by the
                    # operator.  The replacement V3 valve fills slowly, so
                    # every Default-1 MPC+LLM lap precharges it near the
                    # positive-phi peak before using it for the main return.
                    default1_phi_v3_assist = bool(
                        codex_mpc and fixed_task_id == "1")
                    # 只允许 Codex 的 GUI 自动续轮保留控制响应；普通手动播放仍从空图开始。
                    preserve_response = codex_mpc and bool(data.get("preserve_response", False))
                    shadow_forward = new_mpc_fel or shadow_forward_only
                    model_source = str(data.get("model_source", "latest")).strip().lower()
                    if model_source not in ("latest", "original"):
                        self._out({"err": f"未知模型版本 {model_source}"}); return
                    # 正/逆模型成对选版只属于完整 NEW MPC+FEL。
                    # 其他模式继续沿用各自原有的模型路径。
                    if not new_mpc_fel:
                        model_source = "latest"
                    if shadow_forward and ctrl != "mpc":
                        self._out({"err": "影子顺模型只能与 MPC 控制器一起使用"}); return
                    if new_mpc_fel and shadow_forward_only:
                        self._out({"err": "完整FEL与无逆模型影子模式不能同时启用"}); return
                    if ilc_mpc and (new_mpc_fel or shadow_forward_only):
                        self._out({"err": "约束ILC不能同时启用FEL或影子正模型"}); return
                    if codex_mpc and ctrl != "mpc":
                        self._out({"err": "Codex补偿器只能与MPC控制器一起使用"}); return
                    if codex_mpc and (new_mpc_fel or shadow_forward_only or ilc_mpc):
                        self._out({"err": "Codex补偿器必须使用独立的原始LTV MPC基线，不能叠加FEL、影子学习或ILC"}); return
                    replay_blob = BR.get("codex_replay")
                    replay_requested = bool(
                        codex_mpc and isinstance(replay_blob, dict) and
                        replay_blob.get("active"))
                    if replay_requested:
                        replay_rounds = replay_blob.get("rounds") or []
                        if replay_blob.get("duration_min"):
                            deadline = replay_blob.get("deadline_monotonic")
                            if deadline is not None and time.monotonic() >= deadline:
                                self._out({"err": "Load 2的30分钟任务已结束，请重新执行以开始新任务"}); return
                            replay_blob.setdefault("deadline_monotonic", time.monotonic()+1800.0)
                        replay_index = int(replay_blob.get("index", 0))
                        if not replay_rounds or not 0 <= replay_index < len(replay_rounds):
                            self._out({"err": "过程回放已经结束；请点击回放按钮重新复位"}); return
                    BR["codex_replay_active"] = replay_requested
                    plain_mpc_baseline = bool(replay_requested and
                        replay_blob.get("comparison_variant") == 2 and
                        replay_rounds[replay_index].get("comparison_role") == "baseline_no_v3")
                    BR["plain_mpc_baseline"] = plain_mpc_baseline
                    if plain_mpc_baseline:
                        print("# Load 2首轮：普通MPC模型/参数/原始输出；仅记录，不屏蔽V3，不叠加补偿器限速")
                    BR["replay_forbid_v3"] = bool(replay_requested and
                        replay_rounds[replay_index].get("forbid_v3", False) and not plain_mpc_baseline)
                    if replay_requested:
                        # Restore the runtime assistance used in the selected
                        # measured lap as well as its Python compensator.
                        default1_phi_v3_assist = bool(
                            replay_rounds[replay_index].get("default1_phi_v3_assist", False)
                            and not BR["replay_forbid_v3"])
                    auto_job = BR.get("codex_auto_thread")
                    if (codex_mpc and not replay_requested and auto_job is not None and
                            auto_job.is_alive()):
                        self._out({"err": "Codex正在自动读取上一轮response并设计候选，请等GUI显示‘候选已生成’后再播放"}); return
                    if new_mpc_fel and model_source == "original":
                        try:
                            _reset_original_new_mpc_fel_branch()
                        except Exception as exc:
                            self._out({"err": str(exc)}); return
                    run_dt = PI_DT if ctrl in ("pi", "musclepi") else DT
                    BR["dt"] = run_dt
                    reference_mode = str(data.get("reference_mode", "smooth")).lower()
                    if reference_mode not in ("smooth", "step"):
                        self._out({"err": f"未知参考方式 {reference_mode}"}); return
                    step_reference = reference_mode == "step"
                    BR["reference_mode"] = reference_mode
                    codex_loop_requested = codex_mpc and bool(data.get("loop"))
                    # Codex需要完整单轮response做设计，所以控制站一次只执行一轮；
                    # Qt GUI 勾选循环时会等候候选就绪，再自动发起下一轮 /start。
                    BR["loop"] = bool(data.get("loop")) and not codex_mpc
                    BR["loop_limit_runs"] = None
                    BR["loop_deadline_monotonic"] = None
                    if BR["loop"]:
                        try:
                            raw_runs = data.get("loop_limit_runs")
                            if raw_runs is not None:
                                BR["loop_limit_runs"] = max(1, min(999, int(float(raw_runs))))
                        except (TypeError, ValueError):
                            BR["loop_limit_runs"] = None
                        try:
                            raw_minutes = data.get("loop_limit_min")
                            if raw_minutes is not None:
                                minutes = max(0.1, min(999.0, float(raw_minutes)))
                                BR["loop_deadline_monotonic"] = time.monotonic() + minutes * 60.0
                        except (TypeError, ValueError):
                            BR["loop_deadline_monotonic"] = None
                    BR["fel"] = bool(data.get("fel", True))    # FEL 学习层开关（默认开）
                    BR["new_mpc_fel"] = new_mpc_fel
                    BR["shadow_forward_only"] = shadow_forward_only
                    BR["shadow_forward"] = shadow_forward
                    BR["ilc_mpc"] = ilc_mpc
                    BR["codex_mpc"] = codex_mpc
                    BR["fixed_task_id"] = fixed_task_id
                    BR["execution_round"] = execution_round
                    BR["default1_phi_v3_assist"] = default1_phi_v3_assist
                    BR["default1_phi_v3_releasing"] = False
                    BR["load2_v3_precharge"] = bool(replay_requested and
                        replay_rounds[replay_index].get("comparison_role") ==
                        "round5_with_round4_phi_descent_load2")
                    BR["load2_v3_timing"] = {} if BR["load2_v3_precharge"] else None
                    BR["load2_y_return_state"] = {} if BR["load2_v3_precharge"] else None
                    BR["model_source"] = model_source
                    # 模型来源只决定初始化分支，不关闭学习。
                    BR["models_frozen"] = False
                    BR["new_round_complete"] = False
                    if new_mpc_fel:
                        BR["fel"] = True                        # 新模式必须同时训练逆模型
                    if shadow_forward_only:
                        BR["fel"] = False                       # 只学习顺模型：逆模型不加载、不输出、不训练
                    if ilc_mpc:
                        BR["fel"] = False                       # 只学逐拍ILC前馈；真正模型和逆模型都冻结
                    if codex_mpc:
                        BR["fel"] = False                       # 固定真正模型；只在轮间更换受限补偿参数
                    if ctrl == "musclepi":
                        BR["fel"] = False                       # 长度 PI 是独立基线，不接 DataModel/FEL
                    if not BR["fel"]:
                        BR["nn"] = None
                        BR["fel_path"] = None
                    else:
                        if new_mpc_fel and model_source == "original":
                            inverse_paths = (
                                ORIGINAL_INIT_INVERSE_ACTIVE_PATH,
                                ORIGINAL_INIT_INVERSE_SHADOW_PATH,
                                ORIGINAL_INIT_INVERSE_PREVIOUS_PATH,
                                ORIGINAL_INIT_INVERSE_STATE_PATH)
                            inverse_bootstrap = ORIGINAL_INVERSE_PATH
                        elif new_mpc_fel:
                            inverse_paths = (
                                NEW_MPC_FEL_NN_PATH,
                                NEW_MPC_FEL_NN_SHADOW_PATH,
                                NEW_MPC_FEL_NN_PREVIOUS_PATH,
                                NEW_MPC_FEL_NN_STATE_PATH)
                            inverse_bootstrap = FEL_NN_PATH
                        else:
                            inverse_paths = (FEL_NN_PATH, None, None, None)
                            inverse_bootstrap = None
                        desired_fel_path = inverse_paths[0]
                        BR["inverse_active_path"] = inverse_paths[0]
                        BR["inverse_shadow_path"] = inverse_paths[1]
                        BR["inverse_previous_path"] = inverse_paths[2]
                        BR["inverse_state_path"] = inverse_paths[3]
                        if BR.get("nn") is None or BR.get("fel_path") != desired_fel_path:
                            BR["nn"] = _fel_load(
                                desired_fel_path,
                                bootstrap_path=inverse_bootstrap)
                        BR["fel_path"] = desired_fel_path
                        if new_mpc_fel and model_source == "original" and BR.get("nn") is None:
                            self._out({"err": "原始逆模型快照无法加载"}); return
                    if new_mpc_fel:
                        _new_inverse_prepare()
                    else:
                        BR["inverse_shadow"] = None
                        BR["inverse_replay"] = []
                        BR["inverse_rounds_seen"] = 0
                        BR["inverse_state"] = None
                    GN = _gains_load()                         # 每次播放重读增益面板（改完不用重启）
                    if replay_requested and replay_blob.get("comparison_variant") == 2 and not plain_mpc_baseline:
                        # Per-run override only: never persist to gains.json.
                        GN = dict(GN)
                        GN["mpc_w_q"] = [1000.0] * 3
                        GN["mpc_w_v"] = [300.0 if replay_index == 0 and
                            replay_rounds[replay_index].get("comparison_role") ==
                            "baseline_no_v3" else 200.0] * 3
                    BR["codex_auto_enabled"] = bool(GN.get("codex_auto_design", True))
                    BR["llm_provider"] = str(GN.get("llm_provider", "gpt")).strip().lower()
                    BR["llm_auth_method"] = str(
                        GN.get("llm_auth_method", "account")).strip().lower()
                    BR["codex_model"] = str(GN.get("codex_model", "")).strip()
                    try:
                        BR["codex_reasoning_effort"] = normalize_reasoning_effort(
                            GN.get("codex_reasoning_effort", "medium"))
                    except ValueError as exc:
                        self._out({"err": str(exc)}); return
                    BR["codex_fast_mode"] = bool(GN.get("codex_fast_mode", False))
                    BR["codex_auto_timeout_s"] = float(np.clip(
                        float(GN.get("codex_auto_timeout_s", 300.0)), 30.0, 900.0))
                    BR["codex_reuse_mse_threshold_deg2"] = float(np.clip(
                        float(GN.get("codex_reuse_mse_threshold_deg2", 15.0)),
                        0.0, 10000.0))
                    forward_arch = str(GN.get("mpc_forward_model", "mlp")).strip().lower()
                    if forward_arch not in ("mlp", "history_mlp", "gru"):
                        self._out({"err": "gains.json 的 mpc_forward_model 只能是 mlp、history_mlp 或 gru"}); return
                    if new_mpc_fel and forward_arch != "mlp":
                        self._out({"err": "完整NEW MPC+FEL的成对正/逆模型当前仍使用单拍MLP；"
                                          "10拍history_mlp/GRU请使用“MPC + 影子正模型（无逆模型）”"}); return
                    if ilc_mpc or (codex_mpc and not plain_mpc_baseline):
                        forward_arch = "mlp"                    # 固定原始单拍真正模型作为对照底座
                    BR["forward_arch"] = forward_arch
                    BR["mpc_weights"] = {
                        "wq": float(np.mean(GN["mpc_w_q"])),
                        "wv": float(np.mean(GN["mpc_w_v"]))}
                    BR["pid"] = {"kp": float(GN["pi_kp"]), "ki": float(GN["pi_ki"]),
                                 "kd": float(GN.get("pi_kd", 0.0)),
                                 "kv": float(GN.get("pi_kv", PI_KV)),
                                 "vmax": float(GN.get("pi_vmax_deg", PI_VMAX_DEG))}
                    BR["pressure_pid"] = {
                        "kp": float(GN.get("pressure_hold_kp", PRESSURE_HOLD_KP)),
                        "ki": float(GN.get("pressure_hold_ki", PRESSURE_HOLD_KI)),
                        "deadband": float(GN.get("pressure_hold_deadband_kpa", PRESSURE_HOLD_DEADBAND_KPA))}
                    BR["muscle_pid"] = {
                        "kp": float(GN.get("muscle_pi_kp", MUSCLE_PI_KP)),
                        "ki": float(GN.get("muscle_pi_ki", MUSCLE_PI_KI)),
                        "deadband": float(GN.get("muscle_pi_deadband_mm", MUSCLE_PI_DEADBAND_MM))}
                    NH = int(np.clip(int(GN["mpc_N"]), 4, 40)); BR["N"] = NH
                    # 参考直接按三分量 (sx,sy,ψ) 构建（复检修正：旧路径把 ψ 丢了）；
                    # 循环模式追加「末点→首点」2s 回绕段并回绕到首点驻留处——参考连续，
                    # 不再每圈从末点硬跳回零（旧实现真机每圈哆嗦一下，FEL 还把跳变学进权重）
                    # 若已完成六方向测速，用70%可重复速度限制五次S曲线的峰值速度。
                    # smoothstep5 的最大归一化斜率为1.875，因此所需段时长 = 1.875*Δ角/速度。
                    speed_dirs = {}
                    try:
                        speed_blob = json.load(open(SPEED_LIMIT_PATH, encoding="utf-8"))
                        # coordinate_version=2 才是修正 x/φ 映射后的实机测速。
                        # 旧文件会被忽略，防止错误速度继续影响轨迹时长。
                        if int(speed_blob.get("coordinate_version", 0)) == 2:
                            speed_dirs = speed_blob.get("directions", {})
                    except Exception:
                        pass
                    speed_keys = (("x-", "x+"), ("y-", "y+"), ("φ-", "φ+"))
                    speed_adjustments = []
                    axis_schedule_desc = []
                    if not pts:
                        self._out({"err": "至少需要一个示教点"}); return
                    pts_qm = []
                    ref_l = [np.zeros(3)]; cur = np.zeros(3)
                    length_ref_l = [np.full(NCH, MUSCLE_SENSOR_L0_MM)]
                    length_start_weight = [1.0]     # 首段按首拍实测长度重建，终点仍是示教长度
                    cur_length = np.full(NCH, MUSCLE_SENSOR_L0_MM)
                    plot_ref_l = [np.zeros(3)]       # 原始示教角度，仅用于曲线/日志，不参与控制
                    cur_gui = np.zeros(3)
                    axis_plan = [-1]
                    channel_plan = [-1]
                    point_plan = [0]
                    prev_gui = np.zeros(3)
                    prev_raw_gui = np.zeros(3)
                    for point_i, pt in enumerate(pts, 1):
                        v3 = [float(v) for v in (list(pt["q"]) + [0, 0, 0])[:3]]
                        raw_v3 = [float(v) for v in (list(pt.get("q_raw", pt["q"])) + [0, 0, 0])[:3]]
                        requested_dur = float(pt.get("t", 2))
                        qm = q_model_from_st(v3[0], v3[1], v3[2], args.mount)
                        n_before = len(ref_l)
                        if ctrl == "pi":
                            add_ref, add_axis, add_channel, cur_gui, effective_dur, desc, dead_ignored, small_ignored, x_ignored = _pi_plan_segment(
                                cur_gui, v3, np.asarray(v3, float)-prev_gui,
                                requested_dur, run_dt, speed_dirs,
                                GN.get("pi_vmax_deg", PI_VMAX_DEG), step_reference)
                            cur = q_model_from_st(*cur_gui)
                            ref_l.extend(add_ref); axis_plan.extend(add_axis)
                            channel_plan.extend(add_channel)
                            point_plan.extend([point_i-1]*len(add_ref))
                            axis_schedule_desc.append(
                                (point_i, desc, dead_ignored, small_ignored, x_ignored, effective_dur))
                        elif ctrl == "musclepi":
                            saved = pt.get("channel_length_mm")
                            if not isinstance(saved, list) or len(saved) < NCH:
                                self._out({"err": f"第{point_i}个示教点没有六组肌肉长度；请用20路传感器重新记录"}); return
                            try:
                                gui_length = np.array([float(v) for v in saved[:NCH]], float)
                            except (TypeError, ValueError):
                                self._out({"err": f"第{point_i}个示教点存在无效肌肉长度；请检查20路传感器"}); return
                            if not np.all(np.isfinite(gui_length)):
                                self._out({"err": f"第{point_i}个示教点存在掉线的肌肉长度；请检查20路传感器"}); return
                            target_length = gui_length[MUSCLE_GUI_TO_PHYSICAL]
                            # 肌肉 PI 复用角度 PI 的 X/Y/φ 意图判断、最多两轴、顺序选轴、
                            # 实测速限和5秒上限；角度只做离线调度，实时反馈仍是肌肉长度。
                            add_ref, add_axis, add_channel, cur_gui, effective_dur, desc, dead_ignored, small_ignored, x_ignored = _pi_plan_segment(
                                cur_gui, v3, np.asarray(v3, float)-prev_gui,
                                requested_dur, run_dt, speed_dirs,
                                GN.get("pi_vmax_deg", PI_VMAX_DEG), step_reference)
                            add_length, cur_length = _muscle_length_refs_for_plan(
                                cur_length, target_length, add_channel, step_reference)
                            cur = q_model_from_st(*cur_gui)
                            ref_l.extend(add_ref); axis_plan.extend(add_axis)
                            channel_plan.extend(add_channel)
                            length_ref_l.extend(add_length)
                            nseg = len(add_ref)
                            length_start_weight.extend([
                                1.0-_s_curve5(j/max(1, nseg)) if point_i == 1 else 0.0
                                for j in range(1, nseg+1)])
                            point_plan.extend([point_i-1]*nseg)
                            axis_schedule_desc.append(
                                (point_i, desc, dead_ignored, small_ignored, x_ignored, effective_dur))
                        else:
                            required_dur = 0.0
                            if speed_dirs:
                                delta = np.asarray(v3, float)-prev_gui
                                for axis in range(3):
                                    key = speed_keys[axis][1 if delta[axis] >= 0.0 else 0]
                                    lim = float(speed_dirs.get(key, {}).get("planning_deg_s", 0.0))
                                    if lim > 1e-6:
                                        required_dur = max(required_dur, 1.875*abs(float(delta[axis]))/lim)
                            effective_dur = min(PI_PLAN_MAX_DURATION_S,
                                                max(requested_dur, required_dur))
                            nseg = max(1, int(round(effective_dur/run_dt)))
                            for j in range(1, nseg+1):
                                ref_l.append(qm.copy() if step_reference else
                                             cur + (qm-cur)*_s_curve5(j/nseg)); axis_plan.append(-1)
                                channel_plan.append(-1)
                                point_plan.append(point_i-1)
                            cur = qm
                        n_added = len(ref_l) - n_before
                        if ctrl != "musclepi":
                            length_ref_l.extend([np.full(NCH, np.nan) for _ in range(n_added)])
                            length_start_weight.extend([0.0]*n_added)
                        for j in range(1, n_added+1):
                            raw_gui = (np.asarray(raw_v3, float) if step_reference else
                                       prev_raw_gui + (np.asarray(raw_v3)-prev_raw_gui)*_s_curve5(j/n_added))
                            plot_ref_l.append(q_model_from_st(*raw_gui))
                        if effective_dur > requested_dur + 0.01:
                            speed_adjustments.append((point_i, requested_dur, effective_dur))
                        pts_qm.append((cur.copy(), effective_dur))
                        prev_gui = np.asarray(v3, float)
                        prev_raw_gui = np.asarray(raw_v3, float)
                    # 默认轨迹追加隐藏的 (0,0,0) 回零段；显式关闭时，末点完成后
                    # 直接进入强制全机放气。循环模式都要等放气结束才开始下一轮。
                    if pts_qm and auto_hidden_return_to_zero:
                        n_before = len(ref_l)
                        if ctrl == "pi":
                            add_ref, add_axis, add_channel, cur_gui, return_dur, desc, dead_ignored, small_ignored, x_ignored = _pi_plan_segment(
                                cur_gui, np.zeros(3), -prev_gui,
                                2.0, run_dt, speed_dirs,
                                GN.get("pi_vmax_deg", PI_VMAX_DEG), step_reference)
                            cur = q_model_from_st(*cur_gui)
                            ref_l.extend(add_ref); axis_plan.extend(add_axis)
                            channel_plan.extend(add_channel)
                            point_plan.extend([-1]*len(add_ref))
                            axis_schedule_desc.append(
                                ("回零", desc, dead_ignored, small_ignored, x_ignored, return_dur))
                        elif ctrl == "musclepi":
                            add_ref, add_axis, add_channel, cur_gui, return_dur, desc, dead_ignored, small_ignored, x_ignored = _pi_plan_segment(
                                cur_gui, np.zeros(3), -prev_gui,
                                2.0, run_dt, speed_dirs,
                                GN.get("pi_vmax_deg", PI_VMAX_DEG), step_reference)
                            add_length, cur_length = _muscle_length_refs_for_plan(
                                cur_length, np.full(NCH, MUSCLE_SENSOR_L0_MM), add_channel,
                                step_reference)
                            cur = q_model_from_st(*cur_gui)
                            ref_l.extend(add_ref); axis_plan.extend(add_axis)
                            channel_plan.extend(add_channel)
                            length_ref_l.extend(add_length)
                            length_start_weight.extend([0.0]*len(add_ref))
                            point_plan.extend([-1]*len(add_ref))
                            axis_schedule_desc.append(
                                ("回零", desc, dead_ignored, small_ignored, x_ignored, return_dur))
                        else:
                            nseg = int(2.0/run_dt)
                            for j in range(1, nseg+1):
                                s5 = _s_curve5(j/nseg)
                                ref_l.append(np.zeros(3) if step_reference else cur*(1 - s5)); axis_plan.append(-1)
                                channel_plan.append(-1)
                                point_plan.append(-1)
                            cur = np.zeros(3)
                        n_added = len(ref_l) - n_before
                        if ctrl != "musclepi":
                            length_ref_l.extend([np.full(NCH, np.nan) for _ in range(n_added)])
                            length_start_weight.extend([0.0]*n_added)
                        for j in range(1, n_added+1):
                            raw_gui = (np.zeros(3) if step_reference else
                                       prev_raw_gui*(1.0-_s_curve5(j/n_added)))
                            plot_ref_l.append(q_model_from_st(*raw_gui))
                        prev_raw_gui = np.zeros(3)
                        # 零位驻留不放参考里——轮末走【强制放气 2s】（u=0 开环，见回绕处），
                        # 发散状态下闭环追零还是会顶电压，作者要的是真放气
                    BR["wrap_k"] = 0
                    BR["auto_hidden_return_to_zero"] = auto_hidden_return_to_zero
                    # 阶段编号只对应 GUI 中的示教点/示教段。同一段内依次调整 Y、φ、X
                    # 只更新阶段说明，不重复增加阶段总数，否则 4 段会被显示成 8/10 段。
                    stage_ids, stage_labels = [], []
                    axis_stage_names = ("Y轴", "φ轴", "X轴")
                    for point_i0, axis_code in zip(point_plan, axis_plan):
                        stage_no = int(point_i0) + 1 if int(point_i0) >= 0 else 0
                        base = (f"第{point_i0+1}段" if point_i0 >= 0 else "回零段")
                        if ctrl == "mpc":
                            label = base + "·三轴MPC跟踪"
                        elif ctrl == "musclepi" and axis_code >= 0:
                            label = base + f"·长度PI调整{axis_stage_names[axis_code]}"
                        elif ctrl == "musclepi":
                            label = base + "·长度PI保持"
                        elif axis_code >= 0:
                            label = base + f"·调整{axis_stage_names[axis_code]}"
                        else:
                            label = base + "·保持"
                        stage_ids.append(stage_no)
                        stage_labels.append(label)
                    BR["ref"] = np.array(ref_l)
                    BR["length_ref"] = np.array(length_ref_l)
                    BR["length_ref_base"] = BR["length_ref"].copy()
                    BR["length_start_weight"] = np.asarray(length_start_weight, dtype=float)
                    BR["length_ref_seeded"] = ctrl != "musclepi"
                    BR["plot_ref"] = np.array(plot_ref_l)
                    BR["axis_plan"] = np.asarray(axis_plan, dtype=int)
                    BR["channel_plan"] = np.asarray(channel_plan, dtype=int)
                    BR["point_plan"] = np.asarray(point_plan, dtype=int)
                    BR["segment_duration_s"] = np.asarray(
                        [float(item[1]) for item in pts_qm], dtype=float)
                    BR["stage_plan"] = np.asarray(stage_ids, dtype=int)
                    BR["stage_labels"] = stage_labels
                    BR["stage_total"] = max(1, len(pts))
                    BR["ctrl"] = ctrl
                    if BR.get("pi") is None:               # 方向表：PI 投影用
                        print("# 用模型算方向表（PI 用）…")
                        BR["pi"] = PIController(plant)
                    BR["pi"].set_gains(BR["pid"]["kp"], BR["pid"]["ki"], BR["pid"]["kd"],
                                       BR["pid"]["kv"], BR["pid"]["vmax"])
                    BR["pi"].set_dt(run_dt)
                    BR["pi"].reset()
                    if BR.get("muscle_pi") is None:
                        BR["muscle_pi"] = MuscleLengthPIController()
                    BR["muscle_pi"].set_gains(
                        BR["muscle_pid"]["kp"], BR["muscle_pid"]["ki"],
                        BR["muscle_pid"]["deadband"])
                    BR["muscle_pi"].set_dt(run_dt)
                    BR["muscle_pi"].reset()
                    # 角度 PI 需要固定物理一步预测器；肌肉长度 PI 只用角度做预先选轴，
                    # 实时闭环不使用角度模型。
                    BR["mpc"] = None if ctrl == "musclepi" else MPC(plant, N=NH)
                    BR["tss"] = 0.0; BR["tn"] = 0
                    BR["length_ess"] = 0.0; BR["length_en"] = 0
                    if ctrl == "mpc":                      # 数据锚定 MPC：预测模型来自真机数据
                        BR["mpc_cfg"] = {
                            "N": NH,
                            "w_q": tuple(float(v) for v in GN["mpc_w_q"]),
                            "w_v": tuple(float(v) for v in GN["mpc_w_v"]),
                            "rd": float(GN["mpc_rd"]),
                            "rho": float(GN["mpc_rho"]),
                            "l1": float(GN["mpc_l1"]),
                            "x_comp_ratio": float(GN.get("mpc_x_comp_ratio", 0.5)),
                            "lambda_c": float(GN.get("mpc_lambda_c", 0.005)),
                            "lock_lead_ms": float(GN.get("mpc_lock_lead_ms", 100.0)),
                            "solver": ("ltv" if (ilc_mpc or (codex_mpc and not plain_mpc_baseline)) else
                                       str(GN.get("mpc_solver", "mppi"))),
                            "nl_iters": int(GN.get("mpc_nonlinear_iterations", 20)),
                            "nl_lr_mv": float(GN.get("mpc_nonlinear_lr_mv", 180.0)),
                            "nl_budget_ms": float(GN.get("mpc_nonlinear_budget_ms", 40.0)),
                            "mppi_samples": int(GN.get("mpc_mppi_samples", 192)),
                            "mppi_iterations": int(GN.get("mpc_mppi_iterations", 2)),
                            "mppi_sigma_mv": float(GN.get("mpc_mppi_sigma_mv", 700.0)),
                            "mppi_temperature": float(GN.get("mpc_mppi_temperature", 1.0)),
                            "mppi_knots": int(GN.get("mpc_mppi_knots", 5)),
                            "sqp_iterations": int(GN.get("mpc_sqp_iterations", 2)),
                            "sqp_max_qp_iterations": int(GN.get("mpc_sqp_max_qp_iterations", 100)),
                            "acados_qp_solver": str(GN.get(
                                "mpc_acados_qp_solver", "PARTIAL_CONDENSING_HPIPM")),
                            "acados_cond_N": int(GN.get("mpc_acados_cond_N", 5)),
                            "acados_cmake_generator": str(GN.get(
                                "mpc_acados_cmake_generator", "")),
                            "acados_codegen_dir": str(GN.get(
                                "mpc_acados_codegen_dir", "")),
                            "acados_source_dir": str(GN.get(
                                "mpc_acados_source_dir", "")),
                        }
                        if ilc_mpc or (codex_mpc and not plain_mpc_baseline):
                            try:
                                _ensure_original_model_pair()
                                BR["dm"] = PureDataModel(
                                    ORIGINAL_FORWARD_PATH, ORIGINAL_FORWARD_META_PATH)
                            except Exception as exc:
                                branch = "Codex补偿器" if codex_mpc else "约束ILC"
                                self._out({"err": f"{branch}无法加载原始真正模型：{exc}"}); return
                            BR["forward_learner"] = None
                            BR["forward_last"] = None
                            BR["dm_kind"] = ("mpc_codex_original_ltv" if codex_mpc
                                             else "mpc_ilc_original_ltv")
                        elif BR.get("shadow_forward"):
                            from new_mpc_fel import (ShadowForwardLearner,
                                                     SequenceShadowForwardLearner)
                            if (BR.get("shadow_forward_only") and
                                    forward_arch in MPC_SHADOW_SEQUENCE_PATHS):
                                sequence_paths = MPC_SHADOW_SEQUENCE_PATHS[forward_arch]
                                learner = SequenceShadowForwardLearner(
                                    forward_arch,
                                    sequence_base_path=sequence_paths["base"],
                                    physical_base_path=PURE_DM_PATH,
                                    active_model_path=sequence_paths["active"],
                                    shadow_model_path=sequence_paths["shadow"],
                                    previous_model_path=sequence_paths["previous"],
                                    state_path=sequence_paths["state"])
                                BR["forward_meta_path"] = sequence_paths["meta"]
                                BR["dm_kind"] = f"shadow_forward_only_{forward_arch}"
                                shadow_name = f"mpc_shadow（10拍{forward_arch}，无逆模型独立分支）"
                                if str(BR["mpc_cfg"].get("solver", "")).lower() not in (
                                        "acados", "sqp_rti", "acados-sqp-rti",
                                        "acados_sqp_rti"):
                                    BR["mpc_cfg"]["solver"] = "ltv"
                            elif (BR.get("new_mpc_fel") and
                                    BR.get("model_source") == "original"):
                                learner = ShadowForwardLearner(
                                    base_model_path=ORIGINAL_FORWARD_PATH,
                                    active_model_path=ORIGINAL_INIT_FORWARD_ACTIVE_PATH,
                                    shadow_model_path=ORIGINAL_INIT_FORWARD_SHADOW_PATH,
                                    previous_model_path=ORIGINAL_INIT_FORWARD_PREVIOUS_PATH,
                                    state_path=ORIGINAL_INIT_FORWARD_STATE_PATH)
                                BR["forward_meta_path"] = ORIGINAL_FORWARD_META_PATH
                                BR["dm_kind"] = "shadow_forward_original_init"
                                shadow_name = "new_mpc_fel（原始模型初始化分支）"
                            elif BR.get("shadow_forward_only"):
                                learner = ShadowForwardLearner(
                                    base_model_path=PURE_DM_PATH,
                                    active_model_path=MPC_SHADOW_FORWARD_ACTIVE_PATH,
                                    shadow_model_path=MPC_SHADOW_FORWARD_SHADOW_PATH,
                                    previous_model_path=MPC_SHADOW_FORWARD_PREVIOUS_PATH,
                                    state_path=MPC_SHADOW_FORWARD_STATE_PATH)
                                BR["forward_meta_path"] = PURE_DM_META_PATH
                                BR["dm_kind"] = "shadow_forward_only"
                                shadow_name = "mpc_shadow（无逆模型独立分支）"
                            else:
                                learner = ShadowForwardLearner()
                                BR["forward_meta_path"] = PURE_DM_META_PATH
                                BR["dm_kind"] = "shadow_forward"
                                shadow_name = "new_mpc_fel"
                            BR["forward_learner"] = learner
                            BR["forward_last"] = None
                            if forward_arch in ("history_mlp", "gru"):
                                BR["dm"] = SequencePureDataModel(
                                    forward_arch,
                                    model_path=str(learner.active_path),
                                    meta_path=BR["forward_meta_path"])
                            else:
                                BR["dm"] = PureDataModel(
                                    model_path=str(learner.active_path),
                                    meta_path=BR["forward_meta_path"])
                            print(f"# {shadow_name}：真正模型 generation={learner.generation}；"
                                  "影子正模型只学习，每三轮训练后审核通过1次即可替换")
                        else:
                            BR["forward_learner"] = None
                            BR["forward_last"] = None
                            dm_stamp = _model_file_stamp(forward_arch)
                            if (BR.get("dm") is None or BR.get("dm_stamp") != dm_stamp or
                                    BR.get("dm_kind") != f"standard_{forward_arch}"):
                                BR["dm"] = data_model_ready(plant, forward_arch)  # 保存gains后，下次播放自动切模型
                                BR["dm_stamp"] = dm_stamp
                            BR["dm_kind"] = f"standard_{forward_arch}"
                            if forward_arch in ("history_mlp", "gru"):
                                if str(BR["mpc_cfg"].get("solver", "")).lower() not in (
                                        "acados", "sqp_rti", "acados-sqp-rti",
                                        "acados_sqp_rti"):
                                    BR["mpc_cfg"]["solver"] = "ltv"
                        BR["dampc"] = DAMPC(BR["dm"], **BR["mpc_cfg"])
                        if ilc_mpc:
                            try:
                                from mpc_ilc import ConstrainedILC
                                BR["ilc"] = ConstrainedILC(
                                    MPC_ILC_STATE_PATH, BR["ref"], BR["plot_ref"], run_dt,
                                    CLAMP_MV,
                                    learning_rate=float(GN.get("ilc_learning_rate", 0.15)),
                                    max_update_mv=float(GN.get("ilc_max_update_mv", 100.0)),
                                    smoothing_ticks=int(GN.get("ilc_smoothing_ticks", 5)),
                                    max_slew_mv_s=float(GN.get("ilc_max_slew_mv_s", 2500.0)),
                                    accept_ratio=float(GN.get("ilc_accept_ratio", 1.0)))
                                _ilc_buffers_reset()
                            except Exception as exc:
                                self._out({"err": f"约束ILC初始化失败：{exc}"}); return
                        else:
                            BR["ilc"] = None
                            _ilc_buffers_reset()
                        if codex_mpc:
                            try:
                                from codex_compensator import CodexCompensator
                                BR["codex_task_ref"] = np.asarray(
                                    [st3_from_q_model(row) for row in BR["plot_ref"]],
                                    dtype=float)
                                if replay_requested:
                                    CodexCompensator.reset_files(
                                        CODEX_REPLAY_STATE_PATH,
                                        CODEX_REPLAY_CANDIDATE_PATH,
                                        CODEX_REPLAY_REQUEST_PATH)
                                    state_path = CODEX_REPLAY_STATE_PATH
                                    candidate_path = CODEX_REPLAY_CANDIDATE_PATH
                                    request_path = CODEX_REPLAY_REQUEST_PATH
                                else:
                                    state_path = CODEX_COMP_STATE_PATH
                                    candidate_path = CODEX_COMP_CANDIDATE_PATH
                                    request_path = CODEX_COMP_REQUEST_PATH
                                BR["codex_comp"] = CodexCompensator(
                                    state_path, candidate_path, request_path,
                                    BR["ref"], BR["codex_task_ref"], run_dt, CLAMP_MV,
                                    accept_ratio=float(GN.get("codex_comp_accept_ratio", 1.0)),
                                    angle_jump_limit_deg=float(GN.get(
                                        "codex_comp_angle_jump_deg", 10.0)),
                                    target_mse_deg2=float(BR.get(
                                        "codex_reuse_mse_threshold_deg2", 15.0)),
                                    final_fall_slew_mv_s=float(GN.get(
                                        "codex_final_fall_slew_mv_s", 1200.0)),
                                    final_phi_rise_slew_mv_s=float(GN.get(
                                        "codex_final_phi_rise_slew_mv_s", 600.0)),
                                    unlock_pressure_kpa=PRESSURE_EMPTY_KPA)
                                if replay_requested:
                                    replay_rounds = replay_blob.get("rounds") or []
                                    replay_index = int(replay_blob.get("index", 0))
                                    replay_item = replay_rounds[replay_index]
                                    expected_hash = str(replay_blob.get("task_hash", ""))
                                    if (not expected_hash or
                                            BR["codex_comp"].task_hash != expected_hash):
                                        self._out({"err": "回放包轨迹与当前示教轨迹不一致；请重新载入回放包"}); return
                                    installed = BR["codex_comp"].install_replay_controller(
                                        replay_item["controller"],
                                        replay_item.get("generation", replay_index))
                                    BR["codex_begin_status"] = {
                                        **installed, "candidate": {"loaded": False},
                                        "replay": True, "round": replay_index + 1}
                                    BR["codex_replay_round_index"] = replay_index
                                    BR["codex_auto_status"] = {
                                        "state": "replay_testing",
                                        "message": (f"预设对比回放第{replay_index+1}/"
                                                    f"{len(replay_rounds)}轮控制器；不调用LLM"),
                                        "replay_round": replay_index + 1,
                                        "round_count": len(replay_rounds),
                                    }
                                    print(f"# MPC+LLM过程回放：第{replay_index+1}/"
                                          f"{len(replay_rounds)}轮，generation="
                                          f"{installed['generation']}")
                                else:
                                    BR["codex_begin_status"] = BR["codex_comp"].begin_round()
                                    candidate = BR["codex_begin_status"].get("candidate") or {}
                                    if candidate.get("loaded"):
                                        BR["codex_auto_status"] = {
                                            "state": "testing",
                                            "message": (f"正在实机审核候选generation "
                                                        f"{BR['codex_begin_status']['generation']}")}
                                        print(f"# Codex补偿器：加载候选 generation="
                                              f"{BR['codex_begin_status']['generation']}，本轮进行真机验证")
                                    elif candidate.get("error"):
                                        print(f"# ⚠ Codex候选未加载：{candidate.get('reason')}；本轮使用稳定版")
                                    elif (BR.get("codex_auto_status") or {}).get("state") == "retry":
                                        BR["codex_auto_status"] = {
                                            "state": "testing",
                                            "message": (f"正在重跑稳定补偿器 generation "
                                                        f"{BR['codex_begin_status']['generation']}；"
                                                        "有效完整轮结束后继续设计"),
                                            "retrying_invalid_lap": True}
                                        print(f"# Codex补偿器：正在重跑稳定 generation="
                                              f"{BR['codex_begin_status']['generation']}，"
                                              "有效轮结束后继续LLM设计")
                            except Exception as exc:
                                self._out({"err": f"Codex补偿器初始化失败：{exc}"}); return
                        else:
                            BR["codex_comp"] = None
                            BR["codex_begin_status"] = None
                            BR["codex_task_ref"] = None
                        BR["codex_u_comp"] = np.zeros(NCH)
                    BR["xd"] = np.zeros(12)                # 数据模型状态 [q, Δq, P]
                    # 纯 MPC 只读取并使用 DataModel；MPC+FEL 才采样并逐轮更新它。
                    BR["dmX"] = []; BR["dmY"] = []; BR["dmU"] = []; BR["dmPa"] = []; BR["dmPb"] = []
                    BR["dmBlk"] = [([], [])]               # MPC+FEL 气路多步拟合用的连续段
                    BR["dm_pend"] = None
                    BR["felPhi"], BR["felErr"], BR["felMask"] = [], [], []
                    # 上一拍发给固件的请求值。它不等于实际比例阀输出：锁状态 o/w 时固件强制输出 0。
                    BR["last_u_cmd"] = np.zeros(NCH)
                    # PI顺序控制：当前只调整一个轴；已经调好的其他轴保持切换瞬间压力。
                    BR["pi_hold_u"] = np.zeros(NCH)
                    BR["pi_hold_p"] = np.full(NCH, np.nan)
                    BR["pi_hold_i"] = np.zeros(NCH)
                    BR["pi_control_axis"] = None
                    BR["pi_control_channel"] = None
                    BR["pi_x_comp_ch"] = None
                    if BR.get("fel") and BR.get("nn") is None:
                        BR["nn"] = _fel_new()
                        print(f"# FEL PyTorch MLP 从零开始：6 路 {FEL_NF}→{FEL_H1}→{FEL_H2}→1")
                    # ── 学习层：FEL（作者 PPT 框架：逆模型在线训练）──────────────
                    # u = u_ff(逆模型) + u_fb(MPC/PI)；圈末用 u_fb 的一部分训练 NN。
                    # φ 同时包含参考、实测姿态/速度、误差、气压、上一拍输出与锁状态；
                    # 处于 w 预锁等待的通道不训练，六路独立网络互不污染。
                    BR["elog"] = np.zeros((len(BR["ref"]), 3))   # 圈末 RMS 统计用
                    BR["raw_elog"] = np.zeros((len(BR["ref"]), 3))
                    BR["completed_track_mse_deg2"] = None
                    BR["lap"] = 0
                    BR["k"] = 0; BR["n"] = 0; BR["x"] = np.zeros(12); BR["q_prev"] = None; BR["qref"] = None
                    # 每份训练/审核日志的 t 仍从 0 开始；控制响应窗口单独使用连续偏移。
                    # 这样不改变 Codex 每轮 response 的含义，只改变跨轮显示。
                    if preserve_response and TELEM:
                        try:
                            BR["telem_time_offset"] = float(TELEM[-1].get("t", 0.0)) + run_dt
                        except (TypeError, ValueError):
                            BR["telem_time_offset"] = 0.0
                    else:
                        BR["telem_time_offset"] = 0.0
                    BR["pi_stage_wait_s"] = 0.0
                    BR["pi_stage_wait_point"] = None
                    BR["stage_key"] = None; BR["stage_text"] = "准备开始"; BR["stage_point"] = -1
                    BR["frozen_n"] = 0; BR["last_fn"] = None; BR["last_qv"] = None; BR["venting"] = 0
                    BR["final_vent"] = False
                    BR["mode"] = "mpc"
                    os.makedirs(LOG_DIR, exist_ok=True)
                    log_prefix = ("mpc_codex" if codex_mpc else
                                  "mpc_ilc" if ilc_mpc else
                                  (("new_mpc_fel_original" if model_source == "original"
                                   else "new_mpc_fel") if new_mpc_fel else
                                  "mpc_shadow" if shadow_forward_only else ctrl))
                    BR["path"] = os.path.join(LOG_DIR, time.strftime(log_prefix + "_%Y%m%d_%H%M%S.csv"))
                    BR["log"] = []
                    if not preserve_response:
                        TELEM.clear()
                    # 可达性检查（单边+2.2V 下这些姿态撑不撑得住）：线性近似 G·u ≈ q，
                    # u∈[0,2000] 带界最小二乘，残差大 = 示教点超出可达域，警告但不拦。
                    warns = []
                    if default1_phi_v3_assist:
                        warns.append(
                            f"Default 1 第{execution_round}轮：phi主下降段启用V3回程辅助，"
                            "目标1000mV；保留100ms预锁与30mV/拍上升限制")
                    if codex_loop_requested:
                        warns.append(
                            "MPC+LLM过程回放按保存的轮次控制器自动连续执行，不调用LLM"
                            if replay_requested else
                            "Codex补偿器按完整单轮导出response；候选就绪后由Qt GUI自动开始下一轮")
                    if ctrl == "mpc" and BR.get("dampc") is not None:
                        requested_solver = str(getattr(
                            BR["dampc"], "solver_requested", BR["dampc"].solver))
                        if requested_solver != BR["dampc"].solver:
                            warns.append(
                                f"请求求解器 {requested_solver} 不可用，实际使用 "
                                f"{BR['dampc'].solver}；请看终端中的依赖错误")
                    try:
                        if ctrl == "musclepi":
                            raise StopIteration
                        if BR.get("G") is None:
                            print("# 建可达性矩阵（每路 600mV 平衡差分）…")
                            Gm = np.zeros((3, NCH))
                            for ch in range(NCH):
                                uv = np.zeros(NCH); uv[ch] = 600.0
                                Gm[:, ch] = _sim_settle(plant, uv) / 600.0
                            BR["G"] = Gm
                        from scipy.optimize import lsq_linear
                        for i, pt in enumerate(pts):
                            v3 = [float(v) for v in (list(pt["q"]) + [0, 0, 0])[:3]]
                            qm = q_model_from_st(v3[0], v3[1], v3[2], args.mount)
                            sol = lsq_linear(BR["G"], qm, bounds=(CO_CONTRACT_MV, CLAMP_MV))
                            res = math.degrees(np.linalg.norm(BR["G"] @ sol.x - qm))
                            if res > 5.0:
                                warns.append(f"点{i+1} 超出可达域约 {res:.0f}°（单边拉+2.2V 撑不住）")
                    except StopIteration:
                        pass
                    except Exception as e:
                        print(f"# 可达性检查跳过：{e}")
                    for w in warns: print("#  ⚠ " + w)
                    for seg, old_t, new_t in speed_adjustments:
                        print(f"# 速度规划：第{seg}段按实测速度由 {old_t:.2f}s 延长到 {new_t:.2f}s")
                    if ctrl in ("pi", "musclepi"):
                        axis_names = ("Y", "φ", "X")
                        for seg, desc, dead_ignored, small_ignored, x_ignored, total_t in axis_schedule_desc:
                            parts = [f"{key} {abs(deg):.1f}°/{dur:.2f}s" for key, deg, dur in desc]
                            msg = " → ".join(parts) if parts else "保持"
                            if dead_ignored:
                                msg += (f"；忽略≤{PI_PLAN_DEADBAND_DEG:g}°：" +
                                        ", ".join(axis_names[a] for a in dead_ignored))
                            if small_ignored:
                                msg += ("；三轴中最小者不调整：" +
                                        ", ".join(axis_names[a] for a in small_ignored))
                            if x_ignored:
                                msg += "；X非最大变化，禁用肌肉9/10"
                            plan_name = "肌肉PI顺序规划" if ctrl == "musclepi" else "PI顺序规划"
                            print(f"# {plan_name}：第{seg}段 {msg}，规划 {total_t:.2f}s，"
                                  f"最长 {total_t*(1.0+PI_STAGE_EXTENSION_RATIO):.2f}s")
                    learn_desc = ("真实长度闭环，角度负责顺序选轴" if ctrl == "musclepi" else
                                  "原始真正模型+LTV冻结，Codex按每轮response离线设计受限补偿器" if codex_mpc else
                                  "原始真正模型+LTV冻结，逐轮真实误差审核ILC前馈" if ilc_mpc else
                                  "原始正/逆模型初始化分支 + 影子学习"
                                  if new_mpc_fel and model_source == "original" else
                                  "最新已验证正/逆模型 + 影子学习"
                                  if new_mpc_fel else
                                  "真正模型控制 + 影子正模型学习（无逆模型）" if shadow_forward_only else
                                  "顺模型冻结 + 逆模型NN逐轮学习" if ctrl == "mpc" and BR["fel"] else
                                  "NN 逐轮学习" if BR["fel"] else
                                  "DataModel/权重冻结")
                    ctrl_gain_desc = ""
                    if ctrl == "pi":
                        ctrl_gain_desc = (
                            f"PID kp={GN['pi_kp']:g} ki={GN['pi_ki']:g} kd={GN.get('pi_kd', 0.0):g} "
                            f"speed kv={GN.get('pi_kv', PI_KV):g} "
                            f"vmax={GN.get('pi_vmax_deg', PI_VMAX_DEG):g}°/s "
                            f"du_max={PI_DU_MAX_MV_PER_S:g}mV/s "
                            f"single-axis hold={PI_AXIS_MIN_HOLD_S:g}s "
                            f"hyst={PI_AXIS_HYST_DEG:g}° dead={PI_AXIS_DONE_DEG:g}° "
                            f"pressure-hold kp={BR['pressure_pid']['kp']:g} "
                            f"ki={BR['pressure_pid']['ki']:g} "
                            f"dead={BR['pressure_pid']['deadband']:g}kPa "
                            f"x-comp 9+67={_x_phi_comp_ratio(4):g}x "
                            f"10+23={_x_phi_comp_ratio(5):g}x ")
                    elif ctrl == "musclepi":
                        ctrl_gain_desc = (
                            f"length-PI kp={BR['muscle_pid']['kp']:g}mV/mm "
                            f"ki={BR['muscle_pid']['ki']:g}mV/(mm*s) "
                            f"dead={BR['muscle_pid']['deadband']:g}mm "
                            f"du_max={PI_DU_MAX_MV_PER_S:g}mV/s "
                            f"pressure-hold kp={BR['pressure_pid']['kp']:g} "
                            f"ki={BR['pressure_pid']['ki']:g} "
                            f"dead={BR['pressure_pid']['deadband']:g}kPa "
                            f"x-comp 9+67={_x_phi_comp_ratio(4):g}x "
                            f"10+23={_x_phi_comp_ratio(5):g}x ")
                    elif ctrl == "mpc":
                        solver_name = str(getattr(BR.get("dampc"), "solver",
                                                  GN.get("mpc_solver", "mppi"))).lower()
                        if solver_name == "nonlinear":
                            ctrl_gain_desc = (
                                f"predict=真正模型直接非线性滚动 "
                                f"opt≤{int(GN.get('mpc_nonlinear_iterations', 20))}iter/"
                                f"{float(GN.get('mpc_nonlinear_budget_ms', 40.0)):g}ms ")
                        elif solver_name == "mppi":
                            ctrl_gain_desc = (
                                f"predict=MPPI无梯度批量滚动 "
                                f"S={int(GN.get('mpc_mppi_samples', 192))} "
                                f"iter≤{int(GN.get('mpc_mppi_iterations', 2))} "
                                f"sigma={float(GN.get('mpc_mppi_sigma_mv', 700.0)):g}mV "
                                f"K={int(GN.get('mpc_mppi_knots', 5))}/"
                                f"{float(GN.get('mpc_nonlinear_budget_ms', 40.0)):g}ms ")
                        elif solver_name == "sqp":
                            ctrl_gain_desc = (
                                f"predict=SQP-RTI+OSQP "
                                f"SQP≤{int(GN.get('mpc_sqp_iterations', 2))}iter "
                                f"QP≤{int(GN.get('mpc_sqp_max_qp_iterations', 100))}iter/"
                                f"{float(GN.get('mpc_nonlinear_budget_ms', 40.0)):g}ms ")
                        elif solver_name == "acados_sqp_rti":
                            ctrl_gain_desc = (
                                f"predict=acados SQP-RTI "
                                f"QP={str(GN.get('mpc_acados_qp_solver', 'PARTIAL_CONDENSING_HPIPM'))} "
                                f"condN={int(GN.get('mpc_acados_cond_N', 5))} ")
                        else:
                            ctrl_gain_desc = "predict=LTV逐拍线性化 "
                        ctrl_gain_desc += (
                            f"9→67软补偿≥{float(GN.get('mpc_x_comp_ratio', 0.5)):g}x"
                            f"(λc={float(GN.get('mpc_lambda_c', 0.005)):g}) ")
                        if ilc_mpc and BR.get("ilc") is not None:
                            ilc_status = BR["ilc"].status()
                            ctrl_gain_desc += (
                                f"ILC gen={ilc_status['generation']} "
                                f"α={float(GN.get('ilc_learning_rate', 0.15)):g} "
                                f"Δu≤{float(GN.get('ilc_max_update_mv', 100.0)):g}mV/轮 "
                                f"ff_peak={ilc_status['ff_peak_mv']:.0f}mV ")
                        if codex_mpc and BR.get("codex_comp") is not None:
                            codex_status = BR["codex_comp"].status()
                            ctrl_gain_desc += (
                                f"Codex gen={codex_status['generation']} "
                                "参数与补偿不限幅（仅最终电压截断） ")
                    if ctrl == "musclepi":
                        print(f"# GUI 启动肌肉长度 PI：{len(pts)} 点，{len(BR['ref'])*run_dt:.0f}s"
                              f"，{1/run_dt:.0f}Hz｜参考 {reference_mode.upper()}｜{learn_desc}｜{ctrl_gain_desc}"
                              f"u_max={CLAMP_MV:.0f}mV")
                    else:
                        mode_name = ("MPC_CODEX" if codex_mpc else
                                     "MPC_ILC" if ilc_mpc else
                                     "NEW_MPC_FEL" if new_mpc_fel else
                                     "MPC_SHADOW" if shadow_forward_only else ctrl.upper())
                        print(f"# GUI 启动 {mode_name}：{len(pts)} 点，{len(BR['ref'])*run_dt:.0f}s"
                              f"，{1/run_dt:.0f}Hz｜映射 {mount}"
                              f"｜参考 {reference_mode.upper()}"
                              f"｜{learn_desc}"
                              f"｜增益 N={NH} w_q={GN['mpc_w_q']} w_v={GN['mpc_w_v']} "
                              f"rd={GN['mpc_rd']:g} rho={GN['mpc_rho']:g} L1={GN['mpc_l1']:g} "
                              f"{ctrl_gain_desc}u_max={CLAMP_MV:.0f}mV"
                              + (f" lock={float(GN.get('mpc_lock_lead_ms', 100.0)):g}ms"
                                 if ctrl == "mpc" else ""))
                    self._out({"ok": 1, "warn": warns,
                               "planned_duration_s": round(len(BR["ref"])*run_dt, 2),
                               "segment_timing": [
                                   {"segment": i+1,
                                    "requested_s": round(float(pt.get("t", 2)), 2),
                                    "planned_s": round(float(pts_qm[i][1]), 2),
                                    "max_s": round(float(pts_qm[i][1]) *
                                                   (1.0+PI_STAGE_EXTENSION_RATIO), 2)
                                             if ctrl in ("pi", "musclepi") else round(float(pts_qm[i][1]), 2)}
                                   for i, pt in enumerate(pts)
                               ],
                               "sequential_axis_plan": ctrl in ("pi", "musclepi"),
                               "muscle_length_control": ctrl == "musclepi",
                               "reference_mode": reference_mode,
                               "forward_model_arch": BR.get("forward_arch", "mlp"),
                               "new_mpc_fel": new_mpc_fel,
                                "shadow_forward_only": shadow_forward_only,
                                "ilc_mpc": ilc_mpc,
                                "codex_compensator": codex_mpc,
                                "codex_status": (BR["codex_comp"].status()
                                                 if BR.get("codex_comp") is not None else None),
                               "ilc_status": (BR["ilc"].status()
                                              if ilc_mpc and BR.get("ilc") is not None else None),
                               "model_source": model_source,
                               "models_frozen": BR.get("models_frozen", False),
                               "forward_generation": (BR["forward_learner"].generation
                                                      if BR.get("forward_learner") is not None
                                                      else 0 if shadow_forward else None)}); return
                if self.path == "/stop":
                    if BR["mode"] == "mpc": _finish("GUI 停止")
                    self._out({"u": [0]*NCH, "mode": "idle"}); return
                if self.path == "/telem":              # 开环遥测：页面直发，仅供响应窗口绘图
                    if BR["mode"] != "mpc":            # MPC/PI 运行时不让开环流搅局
                        qv = np.array(data.get("qrel", []), float)
                        if qv.size >= 4:
                            qv = qv/max(1e-9, np.linalg.norm(qv))
                            q = q_model_from_quat(qv, np.array([1.0, 0, 0, 0]), args.mount)
                            r3 = [float(v) for v in (list(data.get("ref", [])) + [0, 0, 0])[:3]]
                            r = q_model_from_st(r3[0], r3[1], r3[2], args.mount)
                            now = time.time()
                            if now - BR.get("ol_t", 0) > 1.0:   # 新一轮开环回放：清屏重画
                                TELEM.clear(); BR["ol_n"] = 0
                            BR["ol_t"] = now
                            tg = BR.get("ol_n", 0)*DT; BR["ol_n"] = BR.get("ol_n", 0) + 1
                            TELEM.append({"t": tg, "wall": now, "source": "control",
                                          "q": [math.degrees(v) for v in q],
                                          "r": [math.degrees(v) for v in r],
                                          "q_gui": list(st3_from_q_model(q)),
                                          "r_gui": list(st3_from_q_model(r)),
                                          "u": [int(v) for v in (list(data.get("u", [])) + [0]*NCH)[:NCH]],
                                          "e": math.degrees(np.linalg.norm(r - q))})
                    self._out({"ok": 1}); return
                if self.path == "/step":
                    if BR["mode"] != "mpc": self._out({"u": [0]*NCH, "mode": BR["mode"]}); return
                    run_dt = float(BR.get("dt", DT))
                    qv = np.array(data.get("quat", []), float)
                    if qv.size < 4: self._out({"u": [0]*NCH, "mode": "mpc"}); return
                    qv = qv/max(1e-9, np.linalg.norm(qv))
                    # 帧流冻结检测（改 2026-09-04）：按 GUI 报的帧计数 n 判——计数 3s 不涨
                    # 才是固件真没发新帧。旧判据"四元数逐位相同 3s"在臂静止时误杀：BNO055
                    # 静止输出可以一位不变，轮末放气 2s + 慢起步 > 3s 全同，每轮末都被当成
                    # 僵死停掉（作者实机："点了循环没有用"）。旧 GUI 不报 n 时才退回旧判据。
                    fn = data.get("n")
                    if isinstance(fn, (int, float)):
                        frozen = (fn == BR.get("last_fn"))
                        BR["last_fn"] = fn
                    else:
                        frozen = np.array_equal(qv, BR.get("last_qv"))
                    if frozen:
                        BR["frozen_n"] = BR.get("frozen_n", 0) + 1
                        if BR["frozen_n"] == max(1, int(round(3.0/run_dt))):
                            print("# !! 3 秒没有新 IMU 帧——帧流冻结（串口断了？固件僵死？）。已停止本次运行")
                            for kk in ("felPhi", "felErr", "felMask", "dmX", "dmY", "dmU", "dmPa", "dmPb"):
                                BR.pop(kk, None)               # 冻结段样本不是真机数据，不训练不落盘
                            _finish("IMU 帧流冻结，停止")
                            self._out({"u": [0]*NCH, "mode": "idle"}); return
                    else:
                        BR["frozen_n"] = 0
                    BR["last_qv"] = qv.copy()
                    if BR["qref"] is None: BR["qref"] = qv.copy()      # 首拍对零
                    q = q_model_from_quat(qv, BR["qref"], args.mount)
                    # 角度急停已按作者要求取消（2026-09-03）：安全由气源侧最大气压兜底
                    # ── 轮间强制放气窗：u=0 开环 2s，不跑控制器、不采样、不学习 ──
                    if BR.get("venting", 0) > 0:
                        BR["venting"] -= 1
                        vent_round = max(1, int(BR.get("lap", 0)))
                        finish_after_vent = bool(BR.get("final_vent", False))
                        hidden_return = bool(BR.get("auto_hidden_return_to_zero", True))
                        vent_plot_model = (np.zeros(3) if hidden_return else
                                           np.asarray(BR["plot_ref"][-1], dtype=float))
                        vent_control_model = (np.zeros(3) if hidden_return else
                                              np.asarray(BR["ref"][-1], dtype=float))
                        vent_ref_gui = list(st3_from_q_model(vent_plot_model))
                        vent_control_gui = list(st3_from_q_model(vent_control_model))
                        endpoint_text = ("隐藏终点·(0,0,0)" if hidden_return else "末点")
                        vent_text = (f"{endpoint_text}·全机放气" if finish_after_vent else
                                     f"第{vent_round}轮·{endpoint_text}·全机放气")
                        stage_payload = _stage_update(
                            ("vent", vent_round, finish_after_vent), vent_text, -1)
                        tg = BR.get("n", 0)*run_dt
                        BR["n"] = BR.get("n", 0) + 1
                        vent_pressure = data.get("kpa")
                        if not isinstance(vent_pressure, list) or len(vent_pressure) < NCH:
                            vent_pressure = [None]*NCH
                        vent_length = _muscle_lengths_from_lens(data.get("lens"))
                        vent_group_displacement = _group_signed_displacements_from_lens(
                            data.get("lens"))
                        TELEM.append({"t": BR.get("telem_time_offset", 0.0) + tg,
                                      "wall": time.time(), "source": "control",
                                      "q": [math.degrees(v) for v in q],
                                      "r": [math.degrees(v) for v in vent_plot_model],
                                      "q_gui": list(st3_from_q_model(q)),
                                      "r_gui": vent_ref_gui, "u": [0]*NCH,
                                      "pressure_kpa": list(vent_pressure[:NCH]),
                                      "channel_length_mm": [round(float(v), 3) if np.isfinite(v) else None
                                                            for v in vent_length],
                                      "l15_displacement_raw_mm": [
                                          round(float(v), 3) if np.isfinite(v) else None
                                          for v in vent_group_displacement[0]],
                                      "group_displacement_raw_mm": [
                                          [round(float(v), 3) if np.isfinite(v) else None
                                           for v in pair]
                                          for pair in vent_group_displacement],
                                      "force_gui_yzx_n": data.get("force_gui_yzx_n"),
                                      "torque_xyz_nm": data.get("torque_xyz_nm"),
                                      "force_pred_xyz_n": data.get("force_pred_xyz_n"),
                                      "e": math.degrees(np.linalg.norm(vent_control_model-q)),
                                      **stage_payload})
                        if BR["venting"] == 0:                 # 放气完毕才复位瞬态：每轮从干净初始态出发
                            m_ = BR.get("mpc")
                            if m_ is not None:
                                m_.d[:] = 0; m_.q_pred = None; m_.u_prev[:] = 0
                            if BR.get("dampc") is not None:
                                BR["dampc"].reset_runtime()
                            BR["xd"][:] = 0; BR["dm_pend"] = None
                            BR["forward_last"] = None
                            if BR.get("forward_learner") is not None:
                                BR["forward_learner"].discard_pending()
                            BR["last_u_cmd"][:] = 0
                            BR["pi_hold_u"][:] = 0
                            BR["pi_hold_p"][:] = np.nan
                            BR["pi_hold_i"][:] = 0
                            BR["pi_control_axis"] = None
                            BR["pi_control_channel"] = None
                            BR["pi_x_comp_ch"] = None
                            if BR.get("dmBlk") and BR["dmBlk"][-1][0]: BR["dmBlk"].append(([], []))
                            BR["x"][:] = 0; BR["q_prev"] = None
                            if BR.get("pi"): BR["pi"].reset()
                            if BR.get("muscle_pi"): BR["muscle_pi"].reset()
                            if BR.get("ctrl") == "musclepi" and not finish_after_vent:
                                BR["length_ref"] = BR["length_ref_base"].copy()
                                BR["length_ref_seeded"] = False
                            if finish_after_vent:
                                BR["final_vent"] = False
                                _finish("轨迹完成")
                            else:
                                print(f"# 放气完毕，第 {BR['lap']+1} 轮开始")
                        vent_out = {"u": [0]*NCH,
                                    "mode": "done" if finish_after_vent and BR["venting"] == 0 else "mpc",
                                    "t": round(tg, 2),
                                    "ref": vent_ref_gui,
                                    "control_ref": vent_control_gui,
                                    "meas": list(st3_from_q_model(q)),
                                    **stage_payload}
                        if BR.get("ctrl") == "mpc":
                            vent_out["lock"] = [0]*NCH
                        self._out(vent_out); return
                    qd = (q - BR["q_prev"])/run_dt if BR["q_prev"] is not None else np.zeros(3)
                    BR["q_prev"] = q
                    k = BR["k"]; ref = BR["ref"]
                    r_now = ref[min(k, len(ref)-1)]
                    ki = min(k, len(ref)-1)
                    plot_ref = BR.get("plot_ref", ref)
                    r_plot = plot_ref[min(ki, len(plot_ref)-1)]
                    r_next = ref[min(ki+1, len(ref)-1)]
                    length_now = _muscle_lengths_from_lens(data.get("lens"))
                    group_displacement_raw = _group_signed_displacements_from_lens(
                        data.get("lens"))
                    length_plan = BR.get("length_ref")
                    if BR.get("ctrl") == "musclepi" and not BR.get("length_ref_seeded", False):
                        # 235mm 是名义原长，不应被当作每次播放的精确起始反馈值。
                        # 用首个完整传感器帧重建第一段起点，权重到第一个示教点衰减为0，
                        # 所以只消除开机跳变，不改变用户记录的任何示教终点。
                        if not np.all(np.isfinite(length_now)):
                            missing = [i for i, v in enumerate(length_now) if not np.isfinite(v)]
                            wait_stage = _stage_update(
                                ("length-sensor-wait", tuple(missing)),
                                "等待六组肌肉长度传感器", -1)
                            self._out({"u": [0]*NCH, "mode": "mpc", "t": round(k*run_dt, 2),
                                       "ref": list(st3_from_q_model(r_plot)),
                                       "control_ref": list(st3_from_q_model(r_now)),
                                       "meas": list(st3_from_q_model(q)),
                                       "length": [round(float(v), 3) if np.isfinite(v) else None
                                                  for v in length_now],
                                       **wait_stage}); return
                        weights = np.asarray(BR.get("length_start_weight", []), float)
                        if length_plan is not None and len(weights) == len(length_plan):
                            length_plan += weights[:, None] * (length_now-MUSCLE_SENSOR_L0_MM)[None, :]
                        BR["length_ref_seeded"] = True
                        print("# 肌肉长度 PI：第一段已用播放首拍实测长度对齐，起步输出不会追名义235mm")
                    length_target = (length_plan[min(ki, len(length_plan)-1)].copy()
                                     if length_plan is not None and len(length_plan)
                                     else np.full(NCH, np.nan))
                    length_error = length_now - length_target
                    plan = BR.get("axis_plan")
                    plan_code = int(plan[ki]) if plan is not None and ki < len(plan) else -1
                    planned_axis = plan_code if plan_code >= 0 else -1
                    channels = BR.get("channel_plan")
                    planned_channel = (int(channels[ki])
                                       if channels is not None and ki < len(channels) else -1)
                    stage_no = int(BR["stage_plan"][ki]) if ki < len(BR.get("stage_plan", [])) else 1
                    stage_label = (BR["stage_labels"][ki]
                                   if ki < len(BR.get("stage_labels", [])) else "轨迹跟踪")
                    stage_point = (int(BR["point_plan"][ki])
                                   if ki < len(BR.get("point_plan", [])) else -1)
                    if stage_point >= 0:
                        stage_text = (f"第{BR.get('lap', 0)+1}轮·阶段{stage_no}/"
                                      f"{max(1, BR.get('stage_total', 1))}：{stage_label}")
                    else:
                        stage_text = f"第{BR.get('lap', 0)+1}轮·{stage_label}"
                    hold_ref = False
                    if BR.get("ctrl") in ("pi", "musclepi"):
                        # 延长预算按整个 GUI 段共享，而不是每个顺序轴各加30%。例如该段
                        # 基础规划3.50s，无论段内安排一个还是两个轴，总时长都不超过4.55s。
                        if BR.get("pi_stage_wait_point") != stage_point:
                            BR["pi_stage_wait_point"] = stage_point
                            BR["pi_stage_wait_s"] = 0.0
                        next_i = min(ki+1, len(plan)-1) if plan is not None else ki
                        next_code = int(plan[next_i]) if plan is not None else -1
                        next_channel = (int(channels[next_i])
                                        if channels is not None and next_i < len(channels) else -1)
                        point_plan_now = BR.get("point_plan")
                        next_point = (int(point_plan_now[next_i])
                                      if point_plan_now is not None and next_i < len(point_plan_now)
                                      else stage_point)
                        at_axis_end = (planned_axis >= 0 and
                                       (ki >= len(plan)-1 or next_code != planned_axis or
                                        next_channel != planned_channel or
                                        next_point != stage_point))
                        if at_axis_end:
                            if BR.get("ctrl") == "musclepi":
                                signed_stage_err = (float(MUSCLE_PI_SENSOR_SIGN[planned_channel] *
                                                          length_error[planned_channel])
                                                    if planned_channel >= 0 else 0.0)
                                stage_err = max(0.0, signed_stage_err)
                                stage_crossed = signed_stage_err <= 0.0
                                done_threshold = float(BR["muscle_pid"]["deadband"])
                                err_unit = "mm"
                            else:
                                signed_stage_err = float(r_now[planned_axis]-q[planned_axis])
                                stage_err = abs(math.degrees(signed_stage_err))
                                stage_crossed = (planned_channel >= 0 and
                                                 PID_AXIS[planned_channel] == planned_axis and
                                                 PID_SIGN[planned_channel]*signed_stage_err <= 0.0)
                                done_threshold = PI_STAGE_DONE_DEG
                                err_unit = "°"
                            durations = BR.get("segment_duration_s", [])
                            base_s = (float(durations[stage_point])
                                      if 0 <= stage_point < len(durations) else 0.0)
                            budget_s = PI_STAGE_EXTENSION_RATIO * base_s
                            used_s = float(BR.get("pi_stage_wait_s", 0.0))
                            needs_more = (np.isfinite(stage_err) and
                                          stage_err > done_threshold and not stage_crossed)
                            if needs_more and used_s + run_dt <= budget_s + 1e-9:
                                hold_ref = True
                                used_s += run_dt
                                BR["pi_stage_wait_s"] = used_s
                                r_next = r_now
                                stage_text += (f"·等待到位 +{used_s:.2f}/"
                                               f"{budget_s:.2f}s")
                            elif needs_more and budget_s > 0.0:
                                print(f"# ⚠ 第{stage_no}段已用完30%延长预算，"
                                      f"当前通道仍差{stage_err:.1f}{err_unit}，按时进入下一阶段")
                    stage_payload = _stage_update(
                        (BR.get("lap", 0), stage_no, bool(hold_ref), planned_axis, planned_channel),
                        stage_text, stage_point)
                    r_vel = (r_next-r_now)/run_dt
                    fel_on = bool(BR.get("fel", True))
                    # 实测气压（固件帧 v2 第 8 段：AD7606 读 regulator 的 1–5V 监测口）——
                    # 有就直接当气压状态，不再靠 τ_up/τ_dn 推算；旧固件/无 ADC 传 null，退回推算
                    P_meas = None
                    kpa = data.get("kpa")
                    if isinstance(kpa, list) and len(kpa) >= NCH:
                        try:
                            P_meas = np.clip(np.array([float(v) for v in kpa[:NCH]]), 0.0, P_MAX_KPA)
                            if not np.all(np.isfinite(P_meas)): P_meas = None
                        except (TypeError, ValueError):
                            P_meas = None
                    lock_vec, lock_st, lock_valid = _fel_lock_state(data.get("locks") or HOST_LOCKS)
                    qd_alpha = PID_D_ALPHA**(run_dt/DT)
                    x = BR["x"]; x[:3] = q; x[3:6] = qd_alpha*x[3:6] + (1.0-qd_alpha)*qd
                    qd_f = x[3:6].copy()
                    last_u_cmd = np.asarray(BR.get("last_u_cmd", np.zeros(NCH)), float)
                    lock_cmd = np.zeros(NCH, dtype=int)
                    codex_applied = np.zeros(NCH, dtype=float)
                    phi_v3_assist_mv = 0.0
                    phi_v3_assist_tracking = False
                    phi_v3_assist_override = False
                    # 固件只有在 L(LOCKED) 才把请求送到比例阀；o(OPEN)/w(LOCKING) 的实际输出都是 0。
                    # 每拍用实测锁状态覆盖 MPC 的上一输入，避免100ms预锁期被模型误认为已经充气。
                    fallback_ctrl = (BR.get("dampc") if BR.get("ctrl") == "mpc"
                                     else BR.get("mpc"))
                    u_prev_effective = (np.where(np.asarray(lock_st) == "L", last_u_cmd, 0.0)
                                        if lock_valid else
                                        (fallback_ctrl.u_prev.copy() if fallback_ctrl is not None
                                         else last_u_cmd.copy()))
                    if BR.get("ctrl") == "mpc" and lock_valid:
                        BR["dampc"].u_prev[:] = u_prev_effective
                    if P_meas is not None and BR.get("ctrl") == "mpc":
                        dm_ = BR["dm"]
                        P_state = dm_.p_estimate(P_meas, u_prev_effective, BR["xd"][6:]) \
                            if any(dm_.p_dead) else P_meas.copy()
                    elif P_meas is not None:
                        P_state = P_meas.copy()
                    else:
                        P_state = (BR["xd"][6:] if BR.get("ctrl") == "mpc" else x[6:]).copy()
                    phi_now = _fel_phi(r_now, r_vel, q, qd_f, P_state,
                                       last_u_cmd, lock_vec)
                    u_ff_now = _fel_predict(BR["nn"], phi_now) if fel_on else np.zeros(NCH)
                    # ── FEL 集成（作者定五模式 2026-09-04：纯MPC / MPC+FEL / 纯PI / PI+FEL / 开环）──
                    # fel=False：u_ff≡0、不采样、不训练，就是纯反馈控制器。
                    # PI+FEL：经典 Kawato 叠加 u=u_ff+u_fb，教师=u_fb。
                    # MPC+FEL：u_ff 不是外挂——整段未来前馈序列作为【已知名义输入】进 QP 预测，
                    #   QP 只解其上的反馈修正 v，ρ 罚 v（前馈学好→反馈自然缩零）。教师=v。
                    #   训练特征与 QP 实际采用的第一拍前馈严格对齐，不再错开一拍。
                    if BR.get("ctrl") == "musclepi":
                        # 角度示教只决定预先规划好的轴和方向；本拍只允许该物理通道使用
                        # 真实肌肉长度 PI。其他已完成通道按交棒瞬间的实测压力闭环保持。
                        allowed = np.zeros(NCH, dtype=bool)
                        if planned_channel >= 0:
                            allowed[planned_channel] = True
                        candidate, length_error = BR["muscle_pi"].step(
                            length_now, length_target, allowed_channels=allowed)
                        control_axis = planned_axis if planned_channel >= 0 else None
                        x_comp_ch = (PI_X_PHI_COMP_CHANNEL.get(int(planned_channel))
                                     if control_axis == 2 else None)
                        if x_comp_ch is not None:
                            candidate[x_comp_ch] = np.clip(
                                _x_phi_comp_ratio(planned_channel)*candidate[planned_channel],
                                0.0, CLAMP_MV)

                        hold_u = np.asarray(BR.get("pi_hold_u", np.zeros(NCH)), float).copy()
                        hold_p = np.asarray(BR.get("pi_hold_p", np.full(NCH, np.nan)), float).copy()
                        hold_i = np.asarray(BR.get("pi_hold_i", np.zeros(NCH)), float).copy()
                        prev_axis = BR.get("pi_control_axis")
                        prev_channel = BR.get("pi_control_channel")

                        # 换轴或同轴换方向时，冻结刚完成通道的当前实测压力。X 的φ补偿通道
                        # 与主通道一起交棒。低于30kPa按无有效气压处理，不建立保持目标。
                        if prev_channel is not None and (
                                control_axis != prev_axis or planned_channel != prev_channel):
                            old_channels = [int(prev_channel)]
                            old_x_comp = BR.get("pi_x_comp_ch")
                            if old_x_comp is not None and int(old_x_comp) not in old_channels:
                                old_channels.append(int(old_x_comp))
                            for ch in old_channels:
                                if P_meas is not None and float(P_meas[ch]) > PRESSURE_EMPTY_KPA:
                                    p_hold = float(P_meas[ch])
                                    hold_p[ch] = p_hold
                                    hold_i[ch] = 0.0
                                    hold_u[ch] = np.clip(p_hold/KPA_PER_MV, 0.0, CLAMP_MV)
                                elif P_meas is None and last_u_cmd[ch] > 0.5:
                                    hold_p[ch] = np.nan
                                    hold_i[ch] = 0.0
                                    hold_u[ch] = last_u_cmd[ch]
                                else:
                                    hold_p[ch] = np.nan
                                    hold_i[ch] = 0.0
                                    hold_u[ch] = 0.0

                        # 当前计划轴接管对应拮抗对，只保留计划方向；长度达到目标或目标要求
                        # 拉长时该方向自然归零，不会临时切到另一侧。
                        if control_axis is not None:
                            pair = np.flatnonzero(PID_AXIS == int(control_axis))
                            hold_p[pair] = np.nan
                            hold_i[pair] = 0.0
                            hold_u[pair] = 0.0
                            hold_u[planned_channel] = candidate[planned_channel]
                            if int(control_axis) == 2:
                                phi_pair = np.flatnonzero(PID_AXIS == 1)
                                hold_p[phi_pair] = np.nan
                                hold_i[phi_pair] = 0.0
                                hold_u[phi_pair] = 0.0
                                if x_comp_ch is not None:
                                    hold_u[x_comp_ch] = candidate[x_comp_ch]

                        if P_meas is not None:
                            pressure_pid = BR["pressure_pid"]
                            for ch in range(NCH):
                                if not np.isfinite(hold_p[ch]):
                                    continue
                                err_p = float(hold_p[ch]-P_meas[ch])
                                p_dead = float(pressure_pid["deadband"])
                                err_eff = (0.0 if abs(err_p) <= p_dead
                                           else math.copysign(abs(err_p)-p_dead, err_p))
                                i_try = np.clip(
                                    hold_i[ch] + float(pressure_pid["ki"])*err_eff*run_dt,
                                    -PRESSURE_HOLD_I_MAX_MV, PRESSURE_HOLD_I_MAX_MV)
                                u_raw = (hold_p[ch]/KPA_PER_MV +
                                         float(pressure_pid["kp"])*err_eff + i_try)
                                u_sat = float(np.clip(u_raw, 0.0, CLAMP_MV))
                                if ((u_raw <= CLAMP_MV or err_eff < 0.0) and
                                        (u_raw >= 0.0 or err_eff > 0.0)):
                                    hold_i[ch] = i_try
                                hold_u[ch] = u_sat

                        u_target = _antagonist_project(np.clip(hold_u, 0.0, CLAMP_MV))
                        du_max = PI_DU_MAX_MV_PER_S * run_dt
                        u = np.clip(u_target, last_u_cmd-du_max, last_u_cmd+du_max)
                        u = _antagonist_project(np.clip(u, 0.0, CLAMP_MV))
                        u[~np.isfinite(length_now)] = 0.0
                        pid_mask = allowed & np.isfinite(length_now) & (candidate > 0.0)
                        BR["pi_hold_u"] = u.copy()
                        BR["pi_hold_p"] = hold_p.copy()
                        BR["pi_hold_i"] = hold_i.copy()
                        BR["pi_control_axis"] = control_axis
                        BR["pi_control_channel"] = (planned_channel
                                                     if control_axis is not None else None)
                        BR["pi_x_comp_ch"] = x_comp_ch if control_axis == 2 else None
                        u_ff_used = np.zeros(NCH)
                        teacher = u.copy()
                        phi_train = phi_now
                        if P_meas is not None: x[6:] = P_state
                        xn = x.copy()
                    elif BR.get("ctrl") == "pi":
                        # 顺序 PI：本拍只调整计划轴；已经完成的其他轴保持切换瞬间压力，
                        # 而不是把其电压清零后放气。PI+FEL 前馈也只能修改当前轴。
                        # 只有 axis_plan 明确指定的轴可以产生角度控制。-1（保持）强制为
                        # 无活动轴，禁止再按三轴当前误差动态选中被规划忽略的第三轴。
                        forced_axis = planned_axis if planned_axis >= 0 else -1
                        u_fb = np.asarray(BR["pi"].step(
                            q, r_now, r_vel, forced_axis=forced_axis,
                            forced_channel=(planned_channel if planned_channel >= 0 else None)), float)
                        pid_mask = BR["pi"].active_mask(q, r_now)
                        if planned_channel >= 0:
                            pid_mask &= np.arange(NCH) == planned_channel
                        else:
                            pid_mask[:] = False
                        u_ff_used = np.where(pid_mask, u_ff_now, 0.0)
                        candidate = _antagonist_project(
                            np.clip(u_ff_used + u_fb, CO_CONTRACT_MV, CLAMP_MV))
                        hold_u = np.asarray(BR.get("pi_hold_u", np.zeros(NCH)), float).copy()
                        hold_p = np.asarray(BR.get("pi_hold_p", np.full(NCH, np.nan)), float).copy()
                        hold_i = np.asarray(BR.get("pi_hold_i", np.zeros(NCH)), float).copy()
                        prev_axis = BR.get("pi_control_axis")
                        control_axis = BR["pi"].active_axis
                        # X轴由9/10号驱动时，用对应φ通道抵消其伴随扭转：
                        # V4(肌肉9) + V3(肌肉6/7)，V5(肌肉10) + V2(肌肉2/3)。
                        # 两路目标同时生成，随后仍统一经过gains.json上限、拮抗硬约束和斜率限制。
                        x_comp_ch = (PI_X_PHI_COMP_CHANNEL.get(int(planned_channel))
                                     if control_axis == 2 and planned_channel >= 0 else None)
                        if x_comp_ch is not None:
                            candidate[x_comp_ch] = np.clip(
                                _x_phi_comp_ratio(planned_channel) * candidate[int(planned_channel)],
                                CO_CONTRACT_MV, CLAMP_MV)

                        # 交棒时冻结旧轴的实测压力目标。P/0.09 只是 regulator 的前馈设定，
                        # 后续还会由 ADC 压力内环持续修正；没有压力帧时才退回保持上一拍。
                        if prev_axis is not None and control_axis != prev_axis:
                            old_pair = list(np.flatnonzero(PID_AXIS == int(prev_axis)))
                            old_x_comp = BR.get("pi_x_comp_ch")
                            if int(prev_axis) == 2 and old_x_comp is not None and old_x_comp not in old_pair:
                                old_pair.append(int(old_x_comp))
                            for ch in old_pair:
                                was_active = hold_u[ch] > 0.5 or last_u_cmd[ch] > 0.5
                                if was_active and P_meas is not None:
                                    p_hold = float(P_meas[ch])
                                    if p_hold > PRESSURE_EMPTY_KPA:
                                        hold_p[ch] = p_hold
                                        hold_i[ch] = 0.0
                                        hold_u[ch] = np.clip(p_hold/KPA_PER_MV, 0.0, CLAMP_MV)
                                    else:
                                        hold_p[ch] = np.nan
                                        hold_i[ch] = 0.0
                                        hold_u[ch] = 0.0
                                elif not was_active:
                                    hold_p[ch] = np.nan
                                    hold_i[ch] = 0.0
                                    hold_u[ch] = 0.0

                        # 当前轴误差超过死区时才更新它的一对阀；其余轴保持各自压力目标。
                        if control_axis is not None:
                            pair = np.flatnonzero(PID_AXIS == int(control_axis))
                            hold_p[pair] = np.nan
                            hold_i[pair] = 0.0
                            # 当前规划轴每拍覆盖整对阀。进入死区或误差过零时，两侧都归零；
                            # 不再保留上一拍电压，也绝不在同一子阶段切到对侧阀。
                            hold_u[pair] = 0.0
                            if np.any(pid_mask[pair]):
                                hold_u[pair] = candidate[pair]
                            if int(control_axis) == 2:
                                # X补偿接管φ对；先清掉另一侧，确保2/3与6/7不同时充气。
                                phi_pair = np.flatnonzero(PID_AXIS == 1)
                                hold_p[phi_pair] = np.nan
                                hold_i[phi_pair] = 0.0
                                hold_u[phi_pair] = 0.0
                                if x_comp_ch is not None:
                                    hold_u[x_comp_ch] = candidate[x_comp_ch]

                        # 压力保持内环：切换到其他轴后，以切换瞬间的实测压力为目标，
                        # 在 PI 的 20 Hz 周期内修正 DAC，而不再把一个换算电压开环保持到底。
                        if P_meas is not None:
                            pressure_pid = BR["pressure_pid"]
                            for ch in range(NCH):
                                if not np.isfinite(hold_p[ch]):
                                    continue
                                err_p = float(hold_p[ch] - P_meas[ch])
                                p_dead = float(pressure_pid["deadband"])
                                err_eff = (0.0 if abs(err_p) <= p_dead
                                           else math.copysign(abs(err_p)-p_dead, err_p))
                                i_try = np.clip(hold_i[ch] + float(pressure_pid["ki"])*err_eff*run_dt,
                                                -PRESSURE_HOLD_I_MAX_MV, PRESSURE_HOLD_I_MAX_MV)
                                u_raw = (hold_p[ch]/KPA_PER_MV +
                                         float(pressure_pid["kp"])*err_eff + i_try)
                                u_sat = float(np.clip(u_raw, 0.0, CLAMP_MV))
                                if ((u_raw <= CLAMP_MV or err_eff < 0.0) and
                                        (u_raw >= 0.0 or err_eff > 0.0)):
                                    hold_i[ch] = i_try
                                hold_u[ch] = u_sat
                        u_target = _antagonist_project(np.clip(hold_u, CO_CONTRACT_MV, CLAMP_MV))
                        # PI/保压统一经过输出变化率限制，避免 ADC 跳动或饱和切换造成
                        # 2200→0→2200 的阀指令冲击。急停、停止和轮间放气不走这里，
                        # 仍会立即清零。限速后再次做拮抗投影，维持两侧绝不同时充气。
                        du_max = PI_DU_MAX_MV_PER_S * run_dt
                        u = np.clip(u_target, last_u_cmd-du_max, last_u_cmd+du_max)
                        u = _antagonist_project(np.clip(u, CO_CONTRACT_MV, CLAMP_MV))
                        BR["pi_hold_u"] = u.copy()
                        BR["pi_hold_p"] = hold_p.copy()
                        BR["pi_hold_i"] = hold_i.copy()
                        BR["pi_control_axis"] = control_axis
                        BR["pi_x_comp_ch"] = x_comp_ch if control_axis == 2 else None
                        teacher = np.where(pid_mask, u-u_ff_used, 0.0)
                        phi_train = phi_now
                        m_ = BR["mpc"]                     # PI 分支只使用固定物理一步预测
                        if P_meas is not None: x[6:] = P_state
                        xn = m_.x0 + m_.Ad @ (x - m_.x0) + m_.Bd @ (u - m_.u0)
                        if P_meas is None: x[6:] = xn[6:]
                    else:
                        L_ = len(ref)
                        NH_ = BR.get("N", args.N)
                        hor = [ref[min(k+j, L_-1)] for j in range(1, NH_+1)]
                        u_ff_hor = None
                        if fel_on:                         # 未来 N 拍的逆模型前馈序列
                            rows, phi_rows = [], []
                            for j in range(1, NH_+1):
                                rj = ref[min(k+j, L_-1)]; rj1 = ref[min(k+j+1, L_-1)]
                                phj = _fel_phi(rj, (rj1-rj)/run_dt, q, qd_f, P_state,
                                               last_u_cmd, lock_vec)
                                phi_rows.append(phj)
                                rows.append(_fel_predict(BR["nn"], phj))
                            u_ff_hor = np.array(rows)
                        elif BR.get("ilc_mpc") and BR.get("ilc") is not None:
                            # 任务索引前馈直接来自上一轮真机反馈，不经过顺模型反传。
                            # 第0代严格全零，下面会转成None，保证第一轮与纯LTV MPC同路。
                            u_ff_hor = BR["ilc"].horizon(k, NH_)
                        # A freshly reset inverse network is exactly zero.  In
                        # that case use the identical MPC path as the no-inverse
                        # baseline: DAMPC distinguishes ``None`` (hold the last
                        # nominal input) from an explicit all-zero trajectory.
                        # Passing zero used to make the two generation-0 first
                        # laps differ even though the inverse added no voltage.
                        solver_u_ff_hor = u_ff_hor
                        if (u_ff_hor is not None and
                                not np.any(np.abs(u_ff_hor) > 1.0e-9)):
                            solver_u_ff_hor = None
                        # ── 数据锚定 MPC（2026-09-04）：状态 [q, Δq, P]，P 用实测 ──
                        xd = BR["xd"]; xd[:3] = q; xd[3:6] = qd*run_dt
                        if P_meas is not None:
                            xd[6:] = P_state
                        forward_state_now = xd.copy()
                        if BR.get("shadow_forward"):
                            learner = BR.get("forward_learner")
                            if learner is not None and P_meas is not None:
                                BR["forward_last"] = learner.observe_next(
                                    forward_state_now,
                                    BR.get("n", k)*run_dt,
                                    BR.get("lap", 0))
                            elif learner is not None:
                                learner.discard_pending()
                        # DataModel 学习只属于 MPC+FEL；纯 MPC 不留样本、不改变模型文件。
                        pend = BR.get("dm_pend") if fel_on else None
                        if pend is not None and P_meas is not None:
                            f_prev, q_prev_, u_prev_, P_prev_ = pend
                            if np.all(np.abs(q - q_prev_) < 0.35):
                                BR["dmX"].append(f_prev); BR["dmY"].append(q - q_prev_)
                                BR["dmU"].append(u_prev_); BR["dmPa"].append(P_prev_); BR["dmPb"].append(P_meas.copy())
                                BR["dmBlk"][-1][0].append(u_prev_); BR["dmBlk"][-1][1].append(P_prev_)
                                if len(BR["dmX"]) > 6000:
                                    for kk in ("dmX", "dmY", "dmU", "dmPa", "dmPb"): BR[kk] = BR[kk][-4000:]
                        elif fel_on and BR["dmBlk"][-1][0]:  # 断流（缺实测压/刚放完气）→ 起新连续段
                            BR["dmBlk"].append(([], []))
                        # 把固件 o/w/L 状态交给求解器：未锁通道先规划预锁拍，
                        # 只有进入 L 后才把电压作为 uact 送进真正模型滚动。
                        solver_locks = lock_st if lock_valid else ["o"]*NCH
                        u_arr, xn = BR["dampc"].step(
                            xd.copy(), hor, solver_u_ff_hor,
                            lock_states=solver_locks)
                        u = np.asarray(u_arr, float)
                        lock_cmd = np.asarray(BR["dampc"].lock_cmd, int).copy()
                        base_mpc_u = u.copy()
                        if BR.get("codex_mpc") and BR.get("codex_comp") is not None:
                            task_ref = BR["codex_task_ref"]
                            task_r = task_ref[ki].copy()
                            task_r_next = (task_r if hold_ref else
                                           task_ref[min(ki+1, len(task_ref)-1)].copy())
                            task_ref_delta = task_r_next-task_r
                            task_ref_delta[2] = (task_ref_delta[2]+180.0) % 360.0-180.0
                            task_rv = task_ref_delta/run_dt
                            task_q = np.asarray(st3_from_q_model(q), dtype=float)
                            task_q_next = np.asarray(
                                st3_from_q_model(q+qd_f*run_dt), dtype=float)
                            task_q_delta = task_q_next-task_q
                            task_q_delta[2] = (task_q_delta[2]+180.0) % 360.0-180.0
                            task_qv = task_q_delta/run_dt
                            # Default 1 hardware exception: the replacement V3
                            # valve fills slowly.  Precharge it to 300 mV near
                            # the positive-phi top hold, then raise the target
                            # to 1000 mV through the return.  Keeping this here
                            # permits explicit V2/V3 co-inflation.
                            phi_v3_precharge = bool(
                                BR.get("default1_phi_v3_assist", False) and
                                float(task_r[2]) >= 20.0 and
                                abs(float(task_rv[2])) <= 0.5)
                            phi_v3_assist_tracking = bool(
                                BR.get("default1_phi_v3_assist", False) and
                                float(task_rv[2]) < -0.1 and float(task_r[2]) > 1.0)
                            phi_v3_wait = False
                            timing = BR.get("load2_v3_timing")
                            if timing is not None and BR.get("default1_phi_v3_assist", False):
                                from comparison_replay import load2_v3_phase
                                phase = load2_v3_phase(
                                    timing, BR.get("n", k)*run_dt, phi_v3_precharge)
                                if phase != "normal":
                                    phi_v3_wait = phase == "wait"
                                    phi_v3_precharge = phase == "precharge"
                                    phi_v3_assist_tracking = False
                            phi_v3_assist_override = bool(
                                phi_v3_wait or phi_v3_precharge or phi_v3_assist_tracking or
                                BR.get("default1_phi_v3_releasing", False))
                            u, codex_applied = BR["codex_comp"].apply(
                                task_r, task_q, task_rv, task_qv, base_mpc_u,
                                P_state, lock_st if lock_valid else [],
                                BR.get("n", k)*run_dt,
                                (lock_cmd if lock_valid and
                                 not phi_v3_assist_override else None),
                                record_only=bool(BR.get("plain_mpc_baseline")))
                            if lock_valid and not phi_v3_assist_override and not BR.get("plain_mpc_baseline"):
                                lock_cmd = BR["codex_comp"].last_guarded_lock_cmd.copy()
                            if phi_v3_assist_override and lock_valid:
                                target_v3 = (1000.0 if phi_v3_assist_tracking else
                                             (500.0 if BR.get("load2_v3_precharge", False) else 300.0)
                                             if phi_v3_precharge else 0.0)
                                if phi_v3_assist_tracking or (
                                        BR.get("load2_v3_precharge", False) and phi_v3_precharge):
                                    # Anticipate remaining travel and brake when
                                    # measured descent outruns the reference.
                                    remaining_phi = float(task_q[2]) + 0.4*min(float(task_qv[2]), 0.0)
                                    stop_scale = float(np.clip((remaining_phi-1.0)/8.0, 0.0, 1.0))
                                    speed_scale = float(np.clip(1.0-max(
                                        float(task_rv[2])-float(task_qv[2])-3.0, 0.0)/12.0, 0.0, 1.0))
                                    target_v3 *= min(stop_scale, speed_scale)
                                previous_v3 = float(last_u_cmd[3])
                                rise_slew = (BR["codex_comp"].final_phi_rise_slew_mv_s
                                             or 600.0)
                                if BR.get("load2_v3_precharge", False):
                                    rise_slew = 1200.0  # V3 only: at most +60 mV per 50 ms.
                                fall_slew = (BR["codex_comp"].final_fall_slew_mv_s
                                             or 1200.0)
                                lower = max(0.0, previous_v3-fall_slew*run_dt)
                                upper = min(float(CLAMP_MV),
                                            previous_v3+rise_slew*run_dt)
                                # Explicit Load 2 experiment: step to the hold
                                # target after the delay; preserve falling slew.
                                if BR.get("load2_v3_precharge", False) and phi_v3_precharge:
                                    upper = float(CLAMP_MV)
                                phi_v3_assist_mv = float(np.clip(target_v3, lower, upper))
                                if lock_st[3] != "L":
                                    phi_v3_assist_mv = 0.0
                                u = np.asarray(u, dtype=float).copy()
                                u[3] = phi_v3_assist_mv
                                # V2 remains exactly as the normal MPC+LLM
                                # path requested.  V3 owns a separate lock
                                # request for this explicit co-contraction test.
                                lock_cmd = np.asarray(lock_cmd, dtype=int).copy()
                                lock_cmd[3] = 1 if (
                                    phi_v3_precharge or phi_v3_assist_tracking or
                                    phi_v3_assist_mv > 0.5) else 0
                                BR["default1_phi_v3_releasing"] = bool(
                                    phi_v3_precharge or phi_v3_assist_tracking or
                                    phi_v3_assist_mv > 0.5)
                                codex_applied = u-base_mpc_u
                                # The per-round report must contain the voltage
                                # actually sent to the robot, not the value
                                # before this named experimental override.
                                if BR["codex_comp"].records:
                                    record = BR["codex_comp"].records[-1]
                                    record["total_u_mv"] = u.tolist()
                                    record["compensation_mv"] = codex_applied.tolist()
                                    record["default1_phi_v3_precharge"] = bool(
                                        phi_v3_precharge)
                                    record["default1_phi_v3_assist_mv"] = phi_v3_assist_mv
                            y_return_state = BR.get("load2_y_return_state")
                            if y_return_state is not None and lock_valid:
                                from comparison_replay import load2_y_return_assist
                                u, lock_cmd = load2_y_return_assist(
                                    y_return_state, BR.get("n", k)*run_dt,
                                    float(task_r[1]), float(task_q[1]), float(task_rv[1]),
                                    u, lock_cmd, lock_st, P_state, last_u_cmd, run_dt)
                                u = np.clip(np.asarray(u, dtype=float), 0.0, CLAMP_MV)
                                lock_cmd = np.asarray(lock_cmd, dtype=int)
                                codex_applied = u-base_mpc_u
                                BR["codex_comp"].previous_total = u.copy()
                                if BR["codex_comp"].records:
                                    record = BR["codex_comp"].records[-1]
                                    record["total_u_mv"] = u.tolist()
                                    record["compensation_mv"] = codex_applied.tolist()
                                    record["load2_y_return_assist"] = dict(y_return_state)
                            # 下一拍MPC的Δu代价必须从真机实际收到的总指令开始，
                            # 不能误以为只执行了补偿前的base MPC电压。
                            BR["dampc"].u_prev[:] = u
                        if BR.get("replay_forbid_v3", False):
                            u = np.asarray(u, dtype=float).copy()
                            u[3] = 0.0
                            lock_cmd[3] = 0
                            phi_v3_assist_override = False
                            BR["dampc"].u_prev[:] = u
                            codex_applied = u-base_mpc_u
                            if BR.get("codex_comp") is not None and BR["codex_comp"].records:
                                record = BR["codex_comp"].records[-1]
                                record["total_u_mv"] = u.tolist()
                                record["compensation_mv"] = codex_applied.tolist()
                                record["comparison_forbid_v3"] = True
                        if P_meas is None: xd[6:] = xn[6:]
                        x[6:] = xd[6:]
                        u_ff_used = (u_ff_hor[0].copy() if u_ff_hor is not None else
                                     np.zeros(NCH))
                        teacher = (base_mpc_u if BR.get("codex_mpc") else
                                   u - u_ff_used)
                        phi_train = phi_rows[0] if fel_on else phi_now
                    BR["codex_u_comp"] = codex_applied.copy()
                    # 门卫：QP 异常拍、锁状态缺失以及 w 预锁等待拍均不进入逆模型。
                    tick_ok = getattr(BR.get("dampc"), "ok", True) if BR.get("ctrl") == "mpc" else True
                    if BR.get("ilc_mpc"):
                        ilc_valid = (np.asarray(lock_st) == "L" if lock_valid else
                                     np.zeros(NCH, dtype=bool))
                        if not tick_ok:
                            ilc_valid[:] = False
                        BR["ilc_u"].append(np.asarray(u, float).copy())
                        BR["ilc_ff"].append(np.asarray(u_ff_used, float).copy())
                        BR["ilc_mask"].append(ilc_valid.copy())
                    if fel_on and tick_ok and lock_valid:
                        learn_mask = np.asarray(lock_st) == "L"  # 只训练确实可施加电压的通道
                        if BR.get("ctrl") == "pi":
                            learn_mask &= pid_mask               # 保压轴冻结，不拿零教师覆盖其网络
                        if np.any(learn_mask):
                            BR["felPhi"].append(phi_train.copy())
                            BR["felErr"].append(np.asarray(teacher, float).copy())
                            BR["felMask"].append(learn_mask.copy())
                    # DataModel 也不能把预锁期间“命令有值、实际阀为零”误当成对象响应。
                    actuation_ok = (lock_valid and not any(s == "w" for s in lock_st)
                                    and all(float(u[c]) <= 0.5 or lock_st[c] == "L" for c in range(NCH)))
                    if BR.get("ctrl") == "mpc":
                        BR["dm_pend"] = (DataModel.feats(q, qd*run_dt, BR["xd"][6:]), q.copy(),
                                         u.copy(), BR["xd"][6:].copy()) \
                            if fel_on and tick_ok and P_meas is not None and actuation_ok else None
                    # MPC 的 u 已经在求解器出口按 L 状态门控；此处再用实测锁状态
                    # 复核得到 u_actual。正常显式锁协议下两者逐拍相等。
                    u_actual = (np.where(np.asarray(lock_st) == "L", u, 0.0)
                                if lock_valid else np.full(NCH, np.nan))
                    # The ten-tick models keep only real measured history here.
                    # Candidate controls used inside MPC are shifted separately
                    # by DAMPC during its nominal rollout.
                    history_dm = getattr(BR.get("dampc"), "dm", None)
                    if hasattr(history_dm, "observe_actual"):
                        if lock_valid and np.all(np.isfinite(u_actual)):
                            history_dm.observe_actual(forward_state_now, u_actual)
                        else:
                            history_dm.reset_history()
                    if BR.get("shadow_forward") and BR.get("forward_learner") is not None:
                        if (BR.get("ctrl") == "mpc" and P_meas is not None and
                                lock_valid and np.all(np.isfinite(u_actual))):
                            try:
                                BR["forward_learner"].begin_transition(
                                    forward_state_now, u_actual,
                                    BR.get("n", k)*run_dt,
                                    BR.get("lap", 0), reference=r_plot)
                            except Exception as exc:
                                BR["forward_learner"].discard_pending()
                                print(f"# ⚠ 顺模型 x_hat_next 计算失败，本拍不学习：{exc}")
                        else:
                            BR["forward_learner"].discard_pending()
                    BR["last_u_cmd"] = u.copy()
                    if tick_ok:
                        BR["tss"] = BR.get("tss", 0.0) + float(np.sum(teacher**2))
                        BR["tn"] = BR.get("tn", 0) + NCH
                    length_valid = np.isfinite(length_error)
                    if BR.get("ctrl") == "musclepi" and np.any(length_valid):
                        BR["length_ess"] = BR.get("length_ess", 0.0) + float(
                            np.sum(length_error[length_valid]**2))
                        BR["length_en"] = BR.get("length_en", 0) + int(np.sum(length_valid))
                    BR["elog"][ki] = r_now - q
                    BR["raw_elog"][ki] = r_plot - q
                    tg = BR.get("n", k) * run_dt           # 全局时间：循环回绕也单调递增
                    BR["n"] = BR.get("n", k) + 1
                    forward_payload = {}
                    if BR.get("shadow_forward"):
                        learner = BR.get("forward_learner")
                        obs = BR.get("forward_last") or {}
                        original_pair = (BR.get("new_mpc_fel") and
                                         BR.get("model_source") == "original")
                        forward_arch_label = str(BR.get("forward_arch", "mlp"))
                        forward_payload = {
                            "forward_mode": (("NEW MPC+FEL｜原始初始化分支"
                                              if original_pair else
                                              "NEW MPC+FEL｜最新已验证"
                                              if BR.get("new_mpc_fel") else
                                              "MPC+影子正模型（无逆模型）") +
                                             f"｜{forward_arch_label}"),
                            "forward_status": obs.get("status", "等待下一拍"),
                            "forward_generation": (learner.generation if learner is not None else 0),
                            "forward_error_norm_deg": obs.get("active_error_norm_deg"),
                            "shadow_error_norm_deg": obs.get("shadow_error_norm_deg"),
                            "forward_error_deg": obs.get("active_error_deg"),
                            "shadow_error_deg": obs.get("shadow_error_deg"),
                        }
                    v_ref_axis = (np.clip(r_vel + PI_VPOS*(r_now-q), -BR["pi"].vmax, BR["pi"].vmax)
                                  if BR.get("ctrl") == "pi" else r_vel)
                    BR["log"].append([round(tg, 2)] + [round(math.degrees(v), 3) for v in q]
                                     + [round(math.degrees(v), 3) for v in r_now]
                                     + [round(math.degrees(v), 3) for v in r_plot]
                                     + [round(math.degrees(v), 3) for v in qd_f]
                                     + [round(math.degrees(v), 3) for v in v_ref_axis]
                                     + [int(v) for v in u]
                                     + ([int(round(v)) for v in u_actual] if lock_valid else [""]*NCH)
                                     + ([int(v) for v in lock_cmd] if BR.get("ctrl") == "mpc" else [""]*NCH)
                                     + [int(round(v)) for v in u_ff_used]
                                     + [int(round(v)) for v in teacher]
                                     + [int(round(v)) for v in codex_applied]
                                     + ([int(round(float(v))) for v in P_meas] if P_meas is not None else [""]*NCH)
                                     + (lock_st if lock_valid else [""]*NCH)
                                     + ([round(float(v), 3) if np.isfinite(v) else "" for v in length_now])
                                     + ([round(float(v), 3) if np.isfinite(v) else "" for v in length_target])
                                     + [plan_code, planned_channel, int(hold_ref)]
                                     + ([getattr(BR.get("dampc"), "solver", ""),
                                         round(float(getattr(BR.get("dampc"), "last_solve_ms", 0.0)), 3),
                                         int(getattr(BR.get("dampc"), "last_nl_iters", 0))]
                                        if BR.get("ctrl") == "mpc" else ["", "", ""]))
                    TELEM.append({"t": BR.get("telem_time_offset", 0.0) + tg,
                                  "wall": time.time(), "source": "control",
                                  "q": [math.degrees(v) for v in q],
                                  "r": [math.degrees(v) for v in r_plot],
                                  "r_control": [math.degrees(v) for v in r_now],
                                  "q_gui": list(st3_from_q_model(q)),
                                  "r_gui": list(st3_from_q_model(r_plot)),
                                  "r_control_gui": list(st3_from_q_model(r_now)),
                                  "v": [math.degrees(v) for v in qd_f],
                                  "vr": [math.degrees(v) for v in v_ref_axis],
                                   "u": [int(v) for v in u],
                                  "u_comp": [int(round(v)) for v in codex_applied],
                                  "pressure_kpa": ([float(v) for v in P_meas]
                                                   if P_meas is not None else [None]*NCH),
                                  "channel_length_mm": [round(float(v), 3) if np.isfinite(v) else None
                                                        for v in length_now],
                                  "l15_displacement_raw_mm": [
                                      round(float(v), 3) if np.isfinite(v) else None
                                      for v in group_displacement_raw[0]],
                                  "group_displacement_raw_mm": [
                                      [round(float(v), 3) if np.isfinite(v) else None
                                       for v in pair]
                                      for pair in group_displacement_raw],
                                  "force_gui_yzx_n": data.get("force_gui_yzx_n"),
                                  "torque_xyz_nm": data.get("torque_xyz_nm"),
                                  "force_pred_xyz_n": data.get("force_pred_xyz_n"),
                                  "lock_cmd": ([int(v) for v in lock_cmd]
                                               if BR.get("ctrl") == "mpc" else None),
                                  "default1_phi_v3_assist_mv": float(phi_v3_assist_mv),
                                  "e": math.degrees(np.linalg.norm(r_now - q)),
                                  "length": [round(float(v), 3) if np.isfinite(v) else None for v in length_now],
                                  "length_ref": [round(float(v), 3) if np.isfinite(v) else None for v in length_target],
                                  **stage_payload, **forward_payload})
                    if not hold_ref:
                        BR["k"] += 1
                    done = BR["k"] >= len(ref)
                    if done and BR.get("loop"):
                        completed_laps = int(BR.get("lap", 0)) + 1
                        run_limit = BR.get("loop_limit_runs")
                        deadline = BR.get("loop_deadline_monotonic")
                        if ((run_limit is not None and completed_laps >= int(run_limit)) or
                                (deadline is not None and time.monotonic() >= float(deadline))):
                            # 限制只在完整轨迹轮次结束后生效，不在示教段中间硬切断。
                            BR["loop"] = False
                    if done and BR.get("loop"):
                        BR["k"] = 0; done = False
                        BR["pi_stage_wait_s"] = 0.0
                        BR["pi_stage_wait_point"] = None
                        BR["venting"] = int(2.0/run_dt)        # 轮间强制放气 2s：u=0 开环，不闭环追零
                        BR["final_vent"] = False
                        if len(BR["log"]) > 36000: BR["log"] = BR["log"][-18000:]
                        BR["lap"] += 1
                        rms = math.degrees(math.sqrt(float(np.mean(np.sum(BR["elog"]**2, axis=1)))))
                        raw_mse_deg2 = float(np.mean(np.sum(
                            np.degrees(BR["raw_elog"])**2, axis=1)))
                        ilc_result = (_ilc_round_update(raw_mse_deg2)
                                      if BR.get("ilc_mpc") else None)
                        rmse = math.sqrt(BR.get("tss", 0.0)/max(1, BR.get("tn", 1)))
                        BR["tss"] = 0.0; BR["tn"] = 0
                        length_rmse = math.sqrt(BR.get("length_ess", 0.0) /
                                                max(1, BR.get("length_en", 1)))
                        BR["length_ess"] = 0.0; BR["length_en"] = 0
                        forward_result = (_new_forward_round_update(raw_mse_deg2)
                                          if BR.get("shadow_forward") else None)
                        nu = _fel_round_update(raw_mse_deg2) if fel_on else 0
                        nd = (_dm_round_update()
                              if fel_on and not BR.get("new_mpc_fel") else 0)
                        if fel_on: _save_w()                   # 保存本圈刚更新后的 NN
                        wn = _fel_weight_mean(BR.get("nn"))
                        BR.setdefault("lapstats", []).append(
                            [BR["lap"], round(rms, 3), round(rmse, 1), round(wn, 2)])
                        if BR.get("ctrl") == "musclepi":
                            print(f"# 第 {BR['lap']} 轮完  肌肉长度RMSE {length_rmse:.2f}mm  六路长度PI")
                        elif BR.get("ilc_mpc"):
                            action = ("回滚候选" if ilc_result and ilc_result.get("rolled_back") else
                                      "下一轮测试新候选" if ilc_result and ilc_result.get("proposed") else
                                      "保持当前前馈")
                            generation = (BR["ilc"].generation if BR.get("ilc") is not None else 0)
                            print(f"# 第 {BR['lap']} 轮完  跟踪RMS {rms:.2f}°  "
                                  f"ILC反馈RMSE {rmse:.0f}mV；{action}，generation={generation}")
                        elif BR.get("shadow_forward"):
                            original_pair = (BR.get("new_mpc_fel") and
                                             BR.get("model_source") == "original")
                            promoted_text = ("；影子顺模型已替换活动模型"
                                             if forward_result and forward_result.get("promoted") else
                                             "；活动顺模型保持不变")
                            if BR.get("new_mpc_fel"):
                                inverse_result = BR.get("inverse_last_result") or {}
                                inverse_text = (
                                    f"逆模型NN用{inverse_result.get('replay_rounds', 0)}轮/"
                                    f"{nu}样本训练{inverse_result.get('epochs', 0)}epochs"
                                    if inverse_result.get("trained") else
                                    f"逆模型本轮未训练：{inverse_result.get('reason', '样本不足')}")
                                if original_pair:
                                    print(f"# 第 {BR['lap']} 轮完  跟踪RMS {rms:.2f}°  反馈RMSE {rmse:.0f}mV"
                                          f"  ← 原始初始化分支{inverse_text}{promoted_text}")
                                else:
                                    print(f"# 第 {BR['lap']} 轮完  跟踪RMS {rms:.2f}°  反馈RMSE {rmse:.0f}mV"
                                          f"  ← {inverse_text}{promoted_text}")
                            else:
                                print(f"# 第 {BR['lap']} 轮完  跟踪RMS {rms:.2f}°  "
                                      f"纯MPC反馈、无逆模型{promoted_text}")
                        elif fel_on:
                            print(f"# 第 {BR['lap']} 轮完  跟踪RMS {rms:.2f}°  反馈RMSE {rmse:.0f}mV"
                                  f"  ← DataModel {nd} / NN {nu} 样本训练毕，第 {BR['lap']+1} 轮换新模型")
                        else:
                            print(f"# 第 {BR['lap']} 轮完  跟踪RMS {rms:.2f}°  纯控制模式：模型冻结")
                    elif done:
                        # 单程末点完成后先下发全零放气2s，再真正结束。
                        if BR.get("ctrl") == "musclepi":
                            length_rmse = math.sqrt(BR.get("length_ess", 0.0) /
                                                    max(1, BR.get("length_en", 1)))
                            print(f"# 单程控制结束  肌肉长度RMSE {length_rmse:.2f}mm，开始全机放气")
                        elif len(BR.get("elog", [])):
                            rms = math.degrees(math.sqrt(float(np.mean(
                                np.sum(BR["elog"]**2, axis=1)))))
                            if len(BR.get("raw_elog", [])):
                                BR["completed_track_mse_deg2"] = float(np.mean(np.sum(
                                    np.degrees(BR["raw_elog"])**2, axis=1)))
                            print(f"# 单程控制结束  整体跟踪RMS {rms:.2f}°，开始全机放气")
                        done = False
                        BR["pi_stage_wait_s"] = 0.0
                        BR["pi_stage_wait_point"] = None
                        BR["venting"] = int(2.0/run_dt)
                        BR["final_vent"] = True
                        BR["new_round_complete"] = True
                    # 用真实控制拍计数限制为 0.1s 打印一次。等待到位时 k 会冻结，
                    # 若用 k 取模，会在同一个 t 上每个 20Hz 控制拍都重复打印。
                    actual_tick = max(0, int(BR.get("n", 1))-1)
                    if actual_tick % max(1, int(round(0.1/run_dt))) == 0:
                        p_text = ([int(round(float(v))) for v in P_meas]
                                  if P_meas is not None else ["--"]*NCH)
                        if BR.get("ctrl") == "musclepi":
                            le = length_error[np.isfinite(length_error)]
                            le_norm = math.sqrt(float(np.mean(le**2))) if le.size else float("nan")
                            l_text = [round(float(v), 1) if np.isfinite(v) else "--" for v in length_now]
                            lr_text = [round(float(v), 1) if np.isfinite(v) else "--" for v in length_target]
                            print(f"  t={k*run_dt:5.1f}s |eL|={le_norm:5.2f}mm "
                                  f"u={[int(v) for v in u]} L={l_text} Lr={lr_text} P={p_text} kPa")
                        else:
                            fm_text = ""
                            if BR.get("shadow_forward"):
                                fm_value = forward_payload.get("forward_error_norm_deg")
                                fm_text = (f" FM={float(fm_value):.2f}°/"
                                           f"{forward_payload.get('forward_status')}"
                                           if fm_value is not None else " FM=等待")
                            lock_text = (f" K={[int(v) for v in lock_cmd]}"
                                         if BR.get("ctrl") == "mpc" else "")
                            print(f"  t={k*run_dt:5.1f}s |e|={math.degrees(np.linalg.norm(r_now-q)):5.2f}° "
                                  f"u={[int(v) for v in u]}{lock_text} P={p_text} kPa{fm_text}")
                    out = {"u": [int(v) for v in u], "mode": "mpc", "t": round(k*run_dt, 2),
                           "ref": list(st3_from_q_model(r_plot)),
                           "control_ref": list(st3_from_q_model(r_now)),
                           "meas": list(st3_from_q_model(q)),
                           "length": [round(float(v), 3) if np.isfinite(v) else None for v in length_now],
                           "length_ref": [round(float(v), 3) if np.isfinite(v) else None for v in length_target],
                           **stage_payload, **forward_payload}
                    if BR.get("ctrl") == "mpc":
                        out["lock"] = [int(v) for v in lock_cmd]
                        out["phi_pair_assist"] = bool(phi_v3_assist_override)
                    if BR.get("venting", 0) > 0:
                        BR["last_u_cmd"][:] = 0
                        BR["pi_hold_u"][:] = 0
                        BR["pi_hold_p"][:] = np.nan
                        BR["pi_hold_i"][:] = 0
                        BR["pi_control_axis"] = None
                        BR["pi_control_channel"] = None
                        BR["pi_x_comp_ch"] = None
                        vent_round = max(1, int(BR.get("lap", 0)))
                        finish_after_vent = bool(BR.get("final_vent", False))
                        hidden_return = bool(BR.get("auto_hidden_return_to_zero", True))
                        vent_plot_model = (np.zeros(3) if hidden_return else
                                           np.asarray(BR["plot_ref"][-1], dtype=float))
                        vent_control_model = (np.zeros(3) if hidden_return else
                                              np.asarray(BR["ref"][-1], dtype=float))
                        vent_ref_gui = list(st3_from_q_model(vent_plot_model))
                        vent_control_gui = list(st3_from_q_model(vent_control_model))
                        endpoint_text = ("隐藏终点·(0,0,0)" if hidden_return else "末点")
                        vent_text = (f"{endpoint_text}·全机放气" if finish_after_vent else
                                     f"第{vent_round}轮·{endpoint_text}·全机放气")
                        out.update({"u": [0]*NCH, "ref": vent_ref_gui,
                                    "control_ref": vent_control_gui,
                                    **_stage_update(("vent", vent_round, finish_after_vent),
                                                    vent_text, -1)})
                        if BR.get("ctrl") == "mpc":
                            out["lock"] = [0]*NCH
                    if done:
                        done_stage = _stage_update(("done", BR.get("lap", 0)), "轨迹完成", -1)
                        _finish("轨迹完成"); out["mode"] = "done"; out["u"] = [0]*NCH
                        if BR.get("ctrl") == "mpc":
                            out["lock"] = [0]*NCH
                        out.update(done_stage)
                    self._out(out); return
            self._out({"err": "unknown path"})
    SRV["mount"] = args.mount                      # st_from_q_model 用
    print("# bridge 模式：纯计算，不碰串口。GUI 保持「连接 IMU」，选「MPC·控制站」即可。")
    print(f"# http://127.0.0.1:{args.http}  （Ctrl+C 退出）")
    ThreadingHTTPServer(("127.0.0.1", args.http), H).serve_forever()

# ───────────────────────── main ─────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PAM 肩关节 MPC 控制站（作者模型灰盒）")
    ap.add_argument("mode", choices=["ident", "fit", "mpc", "serve", "bridge"])
    ap.add_argument("--port", default="")
    ap.add_argument("--levels", default="800,1600")
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--traj", default="")
    ap.add_argument("--targets", default="")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--mount", default="ry-", choices=list(MOUNTS))
    ap.add_argument("--http", type=int, default=8765, help="serve 模式的 GUI 遥控端口")
    ap.add_argument("--noplot", action="store_true")
    args = ap.parse_args()
    {"ident": mode_ident, "fit": mode_fit, "mpc": mode_mpc,
     "serve": mode_serve, "bridge": mode_bridge}[args.mode](args)

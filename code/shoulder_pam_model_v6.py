# -*- coding: utf-8 -*-
"""TRO shoulder model V6.

V6 combines the corrected V5.1 ten-PAM geometry/rigid-body dynamics with the
self-locking muscle-system constitutive law documented in ``肌肉.docx``.

This module is deliberately offline-only: the live controller still imports
``shoulder_pam_model.py``.  Switching the hardware controller to V6 therefore
requires an explicit later decision after parameter identification.

Conventions
-----------
* Geometry rows follow Table 1 of the V5.1 report.
* The hardware valve order remains exactly
  ``[V0=15, V1=48, V2=23, V3=67, V4=9, V5=10]``.
* ``lock=0`` means released; ``lock=1`` means locked.
* 30 kPa is used only by :func:`unlock_permitted`; it never gates pressure,
  muscle force, telemetry, or a model state.
* Lengths in the muscle law use mm, rates use mm/s, and forces use N.
  Geometry is converted to metres only where a moment arm is required.
* The PSS-PAM three-element coefficients are the experimentally identified
  10 cm actuator values supplied with the project.  Their published SI
  ``N/m`` and ``N*s/m`` coefficients are divided by 1000 for this module's
  millimetre state convention.  They were identified only above 25 kPa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np


MODEL_VERSION = "V6"
N_PAM = 10
N_VALVE = 6
MM_TO_M = 1.0e-3
GRAVITY_M_S2 = 9.81
LOCK_LEAD_MS = 100.0
UNLOCK_PRESSURE_KPA = 30.0
MUSCLE_IDENTIFIED_PRESSURE_MIN_KPA = 25.0
ANTAGONIST_PAIRS = ((0, 1), (2, 3), (4, 5))

# V5.1 Table 1, fixed-frame anchors relative to O_B and moving anchors
# relative to the geometric ball centre O, before the +66 mm pivot correction.
A_B_MM = np.array([
    [187.00, 56.00, 23.00],
    [187.00, 23.00, 56.00],
    [187.00, -23.00, 56.00],
    [187.00, -56.00, 23.00],
    [187.00, 56.00, -23.00],
    [187.00, 23.00, -56.00],
    [187.00, -23.00, -56.00],
    [187.00, -56.00, -23.00],
    [30.54, -111.50, 63.18],
    [30.54, 111.50, 63.18],
], dtype=float)

B_BALL_MM = np.array([
    [20.00, 20.00, 8.00],
    [20.00, 8.00, 20.00],
    [20.00, -8.00, 20.00],
    [20.00, -20.00, 8.00],
    [20.00, 20.00, -8.00],
    [20.00, 8.00, -20.00],
    [20.00, -8.00, -20.00],
    [20.00, -20.00, -8.00],
    [26.23, -20.49, -110.55],
    [26.23, 20.49, -110.55],
], dtype=float)

P_BALL_B_MM = np.array([-131.0, 0.0, 0.0])
PIVOT_DX_MM = 66.0
P_PIVOT_B_MM = P_BALL_B_MM + np.array([PIVOT_DX_MM, 0.0, 0.0])
A_PIVOT_MM = A_B_MM - P_PIVOT_B_MM
B_PIVOT_MM = B_BALL_MM - np.array([PIVOT_DX_MM, 0.0, 0.0])
NORM2_MM2 = np.sum(A_PIVOT_MM**2, axis=1) + np.sum(B_PIVOT_MM**2, axis=1)

# Physical encoder/muscle number -> V5.1 Table-1 geometry row.  This is the
# already deployed mapping in control/mpc_station.py and must not be reordered.
PHYSICAL_TO_GEOMETRY_ROW = np.array([2, 3, 7, 6, 1, 0, 4, 5, 8, 9], dtype=int)
VALVE_PHYSICAL_MUSCLES = ((1, 5), (4, 8), (2, 3), (6, 7), (9,), (10,))
VALVE_GEOMETRY_ROWS = tuple(
    tuple(int(PHYSICAL_TO_GEOMETRY_ROW[muscle - 1]) for muscle in muscles)
    for muscles in VALVE_PHYSICAL_MUSCLES
)
GEOMETRY_ROW_TO_VALVE = np.empty(N_PAM, dtype=int)
for _valve, _rows in enumerate(VALVE_GEOMETRY_ROWS):
    for _row in _rows:
        GEOMETRY_ROW_TO_VALVE[_row] = _valve

# Mapping F = S u_v in the deployed hardware valve order.
SIX_VALVE_MAP = np.zeros((N_PAM, N_VALVE), dtype=float)
SIX_VALVE_MAP[np.arange(N_PAM), GEOMETRY_ROW_TO_VALVE] = 1.0


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix(q: Iterable[float]) -> np.ndarray:
    """V5.1 Eq. (6): R = Rz(q2) Ry(-q1) Rz(-q3)."""
    q1, q2, q3 = np.asarray(q, dtype=float).reshape(3)
    return rot_z(q2) @ rot_y(-q1) @ rot_z(-q3)


def angular_velocity_matrix(q: Iterable[float]) -> np.ndarray:
    """V5.1 Eq. (28), fixed-frame angular velocity omega = E(q) qdot."""
    q1, q2, _ = np.asarray(q, dtype=float).reshape(3)
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    return np.array([
        [s2, 0.0, s1*c2],
        [-c2, 0.0, s1*s2],
        [0.0, 1.0, -c1],
    ])


def angular_velocity_matrix_dot(q: Iterable[float], qd: Iterable[float]) -> np.ndarray:
    q1, q2, _ = np.asarray(q, dtype=float).reshape(3)
    qd1, qd2, _ = np.asarray(qd, dtype=float).reshape(3)
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    return np.array([
        [c2*qd2, 0.0, c1*c2*qd1 - s1*s2*qd2],
        [s2*qd2, 0.0, c1*s2*qd1 + s1*c2*qd2],
        [0.0, 0.0, s1*qd1],
    ])


def q_from_rotation_matrix(rotation: np.ndarray, eps: float = 1.0e-10) -> np.ndarray:
    """Invert the V5.1 parameterisation using the q1 in [0, pi] branch."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    q1 = float(np.arccos(np.clip(r[2, 2], -1.0, 1.0)))
    s1 = float(np.sin(q1))
    if abs(s1) <= eps:
        # Only q2-q3 is observable at the singularity.  Use q3=0 so the
        # conversion is deterministic; callers must not infer rank from it.
        return np.array([q1, float(np.arctan2(r[1, 0], r[0, 0])), 0.0])
    q2 = float(np.arctan2(-r[1, 2], -r[0, 2]))
    q3 = float(np.arctan2(r[2, 1], r[2, 0]))
    return np.array([q1, q2, q3])


def q_from_gui_degrees(sx_deg: float, sy_deg: float, phi_deg: float = 0.0) -> np.ndarray:
    """Convert the GUI swing-twist coordinates to the V5.1 q coordinates."""
    sx, sy, phi = np.radians([sx_deg, sy_deg, phi_deg])
    theta = float(np.hypot(sx, sy))
    if theta < 1.0e-12:
        swing = np.eye(3)
    else:
        axis = np.array([sx/theta, sy/theta, 0.0])
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        swing = np.eye(3) + np.sin(theta)*skew + (1.0-np.cos(theta))*(skew @ skew)
    return q_from_rotation_matrix(swing @ rot_z(phi))


def moving_points_mm(q: Iterable[float]) -> np.ndarray:
    return P_PIVOT_B_MM + moment_arm_vectors_mm(q)


def moment_arm_vectors_mm(q: Iterable[float]) -> np.ndarray:
    return B_PIVOT_MM @ rotation_matrix(q).T


def pam_vectors_mm(q: Iterable[float]) -> np.ndarray:
    return A_PIVOT_MM - moment_arm_vectors_mm(q)


def pam_lengths_mm(q: Iterable[float]) -> np.ndarray:
    r = rotation_matrix(q)
    inner = np.einsum("ij,jk,ik->i", A_PIVOT_MM, r, B_PIVOT_MM)
    return np.sqrt(np.maximum(0.0, NORM2_MM2 - 2.0*inner))


def pam_unit_directions(q: Iterable[float]) -> np.ndarray:
    vectors = pam_vectors_mm(q)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def effective_moment_arms_m(q: Iterable[float]) -> np.ndarray:
    arms = moment_arm_vectors_mm(q)
    lengths = pam_lengths_mm(q)
    return np.cross(arms, A_PIVOT_MM) / lengths[:, None] * MM_TO_M


def H_matrix(q: Iterable[float]) -> np.ndarray:
    return effective_moment_arms_m(q).T


def length_jacobian(q: Iterable[float]) -> np.ndarray:
    return -H_matrix(q).T @ angular_velocity_matrix(q)


def valve_to_geometry_rows(valve_values: Iterable[float]) -> np.ndarray:
    return SIX_VALVE_MAP @ np.asarray(valve_values, dtype=float).reshape(N_VALVE)


def geometry_rows_to_valve_mean(row_values: Iterable[float]) -> np.ndarray:
    rows = np.asarray(row_values, dtype=float).reshape(N_PAM)
    return np.array([float(np.mean(rows[list(group)])) for group in VALVE_GEOMETRY_ROWS])


def project_antagonist_valves(valve_values: Iterable[float]) -> np.ndarray:
    """Hard constraint: each antagonist pair retains at most one nonzero side."""
    values = np.maximum(0.0, np.asarray(valve_values, dtype=float).reshape(N_VALVE)).copy()
    for left, right in ANTAGONIST_PAIRS:
        if values[left] > 0.0 and values[right] > 0.0:
            if values[left] >= values[right]:
                values[right] = 0.0
            else:
                values[left] = 0.0
    return values


def lock_lead_steps(dt_s: float, lead_ms: float = LOCK_LEAD_MS) -> int:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    return max(1, int(np.ceil(max(0.0, float(lead_ms))/1000.0/dt_s)))


def unlock_permitted(pressure_kpa: float, requested: bool = True,
                     threshold_kpa: float = UNLOCK_PRESSURE_KPA) -> bool:
    """The sole V6 use of the 30 kPa threshold: an unlock safety decision."""
    return bool(requested and np.isfinite(pressure_kpa) and pressure_kpa <= threshold_kpa)


@dataclass
class ArmParameters:
    mass_kg: float = 0.2219
    centre_of_mass_m: np.ndarray = field(default_factory=lambda: np.array(
        [-0.048728, -0.0000187, -0.090313], dtype=float))
    inertia_about_pivot_kg_m2: np.ndarray = field(default_factory=lambda: np.array([
        [0.00275383266, -2.50198298e-7, -0.000970788277],
        [-2.50198298e-7, 0.00326525194, -4.59756503e-7],
        [-0.000970788277, -4.59756503e-7, 0.000564081428],
    ], dtype=float))
    gravity_m_s2: float = GRAVITY_M_S2

    def __post_init__(self) -> None:
        self.centre_of_mass_m = np.asarray(self.centre_of_mass_m, dtype=float).reshape(3)
        self.inertia_about_pivot_kg_m2 = np.asarray(
            self.inertia_about_pivot_kg_m2, dtype=float).reshape(3, 3)
        if not np.allclose(self.inertia_about_pivot_kg_m2,
                           self.inertia_about_pivot_kg_m2.T, atol=1.0e-12):
            raise ValueError("inertia tensor must be symmetric")


def inertia_in_fixed_frame(q: Iterable[float], arm: ArmParameters) -> np.ndarray:
    r = rotation_matrix(q)
    return r @ arm.inertia_about_pivot_kg_m2 @ r.T


def gravity_torque(q: Iterable[float], arm: ArmParameters) -> np.ndarray:
    position = rotation_matrix(q) @ arm.centre_of_mass_m
    return np.cross(position, np.array([0.0, 0.0, -arm.mass_kg*arm.gravity_m_s2]))


def mass_matrix(q: Iterable[float], arm: ArmParameters) -> np.ndarray:
    e = angular_velocity_matrix(q)
    return e.T @ inertia_in_fixed_frame(q, arm) @ e


def gravity_vector(q: Iterable[float], arm: ArmParameters) -> np.ndarray:
    e = angular_velocity_matrix(q)
    return -e.T @ gravity_torque(q, arm)


def coriolis_times_velocity(q: Iterable[float], qd: Iterable[float],
                            arm: ArmParameters) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(3)
    qd = np.asarray(qd, dtype=float).reshape(3)
    e = angular_velocity_matrix(q)
    inertia = inertia_in_fixed_frame(q, arm)
    omega = e @ qd
    return e.T @ (
        inertia @ (angular_velocity_matrix_dot(q, qd) @ qd)
        + np.cross(omega, inertia @ omega)
    )


def generalized_pam_force(q: Iterable[float], muscle_forces_n: Iterable[float]) -> np.ndarray:
    return -length_jacobian(q).T @ np.asarray(muscle_forces_n, dtype=float).reshape(N_PAM)


def forward_dynamics(q: Iterable[float], qd: Iterable[float],
                     muscle_forces_n: Iterable[float], arm: ArmParameters,
                     silicone_generalized: Optional[Iterable[float]] = None,
                     friction_generalized: Optional[Iterable[float]] = None) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(3)
    qd = np.asarray(qd, dtype=float).reshape(3)
    qs = np.zeros(3) if silicone_generalized is None else np.asarray(
        silicone_generalized, dtype=float).reshape(3)
    qf = np.zeros(3) if friction_generalized is None else np.asarray(
        friction_generalized, dtype=float).reshape(3)
    rhs = (generalized_pam_force(q, muscle_forces_n) + qs
           - coriolis_times_velocity(q, qd, arm) - gravity_vector(q, arm) - qf)
    return np.linalg.solve(mass_matrix(q, arm), rhs)


def _ten(value: float | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(N_PAM, float(array))
    if array.size != N_PAM:
        raise ValueError(f"{name} must be scalar or length {N_PAM}")
    return array.reshape(N_PAM).copy()


@dataclass
class MuscleSystemParameters:
    """Parameters of the three-element PAM plus Kss/Ksl self-locking system.

    The defaults below are the identified PSS-PAM relations, converted from
    metres to millimetres::

        K(P)    = 0.33547 + 0.01567 P       N/mm
        Binf(P) = 1.10000 + 0.00405 P       N*s/mm
        Bdef(P) = 0.70000 - 0.00261 P       N*s/mm
        F(P)    = -10.278 + 0.585 P         N

    with pressure ``P`` in kPa.  These relations are calibrated only for
    ``P > 25 kPa`` and for the actuator used in the cited experiment.  Kss,
    Ksl and their pre-load remain installation-specific parameters; their
    defaults are still placeholders and must not be described as calibrated.
    """

    f0_n: float | Iterable[float] = -10.278
    f1_n_per_kpa: float | Iterable[float] = 0.585
    k0_n_per_mm: float | Iterable[float] = 0.33547
    k1_n_per_mm_kpa: float | Iterable[float] = 0.01567
    b_inflate_n_s_per_mm: float | Iterable[float] = 1.10000
    b_inflate_slope_n_s_per_mm_kpa: float | Iterable[float] = 0.00405
    b_deflate_n_s_per_mm: float | Iterable[float] = 0.70000
    b_deflate_slope_n_s_per_mm_kpa: float | Iterable[float] = -0.00261
    k_ss_n_per_mm: float | Iterable[float] = 0.0
    k_sl_n_per_mm: float | Iterable[float] = 3.0
    x_ss_pre_mm: float | Iterable[float] = 0.0
    max_force_n: float | Iterable[float] = 500.0

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: _ten(getattr(self, name), name) for name in (
            "f0_n", "f1_n_per_kpa", "k0_n_per_mm", "k1_n_per_mm_kpa",
            "b_inflate_n_s_per_mm", "b_inflate_slope_n_s_per_mm_kpa",
            "b_deflate_n_s_per_mm", "b_deflate_slope_n_s_per_mm_kpa",
            "k_ss_n_per_mm", "k_sl_n_per_mm", "x_ss_pre_mm", "max_force_n")}


def damping_coefficient(pressure_kpa: Iterable[float], inflating: Iterable[bool],
                        params: MuscleSystemParameters) -> np.ndarray:
    """Return pressure-dependent damping in N*s/mm.

    No 25/30 kPa clipping is performed here: 25 kPa is the identification
    validity boundary, while 30 kPa belongs solely to the unlock decision.
    """
    p = np.maximum(0.0, np.asarray(pressure_kpa, dtype=float).reshape(N_PAM))
    up = np.asarray(inflating, dtype=bool).reshape(N_PAM)
    a = params.arrays()
    b_inf = (a["b_inflate_n_s_per_mm"]
             + a["b_inflate_slope_n_s_per_mm_kpa"]*p)
    b_def = (a["b_deflate_n_s_per_mm"]
             + a["b_deflate_slope_n_s_per_mm_kpa"]*p)
    result = np.where(up, b_inf, b_def)
    if np.any(result <= 0.0):
        raise ValueError("identified damping relation is non-positive at this pressure")
    return result


def muscle_components(pressure_kpa: Iterable[float], x_ss_mm: Iterable[float],
                      x_ss_rate_mm_s: Iterable[float], inflating: Iterable[bool],
                      params: MuscleSystemParameters) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(F_PAM, F_ss)`` using the corrected sign convention."""
    p = np.maximum(0.0, np.asarray(pressure_kpa, dtype=float).reshape(N_PAM))
    x = np.asarray(x_ss_mm, dtype=float).reshape(N_PAM)
    xd = np.asarray(x_ss_rate_mm_s, dtype=float).reshape(N_PAM)
    up = np.asarray(inflating, dtype=bool).reshape(N_PAM)
    a = params.arrays()
    fm = a["f0_n"] + a["f1_n_per_kpa"]*p
    km = a["k0_n_per_mm"] + a["k1_n_per_mm_kpa"]*p
    bm = damping_coefficient(p, up, params)
    f_pam = fm + bm*xd + km*x
    f_ss = a["k_ss_n_per_mm"]*(a["x_ss_pre_mm"] + x)
    return f_pam, f_ss


def system_force(pressure_kpa: Iterable[float], x_ss_mm: Iterable[float],
                 x_ss_rate_mm_s: Iterable[float], x_sl_mm: Iterable[float],
                 locked: Iterable[bool], inflating: Iterable[bool],
                 params: MuscleSystemParameters) -> np.ndarray:
    """Corrected document Eq. (50).

    F_sys = P1*(F_PAM + F_ss) + (1-P1)*K_sl*x_sl.
    ``P1`` multiplies the complete active branch; omitting that multiplier is
    the algebraic error in the source document.
    """
    lock = np.asarray(locked, dtype=float).reshape(N_PAM)
    x_sl = np.asarray(x_sl_mm, dtype=float).reshape(N_PAM)
    a = params.arrays()
    f_pam, f_ss = muscle_components(
        pressure_kpa, x_ss_mm, x_ss_rate_mm_s, inflating, params)
    f_sl = a["k_sl_n_per_mm"]*x_sl
    force = lock*(f_pam + f_ss) + (1.0-lock)*f_sl
    return np.clip(force, 0.0, a["max_force_n"])


class SelfLockingMuscleBank:
    """Ten independent self-locking muscle systems driven by six valves.

    The released branch integrates the document equilibrium equation.  At a
    lock transition the current Ksl displacement is captured; while locked,
    the active-side displacement follows geometric length with that boundary
    held fixed.  The class contains no pressure threshold.
    """

    def __init__(self, rest_lengths_mm: Optional[Iterable[float]] = None,
                 params: Optional[MuscleSystemParameters] = None):
        self.params = params or MuscleSystemParameters()
        self.rest_lengths_mm = (pam_lengths_mm(np.zeros(3)) if rest_lengths_mm is None
                                else np.asarray(rest_lengths_mm, dtype=float).reshape(N_PAM))
        self.x_ss_mm = np.zeros(N_PAM)
        self.x_sl_lock_mm = np.zeros(N_PAM)
        self.previous_delta_mm = np.zeros(N_PAM)
        self.previous_pressure_kpa = np.zeros(N_PAM)
        self.previous_locked = np.zeros(N_PAM, dtype=bool)

    def reset(self) -> None:
        self.x_ss_mm[:] = 0.0
        self.x_sl_lock_mm[:] = 0.0
        self.previous_delta_mm[:] = 0.0
        self.previous_pressure_kpa[:] = 0.0
        self.previous_locked[:] = False

    def step(self, lengths_mm: Iterable[float], valve_pressure_kpa: Iterable[float],
             locked_valves: Iterable[bool], dt_s: float) -> np.ndarray:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        length = np.asarray(lengths_mm, dtype=float).reshape(N_PAM)
        pressure = valve_to_geometry_rows(valve_pressure_kpa)
        locked = valve_to_geometry_rows(
            np.asarray(locked_valves, dtype=float).reshape(N_VALVE)) > 0.5
        delta = length - self.rest_lengths_mm
        delta_rate = (delta - self.previous_delta_mm)/dt_s
        inflating = pressure >= self.previous_pressure_kpa
        a = self.params.arrays()

        newly_locked = locked & ~self.previous_locked
        self.x_sl_lock_mm[newly_locked] = (
            delta[newly_locked] - self.x_ss_mm[newly_locked])

        released = ~locked
        if np.any(released):
            fm = a["f0_n"] + a["f1_n_per_kpa"]*pressure
            km = a["k0_n_per_mm"] + a["k1_n_per_mm_kpa"]*pressure
            bm = damping_coefficient(pressure, inflating, self.params)
            stiffness = km + a["k_ss_n_per_mm"] + a["k_sl_n_per_mm"]
            numerator = (-fm - stiffness*self.x_ss_mm
                         - a["k_ss_n_per_mm"]*a["x_ss_pre_mm"]
                         + a["k_sl_n_per_mm"]*delta)
            dynamic = released & (bm > 1.0e-12)
            self.x_ss_mm[dynamic] += dt_s*numerator[dynamic]/bm[dynamic]
            algebraic = released & ~dynamic
            safe = np.where(np.abs(stiffness) > 1.0e-12, stiffness, 1.0)
            self.x_ss_mm[algebraic] = (
                (a["k_sl_n_per_mm"]*delta - fm
                 - a["k_ss_n_per_mm"]*a["x_ss_pre_mm"])/safe)[algebraic]

        self.x_ss_mm[locked] = delta[locked] - self.x_sl_lock_mm[locked]
        x_ss_rate = np.where(
            locked, delta_rate,
            (-((a["f0_n"] + a["f1_n_per_kpa"]*pressure)
               + (a["k0_n_per_mm"] + a["k1_n_per_mm_kpa"]*pressure
                  + a["k_ss_n_per_mm"] + a["k_sl_n_per_mm"])*self.x_ss_mm
               + a["k_ss_n_per_mm"]*a["x_ss_pre_mm"]
               - a["k_sl_n_per_mm"]*delta)
             / np.maximum(damping_coefficient(
                 pressure, inflating, self.params), 1.0e-12)))
        x_sl = np.where(locked, self.x_sl_lock_mm, delta-self.x_ss_mm)
        force = system_force(
            pressure, self.x_ss_mm, x_ss_rate, x_sl, locked, inflating, self.params)

        self.previous_delta_mm = delta.copy()
        self.previous_pressure_kpa = pressure.copy()
        self.previous_locked = locked.copy()
        return force


def spatial_moment_arm_metrics(q: Iterable[float]) -> Tuple[int, np.ndarray, float]:
    singular_values = np.linalg.svd(H_matrix(q), compute_uv=False)
    rank = int(np.linalg.matrix_rank(H_matrix(q)))
    condition = float(singular_values[0]/singular_values[-1])
    return rank, singular_values, condition


__all__ = [name for name in globals() if not name.startswith("_")]

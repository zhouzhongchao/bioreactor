# -*- coding: utf-8 -*-
"""acados SQP-RTI backend for the data-driven shoulder MPC.

The neural forward model is deliberately kept outside CasADi.  Every control
tick the caller evaluates/linearizes that model along the current nominal
trajectory and supplies

    x[k+1] = A[k] x[k] + B[k] u_act[k] + c[k]

as stage parameters.  acados then solves the resulting constrained OCP with
its native SQP_RTI implementation and HPIPM QP solver.  The previous applied
voltage is appended to the state so ``rd * ||u_act[k]-u_act[k-1]||^2`` is part
of the OCP rather than an after-solve filter.

This module is optional: importing mpc_station.py does not require acados.
An acados installation must provide both ``casadi``/``acados_template`` and
the compiled acados shared libraries.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import numpy as np


NX_PLANT = 12
NU = 6
NX = NX_PLANT + NU


@dataclass(frozen=True)
class _ParameterSlices:
    A: slice
    B: slice
    c: slice
    ref: slice
    center: slice
    availability: slice
    wq: slice
    wv: slice
    rd: int
    rho: int
    l1: int
    x_comp_ratio: int
    lambda_c: int
    size: int


def _parameter_slices() -> _ParameterSlices:
    i = 0

    def take(n):
        nonlocal i
        ans = slice(i, i+n)
        i += n
        return ans

    A = take(NX_PLANT*NX_PLANT)
    B = take(NX_PLANT*NU)
    c = take(NX_PLANT)
    ref = take(3)
    center = take(NU)
    availability = take(NU)
    wq = take(3)
    wv = take(3)
    rd = i; i += 1
    rho = i; i += 1
    l1 = i; i += 1
    x_comp_ratio = i; i += 1
    lambda_c = i; i += 1
    return _ParameterSlices(A, B, c, ref, center, availability,
                            wq, wv, rd, rho, l1, x_comp_ratio,
                            lambda_c, i)


PS = _parameter_slices()


def availability_message(exc: BaseException) -> str:
    """Turn acados' often long import/build errors into one useful sentence."""
    detail = str(exc).strip().replace("\n", " ")
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return ("缺少 casadi/acados_template；还需要已编译的 acados 动态库"
                + (f"（{detail}）" if detail else ""))
    return "acados生成或加载失败" + (f"：{detail}" if detail else "")


class AcadosSQPRTIBackend:
    """Reusable, parameterized acados SQP-RTI OCP solver."""

    def __init__(self, horizon, sample_dt, u_max, *,
                 qp_solver="PARTIAL_CONDENSING_HPIPM", cond_N=5,
                 qp_iter_max=100, codegen_dir=None, cmake_generator="",
                 source_dir=""):
        source_dir = str(source_dir or os.environ.get(
            "ACADOS_SOURCE_DIR", "")).strip()
        if source_dir:
            os.environ["ACADOS_SOURCE_DIR"] = os.path.abspath(source_dir)
            interface_dir = os.path.join(
                os.path.abspath(source_dir), "interfaces", "acados_template")
            if os.path.isdir(interface_dir) and interface_dir not in sys.path:
                sys.path.insert(0, interface_dir)
        try:
            import casadi as ca
            from acados_template import (AcadosModel, AcadosOcp,
                                         AcadosOcpSolver,
                                         ocp_get_default_cmake_builder)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(availability_message(exc)) from exc

        self.N = max(1, int(horizon))
        self.dt = float(sample_dt)
        self.u_max = float(u_max)
        self.last_qp_iter = 0
        self.last_preparation_ms = 0.0
        self.last_feedback_ms = 0.0

        solver_name = re.sub(r"[^0-9A-Za-z_]", "_", str(qp_solver).lower())
        model = AcadosModel()
        model.name = f"tro_ltv_sqp_rti_n{self.N}_{solver_name}"
        z = ca.SX.sym("z", NX)
        u = ca.SX.sym("u", NU)
        p = ca.SX.sym("p", PS.size)
        x = z[:NX_PLANT]
        u_previous = z[NX_PLANT:]

        # numpy arrays are packed row-major; transpose after CasADi's
        # column-major reshape to recover the original A and B matrices.
        A = ca.reshape(p[PS.A], NX_PLANT, NX_PLANT).T
        B = ca.reshape(p[PS.B], NU, NX_PLANT).T
        c = p[PS.c]
        ref = p[PS.ref]
        center = p[PS.center]
        available = p[PS.availability]
        wq = p[PS.wq]
        wv = p[PS.wv]
        u_actual = available*u
        x_next = ca.mtimes(A, x) + ca.mtimes(B, u_actual) + c

        model.x = z
        model.u = u
        model.p = p
        model.disc_dyn_expr = ca.vertcat(x_next, u_actual)

        angle_error = x_next[:3] - ref
        velocity = x_next[3:6] / self.dt
        du = u_actual - u_previous
        rho_error = u_actual - available*center
        compensation_shortfall = ca.fmax(
            p[PS.x_comp_ratio]*u_actual[4] - u_actual[3], 0.0)
        model.cost_expr_ext_cost = (
            ca.dot(wq, angle_error*angle_error)
            + ca.dot(wv, velocity*velocity)
            + p[PS.rd]*ca.dot(du, du)
            + p[PS.rho]*ca.dot(rho_error, rho_error)
            + p[PS.l1]*ca.sum1(u_actual)
            + 0.5*p[PS.lambda_c]*compensation_shortfall**2
        )
        # The stage cost above already scores x[k+1] for all N transitions.
        model.cost_expr_ext_cost_e = 0.0*ca.sum1(z)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N*self.dt
        ocp.parameter_values = np.zeros(PS.size)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.x0 = np.zeros(NX)
        ocp.constraints.idxbu = np.arange(NU, dtype=np.int64)
        ocp.constraints.lbu = np.zeros(NU)
        ocp.constraints.ubu = np.full(NU, self.u_max)
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = str(qp_solver).strip().upper()
        ocp.solver_options.qp_solver_cond_N = int(np.clip(cond_N, 1, self.N))
        ocp.solver_options.qp_solver_iter_max = max(20, int(qp_iter_max))
        ocp.solver_options.qp_solver_warm_start = 2
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.regularize_method = "CONVEXIFY"
        ocp.solver_options.levenberg_marquardt = 1e-8
        ocp.solver_options.print_level = 0

        if codegen_dir is None:
            codegen_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "acados_generated",
                model.name)
        codegen_dir = os.path.abspath(codegen_dir)
        os.makedirs(codegen_dir, exist_ok=True)
        ocp.code_gen_options.code_export_directory = codegen_dir
        ocp.code_gen_options.json_file = os.path.join(
            codegen_dir, f"{model.name}.json")

        builder = ocp_get_default_cmake_builder()
        builder.build_dir = os.path.join(codegen_dir, "build")
        requested_generator = str(cmake_generator or "").strip()
        if requested_generator:
            builder.generator = requested_generator
        json_path = ocp.code_gen_options.json_file
        # Newer acados versions add an ``ocp_`` prefix and a content hash to
        # the generated library name.  Looking only for the old exact name
        # would rebuild the same solver on every GUI start.
        generated_library_prefixes = (
            "acados_ocp_solver_", "libacados_ocp_solver_")
        reuse_generated_code = (os.path.isfile(json_path) and any(
            name.startswith(generated_library_prefixes)
            and name.lower().endswith((".dll", ".so", ".dylib"))
            for name in os.listdir(codegen_dir)))
        try:
            self.solver = AcadosOcpSolver(
                ocp, cmake_builder=builder, verbose=False,
                generate=not reuse_generated_code,
                build=not reuse_generated_code,
                check_reuse_possible=True)
        except (Exception, SystemExit) as exc:
            raise RuntimeError(availability_message(exc)) from exc

    @staticmethod
    def pack_parameters(A, B, c, ref, center, availability,
                        wq, wv, rd, rho, l1, x_comp_ratio, lambda_c):
        values = np.empty(PS.size, dtype=float)
        values[PS.A] = np.asarray(A, float).reshape(NX_PLANT, NX_PLANT).ravel()
        values[PS.B] = np.asarray(B, float).reshape(NX_PLANT, NU).ravel()
        values[PS.c] = np.asarray(c, float).reshape(NX_PLANT)
        values[PS.ref] = np.asarray(ref, float).reshape(3)
        values[PS.center] = np.asarray(center, float).reshape(NU)
        values[PS.availability] = np.asarray(availability, float).reshape(NU)
        values[PS.wq] = np.asarray(wq, float).reshape(3)
        values[PS.wv] = np.asarray(wv, float).reshape(3)
        values[PS.rd] = float(rd)
        values[PS.rho] = float(rho)
        values[PS.l1] = float(l1)
        values[PS.x_comp_ratio] = float(x_comp_ratio)
        values[PS.lambda_c] = float(lambda_c)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("acados阶段参数含NaN/Inf")
        return values

    def solve(self, *, x0, A, B, c, refs, centers, availability,
              wq, wv, rd, rho, l1, x_comp_ratio, lambda_c,
              lower, upper, nominal_states, nominal_controls,
              previous_control):
        """Run preparation + feedback phases and return the N-by-6 plan."""
        A = np.asarray(A, float).reshape(self.N, NX_PLANT, NX_PLANT)
        B = np.asarray(B, float).reshape(self.N, NX_PLANT, NU)
        c = np.asarray(c, float).reshape(self.N, NX_PLANT)
        refs = np.asarray(refs, float).reshape(self.N, 3)
        centers = np.asarray(centers, float).reshape(self.N, NU)
        availability = np.asarray(availability, float).reshape(self.N, NU)
        lower = np.asarray(lower, float).reshape(self.N, NU)
        upper = np.asarray(upper, float).reshape(self.N, NU)
        nominal_states = np.asarray(nominal_states, float).reshape(self.N+1, NX_PLANT)
        nominal_controls = np.asarray(nominal_controls, float).reshape(self.N, NU)
        x0_aug = np.r_[np.asarray(x0, float).reshape(NX_PLANT),
                       np.asarray(previous_control, float).reshape(NU)]

        previous = np.asarray(previous_control, float).reshape(NU)
        for k in range(self.N):
            pk = self.pack_parameters(
                A[k], B[k], c[k], refs[k], centers[k], availability[k],
                wq, wv, rd, rho, l1, x_comp_ratio, lambda_c)
            self.solver.set(k, "p", pk)
            self.solver.constraints_set(k, "lbu", lower[k])
            self.solver.constraints_set(k, "ubu", upper[k])
            self.solver.set(k, "x", np.r_[nominal_states[k], previous])
            self.solver.set(k, "u", nominal_controls[k])
            previous = availability[k]*nominal_controls[k]
        self.solver.set(self.N, "x", np.r_[nominal_states[-1], previous])

        # acados' genuine RTI split: prepare the QP first, then inject the
        # newest measured state and perform the feedback solve.
        self.solver.options_set("rti_phase", 1)
        status_prepare = int(self.solver.solve())
        self.last_preparation_ms = 1000.0*float(self.solver.get_stats("time_tot"))
        self.solver.set(0, "lbx", x0_aug)
        self.solver.set(0, "ubx", x0_aug)
        self.solver.options_set("rti_phase", 2)
        status_feedback = int(self.solver.solve())
        self.last_feedback_ms = 1000.0*float(self.solver.get_stats("time_tot"))
        # In current acados, preparation-only RTI deliberately returns
        # ACADOS_READY (=5); the feedback phase must return SUCCESS (=0).
        if status_prepare not in (0, 5) or status_feedback != 0:
            raise RuntimeError(
                f"acados SQP-RTI未收敛：prepare={status_prepare}, "
                f"feedback={status_feedback}")

        plan = np.vstack([np.asarray(self.solver.get(k, "u"), float)
                          for k in range(self.N)])
        if plan.shape != (self.N, NU) or not np.all(np.isfinite(plan)):
            raise FloatingPointError("acados返回的电压序列含NaN/Inf")
        try:
            self.last_qp_iter = int(np.sum(self.solver.get_stats("qp_iter")))
        except Exception:
            self.last_qp_iter = 0
        return plan

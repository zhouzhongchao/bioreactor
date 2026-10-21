# -*- coding: utf-8 -*-
"""Human-triggered Codex adaptive compensator for the fixed LTV MPC.

This module implements the deployment side of the LLM-guided adaptive
compensator framework described in arXiv:2507.20509.  A completed hardware lap
is summarized for an offline LLM.  The LLM returns a restricted numerical
``Compensator`` class rather than a fixed PID parameter vector.  The class is
statically checked, replayed twice on logged inputs, and only then admitted to
one real-hardware canary lap.  Final voltage projection, lock gating and
antagonist exclusion remain outside generated code; a worse lap rolls back.
"""
from __future__ import annotations

import hashlib
import ast
import builtins
import json
import math
import os
import glob
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


NAX = 3
NCH = 6
ANTAGONIST_PAIRS = ((0, 1), (2, 3), (4, 5))
CODEX_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "codex_compensator_prompt.json")
PLUS_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "codex_compensator_prompt_advanced.json")
PRO_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "codex_compensator_prompt_pro.json")
PROMPT_TEMPLATE_PATHS = {
    "base": PROMPT_TEMPLATE_PATH,
    "plus": PLUS_PROMPT_TEMPLATE_PATH,
    "pro": PRO_PROMPT_TEMPLATE_PATH,
}
LLM_RECORDING_SESSION_ENV = "TRO_RECORDING_SESSION_DIR"


def _llm_recording_paths(request_id: str) -> Dict[str, str]:
    """Return per-request audit paths inside the active GUI recording folder."""
    folder = str(os.environ.get(LLM_RECORDING_SESSION_ENV, "") or "").strip()
    if not folder:
        return {}
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return {}
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                      for ch in str(request_id or "unknown"))[:120] or "unknown"
    return {
        "reasoning": os.path.join(folder, f"llm_reasoning_{safe_id}.jsonl"),
        "final": os.path.join(folder, f"llm_final_{safe_id}.json"),
    }


def _append_llm_audit_events(path: str, phase: str, stdout: str = "",
                             stderr: str = "", payload: object = None) -> None:
    """Append provider-visible events without rewriting or interpreting them."""
    if not path:
        return
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records = []
    if payload is not None:
        records.append({"recorded_at": recorded_at, "phase": str(phase),
                        "stream": "local", "event": payload})
    for stream_name, text in (("stdout", stdout), ("stderr", stderr)):
        for line in str(text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                event = {"type": "text", "text": line}
            records.append({"recorded_at": recorded_at, "phase": str(phase),
                            "stream": stream_name, "event": event})
    if not records:
        return
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(
                    record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        # Audit output must never stop the fixed MPC or candidate validation.
        pass


def _write_llm_final_result(path: str, candidate: object,
                            validation_error: str = "",
                            raw_output: str = "") -> None:
    """Atomically save the final returned design, including rejected output."""
    if not path:
        return
    if validation_error:
        result = {
            "state": "validation_failed",
            "validation_error": str(validation_error),
            "model_result": candidate,
        }
        if candidate is None and raw_output:
            result["raw_output"] = str(raw_output)
    else:
        result = candidate
    try:
        folder = os.path.dirname(os.path.abspath(path))
        fd, temp_path = tempfile.mkstemp(
            prefix=".llm_final_", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2,
                          allow_nan=False)
                stream.write("\n")
            os.replace(temp_path, path)
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError):
        # A recording failure must not change the deployed compensator.
        pass

DEFAULT_COMPENSATOR_SOURCE = """class Compensator:
    def __init__(self):
        self.kp_matrix = np.zeros((6, 3))
        self.kd_matrix = np.zeros((6, 3))
        self.ki_matrix = np.zeros((6, 3))
        self.integral_limit_deg_s = np.zeros(3)
        self.output_limit_mv = np.ones(6)
        self.output_slew_mv_per_s = np.ones(6)
        self.integral_error = np.zeros(3)

    def reset(self, feedback):
        self.integral_error = np.zeros(3)

    def compute(self, u_mpc, pose, velocity, reference_pose,
                reference_velocity, pressure, lock_state, trajectory_time):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
"""

_SAFE_NUMPY_CALLS = {
    "array", "asarray", "zeros", "ones", "clip", "maximum", "minimum",
    "where", "dot", "matmul", "mean", "sum", "abs", "sign", "sqrt",
}
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "float": float, "int": int,
    "len": len, "bool": bool,
}


def _numpy_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Support NumPy's lazy C-level imports inside restricted numeric code."""
    if level != 0 or not (name == "numpy" or name.startswith("numpy.")):
        raise ImportError(f"Compensator runtime import denied: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _validate_compensator_source(source: object,
                                 require_design_contract: bool = False):
    """Compile a small loop-free numeric Compensator class in a restricted namespace."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("candidate source_code must be a nonempty string")
    source = source.strip()
    source_lines = source.splitlines()
    if source_lines and source_lines[0].strip() == "import numpy as np":
        source = "\n".join(source_lines[1:]).strip()
    if len(source) > 20000 or "\x00" in source:
        raise ValueError("candidate source_code is too large or contains NUL")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"candidate source_code syntax error: {exc}") from exc
    ast_node_count = len(list(ast.walk(tree)))
    if ast_node_count > 2400:
        raise ValueError(
            f"candidate source_code AST is too large ({ast_node_count} > 2400)")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "Compensator":
        raise ValueError("source_code must contain exactly one Compensator class")
    if len(tree.body) != 1:
        raise ValueError(
            "source_code may contain only the Compensator class; np is already provided")
    cls = classes[0]
    if cls.bases or cls.keywords or cls.decorator_list:
        raise ValueError("Compensator inheritance and decorators are not allowed")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    required_signatures = {
        "__init__": ["self"],
        "reset": ["self", "feedback"],
        "compute": ["self", "u_mpc", "pose", "velocity", "reference_pose",
                    "reference_velocity", "pressure", "lock_state", "trajectory_time"],
    }
    if set(methods) != set(required_signatures):
        raise ValueError("Compensator must define only __init__, reset, and compute")
    for name, expected in required_signatures.items():
        function = methods[name]
        actual = [argument.arg for argument in function.args.args]
        if (actual != expected or function.args.vararg or function.args.kwarg or
                function.args.kwonlyargs or function.decorator_list):
            raise ValueError(f"invalid Compensator.{name} signature")

    forbidden = (ast.Import, ast.ImportFrom, ast.For, ast.AsyncFor, ast.While,
                 ast.Try, ast.With, ast.AsyncWith, ast.Lambda, ast.ListComp,
                 ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Await,
                 ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal, ast.Delete,
                 ast.Raise, ast.Assert, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"forbidden source_code syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and abs(float(node.value)) > 1.0e9:
                raise ValueError("numeric literal is too large")
            if isinstance(node.value, (bytes, complex)):
                raise ValueError("bytes and complex literals are not allowed")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in _SAFE_NUMPY_CALLS:
                    raise ValueError(f"np.{node.attr} is not allowed")
            elif isinstance(node.value, ast.Name) and node.value.id == "self":
                if node.attr.startswith("_"):
                    raise ValueError("private/dunder state attributes are not allowed")
            else:
                raise ValueError("attribute access is limited to self state and safe np functions")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_BUILTINS:
                    raise ValueError(f"call to {node.func.id} is not allowed")
            elif (isinstance(node.func, ast.Attribute) and
                  isinstance(node.func.value, ast.Name) and
                  node.func.value.id == "np"):
                if node.func.attr not in _SAFE_NUMPY_CALLS:
                    raise ValueError(f"np.{node.func.attr} is not allowed")
            else:
                raise ValueError("only safe numeric function calls are allowed")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (node.targets if isinstance(node, ast.Assign) else [node.target])
            for target in targets:
                pending = [target]
                while pending:
                    current = pending.pop()
                    if isinstance(current, ast.Name):
                        continue
                    if isinstance(current, (ast.Tuple, ast.List)):
                        pending.extend(current.elts)
                        continue
                    if isinstance(current, ast.Subscript):
                        base = current.value
                        while isinstance(base, ast.Subscript):
                            base = base.value
                        if isinstance(base, ast.Name) and base.id != "np":
                            continue
                        if (isinstance(base, ast.Attribute) and
                                isinstance(base.value, ast.Name) and
                                base.value.id == "self" and
                                not base.attr.startswith("_")):
                            continue
                    if (isinstance(current, ast.Attribute) and
                            isinstance(current.value, ast.Name) and
                            current.value.id == "self" and
                            not current.attr.startswith("_")):
                        continue
                    raise ValueError(
                        "assignment targets must be local variables, local array items, or self state")

    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__build_class__"] = builtins.__build_class__
    # Not in _SAFE_BUILTINS: explicit candidate calls/imports stay forbidden.
    # NumPy string comparisons may consult the caller's builtins lazily.
    safe_builtins["__import__"] = _numpy_runtime_import
    namespace = {"__builtins__": safe_builtins,
                 "__name__": "validated_compensator", "np": np}
    try:
        exec(compile(tree, "<candidate_compensator>", "exec"), namespace, namespace)
        instance = namespace["Compensator"]()
        instance.reset(np.zeros(NAX, dtype=float))
    except Exception as exc:
        raise ValueError(f"candidate Compensator cannot initialize: {exc}") from exc
    if require_design_contract:
        limits = np.asarray(
            getattr(instance, "output_limit_mv", None), dtype=float)
        slew = np.asarray(
            getattr(instance, "output_slew_mv_per_s", None), dtype=float)
        # PID structure is intentionally not prescribed.  Conventional matrix
        # gains are validated when a candidate chooses to expose them, but a
        # valid PID may instead differentiate angle error, filter its D term,
        # or use another bounded state representation.
        for name in ("kp_matrix", "kd_matrix", "ki_matrix"):
            value = getattr(instance, name, None)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (NCH, NAX) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Compensator.{name} must be finite 6x3 when provided")
        integral_value = getattr(instance, "integral_limit_deg_s", None)
        if integral_value is not None:
            integral_limit = np.asarray(integral_value, dtype=float)
            if (integral_limit.shape != (NAX,) or
                    not np.all(np.isfinite(integral_limit)) or
                    np.any(integral_limit < 0.0)):
                raise ValueError(
                    "Compensator.integral_limit_deg_s must contain three finite "
                    "nonnegative values when provided")
        if (limits.shape != (NCH,) or not np.all(np.isfinite(limits)) or
                np.any(limits < 0.0)):
            raise ValueError(
                "Compensator.output_limit_mv must contain six finite nonnegative values")
        if (slew.shape != (NCH,) or not np.all(np.isfinite(slew)) or
                np.any(slew < 0.0) or np.any((limits > 0.0) & (slew <= 0.0))):
            raise ValueError(
                "Compensator.output_slew_mv_per_s must contain six finite "
                "nonnegative values and be positive wherever output_limit_mv is positive")
        compute_tree = methods["compute"]
        loaded_self_attributes = {
            node.attr for node in ast.walk(compute_tree)
            if (isinstance(node, ast.Attribute) and
                isinstance(node.value, ast.Name) and node.value.id == "self" and
                isinstance(node.ctx, ast.Load))
        }
        required_uses = {"output_limit_mv", "output_slew_mv_per_s"}
        if not required_uses.issubset(loaded_self_attributes):
            missing = sorted(required_uses-loaded_self_attributes)
            raise ValueError(
                f"Compensator.compute does not use required design fields: {missing}")
        loaded_inputs = {
            node.id for node in ast.walk(compute_tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if not {"pose", "reference_pose"}.issubset(loaded_inputs):
            raise ValueError(
                "Compensator.compute must use pose and reference_pose for PID tracking")
        uses_lock_state = any(
            isinstance(node, ast.Name) and node.id == "lock_state" and
            isinstance(node.ctx, ast.Load)
            for node in ast.walk(compute_tree))
        if not uses_lock_state:
            raise ValueError(
                "Compensator.compute must gate its slew state with lock_state")
    return source, instance


def _run_compensator(instance, u_mpc, pose, velocity, reference_pose,
                     reference_velocity, pressure, lock_state, trajectory_time):
    output = instance.compute(
        np.asarray(u_mpc, dtype=float).copy(),
        np.asarray(pose, dtype=float).copy(),
        np.asarray(velocity, dtype=float).copy(),
        np.asarray(reference_pose, dtype=float).copy(),
        np.asarray(reference_velocity, dtype=float).copy(),
        np.asarray(pressure, dtype=float).copy(), tuple(lock_state),
        float(trajectory_time))
    values = np.asarray(output, dtype=float)
    if values.shape != (NCH,) or not np.all(np.isfinite(values)):
        raise ValueError("Compensator.compute must return six finite numeric values")
    return values


def _optional_compensator_array(instance, name: str, shape):
    """Return an optional conventional PID field without prescribing PID form."""
    value = getattr(instance, name, None)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"Compensator.{name} must be finite with shape {tuple(shape)} when provided")
    return array.tolist()


def _probe_compensator_source(source: object, trace: Optional[Iterable[dict]] = None) -> str:
    clean, first = _validate_compensator_source(
        source, require_design_contract=True)
    _, second = _validate_compensator_source(
        clean, require_design_contract=True)
    rows = [row for row in (trace or []) if isinstance(row, dict)]
    if not rows:
        rows = [{
            "t": value,
            "base_u_mv": [400.0, 0.0, 300.0, 0.0, 200.0, 0.0],
            "response_deg": [0.0, 0.0, 0.0],
            "velocity_deg_s": [0.0, 0.0, 0.0],
            "reference_deg": [10.0, 20.0, -10.0],
            "reference_velocity_deg_s": [5.0, 5.0, -5.0],
            "pressure_kpa": [20.0] * NCH,
            "locks": ["L"] * NCH,
        } for value in (0.0, 0.5, 1.0, 5.0)]
    feedback = np.asarray(rows[0].get("response_deg", [0.0] * NAX), dtype=float)
    first.reset(feedback.copy())
    second.reset(feedback.copy())
    previous_output = np.zeros(NCH, dtype=float)
    previous_time = 0.0
    for row in rows[:256]:
        pressure = [0.0 if value is None else value
                    for value in row.get("pressure_kpa", [0.0] * NCH)]
        arguments = (
            row.get("base_u_mv", [0.0] * NCH),
            row.get("response_deg", [0.0] * NAX),
            row.get("velocity_deg_s", [0.0] * NAX),
            row.get("reference_deg", [0.0] * NAX),
            row.get("reference_velocity_deg_s", [0.0] * NAX),
            pressure, row.get("locks", ["L"] * NCH), row.get("t", 0.0),
        )
        one = _run_compensator(first, *arguments)
        two = _run_compensator(second, *arguments)
        if not np.allclose(one, two, rtol=0.0, atol=1.0e-9):
            raise ValueError("Compensator is not deterministic on replay")
        limits = np.asarray(first.output_limit_mv, dtype=float)
        slew = np.asarray(first.output_slew_mv_per_s, dtype=float)
        if np.any(np.abs(one) > limits + 1.0e-6):
            raise ValueError("Compensator output exceeds output_limit_mv on replay")
        current_time = float(arguments[-1])
        elapsed = max(0.0, current_time-previous_time)
        locked = np.asarray(
            [str(value).upper() == "L" for value in arguments[-2]], dtype=bool)
        if np.any(np.abs(one[~locked]) > 1.0e-6):
            raise ValueError(
                "Compensator must return zero on unlocked channels")
        if np.any(np.abs(one[locked]-previous_output[locked]) >
                  slew[locked]*elapsed + 1.0e-6):
            raise ValueError(
                "Compensator output exceeds output_slew_mv_per_s on replay")
        previous_output = one
        previous_time = current_time
    return clean


def normalize_reasoning_effort(value: object) -> str:
    """Return a Codex effort that is valid for every GPT model offered by the GUI.

    ``minimal`` was accepted by older GPT models but is not accepted by the
    current GPT-5.6 family.  Treat a saved legacy value as ``low`` so old
    installations migrate safely instead of failing after a hardware lap.
    """
    effort = str(value or "medium").strip().lower()
    if effort == "minimal":
        effort = "low"
    if effort not in CODEX_REASONING_EFFORTS:
        raise ValueError(
            "不支持的Codex推理强度："
            f"{effort}；请选择 low/medium/high/xhigh")
    return effort


def normalize_prompt_profile(value: object) -> str:
    profile = str(value or "base").strip().lower()
    if profile not in PROMPT_TEMPLATE_PATHS:
        raise ValueError("不支持的补偿器设计版本：" + profile)
    return profile


def _zero_parameters() -> Dict[str, object]:
    return {
        "kp_mv_per_deg": np.zeros((NCH, NAX), dtype=float).tolist(),
        "kd_mv_per_deg_s": np.zeros((NCH, NAX), dtype=float).tolist(),
        "ki_mv_per_deg_s": np.zeros((NCH, NAX), dtype=float).tolist(),
        "bias_mv": np.zeros(NCH, dtype=float).tolist(),
        "feedforward_knots": [],
    }


class CodexCompensator:
    """MIMO feedback + phase feedforward proposed off-line by Codex."""

    SCHEMA = 1
    PARAMETER_KEYS = (
        "kp_mv_per_deg", "kd_mv_per_deg_s", "ki_mv_per_deg_s", "bias_mv")
    # These are minimum changes for rejecting no-op candidates, never maxima.
    MIN_MEANINGFUL_CHANGE = {
        "kp_mv_per_deg": 1.0,
        "kd_mv_per_deg_s": 0.5,
        "ki_mv_per_deg_s": 0.25,
        "bias_mv": 40.0,
        "feedforward_mv": 40.0,
    }
    FEEDFORWARD_MAX_KNOTS = 16
    FEEDFORWARD_MIN_SPACING_S = 0.05
    SLOW_RESPONSE_LAG_S = 0.15
    SEVERE_RESPONSE_LAG_S = 0.30
    HOTSPOT_PHASE_BINS = 12
    HOTSPOT_MIN_RMSE_DEG = 2.0
    RECURRENT_HOTSPOT_MIN_RMSE_DEG = 3.0
    DESIGN_MEMORY_MAX = 5
    DESIGN_MEMORY_CHANGE_MAX = 12

    def __init__(self, state_path: str, candidate_path: str, request_path: str,
                 model_reference: np.ndarray,
                 task_reference_deg: np.ndarray, dt: float,
                 u_max: float,
                 max_slew_mv_s: float = 1200.0,
                 integral_limit_deg_s: float = 30.0,
                 accept_ratio: float = 1.0,
                 angle_jump_limit_deg: float = 10.0,
                 target_mse_deg2: float = 15.0,
                 final_fall_slew_mv_s: Optional[float] = None,
                 final_phi_rise_slew_mv_s: Optional[float] = None,
                 unlock_pressure_kpa: float = 30.0):
        self.state_path = os.path.abspath(state_path)
        self.candidate_path = os.path.abspath(candidate_path)
        self.request_path = os.path.abspath(request_path)
        self.model_reference = np.asarray(model_reference, dtype=float)
        self.task_reference_deg = np.asarray(task_reference_deg, dtype=float)
        if self.model_reference.ndim != 2 or self.model_reference.shape[1] != NAX:
            raise ValueError("Codex compensator model reference must have shape T x 3")
        if self.task_reference_deg.shape != self.model_reference.shape:
            raise ValueError("Codex task-space reference must match model reference")
        self.length = len(self.model_reference)
        self.dt = float(dt)
        self.u_max = max(0.0, float(u_max))
        # Kept in the signature for old callers/config files, but intentionally
        # not applied.  The user-selected final voltage projection is the sole
        # numeric amplitude/rate limiter for the residual controller.
        self.max_slew_mv_s = None
        fall_slew = (float(final_fall_slew_mv_s)
                     if final_fall_slew_mv_s is not None else 0.0)
        self.final_fall_slew_mv_s = (
            fall_slew if np.isfinite(fall_slew) and fall_slew > 0.0 else None)
        phi_rise_slew = (float(final_phi_rise_slew_mv_s)
                         if final_phi_rise_slew_mv_s is not None else 0.0)
        self.final_phi_rise_slew_mv_s = (
            phi_rise_slew
            if np.isfinite(phi_rise_slew) and phi_rise_slew > 0.0 else None)
        unlock_pressure = float(unlock_pressure_kpa)
        self.unlock_pressure_kpa = (
            unlock_pressure if np.isfinite(unlock_pressure) and unlock_pressure >= 0.0
            else 30.0)
        self.integral_limit_deg_s = None
        self.accept_ratio = max(0.0, float(accept_ratio))
        self.angle_jump_limit_deg = max(0.0, float(angle_jump_limit_deg))
        target = float(target_mse_deg2)
        self.target_mse_deg2 = target if np.isfinite(target) and target > 0.0 else None
        self.task_hash = self._fingerprint()

        self.generation = 0
        self.active = _zero_parameters()
        self.previous = _zero_parameters()
        self.active_kind = "legacy_parameters"
        self.previous_kind = "legacy_parameters"
        self.active_source = DEFAULT_COMPENSATOR_SOURCE
        self.previous_source = DEFAULT_COMPENSATOR_SOURCE
        self._program = None
        self.canary_active = False
        self.baseline_mse = float("inf")
        self.last_valid_mse = None
        self.last_candidate_digest = ""
        self.history = []
        self.design_memory = []
        self.last_design = {}
        self._load()
        self._recover_interrupted_canary()
        self._reset_runtime()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([self.dt], dtype=np.float64).tobytes())
        # Task identity follows the user's true GUI reference.  Internal MPC
        # coordinate filtering may change without creating a different task.
        digest.update(np.round(self.task_reference_deg, 7).astype(np.float64).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _atomic_json(path: str, payload: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)

    @staticmethod
    def reset_files(state_path: str, candidate_path: str,
                    request_path: Optional[str] = None) -> None:
        for path in (state_path, candidate_path, request_path):
            if path and os.path.isfile(path):
                os.remove(path)

    def _state_payload(self) -> Dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "task_hash": self.task_hash,
            "generation": int(self.generation),
            "active": self.active,
            "previous": self.previous,
            "active_kind": self.active_kind,
            "previous_kind": self.previous_kind,
            "active_source": self.active_source,
            "previous_source": self.previous_source,
            "canary_active": bool(self.canary_active),
            "baseline_mse_deg2": (float(self.baseline_mse)
                                   if np.isfinite(self.baseline_mse) else None),
            "last_valid_mse_deg2": (float(self.last_valid_mse)
                                      if self.last_valid_mse is not None else None),
            "last_candidate_digest": self.last_candidate_digest,
            "last_design": self.last_design,
            "history": self.history[-30:],
            # Compact cross-run evidence supplied to the next offline design.
            # Keep this deliberately small; the full 20 Hz traces remain in
            # the archived request files and do not belong in persistent state.
            "design_memory": self.design_memory[-self.DESIGN_MEMORY_MAX:],
        }

    def _save(self) -> None:
        self._atomic_json(self.state_path, self._state_payload())

    def _load(self) -> None:
        if not os.path.isfile(self.state_path):
            self._save()
            return
        try:
            with open(self.state_path, encoding="utf-8") as stream:
                blob = json.load(stream)
            if (int(blob.get("schema", -1)) != self.SCHEMA or
                    str(blob.get("task_hash", "")) != self.task_hash):
                self._save()
                return
            self.active = self._validate_parameters(blob.get("active"), None)
            self.previous = self._validate_parameters(blob.get("previous"), None)
            self.active_kind = str(blob.get("active_kind", "legacy_parameters"))
            self.previous_kind = str(blob.get("previous_kind", "legacy_parameters"))
            if self.active_kind not in ("legacy_parameters", "python_class"):
                self.active_kind = "legacy_parameters"
            if self.previous_kind not in ("legacy_parameters", "python_class"):
                self.previous_kind = "legacy_parameters"
            self.active_source = str(blob.get(
                "active_source", DEFAULT_COMPENSATOR_SOURCE))
            self.previous_source = str(blob.get(
                "previous_source", DEFAULT_COMPENSATOR_SOURCE))
            if self.active_kind == "python_class":
                self.active_source, _ = _validate_compensator_source(self.active_source)
            if self.previous_kind == "python_class":
                self.previous_source, _ = _validate_compensator_source(self.previous_source)
            self.generation = max(0, int(blob.get("generation", 0)))
            self.canary_active = bool(blob.get("canary_active", False))
            base = blob.get("baseline_mse_deg2")
            self.baseline_mse = float(base) if base is not None else float("inf")
            last = blob.get("last_valid_mse_deg2")
            self.last_valid_mse = float(last) if last is not None else None
            self.last_candidate_digest = str(blob.get("last_candidate_digest", ""))
            self.last_design = dict(blob.get("last_design") or {})
            self.history = list(blob.get("history") or [])[-30:]
            memory = blob.get("design_memory") or []
            self.design_memory = ([dict(item) for item in memory
                                   if isinstance(item, dict)]
                                  [-self.DESIGN_MEMORY_MAX:])
        except Exception:
            self.generation = 0
            self.active = _zero_parameters()
            self.previous = _zero_parameters()
            self.active_kind = "legacy_parameters"
            self.previous_kind = "legacy_parameters"
            self.active_source = DEFAULT_COMPENSATOR_SOURCE
            self.previous_source = DEFAULT_COMPENSATOR_SOURCE
            self.canary_active = False
            self.baseline_mse = float("inf")
            self.last_valid_mse = None
            self.last_candidate_digest = ""
            self.history = []
            self.design_memory = []
            self.last_design = {}
            self._save()

    def _recover_interrupted_canary(self) -> None:
        # A candidate must be judged by one complete lap.  A process exit or
        # emergency stop cannot count as evidence, so restore the last stable
        # compensator before accepting another candidate.
        if self.canary_active:
            self._remember_interrupted_trial(
                "process restarted before the real-hardware canary lap completed")
            self.active = deepcopy(self.previous)
            self.active_kind = self.previous_kind
            self.active_source = self.previous_source
            self.generation = max(0, self.generation - 1)
            self.canary_active = False
            self.last_design = dict(self.last_design)
            self.last_design["recovery"] = "interrupted canary rolled back"
            self._save()

    @classmethod
    def _parameter_changes(cls, before: Dict[str, object],
                           after: Dict[str, object]) -> list:
        """Return a compact, ranked summary of one candidate's parameter edits."""
        changes = []
        axes = ("X", "Y", "phi")
        channels = ("V0", "V1", "V2", "V3", "V4", "V5")
        for key in cls.PARAMETER_KEYS:
            old = np.asarray(before.get(key), dtype=float)
            new = np.asarray(after.get(key), dtype=float)
            if old.shape != new.shape:
                continue
            for index in np.argwhere(np.abs(new-old) > 1.0e-12):
                idx = tuple(int(value) for value in index)
                if len(idx) == 1:
                    label = f"{key}[{channels[idx[0]]}]"
                else:
                    label = f"{key}[{channels[idx[0]]},{axes[idx[1]]}]"
                old_value = float(old[idx])
                new_value = float(new[idx])
                changes.append({
                    "parameter": label,
                    "from": round(old_value, 6),
                    "to": round(new_value, 6),
                    "delta": round(new_value-old_value, 6),
                })

        old_knots = list(before.get("feedforward_knots") or [])
        new_knots = list(after.get("feedforward_knots") or [])
        times = sorted({float(item["time_s"]) for item in old_knots} |
                       {float(item["time_s"]) for item in new_knots})
        for sample_time in times:
            old_value = cls._feedforward_at(old_knots, sample_time)
            new_value = cls._feedforward_at(new_knots, sample_time)
            for channel in np.flatnonzero(np.abs(new_value-old_value) > 1.0e-12):
                changes.append({
                    "parameter": (f"feedforward_mv@{sample_time:.3f}s"
                                  f"[{channels[int(channel)]}]"),
                    "from": round(float(old_value[channel]), 6),
                    "to": round(float(new_value[channel]), 6),
                    "delta": round(float(new_value[channel]-old_value[channel]), 6),
                })
        changes.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
        omitted = max(0, len(changes)-cls.DESIGN_MEMORY_CHANGE_MAX)
        compact = changes[:cls.DESIGN_MEMORY_CHANGE_MAX]
        if omitted:
            compact.append({"omitted_smaller_changes": omitted})
        return compact

    @classmethod
    def _candidate_change_profile(cls, before: Dict[str, object],
                                  after: Dict[str, object]) -> Dict[str, float]:
        """Return maximum absolute edits; no per-round upper bound is applied."""
        profile = {}
        for key in cls.PARAMETER_KEYS:
            old = np.asarray(before.get(key), dtype=float)
            new = np.asarray(after.get(key), dtype=float)
            profile[key] = (float(np.max(np.abs(new-old)))
                            if old.shape == new.shape else 0.0)
        old_knots = list(before.get("feedforward_knots") or [])
        new_knots = list(after.get("feedforward_knots") or [])
        times = sorted({float(item["time_s"]) for item in old_knots} |
                       {float(item["time_s"]) for item in new_knots})
        profile["feedforward_mv"] = (
            max(float(np.max(np.abs(
                cls._feedforward_at(new_knots, sample_time)-
                cls._feedforward_at(old_knots, sample_time))))
                for sample_time in times) if times else 0.0)
        return profile

    @staticmethod
    def _meets_minimum_change(profile: Dict[str, float],
                              minimums: Dict[str, float]) -> bool:
        """True when at least one supported parameter change is large enough."""
        return any(float(profile.get(key, 0.0)) >= float(minimum)-1.0e-12
                   for key, minimum in minimums.items())

    @classmethod
    def _phase_error_hotspots(cls, records: list) -> list:
        """Summarize the worst repeatable trajectory regions without full traces."""
        if not records:
            return []
        reference = np.asarray([row["reference_deg"] for row in records], dtype=float)
        response = np.asarray([row["response_deg"] for row in records], dtype=float)
        if (reference.ndim != 2 or reference.shape != response.shape or
                reference.shape[1] != NAX or not np.all(np.isfinite(reference)) or
                not np.all(np.isfinite(response))):
            return []
        errors = reference-response
        n_sample = len(records)
        n_bins = min(cls.HOTSPOT_PHASE_BINS, n_sample)
        edges = np.linspace(0, n_sample, n_bins+1).round().astype(int)
        by_axis = {name: [] for name in ("X", "Y", "phi")}
        for phase_bin in range(n_bins):
            start, stop = int(edges[phase_bin]), int(edges[phase_bin+1])
            if stop <= start:
                continue
            segment = errors[start:stop]
            reference_mean = np.mean(reference[start:stop], axis=0)
            response_mean = np.mean(response[start:stop], axis=0)
            for axis_index, axis_name in enumerate(("X", "Y", "phi")):
                values = segment[:, axis_index]
                rmse = float(np.sqrt(np.mean(values*values)))
                if rmse < cls.HOTSPOT_MIN_RMSE_DEG:
                    continue
                by_axis[axis_name].append({
                    "axis": axis_name,
                    "phase_bin": int(phase_bin),
                    "phase_fraction": [round(start/n_sample, 4),
                                       round(stop/n_sample, 4)],
                    "time_s": [round(float(records[start]["t"]), 4),
                               round(float(records[stop-1]["t"]), 4)],
                    "reference_pose_mean_deg": np.round(reference_mean, 4).tolist(),
                    "response_pose_mean_deg": np.round(response_mean, 4).tolist(),
                    "rmse_deg": round(rmse, 6),
                    "mean_signed_error_deg": round(float(np.mean(values)), 6),
                    "max_abs_error_deg": round(float(np.max(np.abs(values))), 6),
                })
        hotspots = []
        for axis_name in ("X", "Y", "phi"):
            ranked = sorted(by_axis[axis_name],
                            key=lambda item: item["rmse_deg"], reverse=True)
            hotspots.extend(ranked[:2])
        return hotspots

    def _recurrent_phase_error_hotspots(self) -> list:
        """Find the same axis/phase hotspot recurring in at least two valid laps."""
        valid_memories = [item for item in self.design_memory
                          if item.get("valid_lap")]
        groups = {}
        for memory in valid_memories:
            for hotspot in memory.get("phase_error_hotspots") or []:
                try:
                    key = (str(hotspot["axis"]), int(hotspot["phase_bin"]))
                    groups.setdefault(key, []).append(hotspot)
                except (KeyError, TypeError, ValueError):
                    continue
        recurrent = []
        for (axis, phase_bin), items in groups.items():
            if len(items) < 2:
                continue
            rmse_values = [float(item["rmse_deg"]) for item in items]
            mean_rmse = float(np.mean(rmse_values))
            if mean_rmse < self.RECURRENT_HOTSPOT_MIN_RMSE_DEG:
                continue
            signed = [float(item.get("mean_signed_error_deg", 0.0))
                      for item in items]
            nonzero_signs = {int(np.sign(value)) for value in signed
                             if abs(value) > 1.0e-9}
            latest = items[-1]
            recurrent.append({
                "priority": "repeated_large_error",
                "axis": axis,
                "phase_bin": phase_bin,
                "occurrences_in_recent_valid_laps": len(items),
                "valid_laps_considered": len(valid_memories),
                "mean_rmse_deg": round(mean_rmse, 6),
                "latest_rmse_deg": round(rmse_values[-1], 6),
                "mean_signed_error_deg": round(float(np.mean(signed)), 6),
                "signed_error_consistency": (
                    "consistent" if len(nonzero_signs) <= 1 else "mixed"),
                "latest_time_s": latest.get("time_s"),
                "latest_phase_fraction": latest.get("phase_fraction"),
                "latest_reference_pose_mean_deg": latest.get(
                    "reference_pose_mean_deg"),
            })
        recurrent.sort(
            key=lambda item: (item["occurrences_in_recent_valid_laps"],
                              item["mean_rmse_deg"]), reverse=True)
        return recurrent[:6]

    def _append_design_memory(self, entry: Dict[str, object]) -> None:
        self.design_memory.append(entry)
        self.design_memory = self.design_memory[-self.DESIGN_MEMORY_MAX:]

    def _remember_interrupted_trial(self, reason: str) -> None:
        """Remember why a pending candidate was rejected without grading it."""
        design = dict(self.last_design or {})
        self._append_design_memory({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "outcome": "interrupted_rolled_back",
            "generation_tested": int(self.generation),
            "generation_after": max(0, int(self.generation)-1),
            "hypothesis": str(design.get("rationale", ""))[:1000],
            "design_strategy": str(design.get("design_strategy", ""))[:1000],
            "expected_effect": str(design.get("expected_effect", ""))[:500],
            "designer_failure_analysis": str(
                design.get("failure_analysis", ""))[:1000],
            "parameter_changes": self._parameter_changes(self.previous, self.active),
            "interruption_reason": str(reason)[:1000],
        })

    @classmethod
    def _validate_parameters(cls, raw: object,
                             base: Optional[Dict[str, object]]) -> Dict[str, object]:
        if not isinstance(raw, dict):
            raise ValueError("compensator parameters must be an object")
        answer = {}
        for key in cls.PARAMETER_KEYS:
            shape = (NCH,) if key == "bias_mv" else (NCH, NAX)
            arr = np.asarray(raw.get(key), dtype=float)
            if arr.shape != shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"{key} must have finite shape {shape}")
            answer[key] = arr.tolist()
        knots = raw.get("feedforward_knots", [])
        if not isinstance(knots, list) or len(knots) > cls.FEEDFORWARD_MAX_KNOTS:
            raise ValueError(
                f"feedforward_knots must be a list with at most {cls.FEEDFORWARD_MAX_KNOTS} entries")
        clean_knots = []
        last_t = -float("inf")
        for item in knots:
            if not isinstance(item, dict):
                raise ValueError("each feedforward knot must be an object")
            t = float(item.get("time_s", float("nan")))
            values = np.asarray(item.get("delta_u_mv"), dtype=float)
            if (not np.isfinite(t) or t < 0.0 or t > 3600.0 or
                    values.shape != (NCH,) or not np.all(np.isfinite(values))):
                raise ValueError("invalid feedforward knot time or six-output value")
            if t-last_t < cls.FEEDFORWARD_MIN_SPACING_S-1.0e-9:
                raise ValueError("feedforward knot times must be strictly increasing by >=0.05s")
            clean_knots.append({"time_s": t, "delta_u_mv": values.tolist()})
            last_t = t
        answer["feedforward_knots"] = clean_knots
        return answer

    @staticmethod
    def _feedforward_at(knots: list, time_s: float) -> np.ndarray:
        """Interpolate a local phase curve; outside its knot interval it is zero."""
        if not knots:
            return np.zeros(NCH, dtype=float)
        times = np.asarray([float(v["time_s"]) for v in knots], dtype=float)
        values = np.asarray([v["delta_u_mv"] for v in knots], dtype=float)
        t = float(time_s)
        return np.asarray([
            np.interp(t, times, values[:, ch], left=0.0, right=0.0)
            for ch in range(NCH)
        ], dtype=float)

    def _reset_runtime(self) -> None:
        self.integral = np.zeros(NAX, dtype=float)
        self.previous_comp = np.zeros(NCH, dtype=float)
        self.previous_total = None
        self.previous_total_time = None
        self.last_guarded_lock_cmd = np.zeros(NCH, dtype=int)
        self.pair_lock_owner = np.full(len(ANTAGONIST_PAIRS), -1, dtype=int)
        self.pair_releasing = np.zeros(len(ANTAGONIST_PAIRS), dtype=bool)
        self.records = []
        self._program = None
        if self.active_kind == "python_class":
            self.active_source, self._program = _validate_compensator_source(
                self.active_source)
            self._program.reset(np.zeros(NAX, dtype=float))

    def _candidate_digest(self, blob: Dict[str, object]) -> str:
        raw = json.dumps(blob, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def consume_candidate(self) -> Dict[str, object]:
        if not os.path.isfile(self.candidate_path):
            return {"loaded": False, "reason": "no candidate file"}
        try:
            with open(self.candidate_path, encoding="utf-8") as stream:
                blob = json.load(stream)
            digest = self._candidate_digest(blob)
            if digest == self.last_candidate_digest:
                return {"loaded": False, "reason": "candidate already consumed"}
            if int(blob.get("schema", -1)) != self.SCHEMA:
                raise ValueError("candidate schema mismatch")
            if str(blob.get("task_hash", "")) != self.task_hash:
                raise ValueError("candidate belongs to a different trajectory")
            if int(blob.get("base_generation", -1)) != self.generation:
                raise ValueError("candidate base_generation is stale")
            if self.canary_active:
                raise ValueError("a candidate is already under canary test")
            if self.last_valid_mse is None:
                raise ValueError("run one valid baseline lap before compensation")
            source_code = blob.get("source_code")
            kp_matrix = None
            kd_matrix = None
            ki_matrix = None
            integral_limit_deg_s = None
            output_limit_mv = None
            output_slew_mv_per_s = None
            if source_code is not None:
                trace = []
                if os.path.isfile(self.request_path):
                    try:
                        with open(self.request_path, encoding="utf-8") as stream:
                            trace = list((json.load(stream).get("full_trace") or []))
                    except Exception:
                        trace = []
                source_code = _probe_compensator_source(source_code, trace)
                _, contract_instance = _validate_compensator_source(
                    source_code, require_design_contract=True)
                kp_matrix = _optional_compensator_array(
                    contract_instance, "kp_matrix", (NCH, NAX))
                kd_matrix = _optional_compensator_array(
                    contract_instance, "kd_matrix", (NCH, NAX))
                ki_matrix = _optional_compensator_array(
                    contract_instance, "ki_matrix", (NCH, NAX))
                integral_limit_deg_s = _optional_compensator_array(
                    contract_instance, "integral_limit_deg_s", (NAX,))
                output_limit_mv = _optional_compensator_array(
                    contract_instance, "output_limit_mv", (NCH,))
                output_slew_mv_per_s = _optional_compensator_array(
                    contract_instance, "output_slew_mv_per_s", (NCH,))
                if (self.active_kind == "python_class" and
                        source_code.strip() == self.active_source.strip()):
                    raise ValueError("candidate implementation is unchanged")
                params = _zero_parameters()
                next_kind = "python_class"
            else:
                params = self._validate_parameters(blob.get("parameters"), self.active)
                change_profile = self._candidate_change_profile(self.active, params)
                if not self._meets_minimum_change(
                        change_profile, self.MIN_MEANINGFUL_CHANGE):
                    raise ValueError(
                        "candidate is a no-op or too small to be a meaningful hardware "
                        f"experiment (changes={change_profile}; minimums="
                        f"{self.MIN_MEANINGFUL_CHANGE})")
                source_code = DEFAULT_COMPENSATOR_SOURCE
                next_kind = "legacy_parameters"
            self.previous = deepcopy(self.active)
            self.previous_kind = self.active_kind
            self.previous_source = self.active_source
            self.active = params
            self.active_kind = next_kind
            self.active_source = source_code
            self.generation += 1
            self.canary_active = True
            self.baseline_mse = float(self.last_valid_mse)
            self.last_candidate_digest = digest
            self.last_design = {
                "generation": self.generation,
                "rationale": str(blob.get("rationale", ""))[:2000],
                "design_strategy": str(blob.get("design_strategy", ""))[:2000],
                "expected_effect": str(blob.get("expected_effect", ""))[:1000],
                "failure_analysis": str(blob.get("failure_analysis", ""))[:2000],
                "created_by": str(blob.get("created_by", "Codex"))[:100],
                "implementation_kind": self.active_kind,
                "source_sha256": hashlib.sha256(
                    self.active_source.encode("utf-8")).hexdigest(),
                "kp_matrix": kp_matrix,
                "kd_matrix": kd_matrix,
                "ki_matrix": ki_matrix,
                "integral_limit_deg_s": integral_limit_deg_s,
                "output_limit_mv": output_limit_mv,
                "output_slew_mv_per_s": output_slew_mv_per_s,
            }
            self._save()
            self._reset_runtime()
            return {"loaded": True, "generation": self.generation,
                    "baseline_mse_deg2": self.baseline_mse}
        except Exception as exc:
            return {"loaded": False, "reason": str(exc), "error": True}

    def begin_round(self) -> Dict[str, object]:
        self._reset_runtime()
        candidate = self.consume_candidate()
        return {"generation": self.generation,
                "canary_active": self.canary_active,
                "candidate": candidate}

    def install_replay_controller(self, controller: Dict[str, object],
                                  generation: int = 0) -> Dict[str, object]:
        """Install one previously audited lap controller without touching normal state.

        Replay users construct this object with separate temporary state paths.  The
        same AST/source and parameter validation used for live LLM candidates is
        applied here, so loading a saved process cannot bypass controller checks.
        """
        if not isinstance(controller, dict):
            raise ValueError("replay controller must be an object")
        kind = str(controller.get("kind", "")).strip()
        if kind == "python_class":
            source = str(controller.get("source_code") or "")
            source, _ = _validate_compensator_source(source)
            self.active = _zero_parameters()
            self.active_kind = "python_class"
            self.active_source = source
        elif kind == "legacy_parameters":
            parameters = controller.get("legacy_parameters")
            self.active = self._validate_parameters(parameters, None)
            self.active_kind = "legacy_parameters"
            self.active_source = DEFAULT_COMPENSATOR_SOURCE
        else:
            raise ValueError("replay controller kind must be python_class or legacy_parameters")
        self.previous = deepcopy(self.active)
        self.previous_kind = self.active_kind
        self.previous_source = self.active_source
        self.generation = max(0, int(generation))
        self.canary_active = False
        self.last_design = {
            "created_by": "saved MPC+LLM process replay",
            "implementation_kind": self.active_kind,
        }
        self._reset_runtime()
        return {
            "generation": int(self.generation),
            "implementation_kind": self.active_kind,
        }

    @staticmethod
    def _antagonist_project(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        for left, right in ANTAGONIST_PAIRS:
            if out[left] > 0.0 and out[right] > 0.0:
                if out[left] >= out[right]:
                    out[right] = 0.0
                else:
                    out[left] = 0.0
        return out

    def _guard_final_total(self, desired_total: np.ndarray,
                           lock_states: list, time_s: float) -> np.ndarray:
        """Limit LLM+MPC final-voltage slew and enforce break-before-make."""
        desired = self._antagonist_project(
            np.clip(np.asarray(desired_total, dtype=float), 0.0, self.u_max))
        if (self.final_fall_slew_mv_s is None and
                self.final_phi_rise_slew_mv_s is None):
            self.previous_total = desired.copy()
            self.previous_total_time = float(time_s)
            return desired

        now = float(time_s)
        if self.previous_total is None:
            previous = np.zeros(NCH, dtype=float)
            elapsed = self.dt
        else:
            previous = np.asarray(self.previous_total, dtype=float).copy()
            # A command cannot be physically retained once telemetry confirms
            # that its regulator is no longer locked.
            locked = np.asarray([
                index < len(lock_states) and lock_states[index] == "L"
                for index in range(NCH)
            ], dtype=bool)
            previous = np.where(locked, previous, 0.0)
            elapsed = (now-float(self.previous_total_time)
                       if self.previous_total_time is not None else self.dt)
            if not np.isfinite(elapsed) or elapsed <= 0.0:
                elapsed = self.dt
            # A delayed HTTP tick must not create one correspondingly large
            # voltage step; the control period remains the hard step basis.
            elapsed = min(elapsed, self.dt)
        target = desired.copy()
        # Never raise the opposite side while the old side is being ramped
        # down.  The lock guard below then waits for real unlock.
        for left, right in ANTAGONIST_PAIRS:
            if previous[left] > 0.5 and target[right] > 0.5:
                target[right] = 0.0
            elif previous[right] > 0.5 and target[left] > 0.5:
                target[left] = 0.0
        total = target.copy()
        if self.final_fall_slew_mv_s is not None:
            maximum_fall = self.final_fall_slew_mv_s*elapsed
            floor = np.maximum(0.0, previous-maximum_fall)
            total = np.where(target < previous,
                             np.maximum(target, floor), target)
        if self.final_phi_rise_slew_mv_s is not None:
            maximum_rise = self.final_phi_rise_slew_mv_s*elapsed
            # V2/V3 are the phi+/phi- antagonist pair.  Limit their final
            # MPC+LLM voltage rise, including the first nonzero control tick.
            for channel in (2, 3):
                if total[channel] > previous[channel]:
                    total[channel] = min(
                        total[channel], previous[channel]+maximum_rise)
        total = self._antagonist_project(total)
        self.previous_total = total.copy()
        self.previous_total_time = now
        return total

    def _guard_lock_handoff(self, desired_lock_cmd: Iterable[int],
                            lock_states: list, pressure: np.ndarray,
                            total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Keep an antagonist switch break-before-make at the final output."""
        raw_cmd = list(desired_lock_cmd)
        cmd = np.asarray([
            1 if index < len(raw_cmd) and int(raw_cmd[index]) else 0
            for index in range(NCH)
        ], dtype=int)
        guarded_total = np.asarray(total, dtype=float).copy()
        for pair_index, (left, right) in enumerate(ANTAGONIST_PAIRS):
            pair = (left, right)
            busy = [index for index in pair
                    if index < len(lock_states) and lock_states[index] in ("L", "w")]
            if len(busy) > 1:
                # Recover deterministically from impossible dual ownership.
                owner = max(busy, key=lambda index: (
                    lock_states[index] == "L", guarded_total[index]))
                other = right if owner == left else left
                cmd[other] = 0
                guarded_total[other] = 0.0
                busy = [owner]
            if busy:
                owner = busy[0]
                self.pair_lock_owner[pair_index] = owner
                other = right if owner == left else left
                cmd[other] = 0
                guarded_total[other] = 0.0
                if lock_states[owner] == "w":
                    # Latch the first 100 ms lock transaction.  A new MPC
                    # winner cannot cancel it and start the antagonist.
                    cmd[owner] = 1
                elif guarded_total[owner] > 0.5 or raw_cmd[owner] != 0:
                    # Retain the locked regulator throughout its voltage ramp.
                    cmd[owner] = 1
                    self.pair_releasing[pair_index] = False
                else:
                    # At zero request unlock.  The antagonist stays blocked
                    # until telemetry reports OPEN (the firmware's 30 kPa rule).
                    cmd[owner] = 0
                    self.pair_releasing[pair_index] = True
                continue

            if self.pair_releasing[pair_index]:
                # No L/w owner remains: measured unlock has completed.
                self.pair_releasing[pair_index] = False
                self.pair_lock_owner[pair_index] = -1
            elif self.pair_lock_owner[pair_index] in pair:
                # The first lock request may precede w telemetry by one or more
                # HTTP samples.  Keep it latched instead of following a new
                # winner on every 50 ms control tick.
                owner = int(self.pair_lock_owner[pair_index])
                other = right if owner == left else left
                cmd[owner] = 1
                cmd[other] = 0
                guarded_total[other] = 0.0
                continue

            if cmd[left] and cmd[right]:
                owner = left if guarded_total[left] >= guarded_total[right] else right
                other = right if owner == left else left
                cmd[other] = 0
                guarded_total[other] = 0.0
            elif cmd[left]:
                owner = left
            elif cmd[right]:
                owner = right
            else:
                owner = -1
            self.pair_lock_owner[pair_index] = owner

        return cmd, self._antagonist_project(guarded_total)

    def apply(self, reference_task_deg: np.ndarray,
              measurement_task_deg: np.ndarray,
              reference_velocity_deg_s: np.ndarray,
              velocity_deg_s: np.ndarray,
              base_u: np.ndarray, pressure: np.ndarray,
              lock_states: Iterable[str], time_s: float,
              desired_lock_cmd: Optional[Iterable[int]] = None,
              record_only: bool = False
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the locally evaluated compensator and record one control tick."""
        r = np.asarray(reference_task_deg, dtype=float)
        q = np.asarray(measurement_task_deg, dtype=float)
        rv = np.asarray(reference_velocity_deg_s, dtype=float)
        qv = np.asarray(velocity_deg_s, dtype=float)
        base = np.clip(np.asarray(base_u, dtype=float), 0.0, self.u_max)
        pressure = np.asarray(pressure, dtype=float)
        locks = list(lock_states)
        if any(a.shape != (NAX,) for a in (r, q, rv, qv)) or base.shape != (NCH,):
            raise ValueError("invalid compensator tick shape")

        e = r-q
        ev = rv-qv
        if record_only:
            requested = np.zeros(NCH, dtype=float)
            phase_feedforward = np.zeros(NCH, dtype=float)
        elif self.active_kind == "python_class":
            if self._program is None:
                self.active_source, self._program = _validate_compensator_source(
                    self.active_source)
            if not self.records:
                self._program.reset(q.copy())
            requested = _run_compensator(
                self._program, base, q, qv, r, rv,
                np.nan_to_num(pressure, nan=0.0), locks, time_s)
            phase_feedforward = np.zeros(NCH, dtype=float)
        else:
            self.integral = self.integral + e*self.dt
            kp = np.asarray(self.active["kp_mv_per_deg"], dtype=float)
            kd = np.asarray(self.active["kd_mv_per_deg_s"], dtype=float)
            ki = np.asarray(self.active["ki_mv_per_deg_s"], dtype=float)
            bias = np.asarray(self.active["bias_mv"], dtype=float)
            phase_feedforward = self._feedforward_at(
                list(self.active.get("feedforward_knots") or []), time_s)
            requested = kp@e + kd@ev + ki@self.integral + bias + phase_feedforward
        # The compensator may modify only channels already confirmed LOCKED.
        # It therefore cannot bypass the MPC's explicit lock planning action.
        allowed = np.asarray([i < len(locks) and locks[i] == "L"
                              for i in range(NCH)], dtype=bool)
        requested = np.where(allowed, requested, 0.0)
        total = base.copy() if record_only else self._guard_final_total(
            self._antagonist_project(np.clip(base+requested, 0.0, self.u_max)),
            locks, time_s)
        if desired_lock_cmd is not None and not record_only:
            self.last_guarded_lock_cmd, total = self._guard_lock_handoff(
                desired_lock_cmd, locks, pressure, total)
            self.previous_total = total.copy()
        applied = total-base
        self.previous_comp = applied.copy()

        self.records.append({
            "t": float(time_s),
            "reference_deg": r.tolist(),
            "response_deg": q.tolist(),
            "reference_velocity_deg_s": rv.tolist(),
            "velocity_deg_s": qv.tolist(),
            "error_deg": e.tolist(),
            "base_u_mv": base.tolist(),
            "compensation_mv": applied.tolist(),
            "phase_feedforward_mv": phase_feedforward.tolist(),
            "total_u_mv": total.tolist(),
            "pressure_kpa": ([float(value) if np.isfinite(value) else None
                              for value in pressure]
                             if pressure.shape == (NCH,) else [None]*NCH),
            "locks": locks[:NCH],
        })
        return total, applied

    @staticmethod
    def _lag_seconds(reference: np.ndarray, response: np.ndarray, dt: float) -> list:
        answer = []
        max_shift = min(20, max(1, len(reference)//5))
        for axis in range(NAX):
            best = (float("inf"), 0)
            for shift in range(-max_shift, max_shift+1):
                if shift < 0:
                    a, b = reference[-shift:, axis], response[:shift, axis]
                elif shift > 0:
                    a, b = reference[:-shift, axis], response[shift:, axis]
                else:
                    a, b = reference[:, axis], response[:, axis]
                if len(a) < 5:
                    continue
                mse = float(np.mean((a-b)**2))
                if mse < best[0]:
                    best = (mse, shift)
            answer.append(round(best[1]*dt, 3))
        return answer

    def _build_request(self, score: float, audit: Dict[str, object],
                       valid: bool, invalid_reasons: list,
                       tested_parameters: Dict[str, object],
                       tested_compensator: Dict[str, object]) -> Dict[str, object]:
        r = np.asarray([v["reference_deg"] for v in self.records], dtype=float)
        q = np.asarray([v["response_deg"] for v in self.records], dtype=float)
        rv = np.asarray([v["reference_velocity_deg_s"] for v in self.records], dtype=float)
        qv = np.asarray([v["velocity_deg_s"] for v in self.records], dtype=float)
        base = np.asarray([v["base_u_mv"] for v in self.records], dtype=float)
        comp = np.asarray([v["compensation_mv"] for v in self.records], dtype=float)
        total = np.asarray([v["total_u_mv"] for v in self.records], dtype=float)
        pressure = np.asarray([[np.nan if x is None else float(x) for x in v["pressure_kpa"]]
                               for v in self.records], dtype=float)
        e = r-q
        tail = max(1, len(e)//10)
        axis = {
            name: {
                "rmse_deg": round(float(np.sqrt(np.mean(e[:, i]**2))), 4),
                "mae_deg": round(float(np.mean(np.abs(e[:, i]))), 4),
                "max_abs_error_deg": round(float(np.max(np.abs(e[:, i]))), 4),
                "final_mean_error_deg": round(float(np.mean(e[-tail:, i])), 4),
                "velocity_rmse_deg_s": round(float(np.sqrt(np.mean((rv[:, i]-qv[:, i])**2))), 4),
            } for i, name in enumerate(("X", "Y", "phi"))
        }
        lag_seconds = self._lag_seconds(r, q, self.dt)
        channel_pairs = {
            "X": [(4, "V4=9(x-)"), (5, "V5=10(x+)")],
            "Y": [(0, "V0=15(y+)"), (1, "V1=48(y-)")],
            "phi": [(2, "V2=23(phi+)"), (3, "V3=67(phi-)")],
        }
        slow_response_priority = []
        for axis_index, axis_name in enumerate(("X", "Y", "phi")):
            lag = float(lag_seconds[axis_index])
            if lag < self.SLOW_RESPONSE_LAG_S:
                continue
            drive_context = []
            for channel_index, channel_label in channel_pairs[axis_name]:
                base_peak = float(np.max(base[:, channel_index]))
                headroom = max(0.0, self.u_max-base_peak)
                finite_pressure = pressure[:, channel_index][
                    np.isfinite(pressure[:, channel_index])]
                pressure_peak = (float(np.max(finite_pressure))
                                 if len(finite_pressure) else None)
                suggested = None
                if base_peak >= 50.0 and headroom >= 1.0:
                    low = min(headroom, max(150.0, 0.35*headroom))
                    high = min(headroom, max(low, min(600.0, 0.70*headroom)))
                    suggested = [round(low, 1), round(high, 1)]
                drive_context.append({
                    "channel": channel_label,
                    "base_peak_mv": round(base_peak, 2),
                    "remaining_command_headroom_mv": round(headroom, 2),
                    "measured_pressure_peak_kpa": (
                        round(pressure_peak, 2) if pressure_peak is not None else None),
                    "suggested_additional_positive_command_mv": suggested,
                })
            slow_response_priority.append({
                "priority": ("severe_slow_response" if
                             lag >= self.SEVERE_RESPONSE_LAG_S else
                             "slow_response"),
                "axis": axis_name,
                "estimated_response_lag_s": round(lag, 4),
                "axis_rmse_deg": axis[axis_name]["rmse_deg"],
                "velocity_rmse_deg_s": axis[axis_name]["velocity_rmse_deg_s"],
                "nominal_agonist_pair": [label for _, label in
                                          channel_pairs[axis_name]],
                "channel_drive_context": drive_context,
            })
        slow_response_priority.sort(
            key=lambda item: item["estimated_response_lag_s"], reverse=True)
        n_sample = min(32, len(self.records))
        indexes = np.unique(np.linspace(0, len(self.records)-1, n_sample).round().astype(int))
        samples = [self.records[int(i)] for i in indexes]
        request_id = hashlib.sha256(
            (self.task_hash+str(time.time_ns())+str(score)).encode("utf-8")).hexdigest()[:16]
        return {
            "schema": self.SCHEMA,
            "request_id": request_id,
            "paper_framework": "arXiv:2507.20509 LLM-guided adaptive compensator",
            "task_hash": self.task_hash,
            "sample_dt_s": self.dt,
            "samples_in_lap": len(self.records),
            "lap_valid_for_design": bool(valid),
            "invalid_reasons": invalid_reasons,
            "audit": audit,
            "tracking_mse_deg2": round(float(score), 6),
            "tracking_rmse_deg": round(math.sqrt(max(0.0, float(score))), 6),
            "axis_metrics": axis,
            "estimated_response_lag_s": lag_seconds,
            "slow_response_priority": slow_response_priority,
            "initial_error_deg": np.round(e[0], 4).tolist(),
            "initial_pressure_kpa": [
                (round(float(value), 2) if np.isfinite(value) else None)
                for value in pressure[0]
            ],
            "pressure_peak_kpa": [
                (round(float(np.max(column[np.isfinite(column)])), 2)
                 if np.any(np.isfinite(column)) else None)
                for column in pressure.T
            ],
            "base_u_mean_mv": np.round(np.mean(base, axis=0), 2).tolist(),
            "base_u_peak_mv": np.round(np.max(base, axis=0), 2).tolist(),
            "compensation_rms_mv": np.round(np.sqrt(np.mean(comp*comp, axis=0)), 2).tolist(),
            "compensation_peak_abs_mv": np.round(np.max(np.abs(comp), axis=0), 2).tolist(),
            "total_u_saturation_fraction": round(float(np.mean(total >= self.u_max-1.0)), 6),
            "channel_order": ["V0=15(y+)", "V1=48(y-)", "V2=23(phi+)",
                              "V3=67(phi-)", "V4=9(x-)", "V5=10(x+)"],
            "axis_order": ["GUI X", "GUI Y", "GUI phi"],
            "hard_constraints": {
                "total_u_mv": [0.0, self.u_max],
                "compensation_abs_max_mv": None,
                "compensation_slew_mv_s": None,
                "final_total_fall_slew_mv_s": self.final_fall_slew_mv_s,
                "final_phi_rise_slew_mv_s": self.final_phi_rise_slew_mv_s,
                "antagonist_handoff": "break-before-make; old side ramps to zero and unlocks before new side locks",
                "unlock_pressure_kpa": self.unlock_pressure_kpa,
                "integral_state_abs_max_deg_s": None,
                "antagonist_pairs": [[0, 1], [2, 3], [4, 5]],
                "slow_response_lag_s": self.SLOW_RESPONSE_LAG_S,
                "severe_response_lag_s": self.SEVERE_RESPONSE_LAG_S,
            },
            # Keep the rejected candidate visible.  A rollback changes only
            # what the next hardware lap executes; its failed response remains
            # evidence for the next Codex design.
            "tested_parameters": tested_parameters,
            "tested_compensator": tested_compensator,
            "active_generation": self.generation,
            "stable_parameters_after_audit": self.active,
            "stable_compensator_after_audit": {
                "kind": self.active_kind,
                "source_code": (self.active_source
                                if self.active_kind == "python_class" else None),
                "legacy_parameters": (self.active
                                      if self.active_kind == "legacy_parameters" else None),
            },
            "last_design": self.last_design,
            "lap_history": self.history[-12:],
            "design_memory_last_5": self.design_memory[-self.DESIGN_MEMORY_MAX:],
            "recurrent_phase_error_hotspots": self._recurrent_phase_error_hotspots(),
            "phase_samples": samples,
            "full_trace": self.records,
        }

    def _build_trial_memory(self, score: float, valid: bool,
                            invalid_reasons: list,
                            result: Dict[str, object],
                            tested_parameters: Dict[str, object],
                            comparison_parameters: Dict[str, object],
                            design: Dict[str, object],
                            prior_valid_mse: Optional[float]) -> Dict[str, object]:
        """Create the concise evidence record that the next LLM must inspect."""
        was_canary = bool(result.get("canary_checked"))
        accepted = result.get("accepted")
        if was_canary and accepted:
            outcome = "accepted"
        elif was_canary and result.get("rolled_back"):
            outcome = "invalid_rolled_back" if not valid else "rolled_back"
        elif valid:
            outcome = "stable_baseline"
        else:
            outcome = "invalid_lap"

        current_axis = None
        if self.records:
            errors = np.asarray([row["error_deg"] for row in self.records], dtype=float)
            if errors.ndim == 2 and errors.shape[1] == NAX and np.all(np.isfinite(errors)):
                values = np.sqrt(np.mean(errors*errors, axis=0))
                current_axis = {name: round(float(values[index]), 6)
                                for index, name in enumerate(("X", "Y", "phi"))}

        previous_axis = None
        for memory in reversed(self.design_memory):
            if memory.get("outcome") in ("accepted", "stable_baseline"):
                candidate = memory.get("axis_rmse_deg")
                if isinstance(candidate, dict):
                    previous_axis = candidate
                    break
        axis_change = None
        if current_axis is not None and previous_axis is not None:
            try:
                axis_change = {
                    name: round(current_axis[name]-float(previous_axis[name]), 6)
                    for name in ("X", "Y", "phi")
                }
            except (KeyError, TypeError, ValueError):
                axis_change = None

        comparison_mse = None
        if was_canary and np.isfinite(self.baseline_mse):
            comparison_mse = float(self.baseline_mse)
        elif prior_valid_mse is not None and np.isfinite(prior_valid_mse):
            comparison_mse = float(prior_valid_mse)
        score_value = float(score) if np.isfinite(score) else None
        mse_delta = (float(score)-comparison_mse
                     if score_value is not None and comparison_mse is not None else None)
        mse_ratio = (float(score)/max(comparison_mse, 1.0e-12)
                     if score_value is not None and comparison_mse is not None else None)

        return {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "outcome": outcome,
            "generation_tested": int(result.get("generation_before", self.generation)),
            "generation_after": int(result.get("generation_after", self.generation)),
            "valid_lap": bool(valid),
            "hypothesis": str(design.get("rationale", ""))[:1000],
            "design_strategy": str(design.get("design_strategy", ""))[:1000],
            "expected_effect": str(design.get("expected_effect", ""))[:500],
            "designer_failure_analysis": str(
                design.get("failure_analysis", ""))[:1000],
            "parameter_changes": (
                [] if result.get("tested_implementation_kind") == "python_class"
                else self._parameter_changes(
                    comparison_parameters, tested_parameters)),
            "implementation_kind": result.get("tested_implementation_kind"),
            "source_sha256": result.get("tested_source_sha256"),
            "measured_mse_deg2": (round(score_value, 6)
                                  if score_value is not None else None),
            "comparison_mse_deg2": (round(comparison_mse, 6)
                                     if comparison_mse is not None else None),
            "mse_delta_deg2": (round(mse_delta, 6) if mse_delta is not None else None),
            "mse_ratio": (round(mse_ratio, 6) if mse_ratio is not None else None),
            "axis_rmse_deg": current_axis,
            "axis_rmse_change_vs_previous_stable_deg": axis_change,
            "phase_error_hotspots": self._phase_error_hotspots(self.records),
            "acceptance_limit_mse_deg2": (
                round(self.accept_ratio*self.baseline_mse, 6)
                if was_canary and np.isfinite(self.baseline_mse) else None),
            "audit_invalid_reasons": list(invalid_reasons),
        }

    def finish_round(self, tracking_mse_deg2: float) -> Dict[str, object]:
        supplied_score = float(tracking_mse_deg2)
        # The rest of the controller internally uses model Euler q.  Codex is
        # graded only in the user's actual GUI task coordinates [X,Y,phi].
        if self.records:
            task_error = np.asarray(
                [row["error_deg"] for row in self.records], dtype=float)
            score = float(np.mean(np.sum(task_error*task_error, axis=1)))
        else:
            score = supplied_score
        tested_parameters = deepcopy(self.active)
        tested_compensator = {
            "kind": self.active_kind,
            "source_code": (self.active_source
                            if self.active_kind == "python_class" else None),
            "legacy_parameters": (deepcopy(self.active)
                                  if self.active_kind == "legacy_parameters" else None),
        }
        comparison_parameters = deepcopy(
            self.previous if self.canary_active else self.active)
        design_snapshot = deepcopy(self.last_design)
        prior_valid_mse = self.last_valid_mse
        result = {
            "tracking_mse_deg2": score,
            "controller_model_coordinate_mse_deg2": supplied_score,
            "generation_before": self.generation,
            "canary_checked": bool(self.canary_active),
            "accepted": None,
            "rolled_back": False,
            "tested_implementation_kind": tested_compensator["kind"],
            "tested_source_sha256": (
                hashlib.sha256(
                    tested_compensator["source_code"].encode("utf-8")
                ).hexdigest()
                if tested_compensator["source_code"] else None),
        }
        invalid_reasons = []
        if len(self.records) != self.length:
            invalid_reasons.append(f"incomplete lap {len(self.records)}/{self.length}")
        if not np.isfinite(score):
            invalid_reasons.append("non-finite tracking MSE")
        if len(self.records) >= 2:
            q = np.asarray([v["response_deg"] for v in self.records], dtype=float)
            max_jump = float(np.max(np.abs(np.diff(q, axis=0))))
            result["max_angle_jump_deg"] = max_jump
            if max_jump > self.angle_jump_limit_deg:
                invalid_reasons.append(
                    f"one-tick angle jump {max_jump:.2f}deg exceeds {self.angle_jump_limit_deg:.2f}deg")
        valid = not invalid_reasons

        if self.canary_active:
            accepted = bool(valid and score <= self.accept_ratio*self.baseline_mse)
            result.update({"accepted": accepted,
                           "baseline_mse_deg2": self.baseline_mse,
                           "ratio": (score/max(self.baseline_mse, 1.0e-12)
                                     if np.isfinite(score) else None)})
            if accepted:
                self.canary_active = False
                self.previous = deepcopy(self.active)
                self.previous_kind = self.active_kind
                self.previous_source = self.active_source
                self.last_valid_mse = score
            else:
                self.active = deepcopy(self.previous)
                self.active_kind = self.previous_kind
                self.active_source = self.previous_source
                self.generation = max(0, self.generation-1)
                self.canary_active = False
                result["rolled_back"] = True
        elif valid:
            self.last_valid_mse = score

        result["valid_for_design"] = valid
        result["invalid_reasons"] = invalid_reasons
        result["generation_after"] = self.generation
        postmortem = self._build_trial_memory(
            score, valid, invalid_reasons, result, tested_parameters,
            comparison_parameters, design_snapshot, prior_valid_mse)
        self._append_design_memory(postmortem)
        result["postmortem"] = postmortem
        if np.isfinite(score):
            self.history.append({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mse_deg2": score,
                "generation": self.generation,
                "valid": valid,
                "accepted": result["accepted"],
                "rolled_back": result["rolled_back"],
            })
        result["completed_lap_index"] = len(self.history)
        if self.records:
            request = self._build_request(
                score, result, valid, invalid_reasons, tested_parameters,
                tested_compensator)
            self._atomic_json(self.request_path, request)
            archive = os.path.splitext(self.request_path)[0] + "_" + request["request_id"] + ".json"
            self._atomic_json(archive, request)
            result["request_path"] = self.request_path
            result["request_id"] = request["request_id"]
        self._save()
        self._reset_runtime()
        return result

    def abort_round(self) -> Dict[str, object]:
        rolled_back = bool(self.canary_active)
        if rolled_back:
            self._remember_interrupted_trial(
                "operator or controller stopped before the real-hardware canary lap completed")
            self.active = deepcopy(self.previous)
            self.active_kind = self.previous_kind
            self.active_source = self.previous_source
            self.generation = max(0, self.generation-1)
            self.canary_active = False
            self._save()
        self._reset_runtime()
        return {"rolled_back": rolled_back, "generation": self.generation}

    def status(self) -> Dict[str, object]:
        if self.active_kind == "python_class":
            nonzero = 1
        else:
            nonzero = 0
            for key in self.PARAMETER_KEYS:
                nonzero += int(np.sum(np.abs(np.asarray(
                    self.active.get(key), dtype=float)) > 1.0e-12))
            for knot in self.active.get("feedforward_knots", []):
                nonzero += int(np.sum(np.abs(np.asarray(
                    knot.get("delta_u_mv", []), dtype=float)) > 1.0e-12))
        return {
            "generation": int(self.generation),
            "canary_active": bool(self.canary_active),
            "baseline_mse_deg2": (float(self.baseline_mse)
                                   if np.isfinite(self.baseline_mse) else None),
            "last_valid_mse_deg2": self.last_valid_mse,
            "nonzero_parameters": nonzero,
            "implementation_kind": self.active_kind,
            "design_memory_count": len(self.design_memory),
            "last_trial": (self.design_memory[-1] if self.design_memory else None),
            "request_path": self.request_path,
            "candidate_path": self.candidate_path,
        }


def _find_codex_executable() -> Optional[str]:
    """Find the signed-in Codex CLI without depending on the Python env."""
    found = shutil.which("codex")
    if found:
        return os.path.abspath(found)
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        pattern = os.path.join(
            user_profile, ".vscode", "extensions", "openai.chatgpt-*-win32-x64",
            "bin", "windows-x86_64", "codex.exe")
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return os.path.abspath(matches[0])
    return None


def _find_claude_executable() -> Optional[str]:
    """Find the Claude Code CLI used for Anthropic account authentication."""
    found = shutil.which("claude")
    if found:
        return os.path.abspath(found)
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates = [
            os.path.join(user_profile, ".local", "bin", "claude.exe"),
            os.path.join(user_profile, "AppData", "Local", "Programs", "claude", "claude.exe"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None


def _build_design_prompt(report: Dict[str, object], report_context: str = "",
                         request_id: str = "test-request",
                         task_hash: str = "0" * 64,
                         base_generation: int = 0,
                         operator_feedback: str = "",
                         prompt_path: str = PROMPT_TEMPLATE_PATH) -> str:
    with open(prompt_path, encoding="utf-8") as stream:
        templates = json.load(stream)
    required_templates = {
        "main_prompt", "first_round", "later_round", "quality_above_30",
        "quality_at_or_below_30", "quality_unavailable",
    }
    if set(templates) != required_templates:
        raise ValueError("invalid codex_compensator_prompt.json structure")
    normalized = {}
    for key in required_templates:
        value = templates[key]
        if isinstance(value, str):
            parts = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            parts = value
        else:
            raise ValueError("invalid codex_compensator_prompt.json structure")
        if not parts or not all(item.strip() for item in parts):
            raise ValueError("invalid codex_compensator_prompt.json structure")
        normalized[key] = "\n\n".join(parts)
    templates = normalized
    candidate_outcomes = {
        "accepted", "rolled_back", "invalid_rolled_back",
        "interrupted_rolled_back",
    }
    memories = [item for item in (report.get("design_memory_last_5") or [])
                if isinstance(item, dict)]
    prior_trials = [item for item in memories
                    if str(item.get("outcome", "")) in candidate_outcomes]
    last_design = report.get("last_design")
    has_previous_design = bool(prior_trials) or bool(
        isinstance(last_design, dict) and last_design.get("generation"))

    if not has_previous_design:
        round_instruction = str(templates["first_round"])
    else:
        try:
            latest_mse = float(report.get("tracking_mse_deg2"))
        except (TypeError, ValueError):
            latest_mse = float("nan")
        if np.isfinite(latest_mse) and latest_mse > 30.0:
            quality = str(templates["quality_above_30"]).replace(
                "{latest_mse:.6f}", f"{latest_mse:.6f}")
        elif np.isfinite(latest_mse):
            quality = str(templates["quality_at_or_below_30"]).replace(
                "{latest_mse:.6f}", f"{latest_mse:.6f}")
        else:
            quality = str(templates["quality_unavailable"])
        round_instruction = str(templates["later_round"]).replace(
            "{quality_instruction}", quality)
    replacements = {
        "{round_instruction}": round_instruction,
        "{request_id}": request_id,
        "{task_hash}": task_hash,
        "{base_generation}": str(int(base_generation)),
        "{report_context}": report_context,
    }
    prompt = str(templates["main_prompt"])
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    operator_feedback = str(operator_feedback or "").strip()[:4000]
    if operator_feedback:
        prompt += "\n\n" + operator_feedback
    return prompt


def design_candidate_with_codex(request_path: str, candidate_path: str,
                                schema_path: str, workspace: str,
                                timeout_s: float = 300.0,
                                model: str = "",
                                reasoning_effort: str = "medium",
                                fast_mode: bool = False,
                                provider: str = "gpt",
                                api_key: str = "",
                                auth_method: str = "account",
                                operator_feedback: str = "",
                                operator_feedback_count: int = 0,
                                operator_feedback_items: Optional[List[str]] = None,
                                feedback_provider: Optional[
                                    Callable[[], List[str]]] = None,
                                prompt_profile: str = "base") -> Dict[str, object]:
    """Read one completed-lap report and ask the selected LLM for validated JSON only.

    GPT runs through Codex in a read-only sandbox; Claude receives only the
    compact report and schema over its API.  Neither path can edit the controller.
    This function validates the structured reply and performs the only candidate
    file write itself.
    """
    started = time.time()
    request_path = os.path.abspath(request_path)
    candidate_path = os.path.abspath(candidate_path)
    schema_path = os.path.abspath(schema_path)
    workspace = os.path.abspath(workspace)
    with open(request_path, encoding="utf-8") as stream:
        report = json.load(stream)
    if not bool(report.get("lap_valid_for_design", False)):
        reasons = report.get("invalid_reasons") or ["lap marked invalid"]
        raise ValueError("本轮传感器/样本审核未通过，不自动设计：" + "; ".join(map(str, reasons)))
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Codex候选JSON Schema不存在：{schema_path}")
    provider = str(provider or "gpt").strip().lower()
    if provider not in ("gpt", "claude"):
        raise ValueError(f"不支持的LLM提供商：{provider}")
    auth_method = str(auth_method or "account").strip().lower()
    if auth_method not in ("account", "api"):
        raise ValueError(f"不支持的LLM认证方式：{auth_method}")
    executable = (_find_codex_executable() if provider == "gpt" else
                  _find_claude_executable() if auth_method == "account" else None)
    if provider == "gpt" and not executable:
        raise FileNotFoundError("找不到 codex 命令；请在VS Code中安装/启用Codex并完成登录")
    api_key = str(api_key or "").strip()
    if provider == "claude" and auth_method == "account" and not executable:
        raise FileNotFoundError("找不到 claude 命令；请安装Claude Code并完成账号登录")
    if provider == "claude" and auth_method == "api" and len(api_key) < 20:
        raise ValueError("Claude API Key尚未配置")
    model = str(model or "").strip()
    if len(model) > 100 or any(ord(ch) < 32 for ch in model):
        raise ValueError("Codex模型名称无效")
    reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    fast_mode = bool(fast_mode)
    prompt_profile = normalize_prompt_profile(prompt_profile)
    prompt_path = PROMPT_TEMPLATE_PATHS[prompt_profile]

    request_id = str(report.get("request_id", ""))
    task_hash = str(report.get("task_hash", ""))
    base_generation = int(report.get("active_generation", -1))
    if not request_id or len(task_hash) != 64 or base_generation < 0:
        raise ValueError("Codex设计报告缺少request_id/task_hash/base_generation")
    audit_paths = _llm_recording_paths(request_id)
    candidate = None
    raw_final_output = ""
    final_result_written = False

    # Keep the full 20 Hz report on disk for engineering audit, but give the
    # LLM a uniformly decimated trace plus all precomputed metrics.  Around
    # 100 temporal samples preserve lag/reversal information without spending
    # most of the context window on adjacent near-duplicate sensor frames.
    compact_report = deepcopy(report)
    compact_report.pop("tested_parameters", None)
    compact_report.pop("stable_parameters_after_audit", None)
    for key in ("tested_compensator", "stable_compensator_after_audit"):
        implementation = compact_report.get(key)
        if isinstance(implementation, dict):
            implementation.pop("legacy_parameters", None)
    full_trace = list(compact_report.get("full_trace") or [])
    if len(full_trace) > 96:
        indexes = np.unique(
            np.linspace(0, len(full_trace)-1, 96).round().astype(int))
        compact_report["full_trace"] = [full_trace[int(i)] for i in indexes]
        compact_report["full_trace_decimated_from"] = len(full_trace)
        compact_report["full_trace_samples_supplied"] = len(indexes)
    if prompt_profile == "base":
        compact_report.pop("phase_samples", None)
    report_context = json.dumps(
        compact_report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    feedback_messages = [
        str(item).strip()[:1200] for item in (operator_feedback_items or [])
        if str(item).strip()
    ]
    operator_feedback = str(operator_feedback or "").strip()[:4000]
    if feedback_messages:
        operator_feedback = "\n".join(feedback_messages)[:4000]
    elif operator_feedback:
        feedback_messages = [operator_feedback]
    prompt = _build_design_prompt(
        compact_report, report_context, request_id, task_hash, base_generation,
        operator_feedback=operator_feedback, prompt_path=prompt_path)
    fd, output_path = tempfile.mkstemp(
        prefix=f"codex_candidate_{request_id}_", suffix=".json",
        dir=os.path.dirname(candidate_path))
    os.close(fd)
    thread_id = ""
    _append_llm_audit_events(
        audit_paths.get("reasoning", ""), "request_started",
        payload={
            "type": "request_started", "request_id": request_id,
            "provider": provider, "model": model or "default",
            "reasoning_effort": reasoning_effort, "fast_mode": fast_mode,
            "prompt_profile": prompt_profile,
            "base_generation": base_generation,
            "operator_feedback": list(feedback_messages),
        })
    try:
        feedback_count = (max(len(feedback_messages),
                              max(1, int(operator_feedback_count)))
                          if operator_feedback else 0)
        if provider == "gpt":
            # Plus is structurally larger and Pro deliberately leaves the
            # architecture open. Keep both threads resumable so a rejected
            # draft can see the exact local validator error and repair itself
            # before publication.
            resumable = (callable(feedback_provider) or
                         prompt_profile in ("plus", "pro"))
            command = [
                executable, "exec", "--sandbox", "read-only",
                "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
                "--color", "never", "-C", workspace,
            ]
            if not resumable:
                command.append("--ephemeral")
            if model:
                command.extend(["--model", model])
            if fast_mode:
                # This invocation ignores user config, so Fast mode must be
                # supplied explicitly for the offline compensator request.
                command.extend([
                    "--config", 'service_tier="fast"',
                    "--config", "features.fast_mode=true",
                ])
            command.extend([
                "--config", f'model_reasoning_effort="{reasoning_effort}"',
                "--output-schema", schema_path, "-o", output_path, "-",
            ])
            if resumable:
                command.insert(-1, "--json")
            kwargs = {}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(
                command, cwd=workspace, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                input=prompt,
                timeout=max(30.0, min(float(timeout_s), 900.0)), check=False, **kwargs)
            _append_llm_audit_events(
                audit_paths.get("reasoning", ""), "initial_design",
                completed.stdout, completed.stderr,
                {"type": "provider_exit", "returncode": completed.returncode})
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown error").strip()
                raise RuntimeError(
                    f"codex exec退出码{completed.returncode}：{detail[-1200:]}")
            if resumable:
                for line in (completed.stdout or "").splitlines():
                    try:
                        event = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if event.get("type") == "thread.started":
                        thread_id = str(event.get("thread_id") or "").strip()
            with open(output_path, encoding="utf-8") as stream:
                raw_final_output = stream.read()
            candidate = json.loads(raw_final_output)

            # codex exec cannot accept another user message while a turn is
            # still sampling.  Messages entered in the GUI during that turn
            # are therefore drained immediately afterwards and sent through
            # `exec resume` before the candidate is validated or exposed to
            # the hardware.  Multiple interventions can be handled without
            # loading any intermediate draft.
            for _ in range(4):
                notes = feedback_provider() if callable(feedback_provider) else []
                notes = [str(note).strip()[:1200] for note in (notes or [])
                         if str(note).strip()]
                if not notes:
                    break
                if not thread_id:
                    raise RuntimeError(
                        "Codex未返回可续接的会话ID，本轮人工意见未应用；稳定补偿器保持不变")
                feedback_count += len(notes)
                feedback_messages.extend(notes)
                revision_prompt = (
                    "The operator added the following requirements while you were "
                    "reasoning:\n- " + "\n- ".join(notes) +
                    "\nReview your just-produced candidate against these requirements. "
                    "Return a complete corrected JSON object matching the original schema, "
                    "with the same request_id, task_hash, and base_generation. The draft must "
                    "not be deployed before this review is complete."
                )
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass
                resume_command = [
                    executable, "exec", "resume", "--skip-git-repo-check",
                    "--ignore-user-config", "--ignore-rules",
                ]
                if model:
                    resume_command.extend(["--model", model])
                if fast_mode:
                    resume_command.extend([
                        "--config", 'service_tier="fast"',
                        "--config", "features.fast_mode=true",
                    ])
                resume_command.extend([
                    "--config", f'model_reasoning_effort="{reasoning_effort}"',
                    "--output-schema", schema_path, "-o", output_path,
                    "--json", thread_id, "-",
                ])
                revised = subprocess.run(
                    resume_command, cwd=workspace, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                    errors="replace", input=revision_prompt,
                    timeout=max(30.0, min(float(timeout_s), 900.0)),
                    check=False, **kwargs)
                _append_llm_audit_events(
                    audit_paths.get("reasoning", ""), "operator_review",
                    revised.stdout, revised.stderr,
                    {"type": "provider_exit", "returncode": revised.returncode,
                     "operator_feedback": notes})
                if revised.returncode != 0:
                    detail = (revised.stderr or revised.stdout or "unknown error").strip()
                    raise RuntimeError(
                        f"Codex人工复审退出码{revised.returncode}：{detail[-1200:]}")
                with open(output_path, encoding="utf-8") as stream:
                    raw_final_output = stream.read()
                candidate = json.loads(raw_final_output)
        elif auth_method == "api":
            with open(schema_path, encoding="utf-8") as stream:
                output_schema = json.load(stream)
            claude_model = model or "claude-sonnet-5"
            claude_effort = ({"low": "low", "medium": "medium",
                              "high": "high", "xhigh": "high"})[reasoning_effort]
            body = json.dumps({
                "model": claude_model,
                "max_tokens": 16000,
                "messages": [{"role": "user", "content": prompt}],
                "output_config": {
                    "effort": claude_effort,
                    "format": {"type": "json_schema", "schema": output_schema},
                },
            }, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body, method="POST",
                headers={"content-type": "application/json", "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"})
            try:
                with urllib.request.urlopen(
                        request, timeout=max(30.0, min(float(timeout_s), 900.0))) as response:
                    answer = json.loads(response.read().decode("utf-8"))
                _append_llm_audit_events(
                    audit_paths.get("reasoning", ""), "initial_design",
                    payload={"type": "provider_response", "response": answer})
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[-1200:]
                raise RuntimeError(f"Claude API HTTP {exc.code}：{detail}") from exc
            blocks = answer.get("content") or []
            text_blocks = [str(item.get("text", "")) for item in blocks
                           if isinstance(item, dict) and item.get("type") == "text"]
            if not text_blocks:
                raise RuntimeError("Claude API没有返回结构化候选")
            candidate = json.loads("".join(text_blocks))
            raw_final_output = json.dumps(candidate, ensure_ascii=False)
        else:
            with open(schema_path, encoding="utf-8") as stream:
                schema_text = json.dumps(json.load(stream), separators=(",", ":"))
            command = [
                executable, "--print", "--output-format", "json",
                "--json-schema", schema_text, "--model", model or "claude-sonnet-5",
                "--effort", ({"low": "low", "medium": "medium",
                               "high": "high", "xhigh": "max"})[reasoning_effort],
                "--no-session-persistence", "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--setting-sources", "",
            ]
            kwargs = {}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(
                command, cwd=workspace, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                input=prompt,
                timeout=max(30.0, min(float(timeout_s), 900.0)), check=False, **kwargs)
            _append_llm_audit_events(
                audit_paths.get("reasoning", ""), "initial_design",
                completed.stdout, completed.stderr,
                {"type": "provider_exit", "returncode": completed.returncode})
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown error").strip()
                raise RuntimeError(
                    f"claude退出码{completed.returncode}：{detail[-1200:]}")
            decoded = json.loads(completed.stdout or "{}")
            if isinstance(decoded, dict) and decoded.get("structured_output") is not None:
                candidate = decoded["structured_output"]
            elif isinstance(decoded, dict) and decoded.get("result") is not None:
                candidate = decoded["result"]
            else:
                candidate = decoded
            if isinstance(candidate, str):
                candidate = json.loads(candidate)
            raw_final_output = json.dumps(candidate, ensure_ascii=False)
        def validate_candidate_envelope(value):
            if not isinstance(value, dict):
                raise ValueError("Codex candidate must be one JSON object")
            if int(value.get("schema", -1)) != CodexCompensator.SCHEMA:
                raise ValueError("Codex returned a mismatched candidate schema")
            if str(value.get("request_id", "")) != request_id:
                raise ValueError("Codex returned the wrong request_id")
            if str(value.get("task_hash", "")) != task_hash:
                raise ValueError("Codex returned the wrong task_hash")
            if int(value.get("base_generation", -1)) != base_generation:
                raise ValueError("Codex returned a stale base_generation")
            for field, max_length in (("rationale", 2000),
                                      ("design_strategy", 2000),
                                      ("expected_effect", 1000),
                                      ("failure_analysis", 2000)):
                field_value = value.get(field)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(f"Codex candidate missing required {field}")
                value[field] = field_value.strip()[:max_length]
            return value

        # Plus has a larger executable contract and Pro has no mandatory
        # implementation template. If either first draft violates the local
        # contract, the same Codex thread receives the exact validator error
        # and may correct it twice. Invalid drafts are never written to the
        # candidate path or exposed to hardware control.
        for repair_index in range(3):
            try:
                candidate = validate_candidate_envelope(candidate)
                candidate["source_code"] = _probe_compensator_source(
                    candidate.get("source_code"), report.get("full_trace") or [])
                break
            except (TypeError, ValueError) as validation_error:
                can_repair = (prompt_profile in ("plus", "pro") and
                              provider == "gpt" and bool(thread_id) and
                              repair_index < 2)
                if not can_repair:
                    raise
                repair_prompt = (
                    "Your candidate was rejected by the local deterministic validator. "
                    "Nothing was deployed. Validator error:\n" +
                    str(validation_error)[:1600] +
                    "\nReturn the entire corrected JSON object with the exact original "
                    "request_id, task_hash, base_generation, and schema. Preserve your "
                    "control objective, but correct the executable contract violation. "
                    "Re-read and self-audit every source-code restriction in the original "
                    "selected design prompt before returning."
                )
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass
                repair_command = [
                    executable, "exec", "resume", "--skip-git-repo-check",
                    "--ignore-user-config", "--ignore-rules",
                ]
                if model:
                    repair_command.extend(["--model", model])
                if fast_mode:
                    repair_command.extend([
                        "--config", 'service_tier="fast"',
                        "--config", "features.fast_mode=true",
                    ])
                repair_command.extend([
                    "--config", f'model_reasoning_effort="{reasoning_effort}"',
                    "--output-schema", schema_path, "-o", output_path,
                    "--json", thread_id, "-",
                ])
                repaired = subprocess.run(
                    repair_command, cwd=workspace, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                    errors="replace", input=repair_prompt,
                    timeout=max(30.0, min(float(timeout_s), 900.0)),
                    check=False, **kwargs)
                _append_llm_audit_events(
                    audit_paths.get("reasoning", ""), "validator_repair",
                    repaired.stdout, repaired.stderr,
                    {"type": "provider_exit", "returncode": repaired.returncode,
                     "repair_attempt": repair_index + 1,
                     "validation_error": str(validation_error)})
                if repaired.returncode != 0:
                    detail = (repaired.stderr or repaired.stdout or
                              "unknown error").strip()
                    raise RuntimeError(
                        f"Codex validator repair exited {repaired.returncode}: "
                        f"{detail[-1200:]}")
                with open(output_path, encoding="utf-8") as stream:
                    raw_final_output = stream.read()
                candidate = json.loads(raw_final_output)
        _, contract_instance = _validate_compensator_source(
            candidate["source_code"], require_design_contract=True)
        candidate["kp_matrix"] = _optional_compensator_array(
            contract_instance, "kp_matrix", (NCH, NAX))
        candidate["kd_matrix"] = _optional_compensator_array(
            contract_instance, "kd_matrix", (NCH, NAX))
        candidate["ki_matrix"] = _optional_compensator_array(
            contract_instance, "ki_matrix", (NCH, NAX))
        candidate["integral_limit_deg_s"] = _optional_compensator_array(
            contract_instance, "integral_limit_deg_s", (NAX,))
        candidate["output_limit_mv"] = _optional_compensator_array(
            contract_instance, "output_limit_mv", (NCH,))
        candidate["output_slew_mv_per_s"] = _optional_compensator_array(
            contract_instance, "output_slew_mv_per_s", (NCH,))
        # Persist the exact operator text after schema validation.  This field
        # is local audit metadata, not something the LLM is asked to produce.
        candidate["operator_feedback_used"] = list(feedback_messages)
        candidate["human_feedback_count"] = int(feedback_count)
        candidate["prompt_profile"] = prompt_profile
        _write_llm_final_result(audit_paths.get("final", ""), candidate)
        final_result_written = True
        _append_llm_audit_events(
            audit_paths.get("reasoning", ""), "validation_complete",
            payload={
                "type": "validation_complete", "request_id": request_id,
                "base_generation": base_generation,
                "human_feedback_count": int(feedback_count),
            })
        stable = report.get("stable_compensator_after_audit") or {}
        stable_source = str(stable.get("source_code") or "").strip()
        if stable_source and candidate["source_code"].strip() == stable_source:
            # An unchanged answer is a valid decision to keep the audited
            # controller, not a fatal design error.  Archive the LLM's
            # explanation, but do not publish it as a new candidate (otherwise
            # begin_round would try to create a meaningless new generation).
            candidate["disposition"] = "unchanged_reused"
            archive = os.path.splitext(candidate_path)[0] + "_" + request_id + ".json"
            CodexCompensator._atomic_json(archive, candidate)
            return {
                "state": "reused",
                "message": (f"LLM未修改generation {base_generation}；"
                            "沿用当前稳定补偿器并自动继续下一轮"),
                "request_id": request_id,
                "base_generation": base_generation,
                "generation": base_generation,
                "implementation_sha256": hashlib.sha256(
                    candidate["source_code"].encode("utf-8")).hexdigest(),
                "archive_path": archive,
                "fast_mode": bool(fast_mode and provider == "gpt"),
                "prompt_profile": prompt_profile,
                "human_feedback_count": int(feedback_count),
                "operator_feedback_used": list(feedback_messages),
                "duration_s": round(time.time()-started, 2),
                "rationale": str(candidate.get("rationale", ""))[:500],
                "design_strategy": str(candidate.get("design_strategy", ""))[:500],
                "expected_effect": str(candidate.get("expected_effect", ""))[:500],
                "failure_analysis": str(candidate.get("failure_analysis", ""))[:500],
                "source_code": candidate["source_code"],
                "kp_matrix": candidate["kp_matrix"],
                "kd_matrix": candidate["kd_matrix"],
                "ki_matrix": candidate["ki_matrix"],
                "integral_limit_deg_s": candidate["integral_limit_deg_s"],
                "output_limit_mv": candidate["output_limit_mv"],
                "output_slew_mv_per_s": candidate["output_slew_mv_per_s"],
            }
        CodexCompensator._atomic_json(candidate_path, candidate)
        archive = os.path.splitext(candidate_path)[0] + "_" + request_id + ".json"
        CodexCompensator._atomic_json(archive, candidate)
        return {
            "state": "ready",
            "message": f"候选generation {base_generation+1}已生成，按播放开始实机审核",
            "request_id": request_id,
            "base_generation": base_generation,
            "candidate_generation": base_generation+1,
            "implementation_sha256": hashlib.sha256(
                candidate["source_code"].encode("utf-8")).hexdigest(),
            "kp_matrix": candidate["kp_matrix"],
            "kd_matrix": candidate["kd_matrix"],
            "ki_matrix": candidate["ki_matrix"],
            "integral_limit_deg_s": candidate["integral_limit_deg_s"],
            "output_limit_mv": candidate["output_limit_mv"],
            "output_slew_mv_per_s": candidate["output_slew_mv_per_s"],
            "candidate_path": candidate_path,
            "fast_mode": bool(fast_mode and provider == "gpt"),
            "prompt_profile": prompt_profile,
            "human_feedback_count": int(feedback_count),
            "operator_feedback_used": list(feedback_messages),
            "duration_s": round(time.time()-started, 2),
            "rationale": str(candidate.get("rationale", ""))[:500],
            "design_strategy": str(candidate.get("design_strategy", ""))[:500],
            "expected_effect": str(candidate.get("expected_effect", ""))[:500],
            "failure_analysis": str(candidate.get("failure_analysis", ""))[:500],
            "source_code": candidate["source_code"],
        }
    except Exception as exc:
        if not final_result_written:
            _write_llm_final_result(
                audit_paths.get("final", ""), candidate,
                validation_error=str(exc), raw_output=raw_final_output)
        _append_llm_audit_events(
            audit_paths.get("reasoning", ""), "design_failed",
            payload={"type": "design_failed", "request_id": request_id,
                     "error": str(exc)})
        raise
    finally:
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass

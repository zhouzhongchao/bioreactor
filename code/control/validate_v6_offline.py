# -*- coding: utf-8 -*-
"""Offline-only validation for the TRO shoulder V6 model.

The script reads existing CSV logs and never opens a serial port, starts the
GUI, or sends a robot command.  It compares the deployed legacy geometry with
the corrected V6 geometry using zero-aligned measured muscle-system lengths,
and audits the logged lock/antagonist invariants.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from project_paths import CONTROL_LOG_DIR

import shoulder_pam_model as legacy  # noqa: E402
import shoulder_pam_model_v6 as v6  # noqa: E402


def _prepare_legacy() -> None:
    if legacy.N_PAM == 8:
        legacy.use_ten_pam()
        legacy.set_pivot_dx(66.0)


def _float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _geometry_channel_lengths(model: str, row: dict[str, str]) -> np.ndarray:
    # mpc_station logs q1..q3 as degrees of the deployed legacy model
    # coordinates, not the GUI swing/twist coordinates.  Reconstruct the
    # physical rotation with that model, then re-parameterise it for V6.
    legacy_q = np.radians([_float(row, f"q{i}") for i in range(1, 4)])
    if not np.all(np.isfinite(legacy_q)):
        return np.full(v6.N_VALVE, np.nan)
    if model == "v6":
        rotation = legacy.R_M(legacy_q)
        lengths = v6.pam_lengths_mm(v6.q_from_rotation_matrix(rotation))
    else:
        lengths = legacy.pam_lengths_mm(legacy_q)
    return np.array([
        float(np.mean(lengths[list(rows)])) for rows in v6.VALVE_GEOMETRY_ROWS
    ])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _metric(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    good = np.isfinite(measured) & np.isfinite(predicted)
    measured = measured[good]
    predicted = predicted[good]
    if measured.size == 0:
        return {"samples": 0, "rmse_mm": None, "mae_mm": None,
                "correlation": None, "best_scale": None,
                "scaled_rmse_mm": None}
    error = predicted-measured
    if measured.size > 2 and np.std(measured) > 1.0e-12 and np.std(predicted) > 1.0e-12:
        correlation = float(np.corrcoef(measured, predicted)[0, 1])
    else:
        correlation = None
    denominator = float(predicted @ predicted)
    scale = float((predicted @ measured)/denominator) if denominator > 1.0e-12 else 0.0
    scaled_error = scale*predicted-measured
    return {
        "samples": int(measured.size),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "mae_mm": float(np.mean(np.abs(error))),
        "correlation": correlation,
        "best_scale": scale,
        "scaled_rmse_mm": float(np.sqrt(np.mean(scaled_error**2))),
    }


def _geometry_comparison(paths: Iterable[Path]) -> tuple[dict, int, float]:
    measured_by_channel = [[] for _ in range(v6.N_VALVE)]
    legacy_by_channel = [[] for _ in range(v6.N_VALVE)]
    v6_by_channel = [[] for _ in range(v6.N_VALVE)]
    used_files = 0
    max_legacy_v6_difference_mm = 0.0
    for path in paths:
        rows = _read_rows(path)
        if not rows or not all(f"L{i}_mm" in rows[0] for i in range(v6.N_VALVE)):
            continue
        measured = np.array([
            [_float(row, f"L{i}_mm") for i in range(v6.N_VALVE)] for row in rows
        ])
        legacy_length = np.array([_geometry_channel_lengths("legacy", row) for row in rows])
        v6_length = np.array([_geometry_channel_lengths("v6", row) for row in rows])
        valid_rows = (np.all(np.isfinite(measured), axis=1)
                      & np.all(np.isfinite(legacy_length), axis=1)
                      & np.all(np.isfinite(v6_length), axis=1))
        if np.count_nonzero(valid_rows) < 2:
            continue
        measured = measured[valid_rows]
        legacy_length = legacy_length[valid_rows]
        v6_length = v6_length[valid_rows]
        max_legacy_v6_difference_mm = max(
            max_legacy_v6_difference_mm,
            float(np.max(np.abs(legacy_length-v6_length))))
        # Encoder absolute offsets differ between sessions.  Geometry validates
        # length change, so each run is aligned at its first complete sample.
        measured -= measured[0]
        legacy_length -= legacy_length[0]
        v6_length -= v6_length[0]
        for channel in range(v6.N_VALVE):
            measured_by_channel[channel].append(measured[:, channel])
            legacy_by_channel[channel].append(legacy_length[:, channel])
            v6_by_channel[channel].append(v6_length[:, channel])
        used_files += 1

    channels = {}
    for channel in range(v6.N_VALVE):
        measured = (np.concatenate(measured_by_channel[channel])
                    if measured_by_channel[channel] else np.array([]))
        old = (np.concatenate(legacy_by_channel[channel])
               if legacy_by_channel[channel] else np.array([]))
        new = (np.concatenate(v6_by_channel[channel])
               if v6_by_channel[channel] else np.array([]))
        channels[f"V{channel}"] = {
            "physical_muscles": list(v6.VALVE_PHYSICAL_MUSCLES[channel]),
            "legacy": _metric(measured, old),
            "v6": _metric(measured, new),
        }
    return channels, used_files, max_legacy_v6_difference_mm


def _lock_audit(paths: Iterable[Path]) -> dict[str, int | float | list]:
    positive_without_locked = 0
    antagonist_violations = 0
    unlock_above_30kpa = 0
    lead_delays = []
    short_lead_events = []
    high_pressure_unlock_events = []
    audited_rows = 0
    for path in paths:
        rows = _read_rows(path)
        if not rows:
            continue
        header = rows[0]
        if not all(f"lock{i}" in header for i in range(v6.N_VALVE)):
            continue
        lock_started = [None]*v6.N_VALVE
        previous_lock = ["o"]*v6.N_VALVE
        have_previous = False
        for row in rows:
            time_s = _float(row, "t")
            uact = np.array([_float(row, f"uact{i}") for i in range(v6.N_VALVE)])
            kcmd = np.array([
                int(round(_float(row, f"kcmd{i}"))) if math.isfinite(_float(row, f"kcmd{i}")) else 0
                for i in range(v6.N_VALVE)
            ])
            locks = [str(row.get(f"lock{i}", "")) for i in range(v6.N_VALVE)]
            pressure = np.array([_float(row, f"P{i}") for i in range(v6.N_VALVE)])
            if np.all(np.isfinite(uact)):
                positive_without_locked += sum(
                    1 for channel in range(v6.N_VALVE)
                    if uact[channel] > 0.5 and locks[channel] != "L")
                for left, right in v6.ANTAGONIST_PAIRS:
                    antagonist_violations += int(uact[left] > 0.5 and uact[right] > 0.5)
            for channel in range(v6.N_VALVE):
                if (have_previous and previous_lock[channel] == "o"
                        and locks[channel] == "w" and math.isfinite(time_s)):
                    lock_started[channel] = time_s
                if locks[channel] == "o":
                    lock_started[channel] = None
                if (math.isfinite(uact[channel]) and uact[channel] > 0.5
                        and lock_started[channel] is not None):
                    delay = time_s-lock_started[channel]
                    lead_delays.append(delay)
                    if delay < 0.1-1.0e-9 and len(short_lead_events) < 20:
                        short_lead_events.append({
                            "file": path.name, "time_s": time_s, "channel": channel,
                            "delay_s": delay, "lock": locks[channel],
                        })
                    lock_started[channel] = None
                if (have_previous and previous_lock[channel] in ("w", "L")
                        and locks[channel] == "o"
                        and math.isfinite(pressure[channel])
                        and pressure[channel] > v6.UNLOCK_PRESSURE_KPA + 1.0e-9):
                    unlock_above_30kpa += 1
                    if len(high_pressure_unlock_events) < 20:
                        high_pressure_unlock_events.append({
                            "file": path.name, "time_s": time_s, "channel": channel,
                            "pressure_kpa": float(pressure[channel]),
                            "previous_lock": previous_lock[channel],
                        })
            previous_lock = locks
            have_previous = True
            audited_rows += 1
    delays = np.asarray(lead_delays, dtype=float)
    return {
        "rows": audited_rows,
        "positive_uact_without_lock_L": positive_without_locked,
        "antagonist_pair_violations": antagonist_violations,
        "unlock_transitions_above_30kpa": unlock_above_30kpa,
        "observed_locking_state_to_actuation_events": int(delays.size),
        "minimum_observed_telemetry_gap_s": (
            float(np.min(delays)) if delays.size else None),
        "telemetry_gaps_below_100ms": int(np.count_nonzero(delays < 0.1-1.0e-9)),
        "telemetry_gaps_below_100ms_examples": short_lead_events,
        "telemetry_gap_interpretation": (
            "Rows are sampled at 50 ms and command/state reporting is not atomic; "
            "the sampled o->w gap can underestimate firmware wall-clock lock time. "
            "Use the firmware source audit for the exact 100 ms contract."),
        "unlock_above_30kpa_examples": high_pressure_unlock_events,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_constraint_audit() -> dict:
    firmware_path = ROOT / "arduino test" / "teach_mode" / "src" / "teach_mode" / "teach_mode.ino"
    gains_path = HERE / "gains.json"
    firmware = firmware_path.read_text(encoding="utf-8")
    gains = json.loads(gains_path.read_text(encoding="utf-8"))
    checks = {
        "firmware_lock_lead_constant_100ms": "LOCK_LEAD_MS = 100" in firmware,
        "firmware_enforces_elapsed_lock_lead": (
            "millis() - lkT0[c] >= LOCK_LEAD_MS" in firmware),
        "firmware_unlock_threshold_constant_30kpa": "KPA_ZERO_NOISE = 30.0f" in firmware,
        "firmware_30kpa_used_by_pressure_empty": (
            "chKpa(ch) <= KPA_ZERO_NOISE" in firmware),
        "host_lock_lead_setting_100ms": float(gains.get("mpc_lock_lead_ms", -1.0)) == 100.0,
        "v6_lock_lead_steps_at_20hz_is_two": v6.lock_lead_steps(0.05) == 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "firmware_path": str(firmware_path),
        "firmware_sha256": _sha256(firmware_path),
        "gains_path": str(gains_path),
    }


def validate(paths: list[Path]) -> dict:
    _prepare_legacy()
    channels, used_files, max_geometry_difference = _geometry_comparison(paths)
    h0 = v6.H_matrix(np.zeros(3))
    rank, singular_values, condition = v6.spatial_moment_arm_metrics(np.zeros(3))
    return {
        "model_version": v6.MODEL_VERSION,
        "mode": "offline_read_only",
        "log_files_considered": len(paths),
        "log_files_with_complete_length_data": used_files,
        "geometry_length_change_comparison": channels,
        "legacy_v6_physical_length_max_difference_mm": max_geometry_difference,
        "legacy_v6_geometry_interpretation": (
            "The two parameterisations reconstruct the same physical rotation, so their "
            "PAM lengths match; V6 removes the report/code coordinate inconsistency in E, "
            "J_L and generalized dynamics rather than changing spatial anchor geometry."),
        "logged_pose_coordinate_interpretation": (
            "CSV q1/q2/q3 are deployed legacy-model Euler coordinates in degrees.  The "
            "validator reconstructs the physical rotation with the legacy model and then "
            "re-parameterises that rotation in V6 coordinates."),
        "v51_zero_pose": {
            "lengths_mm": v6.pam_lengths_mm(np.zeros(3)).tolist(),
            "H_m": h0.tolist(),
            "rank": rank,
            "singular_values_m": singular_values.tolist(),
            "condition_number": condition,
        },
        "constraints": {
            "valve_order": ["V0=15", "V1=48", "V2=23", "V3=67", "V4=9", "V5=10"],
            "valve_physical_muscles": [list(group) for group in v6.VALVE_PHYSICAL_MUSCLES],
            "antagonist_pairs": [list(pair) for pair in v6.ANTAGONIST_PAIRS],
            "lock_lead_ms": v6.LOCK_LEAD_MS,
            "lock_lead_steps_at_20hz": v6.lock_lead_steps(0.05),
            "unlock_pressure_kpa": v6.UNLOCK_PRESSURE_KPA,
            "unlock_threshold_scope": "unlock judgement only",
        },
        "muscle_calibration": {
            "status": "PSS-PAM three-element coefficients supplied by experiment",
            "identified_pressure_min_kpa": v6.MUSCLE_IDENTIFIED_PRESSURE_MIN_KPA,
            "pressure_unit": "kPa",
            "published_units": {
                "K": "N/m",
                "B": "N*s/m",
                "F": "N",
            },
            "code_units": {
                "K_n_per_mm": "0.33547 + 0.01567*P",
                "B_inflate_n_s_per_mm": "1.10000 + 0.00405*P",
                "B_deflate_n_s_per_mm": "0.70000 - 0.00261*P",
                "F_n": "-10.278 + 0.585*P",
            },
            "scope": (
                "Identifies the three-element PAM body only.  Kss, Ksl, spring "
                "preloads, installation-to-installation variation and the robot "
                "environment dynamics are not identified by these equations."),
            "below_25kpa": (
                "continuous extrapolation only; 30 kPa remains exclusively the "
                "unlock decision threshold and is not a force-model cutoff"),
        },
        "logged_constraint_audit": _lock_audit(paths),
        "source_constraint_audit": _source_constraint_audit(),
        "offline_compensator": {
            "path": str(HERE / "codex_compensator.py"),
            "sha256": _sha256(HERE / "codex_compensator.py"),
            "modified_by_v6_work": False,
        },
        "limitations": [
            "The supplied experiment identifies F0/F1/K/B, but these logs contain no direct ten-muscle force measurement with which to independently validate those coefficients on this installation.",
            "Kss, Ksl, their preloads, silicone restoring force, joint friction and pressure dynamics remain unidentified.",
            "The length comparison removes each run's first-sample encoder offset and validates changes only.",
            "V6 is not imported by the live controller in this change set.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", action="append", dest="patterns",
                        help="CSV glob relative to the workspace; may be repeated")
    parser.add_argument("--output", default=str(CONTROL_LOG_DIR / "v6_offline_validation.json"))
    args = parser.parse_args()
    patterns = args.patterns or [
        str(CONTROL_LOG_DIR / "mpc_codex_*.csv"),
        str(CONTROL_LOG_DIR / "musclepi_*.csv"),
    ]
    paths = sorted({Path(path).resolve() for pattern in patterns for path in glob.glob(pattern)})
    report = validate(paths)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(output),
        "files": report["log_files_considered"],
        "length_files": report["log_files_with_complete_length_data"],
        "lock_audit": report["logged_constraint_audit"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

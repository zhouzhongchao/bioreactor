# -*- coding: utf-8 -*-
"""Offline fitting for the V6 pose-to-length observation layer.

The fitter never opens a serial port and never changes the live controller.  It
compares six measured length channels with the fixed V6 ten-muscle geometry and
reports direct, unscaled agreement using Kendall tau-b and RMSE.
Possible channel mismatches are diagnostic only; the six-valve hardware map
remains immutable.
"""
from __future__ import annotations

import csv
import datetime as _datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shoulder_pam_model as legacy  # noqa: E402
import shoulder_pam_model_v6 as v6  # noqa: E402


MIN_SAMPLES = 120
MIN_RANGE_MM = 3.0
MIN_ABS_KENDALL_TAU_B = 0.60


def _prepare_legacy():
    if legacy.N_PAM == 8:
        legacy.use_ten_pam()
        legacy.set_pivot_dx(66.0)


def predicted_channel_lengths_mm(q_legacy_deg):
    """Return V6 six-channel lengths for deployed legacy Euler coordinates."""
    _prepare_legacy()
    q_deg = np.asarray(q_legacy_deg, dtype=float)
    if q_deg.ndim != 2 or q_deg.shape[1] != 3:
        raise ValueError("q_legacy_deg must have shape (n, 3)")
    answer = np.empty((q_deg.shape[0], v6.N_VALVE), dtype=float)
    for row_index, row in enumerate(q_deg):
        rotation = legacy.R_M(np.radians(row))
        lengths = v6.pam_lengths_mm(v6.q_from_rotation_matrix(rotation))
        answer[row_index] = [
            float(np.mean(lengths[list(group)]))
            for group in v6.VALVE_GEOMETRY_ROWS
        ]
    return answer


def _zero_align(values, session_ids):
    aligned = np.asarray(values, dtype=float).copy()
    ids = np.asarray(session_ids)
    for session_id in dict.fromkeys(ids.tolist()):
        indices = np.flatnonzero(ids == session_id)
        if indices.size:
            aligned[indices] -= aligned[indices[0]]
    return aligned


def _kendall_tau_b(x, y):
    """Return Kendall tau-b, which corrects for ties in either sample."""
    if x.size < 3:
        return None
    result = kendalltau(x, y, variant="b", nan_policy="omit")
    value = float(result.statistic)
    if not np.isfinite(value):
        return None
    return value


def _pair_metrics(predicted, measured, sample_mask=None):
    good = np.isfinite(predicted) & np.isfinite(measured)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != good.shape:
            raise ValueError("sample_mask must match the sample vector shape")
        good &= mask
    x = np.asarray(predicted, dtype=float)[good]
    y = np.asarray(measured, dtype=float)[good]
    if x.size == 0:
        return {
            "samples": 0, "kendall_tau_b": None, "rmse_mm": None,
            "predicted_range_mm": None, "measured_range_mm": None,
            "accepted": False, "reasons": ["no finite samples"],
        }
    x_range = float(np.ptp(x))
    y_range = float(np.ptp(y))
    tau_b = _kendall_tau_b(x, y)
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    reasons = []
    if x.size < MIN_SAMPLES:
        reasons.append("samples<%d" % MIN_SAMPLES)
    if x_range < MIN_RANGE_MM:
        reasons.append("predicted_range<%.1fmm" % MIN_RANGE_MM)
    if y_range < MIN_RANGE_MM:
        reasons.append("measured_range<%.1fmm" % MIN_RANGE_MM)
    if tau_b is None or abs(tau_b) < MIN_ABS_KENDALL_TAU_B:
        reasons.append("abs(kendall_tau_b)<%.2f" % MIN_ABS_KENDALL_TAU_B)
    return {
        "samples": int(x.size),
        "kendall_tau_b": tau_b,
        "rmse_mm": rmse,
        "predicted_range_mm": x_range,
        "measured_range_mm": y_range,
        "accepted": not reasons,
        "reasons": reasons,
    }


def fit_geometry_samples(q_legacy_deg, measured_lengths_mm, session_ids=None,
                         source="unknown", channel_sample_masks=None):
    """Fit a candidate without changing geometry, mappings, or live control."""
    q_deg = np.asarray(q_legacy_deg, dtype=float)
    measured = np.asarray(measured_lengths_mm, dtype=float)
    if q_deg.ndim != 2 or q_deg.shape[1] != 3:
        raise ValueError("q_legacy_deg must have shape (n, 3)")
    if measured.shape != (q_deg.shape[0], v6.N_VALVE):
        raise ValueError("measured_lengths_mm must have shape (n, 6)")
    if session_ids is None:
        session_ids = np.zeros(q_deg.shape[0], dtype=int)
    ids = np.asarray(session_ids)
    if ids.shape != (q_deg.shape[0],):
        raise ValueError("session_ids must have shape (n,)")
    if channel_sample_masks is None:
        channel_masks = np.ones((q_deg.shape[0], v6.N_VALVE), dtype=bool)
    else:
        channel_masks = np.asarray(channel_sample_masks, dtype=bool)
        if channel_masks.shape != (q_deg.shape[0], v6.N_VALVE):
            raise ValueError("channel_sample_masks must have shape (n, 6)")

    finite = np.all(np.isfinite(q_deg), axis=1) & np.all(np.isfinite(measured), axis=1)
    q_deg = q_deg[finite]
    measured = measured[finite]
    ids = ids[finite]
    channel_masks = channel_masks[finite]
    if q_deg.shape[0] < 2:
        raise ValueError("not enough complete calibration samples")

    predicted = predicted_channel_lengths_mm(q_deg)
    measured_delta = _zero_align(measured, ids)
    predicted_delta = _zero_align(predicted, ids)

    channels = []
    mapping_identity = True
    for measured_channel in range(v6.N_VALVE):
        sample_mask = channel_masks[:, measured_channel]
        fixed = _pair_metrics(
            predicted_delta[:, measured_channel], measured_delta[:, measured_channel],
            sample_mask=sample_mask)
        alternatives = []
        for predicted_channel in range(v6.N_VALVE):
            metric = _pair_metrics(
                predicted_delta[:, predicted_channel], measured_delta[:, measured_channel],
                sample_mask=sample_mask)
            alternatives.append({
                "predicted_channel": predicted_channel,
                "kendall_tau_b": metric["kendall_tau_b"],
                "rmse_mm": metric["rmse_mm"],
            })
        ranked = sorted(
            alternatives,
            key=lambda item: (
                -(abs(item["kendall_tau_b"])
                  if item["kendall_tau_b"] is not None else -1.0),
                item["rmse_mm"] if item["rmse_mm"] is not None else float("inf")),
        )
        best = ranked[0]
        same = int(best["predicted_channel"]) == measured_channel
        mapping_identity = mapping_identity and same
        channels.append({
            "valve": "V%d" % measured_channel,
            "physical_muscles": list(v6.VALVE_PHYSICAL_MUSCLES[measured_channel]),
            "fixed_mapping_fit": fixed,
            "diagnostic_best_geometry_channel": "V%d" % int(best["predicted_channel"]),
            "diagnostic_best_abs_kendall_tau_b": (
                abs(float(best["kendall_tau_b"]))
                if best["kendall_tau_b"] is not None else None),
            "diagnostic_mapping_matches_fixed": same,
            "diagnostic_only": True,
        })

    accepted = [bool(channel["fixed_mapping_fit"]["accepted"]) for channel in channels]
    return {
        "schema_version": 3,
        "model_version": v6.MODEL_VERSION,
        "generated_at_utc": _datetime.datetime.now(
            _datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "mode": "candidate_only_offline_fit",
        "source": str(source),
        "samples": int(q_deg.shape[0]),
        "sessions": int(len(set(ids.tolist()))),
        "calibration_definition": (
            "direct comparison of delta_L_measured and delta_L_V6_geometry; "
            "per-session offsets are removed and no scale is fitted"),
        "quality_thresholds": {
            "minimum_samples": MIN_SAMPLES,
            "minimum_range_mm": MIN_RANGE_MM,
            "minimum_abs_kendall_tau_b": MIN_ABS_KENDALL_TAU_B,
            "rmse_mm": "reported without rescaling; no acceptance limit configured",
        },
        "immutable_contract": {
            "valve_order": ["V0=15", "V1=48", "V2=23", "V3=67", "V4=9", "V5=10"],
            "valve_physical_muscles": [list(group) for group in v6.VALVE_PHYSICAL_MUSCLES],
            "antagonist_pairs": [list(pair) for pair in v6.ANTAGONIST_PAIRS],
            "lock_lead_ms": v6.LOCK_LEAD_MS,
            "unlock_pressure_kpa": v6.UNLOCK_PRESSURE_KPA,
            "unlock_threshold_scope": "unlock judgement only",
        },
        "channels": channels,
        "fixed_mapping_quality_passed": all(accepted),
        "mapping_diagnostic_identity": mapping_identity,
        "mapping_diagnostic_is_quality_gate": False,
        "ready_to_apply": bool(all(accepted)),
        "application_state": "not_applied",
        "notes": [
            "The alternative-channel result is diagnostic only and never changes the six-channel map.",
            "Alternative-channel ranking is not an acceptance gate because a single-axis trial can make several geometry channels monotonic together.",
            "Kendall tau-b measures concordant versus discordant value ordering and corrects for ties.",
            "RMSE is the direct zero-aligned length error in mm; no fitted scale is used.",
            "RMSE is reported for review but has no automatic pass threshold until one is specified.",
        ],
    }


def read_log_samples(paths):
    q_rows = []
    length_rows = []
    session_ids = []
    used_paths = []
    for session_id, raw_path in enumerate(paths):
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
        required = (["q1", "q2", "q3"] + ["L%d_mm" % i for i in range(v6.N_VALVE)])
        if not rows or not all(name in rows[0] for name in required):
            continue
        before = len(q_rows)
        for row in rows:
            try:
                q = [float(row["q1"]), float(row["q2"]), float(row["q3"])]
                lengths = [float(row["L%d_mm" % i]) for i in range(v6.N_VALVE)]
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(q)) and np.all(np.isfinite(lengths)):
                q_rows.append(q)
                length_rows.append(lengths)
                session_ids.append(session_id)
        if len(q_rows) > before:
            used_paths.append(str(path.resolve()))
    if not q_rows:
        raise ValueError("no complete q/L rows found in the selected logs")
    return (np.asarray(q_rows, dtype=float), np.asarray(length_rows, dtype=float),
            np.asarray(session_ids), used_paths)


def fit_geometry_logs(paths):
    q, lengths, session_ids, used_paths = read_log_samples(paths)
    result = fit_geometry_samples(q, lengths, session_ids, source="existing_hardware_logs")
    result["source_files"] = used_paths
    return result


def save_candidate(result, path):
    """Atomically save a candidate; this does not make it active."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(target))
    return target


def concise_summary(result):
    parts = []
    for channel in result["channels"]:
        fit = channel["fixed_mapping_fit"]
        corr = fit["kendall_tau_b"]
        corr_text = "nan" if corr is None else "%.2f" % corr
        rmse = fit["rmse_mm"]
        rmse_text = "nan" if rmse is None else "%.2f" % rmse
        mark = "OK" if fit["accepted"] else "需复核"
        parts.append("%s:%s tau_b=%s RMSE=%smm" % (
            channel["valve"], mark, corr_text, rmse_text))
    return "; ".join(parts)


__all__ = [
    "fit_geometry_logs", "fit_geometry_samples", "predicted_channel_lengths_mm",
    "save_candidate", "concise_summary",
]

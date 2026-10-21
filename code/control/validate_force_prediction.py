from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


CHANNEL_LABELS = ("A1+5", "A2+3", "A4+8", "A6+7", "A9", "A10")
FEATURE_TOKENS = (
    "P_kPa",
    "L_mm",
    "d_active_mm",
    "d_passive_mm",
    "dP_dt_kPa_s",
    "dL_dt_mm_s",
)
PREDICTION_PREFIX = "FORCE_PREDICTION "


def _number(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_response(path: Path) -> tuple[list[str], list[tuple]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    iterator = worksheet.iter_rows(values_only=True)
    headers = [str(value) for value in next(iterator)]
    rows = [tuple(row) for row in iterator]
    workbook.close()
    return headers, rows


def _response_value(rows, columns, row_index: int, name: str) -> float:
    return _number(rows[row_index][columns[name]])


def _worker_message(process: subprocess.Popen) -> dict:
    while True:
        line = process.stdout.readline()
        if not line:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(
                f"Force worker exited with code {process.poll()}: {stderr.strip()}"
            )
        if line.startswith(PREDICTION_PREFIX):
            return json.loads(line[len(PREDICTION_PREFIX):])


def validate(experiment_dir: Path, model_dir: Path) -> dict:
    response_path = experiment_dir / "response.xlsx"
    dataset_dir = experiment_dir / "haw60_self_attention_dataset"
    raw_path = dataset_dir / "haw60_matched_engineered_raw.csv"
    standardized_path = dataset_dir / "haw60_matched_standardized.csv"
    worker_path = Path(__file__).resolve().parent / "force_predictor_worker.py"

    headers, response_rows = _read_response(response_path)
    columns = {name: index for index, name in enumerate(headers)}
    raw_rows = _read_csv(raw_path)
    standardized_rows = _read_csv(standardized_path)
    if len(raw_rows) != len(standardized_rows):
        raise ValueError("Raw and standardized datasets have different row counts")

    workbook_map = []
    for channel, label in enumerate(CHANNEL_LABELS, start=1):
        workbook_map.extend([
            (f"ch{channel}_P_kPa", f"P_{label}_kPa"),
            (f"ch{channel}_L_mm", f"L_{label}_mm"),
            (f"ch{channel}_d_active_mm", f"d_active_{label}_mm"),
            (f"ch{channel}_d_passive_mm", f"d_passive_{label}_mm"),
        ])
    max_workbook_difference = {raw_name: 0.0 for raw_name, _ in workbook_map}
    max_workbook_difference.update({
        name: 0.0 for name in (
            "response_time_s", "X_deg", "Y_deg", "phi_deg",
            "X_speed_deg_s", "Y_speed_deg_s", "phi_speed_deg_s", "F_X_N",
        )
    })
    for raw_row in raw_rows:
        row_index = int(raw_row["row_index"])
        if row_index < 0 or row_index >= len(response_rows):
            raise ValueError(f"row_index {row_index} is outside response.xlsx")
        for raw_name, workbook_name in workbook_map:
            difference = abs(
                _number(raw_row[raw_name])
                - _response_value(response_rows, columns, row_index, workbook_name)
            )
            max_workbook_difference[raw_name] = max(
                max_workbook_difference[raw_name], difference
            )
        for name in (
            "response_time_s", "X_deg", "Y_deg", "phi_deg",
            "X_speed_deg_s", "Y_speed_deg_s", "phi_speed_deg_s", "F_X_N",
        ):
            difference = abs(
                _number(raw_row[name])
                - _response_value(response_rows, columns, row_index, name)
            )
            max_workbook_difference[name] = max(
                max_workbook_difference[name], difference
            )

    scaler_rows = _read_csv(model_dir / "train_scaler_stats.csv")
    scaler = {
        row["feature"]: (_number(row["train_mean"]), _number(row["train_std"]))
        for row in scaler_rows
    }
    max_standardized_difference = 0.0
    max_derivative_difference = 0.0
    previous_by_process: dict[int, tuple[float, list[float], list[float]]] = {}
    for raw_row, standardized_row in zip(raw_rows, standardized_rows):
        if raw_row["row_index"] != standardized_row["row_index"]:
            raise ValueError("Raw and standardized row_index values are not aligned")
        process_id = int(raw_row["process_id"])
        timestamp = _number(raw_row["response_time_s"])
        pressures = [_number(raw_row[f"ch{channel}_P_kPa"]) for channel in range(1, 7)]
        lengths = [_number(raw_row[f"ch{channel}_L_mm"]) for channel in range(1, 7)]
        previous = previous_by_process.get(process_id)
        if previous is None:
            pressure_rates = [0.0] * 6
            length_rates = [0.0] * 6
        else:
            previous_time, previous_pressures, previous_lengths = previous
            dt = timestamp - previous_time
            if dt <= 0:
                raise ValueError(f"Non-increasing time in process {process_id}")
            pressure_rates = [
                (current-old) / dt for current, old in zip(pressures, previous_pressures)
            ]
            length_rates = [
                (current-old) / dt for current, old in zip(lengths, previous_lengths)
            ]
        previous_by_process[process_id] = (timestamp, pressures, lengths)
        for channel in range(1, 7):
            max_derivative_difference = max(
                max_derivative_difference,
                abs(pressure_rates[channel-1] - _number(
                    raw_row[f"ch{channel}_dP_dt_kPa_s"])),
                abs(length_rates[channel-1] - _number(
                    raw_row[f"ch{channel}_dL_dt_mm_s"])),
            )
            raw_values = (
                pressures[channel-1],
                lengths[channel-1],
                _number(raw_row[f"ch{channel}_d_active_mm"]),
                _number(raw_row[f"ch{channel}_d_passive_mm"]),
                pressure_rates[channel-1],
                length_rates[channel-1],
            )
            for token, value in zip(FEATURE_TOKENS, raw_values):
                feature_name = f"ch{channel}_{token}"
                mean, std = scaler[feature_name]
                calculated = (value-mean) / std
                stored = _number(standardized_row[feature_name + "_z"])
                max_standardized_difference = max(
                    max_standardized_difference, abs(calculated-stored)
                )

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    for relative_path, expected_hash in manifest["weights_sha256"].items():
        actual_hash = hashlib.sha256((model_dir / relative_path).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Model hash mismatch: {relative_path}")

    process = subprocess.Popen(
        [sys.executable, "-u", str(worker_path), "--model-dir", str(model_dir)],
        cwd=str(worker_path.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    predictions = []
    targets = []
    sample_count = 0
    generation = 0
    previous_process = None
    try:
        ready = _worker_message(process)
        if ready.get("state") != "worker_ready":
            raise RuntimeError(f"Force worker did not start: {ready}")
        for raw_row in raw_rows:
            process_id = int(raw_row["process_id"])
            if raw_row["split"] != "test" or process_id in set(
                manifest.get("excluded_test_processes", [])
            ):
                continue
            if process_id != previous_process:
                generation += 1
                previous_process = process_id
            row_index = int(raw_row["row_index"])
            request = {
                "reset_generation": generation,
                "timestamp_s": _number(raw_row["response_time_s"]),
                "pressure_kpa": [
                    _number(raw_row[f"ch{channel}_P_kPa"])
                    for channel in range(1, 7)
                ],
                "length_mm": [
                    _number(raw_row[f"ch{channel}_L_mm"])
                    for channel in range(1, 7)
                ],
                "active_mm": [
                    _number(raw_row[f"ch{channel}_d_active_mm"])
                    for channel in range(1, 7)
                ],
                "passive_mm": [
                    _number(raw_row[f"ch{channel}_d_passive_mm"])
                    for channel in range(1, 7)
                ],
            }
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
            result = _worker_message(process)
            if result.get("state") != "ready":
                raise RuntimeError(f"Prediction failed at row {row_index}: {result}")
            predictions.append(result["force_xyz_n"])
            targets.append([
                _response_value(response_rows, columns, row_index, "F_X_N"),
                _response_value(response_rows, columns, row_index, "F_Y_N"),
                _response_value(response_rows, columns, row_index, "F_Z_N"),
            ])
            sample_count += 1
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('{"command":"stop"}\n')
                process.stdin.flush()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)

    predicted_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    rmse = np.sqrt(np.mean((predicted_array-target_array) ** 2, axis=0))
    expected = manifest["test_rmse_n_excluding_process_25"]
    result = {
        "workbook_rows": len(response_rows),
        "matched_rows": len(raw_rows),
        "six_channel_input_max_abs_difference": float(max(
            max_workbook_difference[name] for name, _ in workbook_map
        )),
        "workbook_signal_max_abs_difference": float(max(max_workbook_difference.values())),
        "derivative_max_abs_difference": float(max_derivative_difference),
        "standardized_feature_max_abs_difference": float(max_standardized_difference),
        "worker_test_samples": sample_count,
        "worker_test_rmse_n": {
            "fx": float(rmse[0]), "fy": float(rmse[1]), "fz": float(rmse[2]),
            "macro": float(np.mean(rmse)),
        },
        "manifest_test_rmse_n": expected,
    }
    if result["six_channel_input_max_abs_difference"] > 1e-8:
        raise ValueError("Six-channel workbook inputs do not match the model dataset")
    if result["derivative_max_abs_difference"] > 1e-6:
        raise ValueError("Live derivative reconstruction does not match the model dataset")
    if result["standardized_feature_max_abs_difference"] > 1e-6:
        raise ValueError("Live 36D standardized features do not match the model dataset")
    for axis in ("fx", "fy", "fz"):
        if abs(result["worker_test_rmse_n"][axis] - _number(expected[axis])) > 0.01:
            raise ValueError(f"End-to-end {axis} RMSE does not reproduce the manifest")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a recorded experiment through the GUI force-prediction worker"
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "control" / "models" / "force_estimator_v1",
    )
    args = parser.parse_args()
    result = validate(args.experiment_dir.resolve(), args.model_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

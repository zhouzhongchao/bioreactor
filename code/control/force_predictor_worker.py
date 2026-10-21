from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from force_predictor import ForceEstimator


PREFIX = "FORCE_PREDICTION "


def emit(payload) -> None:
    print(PREFIX + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-mode", choices=("fusion", "separate"), default="fusion")
    args = parser.parse_args()
    try:
        estimator = ForceEstimator(args.model_dir, deployment_mode=args.prediction_mode)
    except Exception as exc:
        emit({"state": "error", "message": "Model loading failed: " + str(exc)})
        return 2
    emit({
        "state": "worker_ready", "model_version": estimator.model_version,
        "prediction_mode": args.prediction_mode,
    })
    generation = None
    recent_times = []
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            command = str(request.get("command", "predict"))
            if command == "stop":
                return 0
            if command == "reset":
                estimator.reset_state()
                generation = request.get("reset_generation")
                recent_times = []
                emit({"state": "waiting", "reset_generation": generation})
                continue
            next_generation = int(request.get("reset_generation", 0))
            if next_generation != generation:
                estimator.reset_state()
                generation = next_generation
                recent_times = []
            result = estimator.predict(
                request["pressure_kpa"], request["length_mm"],
                request["active_mm"], request["passive_mm"],
                request["timestamp_s"])
            recent_times.append(float(request["timestamp_s"]))
            recent_times = recent_times[-20:]
            result["hz"] = (
                (len(recent_times)-1)/(recent_times[-1]-recent_times[0])
                if len(recent_times) >= 2 and recent_times[-1] > recent_times[0]
                else 0.0
            )
            result["reset_generation"] = generation
            emit(result)
        except Exception as exc:
            emit({
                "state": "error", "reset_generation": generation,
                "message": "Prediction request failed: " + str(exc),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

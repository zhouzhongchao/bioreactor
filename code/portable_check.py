from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "control"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CONTROL))
from project_paths import DATA_ROOT


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        fail(f"Python 3.10 is required; found {sys.version.split()[0]}")

    modules = (
        "numpy", "scipy", "matplotlib", "PyQt5", "PyQt5.QtWebEngineWidgets",
        "serial", "openpyxl", "imageio_ffmpeg", "pyqtgraph", "torch", "osqp",
    )
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            fail(f"cannot import {name}: {exc}")

    required_files = (
        ROOT / "pam_shoulder_gui.html",
        ROOT / "shoulder_pam_model.py",
        ROOT / "shoulder_pam_model_v6.py",
        CONTROL / "qt_gui.py",
        CONTROL / "mpc_station.py",
        CONTROL / "codex_compensator.py",
        CONTROL / "codex_compensator_prompt.json",
        CONTROL / "codex_compensator_candidate_schema.json",
        CONTROL / "gains.json",
        CONTROL / "params.json",
        CONTROL / "pure_mpc_model.pt",
        CONTROL / "pure_mpc_model.json",
    )
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"required file is missing or empty: {path.relative_to(ROOT)}")

    for path in (CONTROL / "gains.json", CONTROL / "params.json",
                 CONTROL / "pure_mpc_model.json"):
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")

    try:
        import torch
        for model_name in ("pure_mpc_model.pt", "pure_mpc_history_mlp.pt", "pure_mpc_gru.pt"):
            model_path = CONTROL / model_name
            if model_path.is_file():
                torch.jit.load(str(model_path), map_location="cpu").eval()
    except Exception as exc:
        fail(f"MPC neural-network model could not be loaded: {exc}")

    try:
        import qt_gui  # noqa: F401
        import mpc_station  # noqa: F401
        import codex_compensator  # noqa: F401
    except Exception as exc:
        fail(f"application module import failed: {exc}")

    recording_dir = DATA_ROOT / "output" / "recordings"
    try:
        recording_dir.mkdir(parents=True, exist_ok=True)
        probe = recording_dir / ".portable_write_test"
        probe.write_bytes(b"ok")
        probe.unlink()
    except Exception as exc:
        fail(f"recording directory is not writable: {exc}")

    print("[PASS] Musculoskeletal Robotic Shoulder Bioreactor portable environment is ready.")
    print("       Hardware was not connected or commanded by this check.")


if __name__ == "__main__":
    main()

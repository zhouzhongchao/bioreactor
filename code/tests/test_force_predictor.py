from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
from openpyxl import load_workbook
from control.qt_gui import SerialBridge


MODEL_DIR = (Path(__file__).resolve().parents[1] / "control" / "models" /
             "force_estimator_v1")


def _run_predictor_code(source):
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_copied_force_ensemble_reproduces_training_predictions():
    standardized_feature = [
        -0.4958699138, 0.4907512118, 0.6863265731, -1.3740687849,
        0.0030470379, 0.0023936981, -0.4925651061, 0.3556269207,
        0.2774215750, -0.9824067204, 0.0041724172, 0.0036206108,
        -0.5488573722, 0.7344219570, 0.4913424307, -0.6650365604,
        0.0008925905, 0.0052768282, -0.5150462574, 0.7465342310,
        0.7265023870, 0.0090563376, 0.0028952009, -0.0041424533,
        -0.4870959772, 0.2896315600, 0.2429206492, -0.2611872354,
        0.0062996951, 0.0177536810, -0.4390641392, -0.1778958099,
        -0.0568867086, -0.2825343910, 0.0012058966, -0.0031773604,
    ]
    source = (
        "import json; from control.force_predictor import ForceEstimator; "
        f"p=ForceEstimator({str(MODEL_DIR)!r}); "
        f"print(json.dumps(p.predict_standardized({standardized_feature!r}).tolist()))"
    )
    predicted = _run_predictor_code(source)

    np.testing.assert_allclose(
        predicted,
        [0.73130053, -0.67264081, 0.35719276],
        rtol=0.0,
        atol=5e-5,
    )


def test_live_feature_state_resets_derivatives_and_model_history():
    source = (
        "import json; from control.force_predictor import ForceEstimator; "
        f"p=ForceEstimator({str(MODEL_DIR)!r}); "
        "a=p.predict([0]*6,[223,221,224,231,215,228],[0]*6,[0]*6,10.0); "
        "b=p.predict([1]*6,[222.9,220.9,223.9,230.9,214.9,227.9],[-0.1]*6,[0]*6,10.05); "
        "p.reset_state(); "
        "c=p.predict([1]*6,[222.9,220.9,223.9,230.9,214.9,227.9],[0]*6,[0]*6,20.0); "
        "print(json.dumps([a,b,c]))"
    )
    first, second, after_reset = _run_predictor_code(source)

    assert first["history_samples"] == 1
    assert second["history_samples"] == second["window_size"] == 1
    assert after_reset["history_samples"] == 1
    assert np.all(np.isfinite(after_reset["force_xyz_n"]))


def test_serial_bridge_runs_prediction_off_the_gui_thread():
    bridge = SerialBridge()
    frame = {
        "wall": time.time(),
        "kpa": np.zeros(6, dtype=float),
        "channel_length_mm": [235.0] * 6,
        "lens": [0.0] * 20,
    }
    try:
        bridge._force_prediction_on_frame(frame)
        deadline = time.time() + 10.0
        payload = {}
        while time.time() < deadline:
            payload = json.loads(bridge.forcePredictionStatus())
            if payload.get("state") in ("ready", "error"):
                break
            time.sleep(0.02)
        assert payload.get("state") == "ready", payload
        assert len(payload["force_xyz_n"]) == 3
        assert np.all(np.isfinite(payload["force_xyz_n"]))
        assert payload["history_samples"] == 1
    finally:
        bridge.shutdownForcePrediction()


def test_gui_force_prediction_uses_physical_channel_order():
    bridge = SerialBridge()
    bridge._ensure_force_prediction_thread = lambda: None
    first_frame = {
        "wall": 100.0,
        "kpa": np.asarray([10, 20, 30, 40, 50, 60], dtype=float),
        "channel_length_mm": [210, 220, 230, 240, 250, 260],
        "lens": [0.0] * 20,
    }
    second_frame = {
        "wall": 100.05,
        "kpa": np.asarray([11, 21, 31, 41, 51, 61], dtype=float),
        "channel_length_mm": [211, 221, 231, 241, 251, 261],
        "lens": [float(10 + index) for index in range(10)]
        + [float(-20 - index) for index in range(10)],
    }
    bridge._force_prediction_on_frame(first_frame)
    bridge._force_prediction_queue.get_nowait()
    bridge._force_prediction_on_frame(second_frame)
    sample = bridge._force_prediction_queue.get_nowait()

    assert sample["pressure_kpa"] == [11.0, 31.0, 21.0, 41.0, 51.0, 61.0]
    assert sample["length_mm"] == [211.0, 231.0, 221.0, 241.0, 251.0, 261.0]
    np.testing.assert_allclose(sample["active_mm"], [12.0, 11.5, 15.0, 15.5, 18.0, 19.0])
    np.testing.assert_allclose(sample["passive_mm"], [-22.0, -21.5, -25.0, -25.5, -28.0, -29.0])


def test_force_recording_saves_prediction_and_nano25_measurement(tmp_path):
    bridge = SerialBridge()
    bridge._recording_base_dir = str(tmp_path)
    bridge._recording_directory_confirmed = True
    bridge._recording_session = {"token": "test", "folder": str(tmp_path)}
    started = json.loads(bridge.startForceRecording())
    assert started["ok"] is True
    bridge._emit_nano25({
        "state": "running", "received_wall": time.time(),
        "force_n": [1.0, 2.0, 3.0], "torque_nm": [0.1, 0.2, 0.3],
    })
    bridge._emit_force_prediction({
        "state": "ready", "timestamp_s": 100.0,
        "force_xyz_n": [4.0, 5.0, 6.0], "hz": 20.0,
        "latency_ms": 3.5, "model_version": "test-vib",
    })
    stopped = json.loads(bridge.stopForceRecording())
    assert stopped["ok"] is True
    path = Path(stopped["path"])
    deadline = time.time() + 5.0
    while not path.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert path.exists()
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook["Force"].iter_rows(values_only=True))
    workbook.close()
    assert rows[0] == SerialBridge.FORCE_RECORD_HEADERS
    assert rows[1][:10] == (
        0.0, 4.0, 5.0, 6.0, -3.0, -1.0, -2.0, 0.1, 0.2, 0.3,
    )
    assert rows[1][12] == "test-vib"


def test_force_save_uses_visible_nano25_stream_without_prior_task_recording(tmp_path):
    bridge = SerialBridge()
    bridge._recording_base_dir = str(tmp_path)
    bridge._recording_directory_confirmed = True
    bridge._emit_force_prediction({
        "state": "ready", "timestamp_s": 100.0,
        "force_xyz_n": [4.0, 5.0, 6.0], "hz": 20.0,
        "latency_ms": 3.5, "model_version": "test-vib",
    })
    bridge._emit_nano25({
        "state": "running", "received_wall": 200.0,
        "force_n": [1.0, 2.0, 3.0], "torque_nm": [0.1, 0.2, 0.3],
    })

    saved = json.loads(bridge.saveForceSnapshot())

    assert saved["ok"] is True
    assert saved["rows"] == 1
    path = Path(saved["path"])
    deadline = time.time() + 5.0
    while not path.exists() and time.time() < deadline:
        time.sleep(0.02)
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook["Force"].iter_rows(values_only=True))
    workbook.close()
    assert rows[1][:10] == (
        0.0, 4.0, 5.0, 6.0, -3.0, -1.0, -2.0, 0.1, 0.2, 0.3,
    )

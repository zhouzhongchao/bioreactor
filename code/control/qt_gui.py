# -*- coding: utf-8 -*-
"""
qt_gui.py — 桌面入口：把 pam_shoulder_gui.html 原封不动嵌进 Python 窗口

外观 100% 就是网页版（同一个文件）；差别只在水面下：
  串口不再走浏览器的 Web Serial，而是本进程的 pyserial，经 QWebChannel
  桥接给页面（页面里 pyHost 非空时自动切换到这条通道，见 HTML 的宿主适配段）。
好处：不再依赖 Chrome、没有标签页降频、串口归本进程管；MPC 照旧由
  `python mpc_station.py bridge` 提供（纯计算不占串口，页面 HTTP 调它）。

运行：python qt_gui.py
"""
import base64, csv, importlib.metadata, importlib.util, json, os, queue, shutil, subprocess, sys, threading, time
from collections import deque
from datetime import datetime

CONTROL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONTROL_DIR)
for _import_dir in (CONTROL_DIR, PROJECT_ROOT):
    if _import_dir not in sys.path:
        sys.path.insert(0, _import_dir)

from project_paths import DATA_ROOT, CONTROL_LOG_DIR

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView, QWebEngineSettings
from PyQt5.QtWebChannel import QWebChannel
import serial as pyserial
from serial.tools import list_ports

HTML = os.path.join(PROJECT_ROOT, "pam_shoulder_gui.html")
ADC_CAL_PATH = os.path.join(os.path.dirname(__file__), "adc_calibration.json")
SPEED_LIMIT_PATH = os.path.join(os.path.dirname(__file__), "speed_limits.json")
X_COMP_PATH = os.path.join(os.path.dirname(__file__), "x_compensation.json")
FIXED_TASK_PATH = os.path.join(os.path.dirname(__file__), "important_task_trajectory.json")
FIXED_TASK_2_PATH = os.path.join(os.path.dirname(__file__), "important_task_trajectory_2.json")
FIXED_TASK_3_PATH = os.path.join(os.path.dirname(__file__), "important_task_trajectory_3.json")
GAINS_PATH = os.path.join(os.path.dirname(__file__), "gains.json")
V6_GEOMETRY_CANDIDATE_PATH = os.path.join(
    os.path.dirname(__file__), "v6_geometry_calibration_candidate.json")
QUICK_HEALTH_SPEED_CANDIDATE_PATH = os.path.join(
    os.path.dirname(__file__), "quick_health_speed_candidate.json")
RECORDINGS_DIR = str(DATA_ROOT / "output" / "recordings")
LLM_RECORDING_SESSION_ENV = "TRO_RECORDING_SESSION_DIR"
NANO25_RECORDER_PATH = os.path.join(
    PROJECT_ROOT, "setup_for_shuai", "src", "nano25_acquisition_py",
    "record_nano25.py")
NANO25_RECORDINGS_DIR = str(DATA_ROOT / "nano25_recordings")
NANO25_DATA_PREFIX = "NANO25_DATA "
NANO25_EVENT_PREFIX = "NANO25_EVENT "
FORCE_PREDICTOR_MODEL_DIR = os.path.join(
    CONTROL_DIR, "models", "force_estimator_v1")
FORCE_PREDICTOR_WORKER_PATH = os.path.join(
    CONTROL_DIR, "force_predictor_worker.py")
FORCE_PREDICTION_PREFIX = "FORCE_PREDICTION "
DEFAULT_MAX_OUTPUT_MV = 2200
OPEN_LOOP_MAX_OUTPUT_MV = 3000
X_COMP_TEST_MV = 800
X_COMP_TEST_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
PHI_GRAD_LEVELS_MV = (600, 1200, 1800)
REACHABILITY_MAX_KPA = 200.0
REACHABILITY_EXTREMA_MV = 2200.0
REACHABILITY_MIN_SAMPLES = 6
REACHABILITY_MAX_SAMPLES = 120
REACHABILITY_PRELOCK_S = 0.100
REACHABILITY_SETTLE_S = 0.75
REACHABILITY_POINT_TIMEOUT_S = 10.0
REACHABILITY_VENT_STABLE_S = 1.0
REACHABILITY_PRESSURE_TOL_KPA = 5.0
REACHABILITY_PRESSURE_ABORT_KPA = 215.0
REACHABILITY_ARM_LENGTH_MM = 216.0
SLOW_ROM_SLEW_MV_PER_S = 100.0
SLOW_ROM_BACKOFF_MV_PER_S = 300.0
SLOW_ROM_SPEED_HOLD_DEG_S = 3.0
SLOW_ROM_SPEED_BACKOFF_DEG_S = 5.0
SLOW_ROM_POINT_TIMEOUT_S = 45.0
SLOW_ROM_ZERO_TOL_DEG = 3.0
SLOW_ROM_RECENTER_PULSE_S = 0.25
SLOW_ROM_RECENTER_MIN_MV = 250.0
SLOW_ROM_RECENTER_MAX_MV = 500.0
SLOW_ROM_DEFLATE_SLEW_MV_PER_S = 100.0
MANUAL_PRESSURE_DEFLATE_SLEW_MV_PER_S = 100.0
MANUAL_PRESSURE_UNLOCK_KPA = 30.0
SLOW_ROM_MODES = ("slow_rom", "slow_rom_gradual")
REACHABILITY_CHANNEL_INFO = (
    {"channel": 0, "muscles": "1+5", "direction": "Y+"},
    {"channel": 1, "muscles": "4+8", "direction": "Y-"},
    {"channel": 2, "muscles": "2+3", "direction": "phi+"},
    {"channel": 3, "muscles": "6+7", "direction": "phi-"},
    {"channel": 4, "muscles": "9", "direction": "X-"},
    {"channel": 5, "muscles": "10", "direction": "X+"},
)
# Empirical envelope from the current speed tests.  It is a test abort boundary,
# not a model constraint and not a target angle.
REACHABILITY_POSE_MIN_DEG = np.array([-20.0, -38.0, -70.0])
REACHABILITY_POSE_MAX_DEG = np.array([20.0, 80.0, 75.0])
# Plot-only copy of the confirmed mechanical envelope.  Historical dynamic
# overshoot outside these limits is not labelled as safely reachable.
THEORETICAL_REACHABILITY_POSE_MIN_DEG = np.array([-20.0, -38.0, -70.0])
THEORETICAL_REACHABILITY_POSE_MAX_DEG = np.array([20.0, 80.0, 75.0])
PHI_GRAD_CHANNELS = (
    {"name": "23", "ch": 2, "expected_sign": 1.0},
    {"name": "67", "ch": 3, "expected_sign": -1.0},
    {"name": "9", "ch": 4, "expected_sign": 1.0},
    {"name": "10", "ch": 5, "expected_sign": -1.0},
)
X_COMP_TESTS = {
    "9_67": {"label": "9+67", "primary": 4, "comp": 3, "x_sign": -1.0,
             "save_key": "9_plus_67"},
    "10_23": {"label": "10+23", "primary": 5, "comp": 2, "x_sign": 1.0,
              "save_key": "10_plus_23"},
}
MPC_DATA_LEVELS_MV = (400, 1000, 1600, 2200, 800)
MPC_DATA_PAIR_LEVELS_MV = ((500, 900), (1200, 600), (1800, 1400),
                           (2200, 2200), (800, 1600))
MPC_DATA_TRIPLE_LEVELS_MV = ((400, 700, 1000), (900, 500, 1400),
                             (1400, 1800, 800), (2200, 2200, 2200),
                             (800, 1400, 1800))
SPEED_DIRECTIONS = (
    {"name": "y+", "ch": 0, "axis": 1, "sign": 1.0},
    {"name": "y-", "ch": 1, "axis": 1, "sign": -1.0},
    {"name": "φ+", "ch": 2, "axis": 2, "sign": 1.0},
    {"name": "φ-", "ch": 3, "axis": 2, "sign": -1.0},
    {"name": "x-", "ch": 4, "axis": 0, "sign": -1.0},
    {"name": "x+", "ch": 5, "axis": 0, "sign": 1.0},
)
SPEED_AXIS_GROUPS = {
    "y": tuple(d for d in SPEED_DIRECTIONS if d["name"].startswith("y")),
    "phi": tuple(d for d in SPEED_DIRECTIONS if d["name"].startswith("φ")),
    "x": tuple(d for d in SPEED_DIRECTIONS if d["name"].startswith("x")),
}
SPEED_AXIS_LABELS = {"y": "Y", "phi": "φ", "x": "X"}


def configured_max_output_mv():
    """Read the single user-adjustable voltage ceiling from gains.json."""
    try:
        with open(GAINS_PATH, encoding="utf-8") as f:
            value = float(json.load(f).get("u_max", DEFAULT_MAX_OUTPUT_MV))
        return int(np.clip(value, 100.0, OPEN_LOOP_MAX_OUTPUT_MV))
    except Exception:
        return DEFAULT_MAX_OUTPUT_MV


def parse_nano25_stream_line(line):
    """Decode one machine-readable line produced by record_nano25.py."""
    text = str(line or "").strip()
    for prefix in (NANO25_DATA_PREFIX, NANO25_EVENT_PREFIX):
        if text.startswith(prefix):
            payload = json.loads(text[len(prefix):])
            if not isinstance(payload, dict):
                raise ValueError("Nano25 stream payload must be an object")
            return payload
    return None


class SerialBridge(QtCore.QObject):
    """页面里的 pyHost。约定：open/close/write 三个动作 + rx 信号推数据 + lost 失联信号。"""
    rx = QtCore.pyqtSignal(str)
    lost = QtCore.pyqtSignal(str)                    # 串口失联（拔线/驱动异常），页面据此停控
    notice = QtCore.pyqtSignal(str)                  # 非致命串口提醒（例如连接后持续无数据）
    theoretical_reachability_ready = QtCore.pyqtSignal(str)
    analysis_image_ready = QtCore.pyqtSignal(str)
    debug_mode_changed = QtCore.pyqtSignal(bool)
    recording_status = QtCore.pyqtSignal(str)
    codex_auth_status = QtCore.pyqtSignal(str)
    nano25_status = QtCore.pyqtSignal(str)
    force_prediction_status = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ser = None
        self._frame_buf = ""
        self._latest_frame = None
        self._dm_ident = None
        self._adc_cal = None
        self._speed_test = None
        self._x_comp_test = None
        self._mpc_data_collect = None
        self._phi_gradient_test = None
        self._quick_health_running = False
        self._manual_pressure_test = None
        self._reachability_test = None
        self._reachability_windows = []
        self._theoretical_reachability_running = False
        self._v6_log_analysis_running = False
        self._adc_cal_applied = False
        self._video_recordings = {}
        self._video_recording_lock = threading.Lock()
        self._recording_session = None
        self._recording_session_lock = threading.Lock()
        self._last_recording_session_folder = ""
        # A remembered directory is only used as the initial folder shown by
        # the native picker.  Recording must be explicitly armed by choosing a
        # directory once in the current application session, so data never
        # disappears into an implicit/default location.
        self._recording_directory_confirmed = False
        saved_recording_dir = str(QtCore.QSettings(
            "TRO", "PAMShoulder").value("recordingDirectory", RECORDINGS_DIR))
        self._recording_base_dir = (os.path.abspath(saved_recording_dir)
                                    if saved_recording_dir else RECORDINGS_DIR)
        if not os.path.isdir(self._recording_base_dir):
            self._recording_base_dir = RECORDINGS_DIR
        self._native_recorders = {}
        # The visible WebEngine page can switch to the engineering/debug UI.
        # GUI video is an operation-view record, so the native recorder uses
        # this state to omit debug-only frames without ending the recording.
        self._debug_mode = False
        self._data_recording = None
        self._data_recording_lock = threading.Lock()
        # Keep the most recent response buffer available after execution stops,
        # so the operation-mode Save button can always overwrite response.xlsx
        # with the latest complete CONTROL RESPONSE snapshot.
        self._last_data_recording = None
        self._data_save_lock = threading.Lock()
        self._data_save_generation = 0
        # CONTROL RESPONSE is a global live view, not a controller-only view.
        # The browser sends its already-zeroed GUI X/Y/phi pose at 20 Hz.  When
        # MPC is not writing TELEM, these samples keep the same plot/save buffer
        # alive in Automatic, Teaching, Test and Debug modes alike.
        self._live_response_lock = threading.Lock()
        self._live_response_last_wall = 0.0
        self._environment_installing = False
        self._environment_install_lock = threading.Lock()
        self._codex_auth_busy = False
        self._codex_auth_lock = threading.Lock()
        self._nano25_lock = threading.Lock()
        self._nano25_process = None
        self._nano25_thread = None
        self._nano25_stop_requested = False
        self._nano25_control_required = False
        self._nano25_latest = {
            "state": "stopped", "hz": 0.0,
            "force_n": [0.0, 0.0, 0.0],
            "torque_nm": [0.0, 0.0, 0.0],
            "received_wall": None,
            "control_required": False,
            "message": "Nano25 is stopped.",
        }
        self._force_prediction_lock = threading.Lock()
        self._force_prediction_queue = queue.Queue(maxsize=1)
        self._force_prediction_zero_mm = None
        self._force_prediction_reset_generation = 0
        self._force_prediction_last_submit_wall = 0.0
        saved_prediction_mode = str(QtCore.QSettings(
            "TRO", "PAMShoulder").value("forcePredictionMode", "fusion"))
        self._force_prediction_mode = (
            "separate" if saved_prediction_mode == "separate" else "fusion")
        self._force_prediction_latest = {
            "state": "waiting",
            "force_xyz_n": [0.0, 0.0, 0.0],
            "hz": 0.0,
            "latency_ms": 0.0,
            "prediction_mode": self._force_prediction_mode,
            "message": "Waiting for six-channel pressure and length data.",
        }
        # Load PyTorch lazily after the first complete hardware frame.  This
        # keeps startup light and avoids creating idle worker threads in tests.
        self._force_prediction_thread = None
        self._force_prediction_process = None
        self._force_recording = None
        self._last_force_recording = None
        # Keep the latest hour of the already-displayed 20 Hz Nano25 stream.
        # This lets the Save button export Force even when it was armed after
        # the task began, instead of incorrectly reporting that no data exist.
        self._force_recent_rows = deque(maxlen=20 * 60 * 60)
        self._force_recent_last_timestamp_s = float("-inf")
        self._force_recording_lock = threading.Lock()
        self._force_save_lock = threading.Lock()
        self._force_save_generation = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.theoretical_reachability_ready.connect(self._show_reachability_plot)
        self.analysis_image_ready.connect(self._show_reachability_plot)

    @QtCore.pyqtSlot(result=str)
    def listPorts(self):
        try:                                         # 槽内未捕获异常会 qFatal 整个进程
            return json.dumps([p.device for p in list_ports.comports()])
        except Exception:
            return "[]"

    @staticmethod
    def _load_fixed_task_file(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            pts = blob.get("points", [])
            if not pts:
                raise ValueError("固定任务没有轨迹点")
            return json.dumps(blob, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def loadFixedTask(self):
        """Return Default 1, the original reproducible controller benchmark."""
        return self._load_fixed_task_file(FIXED_TASK_PATH)

    @QtCore.pyqtSlot(result=str)
    def loadFixedTask2(self):
        """Return Default 2, the Y-only +30 to -20 degree benchmark."""
        return self._load_fixed_task_file(FIXED_TASK_2_PATH)

    @QtCore.pyqtSlot(result=str)
    def loadFixedTask3(self):
        """Return Default 3: Y reaches +40 degrees before X reaches -10."""
        return self._load_fixed_task_file(FIXED_TASK_3_PATH)

    def _emit_nano25(self, payload):
        update = dict(payload or {})
        with self._nano25_lock:
            merged = dict(self._nano25_latest)
            merged.update(update)
            self._nano25_latest = merged
        self._append_recent_force_sample(merged)
        text = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
        self.nano25_status.emit(text)
        return text

    def _emit_force_prediction(self, payload):
        update = dict(payload or {})
        with self._force_prediction_lock:
            merged = dict(self._force_prediction_latest)
            merged.update(update)
            self._force_prediction_latest = merged
        self._append_force_recording(merged)
        text = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
        self.force_prediction_status.emit(text)
        return text

    @QtCore.pyqtSlot(result=str)
    def forcePredictionStatus(self):
        with self._force_prediction_lock:
            payload = dict(self._force_prediction_latest)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @QtCore.pyqtSlot(str, result=str)
    def setForcePredictionMode(self, mode):
        normalized = "separate" if str(mode).strip().lower() == "separate" else "fusion"
        with self._force_prediction_lock:
            changed = normalized != self._force_prediction_mode
            self._force_prediction_mode = normalized
            process = self._force_prediction_process if changed else None
        QtCore.QSettings("TRO", "PAMShoulder").setValue(
            "forcePredictionMode", normalized)
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        label = "三轴分开模型" if normalized == "separate" else "Fusion单网络"
        return self._reset_force_prediction_state(
            f"Prediction mode: {label}; waiting for the next valid frame.")

    def _reset_force_prediction_state(self, message=None):
        self._force_prediction_zero_mm = None
        self._force_prediction_last_submit_wall = 0.0
        with self._force_prediction_lock:
            self._force_prediction_reset_generation += 1
        while True:
            try:
                self._force_prediction_queue.get_nowait()
            except queue.Empty:
                break
        return self._emit_force_prediction({
            "state": "waiting", "force_xyz_n": [0.0, 0.0, 0.0],
            "hz": 0.0, "latency_ms": 0.0,
            "history_samples": 0,
            "window_size": 16 if self._force_prediction_mode == "separate" else 1,
            "prediction_mode": self._force_prediction_mode,
            "message": message or "Prediction zero reset; waiting for the next valid frame.",
        })

    @QtCore.pyqtSlot(result=str)
    def resetForcePrediction(self):
        return self._reset_force_prediction_state()

    FORCE_RECORD_HEADERS = (
        "force_time_s",
        "predicted_FX_N", "predicted_FY_N", "predicted_FZ_N",
        "measured_FX_N", "measured_FY_N", "measured_FZ_N",
        "measured_TX_Nm", "measured_TY_Nm", "measured_TZ_Nm",
        "prediction_hz", "prediction_latency_ms", "model_version",
    )

    def _append_recent_force_sample(self, nano):
        """Cache each real Nano25 display sample independently of task arming."""
        if nano.get("state") not in ("running", "reconnecting"):
            return
        raw_force = self._record_vector(nano.get("force_n"), 3)
        measured_torque = self._record_vector(nano.get("torque_nm"), 3)
        if (any(value is None for value in raw_force) or
                any(value is None for value in measured_torque)):
            return
        try:
            timestamp = float(nano.get("received_wall"))
        except (TypeError, ValueError):
            timestamp = time.time()
        if not np.isfinite(timestamp):
            timestamp = time.time()
        with self._force_prediction_lock:
            prediction = dict(self._force_prediction_latest)
        predicted = (self._record_vector(prediction.get("force_xyz_n"), 3)
                     if prediction.get("state") == "ready" else [None] * 3)
        measured_force = [-raw_force[2], -raw_force[0], -raw_force[1]]
        with self._force_recording_lock:
            if timestamp <= self._force_recent_last_timestamp_s:
                return
            self._force_recent_last_timestamp_s = timestamp
            self._force_recent_rows.append([
                timestamp, *predicted, *measured_force, *measured_torque,
                float(prediction.get("hz", 0.0) or 0.0),
                float(prediction.get("latency_ms", 0.0) or 0.0),
                str(prediction.get("model_version", "")),
            ])

    def _recent_force_record(self):
        if not self._force_recent_rows:
            return None
        rows = [list(row) for row in self._force_recent_rows]
        first_timestamp = float(rows[0][0])
        for row in rows:
            row[0] = float(row[0]) - first_timestamp
        return {"path": self._recording_path("force", "xlsx"), "rows": rows,
                "first_timestamp_s": first_timestamp,
                "last_timestamp_s": float(self._force_recent_rows[-1][0])}

    def _append_force_recording(self, prediction):
        if prediction.get("state") != "ready":
            return
        predicted = self._record_vector(prediction.get("force_xyz_n"), 3)
        if any(value is None for value in predicted):
            return
        with self._force_recording_lock:
            record = self._force_recording
            if record is None:
                return
            try:
                timestamp = float(prediction.get("timestamp_s"))
            except (TypeError, ValueError):
                timestamp = time.time()
            if not np.isfinite(timestamp):
                timestamp = time.time()
            if record["first_timestamp_s"] is None:
                record["first_timestamp_s"] = timestamp
            if timestamp <= record["last_timestamp_s"]:
                return
            record["last_timestamp_s"] = timestamp
            with self._nano25_lock:
                nano = dict(self._nano25_latest)
            try:
                nano_age = time.time()-float(nano.get("received_wall"))
            except (TypeError, ValueError):
                nano_age = float("inf")
            nano_valid = (nano.get("state") in ("running", "reconnecting")
                          and nano_age <= 0.2)
            raw_force = self._record_vector(
                nano.get("force_n") if nano_valid else None, 3)
            measured_force = (
                [-raw_force[2], -raw_force[0], -raw_force[1]]
                if all(value is not None for value in raw_force) else [None] * 3
            )
            measured_torque = self._record_vector(
                nano.get("torque_nm") if nano_valid else None, 3)
            record["rows"].append([
                timestamp-float(record["first_timestamp_s"]),
                *predicted, *measured_force, *measured_torque,
                float(prediction.get("hz", 0.0) or 0.0),
                float(prediction.get("latency_ms", 0.0) or 0.0),
                str(prediction.get("model_version", "")),
            ])

    @QtCore.pyqtSlot(result=str)
    def startForceRecording(self):
        try:
            importlib.import_module("openpyxl")
        except Exception as exc:
            return json.dumps({"ok": False, "error": "缺少 openpyxl：" + str(exc)},
                              ensure_ascii=False)
        with self._force_recording_lock:
            if self._force_recording is not None:
                return json.dumps({"ok": True, "path": self._force_recording["path"]},
                                  ensure_ascii=False)
            record = {
                "path": self._recording_path("force", "xlsx"),
                "rows": [], "first_timestamp_s": None,
                "last_timestamp_s": float("-inf"),
            }
            self._force_recording = record
            self._last_force_recording = record
        self._recording_emit("force", "recording", "Force数据记录中", record["path"])
        return json.dumps({"ok": True, "path": record["path"]}, ensure_ascii=False)

    @classmethod
    def _write_force_workbook(cls, rows, path):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        workbook = Workbook(write_only=False)
        sheet = workbook.active
        sheet.title = "Force"
        sheet.append(list(cls.FORCE_RECORD_HEADERS))
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="35546F")
        sheet.freeze_panes = "A2"
        for row in rows:
            sheet.append(list(row))
        sheet.auto_filter.ref = f"A1:M{len(rows)+1}"
        for index, heading in enumerate(cls.FORCE_RECORD_HEADERS, 1):
            sheet.column_dimensions[get_column_letter(index)].width = max(
                13, min(24, len(heading)+2))
        for cell in sheet["A"][1:]:
            cell.number_format = "0.000"
        tmp = path + ".tmp"
        workbook.save(tmp)
        os.replace(tmp, path)

    def _save_force_recording_worker(self, rows, path, generation):
        try:
            with self._force_save_lock:
                with self._force_recording_lock:
                    if generation < self._force_save_generation:
                        return
                self._write_force_workbook(rows, path)
            self._recording_emit(
                "force", "saved", f"Force Excel已保存，共{len(rows)}行", path)
        except Exception as exc:
            self._recording_emit("force", "error", "Force Excel保存失败：" + str(exc), path)

    def _queue_force_snapshot(self, record):
        if not record:
            return None
        rows = [list(row) for row in record.get("rows", [])]
        path = str(record.get("path", ""))
        if not rows or not path:
            return None
        self._force_save_generation += 1
        generation = self._force_save_generation
        threading.Thread(
            target=self._save_force_recording_worker,
            args=(rows, path, generation), daemon=True).start()
        return {"ok": True, "path": path, "rows": len(rows)}

    @QtCore.pyqtSlot(result=str)
    def saveForceSnapshot(self):
        with self._force_recording_lock:
            record = self._force_recording or self._last_force_recording
            result = self._queue_force_snapshot(record)
            if result is None:
                record = self._recent_force_record()
                if record is not None:
                    self._last_force_recording = record
                    result = self._queue_force_snapshot(record)
        if result is None:
            return json.dumps({"ok": False, "error": "当前没有可保存的Force数据"},
                              ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def stopForceRecording(self):
        with self._force_recording_lock:
            record = self._force_recording
            self._force_recording = None
            if record is not None:
                self._last_force_recording = record
            result = self._queue_force_snapshot(record)
        if result is None:
            return json.dumps({"ok": False, "error": "Force记录中没有有效数据"},
                              ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _force_prediction_python():
        candidates = [sys.executable]
        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            candidates.append(os.path.join(
                user_profile, ".conda", "envs", "bioreactor", "python.exe"))
        candidates.append(os.path.join(PROJECT_ROOT, ".python310", "python.exe"))
        candidates.append(str(DATA_ROOT / ".python310" / "python.exe"))
        checked = set()
        for executable in candidates:
            executable = os.path.abspath(executable)
            if executable.lower() in checked or not os.path.isfile(executable):
                continue
            checked.add(executable.lower())
            options = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                       "timeout": 15}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                completed = subprocess.run(
                    [executable, "-c", "import numpy, torch"], **options)
                if completed.returncode == 0:
                    return executable
            except Exception:
                continue
        return None

    @staticmethod
    def _read_force_prediction_message(process):
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.strip()
            if not line.startswith(FORCE_PREDICTION_PREFIX):
                continue
            return json.loads(line[len(FORCE_PREDICTION_PREFIX):])
        raise RuntimeError(
            f"Force-prediction worker exited with code {process.poll()}")

    def _start_force_prediction_process(self):
        executable = self._force_prediction_python()
        if executable is None:
            raise RuntimeError("PyTorch is not available in the GUI or bioreactor environment")
        if not os.path.isfile(FORCE_PREDICTOR_WORKER_PATH):
            raise FileNotFoundError(FORCE_PREDICTOR_WORKER_PATH)
        options = {
            "cwd": CONTROL_DIR, "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
            "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        with self._force_prediction_lock:
            prediction_mode = self._force_prediction_mode
        process = subprocess.Popen([
            executable, "-u", FORCE_PREDICTOR_WORKER_PATH,
            "--model-dir", FORCE_PREDICTOR_MODEL_DIR,
            "--prediction-mode", prediction_mode,
        ], **options)
        with self._force_prediction_lock:
            self._force_prediction_process = process
        ready = self._read_force_prediction_message(process)
        if ready.get("state") != "worker_ready":
            raise RuntimeError(ready.get("message") or "Force-prediction worker did not start")
        return process

    def _force_prediction_worker(self):
        process = None
        failed_generation = None
        try:
            while True:
                sample = self._force_prediction_queue.get()
                if sample is None:
                    if process is not None and process.poll() is None:
                        try:
                            process.stdin.write('{"command":"stop"}\n')
                            process.stdin.flush()
                        except Exception:
                            pass
                    return
                generation = int(sample.get("reset_generation", 0))
                if failed_generation == generation:
                    continue
                try:
                    if process is None or process.poll() is not None:
                        self._emit_force_prediction({
                            "state": "loading",
                            "message": "Loading force-estimation ensemble in an isolated process...",
                        })
                        process = self._start_force_prediction_process()
                    request = dict(sample)
                    request["command"] = "predict"
                    process.stdin.write(json.dumps(
                        request, ensure_ascii=False, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                    result = self._read_force_prediction_message(process)
                    with self._force_prediction_lock:
                        current_generation = self._force_prediction_reset_generation
                    if generation != current_generation:
                        continue
                    if result.get("state") == "error":
                        raise RuntimeError(result.get("message") or "Unknown prediction error")
                    result["message"] = (
                        "Prediction ready; monitoring only and not connected to control.")
                    self._emit_force_prediction(result)
                except Exception as exc:
                    failed_generation = generation
                    with self._force_prediction_lock:
                        current_generation = self._force_prediction_reset_generation
                    if generation == current_generation:
                        self._emit_force_prediction({
                            "state": "error", "hz": 0.0,
                            "message": "Force prediction failed: " + str(exc),
                        })
                    failed_process = process
                    if failed_process is not None and failed_process.poll() is None:
                        try:
                            failed_process.terminate()
                        except Exception:
                            pass
                    with self._force_prediction_lock:
                        if self._force_prediction_process is failed_process:
                            self._force_prediction_process = None
                    process = None
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            with self._force_prediction_lock:
                if self._force_prediction_process is process:
                    self._force_prediction_process = None

    def _ensure_force_prediction_thread(self):
        with self._force_prediction_lock:
            thread = self._force_prediction_thread
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(
                target=self._force_prediction_worker, daemon=True)
            self._force_prediction_thread = thread
            thread.start()

    def shutdownForcePrediction(self):
        thread = self._force_prediction_thread
        if thread is None:
            return
        while True:
            try:
                self._force_prediction_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._force_prediction_queue.put_nowait(None)
        except queue.Full:
            pass
        if thread.is_alive():
            thread.join(timeout=3.0)
        if thread.is_alive():
            with self._force_prediction_lock:
                process = self._force_prediction_process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            thread.join(timeout=2.0)
        self._force_prediction_thread = None

    def _force_prediction_on_frame(self, frame):
        now = float(frame.get("wall", time.time()))
        if now-self._force_prediction_last_submit_wall < 0.045:
            return
        pressure_raw = self._record_vector(frame.get("kpa"), 6)
        length_raw = self._record_vector(frame.get("channel_length_mm"), 6)
        group_raw = self._group_signed_displacements(frame.get("lens"))
        if (any(value is None for value in pressure_raw+length_raw) or
                any(value is None for pair in group_raw for value in pair)):
            return
        displacement, zero = self._zero_group_displacements(
            group_raw, self._force_prediction_zero_mm)
        self._force_prediction_zero_mm = zero
        order = self.DATA_RECORD_CHANNEL_ORDER
        sample = {
            "pressure_kpa": [pressure_raw[index] for index in order],
            "length_mm": [length_raw[index] for index in order],
            "active_mm": [displacement[index][0] for index in order],
            "passive_mm": [displacement[index][1] for index in order],
            "timestamp_s": now,
        }
        with self._force_prediction_lock:
            sample["reset_generation"] = self._force_prediction_reset_generation
        self._force_prediction_last_submit_wall = now
        self._ensure_force_prediction_thread()
        try:
            self._force_prediction_queue.put_nowait(sample)
        except queue.Full:
            try:
                self._force_prediction_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._force_prediction_queue.put_nowait(sample)
            except queue.Full:
                pass

    @staticmethod
    def _nano25_python():
        candidates = [sys.executable]
        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            candidates.append(os.path.join(
                user_profile, ".conda", "envs", "bioreactor", "python.exe"))
        portable = os.path.join(PROJECT_ROOT, ".python310", "python.exe")
        candidates.append(portable)
        candidates.append(str(DATA_ROOT / ".python310" / "python.exe"))
        checked = set()
        for executable in candidates:
            executable = os.path.abspath(executable)
            if executable.lower() in checked or not os.path.isfile(executable):
                continue
            checked.add(executable.lower())
            if executable == os.path.abspath(sys.executable):
                if importlib.util.find_spec("nidaqmx") is not None:
                    return executable
                continue
            options = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                       "timeout": 10}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                if subprocess.run(
                        [executable, "-c", "import nidaqmx"],
                        **options).returncode == 0:
                    return executable
            except Exception:
                continue
        return None

    @QtCore.pyqtSlot(result=str)
    def nano25Status(self):
        with self._nano25_lock:
            payload = dict(self._nano25_latest)
            process = self._nano25_process
        if process is not None and process.poll() is None and payload.get("state") == "stopped":
            payload["state"] = "starting"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @QtCore.pyqtSlot(bool, result=str)
    def setNano25ControlRequired(self, required):
        """Prevent the force/torque recorder from being stopped during control."""
        required = bool(required)
        with self._nano25_lock:
            self._nano25_control_required = required
        return self._emit_nano25({"control_required": required})

    def _monitor_nano25(self, process):
        error_message = ""
        try:
            for raw_line in iter(process.stdout.readline, ""):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = parse_nano25_stream_line(line)
                except Exception as exc:
                    error_message = "Nano25 data parse failed: " + str(exc)
                    self._emit_nano25({"state": "error", "message": error_message})
                    continue
                if payload is not None:
                    if "force_n" in payload and "torque_nm" in payload:
                        payload["received_wall"] = time.time()
                    self._emit_nano25(payload)
                elif line.startswith("ERROR:"):
                    error_message = line
                    self._emit_nano25({"state": "error", "message": line})
            return_code = process.wait()
        except Exception as exc:
            return_code = -1
            error_message = str(exc)
        with self._nano25_lock:
            requested = self._nano25_stop_requested
            if self._nano25_process is process:
                self._nano25_process = None
                self._nano25_thread = None
                self._nano25_stop_requested = False
        if requested or return_code == 0:
            self._emit_nano25({"state": "stopped", "hz": 0.0,
                               "message": "Nano25 acquisition stopped."})
        else:
            self._emit_nano25({
                "state": "error", "hz": 0.0,
                "message": error_message or f"Nano25 recorder exited with code {return_code}.",
            })

    @QtCore.pyqtSlot(result=str)
    def startNano25(self):
        with self._nano25_lock:
            if self._nano25_process is not None and self._nano25_process.poll() is None:
                return json.dumps(self._nano25_latest, ensure_ascii=False,
                                  separators=(",", ":"))
        if not os.path.isfile(NANO25_RECORDER_PATH):
            return self._emit_nano25({
                "state": "error", "message": "record_nano25.py was not found.",
            })
        nano_python = self._nano25_python()
        if not nano_python:
            return self._emit_nano25({
                "state": "error",
                "message": "nidaqmx is not installed in the current or bioreactor Python environment.",
            })
        os.makedirs(NANO25_RECORDINGS_DIR, exist_ok=True)
        command = [
            nano_python, "-u", NANO25_RECORDER_PATH,
            "--gui-stream", "--out", NANO25_RECORDINGS_DIR,
        ]
        options = {
            "cwd": PROJECT_ROOT, "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
            "text": True, "encoding": "utf-8", "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(command, **options)
        except Exception as exc:
            return self._emit_nano25({"state": "error", "message": str(exc)})
        thread = threading.Thread(
            target=self._monitor_nano25, args=(process,), daemon=True)
        with self._nano25_lock:
            self._nano25_process = process
            self._nano25_thread = thread
            self._nano25_stop_requested = False
        result = self._emit_nano25({
            "state": "starting", "hz": 0.0,
            "force_n": [0.0, 0.0, 0.0], "torque_nm": [0.0, 0.0, 0.0],
            "received_wall": None,
            "message": "Starting Nano25; keep the sensor unloaded during automatic zeroing.",
        })
        thread.start()
        return result

    def _stop_nano25(self, wait=False, force=False):
        with self._nano25_lock:
            process = self._nano25_process
            thread = self._nano25_thread
            control_required = self._nano25_control_required
            already_stopped = process is None or process.poll() is not None
            if not already_stopped and (force or not control_required):
                self._nano25_stop_requested = True
        if control_required and not force:
            return self._emit_nano25({
                "control_required": True,
                "message": "Nano25 must remain active while control is running.",
            })
        if already_stopped:
            return self._emit_nano25({
                "state": "stopped", "hz": 0.0,
                "message": "Nano25 is already stopped.",
            })
        self._emit_nano25({"state": "stopping", "message": "Stopping Nano25 acquisition..."})
        try:
            process.stdin.write("stop\n")
            process.stdin.flush()
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=4.0)
            if thread.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass
        return json.dumps({"state": "stopping"}, separators=(",", ":"))

    @QtCore.pyqtSlot(result=str)
    def stopNano25(self):
        return self._stop_nano25(wait=False)

    @staticmethod
    def _codex_executable():
        try:
            from codex_compensator import _find_codex_executable
            return _find_codex_executable()
        except Exception:
            return shutil.which("codex")

    @staticmethod
    def _claude_executable():
        try:
            from codex_compensator import _find_claude_executable
            return _find_claude_executable()
        except Exception:
            return shutil.which("claude")

    def _emit_codex_auth(self, state, message="", method="", provider="gpt"):
        self.codex_auth_status.emit(json.dumps({
            "state": str(state), "message": str(message), "method": str(method),
            "provider": str(provider),
        }, ensure_ascii=False, separators=(",", ":")))

    def _codex_auth_status_payload(self):
        executable = self._codex_executable()
        if not executable:
            return {"state": "missing", "message": "未找到Codex CLI", "method": ""}
        try:
            options = {
                "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                "text": True, "encoding": "utf-8", "errors": "replace", "timeout": 15,
            }
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run([executable, "login", "status"], **options)
            output = (completed.stdout or "").strip()
            lowered = output.lower()
            if completed.returncode == 0 and "logged in" in lowered:
                method = "api" if "api key" in lowered else "account"
                return {"state": "signed_in", "message": output, "method": method}
            return {"state": "signed_out", "message": output or "Codex尚未登录", "method": ""}
        except Exception as exc:
            return {"state": "error", "message": str(exc), "method": ""}

    def _finish_codex_auth(self):
        payload = self._codex_auth_status_payload()
        self._emit_codex_auth(**payload)
        with self._codex_auth_lock:
            self._codex_auth_busy = False

    @QtCore.pyqtSlot(result=bool)
    def checkCodexAuth(self):
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth("checking", "正在检查Codex登录状态")
        threading.Thread(target=self._finish_codex_auth, daemon=True).start()
        return True

    def _codex_chatgpt_login_worker(self):
        executable = self._codex_executable()
        if not executable:
            self._emit_codex_auth("missing", "未找到Codex CLI")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
            return
        try:
            options = {"timeout": 600}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_CONSOLE"):
                options["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            completed = subprocess.run([executable, "login"], **options)
            if completed.returncode != 0:
                self._emit_codex_auth("error", "ChatGPT登录未完成")
                with self._codex_auth_lock:
                    self._codex_auth_busy = False
                return
            self._finish_codex_auth()
        except subprocess.TimeoutExpired:
            self._emit_codex_auth("error", "ChatGPT登录等待超过10分钟")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
        except Exception as exc:
            self._emit_codex_auth("error", "无法启动Codex登录：" + str(exc))
            with self._codex_auth_lock:
                self._codex_auth_busy = False

    @QtCore.pyqtSlot(result=bool)
    def loginCodexChatGPT(self):
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth("login_started", "请在新窗口和浏览器中完成ChatGPT登录", "account")
        threading.Thread(target=self._codex_chatgpt_login_worker, daemon=True).start()
        return True

    def _codex_api_login_worker(self, api_key):
        executable = self._codex_executable()
        if not executable:
            self._emit_codex_auth("missing", "未找到Codex CLI")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
            return
        try:
            options = {
                "input": api_key + "\n", "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8",
                "errors": "replace", "timeout": 60,
            }
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run([executable, "login", "--with-api-key"], **options)
            api_key = ""
            if completed.returncode != 0:
                self._emit_codex_auth("error", "API Key登录失败；请检查Key和网络")
                with self._codex_auth_lock:
                    self._codex_auth_busy = False
                return
            self._finish_codex_auth()
        except subprocess.TimeoutExpired:
            self._emit_codex_auth("error", "API Key登录超时")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
        except Exception as exc:
            self._emit_codex_auth("error", "API Key登录失败：" + str(exc))
            with self._codex_auth_lock:
                self._codex_auth_busy = False

    @QtCore.pyqtSlot(str, result=bool)
    def loginCodexApiKey(self, api_key):
        key = str(api_key or "").strip()
        if len(key) < 20 or any(ch.isspace() for ch in key):
            self._emit_codex_auth("error", "API Key格式无效")
            return False
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth("login_started", "正在通过API Key登录", "api")
        threading.Thread(target=self._codex_api_login_worker, args=(key,), daemon=True).start()
        return True

    def _claude_auth_status_payload(self):
        executable = self._claude_executable()
        if not executable:
            return {"state": "missing", "message": "未找到Claude Code CLI",
                    "method": "", "provider": "claude"}
        try:
            options = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                       "text": True, "encoding": "utf-8", "errors": "replace",
                       "timeout": 15}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(
                [executable, "auth", "status", "--text"], **options)
            output = (completed.stdout or "").strip()
            return {"state": "signed_in" if completed.returncode == 0 else "signed_out",
                    "message": output or ("Claude账号已登录" if completed.returncode == 0
                                           else "Claude账号尚未登录"),
                    "method": "account" if completed.returncode == 0 else "",
                    "provider": "claude"}
        except Exception as exc:
            return {"state": "error", "message": str(exc), "method": "",
                    "provider": "claude"}

    def _finish_claude_auth_check(self):
        payload = self._claude_auth_status_payload()
        self._emit_codex_auth(**payload)
        with self._codex_auth_lock:
            self._codex_auth_busy = False

    @QtCore.pyqtSlot(str, result=bool)
    def checkClaudeAuth(self, method):
        selected = str(method or "account").strip().lower()
        if selected == "api":
            try:
                import mpc_station as st
                signed_in = st.has_transient_llm_api_key("claude")
                self._emit_codex_auth(
                    "signed_in" if signed_in else "signed_out",
                    "Claude API Key已载入内存" if signed_in else "Claude尚未配置API Key",
                    "api" if signed_in else "", "claude")
                return True
            except Exception as exc:
                self._emit_codex_auth("error", str(exc), "", "claude")
                return False
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth("checking", "正在检查Claude账号登录状态", "account", "claude")
        threading.Thread(target=self._finish_claude_auth_check, daemon=True).start()
        return True

    def _claude_account_login_worker(self):
        executable = self._claude_executable()
        if not executable:
            self._emit_codex_auth("missing", "未找到Claude Code CLI", "", "claude")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
            return
        try:
            options = {"timeout": 600}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_CONSOLE"):
                options["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            completed = subprocess.run([executable, "auth", "login"], **options)
            if completed.returncode != 0:
                raise RuntimeError("Claude账号登录未完成")
            self._finish_claude_auth_check()
        except subprocess.TimeoutExpired:
            self._emit_codex_auth("error", "Claude账号登录等待超过10分钟", "", "claude")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
        except Exception as exc:
            self._emit_codex_auth("error", str(exc), "", "claude")
            with self._codex_auth_lock:
                self._codex_auth_busy = False

    @QtCore.pyqtSlot(result=bool)
    def loginClaudeAccount(self):
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth(
            "login_started", "请在新窗口和浏览器中完成Claude账号登录",
            "account", "claude")
        threading.Thread(target=self._claude_account_login_worker, daemon=True).start()
        return True

    def _claude_install_worker(self):
        try:
            if os.name != "nt":
                raise RuntimeError("当前自动安装仅支持Windows")
            command = [
                "powershell.exe", "-NoLogo", "-NoProfile",
                "-ExecutionPolicy", "Bypass", "-Command",
                "& { irm https://claude.ai/install.ps1 | iex }",
            ]
            options = {"timeout": 600}
            if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
                options["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            completed = subprocess.run(command, **options)
            if completed.returncode != 0:
                raise RuntimeError(f"安装程序退出码 {completed.returncode}")
            payload = self._claude_auth_status_payload()
            if payload.get("state") == "missing":
                raise RuntimeError("安装完成，但仍未找到claude命令；请重启软件后再试")
            self._emit_codex_auth(
                "signed_out", "Claude Code已安装，请登录账号", "", "claude")
        except subprocess.TimeoutExpired:
            self._emit_codex_auth(
                "error", "Claude Code安装等待超过10分钟", "", "claude")
        except Exception as exc:
            self._emit_codex_auth(
                "error", "Claude Code安装失败：" + str(exc), "", "claude")
        finally:
            with self._codex_auth_lock:
                self._codex_auth_busy = False

    @QtCore.pyqtSlot(result=bool)
    def installClaudeCode(self):
        if self._claude_executable():
            self._emit_codex_auth(
                "signed_out", "Claude Code已安装，请登录账号", "", "claude")
            return True
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth(
            "installing", "正在安装Claude Code", "account", "claude")
        threading.Thread(target=self._claude_install_worker, daemon=True).start()
        return True

    @QtCore.pyqtSlot(str, result=bool)
    def loginClaudeApiKey(self, api_key):
        key = str(api_key or "").strip()
        if len(key) < 20 or any(ch.isspace() for ch in key):
            self._emit_codex_auth("error", "API Key格式无效", "", "claude")
            return False
        try:
            import mpc_station as st
            if not st.set_transient_llm_api_key("claude", key):
                raise RuntimeError("无法设置Claude API Key")
            key = ""
            self._emit_codex_auth("signed_in", "Claude API Key已载入内存", "api", "claude")
            return True
        except Exception as exc:
            self._emit_codex_auth("error", str(exc), "", "claude")
            return False

    def _codex_logout_worker(self):
        executable = self._codex_executable()
        if not executable:
            self._emit_codex_auth("missing", "未找到Codex CLI")
            with self._codex_auth_lock:
                self._codex_auth_busy = False
            return
        try:
            options = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                       "text": True, "encoding": "utf-8", "errors": "replace",
                       "timeout": 30}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run([executable, "logout"], **options)
            if completed.returncode != 0:
                raise RuntimeError((completed.stdout or "Codex退出失败").strip())
            self._emit_codex_auth("signed_out", "Codex已退出登录", "", "gpt")
        except Exception as exc:
            self._emit_codex_auth("error", "Codex退出失败：" + str(exc), "", "gpt")
        finally:
            with self._codex_auth_lock:
                self._codex_auth_busy = False

    def _claude_logout_worker(self):
        executable = self._claude_executable()
        try:
            if not executable:
                raise FileNotFoundError("未找到Claude Code CLI")
            options = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                       "text": True, "encoding": "utf-8", "errors": "replace",
                       "timeout": 30}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run([executable, "auth", "logout"], **options)
            if completed.returncode != 0:
                raise RuntimeError((completed.stdout or "Claude退出失败").strip())
            self._emit_codex_auth("signed_out", "Claude账号已退出登录", "", "claude")
        except Exception as exc:
            self._emit_codex_auth("error", "Claude退出失败：" + str(exc), "", "claude")
        finally:
            with self._codex_auth_lock:
                self._codex_auth_busy = False

    @QtCore.pyqtSlot(str, str, result=bool)
    def logoutLlm(self, provider, method):
        name = str(provider or "gpt").strip().lower()
        selected = str(method or "account").strip().lower()
        if name == "claude" and selected == "api":
            try:
                import mpc_station as st
                st.set_transient_llm_api_key("claude", "")
                self._emit_codex_auth("signed_out", "Claude API Key已从内存清除", "", "claude")
                return True
            except Exception as exc:
                self._emit_codex_auth("error", str(exc), "", "claude")
                return False
        if name == "claude":
            with self._codex_auth_lock:
                if self._codex_auth_busy:
                    return False
                self._codex_auth_busy = True
            self._emit_codex_auth("login_started", "正在退出Claude账号", "account", "claude")
            threading.Thread(target=self._claude_logout_worker, daemon=True).start()
            return True
        with self._codex_auth_lock:
            if self._codex_auth_busy:
                return False
            self._codex_auth_busy = True
        self._emit_codex_auth("login_started", "正在退出Codex", "", "gpt")
        threading.Thread(target=self._codex_logout_worker, daemon=True).start()
        return True

    @QtCore.pyqtSlot(result=str)
    def dashboardData(self):
        """Return a compact, read-only snapshot for the embedded control-response plot."""
        try:
            import mpc_station as st

            source = list(st.TELEM)
            if not source:
                return json.dumps({"rows": [], "stage": "待机"}, ensure_ascii=False)

            def gui_triplet(record, gui_key, model_key):
                value = record.get(gui_key)
                if value is None:
                    value = st.st3_from_q_model(np.radians(record.get(model_key, [0, 0, 0])))
                out = np.asarray(value, dtype=float).reshape(-1)
                if out.size < 3:
                    out = np.pad(out, (0, 3 - out.size))
                return [float(v) if np.isfinite(v) else 0.0 for v in out[:3]]

            rows = []
            for record in source:
                r_gui = gui_triplet(record, "r_gui", "r")
                rc_value = record.get("r_control_gui")
                if rc_value is None:
                    rc_gui = r_gui
                else:
                    rc = np.asarray(rc_value, dtype=float).reshape(-1)
                    if rc.size < 3:
                        rc = np.pad(rc, (0, 3 - rc.size))
                    rc_gui = [float(v) if np.isfinite(v) else 0.0 for v in rc[:3]]
                rows.append({
                    "t": float(record.get("t", 0.0)),
                    "r": r_gui,
                    "rc": rc_gui,
                    "q": gui_triplet(record, "q_gui", "q"),
                })

            latest = source[-1]
            forward = ""
            if "forward_status" in latest:
                status = str(latest.get("forward_status", "等待下一拍"))
                generation = int(latest.get("forward_generation", 0))
                error = latest.get("forward_error_norm_deg")
                error_text = "" if error is None else f"，误差 {float(error):.2f}°"
                forward = f"顺模型 G{generation}，{status}{error_text}"
            return json.dumps({
                "rows": rows,
                "stage": str(latest.get("stage", "轨迹跟踪")),
                "forward": forward,
            }, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            return json.dumps({"rows": [], "stage": "待机", "error": str(exc)}, ensure_ascii=False)

    def _recording_emit(self, kind, state, message, path=""):
        if str(state) == "error" and path:
            try:
                log_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                        "recording_error.log")
                with open(log_path, "a", encoding="utf-8") as stream:
                    stream.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                                 f" [{kind}] {message}\n")
            except Exception:
                pass
        self.recording_status.emit(json.dumps({
            "kind": str(kind), "state": str(state), "message": str(message),
            "path": os.path.abspath(path) if path else "",
        }, ensure_ascii=False, separators=(",", ":")))

    @staticmethod
    def _recording_stamp():
        return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"

    def _recording_path(self, kind, extension):
        os.makedirs(self._recording_base_dir, exist_ok=True)
        with self._recording_session_lock:
            session = self._recording_session
            folder = session["folder"] if session is not None else self._recording_base_dir
        if session is not None:
            return os.path.join(folder, f"{kind}.{extension}")
        return os.path.join(folder, f"{self._recording_stamp()}_{kind}.{extension}")

    @QtCore.pyqtSlot(result=str)
    def startRecordingSession(self):
        """Create one timestamp folder shared by GUI, camera and Excel for this task."""
        with self._recording_session_lock:
            if self._recording_session is not None:
                return json.dumps({"ok": True, **self._recording_session}, ensure_ascii=False)
            if not self._recording_directory_confirmed:
                return json.dumps({
                    "ok": False,
                    "error": "已勾选记录内容，请先点击“路径”并选择保存文件夹",
                }, ensure_ascii=False)
            if not os.path.isdir(self._recording_base_dir):
                self._recording_directory_confirmed = False
                return json.dumps({
                    "ok": False,
                    "error": "所选保存文件夹已不存在，请重新选择路径",
                }, ensure_ascii=False)
            os.makedirs(self._recording_base_dir, exist_ok=True)
            stem = self._recording_stamp()
            folder = os.path.join(self._recording_base_dir, stem)
            suffix = 1
            while os.path.exists(folder):
                folder = os.path.join(self._recording_base_dir, f"{stem}_{suffix:02d}")
                suffix += 1
            os.makedirs(folder)
            session = {"token": f"task-{time.time_ns()}", "folder": folder}
            self._recording_session = session
            self._last_recording_session_folder = os.path.abspath(folder)
            # The MPC/LLM worker runs in another thread in this process.  Keep
            # the current task folder available to it without coupling the
            # controller module to the Qt bridge.
            os.environ[LLM_RECORDING_SESSION_ENV] = os.path.abspath(folder)
        return json.dumps({"ok": True, **session}, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def recordingDirectory(self):
        return os.path.abspath(self._recording_base_dir)

    @QtCore.pyqtSlot(result=str)
    def recordingDirectoryState(self):
        return json.dumps({
            "selected": bool(self._recording_directory_confirmed),
            "path": (os.path.abspath(self._recording_base_dir)
                     if self._recording_directory_confirmed else ""),
        }, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def chooseRecordingDirectory(self):
        with self._recording_session_lock:
            if self._recording_session is not None:
                return json.dumps({"ok": False, "error": "任务记录期间不能更改保存路径"},
                                  ensure_ascii=False)
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            QtWidgets.QApplication.activeWindow(), "选择视频保存路径",
            self._recording_base_dir, QtWidgets.QFileDialog.ShowDirsOnly)
        if not selected:
            return json.dumps({"ok": False, "cancelled": True,
                               "path": self._recording_base_dir}, ensure_ascii=False)
        self._recording_base_dir = os.path.abspath(selected)
        self._recording_directory_confirmed = True
        QtCore.QSettings("TRO", "PAMShoulder").setValue(
            "recordingDirectory", self._recording_base_dir)
        return json.dumps({"ok": True, "path": self._recording_base_dir}, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def chooseCodexReplaySavePath(self):
        """Choose a self-contained JSON package for a completed MPC+LLM process."""
        base = (self._recording_base_dir if os.path.isdir(self._recording_base_dir)
                else RECORDINGS_DIR)
        default_name = os.path.join(
            base, time.strftime("MPC_LLM_10_rounds_%Y%m%d_%H%M%S.json"))
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            QtWidgets.QApplication.activeWindow(), "保存MPC+LLM十轮过程",
            default_name, "MPC+LLM process (*.json)")
        if not selected:
            return json.dumps({"ok": False, "cancelled": True}, ensure_ascii=False)
        if not selected.lower().endswith(".json"):
            selected += ".json"
        return json.dumps({"ok": True, "path": os.path.abspath(selected)},
                          ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def chooseCodexReplayOpenPath(self):
        base = (self._recording_base_dir if os.path.isdir(self._recording_base_dir)
                else RECORDINGS_DIR)
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            QtWidgets.QApplication.activeWindow(), "载入MPC+LLM过程回放",
            base, "MPC+LLM process (*.json)")
        if not selected:
            return json.dumps({"ok": False, "cancelled": True}, ensure_ascii=False)
        return json.dumps({"ok": True, "path": os.path.abspath(selected)},
                          ensure_ascii=False)

    @QtCore.pyqtSlot(str, result=bool)
    def finishRecordingSession(self, token):
        with self._recording_session_lock:
            session = self._recording_session
            if session is None or session["token"] != str(token):
                return False
            self._recording_session = None
            session_folder = os.path.abspath(session["folder"])
            if os.path.abspath(os.environ.get(
                    LLM_RECORDING_SESSION_ENV, "") or ".") == session_folder:
                os.environ.pop(LLM_RECORDING_SESSION_ENV, None)
        return True

    @QtCore.pyqtSlot(str, int, int, int, int, result=str)
    def startNativeRegionRecording(self, kind, x, y, width, height):
        kind = str(kind).strip().lower()
        recorder = self._native_recorders.get(kind)
        if recorder is None or kind != "gui":
            return json.dumps({"ok": False, "error": "未知的GUI录制区域"}, ensure_ascii=False)
        path = self._recording_path(kind, "mp4")
        result = recorder.start(path, (int(x), int(y), int(width), int(height)))
        if not result.get("ok"):
            self._recording_emit(kind, "error", result.get("error", "区域录屏启动失败"), path)
        return json.dumps(result, ensure_ascii=False)

    @QtCore.pyqtSlot(str, result=str)
    def stopNativeRegionRecording(self, kind):
        recorder = self._native_recorders.get(str(kind).strip().lower())
        if recorder is None:
            return json.dumps({"ok": False, "error": "区域录屏器尚未初始化"}, ensure_ascii=False)
        return json.dumps(recorder.stop(), ensure_ascii=False)

    @QtCore.pyqtSlot(str, str, result=str)
    def startVideoRecording(self, kind, mime_type):
        """Open an independent binary sink for one browser MediaRecorder stream."""
        kind = str(kind).strip().lower()
        if kind not in ("gui", "camera"):
            return json.dumps({"ok": False, "error": "未知视频类型"}, ensure_ascii=False)
        extension = "mp4" if "mp4" in str(mime_type).lower() else "webm"
        with self._video_recording_lock:
            if kind in self._video_recordings:
                return json.dumps({"ok": False, "error": "该视频正在录制"}, ensure_ascii=False)
            try:
                path = self._recording_path(kind, extension)
                partial = path + ".part"
                stream = open(partial, "wb")
                token = f"{kind}-{time.time_ns()}"
                self._video_recordings[kind] = {
                    "token": token, "path": path, "partial": partial,
                    "stream": stream, "bytes": 0,
                }
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        self._recording_emit(kind, "recording", "录制中", path)
        return json.dumps({"ok": True, "token": token, "path": path}, ensure_ascii=False)

    @QtCore.pyqtSlot(str, str, str, result=bool)
    def appendVideoChunk(self, kind, token, encoded):
        """Append one ordered MediaRecorder chunk without loading the whole video in Python."""
        try:
            payload = base64.b64decode(str(encoded), validate=True)
        except Exception:
            return False
        if not payload or len(payload) > 8*1024*1024:
            return False
        with self._video_recording_lock:
            item = self._video_recordings.get(str(kind).strip().lower())
            if item is None or item["token"] != str(token):
                return False
            try:
                item["stream"].write(payload)
                item["bytes"] += len(payload)
                return True
            except Exception:
                return False

    @QtCore.pyqtSlot(str, str, result=str)
    def finishVideoRecording(self, kind, token):
        kind = str(kind).strip().lower()
        with self._video_recording_lock:
            item = self._video_recordings.get(kind)
            if item is None or item["token"] != str(token):
                return json.dumps({"ok": False, "error": "录制会话不存在"}, ensure_ascii=False)
            self._video_recordings.pop(kind, None)
        try:
            item["stream"].flush()
            item["stream"].close()
            if item["bytes"] <= 0:
                try:
                    os.remove(item["partial"])
                except FileNotFoundError:
                    pass
                raise RuntimeError("没有收到视频帧")
            os.replace(item["partial"], item["path"])
            self._recording_emit(kind, "saved", "视频已保存", item["path"])
            return json.dumps({"ok": True, "path": item["path"]}, ensure_ascii=False)
        except Exception as exc:
            self._recording_emit(kind, "error", "视频保存失败：" + str(exc), item["path"])
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    # Response workbook: one compact 20 Hz row in the exact physical-channel
    # order used in experiments.  Firmware order is V0=15,V1=48,V2=23,
    # V3=67,V4=9,V5=10, hence the explicit [0,2,1,3,4,5] reorder below.
    DATA_RECORD_HEADERS = (
        "response_time_s", "X_deg", "Y_deg", "phi_deg",
        "X_speed_deg_s", "Y_speed_deg_s", "phi_speed_deg_s",
        "P_A1+5_kPa", "P_A2+3_kPa", "P_A4+8_kPa",
        "P_A6+7_kPa", "P_A9_kPa", "P_A10_kPa",
        "L_A1+5_mm", "d_active_A1+5_mm", "d_passive_A1+5_mm",
        "L_A2+3_mm", "d_active_A2+3_mm", "d_passive_A2+3_mm",
        "L_A4+8_mm", "d_active_A4+8_mm", "d_passive_A4+8_mm",
        "L_A6+7_mm", "d_active_A6+7_mm", "d_passive_A6+7_mm",
        "L_A9_mm", "d_active_A9_mm", "d_passive_A9_mm",
        "L_A10_mm", "d_active_A10_mm", "d_passive_A10_mm",
        "F_X_N", "F_Y_N", "F_Z_N",
        "T_X_Nm", "T_Y_Nm", "T_Z_Nm",
    )
    DATA_RECORD_CHANNEL_ORDER = (0, 2, 1, 3, 4, 5)
    DATA_RECORD_GROUP_LABELS = ("A1+5", "A2+3", "A4+8", "A6+7", "A9", "A10")
    MUSCLE_SENSOR_GROUPS = ((0, 4), (3, 7), (1, 2), (5, 6), (8,), (9,))

    @QtCore.pyqtSlot(result=str)
    def startDataRecording(self):
        """Begin a flat, typed experiment table; the workbook is created on Stop."""
        try:
            importlib.import_module("openpyxl")  # fail early before recording a long run
        except Exception as exc:
            return json.dumps({"ok": False, "error": "缺少 openpyxl：" + str(exc)}, ensure_ascii=False)
        if not self.ser or self._latest_frame is None or time.time()-self._latest_frame["wall"] > 1.0:
            return json.dumps({"ok": False, "error": "请先连接设备并等待实时数据"}, ensure_ascii=False)
        with self._data_recording_lock:
            if self._data_recording is not None:
                return json.dumps({"ok": False, "error": "Excel数据正在记录"}, ensure_ascii=False)
            path = self._recording_path("response", "xlsx")
            self._data_recording = {
                "path": path, "t0": time.time(), "last_wall": 0.0, "rows": [],
                "group_displacement_zero_mm": None,
                "previous_pose_deg": None, "previous_response_time_s": None,
            }
            self._last_data_recording = self._data_recording
        self._recording_emit("data", "recording", "数据记录中", path)
        return json.dumps({"ok": True, "path": path}, ensure_ascii=False)

    @staticmethod
    def _record_vector(value, count, default=None):
        try:
            source = list(value)
        except Exception:
            source = []
        result = []
        for index in range(count):
            try:
                number = float(source[index])
                result.append(number if np.isfinite(number) else default)
            except (IndexError, TypeError, ValueError):
                result.append(default)
        return result

    @classmethod
    def _group_signed_displacements(cls, value):
        """Signed active/passive means for all six valve groups before zeroing."""
        lens = cls._record_vector(value, 20)
        result = []
        for indexes in cls.MUSCLE_SENSOR_GROUPS:
            active = [lens[index] for index in indexes]
            passive = [lens[index+10] for index in indexes]
            if any(item is None for item in active+passive):
                result.append([None, None])
            else:
                result.append([
                    float(np.mean(active)), float(np.mean(passive)),
                ])
        return result

    @classmethod
    def _zero_group_displacements(cls, raw, zero):
        """Zero each signed active/passive sensor group at its first valid sample."""
        raw_pairs = [cls._record_vector(pair, 2) for pair in list(raw or [])[:6]]
        raw_pairs.extend([[None, None] for _ in range(6-len(raw_pairs))])
        if not isinstance(zero, list) or len(zero) != 6:
            zero = [[None, None] for _ in range(6)]
        else:
            zero = [cls._record_vector(pair, 2) for pair in zero]
        displacement = []
        for channel in range(6):
            pair_delta = []
            for side in range(2):
                value = raw_pairs[channel][side]
                if value is None:
                    pair_delta.append(None)
                    continue
                if zero[channel][side] is None:
                    zero[channel][side] = float(value)
                pair_delta.append(float(value)-float(zero[channel][side]))
            displacement.append(pair_delta)
        return displacement, zero

    @classmethod
    def _l15_signed_displacements(cls, value):
        """Backward-compatible A1+A5 view."""
        return cls._group_signed_displacements(value)[0]

    @classmethod
    def _zero_l15_displacements(cls, raw, zero):
        """Backward-compatible zeroing helper for older callers and tests."""
        zero_groups = None if zero is None else [zero]+[[None, None] for _ in range(5)]
        displacement, zero_groups = cls._zero_group_displacements(
            [raw]+[[None, None] for _ in range(5)], zero_groups)
        return displacement[0], zero_groups[0]

    @classmethod
    def _pose_speed_deg_s(cls, pose, sample_time, previous_pose, previous_time):
        """Finite-difference GUI X/Y/phi with a zero-valued first sample."""
        current = cls._record_vector(pose, 3)
        try:
            current_time = float(sample_time)
        except (TypeError, ValueError):
            current_time = float("nan")
        valid_current = all(value is not None for value in current)
        previous = cls._record_vector(previous_pose, 3)
        try:
            dt = current_time-float(previous_time)
        except (TypeError, ValueError):
            dt = float("nan")
        if (not valid_current or any(value is None for value in previous) or
                not np.isfinite(dt) or dt <= 1e-6):
            speed = [0.0, 0.0, 0.0]
        else:
            delta = np.asarray(current, dtype=float)-np.asarray(previous, dtype=float)
            # Avoid an artificial ±360 deg/s spike if an angle crosses its wrap.
            delta = (delta+180.0) % 360.0-180.0
            speed = (delta/dt).tolist()
        return speed, (list(current) if valid_current else None), current_time

    def _append_live_control_response(self, gui, now):
        """Put mode-independent, GUI-zeroed IMU pose into CONTROL RESPONSE."""
        pose = self._record_vector(gui.get("pose"), 3)
        if any(value is None for value in pose):
            return False
        try:
            import mpc_station as st
        except Exception:
            return False
        # Defensive server-side guard for MPC+LLM inter-round analysis.  The
        # browser also sends pause_live_response, but the controller state is
        # authoritative and prevents a delayed UI flag from adding zero rows.
        bridge_state = st.BR
        codex_status = bridge_state.get("codex_auto_status") or {}
        if codex_status.get("state") == "replay_wait":
            return True
        if (bool(bridge_state.get("codex_mpc")) and bridge_state.get("mode") != "mpc" and
                str(codex_status.get("state", "")) == "running" and
                str(codex_status.get("phase", "")) in {
                    "operator_input", "llm_reasoning", "operator_review",
                }):
            return True
        with self._live_response_lock:
            if now-self._live_response_last_wall < 0.045:  # at most about 20 Hz
                return True
            latest = st.TELEM[-1] if st.TELEM else None
            latest_wall = 0.0
            if latest is not None:
                try:
                    latest_wall = float(latest.get("wall", 0.0) or 0.0)
                except (TypeError, ValueError):
                    latest_wall = 0.0
            # /telem and /step contain the authoritative reference and measured
            # pose.  Do not insert a second IMU point while that stream is live.
            if (latest is not None and latest.get("source") == "control" and
                    latest_wall > 0.0 and now-latest_wall < 0.30):
                self._live_response_last_wall = now
                return True
            if latest is None:
                sample_time = 0.0
            else:
                try:
                    previous_t = float(latest.get("t", 0.0))
                except (TypeError, ValueError):
                    previous_t = 0.0
                elapsed = now-latest_wall if latest_wall > 0.0 else 0.05
                sample_time = previous_t+max(0.001, min(0.5, elapsed))
            reference = self._record_vector(gui.get("reference"), 3)
            if any(value is None for value in reference):
                reference = list(pose)
            frame = self._latest_frame or {}
            group_raw = self._group_signed_displacements(frame.get("lens"))
            output = self._record_vector(gui.get("output"), 6, default=0.0)
            st.TELEM.append({
                "t": float(sample_time),
                "wall": float(now),
                "source": "live",
                "stage": str(gui.get("stage") or "实时运动"),
                "q_gui": list(pose),
                "r_gui": list(reference),
                "r_control_gui": list(reference),
                "u": [int(round(value or 0.0)) for value in output],
                "pressure_kpa": self._record_vector(frame.get("kpa"), 6),
                "channel_length_mm": self._record_vector(
                    frame.get("channel_length_mm"), 6),
                "group_displacement_raw_mm": group_raw,
                "l15_displacement_raw_mm": list(group_raw[0]),
            })
            self._live_response_last_wall = now
        return True

    @QtCore.pyqtSlot(str, result=bool)
    def appendDataSample(self, payload):
        """Update CONTROL RESPONSE globally and optionally cache an Excel row."""
        try:
            gui = json.loads(str(payload))
        except Exception:
            return False
        now = time.time()
        # Teaching uses executed trajectory time.  The IMU remains live while
        # idle or while the LLM designs the next lap, but those wall-clock-only
        # frames must not extend the plot or response.xlsx.
        pause_live_response = bool(gui.get("pause_live_response", False))
        live_ok = (True if pause_live_response
                   else self._append_live_control_response(gui, now))
        if pause_live_response or gui.get("record_sample") is False:
            return live_ok
        with self._data_recording_lock:
            record = self._data_recording
            if record is None:
                return live_ok
            if now-record["last_wall"] < 0.045:
                return True
            if len(record["rows"]) >= 1048575:
                return False
            frame = self._latest_frame
            if frame is None or now-frame["wall"] > 1.0:
                return False
            record["last_wall"] = now
            pose = self._record_vector(gui.get("pose"), 3)
            try:
                response_time_s = float(gui.get("response_time_s"))
                if not np.isfinite(response_time_s):
                    raise ValueError("non-finite response time")
            except (TypeError, ValueError):
                # Keep a relative time axis even before the response plot has
                # emitted its first point; never fall back to wall-clock time.
                response_time_s = max(0.0, now-float(record["t0"]))
            pose_speed, previous_pose, previous_time = self._pose_speed_deg_s(
                pose, response_time_s, record.get("previous_pose_deg"),
                record.get("previous_response_time_s"))
            record["previous_pose_deg"] = previous_pose
            record["previous_response_time_s"] = previous_time
            pressure_raw = self._record_vector(frame.get("kpa"), 6)
            length_raw = self._record_vector(frame.get("channel_length_mm"), 6)
            pressure = [pressure_raw[i] for i in self.DATA_RECORD_CHANNEL_ORDER]
            channel_length = [length_raw[i] for i in self.DATA_RECORD_CHANNEL_ORDER]
            group_raw = self._group_signed_displacements(frame.get("lens"))
            group_displacement, group_zero = self._zero_group_displacements(
                group_raw, record.get("group_displacement_zero_mm"))
            record["group_displacement_zero_mm"] = group_zero
            ordered_displacement = [
                group_displacement[index] for index in self.DATA_RECORD_CHANNEL_ORDER
            ]
            length_columns = [
                value
                for length, pair in zip(channel_length, ordered_displacement)
                for value in (length, pair[0], pair[1])
            ]
            with self._nano25_lock:
                nano = dict(self._nano25_latest)
            try:
                nano_age = now-float(nano.get("received_wall"))
            except (TypeError, ValueError):
                nano_age = float("inf")
            nano_valid = (nano.get("state") in ("running", "reconnecting")
                          and nano_age <= 0.2)
            nano_force = self._record_vector(
                nano.get("force_n") if nano_valid else None, 3)
            nano_torque = self._record_vector(
                nano.get("torque_nm") if nano_valid else None, 3)
            # Nano25 force stream is [GUI Y, GUI Z, GUI X] with the opposite
            # positive direction.  Save the workbook in conventional X,Y,Z.
            force_xyz = [-nano_force[2], -nano_force[0], -nano_force[1]] \
                if all(value is not None for value in nano_force) else [None]*3
            record["rows"].append([
                response_time_s, *pose, *pose_speed, *pressure,
                *length_columns,
                *force_xyz, *nano_torque,
            ])
        return True

    @classmethod
    def _write_data_workbook(cls, rows, path):
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        workbook = Workbook(write_only=True)
        sheet = workbook.create_sheet("控制响应")
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
        header_cells = []
        for value in cls.DATA_RECORD_HEADERS:
            cell = WriteOnlyCell(sheet, value=value)
            cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            header_cells.append(cell)
        sheet.append(header_cells)
        for row in rows:
            values = list(row)
            if values:
                response_time_cell = WriteOnlyCell(sheet, value=values[0])
                response_time_cell.number_format = "0.000"
                values[0] = response_time_cell
            sheet.append(values)
        last_column = get_column_letter(len(cls.DATA_RECORD_HEADERS))
        sheet.auto_filter.ref = f"A1:{last_column}{len(rows)+1}"
        for index, heading in enumerate(cls.DATA_RECORD_HEADERS, 1):
            width = 18 if heading == "response_time_s" else max(10, min(18, len(heading)+2))
            sheet.column_dimensions[get_column_letter(index)].width = width
        tmp = path + ".tmp"
        workbook.save(tmp)
        os.replace(tmp, path)

    def _save_data_recording_worker(self, rows, path, generation):
        try:
            with self._data_save_lock:
                with self._data_recording_lock:
                    if generation < self._data_save_generation:
                        return
                self._write_data_workbook(rows, path)
            self._recording_emit("data", "saved", f"Excel已保存，共{len(rows)}行", path)
        except Exception as exc:
            self._recording_emit("data", "error", "Excel保存失败：" + str(exc), path)

    def _queue_data_snapshot(self, record):
        rows = [list(row) for row in record.get("rows", [])]
        path = str(record.get("path", ""))
        if not rows or not path:
            return None
        self._data_save_generation += 1
        generation = self._data_save_generation
        threading.Thread(target=self._save_data_recording_worker,
                         args=(rows, path, generation), daemon=True).start()
        return {"ok": True, "saving": True, "rows": len(rows), "path": path}

    def _control_response_rows(self):
        """Return exactly the samples currently shown by CONTROL RESPONSE."""
        import mpc_station as st

        rows = []
        group_zero = None
        previous_pose = None
        previous_time = None
        for record in list(st.TELEM):
            pose = record.get("q_gui")
            if pose is None:
                pose = st.st3_from_q_model(np.radians(record.get("q", [0, 0, 0])))
            pose = self._record_vector(pose, 3)
            sample_time = float(record.get("t", 0.0))
            pose_speed, previous_pose, previous_time = self._pose_speed_deg_s(
                pose, sample_time, previous_pose, previous_time)
            pressure_raw = self._record_vector(record.get("pressure_kpa"), 6)
            length_raw = self._record_vector(record.get("channel_length_mm"), 6)
            pressure = [pressure_raw[i] for i in self.DATA_RECORD_CHANNEL_ORDER]
            channel_length = [length_raw[i] for i in self.DATA_RECORD_CHANNEL_ORDER]
            group_raw = record.get("group_displacement_raw_mm")
            if not isinstance(group_raw, (list, tuple)) or len(group_raw) < 6:
                group_raw = [[None, None] for _ in range(6)]
                group_raw[0] = self._record_vector(
                    record.get("l15_displacement_raw_mm"), 2)
            group_displacement, group_zero = self._zero_group_displacements(
                group_raw, group_zero)
            ordered_displacement = [
                group_displacement[index] for index in self.DATA_RECORD_CHANNEL_ORDER
            ]
            length_columns = [
                value
                for length, pair in zip(channel_length, ordered_displacement)
                for value in (length, pair[0], pair[1])
            ]
            force_yzx = self._record_vector(record.get("force_gui_yzx_n"), 3)
            force_xyz = ([force_yzx[2], force_yzx[0], force_yzx[1]]
                         if all(value is not None for value in force_yzx) else [None]*3)
            torque_xyz = self._record_vector(record.get("torque_xyz_nm"), 3)
            rows.append([
                sample_time, *pose, *pose_speed, *pressure,
                *length_columns,
                *force_xyz, *torque_xyz,
            ])
        return rows

    def _control_response_workbook_path(self):
        with self._recording_session_lock:
            session = self._recording_session
            folder = (session.get("folder") if session is not None else
                      self._last_recording_session_folder)
        if not folder:
            folder = self._recording_base_dir
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "response.xlsx")

    @QtCore.pyqtSlot(result=str)
    def saveDataSnapshot(self):
        """Overwrite response.xlsx from the same samples drawn in CONTROL RESPONSE."""
        with self._data_recording_lock:
            rows = self._control_response_rows()
            if rows:
                record = {"rows": rows, "path": self._control_response_workbook_path()}
                self._last_data_recording = record
            else:
                # A completed controller run may clear the live telemetry before
                # the operator presses Save.  Preserve and save the most recent
                # completed response instead of replacing it with an empty one.
                record = self._data_recording or self._last_data_recording
            result = self._queue_data_snapshot(record) if record is not None else None
        if result is None:
            return json.dumps({"ok": False, "error": "当前没有可保存的控制响应数据"},
                              ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def stopDataRecording(self):
        with self._data_recording_lock:
            record = self._data_recording
            self._data_recording = None
            rows = self._control_response_rows()
            if rows:
                record = {"rows": rows, "path": self._control_response_workbook_path()}
            if record is not None:
                self._last_data_recording = record
            result = self._queue_data_snapshot(record) if record is not None else None
        if result is None:
            return json.dumps({"ok": False, "error": "当前没有可保存的控制响应数据"},
                              ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    @QtCore.pyqtSlot(bool)
    def setDebugMode(self, enabled):
        """Keep the legacy response window available in debug mode only."""
        self._debug_mode = bool(enabled)
        self.debug_mode_changed.emit(self._debug_mode)

    @QtCore.pyqtSlot(str, result=str)
    def open(self, port):
        """Connect only a positively identified GIGA, never a generic Arduino port."""
        try:
            from giga_serial import select_giga_port
            cands = list(list_ports.comports())
            print("# 候选串口：", [f"{p.device}({p.description})" for p in cands])
            port = select_giga_port(cands, port)
        except Exception as exc:
            return str(exc)
        self.close()
        self._reset_force_prediction_state(
            "Serial connection changed; waiting to establish a new prediction zero.")
        try:
            import mpc_station as st
            st.HOST_LOCKS = ""
            st.HOST_MPC_COMMAND = None
        except Exception:
            pass
        try:
            # write_timeout 必须设：CDC 僵死时 write 会无限阻塞冻死整个 GUI 主线程
            self.ser = pyserial.Serial(port, 115200, timeout=0, write_timeout=0.5)
            self._frame_buf = ""
            self._latest_frame = None
            self._adc_cal_applied = False
            self._last_rx = __import__("time").time()
            self._kicked = False
            self._errs = 0
            # 不做开口 DTR 抖动（2026-09-03 撤销）：僵死是 qt_gui 时代才出现的，
            # 主动抖 DTR 反而可能把 mbed 的 CDC 状态机抖坏。只清缓冲、静静打开；
            # 2 秒无数据时 _poll 里还有一次自救踹。
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self.timer.start(20)
            print(f"# 串口已开 {port}")
            return "ok:" + port
        except Exception as e:
            self.ser = None
            return str(e)

    @QtCore.pyqtSlot()
    def close(self):
        self.timer.stop()
        self._reset_force_prediction_state(
            "Serial device is disconnected; force prediction is waiting.")
        if self._reachability_test is not None:
            self._reachability_abort("串口关闭")
        if self._manual_pressure_test is not None:
            self._manual_pressure_stop("串口关闭")
        self._dm_ident = None
        self._adc_cal = None
        self._speed_test = None
        self._x_comp_test = None
        self._mpc_data_collect = None
        self._phi_gradient_test = None
        self._quick_health_running = False
        self._manual_pressure_test = None
        self._reachability_test = None
        if self.ser:
            try:
                self.ser.write(b"s\n")               # 断开顺手放气
                self.ser.close()
            except Exception:
                pass
            print("# 串口已关（已发 s 放气）")
        self.ser = None
        try:
            import mpc_station as st
            st.HOST_LOCKS = ""
            st.HOST_MPC_COMMAND = None
        except Exception:
            pass

    @QtCore.pyqtSlot(str, result=bool)
    def write(self, txt):
        if not self.ser:
            return False
        try:
            # MPC 的 /step 响应把(lock,uact)放入一次性槽。页面仍发 W 时，
            # 这里核对六路电压后原子改写成 M<6lock>;<6uact>；PI/开环不受影响。
            cmd = txt.strip()
            # 六路手动压力测试由宿主直接维持唯一的 O-stream；拒绝页面中迟到的
            # 轨迹/MPC/单路命令覆盖它。s 急停仍在下方正常生效。
            if (self._manual_pressure_test is not None and cmd and
                    cmd[0].upper() in ("W", "O", "C", "M", "V")):
                return False
            if (self._reachability_test is not None and cmd and
                    cmd[0].upper() in ("W", "O", "C", "M", "V")):
                return False
            # 联合 DataModel/V6 辨识同样由宿主独占串口；采集期间只允许其
            # 内部扫描序列和用户急停，避免页面迟到命令污染同一批辨识数据。
            if (self._dm_ident is not None and cmd and
                    cmd[0].upper() in ("W", "O", "C", "M", "V")):
                return False
            if cmd.startswith("W"):
                try:
                    import mpc_station as st
                    pending = getattr(st, "HOST_MPC_COMMAND", None)
                    st.HOST_MPC_COMMAND = None
                    if pending is not None:
                        planned_locks, planned_u = pending[:2]
                        phi_pair_assist = len(pending) > 2 and bool(pending[2])
                        received_u = [float(v) for v in cmd[1:].split(",")]
                        if (len(received_u) == 6 and
                                np.allclose(received_u, np.asarray(planned_u, float), atol=1.0)):
                            cmd = (("N" if phi_pair_assist else "M") + ",".join("1" if int(v) else "0" for v in planned_locks) +
                                   ";" + ",".join(str(int(v)) for v in planned_u))
                            txt = cmd + "\n"
                except Exception:
                    pass
            if cmd.startswith(("M", "N")) and ";" in cmd:
                lock_part, voltage_part = cmd[1:].split(";", 1)
                locks = lock_part.split(",")
                vals = voltage_part.split(",")
                if len(locks) == 6 and len(vals) == 6:
                    limit = configured_max_output_mv()
                    safe_locks = ["1" if int(float(v)) else "0" for v in locks]
                    safe_vals = [str(int(np.clip(float(v), 0, limit))) for v in vals]
                    txt = cmd[0] + ",".join(safe_locks) + ";" + ",".join(safe_vals) + "\n"
            elif cmd.startswith(("W", "O", "C")):
                vals = cmd[1:].split(",")
                if len(vals) == 6:
                    limit = (OPEN_LOOP_MAX_OUTPUT_MV if cmd.startswith(("O", "C"))
                             else configured_max_output_mv())
                    txt = cmd[0] + ",".join(str(int(np.clip(float(v), 0, limit)))
                                             for v in vals) + "\n"
            elif cmd.startswith("V") and "," in cmd:
                head, val = cmd.split(",", 1)
                txt = f"{head},{int(np.clip(float(val), 0, configured_max_output_mv()))}\n"
            if self._dm_ident is not None and txt.strip().lower().startswith("s"):
                self._ident_abort("用户按了急停")
            if self._adc_cal is not None and txt.strip().lower().startswith("s"):
                self._adc_cal_abort("用户按了急停")
            if self._speed_test is not None and txt.strip().lower().startswith("s"):
                self._speed_abort("用户按了急停")
            if self._x_comp_test is not None and txt.strip().lower().startswith("s"):
                self._x_comp_abort("用户按了急停")
            if self._mpc_data_collect is not None and txt.strip().lower().startswith("s"):
                self._mpc_data_abort("用户按了急停")
            if self._phi_gradient_test is not None and txt.strip().lower().startswith("s"):
                self._phi_gradient_abort("用户按了急停")
            if self._manual_pressure_test is not None and txt.strip().lower().startswith("s"):
                self._manual_pressure_stop("用户按了急停")
            if self._reachability_test is not None and txt.strip().lower().startswith("s"):
                self._reachability_abort("用户按了急停")
            self.ser.write(txt.encode("ascii", "ignore"))
            return True
        except Exception:
            return False

    @QtCore.pyqtSlot(result=str)
    def startAdcCalibration(self):
        """用 V0({1,5}) 的稳定 DAC 档位标定 AD7606 raw→kPa 直线。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        if self._manual_pressure_test is not None:
            return "六路手动压力测试正在运行，请先停止。"
        if self._reachability_test is not None:
            return "Reachability map 正在采样，请先停止。"
        if self._dm_ident is not None:
            return "DataModel 正在辨识，请先停止。"
        if self._adc_cal is not None:
            return "压力 ADC 正在标定；需要停止请按急停。"
        if self._speed_test is not None:
            return "最大速度测试正在运行；需要停止请按急停。"
        if self._x_comp_test is not None:
            return "X耦合补偿测试正在运行；需要停止请按急停。"
        if self._mpc_data_collect is not None:
            return "纯数据MPC采样正在运行；需要停止请按急停。"
        if self._phi_gradient_test is not None:
            return "φ模型梯度验证正在运行；需要停止请按急停。"
        fr = self._latest_frame
        if fr is None or time.time()-fr["wall"] > 1.0:
            return "还没有收到新的固件数据帧。"
        if fr.get("adc_raw") is None or len(fr.get("locks", "")) != 6:
            return "当前固件没有发送 ADC 原始值，请先烧录最新版 teach_mode.ino。"
        self._adc_cal = {
            "stage": "zero", "stage_t": time.time(), "t0": time.time(),
            # DAC 基准 0.09 kPa/mV；全系统硬上限2200mV，对应198kPa。
            "levels": [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2200], "level_i": 0,
            "req": np.zeros(6), "raw_win": [], "points": [], "last_send": 0.0,
        }
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._adc_cal = None
            return "压力 ADC 标定启动失败：" + str(exc)
        self._adc_cal_emit("准备中：全机放气，正在采集 V0 的零压 ADC 原始值。")
        return "压力 ADC 标定已启动；只驱动 V0（肌肉1、5），过程中不要扶动机械臂。"

    def _adc_cal_emit(self, msg):
        print("# 压力 ADC 标定：" + msg)
        self.rx.emit("[adccal] " + msg + "\n")

    def _adc_cal_abort(self, why):
        if self._adc_cal is None:
            return
        self._adc_cal = None
        try:
            if self.ser: self.ser.write(b"s\n")
        except Exception:
            pass
        self._adc_cal_emit("标定已中止：" + why)

    def _adc_cal_send(self, cal):
        try:
            cmd = "W" + ",".join(str(int(v)) for v in cal["req"]) + "\n"
            self.ser.write(cmd.encode("ascii"))
            cal["last_send"] = time.time()
            return True
        except Exception as exc:
            self._adc_cal_abort("串口写入失败：" + str(exc))
            return False

    @staticmethod
    def _adc_raw_stable(values, samples=20, span=60.0, drift=15.0):
        """指定采样窗内 raw 波动及前后半窗漂移均足够小。新版固件帧率约20Hz。"""
        if len(values) < samples:
            return False
        a = np.asarray(values[-samples:], float)
        h = samples//2
        return float(np.ptp(a)) <= span and abs(float(np.mean(a[:h])-np.mean(a[h:]))) <= drift

    def _adc_cal_on_frame(self, fr):
        cal = self._adc_cal
        if cal is None:
            return
        raw = fr.get("adc_raw")
        if raw is None:
            self._adc_cal_abort("固件帧中的 ADC 原始值消失")
            return
        now = fr["wall"]
        if cal["stage"] != "zero" and now-cal["last_send"] >= 0.2:
            if not self._adc_cal_send(cal): return       # 给 W-stream 看门狗续命
        cal["raw_win"].append(float(raw[0]))
        cal["raw_win"] = cal["raw_win"][-240:]

        if cal["stage"] == "zero":
            if now-cal["stage_t"] >= 3.0 and fr["locks"] == "oooooo" and self._adc_raw_stable(cal["raw_win"]):
                raw0 = float(np.mean(cal["raw_win"][-20:]))
                cal["points"].append((raw0, 0.0, 0))
                lv = cal["levels"][0]; cal["req"][0] = lv
                cal["stage"] = "locking"; cal["stage_t"] = now; cal["raw_win"] = []
                if not self._adc_cal_send(cal): return
                self._adc_cal_emit(f"零点 raw={raw0:.1f}；V0 请求 {lv} mV，等待预锁。")
            elif now-cal["stage_t"] > 30.0:
                self._adc_cal_abort("零压 ADC 读数无法稳定")
            return

        if cal["stage"] == "locking":
            if fr["locks"][0] == "L":
                cal["stage"] = "settle"; cal["stage_t"] = now; cal["raw_win"] = []
                self._adc_cal_emit(
                    f"V0 已锁定，等待 {cal['levels'][cal['level_i']]} mV 压力连续稳定10秒。")
            elif now-cal["stage_t"] > 5.0:
                self._adc_cal_abort("V0 预锁超时")
            return

        if cal["stage"] == "settle":
            elapsed = now-cal["stage_t"]
            # 20Hz的200个样本全部来自同一电压，并且整个10秒窗均稳定才取均值。
            if elapsed >= 10.0 and self._adc_raw_stable(
                    cal["raw_win"], samples=200, span=90.0, drift=20.0):
                lv = cal["levels"][cal["level_i"]]
                rv = float(np.mean(cal["raw_win"][-200:]))
                cal["points"].append((rv, 0.09*lv, lv))
                self._adc_cal_emit(f"V0 {lv} mV 已稳定：raw={rv:.1f}，基准压力={0.09*lv:.1f} kPa。")
                cal["level_i"] += 1
                if cal["level_i"] < len(cal["levels"]):
                    lv = cal["levels"][cal["level_i"]]
                    cal["req"][0] = lv; cal["stage_t"] = now; cal["raw_win"] = []
                    if not self._adc_cal_send(cal): return
                    self._adc_cal_emit(f"V0 切换到 {lv} mV，重新累计连续稳定10秒。")
                else:
                    cal["req"][:] = 0; cal["stage"] = "final_zero"; cal["stage_t"] = now; cal["raw_win"] = []
                    if not self._adc_cal_send(cal): return
                    self._adc_cal_emit("九个压力点采集完成，V0 放气并等待重新回零。")
            elif elapsed > 90.0:
                self._adc_cal_abort(f"V0 {cal['levels'][cal['level_i']]} mV 等待稳定超时")
            return

        if cal["stage"] == "final_zero":
            if now-cal["stage_t"] >= 3.0 and fr["locks"] == "oooooo" and self._adc_raw_stable(cal["raw_win"]):
                points = cal["points"]
                x = np.asarray([p[0] for p in points], float)
                y = np.asarray([p[1] for p in points], float)
                if np.any(np.diff(x) < 20.0):
                    self._adc_cal_abort("ADC 原始值没有随 DAC 压力单调增加")
                    return
                dx = x-x[0]
                seg_slope = np.diff(y)/np.diff(dx)
                if np.any(seg_slope < 0.005) or np.any(seg_slope > 0.3):
                    self._adc_cal_abort("相邻压力点的斜率异常，不能建立可靠分段表")
                    return
                # 仍计算一条总直线及其 RMSE，仅作为非线性程度的诊断；实际使用分段插值。
                denom = float(dx @ dx)
                if denom < 1.0:
                    self._adc_cal_abort("ADC 原始值变化量太小")
                    return
                slope = float((dx @ y) / denom)
                intercept = float(-slope*x[0])
                pred = slope*dx
                rmse = float(np.sqrt(np.mean((pred-y)**2)))
                if not (0.01 <= slope <= 0.2):
                    self._adc_cal_abort(f"整体斜率异常：{slope:.6f} kPa/count")
                    return
                table = [{"delta_raw": float(dx[i]), "target_kpa": float(y[i])}
                         for i in range(len(x))]
                blob = {"schema": 2, "channel": 0, "muscles": [1, 5],
                        "formula": "piecewise_linear(delta_raw, target_kpa); delta_raw = adc_raw - per_channel_zero_raw",
                        "slope_kpa_per_count": float(slope), "intercept_kpa": float(intercept),
                        "linear_fit_rmse_kpa": rmse, "table": table,
                        "points": [{"dac_mv": p[2], "target_kpa": p[1], "adc_raw": p[0]} for p in points],
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                try:
                    tmp = ADC_CAL_PATH + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(blob, f, ensure_ascii=False, indent=2)
                    os.replace(tmp, ADC_CAL_PATH)
                    self._send_adc_table(blob)
                except Exception as exc:
                    self._adc_cal_abort("保存或下发公式失败：" + str(exc))
                    return
                self._adc_cal_applied = True
                self._adc_cal = None
                self._adc_cal_emit(
                    f"标定完成：已保存 {len(table)} 点分段表；总直线仅供诊断的 RMSE={rmse:.2f} kPa。"
                    "六路均按各自零点使用该分段表。")
            elif now-cal["stage_t"] > 35.0:
                self._adc_cal_abort("放气后 ADC 无法稳定回零")

    def _apply_saved_adc_cal(self):
        """连接新版固件后，把上次 GUI 标定系数恢复到本次 Arduino 运行。"""
        self._adc_cal_applied = True
        if not os.path.exists(ADC_CAL_PATH) or not self.ser:
            return
        try:
            b = json.load(open(ADC_CAL_PATH, encoding="utf-8"))
            if int(b.get("schema", 1)) >= 2 and len(b.get("table", [])) >= 2:
                self._send_adc_table(b)
                print(f"# 已向固件恢复压力 ADC {len(b['table'])} 点分段表（每路单独扣零点）")
            else:
                slope = float(b["slope_kpa_per_count"]); intercept = float(b["intercept_kpa"])
                self.ser.write(f"A{slope:.9f},{intercept:.6f}\n".encode("ascii"))
                print(f"# 已向固件恢复压力 ADC 斜率：{slope:.9f} kPa/count（每路单独扣零点）")
        except Exception as exc:
            print("# 压力 ADC 标定文件读取/下发失败：" + str(exc))

    def _send_adc_table(self, blob):
        """把分段表装入固件；T! 会在零压安全态重新采集六路 raw 零点。"""
        table = blob["table"]
        self.ser.write(b"T!\n")
        time.sleep(0.4)                         # 固件重采16次零点约需0.32s
        for i, p in enumerate(table):
            cmd = f"T{i},{float(p['delta_raw']):.3f},{float(p['target_kpa']):.3f}\n"
            self.ser.write(cmd.encode("ascii"))

    @QtCore.pyqtSlot(str, str, result=str)
    def startSpeedTest(self, mount, axis_group):
        """按gains.json:u_max单独测一个物理轴正、负方向的实测最大角速度。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        if self._manual_pressure_test is not None:
            return "六路手动压力测试正在运行，请先停止。"
        if self._reachability_test is not None:
            return "Reachability map 正在采样，请先停止。"
        if self._dm_ident is not None:
            return "DataModel 正在辨识，请先停止。"
        if self._adc_cal is not None:
            return "压力 ADC 正在标定，请先停止。"
        if self._speed_test is not None:
            return "最大速度测试已经在运行；需要停止请按急停。"
        if self._x_comp_test is not None:
            return "X耦合补偿测试正在运行；需要停止请按急停。"
        if self._mpc_data_collect is not None:
            return "纯数据MPC采样正在运行；需要停止请按急停。"
        if self._phi_gradient_test is not None:
            return "φ模型梯度验证正在运行；需要停止请按急停。"
        fr = self._latest_frame
        if fr is None or time.time()-fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能安全测速。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        if axis_group not in SPEED_AXIS_GROUPS:
            return "未知测速轴。"
        import mpc_station as st
        st.SRV["mount"] = mount
        directions = SPEED_AXIS_GROUPS[axis_group]
        cases = [(trial, dict(d)) for trial in range(1, 3) for d in directions]
        self._speed_test = {
            "mount": mount, "stage": "prepare", "stage_t": time.time(),
            "ready_t": None, "case_i": 0, "cases": cases, "qref": None,
            "req": np.zeros(6), "last_send": 0.0, "samples": [],
            "rows": [], "results": [], "started": time.time(),
            "axis_group": axis_group, "directions": directions,
        }
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._speed_test = None
            return "最大速度测试启动失败：" + str(exc)
        names = "、".join(d["name"] for d in directions)
        self._speed_emit(f"准备中：全机放气。本次只测 {names}，每个方向2轮。")
        return f"{SPEED_AXIS_LABELS[axis_group]} 轴速度测试已启动；请远离机械臂运动范围，急停随时有效。"

    def _speed_emit(self, msg):
        print("# 最大速度测试：" + msg)
        self.rx.emit("[speed] " + msg + "\n")

    def _speed_abort(self, why):
        if self._speed_test is None:
            return
        self._speed_test = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        self._speed_emit("测试已中止：" + why)

    def _speed_send(self, test):
        try:
            cmd = "W" + ",".join(str(int(v)) for v in test["req"]) + "\n"
            self.ser.write(cmd.encode("ascii"))
            test["last_send"] = time.time()
            return True
        except Exception as exc:
            self._speed_abort("串口写入失败：" + str(exc))
            return False

    @staticmethod
    def _speed_window_velocity(samples, axis, sign):
        """对最近约0.30秒做直线拟合，返回指定物理方向的角速度 deg/s。"""
        w = samples[-7:]
        if len(w) < 4:
            return None
        tt = np.asarray([s[0] for s in w], float)
        yy = np.asarray([s[1][axis] for s in w], float)
        tt -= float(np.mean(tt)); yy -= float(np.mean(yy))
        den = float(tt @ tt)
        return float(sign*(tt @ yy)/den) if den > 1e-9 else None

    def _speed_begin_case(self, test, fr):
        trial, case = test["cases"][test["case_i"]]
        test["qref"] = fr["quat"].copy()
        test["samples"] = []
        test["req"][:] = 0.0
        test["req"][case["ch"]] = configured_max_output_mv()
        test["stage"] = "locking"
        test["stage_t"] = fr["wall"]
        test["ready_t"] = None
        if self._speed_send(test):
            self._speed_emit(
                f"第{trial}/2轮 {case['name']}：V{case['ch']} 请求 {configured_max_output_mv()}mV，等待预锁。")

    def _speed_end_case(self, test, why):
        trial, case = test["cases"][test["case_i"]]
        vv = np.asarray([s[2] for s in test["samples"] if s[2] is not None and s[2] >= 0.0], float)
        peak = float(np.max(vv)) if len(vv) else 0.0
        p95 = float(np.percentile(vv, 95)) if len(vv) else 0.0
        disp = max((case["sign"]*s[1][case["axis"]] for s in test["samples"]), default=0.0)
        test["results"].append({"trial": trial, "direction": case["name"],
                                "channel": case["ch"], "peak_deg_s": peak,
                                "p95_deg_s": p95, "max_displacement_deg": float(disp),
                                "stop_reason": why})
        test["req"][:] = 0.0
        test["stage"] = "vent"
        test["stage_t"] = time.time()
        test["ready_t"] = None
        test["case_i"] += 1
        self._speed_send(test)
        self._speed_emit(f"第{trial}/2轮 {case['name']} 完成：滤波峰值 {peak:.1f}°/s，"
                         f"95%速度 {p95:.1f}°/s；正在放气。")

    def _speed_finish(self, test):
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        summary = {}
        for case in test["directions"]:
            rr = [r for r in test["results"] if r["direction"] == case["name"]]
            peaks = [r["peak_deg_s"] for r in rr]
            p95s = [r["p95_deg_s"] for r in rr]
            robust = float(min(p95s)) if p95s else 0.0
            summary[case["name"]] = {
                "channel": case["ch"],
                "max_filtered_deg_s": round(max(peaks) if peaks else 0.0, 3),
                "repeatable_p95_deg_s": round(robust, 3),
                "planning_deg_s": round(0.70*robust, 3),
                "trials": rr,
            }
        # 三个按钮分次测量：只合并同一新版坐标定义下已有的其他轴结果。
        merged = {}
        try:
            old_blob = json.load(open(SPEED_LIMIT_PATH, encoding="utf-8"))
            if (int(old_blob.get("coordinate_version", 0)) == 2 and
                    old_blob.get("mount") == test["mount"] and
                    int(old_blob.get("voltage_mv", 0)) == configured_max_output_mv()):
                merged.update(old_blob.get("directions", {}))
        except Exception:
            pass
        merged.update(summary)
        blob = {"schema": 2, "coordinate_version": 2,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mount": test["mount"], "voltage_mv": configured_max_output_mv(),
                "trials_per_direction": 2, "velocity_fit_window_s": 0.30,
                "planning_factor": 0.70, "directions": merged}
        log_dir = str(CONTROL_LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        raw_path = os.path.join(
            log_dir, time.strftime(f"speed_test_{test['axis_group']}_%Y%m%d_%H%M%S.csv"))
        try:
            with open(raw_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["trial", "direction", "channel", "t", "x", "y", "phi",
                             "q1", "q2", "q3",
                             "directional_velocity_deg_s"] + [f"P{i}" for i in range(6)] + ["locks"])
                wr.writerows(test["rows"])
            with open(SPEED_LIMIT_PATH, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._speed_test = None
            self._speed_emit("测试完成，但保存失败：" + str(exc))
            return
        self._speed_test = None
        values = "，".join(f"{name}={v['max_filtered_deg_s']:.1f}°/s" for name, v in summary.items())
        self._speed_emit(f"{SPEED_AXIS_LABELS[test['axis_group']]} 轴测试完成：{values}。"
                         f"结果已合并到 speed_limits.json；原始数据 → {raw_path}")

    def _speed_on_frame(self, fr):
        test = self._speed_test
        if test is None:
            return
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            self._speed_abort("压力或锁状态遥测中断")
            return
        now = fr["wall"]
        if test["stage"] in ("locking", "run", "vent") and now-test["last_send"] >= 0.15:
            if not self._speed_send(test):
                return

        empty = fr["locks"] == "oooooo" and float(np.max(fr["kpa"])) <= 30.0
        if test["stage"] in ("prepare", "vent"):
            if empty:
                if test["ready_t"] is None:
                    test["ready_t"] = now
                if now-test["ready_t"] >= 1.5:
                    if test["case_i"] >= len(test["cases"]):
                        self._speed_finish(test)
                    else:
                        self._speed_begin_case(test, fr)
            else:
                test["ready_t"] = None
            if now-test["stage_t"] > 35.0:
                self._speed_abort("等待放气/开锁超过35秒")
            return

        trial, case = test["cases"][test["case_i"]]
        if test["stage"] == "locking":
            if fr["locks"][case["ch"]] == "L":
                test["stage"] = "run"
                test["stage_t"] = now
                test["samples"] = []
                self._speed_emit(f"第{trial}/2轮 {case['name']}：开始测速，固定记录5秒。")
            elif now-test["stage_t"] > 5.0:
                self._speed_abort(f"{case['name']} 预锁超过5秒")
            return

        try:
            import mpc_station as st
            st.SRV["mount"] = test["mount"]
            q = st.q_model_from_quat(fr["quat"], test["qref"], test["mount"])
            pose = np.asarray(st.st3_from_q_model(q), float)
        except Exception as exc:
            self._speed_abort("姿态转换失败：" + str(exc))
            return
        elapsed = now-test["stage_t"]
        test["samples"].append((elapsed, pose.copy(), None))
        vel = self._speed_window_velocity(test["samples"], case["axis"], case["sign"])
        test["samples"][-1] = (elapsed, pose.copy(), vel)
        test["rows"].append([trial, case["name"], case["ch"], round(elapsed, 4),
                             *[round(float(v), 5) for v in pose],
                             *[round(float(v), 7) for v in q],
                             "" if vel is None else round(float(vel), 5),
                             *[round(float(v), 3) for v in fr["kpa"]], fr["locks"]])
        if elapsed >= 5.0:
            self._speed_end_case(test, "达到5秒")

    @QtCore.pyqtSlot(str, str, result=str)
    def startXCompTest(self, mount, test_key):
        """实机扫描 X 主肌与 φ 抵消肌的电压比例。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        if self._manual_pressure_test is not None:
            return "六路手动压力测试正在运行，请先停止。"
        if self._reachability_test is not None:
            return "Reachability map 正在采样，请先停止。"
        if self._dm_ident is not None:
            return "DataModel 正在辨识，请先停止。"
        if self._adc_cal is not None:
            return "压力 ADC 正在标定，请先停止。"
        if self._speed_test is not None:
            return "最大速度测试正在运行，请先停止。"
        if self._x_comp_test is not None:
            return "X耦合补偿测试已经在运行；需要停止请按急停。"
        if self._mpc_data_collect is not None:
            return "纯数据MPC采样正在运行；需要停止请按急停。"
        if self._phi_gradient_test is not None:
            return "φ模型梯度验证正在运行；需要停止请按急停。"
        if test_key not in X_COMP_TESTS:
            return "未知的 X 耦合补偿组合。"
        fr = self._latest_frame
        if fr is None or time.time()-fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能安全测试。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        case = dict(X_COMP_TESTS[test_key])
        self._x_comp_test = {
            **case, "test_key": test_key, "mount": mount,
            "ratios": list(X_COMP_TEST_RATIOS), "ratio_i": 0,
            "stage": "prepare", "stage_t": time.time(), "ready_t": None,
            "qref": None, "req": np.zeros(6), "last_send": 0.0,
            "samples": [], "rows": [], "results": [],
        }
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._x_comp_test = None
            return "X耦合补偿测试启动失败：" + str(exc)
        self._x_comp_emit(f"{case['label']} 准备中：全机放气；随后自动测试比例 " +
                          "/".join(f"{r:g}" for r in X_COMP_TEST_RATIOS) + "。")
        return f"{case['label']} 补偿比例测试已启动；请远离机械臂，急停随时有效。"

    def _x_comp_emit(self, msg):
        print("# X耦合补偿测试：" + msg)
        self.rx.emit("[xcomp] " + msg + "\n")

    def _x_comp_abort(self, why):
        if self._x_comp_test is None:
            return
        self._x_comp_test = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        self._x_comp_emit("测试已中止：" + why)

    def _x_comp_send(self, test):
        try:
            cmd = "W" + ",".join(str(int(v)) for v in test["req"]) + "\n"
            self.ser.write(cmd.encode("ascii"))
            test["last_send"] = time.time()
            return True
        except Exception as exc:
            self._x_comp_abort("串口写入失败：" + str(exc))
            return False

    def _x_comp_begin_case(self, test, fr):
        ratio = float(test["ratios"][test["ratio_i"]])
        test["qref"] = fr["quat"].copy()
        test["samples"] = []
        test["req"][:] = 0.0
        test["req"][test["primary"]] = X_COMP_TEST_MV
        test["req"][test["comp"]] = round(X_COMP_TEST_MV*ratio)
        test["stage"] = "locking"
        test["stage_t"] = fr["wall"]
        test["ready_t"] = None
        if self._x_comp_send(test):
            self._x_comp_emit(
                f"{test['label']} 比例 {ratio:g}：主通道 {X_COMP_TEST_MV}mV，"
                f"补偿通道 {int(test['req'][test['comp']])}mV，等待预锁。")

    def _x_comp_end_case(self, test, why):
        ratio = float(test["ratios"][test["ratio_i"]])
        if test["samples"]:
            tail = np.asarray([s[1] for s in test["samples"][-5:]], float)
            pose = np.median(tail, axis=0)
            elapsed = float(test["samples"][-1][0])
        else:
            pose = np.zeros(3)
            elapsed = 0.0
        directed_x = float(test["x_sign"]*pose[0])
        test["results"].append({
            "ratio": ratio, "elapsed_s": elapsed,
            "delta_x_deg": float(pose[0]), "delta_y_deg": float(pose[1]),
            "delta_phi_deg": float(pose[2]), "directed_x_deg": directed_x,
            "stop_reason": why,
        })
        test["req"][:] = 0.0
        test["stage"] = "vent"
        test["stage_t"] = time.time()
        test["ready_t"] = None
        test["ratio_i"] += 1
        self._x_comp_send(test)
        self._x_comp_emit(
            f"{test['label']} 比例 {ratio:g} 完成：ΔX={pose[0]:+.2f}°，"
            f"Δφ={pose[2]:+.2f}°；正在放气。")

    def _x_comp_finish(self, test):
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        results = test["results"]
        max_x = max((r["directed_x_deg"] for r in results), default=0.0)
        min_x = max(1.0, 0.40*max_x)
        eligible = [r for r in results
                    if r["elapsed_s"] >= 1.5 and r["directed_x_deg"] >= min_x]
        if not eligible:
            self._x_comp_test = None
            self._x_comp_emit(
                f"{test['label']} 测试失败：没有得到足够的有效 X 位移，未修改控制比例。")
            return
        for r in eligible:
            r["coupling_abs_phi_per_x"] = abs(r["delta_phi_deg"])/max(r["directed_x_deg"], 0.5)
        best = min(eligible, key=lambda r: (r["coupling_abs_phi_per_x"],
                                             abs(r["delta_phi_deg"])))
        ratio = float(best["ratio"])

        blob = {"schema": 1, "ratios": {}, "tests": {}}
        try:
            old = json.load(open(X_COMP_PATH, encoding="utf-8"))
            if int(old.get("schema", 0)) == 1:
                blob = old
        except Exception:
            pass
        blob.setdefault("ratios", {})[test["save_key"]] = ratio
        blob.setdefault("tests", {})[test["save_key"]] = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mount": test["mount"], "primary_mv": X_COMP_TEST_MV,
            "recommended_ratio": ratio, "selection": "min_abs_phi_per_effective_x",
            "results": results,
        }
        log_dir = str(CONTROL_LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        raw_path = os.path.join(
            log_dir, time.strftime(f"x_comp_{test['test_key']}_%Y%m%d_%H%M%S.csv"))
        try:
            tmp = X_COMP_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            os.replace(tmp, X_COMP_PATH)
            with open(raw_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["ratio", "t", "x", "y", "phi"] +
                            [f"P{i}" for i in range(6)] + ["locks"])
                wr.writerows(test["rows"])
            import mpc_station as st
            st.PI_X_PHI_COMP_RATIOS[int(test["primary"])] = ratio
        except Exception as exc:
            self._x_comp_test = None
            self._x_comp_emit("测试完成，但保存/应用失败：" + str(exc))
            return
        self._x_comp_test = None
        self._x_comp_emit(
            f"{test['label']} 测试完成：推荐补偿比例 {ratio:g}；"
            f"该点 ΔX={best['delta_x_deg']:+.2f}°、Δφ={best['delta_phi_deg']:+.2f}°。"
            f"已立即应用并保存；原始数据 → {raw_path}")

    def _x_comp_on_frame(self, fr):
        test = self._x_comp_test
        if test is None:
            return
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            self._x_comp_abort("压力或锁状态遥测中断")
            return
        now = fr["wall"]
        if test["stage"] in ("locking", "run", "vent") and now-test["last_send"] >= 0.15:
            if not self._x_comp_send(test):
                return

        empty = fr["locks"] == "oooooo" and float(np.max(fr["kpa"])) <= 30.0
        if test["stage"] in ("prepare", "vent"):
            if empty:
                if test["ready_t"] is None:
                    test["ready_t"] = now
                if now-test["ready_t"] >= 1.5:
                    if test["ratio_i"] >= len(test["ratios"]):
                        self._x_comp_finish(test)
                    else:
                        self._x_comp_begin_case(test, fr)
            else:
                test["ready_t"] = None
            if self._x_comp_test is not None and now-test["stage_t"] > 35.0:
                self._x_comp_abort("等待放气/开锁超过35秒")
            return

        ratio = float(test["ratios"][test["ratio_i"]])
        active_channels = [test["primary"]]
        if ratio > 0.0:
            active_channels.append(test["comp"])
        if test["stage"] == "locking":
            if all(fr["locks"][ch] == "L" for ch in active_channels):
                test["stage"] = "run"
                test["stage_t"] = now
                test["samples"] = []
                self._x_comp_emit(f"{test['label']} 比例 {ratio:g}：开始采集，持续2秒。")
            elif now-test["stage_t"] > 5.0:
                self._x_comp_abort(f"{test['label']} 比例 {ratio:g} 预锁超过5秒")
            return

        try:
            import mpc_station as st
            st.SRV["mount"] = test["mount"]
            q = st.q_model_from_quat(fr["quat"], test["qref"], test["mount"])
            pose = np.asarray(st.st3_from_q_model(q), float)
        except Exception as exc:
            self._x_comp_abort("姿态转换失败：" + str(exc))
            return
        elapsed = now-test["stage_t"]
        test["samples"].append((elapsed, pose.copy()))
        test["rows"].append([
            ratio, round(elapsed, 4), *[round(float(v), 5) for v in pose],
            *[round(float(v), 3) for v in fr["kpa"]], fr["locks"]])
        if max(abs(float(v)) for v in pose) >= 20.0:
            self._x_comp_end_case(test, "任一轴达到20°安全边界")
        elif elapsed >= 2.0:
            self._x_comp_end_case(test, "达到2秒")

    @QtCore.pyqtSlot(str, str, str, result=str)
    def startPhiGradientTest(self, mount, control_mode, model_source):
        """实机与真正模型并行测试 u2/u3/u4/u5→φ 及X补偿组合的多拍响应。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        if any(x is not None for x in (
                self._dm_ident, self._adc_cal, self._speed_test,
                self._x_comp_test, self._mpc_data_collect,
                self._phi_gradient_test,
                self._manual_pressure_test, self._reachability_test)):
            return "已有标定/测试正在运行；需要停止请按急停。"
        fr = self._latest_frame
        if fr is None or time.time()-fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能安全验证。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        try:
            import mpc_station as st
            source = str(model_source or "latest").lower()
            mode = str(control_mode or "mpc").lower()
            if mode == "new_mpc_fel" and source == "original":
                model_path = (st.ORIGINAL_INIT_FORWARD_ACTIVE_PATH
                              if os.path.exists(st.ORIGINAL_INIT_FORWARD_ACTIVE_PATH)
                              else st.ORIGINAL_FORWARD_PATH)
                meta_path = st.ORIGINAL_FORWARD_META_PATH
                model_label = "原始初始化分支真正模型"
            elif mode == "mpc_shadow":
                model_path = (st.MPC_SHADOW_FORWARD_ACTIVE_PATH
                              if os.path.exists(st.MPC_SHADOW_FORWARD_ACTIVE_PATH)
                              else st.PURE_DM_PATH)
                meta_path = st.PURE_DM_META_PATH
                model_label = "无逆模型影子分支真正模型"
            elif mode == "new_mpc_fel":
                model_path = (st.NEW_MPC_FEL_ACTIVE_PATH
                              if os.path.exists(st.NEW_MPC_FEL_ACTIVE_PATH)
                              else st.PURE_DM_PATH)
                meta_path = st.PURE_DM_META_PATH
                model_label = "最新影子体系真正模型"
            else:
                model_path = st.PURE_DM_PATH
                meta_path = st.PURE_DM_META_PATH
                model_label = "纯MPC真正模型"
            model = st.PureDataModel(model_path=model_path, meta_path=meta_path)
        except Exception as exc:
            return "真正模型加载失败：" + str(exc)

        cases = []
        for channel in PHI_GRAD_CHANNELS:
            for level in PHI_GRAD_LEVELS_MV:
                command = np.zeros(6)
                command[channel["ch"]] = level
                cases.append({**channel, "level": level, "kind": "single",
                              "u": command, "active_channels": [channel["ch"]]})
        ratios = {"9_plus_67": 0.5, "10_plus_23": 0.5}
        compensation_mv = 800
        try:
            saved = json.load(open(X_COMP_PATH, encoding="utf-8"))
            for key in ratios:
                ratios[key] = float(saved.get("ratios", {}).get(key, ratios[key]))
        except Exception:
            pass
        for name, primary, comp, ratio_key in (
                ("9+67", 4, 3, "9_plus_67"),
                ("10+23", 5, 2, "10_plus_23")):
            ratio = float(np.clip(ratios[ratio_key], 0.0, 1.5))
            command = np.zeros(6)
            command[primary] = compensation_mv
            command[comp] = round(compensation_mv*ratio)
            cases.append({
                "name": name, "ch": primary, "level": compensation_mv,
                "kind": "compensation", "expected_sign": 0.0,
                "ratio": ratio, "u": command,
                "active_channels": [ch for ch in (primary, comp) if command[ch] > 0],
            })
        self._phi_gradient_test = {
            "mount": mount, "model": model, "model_path": model_path,
            "model_label": model_label, "cases": cases, "case_i": 0,
            "stage": "prepare", "stage_t": time.time(), "ready_t": None,
            "qref": None, "req": np.zeros(6), "last_send": 0.0,
            "samples": [], "rows": [], "results": [], "prediction": None,
        }
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._phi_gradient_test = None
            return "φ梯度验证启动失败：" + str(exc)
        self._phi_gradient_emit(
            f"准备中：全机放气；使用{model_label}。先测试23/67/9/10单路的"
            "600/1200/1800mV，再用已标定比例验证9+67、10+23补偿；均比较5拍和10拍φ响应。")
        return "φ模型梯度验证已启动；请远离机械臂，急停随时有效。"

    def _phi_gradient_emit(self, msg):
        print("# φ模型梯度验证：" + msg)
        self.rx.emit("[phigrad] " + msg + "\n")

    def _phi_gradient_abort(self, why):
        if self._phi_gradient_test is None:
            return
        self._phi_gradient_test = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        self._phi_gradient_emit("验证已中止：" + why)

    def _phi_gradient_send(self, test):
        try:
            cmd = "W" + ",".join(str(int(v)) for v in test["req"]) + "\n"
            self.ser.write(cmd.encode("ascii"))
            test["last_send"] = time.time()
            return True
        except Exception as exc:
            self._phi_gradient_abort("串口写入失败：" + str(exc))
            return False

    @staticmethod
    def _phi_gradient_model_rollout(test, x0, case):
        """返回恒定阶跃相对零输入基线的20拍GUI姿态差，避免模型自身漂移污染梯度。"""
        import mpc_station as st
        model = test["model"]
        u_step = np.asarray(case["u"], dtype=float).copy()
        u_zero = np.zeros(6, dtype=float)
        step_state = np.asarray(x0, float).copy()
        zero_state = np.asarray(x0, float).copy()
        prediction = []
        for _ in range(20):
            step_state = model._predict_batch(step_state[None, :], u_step[None, :])[0]
            zero_state = model._predict_batch(zero_state[None, :], u_zero[None, :])[0]
            for state in (step_state, zero_state):
                state[3:6] = np.clip(state[3:6], -0.3, 0.3)
                state[6:] = np.clip(state[6:], 0.0, st.P_MAX_KPA)
            step_pose = np.asarray(st.st3_from_q_model(step_state[:3]), float)
            zero_pose = np.asarray(st.st3_from_q_model(zero_state[:3]), float)
            prediction.append(step_pose-zero_pose)
        return np.asarray(prediction, float)

    def _phi_gradient_begin_case(self, test, fr):
        case = test["cases"][test["case_i"]]
        test["qref"] = fr["quat"].copy()
        test["samples"] = []
        test["prediction"] = None
        test["req"][:] = np.asarray(case["u"], float)
        test["stage"] = "locking"
        test["stage_t"] = fr["wall"]
        test["ready_t"] = None
        if self._phi_gradient_send(test):
            command_text = "/".join(
                f"u{ch}={int(test['req'][ch])}" for ch in case["active_channels"])
            self._phi_gradient_emit(
                f"{test['case_i']+1}/{len(test['cases'])}：{case['name']} "
                f"{command_text}mV，等待预锁。")

    @staticmethod
    def _phi_gradient_actual_at(samples, target_s):
        if not samples:
            return None
        sample = min(samples, key=lambda item: abs(item[0]-target_s))
        return sample if abs(sample[0]-target_s) <= 0.12 else None

    def _phi_gradient_end_case(self, test, why):
        case = test["cases"][test["case_i"]]
        pred = test.get("prediction")
        actual5 = self._phi_gradient_actual_at(test["samples"], 0.5)
        actual10 = self._phi_gradient_actual_at(test["samples"], 1.0)
        a5 = float(actual5[1][2]) if actual5 is not None else float("nan")
        a10 = float(actual10[1][2]) if actual10 is not None else float("nan")
        p5 = float(pred[9, 2]) if pred is not None and len(pred) >= 10 else float("nan")
        p10 = float(pred[19, 2]) if pred is not None and len(pred) >= 20 else float("nan")
        expected = float(case["expected_sign"])
        is_compensation = case.get("kind") == "compensation"
        actual_direction_ok = bool(np.isfinite(a10) and expected*a10 >= 0.2) \
            if not is_compensation else None
        model_direction_ok = bool(np.isfinite(p10) and expected*p10 >= 0.2) \
            if not is_compensation else None
        gradient_match = bool(actual_direction_ok and model_direction_ok and a10*p10 > 0.0) \
            if not is_compensation else None
        gain_ratio = (abs(p10/a10) if np.isfinite(a10) and np.isfinite(p10)
                      and abs(a10) >= 0.2 else float("nan"))
        compensation_ok = (bool(np.isfinite(a10) and np.isfinite(p10) and
                                abs(a10) <= 2.0 and abs(p10-a10) <= 2.0)
                           if is_compensation else None)
        max_pressure = {
            str(ch): max((float(s[2][ch]) for s in test["samples"]), default=0.0)
            for ch in case["active_channels"]}
        result = {
            "muscle": case["name"], "channel": case["ch"], "level_mv": case["level"],
            "kind": case.get("kind", "single"),
            "command_mv": [int(v) for v in case["u"]],
            "actual_phi_5_deg": a5, "model_phi_5_deg": p5,
            "actual_phi_10_deg": a10, "model_phi_10_deg": p10,
            "expected_sign": expected, "actual_direction_ok": actual_direction_ok,
            "model_direction_ok": model_direction_ok, "gradient_direction_match": gradient_match,
            "compensation_ok": compensation_ok,
            "model_to_actual_gain_10": gain_ratio, "max_pressure_kpa": max_pressure,
            "stop_reason": why,
        }
        test["results"].append(result)
        test["req"][:] = 0.0
        test["stage"] = "vent"
        test["stage_t"] = time.time()
        test["ready_t"] = None
        test["case_i"] += 1
        self._phi_gradient_send(test)
        if is_compensation:
            mark = "✓" if compensation_ok else "✗"
            self._phi_gradient_emit(
                f"{case['name']}（比例{case.get('ratio', 0):.2f}）：5拍残余φ 实{a5:+.2f}°/模{p5:+.2f}°，"
                f"10拍残余φ 实{a10:+.2f}°/模{p10:+.2f}°，补偿{mark}；正在放气。")
        else:
            ratio_text = f"{gain_ratio:.2f}" if np.isfinite(gain_ratio) else "--"
            mark = "✓" if gradient_match else "✗"
            self._phi_gradient_emit(
                f"{case['name']} {case['level']}mV：5拍 实{a5:+.2f}°/模{p5:+.2f}°，"
                f"10拍 实{a10:+.2f}°/模{p10:+.2f}°，方向{mark}，模型/真机增益={ratio_text}；正在放气。")

    def _phi_gradient_finish(self, test):
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        log_dir = str(CONTROL_LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(log_dir, f"phi_gradient_{stamp}.csv")
        json_path = os.path.join(log_dir, f"phi_gradient_{stamp}.json")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                wr.writerow(["muscle", "channel", "level_mv", "t", "x", "y", "phi",
                             "model_delta_phi"] + [f"P{i}" for i in range(6)] + ["locks"])
                wr.writerows(test["rows"])
            payload = {
                "schema": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mount": test["mount"], "model": test["model_label"],
                "model_path": os.path.abspath(test["model_path"]),
                "definition": "constant uact step minus zero-input model baseline",
                "results": test["results"],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._phi_gradient_test = None
            self._phi_gradient_emit("验证完成，但日志保存失败：" + str(exc))
            return
        singles = [r for r in test["results"] if r.get("kind") == "single"]
        compensations = [r for r in test["results"] if r.get("kind") == "compensation"]
        good = sum(bool(r["gradient_direction_match"]) for r in singles)
        comp_good = sum(bool(r["compensation_ok"]) for r in compensations)
        ratios = [r["model_to_actual_gain_10"] for r in singles
                  if np.isfinite(r["model_to_actual_gain_10"])]
        ratio_text = f"{float(np.median(ratios)):.2f}" if ratios else "--"
        self._phi_gradient_test = None
        self._phi_gradient_emit(
            f"验证完成：23/67/9/10单路方向一致 {good}/{len(singles)}，"
            f"10拍模型/真机增益中位数 {ratio_text}；标定补偿通过 {comp_good}/{len(compensations)}；"
            f"结果 → {json_path}")

    def _phi_gradient_on_frame(self, fr):
        test = self._phi_gradient_test
        if test is None:
            return
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            self._phi_gradient_abort("压力或锁状态遥测中断")
            return
        now = fr["wall"]
        if test["stage"] in ("locking", "run", "vent") and now-test["last_send"] >= 0.15:
            if not self._phi_gradient_send(test):
                return
        empty = fr["locks"] == "oooooo" and float(np.max(fr["kpa"])) <= 30.0
        if test["stage"] in ("prepare", "vent"):
            if empty:
                if test["ready_t"] is None:
                    test["ready_t"] = now
                if now-test["ready_t"] >= 1.5:
                    if test["case_i"] >= len(test["cases"]):
                        self._phi_gradient_finish(test)
                    else:
                        self._phi_gradient_begin_case(test, fr)
            else:
                test["ready_t"] = None
            if self._phi_gradient_test is not None and now-test["stage_t"] > 35.0:
                self._phi_gradient_abort("等待放气/开锁超过35秒")
            return

        case = test["cases"][test["case_i"]]
        if test["stage"] == "locking":
            if all(fr["locks"][ch] == "L" for ch in case["active_channels"]):
                try:
                    import mpc_station as st
                    q0 = st.q_model_from_quat(fr["quat"], test["qref"], test["mount"])
                    x0 = np.r_[q0, np.zeros(3), np.asarray(fr["kpa"], float)]
                    test["prediction"] = self._phi_gradient_model_rollout(test, x0, case)
                except Exception as exc:
                    self._phi_gradient_abort("模型滚动失败：" + str(exc))
                    return
                test["stage"] = "run"
                test["stage_t"] = now
                test["samples"] = []
                self._phi_gradient_emit(
                    f"{case['name']} {case['level']}mV：开始记录真实/模型响应，持续1秒。")
            elif now-test["stage_t"] > 5.0:
                self._phi_gradient_abort(f"{case['name']} {case['level']}mV预锁超过5秒")
            return

        try:
            import mpc_station as st
            q = st.q_model_from_quat(fr["quat"], test["qref"], test["mount"])
            pose = np.asarray(st.st3_from_q_model(q), float)
        except Exception as exc:
            self._phi_gradient_abort("姿态转换失败：" + str(exc))
            return
        elapsed = now-test["stage_t"]
        test["samples"].append((elapsed, pose.copy(), np.asarray(fr["kpa"], float).copy()))
        pred = test["prediction"]
        pred_i = int(np.clip(round(elapsed/0.05)-1, 0, len(pred)-1))
        test["rows"].append([
            case["name"], case["ch"], case["level"], round(elapsed, 4),
            *[round(float(v), 5) for v in pose], round(float(pred[pred_i, 2]), 5),
            *[round(float(v), 3) for v in fr["kpa"]], fr["locks"]])
        if max(abs(float(v)) for v in pose) >= 35.0:
            self._phi_gradient_end_case(test, "任一轴达到35°")
        elif elapsed >= 1.05:
            self._phi_gradient_end_case(test, "达到1秒")

    @QtCore.pyqtSlot(str, result=str)
    def startPureMpcDataCollection(self, mount):
        """采集纯实机数据模型需要的持续激励数据，不调用原刚体/PAM动力学。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        if self._manual_pressure_test is not None:
            return "六路手动压力测试正在运行，请先停止。"
        if self._reachability_test is not None:
            return "Reachability map 正在采样，请先停止。"
        if self._dm_ident is not None or self._adc_cal is not None:
            return "辨识或压力标定正在运行，请先停止。"
        if self._speed_test is not None or self._x_comp_test is not None:
            return "实机测试正在运行，请先停止。"
        if self._mpc_data_collect is not None:
            return "纯数据MPC采样已经在运行；需要停止请按急停。"
        if self._phi_gradient_test is not None:
            return "φ模型梯度验证正在运行；需要停止请按急停。"
        fr = self._latest_frame
        if fr is None or time.time()-fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能安全采样。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"

        muscle_names = ("15", "48", "23", "67", "9", "10")
        seed = int(time.time()*1000) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        candidates = []
        # 26种候选工况：6个单通道 + 12个跨轴双通道 + 8个三轴通道。
        # 三通道从每组拮抗对各取一侧，是不违反拮抗硬约束时允许的最大并发数。
        for ch in range(6):
            levels = list(MPC_DATA_LEVELS_MV)
            tail = levels[1:]; rng.shuffle(tail); levels = [levels[0]] + tail
            steps = []
            for level in levels:
                u = np.zeros(6); u[ch] = level
                steps.append({"duration": float(rng.uniform(0.60, 0.90)), "u": u})
            candidates.append({"kind": "single",
                               "name": f"single_V{ch}_M{muscle_names[ch]}", "steps": steps})
        axis_pairs = (((0, 1), (2, 3)), ((0, 1), (4, 5)), ((2, 3), (4, 5)))
        for group_a, group_b in axis_pairs:
            for ca in group_a:
                for cb in group_b:
                    levels = list(MPC_DATA_PAIR_LEVELS_MV)
                    tail = levels[1:]; rng.shuffle(tail); levels = [levels[0]] + tail
                    steps = []
                    for la, lb in levels:
                        u = np.zeros(6); u[ca] = la; u[cb] = lb
                        steps.append({"duration": float(rng.uniform(0.65, 0.90)), "u": u})
                    candidates.append({"kind": "pair", "name": f"pair_V{ca}_V{cb}",
                                       "steps": steps})
        for cy in (0, 1):
            for cp in (2, 3):
                for cx in (4, 5):
                    levels = list(MPC_DATA_TRIPLE_LEVELS_MV)
                    tail = levels[1:]; rng.shuffle(tail); levels = [levels[0]] + tail
                    steps = []
                    for ly, lp, lx in levels:
                        u = np.zeros(6); u[cy] = ly; u[cp] = lp; u[cx] = lx
                        steps.append({"duration": float(rng.uniform(0.65, 0.90)), "u": u})
                    candidates.append({"kind": "triple",
                                       "name": f"triple_V{cy}_V{cp}_V{cx}", "steps": steps})
        # 每轮仍严格5种，但保证一定含2个单通道、2个双通道和1个三通道。
        # Each button press runs only five cases so one bad case does not ruin
        # a long collection.  Always cover all excitation dimensions: two
        # single-channel, two two-channel and one three-channel case.
        by_kind = {kind: [c for c in candidates if c["kind"] == kind]
                   for kind in ("single", "pair", "triple")}
        cases = ([by_kind["single"][int(i)] for i in rng.choice(6, 2, replace=False)] +
                 [by_kind["pair"][int(i)] for i in rng.choice(12, 2, replace=False)] +
                 [by_kind["triple"][int(i)] for i in rng.choice(8, 1, replace=False)])
        rng.shuffle(cases)

        self._mpc_data_collect = {
            "mount": mount, "stage": "prepare", "stage_t": time.time(),
            "ready_t": None, "qref": None, "case_i": 0, "step_i": 0,
            "cases": cases, "req": np.zeros(6), "last_send": 0.0,
            "rows": [], "t0": time.time(), "case_pose0": np.zeros(3), "seed": seed,
        }
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._mpc_data_collect = None
            return "纯数据MPC采样启动失败：" + str(exc)
        self._mpc_data_emit(
            f"准备中：本轮采集{len(cases)}种工况（单通道2 + 双通道2 + 三通道1）；"
            "每种5档并在工况间完全放气，"
            f"发送上限由 gains.json:u_max={configured_max_output_mv()}mV 决定，不设角度截断。")
        return "纯数据MPC采样已启动；不要扶动机械臂，请远离运动范围，急停随时有效。"

    def _mpc_data_emit(self, msg):
        print("# 纯数据MPC采样：" + msg)
        self.rx.emit("[mpcdata] " + msg + "\n")

    def _mpc_data_send(self, test):
        try:
            cmd = "W" + ",".join(str(int(np.clip(v, 0, configured_max_output_mv())))
                                  for v in test["req"]) + "\n"
            self.ser.write(cmd.encode("ascii"))
            test["last_send"] = time.time()
            return True
        except Exception as exc:
            self._mpc_data_abort("串口写入失败：" + str(exc))
            return False

    def _mpc_data_save(self, test, partial=False):
        if not test.get("rows"):
            return None
        log_dir = str(CONTROL_LOG_DIR / "pure_mpc_data")
        os.makedirs(log_dir, exist_ok=True)
        suffix = "_partial" if partial else ""
        stem = time.strftime("pure_mpc_%Y%m%d_%H%M%S") + suffix
        path = os.path.join(log_dir, stem + ".csv")
        header = (["t", "case_index", "case_name", "phase", "step_index",
                   "x_deg", "y_deg", "phi_deg", "q1_rad", "q2_rad", "q3_rad",
                   "qw", "qx", "qy", "qz"] +
                  [f"u_req{i}" for i in range(6)] + [f"u_actual{i}" for i in range(6)] +
                  [f"P{i}_kpa" for i in range(6)] + ["locks"] +
                  [f"sensor{i}" for i in range(1, 21)])
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(header); wr.writerows(test["rows"])
        meta = {
            "schema": 1, "source": "real_hardware_only", "uses_physics_model": False,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mount": test["mount"],
            "random_seed": int(test["seed"]),
            "nominal_frame_hz": 20, "max_output_mv": configured_max_output_mv(),
            "angle_abort_enabled": False,
            "completed_cases": test["case_i"], "total_cases": len(test["cases"]),
            "partial": bool(partial), "columns": header,
            "excitation_plan": [
                {"kind": case.get("kind", "unknown"), "name": case["name"], "steps": [
                    {"duration_s": round(float(step["duration"]), 4),
                     "u_mv": [int(v) for v in step["u"]]}
                    for step in case["steps"]]}
                for case in test["cases"]],
        }
        with open(os.path.join(log_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path

    def _mpc_data_abort(self, why):
        test = self._mpc_data_collect
        if test is None:
            return
        self._mpc_data_collect = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        try:
            path = self._mpc_data_save(test, partial=True)
        except Exception:
            path = None
        self._mpc_data_emit("采样已中止：" + why + (f"；已保存部分数据 → {path}" if path else ""))

    def _mpc_data_begin_case(self, test, fr, pose):
        case = test["cases"][test["case_i"]]
        test["step_i"] = 0
        test["case_pose0"] = pose.copy()
        test["req"] = case["steps"][0]["u"].copy()
        test["stage"] = "locking"
        test["stage_t"] = fr["wall"]
        test["ready_t"] = None
        if self._mpc_data_send(test):
            active = [f"V{i}={int(v)}" for i, v in enumerate(test["req"]) if v > 0]
            self._mpc_data_emit(
                f"{test['case_i']+1}/{len(test['cases'])} {case['name']}："
                f"{', '.join(active)}，等待预锁。")

    def _mpc_data_finish(self, test):
        try:
            if self.ser:
                self.ser.write(b"s\n")
            path = self._mpc_data_save(test, partial=False)
        except Exception as exc:
            self._mpc_data_collect = None
            self._mpc_data_emit("采样完成，但保存失败：" + str(exc))
            return
        self._mpc_data_collect = None
        self._mpc_data_emit(
            f"采样完成：{len(test['rows'])}帧、{len(test['cases'])}组，数据 → {path}。"
            "该文件只含实机测量，下一步可直接拟合纯数据模型。")

    def _mpc_data_on_frame(self, fr):
        test = self._mpc_data_collect
        if test is None:
            return
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            self._mpc_data_abort("压力或锁状态遥测中断")
            return
        now = fr["wall"]
        if test["stage"] in ("locking", "run", "vent") and now-test["last_send"] >= 0.15:
            if not self._mpc_data_send(test):
                return

        pose = None; q = None
        if test["qref"] is not None:
            try:
                import mpc_station as st
                st.SRV["mount"] = test["mount"]
                q = st.q_model_from_quat(fr["quat"], test["qref"], test["mount"])
                pose = np.asarray(st.st3_from_q_model(q), float)
                locks = fr["locks"]
                u_actual = np.where(np.asarray(list(locks)) == "L", test["req"], 0.0)
                case = (test["cases"][test["case_i"]]
                        if test["case_i"] < len(test["cases"]) else {"name": "done"})
                lens = fr.get("lens") or []
                test["rows"].append(
                    [round(now-test["t0"], 4), test["case_i"], case["name"],
                     test["stage"], test["step_i"],
                     *[round(float(v), 6) for v in pose],
                     *[round(float(v), 9) for v in q],
                     *[round(float(v), 9) for v in fr["quat"]],
                     *[int(v) for v in test["req"]], *[int(v) for v in u_actual],
                     *[round(float(v), 4) for v in fr["kpa"]], locks,
                     *(list(lens[:20]) + [None]*max(0, 20-len(lens)))])
            except Exception as exc:
                self._mpc_data_abort("姿态转换或记录失败：" + str(exc))
                return

        empty = fr["locks"] == "oooooo" and float(np.max(fr["kpa"])) <= 30.0
        if test["stage"] in ("prepare", "vent"):
            if empty:
                if test["ready_t"] is None:
                    test["ready_t"] = now
                if now-test["ready_t"] >= 1.0:
                    if test["qref"] is None:
                        test["qref"] = fr["quat"].copy()
                        q = np.zeros(3); pose = np.zeros(3)
                        self._mpc_data_emit("零压稳定，已建立本轮测量零位。")
                    if test["case_i"] >= len(test["cases"]):
                        self._mpc_data_finish(test)
                    else:
                        if pose is None:
                            try:
                                import mpc_station as st
                                q = st.q_model_from_quat(fr["quat"], test["qref"], test["mount"])
                                pose = np.asarray(st.st3_from_q_model(q), float)
                            except Exception as exc:
                                self._mpc_data_abort("建立工况起点失败：" + str(exc)); return
                        self._mpc_data_begin_case(test, fr, pose)
            else:
                test["ready_t"] = None
            if self._mpc_data_collect is not None and now-test["stage_t"] > 40.0:
                self._mpc_data_abort("等待放气/开锁超过40秒")
            return

        case = test["cases"][test["case_i"]]
        if test["stage"] == "locking":
            active = [i for i, v in enumerate(test["req"]) if v > 0]
            if all(fr["locks"][i] == "L" for i in active):
                test["stage"] = "run"; test["stage_t"] = now
                self._mpc_data_emit(
                    f"{test['case_i']+1}/{len(test['cases'])} {case['name']}：开始多档采样。")
            elif now-test["stage_t"] > 5.0:
                self._mpc_data_abort(case["name"] + " 预锁超过5秒")
            return

        step = case["steps"][test["step_i"]]
        if now-test["stage_t"] >= float(step["duration"]):
            test["step_i"] += 1
            if test["step_i"] < len(case["steps"]):
                test["req"] = case["steps"][test["step_i"]]["u"].copy()
                test["stage_t"] = now
                self._mpc_data_send(test)
            else:
                test["req"][:] = 0.0
                test["stage"] = "vent"; test["stage_t"] = now; test["ready_t"] = None
                test["case_i"] += 1
                self._mpc_data_send(test)
                self._mpc_data_emit(
                    f"{case['name']} 完成；已采 {len(test['rows'])} 帧，正在放气。")

    def _quick_health_emit(self, message):
        print("# 快速综合检测：" + message)
        self.rx.emit("[quick] " + message + "\n")

    def _quick_health_status(self, state, items=None, message="", report_path=""):
        payload = {
            "state": str(state), "message": str(message),
            "items": items or {}, "report_path": str(report_path or ""),
        }
        self.rx.emit("[quickjson] " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _python_environment_report():
        """Check the desktop runtime without importing optional training stacks or touching hardware."""
        required = (
            ("numpy", "numpy", "NumPy"),
            ("scipy", "scipy", "SciPy"),
            ("matplotlib", "matplotlib", "Matplotlib"),
            ("PyQt5", "PyQt5", "PyQt5"),
            ("PyQt5.QtWebEngineWidgets", "PyQtWebEngine", "Qt WebEngine"),
            ("serial", "pyserial", "pyserial"),
            ("openpyxl", "openpyxl", "openpyxl"),
            ("pyqtgraph", "pyqtgraph", "pyqtgraph"),
            ("torch", "torch", "PyTorch"),
            ("imageio_ffmpeg", "imageio-ffmpeg", "GUI视频编码器"),
        )
        checks = []
        python_ok = sys.version_info >= (3, 9)
        checks.append({
            "name": "Python", "ok": python_ok,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "detail": "要求 3.9 或更高版本",
        })
        for module_name, distribution, label in required:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            version = ""
            if available:
                try:
                    version = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    version = "已安装"
            checks.append({
                "name": label, "ok": available, "version": version,
                "detail": "" if available else f"缺少模块 {module_name}",
            })

        project_files = [
            (HTML, "GUI页面"),
            (os.path.join(os.path.dirname(__file__), "mpc_station.py"), "MPC控制站"),
            (os.path.join(os.path.dirname(__file__), "codex_compensator.py"), "离线补偿器"),
            (os.path.join(os.path.dirname(__file__), "codex_compensator_prompt.json"), "补偿器Prompt"),
            (os.path.join(os.path.dirname(__file__), "..", "shoulder_pam_model.py"), "默认动力学模型"),
            (os.path.join(os.path.dirname(__file__), "..", "shoulder_pam_model_v6.py"), "V6模型"),
            (GAINS_PATH, "控制参数"),
            (os.path.join(CONTROL_DIR, "force_predictor.py"), "力预测程序"),
            (FORCE_PREDICTOR_WORKER_PATH, "力预测隔离进程"),
            (os.path.join(FORCE_PREDICTOR_MODEL_DIR, "manifest.json"), "力预测模型清单"),
            (os.path.join(FORCE_PREDICTOR_MODEL_DIR, "train_scaler_stats.csv"), "力预测标准化参数"),
        ]
        for family in ("fx", "fy", "fz"):
            for seed in (2025, 2026, 2027):
                project_files.append((
                    os.path.join(FORCE_PREDICTOR_MODEL_DIR, family,
                                 f"model_seed{seed}.pt"),
                    f"力预测模型 {family.upper()} seed{seed}",
                ))
        missing_files = [label for path, label in project_files if not os.path.isfile(path)]
        checks.append({
            "name": "项目运行文件", "ok": not missing_files,
            "version": f"{len(project_files)-len(missing_files)}/{len(project_files)}",
            "detail": "" if not missing_files else "缺少：" + "、".join(missing_files),
        })
        failed = [item for item in checks if not item["ok"]]
        if failed:
            detail = "缺少或不兼容：" + "、".join(item["name"] for item in failed)
        else:
            detail = "环境检查通过"
        return {"state": "fail" if failed else "pass", "detail": detail, "checks": checks}

    @staticmethod
    def _environment_installable_packages(report):
        package_by_name = {
            "NumPy": "numpy", "SciPy": "scipy", "Matplotlib": "matplotlib",
            "PyQt5": "PyQt5", "Qt WebEngine": "PyQtWebEngine",
            "pyserial": "pyserial", "openpyxl": "openpyxl",
            "pyqtgraph": "pyqtgraph", "PyTorch": "torch",
            "GUI视频编码器": "imageio-ffmpeg",
        }
        return [package_by_name[item["name"]] for item in report.get("checks", [])
                if not item.get("ok") and item.get("name") in package_by_name]

    def _environment_emit(self, state, message, report=None):
        payload = {
            "state": str(state), "message": str(message),
            "environment": report or self._python_environment_report(),
        }
        self.rx.emit("[envjson] " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _install_python_packages_worker(self, packages):
        try:
            command = [sys.executable, "-m", "pip", "install",
                       "--disable-pip-version-check", *packages]
            options = {
                "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                "text": True, "encoding": "utf-8", "errors": "replace",
                "timeout": 900,
            }
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(command, **options)
            importlib.invalidate_caches()
            report = self._python_environment_report()
            if completed.returncode == 0 and report["state"] == "pass":
                self._environment_emit(
                    "complete", "缺失组件已自动安装，软件环境通过", report)
            else:
                tail = ""
                output = (completed.stdout or "").strip().splitlines()
                if output:
                    tail = "；" + output[-1][:240]
                self._environment_emit(
                    "failed", "自动安装未完成" + tail, report)
        except subprocess.TimeoutExpired:
            self._environment_emit("failed", "自动安装超过15分钟，已停止等待")
        except Exception as exc:
            self._environment_emit("failed", "自动安装失败：" + str(exc))
        finally:
            with self._environment_install_lock:
                self._environment_installing = False

    @QtCore.pyqtSlot(result=str)
    def installMissingPythonPackages(self):
        """Install only an allow-listed set of missing runtime packages in this interpreter."""
        report = self._python_environment_report()
        if report["state"] == "pass":
            return json.dumps({"ok": True, "started": False,
                               "message": "软件环境已经通过"}, ensure_ascii=False)
        packages = self._environment_installable_packages(report)
        if not packages:
            return json.dumps({
                "ok": False, "started": False,
                "message": "没有可自动安装的缺项；请检查Python版本或项目文件",
            }, ensure_ascii=False)
        with self._environment_install_lock:
            if self._environment_installing:
                return json.dumps({"ok": True, "started": False,
                                   "message": "环境组件正在安装"}, ensure_ascii=False)
            self._environment_installing = True
        self._environment_emit(
            "installing", "正在自动安装：" + "、".join(packages), report)
        threading.Thread(target=self._install_python_packages_worker,
                         args=(packages,), daemon=True).start()
        return json.dumps({
            "ok": True, "started": True, "packages": packages,
            "message": "正在后台自动安装：" + "、".join(packages),
        }, ensure_ascii=False)

    @QtCore.pyqtSlot(result=str)
    def checkPythonEnvironment(self):
        """Return the same environment result shown by the formal operation test page."""
        return json.dumps(self._python_environment_report(), ensure_ascii=False,
                          separators=(",", ":"))

    @QtCore.pyqtSlot(str, result=str)
    def startQuickHealthCheck(self, mount):
        """One short six-channel scan feeding all six operator health checks."""
        environment = self._python_environment_report()
        if environment["state"] != "pass":
            self._quick_health_status(
                "failed", {"environment": environment},
                "Python运行环境未通过，未启动实机扫描")
            return "Python运行环境未通过：" + environment["detail"]
        if not self.ser:
            return "先连接 IMU 串口。"
        if any(x is not None for x in (
                self._dm_ident, self._adc_cal, self._speed_test,
                self._x_comp_test, self._mpc_data_collect,
                self._phi_gradient_test,
                self._manual_pressure_test, self._reachability_test)):
            return "已有辨识、标定或实机测试正在运行，请先停止。"
        fr = self._latest_frame
        if fr is None or time.time()-fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能综合检测。"
        if self._v6_geometry_lengths(fr) is None:
            return "当前帧缺少完整的20路编码器数据，不能综合检测。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        high = configured_max_output_mv()
        levels = [min(600, high), min(1200, high), high,
                  min(1200, high), min(600, high)]
        now = time.time()
        self._dm_ident = {
            "kind": "quick_health", "channel_plan": list(range(6)), "plan_i": 0,
            "mount": mount, "qref": None, "stage": "rezero_wait", "stage_t": now,
            "t0": now, "ch": 0, "level_i": 0, "levels": levels, "hold_s": 1.0,
            "zero_n": 0, "settle_p": [], "req": np.zeros(6), "rows": [],
            "last_send": 0.0, "last_row_t": 0.0,
            "environment": environment,
        }
        self._quick_health_running = True
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._dm_ident = None
            self._quick_health_running = False
            return "快速综合检测启动失败：" + str(exc)
        pending = {key: {"state": "running", "detail": "等待采样"} for key in (
            "pressure", "muscle", "mapping", "imu", "speed")}
        pending["environment"] = environment
        self._quick_health_status("running", pending, "正在全机放气并建立零点")
        self._quick_health_emit(
            f"已启动：V0～V5各执行5秒五档扫描，峰值{high}mV；五项实机检测共用同一批数据。")
        return "快速综合检测已启动；请远离机械臂，急停随时有效。"

    @QtCore.pyqtSlot(result=str)
    def stopQuickHealthCheck(self):
        ident = self._dm_ident
        if ident is None or ident.get("kind") != "quick_health":
            return "快速综合检测未运行。"
        self._ident_abort("用户停止")
        return "快速综合检测已停止并全机放气。"

    @QtCore.pyqtSlot(str, result=str)
    def startDataModelIdent(self, mount):
        """一次六路升降压扫描，同时拟合 DataModel 和 V6 几何候选。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        if self._manual_pressure_test is not None:
            return "六路手动压力测试正在运行，请先停止。"
        if self._reachability_test is not None:
            return "Reachability map 正在采样，请先停止。"
        if self._dm_ident is not None:
            return "DataModel 正在辨识；需要停止请按急停。"
        if self._adc_cal is not None:
            return "压力 ADC 正在标定，请先停止。"
        if self._speed_test is not None:
            return "最大速度测试正在运行，请先停止。"
        if self._x_comp_test is not None:
            return "X耦合补偿测试正在运行，请先停止。"
        if self._mpc_data_collect is not None:
            return "纯数据MPC采样正在运行，请先停止。"
        if self._phi_gradient_test is not None:
            return "φ模型梯度验证正在运行，请先停止。"
        fr = self._latest_frame
        if fr is None or time.time() - fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr["kpa"] is None or len(fr["locks"]) != 6:
            return "当前固件帧没有六路压力/锁状态，不能辨识。"
        if self._v6_geometry_lengths(fr) is None:
            return "当前帧缺少完整的20路编码器数据，不能同时辨识V6几何。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        self._dm_ident = {
            "kind": "combined", "channel_plan": list(range(6)), "plan_i": 0,
            "mount": mount, "qref": None, "stage": "rezero_wait", "stage_t": time.time(),
            "t0": time.time(), "ch": 0, "level_i": 0,
            "levels": [600, 1200, 1800, 1200, 600], "hold_s": 5.0,
            "zero_n": 0, "settle_p": [], "req": np.zeros(6), "rows": [],
            "last_send": 0.0, "last_row_t": 0.0,
        }
        # 从全机安全放气态开始；通道间仍严格等待“压力=0且锁=o”，不会强制跳过。
        try:
            self.ser.write(b"s\n")
        except Exception as e:
            self._dm_ident = None
            return "辨识启动失败：" + str(e)
        self._ident_emit("准备中：全机放气，等待压力读数稳定后自动重新对零。")
        return ("DataModel + V6几何联合辨识已启动；一次自动扫描V0～V5的充气和放气段，"
                "过程中不要扶动机械臂。")

    @QtCore.pyqtSlot(str, int, result=str)
    def startV3RepeatTest(self, mount, repeat_count):
        """Repeat only V3; save validation reports without fitting any control model."""
        if not self.ser:
            return "先连接 IMU 串口。"
        if any(x is not None for x in (
                self._dm_ident, self._adc_cal, self._speed_test,
                self._x_comp_test, self._mpc_data_collect,
                self._phi_gradient_test,
                self._manual_pressure_test, self._reachability_test)):
            return "已有辨识、标定或实机测试正在运行，请先停止。"
        fr = self._latest_frame
        if fr is None or time.time() - fr["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        if fr.get("kpa") is None or len(fr.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能测试。"
        if self._v6_geometry_lengths(fr) is None:
            return "当前帧缺少完整的20路编码器数据，不能验证V3几何。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        repeats = int(np.clip(int(repeat_count), 1, 10))
        self._dm_ident = {
            "kind": "v3_repeat", "channel_plan": [3] * repeats, "plan_i": 0,
            "repeat_count": repeats,
            "mount": mount, "qref": None, "stage": "rezero_wait", "stage_t": time.time(),
            "t0": time.time(), "ch": 3, "level_i": 0,
            "levels": [600, 1200, 1800, 1200, 600], "hold_s": 5.0,
            "zero_n": 0, "settle_p": [], "req": np.zeros(6), "rows": [],
            "last_send": 0.0, "last_row_t": 0.0,
        }
        try:
            self.ser.write(b"s\n")
        except Exception as exc:
            self._dm_ident = None
            return "V3重复测试启动失败：" + str(exc)
        self._v3_repeat_emit(
            f"准备执行 {repeats} 次V3独立扫描；先全机放气并重新对零。"
            "结果只保存为验证报告，不写入DataModel、V6模型或六路候选。")
        return (f"V3重复测试已启动：共{repeats}次，每次自动执行"
                "600/1200/1800/1200/600 mV并完全放气。")

    def _ident_emit(self, msg):
        print("# DataModel 辨识：" + msg)
        self.rx.emit("[ident] " + msg + "\n")

    def _v3_repeat_emit(self, msg):
        print("# V3 重复测试：" + msg)
        self.rx.emit("[v3repeat] " + msg + "\n")

    def _ident_progress(self, msg, ident=None):
        test = self._dm_ident if ident is None else ident
        if test is not None and test.get("kind") == "v3_repeat":
            self._v3_repeat_emit(msg)
        elif test is not None and test.get("kind") == "quick_health":
            self._quick_health_emit(msg)
        else:
            self._ident_emit(msg)

    def _ident_abort(self, why):
        ident = self._dm_ident
        if ident is None:
            return
        self._dm_ident = None
        if ident.get("kind") == "quick_health":
            self._quick_health_running = False
        try:
            if self.ser:
                self.ser.write(b"s\n")             # 任何超时/异常都立即比例阀清零、六路锁全开
        except Exception:
            pass
        self._ident_progress("测试已中止：" + why, ident=ident)
        if ident.get("kind") == "quick_health":
            failed = {key: {"state": "fail", "detail": "检测中止"} for key in (
                "pressure", "muscle", "mapping", "imu", "speed")}
            failed["environment"] = ident.get(
                "environment") or self._python_environment_report()
            self._quick_health_status("failed", failed, "检测已中止：" + why)

    @staticmethod
    def _decode_frame(line):
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) < 9:
            return None
        try:
            qq = np.asarray([float(x) for x in parts[1].split()[:4]], float)
            if qq.size != 4 or not np.all(np.isfinite(qq)) or np.linalg.norm(qq) < 1e-6:
                return None
            qq /= np.linalg.norm(qq)
        except (TypeError, ValueError):
            return None
        try:
            pp = [float(x) for x in parts[7].split()[:6]]
            kpa = np.maximum(0.0, np.asarray(pp, float)) if len(pp) == 6 else None
            if kpa is not None and not np.all(np.isfinite(kpa)):
                kpa = None
        except (TypeError, ValueError):
            kpa = None
        lens = None
        try:
            ll = []
            for token in parts[6].split()[:20]:
                try:
                    value = float(token)
                    ll.append(value if np.isfinite(value) else None)
                except (TypeError, ValueError):
                    ll.append(None)
            if ll:
                lens = (ll + [None] * 20)[:20]
        except (TypeError, ValueError):
            lens = None
        locks = "".join(c for c in parts[8] if c in "owL")[:6]
        adc_raw = None
        if len(parts) >= 10:
            try:
                rr = [float(x) for x in parts[9].split()[:6]]
                if len(rr) == 6 and np.all(np.isfinite(rr)):
                    adc_raw = np.asarray(rr, float)
            except (TypeError, ValueError):
                pass
        muscle_length_mm = None
        channel_length_mm = None
        if lens is not None:
            muscle_length_mm = [
                (235.0 - abs(lens[i]) + abs(lens[i + 10])
                 if lens[i] is not None and lens[i + 10] is not None else None)
                for i in range(10)
            ]
            # 固定物理阀序：V0={1,5}, V1={4,8}, V2={2,3},
            # V3={6,7}, V4={9}, V5={10}。这里不得按旧几何顺序交换V1/V2。
            groups = ((0, 4), (3, 7), (1, 2), (5, 6), (8,), (9,))
            channel_length_mm = [
                (sum(muscle_length_mm[i] for i in group) / len(group)
                 if all(muscle_length_mm[i] is not None for i in group) else None)
                for group in groups
            ]
        return {"quat": qq, "kpa": kpa, "lens": lens,
                "muscle_length_mm": muscle_length_mm,
                "channel_length_mm": channel_length_mm, "locks": locks,
                "adc_raw": adc_raw, "wall": time.time()}

    def _ident_send(self, ident):
        if not self.ser:
            return False
        try:
            cmd = "W" + ",".join(str(int(v)) for v in ident["req"]) + "\n"
            self.ser.write(cmd.encode("ascii"))
            ident["last_send"] = time.time()
            return True
        except Exception as e:
            self._ident_abort("串口写入失败：" + str(e))
            return False

    def _ident_on_frame(self, fr):
        ident = self._dm_ident
        if ident is None or fr["kpa"] is None or len(fr["locks"]) != 6:
            return
        now = fr["wall"]
        if now - ident["last_send"] >= 0.2:
            if not self._ident_send(ident):         # 同时给固件 W-stream 看门狗续命
                return

        # 固件流和控制模型均为20Hz；辨识日志也固定抽样20Hz。
        if ident["qref"] is not None and now-ident.get("last_row_t", 0.0) >= 0.045:
            ident["last_row_t"] = now
            try:
                import mpc_station as st
                q = st.q_model_from_quat(fr["quat"], ident["qref"], ident["mount"])
                req = ident["req"].copy()
                uact = np.where(np.asarray(list(fr["locks"])) == "L", req, 0.0)
                ident["rows"].append({
                    "t": now-ident["t0"], "trial": ident["ch"],
                    "repeat": ident.get("plan_i", ident["ch"]), "stage": ident["stage"],
                    "level": int(req[ident["ch"]]) if ident["ch"] < 6 else 0,
                    "q": np.asarray(q, float), "u": req, "uact": uact,
                    "P": fr["kpa"].copy(), "locks": list(fr["locks"]),
                    "L": (None if self._v6_geometry_lengths(fr) is None else
                          self._v6_geometry_lengths(fr).copy()),
                    "M": (None if fr.get("muscle_length_mm") is None else
                          list(fr["muscle_length_mm"])),
                    "lens": (None if fr.get("lens") is None else list(fr["lens"])),
                })
            except Exception as e:
                self._ident_abort("姿态转换失败：" + str(e))
                return

        if ident["stage"] == "rezero_wait":
            ident["settle_p"].append(fr["kpa"].copy())
            ident["settle_p"] = ident["settle_p"][-10:]
            stable = (len(ident["settle_p"]) >= 10
                      and np.max(np.ptp(np.asarray(ident["settle_p"]), axis=0)) <= 1.0)
            if now - ident["stage_t"] >= 3.0 and stable:
                try:
                    self.ser.write(b"k\n")
                except Exception as e:
                    self._ident_abort("发送压力对零命令失败：" + str(e)); return
                ident["stage"] = "zero"
                ident["stage_t"] = now
                ident["zero_n"] = 0
                self._ident_progress("压力读数已稳定，正在重新采集六路传感器零点。")
            elif now - ident["stage_t"] > 30.0:
                self._ident_abort("放气后压力读数仍不稳定，拒绝错误对零")
            return

        # 30kPa 是固件开关阀的判据，不是 DataModel 的压力截断。
        # 因此这里以固件已经打开六路锁阀为准，不再要求遥测压力精确等于 0。
        all_zero = fr["locks"] == "oooooo"
        if ident["stage"] in ("zero", "vent"):
            ident["zero_n"] = ident["zero_n"] + 1 if all_zero else 0
            if ident["zero_n"] >= 3:
                if ident["qref"] is None:
                    ident["qref"] = fr["quat"].copy()       # 真正零压稳定后才取辨识零位
                if ident["stage"] == "vent":
                    ident["plan_i"] = ident.get("plan_i", ident["ch"]) + 1
                    channel_plan = ident.get("channel_plan", list(range(6)))
                    if ident["plan_i"] >= len(channel_plan):
                        self._ident_finish()
                        return
                    ident["ch"] = int(channel_plan[ident["plan_i"]])
                ident["level_i"] = 0
                ident["req"][:] = 0
                ident["req"][ident["ch"]] = ident["levels"][0]
                ident["stage"] = "locking"
                ident["stage_t"] = now
                ident["zero_n"] = 0
                if not self._ident_send(ident):
                    return
                suffix = (f"（重复 {ident.get('plan_i', 0) + 1}/"
                          f"{len(ident.get('channel_plan', list(range(6))))}）"
                          if ident.get("kind") == "v3_repeat" else "")
                self._ident_progress(
                    f"V{ident['ch']}{suffix}：请求 {ident['levels'][0]} mV，等待预锁完成。")
            elif now - ident["stage_t"] > 40.0:
                self._ident_abort("等待零压超时；当前压力 " + "/".join(f"{v:.0f}" for v in fr["kpa"]) + " kPa")
            return

        ch = ident["ch"]
        if ident["stage"] == "locking":
            if fr["locks"][ch] == "L" and now - ident["stage_t"] >= 0.100:
                ident["stage"] = "hold_up"
                ident["stage_t"] = now
                self._ident_progress(f"V{ch}：已预锁100 ms，保持 {ident['levels'][0]} mV。")
            elif now - ident["stage_t"] > 4.0:
                self._ident_abort(f"V{ch} 预锁超过 4 秒仍未进入 L")
            return

        if ident["stage"].startswith("hold") and now - ident["stage_t"] >= ident["hold_s"]:
            ident["level_i"] += 1
            if ident["level_i"] < len(ident["levels"]):
                lv = ident["levels"][ident["level_i"]]
                ident["req"][ch] = lv
                ident["stage"] = "hold_up" if ident["level_i"] <= 2 else "hold_down"
                ident["stage_t"] = now
                if not self._ident_send(ident):
                    return
                self._ident_progress(f"V{ch}：切换到 {lv} mV（{ident['level_i']+1}/5）。")
            else:
                ident["req"][:] = 0
                ident["stage"] = "vent"
                ident["stage_t"] = now
                ident["zero_n"] = 0
                if not self._ident_send(ident):
                    return
                self._ident_progress(f"V{ch}：扫描完成，等待实测压力回到 0 后开锁。")

    def _ident_finish(self):
        ident = self._dm_ident
        if ident is None:
            return
        try:
            self.ser.write(b"s\n")
        except Exception:
            pass
        rows = ident["rows"]
        kind = ident.get("kind", "combined")
        self._dm_ident = None
        if len(rows) < 100:
            if kind == "quick_health":
                self._quick_health_running = False
            self._ident_progress("测试失败：有效数据太少。", ident=ident)
            if kind == "quick_health":
                failed = {key: {"state": "fail", "detail": "有效数据太少"} for key in (
                    "pressure", "muscle", "mapping", "imu", "speed")}
                failed["environment"] = ident.get(
                    "environment") or self._python_environment_report()
                self._quick_health_status("failed", failed, "有效数据太少")
            return
        log_dir = str(CONTROL_LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        stem = ("v3_repeat_raw_" if kind == "v3_repeat" else
                "quick_health_raw_" if kind == "quick_health" else "dm_ident_raw_")
        path = os.path.join(log_dir, time.strftime(stem + "%Y%m%d_%H%M%S.csv"))
        v3_diagnostic_header = (
            [f"M{i}_mm" for i in range(1, 11)] +
            [f"sensor{i}" for i in range(1, 21)] if kind == "v3_repeat" else [])
        header = (["t", "trial"] + (["repeat"] if kind == "v3_repeat" else [])
                  + ["stage", "level", "q1", "q2", "q3"]
                  + [f"u{i}" for i in range(6)] + [f"uact{i}" for i in range(6)]
                  + [f"P{i}" for i in range(6)] + [f"lock{i}" for i in range(6)]
                  + [f"L{i}_mm" for i in range(6)] + v3_diagnostic_header)
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(header)
            for r in rows:
                wr.writerow([round(r["t"], 3), r["trial"]]
                            + ([r.get("repeat", 0) + 1] if kind == "v3_repeat" else [])
                            + [r["stage"], r["level"]]
                            + [round(float(np.degrees(v)), 5) for v in r["q"]]
                            + [int(v) for v in r["u"]] + [int(v) for v in r["uact"]]
                            + [round(float(v), 3) for v in r["P"]] + r["locks"]
                            + ([round(float(v), 6) for v in r["L"]]
                               if r.get("L") is not None else [""]*6)
                            + (([round(float(v), 6) if v is not None else ""
                                  for v in (r.get("M") or [None] * 10)]
                                + [v if v is not None else ""
                                   for v in (r.get("lens") or [None] * 20)])
                               if kind == "v3_repeat" else []))
        if kind == "v3_repeat":
            self._v3_repeat_emit(
                f"{ident.get('repeat_count', 1)}次V3采集完成（{len(rows)}帧），"
                "正在计算每次及总体Kendall tau-b/RMSE；不会更新任何模型。")
            threading.Thread(target=self._fit_v3_repeat_rows,
                             args=(rows, path, ident["mount"]), daemon=True).start()
        elif kind == "quick_health":
            self._quick_health_running = False
            self._quick_health_emit(
                f"采集完成（{len(rows)}帧），正在生成六项诊断并更新最大速度；原始数据已保存。")
            threading.Thread(target=self._analyze_quick_health,
                             args=(rows, path, ident["mount"], ident.get("environment")),
                             daemon=True).start()
        else:
            self._ident_emit(
                f"联合采集完成（{len(rows)}帧），正在后台分别拟合DataModel和V6几何候选；原始数据已保存。")
            threading.Thread(target=self._fit_ident_logs, args=(path,), daemon=True).start()
            threading.Thread(target=self._fit_v6_geometry_ident_rows,
                             args=(rows, path, ident["mount"]), daemon=True).start()

    def _analyze_quick_health(self, rows, source_path, mount, environment=None):
        """Build six health results; update speed limits but never identify a model."""
        item_names = ("environment", "pressure", "muscle", "mapping", "imu", "speed")
        items = {key: {"state": "fail", "detail": "没有有效结果"} for key in item_names}
        items["environment"] = environment or self._python_environment_report()
        report_path = ""
        try:
            import mpc_station as st

            by_channel = {ch: [row for row in rows if int(row["trial"]) == ch]
                          for ch in range(6)}
            pressure_peaks = [max((float(row["P"][ch]) for row in by_channel[ch]),
                                  default=0.0) for ch in range(6)]
            zero_pressure = []
            for ch in range(6):
                tail = [row for row in by_channel[ch]
                        if float(np.max(row["u"])) <= 1.0 and "".join(row["locks"]) == "oooooo"]
                zero_pressure.extend(tail[-5:])
            zero_max = (float(np.max(np.median(
                np.asarray([row["P"] for row in zero_pressure], float), axis=0)))
                if zero_pressure else float("inf"))
            pressure_ok = zero_max <= 10.0 and all(value >= 8.0 for value in pressure_peaks)
            items["pressure"] = {
                "state": "pass" if pressure_ok else "fail",
                "detail": (f"零压最大{zero_max:.1f}kPa；六路峰值 " +
                           "/".join(f"{value:.0f}" for value in pressure_peaks) + "kPa"),
            }

            complete_lens = [row for row in rows if row.get("lens") is not None and
                             all(value is not None for value in row["lens"])]
            length_ranges = []
            for ch in range(6):
                values = [float(row["L"][ch]) for row in by_channel[ch]
                          if row.get("L") is not None and row["stage"].startswith("hold")]
                length_ranges.append(float(np.ptp(values)) if len(values) >= 2 else 0.0)
            muscle_ok = len(complete_lens) >= max(20, int(0.8*len(rows))) and all(
                value >= 0.3 for value in length_ranges)
            items["muscle"] = {
                "state": "pass" if muscle_ok else "fail",
                "detail": (f"20路完整帧{len(complete_lens)}/{len(rows)}；六组长度变化 " +
                           "/".join(f"{value:.1f}" for value in length_ranges) + "mm"),
            }

            lock_rates = []
            antagonist_rates = []
            for ch in range(6):
                active = [row for row in by_channel[ch] if row["stage"].startswith("hold")]
                lock_rates.append(float(np.mean([row["locks"][ch] == "L" for row in active]))
                                  if active else 0.0)
                antagonist = ch + 1 if ch % 2 == 0 else ch - 1
                antagonist_rates.append(float(np.mean([
                    row["locks"][antagonist] == "L" for row in active])) if active else 1.0)
            mapping_ok = all(rate >= 0.8 for rate in lock_rates) and all(
                rate <= 0.2 for rate in antagonist_rates)
            items["mapping"] = {
                "state": "pass" if mapping_ok else "fail",
                "detail": ("本路锁定率 " + "/".join(f"{100*rate:.0f}%" for rate in lock_rates) +
                           "；拮抗误锁最高" + f"{100*max(antagonist_rates):.0f}%"),
            }

            poses = {}
            directional_motion = []
            speed_values = []
            speed_peak_values = []
            for case in SPEED_DIRECTIONS:
                ch = int(case["ch"])
                samples = []
                for row in by_channel[ch]:
                    if not row["stage"].startswith("hold"):
                        continue
                    pose = np.asarray(st.st3_from_q_model(row["q"]), float)
                    samples.append((float(row["t"]), pose))
                poses[ch] = samples
                if samples:
                    axis_values = np.asarray([sample[1][case["axis"]] for sample in samples], float)
                    directional_motion.append(float(np.max(case["sign"]*axis_values)))
                else:
                    directional_motion.append(0.0)
                vv = []
                for stop in range(4, len(samples)+1):
                    value = self._speed_window_velocity(
                        samples[:stop], case["axis"], case["sign"])
                    if value is not None:
                        vv.append(float(value))
                positive = np.asarray([value for value in vv if value >= 0.0], float)
                speed_peak_values.append(float(np.max(positive)) if len(positive) else 0.0)
                speed_values.append(float(np.percentile(positive, 95)) if len(positive) else 0.0)
            imu_ok = all(value >= 0.5 for value in directional_motion)
            items["imu"] = {
                "state": "pass" if imu_ok else "fail",
                "detail": ("六方向有效位移 " +
                           "/".join(f"{value:.1f}" for value in directional_motion) + "°"),
            }
            speed_ok = all(value >= 1.0 for value in speed_values)
            speed_sensor_gate = bool(pressure_ok and mapping_ok and imu_ok)
            speed_blob = {
                "schema": 2, "coordinate_version": 2,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": os.path.abspath(source_path), "mount": mount,
                "voltage_mv": configured_max_output_mv(),
                "trials_per_direction": 1, "velocity_fit_window_s": 0.30,
                "planning_factor": 0.70,
                "measurement_type": "quick_single_scan",
                "application_state": (
                    "active_speed_limits_updated"
                    if speed_ok and speed_sensor_gate else "candidate_not_applied"),
                "directions": {case["name"]: {
                    "channel": int(case["ch"]),
                    "max_filtered_deg_s": round(float(speed_peak_values[index]), 3),
                    "repeatable_p95_deg_s": round(float(speed_values[index]), 3),
                    "planning_deg_s": round(0.70*float(speed_values[index]), 3),
                    "trials": [{
                        "trial": 1, "direction": case["name"],
                        "channel": int(case["ch"]),
                        "peak_deg_s": float(speed_peak_values[index]),
                        "p95_deg_s": float(speed_values[index]),
                        "max_displacement_deg": float(directional_motion[index]),
                        "stop_reason": "快速综合检测五档扫描",
                    }],
                } for index, case in enumerate(SPEED_DIRECTIONS)},
            }
            with open(QUICK_HEALTH_SPEED_CANDIDATE_PATH, "w", encoding="utf-8") as stream:
                json.dump(speed_blob, stream, ensure_ascii=False, indent=2)

            speed_backup = ""
            if speed_ok and speed_sensor_gate:
                if os.path.exists(SPEED_LIMIT_PATH):
                    speed_backup = os.path.join(
                        os.path.dirname(SPEED_LIMIT_PATH),
                        time.strftime("speed_limits_before_quick_%Y%m%d_%H%M%S.json.bak"))
                    shutil.copy2(SPEED_LIMIT_PATH, speed_backup)
                tmp_speed_path = SPEED_LIMIT_PATH + ".tmp"
                with open(tmp_speed_path, "w", encoding="utf-8") as stream:
                    json.dump(speed_blob, stream, ensure_ascii=False, indent=2)
                os.replace(tmp_speed_path, SPEED_LIMIT_PATH)
            items["speed"] = {
                "state": "pass" if speed_ok and speed_sensor_gate else "fail",
                "detail": ("已更新速度上限：" if speed_ok and speed_sensor_gate
                           else "速度未更新：") +
                          "/".join(f"{value:.1f}" for value in speed_values) + "°/s(P95)" +
                          ("" if speed_sensor_gate else "；压力/映射/IMU检查未通过"),
            }

            overall_pass = all(item["state"] == "pass" for item in items.values())
            report = {
                "schema": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": os.path.abspath(source_path), "mount": mount,
                "overall": "pass" if overall_pass else "attention",
                "items": items,
                "immutable_contract": {"six_channel_mapping": "unchanged",
                    "lock_lead_ms": 100, "unlock_pressure_kpa": 30,
                    "unlock_threshold_scope": "unlock judgement only",
                    "antagonist_pairs": [[0, 1], [2, 3], [4, 5]]},
                "application_state": "health_report_and_speed_update_only",
                "model_identification": "disabled_use_default_model",
                "speed_update": {
                    "applied": bool(speed_ok and speed_sensor_gate),
                    "active_path": os.path.abspath(SPEED_LIMIT_PATH),
                    "backup_path": os.path.abspath(speed_backup) if speed_backup else "",
                },
                "speed_candidate": os.path.abspath(QUICK_HEALTH_SPEED_CANDIDATE_PATH),
            }
            report_path = os.path.join(os.path.dirname(source_path),
                                       time.strftime("quick_health_report_%Y%m%d_%H%M%S.json"))
            with open(report_path, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            message = ("六项全部通过，最大速度已更新" if overall_pass
                       else "检测完成，有项目需要复测或检查")
            self._quick_health_status("complete", items, message, report_path)
            self._quick_health_emit(message + "；报告 " + report_path)
        except Exception as exc:
            self._quick_health_status("failed", items, "分析失败：" + str(exc), report_path)
            self._quick_health_emit("分析失败：" + str(exc))

    def _fit_v3_repeat_rows(self, rows, source_path, mount):
        """Report V3 repeatability only; never save a calibration/model candidate."""
        try:
            from v6_geometry_calibration import (
                fit_geometry_samples, predicted_channel_lengths_mm)

            complete = [row for row in rows if row.get("L") is not None]
            if len(complete) < 120:
                raise RuntimeError(f"完整编码器样本只有{len(complete)}帧")
            q_deg = np.degrees(np.asarray([row["q"] for row in complete], dtype=float))
            lengths = np.asarray([row["L"] for row in complete], dtype=float)
            muscle_lengths = np.asarray([
                (row.get("M") if row.get("M") is not None else [np.nan] * 10)
                for row in complete], dtype=float)
            repeats = np.asarray([row.get("repeat", 0) for row in complete], dtype=int)
            stages = np.asarray([row.get("stage", "") for row in complete], dtype=object)
            levels = np.asarray([row.get("level", 0) for row in complete], dtype=int)

            def v3_fit(mask):
                channel_masks = np.zeros((len(complete), 6), dtype=bool)
                channel_masks[:, 3] = np.asarray(mask, dtype=bool)
                fitted = fit_geometry_samples(
                    q_deg, lengths, session_ids=repeats, source=source_path,
                    channel_sample_masks=channel_masks)
                return fitted["channels"][3]["fixed_mapping_fit"]

            hold_up = stages == "hold_up"
            overall = v3_fit(hold_up)
            per_repeat = []
            for repeat in sorted(set(repeats.tolist())):
                repeat_mask = repeats == repeat
                selected = np.flatnonzero(hold_up & repeat_mask)
                base = np.flatnonzero(repeat_mask)[0]
                metric = v3_fit(hold_up & repeat_mask)
                diagnostics = {}
                if selected.size and np.all(np.isfinite(muscle_lengths[selected][:, [5, 6]])):
                    diagnostics = {
                        "muscle6_measured_range_mm": float(np.ptp(muscle_lengths[selected, 5])),
                        "muscle7_measured_range_mm": float(np.ptp(muscle_lengths[selected, 6])),
                        "muscle6_endpoint_delta_mm": float(
                            muscle_lengths[selected[-1], 5] - muscle_lengths[base, 5]),
                        "muscle7_endpoint_delta_mm": float(
                            muscle_lengths[selected[-1], 6] - muscle_lengths[base, 6]),
                    }
                per_repeat.append({
                    "repeat": int(repeat + 1), **metric,
                    "start_q_deg": [float(v) for v in q_deg[base]],
                    "end_hold_up_q_deg": [float(v) for v in q_deg[selected[-1]]],
                    **diagnostics,
                })
            per_level = []
            for level in (600, 1200, 1800):
                metric = v3_fit(hold_up & (levels == level))
                per_level.append({"level_mv": level, **metric})

            report = {
                "schema": 1, "model_version": "V6", "channel": "V3",
                "physical_muscles": [6, 7],
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mount": mount, "repeat_count": len(per_repeat),
                "levels_mv": [600, 1200, 1800, 1200, 600], "hold_s": 5.0,
                "metric_definition": "direct zero-aligned delta-L comparison; no fitted scale",
                "overall_hold_up": overall, "per_repeat_hold_up": per_repeat,
                "per_level_hold_up": per_level,
                "immutable_contract": {
                    "valve_mapping": "V3=physical muscles 6+7",
                    "lock_lead_ms": 100, "unlock_pressure_kpa": 30,
                    "unlock_threshold_scope": "unlock judgement only",
                },
                "application_state": "not_applied",
                "updates_data_model": False, "updates_v6_model": False,
                "updates_six_channel_candidate": False,
                "raw_log_path": os.path.abspath(source_path),
            }
            stamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                os.path.dirname(source_path), f"v3_repeat_fit_{stamp}.json")
            with open(report_path, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)

            predicted = predicted_channel_lengths_mm(q_deg)[:, 3]
            measured = lengths[:, 3]
            plot_path = str(DATA_ROOT / "output" / "figures" /
                            "TRO_V3_repeat_validation.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            self._save_v3_repeat_plot(
                complete, repeats, stages, predicted, measured, muscle_lengths,
                per_repeat, plot_path)
            self.analysis_image_ready.emit(plot_path)
            detail = "；".join(
                f"第{item['repeat']}次 τb={item['kendall_tau_b']:.4f}, "
                f"RMSE={item['rmse_mm']:.2f}mm"
                for item in per_repeat)
            self._v3_repeat_emit(
                f"分析完成：总体τb={overall['kendall_tau_b']:.4f}、"
                f"RMSE={overall['rmse_mm']:.2f}mm；{detail}。"
                f"图 {plot_path}；报告 {report_path}；未更新模型或六路候选。")
        except Exception as exc:
            self._v3_repeat_emit("分析失败：" + str(exc))

    @staticmethod
    def _save_v3_repeat_plot(rows, repeats, stages, predicted, measured,
                             muscle_lengths, per_repeat, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [2.1, 1.0]})
        for repeat in sorted(set(repeats.tolist())):
            indices = np.flatnonzero(repeats == repeat)
            base = indices[0]
            mask = (repeats == repeat) & (stages == "hold_up")
            selected = np.flatnonzero(mask)
            sample = np.arange(selected.size)
            axes[0].plot(sample, predicted[selected] - predicted[base], linewidth=1.8,
                         label=f"repeat {repeat + 1} theory")
            axes[0].plot(sample, measured[selected] - measured[base], "--", linewidth=1.6,
                         label=f"repeat {repeat + 1} measured")
            if np.all(np.isfinite(muscle_lengths[selected][:, [5, 6]])):
                axes[0].plot(sample, muscle_lengths[selected, 5] - muscle_lengths[base, 5],
                             ":", linewidth=1.0, alpha=0.75,
                             label=f"repeat {repeat + 1} muscle 6")
                axes[0].plot(sample, muscle_lengths[selected, 6] - muscle_lengths[base, 6],
                             ":", linewidth=1.0, alpha=0.75,
                             label=f"repeat {repeat + 1} muscle 7")
        axes[0].set_ylabel("Delta muscle length (mm)")
        axes[0].set_xlabel("20 Hz hold-up sample")
        axes[0].set_title("V3 (muscles 6+7): V6 geometry vs measured contraction")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(ncol=2, fontsize=8)
        labels = [str(item["repeat"]) for item in per_repeat]
        rmse = [item["rmse_mm"] for item in per_repeat]
        bars = axes[1].bar(labels, rmse, color="tab:orange", alpha=0.8)
        axes[1].bar_label(bars, fmt="%.2f mm")
        for index, item in enumerate(per_repeat):
            axes[1].text(index, 0.08 * max(rmse),
                         f"tau-b={item['kendall_tau_b']:.3f}",
                         ha="center", va="bottom", fontsize=9)
        axes[1].set_xlabel("Repeat")
        axes[1].set_ylabel("Direct RMSE (mm)")
        axes[1].set_title("Repeatability summary (no fitted scale)")
        axes[1].grid(True, axis="y", alpha=0.25)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _fit_v6_geometry_ident_rows(self, rows, source_path, mount):
        """Fit the immutable V6 observation map from the same DataModel scan."""
        try:
            complete = [r for r in rows if r.get("L") is not None]
            if len(complete) < 120:
                raise RuntimeError(f"完整编码器样本只有{len(complete)}帧")
            from v6_geometry_calibration import (
                concise_summary, fit_geometry_samples, save_candidate)
            q_deg = np.degrees(np.asarray([r["q"] for r in complete], dtype=float))
            lengths = np.asarray([r["L"] for r in complete], dtype=float)
            sessions = np.asarray([r["trial"] for r in complete], dtype=int)
            stages = np.asarray([r.get("stage", "") for r in complete], dtype=object)
            channel_masks = np.zeros((len(complete), 6), dtype=bool)
            for channel in range(6):
                channel_masks[:, channel] = ((sessions == channel) & (stages == "hold_up"))
            # V1 reaches the known positive q1 mechanical stop near 38 degrees.
            channel_masks[:, 1] &= q_deg[:, 0] < 38.0
            result = fit_geometry_samples(
                q_deg, lengths, session_ids=sessions, source=source_path,
                channel_sample_masks=channel_masks)
            result["mount"] = mount
            result["raw_log_path"] = os.path.abspath(source_path)
            result["acquisition"] = "combined_data_model_and_v6_geometry_scan"
            result["sample_selection"] = {
                "per_channel": "own trial hold_up rows only",
                "v1_mechanical_limit_filter": "q1 < 38 deg",
            }
            candidate = save_candidate(result, V6_GEOMETRY_CANDIDATE_PATH)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            report = save_candidate(
                result, os.path.join(os.path.dirname(source_path),
                                     f"v6_geometry_dm_ident_fit_{stamp}.json"))
            readiness = ("六路均通过，可进入人工复核" if result["ready_to_apply"]
                         else "有通道未通过，候选保持未应用")
            self._ident_emit(
                f"V6几何拟合完成：{concise_summary(result)}；{readiness}。"
                f"候选 {candidate}；报告 {report}。")
        except Exception as exc:
            self._ident_emit("V6几何拟合失败：" + str(exc))

    def _fit_ident_logs(self, newest_path):
        """合并最近三轮专用辨识日志；先用下降段交叉验证，再用全部数据拟合并保存。"""
        try:
            import mpc_station as st
            if not os.path.exists(st.PARAMS_JSON):
                raise RuntimeError("缺少 control/params.json")
            plant = st.Plant(json.load(open(st.PARAMS_JSON, encoding="utf-8")))
            log_dir = os.path.dirname(newest_path)
            prefix = "dm_ident_raw_"
            files = sorted(os.path.join(log_dir, f) for f in os.listdir(log_dir)
                           if f.startswith(prefix) and f.endswith(".csv"))[-3:]
            groups = []
            for fn in files:
                by_trial = {}
                for r in csv.DictReader(open(fn, encoding="utf-8")):
                    try:
                        item = {
                            "t": float(r["t"]), "stage": r["stage"],
                            "q": np.radians([float(r[f"q{i}"]) for i in (1, 2, 3)]),
                            "u": np.array([float(r[f"uact{i}"]) for i in range(6)]),
                            "P": np.array([float(r[f"P{i}"]) for i in range(6)]),
                        }
                        by_trial.setdefault(int(r["trial"]), []).append(item)
                    except (KeyError, TypeError, ValueError):
                        continue
                groups.extend(v for v in by_trial.values() if len(v) >= 30)

            def samples(allowed=None):
                X, Y = [], []
                for g in groups:
                    for k in range(1, len(g)-1):
                        if not (0.05 <= g[k+1]["t"]-g[k]["t"] <= 0.18
                                and 0.05 <= g[k]["t"]-g[k-1]["t"] <= 0.18):
                            continue
                        if allowed is not None and not allowed(g[k]["stage"]):
                            continue
                        X.append(st.DataModel.feats(g[k]["q"], g[k]["q"]-g[k-1]["q"], g[k]["P"]))
                        Y.append(g[k+1]["q"]-g[k]["q"])
                return np.asarray(X), np.asarray(Y)

            Xtr, Ytr = samples(lambda s: s != "hold_down")
            Xva, Yva = samples(lambda s: s == "hold_down")
            Xall, Yall = samples()
            if len(Xall) < 200:
                raise RuntimeError(f"有效连续样本只有 {len(Xall)} 个")
            check = st.DataModel(plant)
            check.fit(Xtr, Ytr)
            val1 = float(np.degrees(np.sqrt(np.mean(np.sum((Yva-Xva@check.theta().T)**2, axis=1))))) \
                if len(Xva) else float("nan")

            dm = st.DataModel(plant)
            fit1 = dm.fit(Xall, Yall)
            blocks = [(np.array([r["u"] for r in g]), np.array([r["P"] for r in g])) for g in groups]
            dm.fit_pneu(blocks, verbose=True)
            dm.src = f"dm-ident-{len(files)}runs"
            if os.path.exists(st.DM_PATH):
                backup = os.path.join(os.path.dirname(st.DM_PATH),
                                      time.strftime("data_model_before_ident_%Y%m%d_%H%M%S.json.bak"))
                shutil.copy2(st.DM_PATH, backup)
            dm.save()
            try:
                gains = dm.gain_table_deg()
                print("# DataModel 辨识稳态增益（每路1000mV→q°）：")
                for c in range(6):
                    print(f"#   V{c}: ({gains[0,c]:+.2f}, {gains[1,c]:+.2f}, {gains[2,c]:+.2f})")
            except np.linalg.LinAlgError:
                print("# DataModel 辨识：K 矩阵接近奇异，已保存模型但无法打印稳态增益表")
            self._ident_emit(f"辨识完成：合并最近 {len(files)} 轮、{len(Xall)} 样本；"
                             f"拟合RMSE {fit1:.2f}°/拍，下降段验证 {val1:.2f}°/拍。"
                             "新 data_model.json 已保存，重新播放即可使用。")
        except Exception as e:
            self._ident_emit("辨识失败：" + str(e))

    # ───────────── 六路手动目标压力测试（kPa 闭环，直到手动停止） ─────────────
    def _manual_pressure_emit(self, msg):
        print("# 六路手动压力：" + msg)
        self.rx.emit("[manp] " + msg + "\n")

    @staticmethod
    def _manual_pressure_gains():
        defaults = {"kp": 10.0, "ki": 18.0, "deadband": 1.5}
        try:
            with open(GAINS_PATH, "r", encoding="utf-8") as stream:
                blob = json.load(stream)
            defaults.update({
                "kp": max(0.0, float(blob.get("pressure_hold_kp", defaults["kp"]))),
                "ki": max(0.0, float(blob.get("pressure_hold_ki", defaults["ki"]))),
                "deadband": max(0.0, float(blob.get(
                    "pressure_hold_deadband_kpa", defaults["deadband"]))),
            })
        except Exception:
            pass
        return defaults

    @staticmethod
    def _validate_manual_pressure_targets(payload, allow_antagonists=False):
        try:
            raw = json.loads(payload)
            targets = np.asarray(raw, dtype=float)
        except Exception as exc:
            raise ValueError("需要六个有效的 kPa 数值") from exc
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            raise ValueError("需要六个有效的 kPa 数值")
        if np.any(targets < 0.0):
            raise ValueError("目标压力不能为负数")
        if not allow_antagonists:
            for left, right in ((0, 1), (2, 3), (4, 5)):
                if targets[left] > 0.0 and targets[right] > 0.0:
                    raise ValueError(
                        f"拮抗通道 V{left}/V{right} 不能同时设为非零压力")
        return targets

    @QtCore.pyqtSlot(str, result=str)
    def setManualPressureTargets(self, payload):
        """启动或在线更新六路目标压力；输入不设预定义档位和持续时间。"""
        if not self.ser:
            return "先连接 IMU 串口。"
        try:
            # Debug manual pressure is the deliberate co-contraction test:
            # V0/V1, V2/V3 and V4/V5 may be pressurised simultaneously here.
            # Reachability/MPC callers keep the default antagonist exclusion.
            targets = self._validate_manual_pressure_targets(
                payload, allow_antagonists=True)
        except ValueError as exc:
            return "六路压力输入无效：" + str(exc)
        if all(float(v) == 0.0 for v in targets):
            if self._manual_pressure_test is not None:
                return self._manual_pressure_begin_gradual_stop("六路目标均设为0")
            try:
                self.ser.write(b"s\n")
            except Exception as exc:
                return "全机放气失败：" + str(exc)
            return "六路目标均为0，已执行全机放气。"
        if self._manual_pressure_test is None:
            if any(x is not None for x in (
                    self._dm_ident, self._adc_cal, self._speed_test,
                    self._x_comp_test, self._mpc_data_collect,
                    self._phi_gradient_test,
                    self._reachability_test)):
                return "已有标定或实机测试正在运行，请先停止。"
            frame = self._latest_frame
            if frame is None or time.time() - frame["wall"] > 1.0:
                return "还没有收到新的 IMU 数据帧。"
            pressure = frame.get("kpa")
            if pressure is None or len(frame.get("locks", "")) != 6:
                return "当前固件帧没有六路压力/锁状态，不能闭环控制压力。"
            # 只在已满足解锁安全判据时开始；30 kPa 不截断目标或实测压力。
            if (frame["locks"] != "oooooo" or
                    float(np.max(np.asarray(pressure, dtype=float))) > 30.0):
                return "请先全机放气，等六路锁=o且实测压力均≤30 kPa 后再开始。"
            now = time.time()
            self._manual_pressure_test = {
                "target": targets.copy(), "integral": np.zeros(6),
                "last_t": now, "last_send": 0.0, "last_ui": 0.0,
                "started": now, "rows": [], "last_u": np.zeros(6),
                "gains": self._manual_pressure_gains(),
            }
            self._manual_pressure_emit(
                "测试已启动：目标=" + "/".join(f"{v:g}" for v in targets) +
                " kPa；无预设时长，按停止/急停结束。")
            return ("六路手动压力测试已启动；目标=" +
                    "/".join(f"{v:g}" for v in targets) + " kPa。")
        test = self._manual_pressure_test
        if test.get("gradual_stop"):
            return "正在缓慢放气，请等待六路压力降至解锁范围。"
        test["target"] = targets.copy()
        self._manual_pressure_emit(
            "目标已更新为 " + "/".join(f"{v:g}" for v in targets) + " kPa。")
        return "六路目标压力已更新。"

    @QtCore.pyqtSlot(result=str)
    def stopManualPressureTest(self):
        if self._manual_pressure_test is None:
            try:
                if self.ser:
                    self.ser.write(b"s\n")
            except Exception:
                pass
            return "手动压力测试未运行；已补发全机放气。"
        return self._manual_pressure_begin_gradual_stop("用户停止")

    def _manual_pressure_begin_gradual_stop(self, why):
        """Ramp the six regulator commands down before opening every lock valve."""
        test = self._manual_pressure_test
        if test is None:
            return "六路手动压力测试没有运行。"
        if test.get("gradual_stop"):
            return "正在缓慢放气，请等待六路压力降至解锁范围。"
        last_u = np.maximum(
            np.asarray(test.get("last_u", np.zeros(6)), dtype=float), 0.0)
        # If Stop is pressed before the first 20 Hz control frame, initialise
        # the ramp from the target feed-forward estimate.  Once a command has
        # actually been sent, always start from that command so Stop can never
        # raise a channel before lowering it.
        if float(test.get("last_send", 0.0) or 0.0) <= 0.0:
            last_u = np.clip(
                np.asarray(test.get("target", np.zeros(6)), dtype=float) / 0.09,
                0.0, float(OPEN_LOOP_MAX_OUTPUT_MV))
        test["last_u"] = last_u
        test["target"] = np.zeros(6, dtype=float)
        test["integral"] = np.zeros(6, dtype=float)
        test["gradual_stop"] = True
        test["gradual_stop_reason"] = str(why)
        test["last_t"] = time.time()
        test["last_send"] = 0.0
        test["last_ui"] = 0.0
        msg = ("开始缓慢放气：六路输出按不超过 "
               f"{MANUAL_PRESSURE_DEFLATE_SLEW_MV_PER_S:.0f} mV/s 降至0；"
               f"实测压力均≤{MANUAL_PRESSURE_UNLOCK_KPA:.0f} kPa且锁阀已逐路打开后，"
               "再结束测试。")
        self._manual_pressure_emit(msg)
        return msg

    def _manual_pressure_stop(self, why):
        test = self._manual_pressure_test
        if test is None:
            return "六路手动压力测试没有运行。"
        self._manual_pressure_test = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        saved = ""
        if test["rows"]:
            try:
                log_dir = str(CONTROL_LOG_DIR)
                os.makedirs(log_dir, exist_ok=True)
                path = os.path.abspath(os.path.join(
                    log_dir, "manual_pressure_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"))
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    names = (["t"] + [f"target{i}_kpa" for i in range(6)] +
                             [f"P{i}_kpa" for i in range(6)] +
                             [f"u{i}_mv" for i in range(6)] + ["locks"] +
                             [f"L{i}_mm" for i in range(6)])
                    writer = csv.DictWriter(stream, fieldnames=names)
                    writer.writeheader()
                    for row in test["rows"]:
                        item = {"t": round(row["t"], 5), "locks": row["locks"]}
                        item.update({f"target{i}_kpa": round(float(row["target"][i]), 5)
                                     for i in range(6)})
                        item.update({f"P{i}_kpa": round(float(row["P"][i]), 5)
                                     for i in range(6)})
                        item.update({f"u{i}_mv": int(round(float(row["u"][i])))
                                     for i in range(6)})
                        item.update({f"L{i}_mm": ("" if row["L"][i] is None else
                                                   round(float(row["L"][i]), 7))
                                     for i in range(6)})
                        writer.writerow(item)
                saved = "；日志 " + path
            except Exception as exc:
                saved = "；日志保存失败：" + str(exc)
        msg = "测试已停止并全机放气：" + why + saved
        self._manual_pressure_emit(msg)
        return msg

    def _manual_pressure_on_frame(self, frame):
        test = self._manual_pressure_test
        if test is None:
            return
        pressure = frame.get("kpa")
        if pressure is None or len(frame.get("locks", "")) != 6:
            self._manual_pressure_stop("压力或锁状态遥测中断")
            return
        now = frame["wall"]
        if now - test["last_send"] < 0.045:
            return
        measured = np.asarray(pressure, dtype=float)
        if not np.all(np.isfinite(measured)):
            self._manual_pressure_stop("压力遥测包含非有限值")
            return
        dt = float(np.clip(now - test["last_t"], 0.01, 0.2))
        test["last_t"] = now
        if test.get("gradual_stop"):
            step = MANUAL_PRESSURE_DEFLATE_SLEW_MV_PER_S * dt
            output = np.maximum(
                np.asarray(test.get("last_u", np.zeros(6)), dtype=float) - step,
                0.0)
            target = np.zeros(6, dtype=float)
            try:
                cmd = "C" + ",".join(
                    str(int(round(v))) for v in output) + "\n"
                self.ser.write(cmd.encode("ascii"))
            except Exception as exc:
                self._manual_pressure_stop("缓慢放气时串口写入失败：" + str(exc))
                return
            test["last_send"] = now
            test["last_u"] = output.copy()
            lengths = frame.get("channel_length_mm")
            if lengths is None or len(lengths) != 6:
                lengths = [None] * 6
            test["rows"].append({
                "t": now - test["started"], "target": target,
                "P": measured.copy(), "u": output.copy(),
                "locks": frame["locks"], "L": list(lengths),
            })
            voltage_zero = bool(np.all(output <= 0.5))
            pressure_safe = bool(np.all(measured <= MANUAL_PRESSURE_UNLOCK_KPA))
            locks_open = frame["locks"] == "oooooo"
            if voltage_zero and pressure_safe and locks_open:
                self._manual_pressure_stop(
                    "缓慢放气完成（" + test.get("gradual_stop_reason", "用户停止") + "）")
                return
            if now - test["last_ui"] >= 0.5:
                test["last_ui"] = now
                waiting = []
                if not voltage_zero:
                    waiting.append("输出降至0")
                if not pressure_safe:
                    waiting.append(f"压力≤{MANUAL_PRESSURE_UNLOCK_KPA:.0f} kPa")
                if not locks_open:
                    waiting.append("锁阀逐路打开")
                self._manual_pressure_emit(
                    "缓慢放气中：输出 " + "/".join(f"{v:.0f}" for v in output) +
                    " mV；压力 " + "/".join(f"{v:.0f}" for v in measured) +
                    " kPa；等待" + "、".join(waiting) + "。")
            return
        target = test["target"].copy()
        gains = test["gains"]
        error = target - measured
        effective = np.where(np.abs(error) <= gains["deadband"], 0.0, error)
        integral = test["integral"].copy()
        integral += gains["ki"] * effective * dt
        integral = np.clip(integral, -1000.0, 1000.0)
        integral[target <= 0.0] = 0.0
        # 固件仍以 mV 接收：0.09 kPa/mV 只作前馈，最终由 ADC 实测压力 PI 修正。
        feedforward = target / 0.09
        raw_u = feedforward + gains["kp"] * effective + integral
        output = np.clip(raw_u, 0.0, float(OPEN_LOOP_MAX_OUTPUT_MV))
        output[target <= 0.0] = 0.0
        # 饱和且误差仍朝饱和方向时冻结积分，避免目标超过供气能力时持续卷积。
        driving_high = (raw_u > OPEN_LOOP_MAX_OUTPUT_MV) & (effective > 0.0)
        driving_low = (raw_u < 0.0) & (effective < 0.0)
        keep = driving_high | driving_low
        integral[keep] = test["integral"][keep]
        test["integral"] = integral
        try:
            # C is the firmware's manual co-contraction stream.  It preserves
            # the same 100 ms per-channel prelock and output ceiling as O, but
            # intentionally does not project antagonist pairs to one side.
            cmd = "C" + ",".join(str(int(round(v))) for v in output) + "\n"
            self.ser.write(cmd.encode("ascii"))
        except Exception as exc:
            self._manual_pressure_stop("串口写入失败：" + str(exc))
            return
        test["last_send"] = now
        test["last_u"] = output.copy()
        lengths = frame.get("channel_length_mm")
        if lengths is None or len(lengths) != 6:
            lengths = [None] * 6
        test["rows"].append({
            "t": now - test["started"], "target": target, "P": measured.copy(),
            "u": output.copy(), "locks": frame["locks"], "L": list(lengths),
        })
        if now - test["last_ui"] >= 0.5:
            test["last_ui"] = now
            saturated = [f"V{i}" for i, v in enumerate(output)
                         if v >= OPEN_LOOP_MAX_OUTPUT_MV - 0.5]
            suffix = ("；输出已饱和 " + ",".join(saturated)) if saturated else ""
            self._manual_pressure_emit(
                "目标 " + "/".join(f"{v:.0f}" for v in target) +
                "；实测 " + "/".join(f"{v:.0f}" for v in measured) +
                " kPa；输出 " + "/".join(f"{v:.0f}" for v in output) + " mV" + suffix)

    # ───────────── V6 理论 Reachability（离线逆静力学，不操作真机） ─────────────
    def _theoretical_reachability_emit(self, msg):
        print("# V6 理论 Reachability：" + msg)
        self.rx.emit("[theoryreach] " + msg + "\n")

    @QtCore.pyqtSlot(int, result=str)
    def calculateTheoreticalReachability(self, sample_count):
        """Start an offline V6 inverse-static reachability calculation."""
        if self._theoretical_reachability_running:
            return "V6 理论可达图正在计算。"
        count = int(np.clip(int(sample_count), 50, 3000))
        seed = int(time.time()) & 0x7fffffff
        self._theoretical_reachability_running = True

        def worker():
            try:
                from v6_reachability import calculate_theoretical_map, save_theoretical_plot

                last_percent = [-1]

                def progress(done, total):
                    percent = int(100 * done / max(1, total))
                    bucket = percent // 10
                    if bucket != last_percent[0]:
                        last_percent[0] = bucket
                        self._theoretical_reachability_emit(
                            f"计算进度 {percent}%（{done}/{total} 个候选姿态）")

                blob = calculate_theoretical_map(
                    count, seed, pressure_limit_kpa=REACHABILITY_MAX_KPA,
                    swing_x_min_deg=float(THEORETICAL_REACHABILITY_POSE_MIN_DEG[0]),
                    swing_x_max_deg=float(THEORETICAL_REACHABILITY_POSE_MAX_DEG[0]),
                    swing_y_min_deg=float(THEORETICAL_REACHABILITY_POSE_MIN_DEG[1]),
                    swing_y_max_deg=float(THEORETICAL_REACHABILITY_POSE_MAX_DEG[1]),
                    twist_min_deg=float(THEORETICAL_REACHABILITY_POSE_MIN_DEG[2]),
                    twist_max_deg=float(THEORETICAL_REACHABILITY_POSE_MAX_DEG[2]),
                    progress=progress)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                log_dir = str(CONTROL_LOG_DIR)
                figure_dir = str(DATA_ROOT / "output" / "figures")
                os.makedirs(log_dir, exist_ok=True)
                os.makedirs(figure_dir, exist_ok=True)
                data_path = os.path.abspath(os.path.join(
                    log_dir, f"v6_theoretical_reachability_{stamp}.json"))
                plot_path = os.path.join(
                    figure_dir, "TRO_V6_theoretical_reachability.png")
                with open(data_path, "w", encoding="utf-8") as stream:
                    json.dump(blob, stream, ensure_ascii=False, indent=2, allow_nan=False)
                multiview_path = save_theoretical_plot(blob, plot_path)
                plot_path = multiview_path or plot_path
                self.theoretical_reachability_ready.emit(plot_path)
                rows = blob["results"]
                pose = np.asarray([row["pose_deg"] for row in rows], dtype=float)
                ranges = ", ".join(
                    f"{name}[{np.min(pose[:, axis]):.1f},{np.max(pose[:, axis]):.1f}]°"
                    for axis, name in enumerate(("X", "Y", "φ")))
                self._theoretical_reachability_emit(
                    f"计算完成：{len(rows)}/{count + 1} 个理论静力可达点；{ranges}；"
                    f"图 {plot_path}；数据 {data_path}。全程未连接或控制真机。")
            except Exception as exc:
                self._theoretical_reachability_emit("计算失败：" + str(exc))
            finally:
                self._theoretical_reachability_running = False

        threading.Thread(target=worker, daemon=True).start()
        return (f"V6 理论可达图已开始计算：{count} 个随机候选姿态、压力 0–"
                f"{REACHABILITY_MAX_KPA:.0f} kPa；不连接、不控制真机。")

    # ───────────── 实机 Reachability map 随机压力采样（≤200 kPa） ─────────────
    def _reachability_emit(self, msg):
        print("# Reachability map：" + msg)
        self.rx.emit("[reach] " + msg + "\n")

    @staticmethod
    def _reachability_targets(sample_count, seed, max_kpa=REACHABILITY_MAX_KPA):
        """Create reproducible pressure targets with antagonist-pair exclusion."""
        count = int(np.clip(sample_count, REACHABILITY_MIN_SAMPLES,
                            REACHABILITY_MAX_SAMPLES))
        ceiling = float(np.clip(max_kpa, 25.0, REACHABILITY_MAX_KPA))
        rng = np.random.default_rng(int(seed))
        targets = []
        random_count = max(0, count - 6)
        for _ in range(random_count):
            target = np.zeros(6, dtype=float)
            while not np.any(target > 0.0):
                target[:] = 0.0
                for left, right in ((0, 1), (2, 3), (4, 5)):
                    choice = int(rng.choice(3, p=(0.45, 0.45, 0.10)))
                    if choice < 2:
                        channel = left if choice == 0 else right
                        target[channel] = float(rng.uniform(25.0, ceiling))
            targets.append(target)
        # Six single-channel boundary points make the random cloud interpretable.
        targets.extend(np.eye(6, dtype=float) * ceiling)
        rng.shuffle(targets)
        return np.asarray(targets[:count], dtype=float)

    @staticmethod
    def _reachability_extrema_targets(max_kpa=REACHABILITY_MAX_KPA):
        """Six single-channel and twelve legal two-channel maximum-pressure cases."""
        ceiling = float(np.clip(max_kpa, 25.0, REACHABILITY_MAX_KPA))
        targets = [np.eye(6, dtype=float)[channel] * ceiling
                   for channel in range(6)]
        antagonists = {(0, 1), (2, 3), (4, 5)}
        for left in range(6):
            for right in range(left + 1, 6):
                if (left, right) in antagonists:
                    continue
                target = np.zeros(6, dtype=float)
                target[[left, right]] = ceiling
                targets.append(target)
        return np.asarray(targets, dtype=float)

    @staticmethod
    def _slow_rom_targets(max_kpa=REACHABILITY_MAX_KPA):
        """Six singles, twelve legal pairs and eight legal three-axis corners."""
        ceiling = float(np.clip(max_kpa, 25.0, REACHABILITY_MAX_KPA))
        targets = list(SerialBridge._reachability_extrema_targets(ceiling))
        for y_channel in (0, 1):
            for phi_channel in (2, 3):
                for x_channel in (4, 5):
                    target = np.zeros(6, dtype=float)
                    target[[y_channel, phi_channel, x_channel]] = ceiling
                    targets.append(target)
        return np.asarray(targets, dtype=float)

    @staticmethod
    def _slow_rom_output_step(previous, target, pose_speed, dt, output_limit):
        """One bounded slow-ROM voltage step with pose-speed hold/backoff."""
        previous = np.asarray(previous, dtype=float)
        target = np.asarray(target, dtype=float)
        pose_speed = np.asarray(pose_speed, dtype=float)
        safe_dt = float(np.clip(dt, 0.0, 0.2))
        maximum_step = SLOW_ROM_SLEW_MV_PER_S * safe_dt
        desired = np.where(target > 0.0, float(output_limit), 0.0)
        peak_speed = (float(np.max(np.abs(pose_speed)))
                      if pose_speed.size and np.all(np.isfinite(pose_speed)) else 0.0)
        if peak_speed >= SLOW_ROM_SPEED_BACKOFF_DEG_S:
            backoff_step = SLOW_ROM_BACKOFF_MV_PER_S * safe_dt
            desired = np.minimum(desired, np.maximum(previous-backoff_step, 0.0))
            # Safety backoff is intentionally faster than the ordinary slow
            # upward sweep; do not clamp it back to the 100 mV/s rise rate.
            return previous + np.clip(
                desired-previous, -backoff_step, maximum_step)
        elif peak_speed >= SLOW_ROM_SPEED_HOLD_DEG_S:
            desired = np.minimum(desired, previous)
        return previous + np.clip(desired-previous, -maximum_step, maximum_step)

    @staticmethod
    def _slow_rom_release_step(previous, pose_speed, dt):
        """Lower voltage slowly; pause the release while the arm is moving fast."""
        previous = np.maximum(np.asarray(previous, dtype=float), 0.0)
        pose_speed = np.asarray(pose_speed, dtype=float)
        peak_speed = (float(np.max(np.abs(pose_speed)))
                      if pose_speed.size and np.all(np.isfinite(pose_speed)) else 0.0)
        if peak_speed >= SLOW_ROM_SPEED_HOLD_DEG_S:
            return previous.copy()
        step = SLOW_ROM_DEFLATE_SLEW_MV_PER_S * float(np.clip(dt, 0.0, 0.2))
        return np.maximum(previous-step, 0.0)

    @staticmethod
    def _slow_rom_recenter_command(pose):
        """Return one antagonist-safe pulse command that nudges X/Y/phi to zero."""
        pose = np.asarray(pose, dtype=float)
        locks = np.zeros(6, dtype=bool)
        output = np.zeros(6, dtype=float)
        # For a positive residual use the negative-direction channel, and vice
        # versa. Axis order is GUI [X,Y,phi].
        opposite_channels = ((4, 5), (1, 0), (3, 2))
        for axis, angle in enumerate(pose[:3]):
            if not np.isfinite(angle) or abs(float(angle)) <= SLOW_ROM_ZERO_TOL_DEG:
                continue
            channel = opposite_channels[axis][0 if angle > 0.0 else 1]
            pulse = np.clip(
                SLOW_ROM_RECENTER_MIN_MV + 5.0*abs(float(angle)),
                SLOW_ROM_RECENTER_MIN_MV, SLOW_ROM_RECENTER_MAX_MV)
            locks[channel] = True
            output[channel] = float(pulse)
        return locks, output

    @staticmethod
    def _reachability_pose_from_frame(test, frame):
        import mpc_station as st

        st.SRV["mount"] = test["mount"]
        q = st.q_model_from_quat(frame["quat"], test["qref"], test["mount"])
        return q, np.asarray(st.st3_from_q_model(q), dtype=float)

    def _slow_rom_append_response(self, test, frame, pose, now):
        """Cache one slow-ROM row in the same schema as operation response.xlsx."""
        if test.get("sampling_mode") not in SLOW_ROM_MODES:
            return
        last_wall = float(test.get("response_last_wall", 0.0) or 0.0)
        if last_wall and now-last_wall < 0.045:
            return
        if test.get("response_t0") is None:
            test["response_t0"] = float(now)
        test["response_last_wall"] = float(now)
        response_time_s = float(now)-float(test["response_t0"])
        pose_values = self._record_vector(pose, 3)
        pose_speed, previous_pose, previous_time = self._pose_speed_deg_s(
            pose_values, response_time_s, test.get("response_previous_pose_deg"),
            test.get("response_previous_time_s"))
        test["response_previous_pose_deg"] = previous_pose
        test["response_previous_time_s"] = previous_time

        pressure_raw = self._record_vector(frame.get("kpa"), 6)
        length_raw = self._record_vector(frame.get("channel_length_mm"), 6)
        pressure = [pressure_raw[i] for i in self.DATA_RECORD_CHANNEL_ORDER]
        channel_length = [length_raw[i] for i in self.DATA_RECORD_CHANNEL_ORDER]
        group_raw = self._group_signed_displacements(frame.get("lens"))
        group_displacement, group_zero = self._zero_group_displacements(
            group_raw, test.get("response_group_zero_mm"))
        test["response_group_zero_mm"] = group_zero
        ordered_displacement = [
            group_displacement[index] for index in self.DATA_RECORD_CHANNEL_ORDER
        ]
        length_columns = [
            value
            for length, pair in zip(channel_length, ordered_displacement)
            for value in (length, pair[0], pair[1])
        ]

        with self._nano25_lock:
            nano = dict(self._nano25_latest)
        try:
            nano_age = time.time()-float(nano.get("received_wall"))
        except (TypeError, ValueError):
            nano_age = float("inf")
        nano_valid = (nano.get("state") in ("running", "reconnecting")
                      and nano_age <= 0.2)
        nano_force = self._record_vector(
            nano.get("force_n") if nano_valid else None, 3)
        nano_torque = self._record_vector(
            nano.get("torque_nm") if nano_valid else None, 3)
        force_xyz = [-nano_force[2], -nano_force[0], -nano_force[1]] \
            if all(value is not None for value in nano_force) else [None]*3

        test["response_rows"].append([
            response_time_s, *pose_values, *pose_speed, *pressure,
            *length_columns,
            *force_xyz, *nano_torque,
        ])

    def _queue_slow_rom_response_save(self, test):
        """Save the available ROM samples once, including interrupted runs."""
        if test.get("sampling_mode") not in SLOW_ROM_MODES:
            return ""
        path = str(test.get("response_path", ""))
        if not path or test.get("response_save_queued"):
            return path
        test["response_save_queued"] = True
        rows = [list(row) for row in test.get("response_rows", [])]

        def worker():
            try:
                with self._data_save_lock:
                    self._write_data_workbook(rows, path)
                self._recording_emit(
                    "slow_rom", "saved", f"慢速ROM Excel已保存，共{len(rows)}行", path)
            except Exception as exc:
                self._recording_emit(
                    "slow_rom", "error", "慢速ROM Excel保存失败：" + str(exc), path)

        # Non-daemon: even if the GUI is closed immediately after Stop, Python
        # waits for the workbook rename to finish instead of discarding data.
        threading.Thread(target=worker, daemon=False).start()
        return path

    @staticmethod
    def _reachability_case_info(target):
        active = [int(channel) for channel in np.flatnonzero(
            np.asarray(target, dtype=float) > 0.0)]
        labels = []
        for channel in active:
            info = REACHABILITY_CHANNEL_INFO[channel]
            labels.append(
                f"V{channel}({info['muscles']},{info['direction']})")
        return {
            "kind": ("single" if len(active) == 1 else
                     "pair" if len(active) == 2 else "triple"),
            "active_channels": active,
            "label": "+".join(labels),
        }

    @staticmethod
    def _reachability_endpoint_mm(pose_deg, arm_length_mm=REACHABILITY_ARM_LENGTH_MM):
        """Convert measured GUI X/Y swing angles to the physical arm-tip coordinate."""
        sx, sy = np.radians(np.asarray(pose_deg, dtype=float)[:2])
        theta = float(np.hypot(sx, sy))
        sinc = float(np.sinc(theta / np.pi))
        length = float(arm_length_mm)
        return np.asarray([
            -length * sinc * sy,
            length * sinc * sx,
            -length * np.cos(theta),
        ], dtype=float)

    @staticmethod
    def _reachability_pose_stop_reason(test, pose, q1_deg):
        """Return a pose stop reason, keeping extrema free of the plot envelope."""
        pose = np.asarray(pose, dtype=float)
        if test.get("sampling_mode") == "extrema":
            if np.asarray(test["target"], dtype=float)[1] > 0.0 and pose[1] <= -38.0:
                return "V1的Y-达到-38度底座机械限位"
            return None
        if test.get("sampling_mode") in SLOW_ROM_MODES:
            # The slow-ROM experiment uses the user's external hardware
            # protection.  Do not truncate the measured workspace with the
            # historical plotting envelope.
            return None
        if (float(q1_deg) >= 38.0 or np.any(pose < REACHABILITY_POSE_MIN_DEG)
                or np.any(pose > REACHABILITY_POSE_MAX_DEG)):
            return "达到经验姿态边界"
        return None

    def _reachability_send(self, test, outputs=None):
        try:
            if outputs is None:
                outputs = np.zeros(6, dtype=float)
            values = np.clip(np.asarray(outputs, dtype=float), 0.0,
                             float(configured_max_output_mv()))
            locks = np.asarray(test.get("requested_locks", np.zeros(6)), dtype=bool)
            command = ("M" + ",".join("1" if value else "0" for value in locks) +
                       ";" + ",".join(str(int(round(value))) for value in values) + "\n")
            self.ser.write(command.encode("ascii"))
            test["last_send"] = time.time()
            test["last_u"] = values.copy()
            return True
        except Exception as exc:
            self._reachability_abort("串口写入失败：" + str(exc))
            return False

    def _reachability_abort(self, why):
        test = self._reachability_test
        if test is None:
            return "Reachability map 未运行。"
        self._reachability_test = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        response_path = self._queue_slow_rom_response_save(test)
        save_text = f"；已有数据正在保存到 {response_path}" if response_path else ""
        self._reachability_emit("采样已中止并全机放气：" + why + save_text)
        return "Reachability map 采样已中止并全机放气：" + why + save_text

    def _start_reachability(self, mount, targets, sampling_mode, seed=None):
        """Start the shared safe state machine for random or sparse extreme sampling."""
        if not self.ser:
            return "先连接 IMU 串口。"
        if any(x is not None for x in (
                self._dm_ident, self._adc_cal, self._speed_test,
                self._x_comp_test, self._mpc_data_collect,
                self._phi_gradient_test,
                self._manual_pressure_test, self._reachability_test)):
            return "已有标定或实机测试正在运行，请先停止。"
        frame = self._latest_frame
        if frame is None or time.time() - frame["wall"] > 1.0:
            return "还没有收到新的 IMU 数据帧。"
        pressure = frame.get("kpa")
        if pressure is None or len(frame.get("locks", "")) != 6:
            return "当前固件帧没有六路压力/锁状态，不能安全采样。"
        if mount == "id":
            mount = "none"
        if mount not in ("ry-", "ry+", "none"):
            return "未知 IMU 映射。"
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 6 or len(targets) == 0:
            return "Reachability 目标集合无效。"
        for target in targets:
            try:
                self._validate_manual_pressure_targets(json.dumps(target.tolist()))
            except ValueError as exc:
                return "Reachability 目标集合无效：" + str(exc)
        response_folder = ""
        response_path = ""
        if sampling_mode in SLOW_ROM_MODES:
            if not self._recording_directory_confirmed:
                return "请先在操作模式点击“路径”并选择慢速ROM数据保存文件夹。"
            base_dir = os.path.abspath(self._recording_base_dir)
            if not os.path.isdir(base_dir):
                self._recording_directory_confirmed = False
                return "操作模式所选保存文件夹已不存在，请重新选择路径。"
            suffix_name = ("slow_rom_gradual" if sampling_mode == "slow_rom_gradual"
                           else "slow_rom")
            stem = self._recording_stamp() + "_" + suffix_name
            response_folder = os.path.join(base_dir, stem)
            suffix = 1
            while os.path.exists(response_folder):
                response_folder = os.path.join(base_dir, f"{stem}_{suffix:02d}")
                suffix += 1
            try:
                os.makedirs(response_folder)
            except Exception as exc:
                return "无法建立慢速ROM保存文件夹：" + str(exc)
            response_path = os.path.join(response_folder, "response.xlsx")
        count = len(targets)
        seed = int(time.time()) & 0x7fffffff if seed is None else int(seed)
        now = time.time()
        self._reachability_test = {
            "mount": mount, "targets": targets, "seed": seed,
            "sampling_mode": str(sampling_mode),
            "stage": "vent", "stage_t": now, "started": now,
            "case_i": 0, "ready_t": None, "qref": None,
            "target": np.zeros(6), "requested_locks": np.zeros(6, dtype=bool),
            "integral": np.zeros(6), "last_t": now, "last_send": 0.0,
            "last_u": np.zeros(6), "settled_t": None,
            "limit_settled_t": None,
            "previous_pose": None, "filtered_pose_speed": np.zeros(3),
            "lens_zero": None,
            "response_folder": response_folder, "response_path": response_path,
            "response_rows": [], "response_t0": None,
            "response_last_wall": 0.0, "response_group_zero_mm": None,
            "response_previous_pose_deg": None, "response_previous_time_s": None,
            "response_save_queued": False,
            "recenter_done_for_case": False,
            "recenter_output": np.zeros(6),
            "pose_samples": [], "rows": [], "results": [],
            "gains": self._manual_pressure_gains(),
        }
        try:
            # Normal preparation uses the firmware's zero-command vent path:
            # locks stay closed until measured pressure satisfies its unlock check.
            self.ser.write(b"W0,0,0,0,0,0\n")
        except Exception as exc:
            self._reachability_test = None
            return "Reachability map 启动失败：" + str(exc)
        if sampling_mode == "extrema":
            self._reachability_emit(
                f"极值采点已启动：{count} 个工况（6个单通道+12个合法双通道），"
                f"活动通道固定 {REACHABILITY_EXTREMA_MV:.0f} mV，"
                f"标称 {REACHABILITY_MAX_KPA:.0f} kPa；正在全机放气并建立零姿态。")
        elif sampling_mode in SLOW_ROM_MODES:
            release_text = (
                f"升压和降压均≤{SLOW_ROM_DEFLATE_SLEW_MV_PER_S:.0f} mV/s"
                if sampling_mode == "slow_rom_gradual" else
                f"升压≤{SLOW_ROM_SLEW_MV_PER_S:.0f} mV/s、点间全放气")
            self._reachability_emit(
                f"慢速ROM采样已启动：{count} 个工况（6单通道+12双通道+8三轴组合），"
                f"{release_text}；姿态速度达到"
                f"{SLOW_ROM_SPEED_HOLD_DEG_S:.0f}°/s暂停升压、"
                f"{SLOW_ROM_SPEED_BACKOFF_DEG_S:.0f}°/s按"
                f"{SLOW_ROM_BACKOFF_MV_PER_S:.0f} mV/s退压；工况间放气后若姿态未回到"
                f"±{SLOW_ROM_ZERO_TOL_DEG:.0f}°，将执行一次反向回零抖动；"
                f"约20 Hz数据将保存到 {response_path}。")
        else:
            self._reachability_emit(
                f"已启动：{count} 个压力点，随机种子 {seed}，每路上限 {REACHABILITY_MAX_KPA:.0f} kPa；"
                "正在全机放气并建立零姿态。")
        return (f"Reachability map 已启动：{count} 个点、每路≤{REACHABILITY_MAX_KPA:.0f} kPa；"
                "急停随时有效。")

    @QtCore.pyqtSlot(str, int, result=str)
    def startReachabilityMap(self, mount, sample_count):
        """Randomly sample safe six-valve pressure combinations and plot pose reachability."""
        count = int(np.clip(int(sample_count), REACHABILITY_MIN_SAMPLES,
                            REACHABILITY_MAX_SAMPLES))
        seed = int(time.time()) & 0x7fffffff
        return self._start_reachability(
            mount, self._reachability_targets(count, seed), "random", seed)

    @QtCore.pyqtSlot(str, result=str)
    def startReachabilityExtrema(self, mount):
        """Measure only the 18 single/pair maximum-pressure hardware boundary cases."""
        return self._start_reachability(
            mount, self._reachability_extrema_targets(), "extrema")

    @QtCore.pyqtSlot(str, result=str)
    def startSlowRomSampling(self, mount):
        """Slowly sweep 26 antagonist-safe boundary cases for dense ROM data."""
        return self._start_reachability(
            mount, self._slow_rom_targets(), "slow_rom")

    @QtCore.pyqtSlot(str, result=str)
    def startSlowRomGradualSampling(self, mount):
        """Slowly inflate and deflate all 26 antagonist-safe ROM cases."""
        return self._start_reachability(
            mount, self._slow_rom_targets(), "slow_rom_gradual")

    @QtCore.pyqtSlot(result=str)
    def stopReachabilityMap(self):
        if self._reachability_test is None:
            try:
                if self.ser:
                    self.ser.write(b"s\n")
            except Exception:
                pass
            return "Reachability map 未运行；已补发全机放气。"
        return self._reachability_abort("用户停止")

    def _reachability_begin_point(self, test, frame):
        index = test["case_i"]
        target = np.asarray(test["targets"][index], dtype=float)
        # Recheck the hard constraint immediately before every hardware command.
        self._validate_manual_pressure_targets(json.dumps(target.tolist()))
        test["target"] = target
        test["requested_locks"] = target > 0.0
        test["integral"][:] = 0.0
        test["last_u"][:] = 0.0
        test["pose_samples"] = []
        test["settled_t"] = None
        test["limit_settled_t"] = None
        test["previous_pose"] = None
        test["filtered_pose_speed"][:] = 0.0
        test["stage"] = "prelock"
        test["stage_t"] = frame["wall"]
        test["last_t"] = frame["wall"]
        test["ready_t"] = None
        if self._reachability_send(test, np.zeros(6)):
            case = self._reachability_case_info(target)
            active = ",".join(f"V{i}={target[i]:.0f}" for i in case["active_channels"])
            self._reachability_emit(
                f"点 {index + 1}/{len(test['targets'])}：{case['label']}，{active} kPa；"
                "先锁定100 ms。")

    def _reachability_end_point(self, test, reason, valid=True):
        recent = [sample for sample in test["pose_samples"]
                  if sample[0] >= test["pose_samples"][-1][0] - 0.5] \
            if test["pose_samples"] else []
        if recent:
            pose = np.median(np.asarray([sample[1] for sample in recent]), axis=0)
            pressure = np.median(np.asarray([sample[2] for sample in recent]), axis=0)
        else:
            pose = np.full(3, np.nan)
            pressure = np.full(6, np.nan)
            valid = False
        pose_json = [float(value) if np.isfinite(value) else None for value in pose]
        pressure_json = [float(value) if np.isfinite(value) else None for value in pressure]
        case = self._reachability_case_info(test["target"])
        endpoint = (self._reachability_endpoint_mm(pose).tolist()
                    if np.all(np.isfinite(pose)) else [None, None, None])
        endpoint_text = (f"({endpoint[0]:.1f},{endpoint[1]:.1f},{endpoint[2]:.1f}) mm"
                         if endpoint[0] is not None else "无有效坐标")
        test["results"].append({
            "sample": test["case_i"] + 1,
            **case,
            "target_kpa": test["target"].tolist(),
            "measured_kpa": pressure_json,
            "pose_deg": pose_json,
            "endpoint_mm": endpoint,
            "valid": bool(valid and np.all(np.isfinite(pose))),
            "stop_reason": reason,
        })
        gradual_release = test.get("sampling_mode") == "slow_rom_gradual"
        self._reachability_emit(
            f"点 {test['case_i'] + 1}/{len(test['targets'])} 完成："
            f"X/Y/φ={pose[0]:.1f}/{pose[1]:.1f}/{pose[2]:.1f}°，"
            f"末端={endpoint_text}"
            f"（{reason}）；正在{'缓慢降压' if gradual_release else '放气'}。")
        test["case_i"] += 1
        now = time.time()
        test["stage_t"] = now
        test["ready_t"] = None
        test["recenter_done_for_case"] = False
        test["recenter_output"][:] = 0.0
        if gradual_release and np.any(np.asarray(test["last_u"], dtype=float) > 0.5):
            test["stage"] = "slow_release"
            test["last_t"] = now
            return
        test["stage"] = "vent"
        test["requested_locks"][:] = False
        try:
            self.ser.write(b"W0,0,0,0,0,0\n")
            test["last_send"] = now
        except Exception as exc:
            self._reachability_abort("放气指令失败：" + str(exc))

    @staticmethod
    def _save_reachability_plot(results, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull

        valid = [row for row in results if row.get("valid")]
        if not valid:
            raise ValueError("没有有效姿态点可绘图")
        pose = np.asarray([row["pose_deg"] for row in valid], dtype=float)
        pressure = np.asarray([row["measured_kpa"] for row in valid], dtype=float)
        color = np.nanmax(pressure, axis=1)
        fig = plt.figure(figsize=(12.5, 9.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 2)
        ax3 = fig.add_subplot(grid[0, 0], projection="3d")
        scat = ax3.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color,
                           cmap="viridis", s=32, alpha=0.85)
        ax3.set_xlabel("X (deg)"); ax3.set_ylabel("Y (deg)"); ax3.set_zlabel("phi (deg)")
        ax3.set_title("Measured 3D reachability")
        projections = [
            (fig.add_subplot(grid[0, 1]), 0, 1, "X (deg)", "Y (deg)", "X-Y"),
            (fig.add_subplot(grid[1, 0]), 0, 2, "X (deg)", "phi (deg)", "X-phi"),
            (fig.add_subplot(grid[1, 1]), 1, 2, "Y (deg)", "phi (deg)", "Y-phi"),
        ]
        for axis, xi, yi, xlabel, ylabel, title in projections:
            axis.scatter(pose[:, xi], pose[:, yi], c=color, cmap="viridis",
                         s=28, alpha=0.82)
            points = pose[:, [xi, yi]]
            if len(points) >= 3 and np.linalg.matrix_rank(points - np.mean(points, axis=0)) >= 2:
                hull = ConvexHull(points)
                boundary = points[np.r_[hull.vertices, hull.vertices[0]]]
                axis.fill(boundary[:, 0], boundary[:, 1], alpha=0.10, color="tab:blue")
                axis.plot(boundary[:, 0], boundary[:, 1], linewidth=1.2,
                          color="tab:blue", label="sampled envelope")
            axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.set_title(title)
            axis.grid(True, alpha=0.25)
        fig.colorbar(scat, ax=list(fig.axes), shrink=0.72, label="Max measured pressure (kPa)")
        fig.suptitle("TRO hardware reachability map (random pressure sampling, limit 200 kPa)")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _show_reachability_plot(self, path):
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            return
        dialog = QtWidgets.QDialog()
        title = ("TRO V3 repeat validation"
                 if "V3_repeat" in os.path.basename(path)
                 else "TRO Reachability map")
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(QtCore.Qt.AlignCenter)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll)
        dialog.resize(1250, 900)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._reachability_windows.append(dialog)
        dialog.destroyed.connect(
            lambda _=None, window=dialog: self._reachability_windows.remove(window)
            if window in self._reachability_windows else None)
        dialog.show()

    def _reachability_finish(self, test):
        self._reachability_test = None
        try:
            if self.ser:
                self.ser.write(b"s\n")
        except Exception:
            pass
        response_path = self._queue_slow_rom_response_save(test)
        log_dir = str(CONTROL_LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sampling_mode = test.get("sampling_mode")
        extrema = sampling_mode == "extrema"
        slow_rom = sampling_mode in SLOW_ROM_MODES
        gradual_release = sampling_mode == "slow_rom_gradual"
        prefix = ("reachability_extrema" if extrema else
                  "slow_rom_gradual" if gradual_release else
                  "slow_rom" if slow_rom else "reachability_map")
        raw_path = os.path.abspath(os.path.join(log_dir, f"{prefix}_raw_{stamp}.csv"))
        result_path = os.path.abspath(os.path.join(log_dir, f"{prefix}_{stamp}.json"))
        plot_path = os.path.abspath(os.path.join(log_dir, f"{prefix}_{stamp}.png"))
        summary_path = (os.path.abspath(os.path.join(
            log_dir, f"{prefix}_summary_{stamp}.csv"))
                        if extrema or slow_rom else None)
        try:
            names = (["sample", "stage", "t", "x", "y", "phi", "q1", "q2", "q3"] +
                     [f"target{i}_kpa" for i in range(6)] +
                     [f"P{i}_kpa" for i in range(6)] +
                     [f"u{i}_mv" for i in range(6)] +
                     ["pose_speed_x_deg_s", "pose_speed_y_deg_s",
                      "pose_speed_phi_deg_s"] +
                     [field for label in self.DATA_RECORD_GROUP_LABELS
                      for field in (f"L_{label}_mm", f"d_active_{label}_mm",
                                    f"d_passive_{label}_mm")] +
                     [f"L_channel{i}_mm" for i in range(6)] +
                     [f"sensor{i}_signed_mm" for i in range(1, 21)] +
                     [f"d_sensor{i}_mm" for i in range(1, 21)] + ["locks"])
            with open(raw_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader(); writer.writerows(test["rows"])
            blob = {
                "schema": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mount": test["mount"], "seed": test["seed"],
                "pressure_limit_kpa": REACHABILITY_MAX_KPA,
                "output_limit_mv": configured_max_output_mv(),
                "extrema_fixed_command_mv": (REACHABILITY_EXTREMA_MV
                                                if extrema else None),
                "slow_rom_voltage_slew_mv_per_s": (SLOW_ROM_SLEW_MV_PER_S
                                                     if slow_rom else None),
                "slow_rom_deflate_slew_mv_per_s": (
                    SLOW_ROM_DEFLATE_SLEW_MV_PER_S if gradual_release else None),
                "slow_rom_backoff_mv_per_s": (SLOW_ROM_BACKOFF_MV_PER_S
                                                if slow_rom else None),
                "slow_rom_speed_hold_deg_s": (SLOW_ROM_SPEED_HOLD_DEG_S
                                                if slow_rom else None),
                "slow_rom_speed_backoff_deg_s": (SLOW_ROM_SPEED_BACKOFF_DEG_S
                                                   if slow_rom else None),
                "slow_rom_zero_tolerance_deg": (SLOW_ROM_ZERO_TOL_DEG
                                                  if slow_rom else None),
                "slow_rom_recenter_pulse_s": (SLOW_ROM_RECENTER_PULSE_S
                                                if slow_rom else None),
                "slow_rom_recenter_voltage_mv": (
                    [SLOW_ROM_RECENTER_MIN_MV, SLOW_ROM_RECENTER_MAX_MV]
                    if slow_rom else None),
                "sampling": (
                    "six single-channel plus twelve legal two-channel maximum-pressure cases"
                    if extrema else
                    "slow inflate and slow deflate sweep of six single, twelve legal pair and eight legal triple cases"
                    if gradual_release else
                    "slow inflate sweep of six single, twelve legal pair and eight legal triple cases"
                    if slow_rom else
                    "random antagonist-exclusive targets plus six single-channel boundary points"),
                "antagonist_hard_constraint": [[0, 1], [2, 3], [4, 5]],
                "pose_envelope_abort_enabled": not (extrema or slow_rom),
                "software_pressure_abort_enabled": not slow_rom,
                "extrema_mode_pose_policy": (
                    "no empirical X/Y/phi envelope abort; only V1 Y- base collision at -38deg"
                    if extrema else None),
                "lock_lead_ms": int(round(1000.0 * REACHABILITY_PRELOCK_S)),
                "unlock_pressure_kpa": 30.0,
                "unlock_threshold_scope": "vent/unlock judgement only",
                "arm_length_mm": REACHABILITY_ARM_LENGTH_MM,
                "coordinate_definition": (
                    "endpoint_mm=[x_B,y_B,z_B], spherical centre is origin, "
                    "zero pose hangs along -z_B; phi does not move the arm-tip centreline"),
                "results": test["results"], "raw_log_path": raw_path,
            }
            if extrema:
                from v6_reachability import measured_extrema_calibration
                blob["calibrated_envelope"] = measured_extrema_calibration(blob)
            with open(result_path, "w", encoding="utf-8") as stream:
                json.dump(blob, stream, ensure_ascii=False, indent=2, allow_nan=False)
            if extrema or slow_rom:
                summary_names = (["sample", "kind", "label", "active_channels",
                                  "valid", "stop_reason", "x_deg", "y_deg", "phi_deg",
                                  "x_B_mm", "y_B_mm", "z_B_mm"] +
                                 [f"target{i}_kpa" for i in range(6)] +
                                 [f"measured{i}_kpa" for i in range(6)])
                with open(summary_path, "w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=summary_names)
                    writer.writeheader()
                    for result in test["results"]:
                        pose = result.get("pose_deg", [None] * 3)
                        endpoint = result.get("endpoint_mm", [None] * 3)
                        target = result.get("target_kpa", [None] * 6)
                        measured_result = result.get("measured_kpa", [None] * 6)
                        row = {
                            "sample": result.get("sample"), "kind": result.get("kind"),
                            "label": result.get("label"),
                            "active_channels": "+".join(
                                str(v) for v in result.get("active_channels", [])),
                            "valid": result.get("valid"),
                            "stop_reason": result.get("stop_reason"),
                            "x_deg": pose[0], "y_deg": pose[1], "phi_deg": pose[2],
                            "x_B_mm": endpoint[0], "y_B_mm": endpoint[1],
                            "z_B_mm": endpoint[2],
                        }
                        row.update({f"target{i}_kpa": target[i] for i in range(6)})
                        row.update({f"measured{i}_kpa": measured_result[i] for i in range(6)})
                        writer.writerow(row)
                if extrema:
                    from v6_reachability import save_measured_extrema_plot
                    rendered_path = save_measured_extrema_plot(blob, plot_path)
                    plot_path = rendered_path or plot_path
                else:
                    self._save_reachability_plot(test["results"], plot_path)
            else:
                self._save_reachability_plot(test["results"], plot_path)
            self._show_reachability_plot(plot_path)
            valid_count = sum(bool(row.get("valid")) for row in test["results"])
            valid_results = [row for row in test["results"] if row.get("valid")]
            range_text = ""
            if valid_results:
                endpoints = np.asarray(
                    [row["endpoint_mm"] for row in valid_results], dtype=float)
                range_text = (f"；末端实测范围 x_B[{np.min(endpoints[:, 0]):.1f},"
                              f"{np.max(endpoints[:, 0]):.1f}] mm，"
                              f"y_B[{np.min(endpoints[:, 1]):.1f},"
                              f"{np.max(endpoints[:, 1]):.1f}] mm，"
                              f"z_B[{np.min(endpoints[:, 2]):.1f},"
                              f"{np.max(endpoints[:, 2]):.1f}] mm")
            self._reachability_emit(
                f"采样完成：{valid_count}/{len(test['results'])} 个有效点{range_text}；"
                f"图 {plot_path}；数据 {result_path}；原始日志 {raw_path}"
                f"{'；汇总 ' + summary_path if summary_path else ''}"
                f"{'；Excel ' + response_path if response_path else ''}。已全机放气。")
        except Exception as exc:
            self._reachability_emit("采样结束并已放气，但保存/绘图失败：" + str(exc))

    def _reachability_on_frame(self, frame):
        test = self._reachability_test
        if test is None:
            return
        pressure = frame.get("kpa")
        if pressure is None or len(frame.get("locks", "")) != 6:
            self._reachability_abort("压力或锁状态遥测中断")
            return
        now = frame["wall"]
        measured = np.asarray(pressure, dtype=float)
        if not np.all(np.isfinite(measured)):
            self._reachability_abort("压力遥测包含非有限值")
            return
        if (test.get("sampling_mode") not in SLOW_ROM_MODES and
                float(np.max(measured)) > REACHABILITY_PRESSURE_ABORT_KPA):
            self._reachability_abort(
                f"实测压力超过 {REACHABILITY_PRESSURE_ABORT_KPA:.0f} kPa 保护线")
            return

        if test["stage"] == "vent":
            response_recorded = False
            if test.get("sampling_mode") in SLOW_ROM_MODES and test.get("qref") is not None:
                try:
                    _, vent_pose = self._reachability_pose_from_frame(test, frame)
                    self._slow_rom_append_response(test, frame, vent_pose, now)
                    response_recorded = True
                except Exception as exc:
                    self._reachability_abort("姿态转换失败：" + str(exc))
                    return
            unlock_safe = (frame["locks"] == "oooooo" and
                           float(np.max(measured)) <= 30.0)
            if now - test["last_send"] >= 0.25:
                try:
                    self.ser.write(b"W0,0,0,0,0,0\n")
                    test["last_send"] = now
                except Exception as exc:
                    self._reachability_abort("放气指令失败：" + str(exc))
                    return
            if unlock_safe:
                if test["ready_t"] is None:
                    test["ready_t"] = now
                if now - test["ready_t"] >= REACHABILITY_VENT_STABLE_S:
                    if test["qref"] is None:
                        test["qref"] = np.asarray(frame["quat"], dtype=float).copy()
                        if test.get("sampling_mode") != "extrema":
                            test["results"].append({
                                "sample": 0, "kind": "origin", "label": "zero pressure",
                                "active_channels": [], "target_kpa": [0.0] * 6,
                                "measured_kpa": measured.tolist(),
                                "pose_deg": [0.0, 0.0, 0.0],
                                "endpoint_mm": [0.0, 0.0, -REACHABILITY_ARM_LENGTH_MM],
                                "valid": True,
                                "stop_reason": "vented origin",
                            })
                    if test.get("sampling_mode") in SLOW_ROM_MODES and not response_recorded:
                        try:
                            _, vent_pose = self._reachability_pose_from_frame(test, frame)
                            self._slow_rom_append_response(test, frame, vent_pose, now)
                        except Exception as exc:
                            self._reachability_abort("姿态转换失败：" + str(exc))
                            return
                    if (test.get("sampling_mode") in SLOW_ROM_MODES and
                            test["case_i"] > 0 and
                            not test.get("recenter_done_for_case", False)):
                        recenter_locks, recenter_output = \
                            self._slow_rom_recenter_command(vent_pose)
                        if np.any(recenter_locks):
                            test["requested_locks"] = recenter_locks
                            test["recenter_output"] = recenter_output
                            test["stage"] = "recenter_prelock"
                            test["stage_t"] = now
                            test["ready_t"] = None
                            if not self._reachability_send(test, np.zeros(6)):
                                return
                            channels = ",".join(
                                f"V{i}={recenter_output[i]:.0f}mV"
                                for i in np.flatnonzero(recenter_locks))
                            self._reachability_emit(
                                f"回零检查：X/Y/φ={vent_pose[0]:.1f}/"
                                f"{vent_pose[1]:.1f}/{vent_pose[2]:.1f}°，"
                                f"超过±{SLOW_ROM_ZERO_TOL_DEG:.0f}°；"
                                f"反向通道 {channels} 先预锁100 ms，再短促充气一次。")
                            return
                    if test["case_i"] >= len(test["targets"]):
                        self._reachability_finish(test)
                    else:
                        self._reachability_begin_point(test, frame)
            else:
                test["ready_t"] = None
            if self._reachability_test is not None and now - test["stage_t"] > 45.0:
                self._reachability_abort("45秒内未完成放气/开锁")
            return

        active = np.flatnonzero(test["requested_locks"])
        if test["stage"] == "slow_release":
            try:
                _, release_pose = self._reachability_pose_from_frame(test, frame)
                self._slow_rom_append_response(test, frame, release_pose, now)
            except Exception as exc:
                self._reachability_abort("姿态转换失败：" + str(exc))
                return
            dt = float(np.clip(now-test["last_t"], 0.01, 0.2))
            test["last_t"] = now
            previous_pose = test.get("previous_pose")
            raw_speed = (np.zeros(3) if previous_pose is None else
                         (release_pose-np.asarray(previous_pose, dtype=float))/dt)
            test["previous_pose"] = release_pose.copy()
            test["filtered_pose_speed"] = np.clip(
                0.8*np.asarray(test.get("filtered_pose_speed", np.zeros(3)), dtype=float) +
                0.2*raw_speed, -200.0, 200.0)
            output = self._slow_rom_release_step(
                test["last_u"], test["filtered_pose_speed"], dt)
            if not self._reachability_send(test, output):
                return
            if np.all(output[active] <= 0.5):
                test["requested_locks"][:] = False
                test["stage"] = "vent"
                test["stage_t"] = now
                test["ready_t"] = None
                try:
                    self.ser.write(b"W0,0,0,0,0,0\n")
                    test["last_send"] = now
                except Exception as exc:
                    self._reachability_abort("缓慢降压后放气失败：" + str(exc))
                    return
                self._reachability_emit("六路电压已缓慢降至0 mV；开始全零放气/解锁。")
            return

        if test["stage"] == "recenter_prelock":
            try:
                _, recenter_pose = self._reachability_pose_from_frame(test, frame)
                self._slow_rom_append_response(test, frame, recenter_pose, now)
            except Exception as exc:
                self._reachability_abort("姿态转换失败：" + str(exc))
                return
            if now - test["last_send"] >= 0.15:
                if not self._reachability_send(test, np.zeros(6)):
                    return
            locked = all(frame["locks"][int(channel)] == "L" for channel in active)
            if locked and now-test["stage_t"] >= REACHABILITY_PRELOCK_S:
                test["stage"] = "recenter_pulse"
                test["stage_t"] = now
                if not self._reachability_send(test, test["recenter_output"]):
                    return
            elif now-test["stage_t"] > 5.0:
                self._reachability_abort("回零抖动预锁超过5秒")
            return

        if test["stage"] == "recenter_pulse":
            try:
                _, recenter_pose = self._reachability_pose_from_frame(test, frame)
                self._slow_rom_append_response(test, frame, recenter_pose, now)
            except Exception as exc:
                self._reachability_abort("姿态转换失败：" + str(exc))
                return
            if now-test["stage_t"] < SLOW_ROM_RECENTER_PULSE_S:
                if now-test["last_send"] >= 0.05:
                    self._reachability_send(test, test["recenter_output"])
                return
            if test.get("sampling_mode") == "slow_rom_gradual":
                test["stage"] = "recenter_slow_release"
                test["stage_t"] = now
                test["last_t"] = now
                test["previous_pose"] = recenter_pose.copy()
                test["filtered_pose_speed"][:] = 0.0
                self._reachability_emit("回零短促充气完成；开始按100 mV/s缓慢降至0 mV。")
                return
            test["recenter_done_for_case"] = True
            test["requested_locks"][:] = False
            test["recenter_output"][:] = 0.0
            test["stage"] = "vent"
            test["stage_t"] = now
            test["ready_t"] = None
            try:
                self.ser.write(b"W0,0,0,0,0,0\n")
                test["last_send"] = now
            except Exception as exc:
                self._reachability_abort("回零抖动后放气失败：" + str(exc))
                return
            self._reachability_emit("回零抖动完成；再次全机放气并检查零位。")
            return

        if test["stage"] == "recenter_slow_release":
            try:
                _, recenter_pose = self._reachability_pose_from_frame(test, frame)
                self._slow_rom_append_response(test, frame, recenter_pose, now)
            except Exception as exc:
                self._reachability_abort("姿态转换失败：" + str(exc))
                return
            dt = float(np.clip(now-test["last_t"], 0.01, 0.2))
            test["last_t"] = now
            previous_pose = test.get("previous_pose")
            raw_speed = (np.zeros(3) if previous_pose is None else
                         (recenter_pose-np.asarray(previous_pose, dtype=float))/dt)
            test["previous_pose"] = recenter_pose.copy()
            test["filtered_pose_speed"] = np.clip(
                0.8*np.asarray(test.get("filtered_pose_speed", np.zeros(3)), dtype=float) +
                0.2*raw_speed, -200.0, 200.0)
            output = self._slow_rom_release_step(
                test["last_u"], test["filtered_pose_speed"], dt)
            if not self._reachability_send(test, output):
                return
            if np.all(output[active] <= 0.5):
                test["recenter_done_for_case"] = True
                test["requested_locks"][:] = False
                test["recenter_output"][:] = 0.0
                test["stage"] = "vent"
                test["stage_t"] = now
                test["ready_t"] = None
                try:
                    self.ser.write(b"W0,0,0,0,0,0\n")
                    test["last_send"] = now
                except Exception as exc:
                    self._reachability_abort("回零缓慢降压后放气失败：" + str(exc))
                    return
                self._reachability_emit("回零电压已缓慢降至0 mV；再次全机放气。")
            return

        if test["stage"] == "prelock":
            if test.get("sampling_mode") in SLOW_ROM_MODES:
                try:
                    _, prelock_pose = self._reachability_pose_from_frame(test, frame)
                    self._slow_rom_append_response(test, frame, prelock_pose, now)
                except Exception as exc:
                    self._reachability_abort("姿态转换失败：" + str(exc))
                    return
            if now - test["last_send"] >= 0.15:
                if not self._reachability_send(test, np.zeros(6)):
                    return
            locked = all(frame["locks"][int(channel)] == "L" for channel in active)
            if locked and now - test["stage_t"] >= REACHABILITY_PRELOCK_S:
                test["stage"] = "run"
                test["stage_t"] = now
                test["last_t"] = now
                test["pose_samples"] = []
                test["settled_t"] = None
                test["limit_settled_t"] = None
            elif now - test["stage_t"] > 5.0:
                self._reachability_abort("预锁超过5秒")
            return

        try:
            q, pose = self._reachability_pose_from_frame(test, frame)
        except Exception as exc:
            self._reachability_abort("姿态转换失败：" + str(exc))
            return
        dt = float(np.clip(now - test["last_t"], 0.01, 0.2))
        test["last_t"] = now
        previous_pose = test.get("previous_pose")
        raw_pose_speed = (np.zeros(3) if previous_pose is None else
                          (pose-np.asarray(previous_pose, dtype=float))/dt)
        test["previous_pose"] = pose.copy()
        test["filtered_pose_speed"] = np.clip(
            0.8*np.asarray(test.get("filtered_pose_speed", np.zeros(3)), dtype=float) +
            0.2*raw_pose_speed, -200.0, 200.0)
        pose_speed = np.asarray(test["filtered_pose_speed"], dtype=float)
        target = test["target"]
        gains = test["gains"]
        error = target - measured
        configured_limit = float(configured_max_output_mv())
        if test.get("sampling_mode") == "extrema":
            # ROM extrema are a fixed-voltage experiment: every selected
            # channel receives exactly 2200 mV (subject only to a deliberately
            # lower configured hardware ceiling), rather than pressure PI
            # reducing voltage as it approaches the nominal 200 kPa point.
            limit = min(float(REACHABILITY_EXTREMA_MV), configured_limit)
            output = np.where(target > 0.0, limit, 0.0)
            raw_u = output.copy()
            test["integral"][:] = 0.0
        elif test.get("sampling_mode") in SLOW_ROM_MODES:
            limit = min(float(REACHABILITY_EXTREMA_MV), configured_limit)
            output = self._slow_rom_output_step(
                test["last_u"], target, pose_speed, dt, limit)
            raw_u = output.copy()
            test["integral"][:] = 0.0
        else:
            effective = np.where(np.abs(error) <= gains["deadband"], 0.0, error)
            integral = np.clip(test["integral"] + gains["ki"] * effective * dt,
                               -1000.0, 1000.0)
            integral[target <= 0.0] = 0.0
            feedforward = target / 0.09
            raw_u = feedforward + gains["kp"] * effective + integral
            limit = configured_limit
            output = np.clip(raw_u, 0.0, limit)
            output[target <= 0.0] = 0.0
            freeze = (((raw_u > limit) & (effective > 0.0)) |
                      ((raw_u < 0.0) & (effective < 0.0)))
            integral[freeze] = test["integral"][freeze]
            test["integral"] = integral
        if not self._reachability_send(test, output):
            return

        elapsed = now - test["stage_t"]
        test["pose_samples"].append((elapsed, pose.copy(), measured.copy()))
        row = {"sample": test["case_i"] + 1, "stage": test["stage"],
               "t": round(now - test["started"], 5), "x": round(float(pose[0]), 6),
               "y": round(float(pose[1]), 6), "phi": round(float(pose[2]), 6),
               "q1": round(float(np.degrees(q[0])), 6),
               "q2": round(float(np.degrees(q[1])), 6),
               "q3": round(float(np.degrees(q[2])), 6), "locks": frame["locks"]}
        row.update({f"target{i}_kpa": round(float(target[i]), 5) for i in range(6)})
        row.update({f"P{i}_kpa": round(float(measured[i]), 5) for i in range(6)})
        row.update({f"u{i}_mv": int(round(float(output[i]))) for i in range(6)})
        row.update({
            "pose_speed_x_deg_s": round(float(pose_speed[0]), 5),
            "pose_speed_y_deg_s": round(float(pose_speed[1]), 5),
            "pose_speed_phi_deg_s": round(float(pose_speed[2]), 5),
        })
        lens = self._record_vector(frame.get("lens"), 20)
        if all(value is not None for value in lens):
            lens_array = np.asarray(lens, dtype=float)
            if test.get("lens_zero") is None:
                test["lens_zero"] = lens_array.copy()
            lens_delta = lens_array-np.asarray(test["lens_zero"], dtype=float)
        else:
            lens_array = np.full(20, np.nan)
            lens_delta = np.full(20, np.nan)
        channel_lengths = self._record_vector(frame.get("channel_length_mm"), 6)
        group_delta = []
        for indexes in self.MUSCLE_SENSOR_GROUPS:
            active_sensors = lens_delta[list(indexes)]
            passive_sensors = lens_delta[[index+10 for index in indexes]]
            group_delta.append([
                float(np.mean(active_sensors))
                if np.all(np.isfinite(active_sensors)) else None,
                float(np.mean(passive_sensors))
                if np.all(np.isfinite(passive_sensors)) else None,
            ])
        for label, channel in zip(self.DATA_RECORD_GROUP_LABELS,
                                  self.DATA_RECORD_CHANNEL_ORDER):
            row[f"L_{label}_mm"] = channel_lengths[channel]
            row[f"d_active_{label}_mm"] = (
                round(group_delta[channel][0], 5)
                if group_delta[channel][0] is not None else None)
            row[f"d_passive_{label}_mm"] = (
                round(group_delta[channel][1], 5)
                if group_delta[channel][1] is not None else None)
        row.update({f"L_channel{i}_mm": channel_lengths[i] for i in range(6)})
        row.update({f"sensor{i+1}_signed_mm": (
            round(float(lens_array[i]), 5) if np.isfinite(lens_array[i]) else None)
                    for i in range(20)})
        row.update({f"d_sensor{i+1}_mm": (
            round(float(lens_delta[i]), 5) if np.isfinite(lens_delta[i]) else None)
                    for i in range(20)})
        test["rows"].append(row)
        self._slow_rom_append_response(test, frame, pose, now)

        q1_deg = float(np.degrees(q[0]))
        # Extreme sampling must be allowed to reveal a workspace larger than
        # the plotting envelope.  The generic q1>=38 test would incorrectly
        # stop Y+ and compound directions; only the confirmed V1/Y- collision
        # remains as a mechanical boundary in that mode.
        pose_stop_reason = self._reachability_pose_stop_reason(test, pose, q1_deg)
        if pose_stop_reason:
            self._reachability_end_point(test, pose_stop_reason, valid=True)
            return
        pressure_ok = bool(active.size and np.all(
            np.abs(error[active]) <= np.maximum(REACHABILITY_PRESSURE_TOL_KPA,
                                                 0.05 * target[active])))
        locks_ok = all(frame["locks"][int(channel)] == "L" for channel in active)
        if pressure_ok and locks_ok:
            if test["settled_t"] is None:
                test["settled_t"] = now
            recent_pose = np.asarray(
                [sample[1] for sample in test["pose_samples"] if sample[0] >= elapsed - 0.5])
            pose_still = (len(recent_pose) >= 5 and
                          float(np.max(np.ptp(recent_pose, axis=0))) <= 1.0)
            if now - test["settled_t"] >= REACHABILITY_SETTLE_S and pose_still:
                self._reachability_end_point(test, "压力和姿态稳定", valid=True)
                return
        else:
            test["settled_t"] = None
        if test.get("sampling_mode") == "extrema" or test.get("sampling_mode") in SLOW_ROM_MODES:
            recent = [sample for sample in test["pose_samples"]
                      if sample[0] >= elapsed - 0.75]
            if len(recent) >= 5:
                recent_pose = np.asarray([sample[1] for sample in recent])
                recent_pressure = np.asarray([sample[2] for sample in recent])
                pose_still = float(np.max(np.ptp(recent_pose, axis=0))) <= 1.0
                pressure_still = bool(active.size and np.all(
                    np.ptp(recent_pressure[:, active], axis=0) <= 2.0))
                output_limited = bool(active.size and np.all(
                    output[active] >= 0.99 * limit))
                if pose_still and pressure_still and output_limited:
                    if test["limit_settled_t"] is None:
                        test["limit_settled_t"] = now
                    elif now - test["limit_settled_t"] >= REACHABILITY_SETTLE_S:
                        self._reachability_end_point(
                            test, f"{limit:.0f} mV固定输出下压力/姿态稳定",
                            valid=True)
                        return
                else:
                    test["limit_settled_t"] = None
        point_timeout = (SLOW_ROM_POINT_TIMEOUT_S
                         if test.get("sampling_mode") in SLOW_ROM_MODES
                         else REACHABILITY_POINT_TIMEOUT_S)
        if elapsed >= point_timeout:
            if test.get("sampling_mode") == "extrema" or test.get("sampling_mode") in SLOW_ROM_MODES:
                self._reachability_end_point(
                    test, (f"慢速爬升{point_timeout:.0f}秒实测边界"
                           if test.get("sampling_mode") in SLOW_ROM_MODES else
                           f"{limit:.0f} mV固定输出下10秒实测极值"),
                    valid=True)
            else:
                self._reachability_end_point(test, "单点10秒超时", valid=False)

    # ───────────── V6 几何/编码器自动充放气标定（候选结果，不接管正常控制） ─────────────
    def _v6_geometry_emit(self, msg):
        print("# V6 几何自动标定：" + msg)
        self.rx.emit("[v6geo] " + msg + "\n")

    @staticmethod
    def _v6_geometry_lengths(frame):
        values = frame.get("channel_length_mm")
        if values is None or len(values) != 6 or any(v is None for v in values):
            return None
        answer = np.asarray(values, dtype=float)
        return answer if np.all(np.isfinite(answer)) else None

    @QtCore.pyqtSlot(result=str)
    def analyzeV6GeometryLogs(self):
        """只读分析现有真机日志并保存候选，不需要串口。"""
        if self._v6_log_analysis_running:
            return "V6 旧日志正在后台分析，请等待完成消息。"
        self._v6_log_analysis_running = True
        threading.Thread(target=self._analyze_v6_geometry_logs_worker,
                         daemon=True).start()
        return "V6 旧日志后台分析已启动；串口轮询和当前控制不会被阻塞。"

    def _analyze_v6_geometry_logs_worker(self):
        try:
            from v6_geometry_calibration import (
                concise_summary, fit_geometry_logs, save_candidate)
            log_dir = str(CONTROL_LOG_DIR)
            paths = []
            if os.path.isdir(log_dir):
                for name in sorted(os.listdir(log_dir)):
                    if not name.lower().endswith(".csv") or name.endswith("_laps.csv"):
                        continue
                    if name.startswith(("mpc_", "musclepi_", "v6_geometry_passive_",
                                        "v6_geometry_auto_")):
                        paths.append(os.path.join(log_dir, name))
            if not paths:
                raise ValueError("没有找到可分析的现有真机日志")
            result = fit_geometry_logs(paths)
            candidate = save_candidate(result, V6_GEOMETRY_CANDIDATE_PATH)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            report = save_candidate(
                result, os.path.join(log_dir, f"v6_geometry_log_fit_{stamp}.json"))
            summary = concise_summary(result)
            msg = (f"旧日志分析完成：{result['samples']} 帧、"
                   f"{len(result.get('source_files', []))} 个日志；{summary}。"
                   f"候选已保存 {candidate}；详细报告 {report}。未应用到控制器。")
            self._v6_geometry_emit(msg)
        except Exception as exc:
            msg = "旧日志分析失败：" + str(exc)
            self._v6_geometry_emit(msg)
        finally:
            self._v6_log_analysis_running = False

    def _poll(self):
        if not self.ser:
            return
        try:
            d = self.ser.read(4096)
            self._errs = 0
        except Exception:
            # 串口失联检测（拔线/驱动崩）：连续 25 次读异常（~0.5s）→ 关口 + 通知页面停控
            self._errs = getattr(self, "_errs", 0) + 1
            if self._errs == 25:
                print("# !! 串口失联（读异常持续 0.5s），已关闭端口并通知页面停止控制")
                if self._manual_pressure_test is not None:
                    self._manual_pressure_stop("串口失联")
                if self._dm_ident is not None:
                    self._ident_abort("串口失联")
                if self._speed_test is not None:
                    self._speed_abort("串口失联")
                if self._reachability_test is not None:
                    self._reachability_abort("串口失联")
                try: self.ser.close()
                except Exception: pass
                self.ser = None
                self.timer.stop()
                self.lost.emit("读串口连续异常——线拔了或驱动崩了")
            return
        import time as _t
        now = _t.time()
        # 心跳：每 250ms 发一个空行——固件以“最近 1s 有无来字节”判断宿主在不在，
        # 不在就停止打印（防 CDC 发送端僵死）。空行对固件解析是无操作。
        if now - getattr(self, "_last_ka", 0) > 0.15:
            self._last_ka = now
            try: self.ser.write(b"\n")
            except Exception: pass
        if d:
            self._last_rx = now
            text = d.decode("ascii", "ignore")
            # 本地 Qt 直接把固件帧第 9 段锁状态交给内嵌 MPC；无需页面中转。
            self._frame_buf += text
            lines = self._frame_buf.splitlines(keepends=True)
            self._frame_buf = "" if not lines or lines[-1].endswith(("\n", "\r")) else lines.pop()
            for line in lines:
                frame = self._decode_frame(line)
                if frame is not None:
                    self._latest_frame = frame
                    locks = frame["locks"]
                    if len(locks) == 6:
                        try:
                            import mpc_station as st
                            st.HOST_LOCKS = locks
                        except Exception:
                            pass
                    # A 指令会重新采六路零点，只能在确认全开锁且压力已低时恢复旧公式。
                    p = frame.get("kpa")
                    adc_apply_safe = (locks == "oooooo" and p is not None and
                                      float(np.max(p)) <= 30.0)
                    if frame.get("adc_raw") is not None and not self._adc_cal_applied \
                            and self._adc_cal is None and adc_apply_safe:
                        self._apply_saved_adc_cal()
                    self._ident_on_frame(frame)
                    self._adc_cal_on_frame(frame)
                    self._speed_on_frame(frame)
                    self._x_comp_on_frame(frame)
                    self._mpc_data_on_frame(frame)
                    self._phi_gradient_on_frame(frame)
                    self._manual_pressure_on_frame(frame)
                    self._reachability_on_frame(frame)
                    self._force_prediction_on_frame(frame)
            self.rx.emit(text)
            return
        # 自动踹一脚：开口 2 秒没任何数据 → 再抖一次 DTR + 发询问（半僵态多能救活）
        if not self._kicked and now - self._last_rx > 2.0:
            self._kicked = True
            message = "连接后 2 秒仍无数据：已自动重抖 DTR 并发送询问；请按板上 RST 重新启动机器。"
            try:
                self.ser.dtr = False; _t.sleep(0.1); self.ser.dtr = True
                self.ser.write(b"\n?\n")
            except Exception:
                message = "连接后 2 秒仍无数据，自动重试未成功；请按板上 RST 重新启动机器。"
            print("# " + message)
            self.notice.emit(message)


class NativeGuiRecorder(QtCore.QObject):
    """Capture the embedded Qt GUI directly; avoids unreliable WebEngine screen sharing."""
    FPS = 12
    MAX_WIDTH = 1600
    MAX_HEIGHT = 1200

    def __init__(self, widget, bridge, kind="gui"):
        super().__init__(widget)
        self.widget = widget
        self.bridge = bridge
        self.kind = str(kind)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._capture)
        self._lock = threading.Lock()
        self._item = None

    @staticmethod
    def _ffmpeg_executable():
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return shutil.which("ffmpeg")

    def _grab_image(self, capture_rect=None):
        rect = self.widget.rect()
        if capture_rect is not None:
            requested = QtCore.QRect(*capture_rect).intersected(rect)
            if requested.width() >= 2 and requested.height() >= 2:
                rect = requested
        return self.widget.grab(rect).toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)

    def start(self, path, capture_rect=None):
        with self._lock:
            if self._item is not None:
                return {"ok": False, "error": "GUI视频正在录制或保存"}
        if bool(getattr(self.bridge, "_debug_mode", False)):
            return {"ok": False, "error": "请先切回操作模式再开始GUI录制"}
        executable = self._ffmpeg_executable()
        if not executable:
            return {"ok": False, "error": "缺少 imageio-ffmpeg/FFmpeg，无法保存GUI视频"}
        image = self._grab_image(capture_rect)
        width, height = image.width(), image.height()
        scale = min(1.0, self.MAX_WIDTH/max(1, width), self.MAX_HEIGHT/max(1, height))
        width = max(2, int(width*scale)//2*2)
        height = max(2, int(height*scale)//2*2)
        partial = path + ".part.mp4"
        command = [
            executable, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s:v", f"{width}x{height}",
            "-r", str(self.FPS), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", partial,
        ]
        options = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL,
                   "stderr": subprocess.DEVNULL}
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(command, **options)
        except Exception as exc:
            return {"ok": False, "error": "无法启动GUI视频编码器：" + str(exc)}
        item = {
            "path": path, "partial": partial, "process": process,
            "width": width, "height": height,
            "capture_rect": capture_rect, "frames": 0, "stopping": False,
            "error": "", "thread": None, "last_payload": None,
            "stop_event": threading.Event(),
        }
        thread = threading.Thread(target=self._writer, args=(item,), daemon=True)
        item["thread"] = thread
        with self._lock:
            self._item = item
        thread.start()
        self.timer.start(round(1000/self.FPS))
        self._capture()
        label = "上方GUI" if self.kind == "gui" else "控制响应"
        self.bridge._recording_emit(self.kind, "recording", label + "视频录制中", path)
        return {"ok": True, "path": path, "width": width, "height": height,
                "fps": self.FPS}

    def _capture(self):
        with self._lock:
            item = self._item
        if item is None or item["stopping"]:
            return
        # The requested GUI movie is an operation-mode record.  While the
        # operator temporarily opens Debug, omit those frames and resume on
        # return.  FFmpeg receives only operation frames, so Debug never
        # changes the recorded crop or appears in the saved video.
        if bool(getattr(self.bridge, "_debug_mode", False)):
            return
        try:
            image = self._grab_image(item.get("capture_rect"))
            if image.width() != item["width"] or image.height() != item["height"]:
                image = image.scaled(item["width"], item["height"],
                                     QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation)
            pointer = image.bits()
            pointer.setsize(image.byteCount())
            payload = bytes(pointer)
            expected = item["width"]*item["height"]*4
            if image.bytesPerLine() != item["width"]*4 or len(payload) != expected:
                payload = b"".join(
                    payload[row*image.bytesPerLine():row*image.bytesPerLine()+item["width"]*4]
                    for row in range(item["height"]))
            # The encoder thread owns the frame clock and keeps writing this last
            # operation-view image even if Qt is blocked by LLM work or Debug is
            # temporarily visible.  Updating one immutable bytes reference is
            # atomic under CPython and does not block the GUI thread.
            item["last_payload"] = payload
        except Exception as exc:
            item["error"] = "画面抓取失败：" + str(exc)
            self.stop()

    def _writer(self, item):
        try:
            period_s = 1.0 / float(self.FPS)
            next_frame_at = time.monotonic()
            while not item["stop_event"].is_set():
                payload = item.get("last_payload")
                if payload is None:
                    if item["stop_event"].wait(0.01):
                        break
                    next_frame_at = time.monotonic()
                    continue
                delay_s = next_frame_at - time.monotonic()
                if delay_s > 0.0 and item["stop_event"].wait(delay_s):
                    break
                item["process"].stdin.write(payload)
                item["frames"] += 1
                next_frame_at += period_s
                # Never generate a catch-up burst after an unusually slow encode;
                # resume from the present while keeping one frame per real period.
                now = time.monotonic()
                if next_frame_at < now - period_s:
                    next_frame_at = now
            item["process"].stdin.close()
            return_code = item["process"].wait(timeout=60)
            if return_code != 0:
                raise RuntimeError(f"FFmpeg退出码 {return_code}")
            if item["frames"] <= 0 or not os.path.isfile(item["partial"]):
                raise RuntimeError("没有捕获到GUI画面")
            os.replace(item["partial"], item["path"])
            label = "上方GUI" if self.kind == "gui" else "控制响应"
            self.bridge._recording_emit(
                self.kind, "saved", label + "视频已保存", item["path"])
        except Exception as exc:
            item["error"] = item["error"] or str(exc)
            try:
                if item["process"].poll() is None:
                    item["process"].kill()
            except Exception:
                pass
            try:
                os.remove(item["partial"])
            except FileNotFoundError:
                pass
            self.bridge._recording_emit(
                self.kind, "error", ("上方GUI" if self.kind == "gui" else "控制响应") +
                "视频保存失败：" + item["error"], item["path"])
        finally:
            with self._lock:
                if self._item is item:
                    self._item = None

    def stop(self):
        self.timer.stop()
        with self._lock:
            item = self._item
            if item is None:
                return {"ok": False, "error": "该区域视频未在录制"}
            if item["stopping"]:
                return {"ok": True, "saving": True, "path": item["path"]}
            item["stopping"] = True
            item["stop_event"].set()
        return {"ok": True, "saving": True, "path": item["path"]}

    def shutdown(self):
        with self._lock:
            item = self._item
        if item is not None:
            self.stop()
            item["thread"].join(timeout=10.0)


class NativePopupWebView(QWebEngineView):
    """Main web view that turns JavaScript ``window.open`` into an OS window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._popup_views = []

    def createWindow(self, window_type):
        popup = QWebEngineView()
        popup.setWindowTitle("Debug Console")
        popup.resize(560, 860)
        popup.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        settings = popup.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.JavascriptCanOpenWindows, True)
        popup.page().windowCloseRequested.connect(popup.close)
        popup.page().geometryChangeRequested.connect(popup.setGeometry)
        self._popup_views.append(popup)

        def _forget_popup(*_args):
            if popup in self._popup_views:
                self._popup_views.remove(popup)

        popup.destroyed.connect(_forget_popup)
        popup.show()
        return popup

    def closeEvent(self, event):
        for popup in list(self._popup_views):
            popup.close()
        self._popup_views.clear()
        super().closeEvent(event)


class DashWindow(QtWidgets.QWidget):
    """「控制响应」独立窗口：按 GUI 的 x/y/φ 坐标显示三轴遥测。"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("控制响应")
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        lay = QtWidgets.QVBoxLayout(self)
        self.stage = QtWidgets.QLabel("当前阶段：待机")
        self.stage.setAlignment(QtCore.Qt.AlignCenter)
        self.stage.setStyleSheet(
            "QLabel {font-size: 17px; font-weight: 700; color: #174a7e; "
            "background: #eaf4ff; border: 1px solid #9fc5ea; border-radius: 6px; padding: 7px;}")
        lay.addWidget(self.stage)
        self.forward_status = QtWidgets.QLabel("")
        self.forward_status.setAlignment(QtCore.Qt.AlignCenter)
        self.forward_status.setVisible(False)
        lay.addWidget(self.forward_status)
        gl = pg.GraphicsLayoutWidget(); lay.addWidget(gl)
        self.c_r, self.c_rc, self.c_q = [], [], []
        plots = []
        axis_names = ["x (s_x)", "y (s_y)", "φ (ψ)"]
        for i, name in enumerate(axis_names):
            p = gl.addPlot(row=i, col=0,
                           title=f"{name}（黑=原始示教，蓝虚线=控制参考，红=actual）")
            p.showGrid(x=True, y=True, alpha=0.3); p.setLabel("left", "deg")
            self.c_r.append(p.plot(pen=pg.mkPen("k", width=1.4)))
            self.c_rc.append(p.plot(pen=pg.mkPen("b", width=1.2, style=QtCore.Qt.DashLine)))
            self.c_q.append(p.plot(pen=pg.mkPen("r", width=2)))
            plots.append(p)
        plots[2].setLabel("bottom", "t (s)")
        plots[1].setXLink(plots[0]); plots[2].setXLink(plots[0])

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(250)

    def _refresh(self):
        import mpc_station as st
        d = list(st.TELEM)
        if not d: return
        self.stage.setText("当前阶段：" + str(d[-1].get("stage", "轨迹跟踪")))
        latest = d[-1]
        if "forward_status" in latest:
            status = str(latest.get("forward_status", "等待下一拍"))
            error = latest.get("forward_error_norm_deg")
            generation = int(latest.get("forward_generation", 0))
            error_text = ("原始正/逆模型只读" if status == "冻结" else
                          "等待第一组 x̂(t+1)" if error is None else
                          f"活动模型误差 {float(error):.2f}°")
            color = "#9b1c1c" if status == "失效" else "#9a6700" if status == "警告" else "#176b3a"
            background = "#fff0f0" if status == "失效" else "#fff8df" if status == "警告" else "#edf9f1"
            mode_text = str(latest.get("forward_mode", "NEW MPC+FEL"))
            self.forward_status.setText(
                f"{mode_text}｜顺模型 generation {generation}｜{error_text}｜{status}")
            self.forward_status.setStyleSheet(
                f"QLabel {{font-size: 13px; font-weight: 650; color: {color}; "
                f"background: {background}; border: 1px solid {color}; border-radius: 5px; padding: 5px;}}")
            self.forward_status.setVisible(True)
        else:
            self.forward_status.setVisible(False)
        t = [r["t"] for r in d]
        for i in range(3):
            # 新数据直接携带 GUI 坐标；兼容窗口启动前留下的旧格式遥测。
            rv = [(r["r_gui"] if "r_gui" in r else
                   st.st3_from_q_model(np.radians(r["r"])))[i] for r in d]
            rcv = [(r.get("r_control_gui", r.get("r_gui")) if "r_gui" in r else
                    st.st3_from_q_model(np.radians(r["r"])))[i] for r in d]
            qv = [(r["q_gui"] if "q_gui" in r else
                   st.st3_from_q_model(np.radians(r["q"])))[i] for r in d]
            self.c_r[i].setData(t, rv)
            self.c_rc[i].setData(t, rcv)
            self.c_q[i].setData(t, qv)


def _start_mpc_brain():
    """把 mpc_station 的 bridge（MPC 计算 + 本机 HTTP）作为线程内嵌到本进程。
    页面里的「MPC·控制站」照旧访问 127.0.0.1:8765，只是不再需要单独开终端。"""
    import threading, types
    import mpc_station as st
    if not os.path.exists(st.PARAMS_JSON):
        print("# 没有 params.json —— MPC 不可用（先跑 ident + fit）；示教/开环/手动不受影响")
        return
    ns = types.SimpleNamespace(mount="ry-", N=20, http=8765)
    def run():
        try:
            st.mode_bridge(ns)
        except OSError:
            print("# 端口 8765 被占（bridge 已在别处跑？）——沿用外部的")
        except Exception as e:
            print(f"# MPC 线程异常：{e}")
    threading.Thread(target=run, daemon=True).start()


def main():
    app = QtWidgets.QApplication(sys.argv)
    _start_mpc_brain()
    view = NativePopupWebView()
    view.setWindowTitle("Musculoskeletal Robotic Shoulder Bioreactor")
    # QWebEngine 默认禁止 file:// 页面访问网络地址——不开这个，页面 fetch
    # 127.0.0.1:8765（内嵌 MPC 线程）会被引擎静默掐死，MPC 模式永远“连不上”。
    s = view.settings()
    s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
    s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
    s.setAttribute(QWebEngineSettings.JavascriptCanOpenWindows, True)

    bridge = SerialBridge()
    gui_recorder = NativeGuiRecorder(view, bridge, "gui")
    bridge._native_recorders = {"gui": gui_recorder}
    channel = QWebChannel()
    channel.registerObject("pyserial", bridge)
    view.page().setWebChannel(channel)

    # 相机只允许本地 GUI 页在用户点开“数据可视化”后使用；其他来源一律拒绝。
    def _feature_permission(origin, feature):
        video_features = {
            QWebEnginePage.MediaVideoCapture,
            QWebEnginePage.MediaAudioVideoCapture,
            QWebEnginePage.DesktopVideoCapture,
            QWebEnginePage.DesktopAudioVideoCapture,
        }
        local_gui = origin.isLocalFile()
        policy = (QWebEnginePage.PermissionGrantedByUser
                  if local_gui and feature in video_features
                  else QWebEnginePage.PermissionDeniedByUser)
        view.page().setFeaturePermission(origin, feature, policy)

    view.page().featurePermissionRequested.connect(_feature_permission)

    # 本地 Qt 专用：MPC 使用页面原生 100 ms 定时器（10 Hz）。启动请求额外附带当前
    # IMU 安装映射，确保示教显示与控制站使用同一个坐标系。
    def _install_qt_overrides(ok):
        if not ok:
            return
        view.page().runJavaScript(r"""
        (() => {
          // 正式操作界面中英双语；debug 始终使用原中文界面。
          const qtOpLanguageSelect = document.getElementById('op-language');
          const qtOpTranslations = {
            zh: {
              language:'语言', operationMode:'操作模式', showFrame:'显示机架', hideFrame:'隐藏机架',
              dataVisualization:'数据可视化', closeVisualization:'关闭可视化', save:'保存',
              response:'响应', camera:'相机', force:'Force', path:'路径', currentSavePath:'当前保存路径',
              viewSelector:'视图选择器', viewSelectorHelp:'点击框面切换视角 · 点击角点进入等轴测 · Alt 选择背面 · Esc 退出',
              frontView:'前视', rearView:'后视', leftView:'左视', rightView:'右视', topView:'上视', bottomView:'下视', isometric:'等轴测',
              controlResponse:'控制响应', currentStage:'当前阶段：', standby:'待机', originalTeaching:'原始示教',
              controlReference:'控制参考', actual:'实际', poseResponse:'姿态 X/Y/φ', forceResponse:'力 FXYZ', cleanResponse:'清空', cleanResponseHelp:'清空当前显示曲线',
              measuredForce:'真值', predictedForce:'预测值', waitingForceResponse:'等待 Nano25 真值和预测值…',
              openCamera:'打开相机', selectCamera:'选择摄像头',
              rotate90:'每次转 90°', mirror:'左右镜像', closeCamera:'关闭相机',
              closeCameraHelp:'停止并释放摄像头，不关闭控制响应区域', jointPose:'关节姿态',
              swingX:'摆动 x', swingY:'摆动 y', elevation:'抬臂角', planeAngle:'平面角',
              spinAngle:'自旋角', axialAngle:'轴向角', locked:'锁定', unlocked:'放开',
              devicePose:'设备连接与姿态', connected:'已连接', disconnected:'未连接', noData:'无数据',
              connect:'连接', disconnect:'断开', zero:'归零', poseDrive:'姿态联动', poseDriveOff:'姿态联动：关',
              defaultZero:'机械归零（默认）', defaultPoseHelp:'启动时的默认姿态为机械零位 (0,0,0)',
              driveDefault:'姿态联动默认开启；连接后点击“归零”。',
              driveEnabled:'姿态联动已开启；归零后当前姿态立即显示为 (0,0,0)。',
              driveDisabled:'姿态联动已关闭。', zeroed:'已归零：当前传感器姿态 = 机械零位 (0,0,0)，姿态联动已开启。',
              muscleParameters:'肌肉参数', muscleParametersHelp:'20路编码器测得的肌肉长度与六路实时压力',
              nano25Title:'Nano25', forcePredictionHide:'隐藏预测力', forcePredictionShow:'显示预测力', nano25Start:'启动 Nano25', nano25Stop:'停止 Nano25', nano25Starting:'启动中…',
              nano25Zeroing:'归零中…', nano25Stopping:'停止中…', nano25Stopped:'未启动', nano25Error:'采集故障',
              nano25Help:'可选传感器；未启动也可执行轨迹。启动前保持无载荷，点击后自动完成约5秒归零。力按GUI的Y、Z、X顺序显示，并已统一反向。',
              nano25ControlRequired:'控制中持续采集', forcePredictionInline:'预测',
              forcePredictionTitle:'模型预测力（20 Hz）', forcePredictionReset:'重置预测零点',
              forcePredictionMode:'预测方式', forcePredictionFusion:'Fusion单网络',
              forcePredictionSeparate:'三轴分开模型',
              forcePredictionWaiting:'等待六路压力和长度数据', forcePredictionLoading:'正在加载预测模型…',
              forcePredictionReady:'仅用于监测，不参与控制', forcePredictionError:'模型预测不可用',
              forcePredictionWarmup:'历史窗 {n}/{window}', forcePredictionOutOfRange:'输入超出训练数据范围，请谨慎使用',
              modes:'模式', ventAll:'全机放气', automatic:'自动', teaching:'示教', test:'测试',
              mode:'模式', loop:'循环', runsUnit:'圈', minutesUnit:'mins', durationHelp:'留空表示无限运行；runs 按完整圈数停止，mins 在当前完整循环结束后停止', infiniteLoop:'无限运行', startAuto:'开始自动运行', stopAuto:'停止自动运行',
              addPoint:'＋ 新增当前点', loadDefault:'载入', clearAll:'清空全部', clearMemory:'清除记忆', recorded:'已记录', points:'个点', noPoints:'还没有记录点',
              saveProcess:'保存10轮过程', loadReplay:'载入过程回放', exitReplay:'退出回放',
              saveProcessHelp:'保存本次10个有效完整轮次的控制器、参考轨迹、实测响应和设计结果',
              loadReplayHelp:'载入已保存的10轮过程；执行时按原轮次控制器回放，不再调用LLM',
              replayLoaded:'已载入 {n} 轮回放', replayExited:'已退出过程回放',
              loadDefaultHelp:'载入所选的 Default 1、Default 2 或 Default 3，并替换当前示教列表',
              clearMemoryHelp:'清除LLM补偿器参数、待审核候选和设计记录；不删除示教点，不修改MPC基础模型',
              controller:'控制器', loopExecution:'循环执行', startTeaching:'开始执行示教轨迹', stopExecution:'停止执行',
              llmSettings:'LLM配置', llmConfiguration:'LLM 配置', llmProvider:'提供商',
              llmIntervention:'介入本轮推理', llmInterventionPlaceholder:'例如：φ上升过快，降低上升段正向补偿并提前卸载',
              llmInterventionSend:'提交意见', llmInterventionIdle:'仅在LLM推理时可提交',
              llmInterventionReady:'LLM正在推理，可追加本轮设计要求',
              llmInterventionQueued:'已提交 {n} 条；当前推理结束后同会话复审',
              llmFeedbackAppliedLabel:'已送入 LLM', llmFeedbackQueuedLabel:'等待送入 LLM',
              llmFeedbackInitial:'初始 prompt', llmFeedbackReview:'同会话复审',
              llmPromptCountdown:'{n}秒后自动开始；点击输入框可暂停',
              llmPromptHeld:'已暂停自动开始；输入意见后提交，留空提交则立即开始',
              llmPromptSubmit:'提交并开始', llmPromptStartNow:'立即开始', llmReasoningAdd:'追加意见',
              gptProvider:'GPT', claudeProvider:'Claude', llmModel:'模型', defaultModel:'默认（推荐）',
              compensatorDesign:'补偿器设计', designBase:'Base（矩阵 PID）', designPlus:'Plus（压力感知分层）', designPro:'Pro（自由设计）',
              reasoningEffort:'推理强度', analysisTimeout:'分析超时（秒）', signOut:'退出登录',
              fastMode:'Fast Mode（更多额度，更快）', fastModeHelp:'仅用于GPT/Codex；保持当前推理强度并使用快速服务，Claude忽略此项',
              automaticDesign:'每轮结束后自动生成下一版',
              reuseMse:'沿用阈值（MSE deg²）', reuseMseHelp:'有效完整轮次的MSE不超过该值时沿用当前已验证补偿器，不再调用LLM改版；0表示关闭',
              authMethod:'认证方式', chatgptAccount:'ChatGPT账号', apiKeyAccess:'API Key',
              authStatus:'登录状态', refreshStatus:'刷新状态', signInChatGPT:'登录ChatGPT',
              signInClaude:'登录Claude', useApiKey:'使用API Key', claudeAccount:'Claude账号',
              installClaude:'安装Claude Code', installingClaude:'正在安装Claude Code…',
              claudeInstallFailed:'Claude Code安装失败', claudeInstalled:'Claude Code已安装，请登录账号',
              claudeNotInstalled:'尚未安装Claude Code', codexNotInstalled:'尚未安装Codex CLI',
              apiKey:'API Key', apiKeyPlaceholder:'只用于本次登录，不写入项目',
              authChecking:'正在检查登录状态…', authStarting:'正在启动登录…',
              signedInChatGPT:'已通过ChatGPT账号登录', signedInClaude:'已通过Claude账号登录', signedInApi:'已通过API Key登录',
              signedOut:'尚未登录', authFailed:'登录失败', apiKeyRequired:'请输入有效的API Key',
              saveSettings:'保存配置', cancel:'取消', llmLoading:'正在读取LLM配置…',
              llmReady:'LLM配置已加载', llmMissing:'所需认证工具不可用',
              llmLoadFailed:'LLM配置读取失败', llmSaved:'LLM配置已保存', llmSaveFailed:'LLM配置保存失败',
              quickStart:'开始快速综合检测', quickStop:'停止检测', pending:'待检测', testing:'检测中',
              passed:'通过', retest:'需复测', fault:'故障', notTested:'尚未检测',
              pythonEnvironment:'软件环境', pythonDetail:'Python版本、运行组件与项目文件',
              pressureSensor:'压力传感器', pressureDetail:'零漂、响应和六路一致性',
              muscleContraction:'肌肉收缩', muscleDetail:'20路在线与六组长度变化',
              valveLockMapping:'阀/锁/映射', mappingDetail:'六路锁定和拮抗通道误锁',
              imuDetail:'六方向运动响应与方向检查', maxSpeed:'最大速度', speedDetail:'从同批扫描快速估计六方向P95',
              selectedMode:'已选择模式', singleRun:'单次', quickRunning:'快速综合检测正在运行',
              runningTeaching:'正在执行示教轨迹', runningMode:'正在运行：模式',
              runningQueue:'顺序运行', queueComplete:'全部选择模式已按顺序完成',
              queueTransition:'模式 {from} 完成，六路清零；1秒后开始模式 {to}',
              finiteQueueRequired:'顺序执行时，最后一个之前开启 Loop 的模式必须填写有限的 runs 或 mins。',
              selectModeFirst:'请至少点选一个 Mode；深色表示已加入执行队列。',
              testStandby:'综合检测待机｜软件环境检查不驱动真机',
              recordedPoint:'已记录示教点', deletePoint:'删除第 {n} 点', pointDeviation:'该点指令与姿态偏差 {v}°',
              waitSampling:'等待采样', testPreparing:'检测准备中…', selectTaskFirst:'请先停止当前自动任务或示教轨迹。',
              connectFirst:'请先连接 IMU/串口。', unsupportedTest:'当前本地宿主不支持快速综合检测。',
              serialResetNotice:'串口没有返回数据；请按板上 RST 重新启动机器。',
              noHostEnv:'当前本地宿主不支持环境检测', envPass:'环境检查通过', envParseFail:'环境检测结果无法解析',
              autoInstall:'正在准备自动安装缺失组件…', installStartFail:'无法启动自动安装',
              recordingDuringPath:'任务记录期间不能更改保存路径', pathFailed:'保存路径选择失败', pathUpdated:'保存路径已更新',
              pathNotSelected:'未选择', pathRequired:'已勾选记录内容，请先点击“路径”并选择保存文件夹',
              recording:'记录中：', recordingCamera:'相机', recordingResponse:'响应',
              savingAllRecords:'正在保存当前全部记录…', nothingToSave:'当前没有正在记录的内容',
              responseRecording:'控制响应记录中', forceRecording:'Force数据记录中', forceStartFailed:'无法开始Force记录',
              savingForce:'正在保存Force Excel（{n}行）', forceSaveFailed:'Force数据保存失败',
              guiStarting:'上方GUI视频正在启动…', guiRecording:'上方GUI视频录制中', cameraRecording:'相机视频录制中',
              desktopVideoOnly:'视频保存仅支持桌面GUI', regionMissing:'找不到要录制的画面区域',
              regionStartFailed:'无法启动区域录制', videoUnsupported:'当前环境不支持视频编码',
              displayRecordingUnsupported:'当前环境不支持GUI窗口录制', cameraNotOpen:'相机尚未打开',
              createVideoFailed:'无法创建视频文件', videoWriteFailed:'视频数据写入失败',
              videoEncodingFailed:'视频编码失败', videoSaveFailed:'视频保存失败', guiVideoSaved:'GUI视频已保存：',
              cameraVideoSaved:'相机视频已保存：', guiVideoLabel:'GUI视频：', cameraVideoLabel:'相机视频：',
              regionStopFailed:'区域视频停止失败', savingGuiVideo:'正在保存上方GUI视频…', finishingVideo:'正在完成视频文件…',
              excelDesktopOnly:'Excel保存仅支持桌面GUI', responseStartFailed:'无法开始响应记录',
              savingResponse:'正在保存响应Excel（{n}行）', responseSaveFailed:'响应保存失败', excelCacheFailed:'有数据帧未写入Excel缓存',
              waitingResponse:'等待控制响应数据…', responseProvided:'控制响应数据由 qt_gui.py 提供',
              responseInvalid:'数据无效', responseReadFail:'控制响应读取失败'
            },
            en: {
              language:'Language', operationMode:'Operation Mode', showFrame:'Show Frame', hideFrame:'Hide Frame',
              dataVisualization:'Data Visualization', closeVisualization:'Close Visualization', save:'Save',
              response:'Response', camera:'Camera', force:'Force', path:'Path', currentSavePath:'Current save path',
              viewSelector:'View Selector', viewSelectorHelp:'Click a face to change view · Click a corner for isometric · Alt for back face · Esc to exit',
              frontView:'Front', rearView:'Rear', leftView:'Left', rightView:'Right', topView:'Top', bottomView:'Bottom', isometric:'Isometric',
              controlResponse:'Control Response', currentStage:'Current stage: ', standby:'Standby', originalTeaching:'Recorded Path',
              controlReference:'Control Reference', actual:'Actual', poseResponse:'Pose X/Y/φ', forceResponse:'Force FXYZ', cleanResponse:'Clean', cleanResponseHelp:'Clear the currently displayed curves',
              measuredForce:'Measured', predictedForce:'Predicted', waitingForceResponse:'Waiting for Nano25 measured and predicted force…',
              openCamera:'Open Camera', selectCamera:'Select camera',
              rotate90:'Rotate 90°', mirror:'Mirror horizontally', closeCamera:'Close Camera',
              closeCameraHelp:'Stop and release the camera without closing the response panel', jointPose:'Joint Pose',
              swingX:'Swing x', swingY:'Swing y', elevation:'Elevation', planeAngle:'Plane of Elevation',
              spinAngle:'Spin Angle', axialAngle:'Axial Rotation', locked:'Locked', unlocked:'Unlocked',
              devicePose:'Device Connection and Pose', connected:'Connected', disconnected:'Disconnected', noData:'No Data',
              connect:'Connect', disconnect:'Disconnect', zero:'Zero', poseDrive:'Pose Sync', poseDriveOff:'Pose Sync: Off',
              defaultZero:'Mechanical Zero (Default)', defaultPoseHelp:'The startup pose is the mechanical zero (0,0,0)',
              driveDefault:'Pose sync is enabled by default. Click “Zero” after connecting.',
              driveEnabled:'Pose sync is enabled. After zeroing, the current pose is displayed as (0,0,0).',
              driveDisabled:'Pose sync is disabled.', zeroed:'Zeroed: the current sensor pose is the mechanical zero (0,0,0), and pose sync is enabled.',
              muscleParameters:'Muscle Parameters', muscleParametersHelp:'Muscle lengths from 20 encoders and six real-time pressures',
              nano25Title:'Nano25', forcePredictionHide:'Hide Pred.', forcePredictionShow:'Show Pred.', nano25Start:'Start Nano25', nano25Stop:'Stop Nano25', nano25Starting:'Starting…',
              nano25Zeroing:'Zeroing…', nano25Stopping:'Stopping…', nano25Stopped:'Not started', nano25Error:'Acquisition fault',
              nano25Help:'Optional sensor; trajectories can run without it. Keep it unloaded before starting. After an approximately 5-second zero, force is displayed in GUI Y, Z, X order with all signs reversed.',
              nano25ControlRequired:'Continuous acquisition during control', forcePredictionInline:'Pred.',
              forcePredictionTitle:'Model-Predicted Force (20 Hz)', forcePredictionReset:'Reset Prediction Zero',
              forcePredictionMode:'Prediction', forcePredictionFusion:'Fusion network',
              forcePredictionSeparate:'Separate axis models',
              forcePredictionWaiting:'Waiting for six-channel pressure and length data', forcePredictionLoading:'Loading prediction models…',
              forcePredictionReady:'Monitoring only; not connected to control', forcePredictionError:'Force prediction unavailable',
              forcePredictionWarmup:'History window {n}/{window}', forcePredictionOutOfRange:'Input is outside the training range; use with caution',
              modes:'Modes', ventAll:'Vent All', automatic:'Automatic', teaching:'Teaching', test:'Test',
              mode:'Mode', loop:'Loop', runsUnit:'runs', minutesUnit:'mins', durationHelp:'Leave blank to run indefinitely; runs stops by completed cycles, mins stops after the current cycle', infiniteLoop:'Infinite Run', startAuto:'Start Automatic Run', stopAuto:'Stop Automatic Run',
              addPoint:'＋ Add Current Point', loadDefault:'Load', clearAll:'Clear All', clearMemory:'Clear Memory', recorded:'Recorded', points:'points', noPoints:'No recorded points',
              saveProcess:'Save 10-Round Process', loadReplay:'Load Process Replay', exitReplay:'Exit Replay',
              saveProcessHelp:'Save the controllers, reference, measured response, and design result for the current 10 valid complete rounds',
              loadReplayHelp:'Load a saved process and replay its round controllers without calling the LLM again',
              replayLoaded:'Loaded {n}-round replay', replayExited:'Process replay exited',
              loadDefaultHelp:'Load the selected Default 1, Default 2, or Default 3 and replace the current teaching list',
              clearMemoryHelp:'Clear LLM compensator parameters, pending candidates, and design records; teaching points and the base MPC model are unchanged',
              controller:'Controller', loopExecution:'Loop Execution', startTeaching:'Execute Taught Trajectory', stopExecution:'Stop Execution',
              llmSettings:'LLM Settings', llmConfiguration:'LLM Configuration', llmProvider:'Provider',
              llmIntervention:'Intervene in This Design', llmInterventionPlaceholder:'Example: phi rises too fast; reduce rise assistance and unload earlier',
              llmInterventionSend:'Send Feedback', llmInterventionIdle:'Available only while the LLM is reasoning',
              llmInterventionReady:'The LLM is reasoning; you can add a requirement for this design',
              llmInterventionQueued:'{n} message(s) queued; the same session will review them after this turn',
              llmFeedbackAppliedLabel:'Sent to LLM', llmFeedbackQueuedLabel:'Queued for LLM',
              llmFeedbackInitial:'initial prompt', llmFeedbackReview:'same-session review',
              llmPromptCountdown:'Starts automatically in {n}s; click the input to pause',
              llmPromptHeld:'Automatic start paused; submit feedback, or submit empty to start now',
              llmPromptSubmit:'Submit and Start', llmPromptStartNow:'Start Now', llmReasoningAdd:'Add Feedback',
              gptProvider:'GPT', claudeProvider:'Claude', llmModel:'Model', defaultModel:'Default (Recommended)',
              compensatorDesign:'Compensator Design', designBase:'Base (Matrix PID)', designPlus:'Plus (Pressure-Aware Hierarchy)', designPro:'Pro (Open Design)',
              reasoningEffort:'Reasoning effort', analysisTimeout:'Analysis timeout (s)', signOut:'Sign Out',
              fastMode:'Fast Mode (more credits, faster)', fastModeHelp:'GPT/Codex only; keeps the selected reasoning effort and uses faster processing. Ignored by Claude.',
              automaticDesign:'Generate the next version after each run',
              reuseMse:'Reuse threshold (MSE deg²)', reuseMseHelp:'Reuse the current validated compensator without another LLM redesign when a valid complete run is at or below this MSE; set 0 to disable',
              authMethod:'Authentication', chatgptAccount:'ChatGPT Account', apiKeyAccess:'API Key',
              authStatus:'Sign-in Status', refreshStatus:'Refresh Status', signInChatGPT:'Sign in with ChatGPT',
              signInClaude:'Sign in with Claude', useApiKey:'Use API Key', claudeAccount:'Claude Account',
              installClaude:'Install Claude Code', installingClaude:'Installing Claude Code…',
              claudeInstallFailed:'Claude Code installation failed', claudeInstalled:'Claude Code installed; sign in to your account',
              claudeNotInstalled:'Claude Code is not installed', codexNotInstalled:'Codex CLI is not installed',
              apiKey:'API Key', apiKeyPlaceholder:'Used only for sign-in; never written to this project',
              authChecking:'Checking sign-in status…', authStarting:'Starting sign-in…',
              signedInChatGPT:'Signed in with a ChatGPT account', signedInClaude:'Signed in with a Claude account', signedInApi:'Signed in with an API key',
              signedOut:'Not signed in', authFailed:'Sign-in failed', apiKeyRequired:'Enter a valid API key',
              saveSettings:'Save Settings', cancel:'Cancel', llmLoading:'Loading LLM settings…',
              llmReady:'LLM configuration loaded', llmMissing:'Required authentication tool is unavailable',
              llmLoadFailed:'Failed to load LLM settings', llmSaved:'LLM settings saved', llmSaveFailed:'Failed to save LLM settings',
              quickStart:'Start Quick System Check', quickStop:'Stop Check', pending:'Pending', testing:'Testing',
              passed:'Passed', retest:'Retest', fault:'Fault', notTested:'Not Tested',
              pythonEnvironment:'Software Environment', pythonDetail:'Python version, runtime components, and project files',
              pressureSensor:'Pressure Sensors', pressureDetail:'Zero drift, response, and six-channel consistency',
              muscleContraction:'Muscle Contraction', muscleDetail:'20-channel availability and six-group length changes',
              valveLockMapping:'Valve / Lock / Mapping', mappingDetail:'Six locks and unintended antagonist locking',
              imuDetail:'Six-direction response and direction verification', maxSpeed:'Maximum Speed', speedDetail:'Estimate six-direction P95 from the same scan',
              selectedMode:'Selected Mode', singleRun:'Single Run', quickRunning:'Quick system check is running',
              runningTeaching:'Executing taught trajectory', runningMode:'Running: Mode',
              runningQueue:'Sequential Run', queueComplete:'All selected modes completed in order',
              queueTransition:'Mode {from} complete; outputs are zero. Mode {to} starts in 1 second',
              finiteQueueRequired:'For sequential execution, every looped mode except the last must have finite runs or mins.',
              selectModeFirst:'Select at least one Mode; dark buttons are included in the execution queue.',
              testStandby:'System check standby | Python environment check does not drive hardware',
              recordedPoint:'Recorded teaching point', deletePoint:'Delete point {n}', pointDeviation:'Command-to-pose deviation at this point: {v}°',
              waitSampling:'Waiting for samples', testPreparing:'Preparing system check…', selectTaskFirst:'Stop the current automatic task or taught trajectory first.',
              connectFirst:'Connect the IMU/serial device first.', unsupportedTest:'Quick system check is not supported by the current local host.',
              serialResetNotice:'No serial data were received. Press the board RST button to restart the machine.',
              noHostEnv:'Environment checking is not supported by the current local host', envPass:'Environment check passed', envParseFail:'Unable to parse the environment-check result',
              autoInstall:'Preparing to install missing components automatically…', installStartFail:'Unable to start automatic installation',
              recordingDuringPath:'The save path cannot be changed while recording', pathFailed:'Failed to select save path', pathUpdated:'Save path updated',
              pathNotSelected:'Not selected', pathRequired:'Recording is selected. Choose a save folder with Path before starting.',
              recording:'Recording: ', recordingCamera:'Camera', recordingResponse:'Response',
              savingAllRecords:'Saving all current recordings…', nothingToSave:'Nothing is currently being recorded',
              responseRecording:'Control response recording', forceRecording:'Recording force data', forceStartFailed:'Failed to start force recording',
              savingForce:'Saving force Excel ({n} rows)', forceSaveFailed:'Failed to save force data',
              guiStarting:'Starting upper-GUI video…', guiRecording:'Recording upper-GUI video', cameraRecording:'Recording camera video',
              desktopVideoOnly:'Video saving is available only in the desktop GUI', regionMissing:'The GUI region to record was not found',
              regionStartFailed:'Failed to start region recording', videoUnsupported:'Video encoding is not supported in this environment',
              displayRecordingUnsupported:'GUI window recording is not supported in this environment', cameraNotOpen:'The camera is not open',
              createVideoFailed:'Failed to create the video file', videoWriteFailed:'Failed to write video data',
              videoEncodingFailed:'Video encoding failed', videoSaveFailed:'Video saving failed', guiVideoSaved:'GUI video saved: ',
              cameraVideoSaved:'Camera video saved: ', guiVideoLabel:'GUI video: ', cameraVideoLabel:'Camera video: ',
              regionStopFailed:'Failed to stop region recording', savingGuiVideo:'Saving upper-GUI video…', finishingVideo:'Finalizing video file…',
              excelDesktopOnly:'Excel saving is available only in the desktop GUI', responseStartFailed:'Failed to start response recording',
              savingResponse:'Saving response Excel ({n} rows)', responseSaveFailed:'Failed to save response data', excelCacheFailed:'A data frame was not written to the Excel cache',
              waitingResponse:'Waiting for control-response data…', responseProvided:'Control-response data are provided by qt_gui.py',
              responseInvalid:'Invalid data', responseReadFail:'Failed to read control response'
            }
          };
          const qtOpEffectiveLanguage = () =>
            (typeof DEBUG !== 'undefined' && DEBUG) ? 'zh' :
            (qtOpLanguageSelect && qtOpLanguageSelect.value === 'en' ? 'en' : 'zh');
          const qtOpText = (key, vars={}) => {
            const lang = qtOpEffectiveLanguage();
            let value = (qtOpTranslations[lang] && qtOpTranslations[lang][key]) ||
              (qtOpTranslations.zh[key] || key);
            Object.keys(vars).forEach(name => { value = value.split('{'+name+'}').join(String(vars[name])); });
            return value;
          };
          window.__qtOpText = qtOpText;
          const qtReadStoredLanguage = () => {
            try { return localStorage.getItem('tro-operation-language') === 'en' ? 'en' : 'zh'; }
            catch (_) { return 'zh'; }
          };
          window.__qtLocalizeOperationRuntime = value => {
            let text = String(value == null ? '' : value);
            if (qtOpEffectiveLanguage() !== 'en') return text;
            const exact = {
              '待机':'Standby', '准备开始':'Preparing', '轨迹跟踪':'Trajectory Tracking',
              '轨迹完成':'Trajectory Complete', '等待六组肌肉长度传感器':'Waiting for six muscle-length sensor groups'
            };
            if (exact[text]) return exact[text];
            return text
              .replace(/第(\d+)轮/g, 'Lap $1')
              .replace(/阶段(\d+)\/(\d+)/g, 'Stage $1/$2')
              .replace(/第(\d+)段/g, 'Segment $1')
              .replace(/回零段/g, 'Return-to-Zero Segment')
              .replace(/三轴MPC跟踪/g, '3-Axis MPC Tracking')
              .replace(/长度PI调整/g, 'Length PI Adjustment: ')
              .replace(/长度PI保持/g, 'Length PI Hold')
              .replace(/Y轴/g, 'Y Axis').replace(/φ轴/g, 'φ Axis').replace(/X轴/g, 'X Axis')
              .replace(/调整/g, 'Adjustment: ').replace(/保持/g, 'Hold')
              .replace(/隐藏终点/g, 'Hidden Endpoint').replace(/全机放气/g, 'Vent All')
              .replace(/等待到位/g, 'Waiting for Target');
          };
          const qtOpSetText = (selector, key) => {
            const element = document.querySelector(selector);
            if (element) element.textContent = qtOpText(key);
          };
          window.__qtApplyOperationLanguage = () => {
            const lang = qtOpEffectiveLanguage();
            document.body.dataset.operationLanguage = lang;
            if (qtOpLanguageSelect) qtOpLanguageSelect.value = qtReadStoredLanguage();
            qtOpSetText('#op-language-label','language');
            qtOpSetText('#mode-user','operationMode');
            qtOpSetText('#view-frame', (typeof showFrame !== 'undefined' && showFrame) ? 'hideFrame' : 'showFrame');
            qtOpSetText('#data-viz', (typeof dataVizOn !== 'undefined' && dataVizOn) ? 'closeVisualization' : 'dataVisualization');
            const dataViz = document.getElementById('data-viz');
            if (dataViz) dataViz.title = qtOpText('dataVisualization');
            qtOpSetText('#task-record-options .task-record-title','save');
            const recordSave = document.getElementById('qt-record-save');
            if (recordSave) recordSave.title = qtOpText('save') + ' GUI / ' +
              qtOpText('response') + ' / ' + qtOpText('camera') + ' / ' + qtOpText('force');
            const responseLabel = document.getElementById('qt-rec-response');
            if (responseLabel && responseLabel.closest('label'))
              responseLabel.closest('label').lastChild.textContent = qtOpText('response');
            const cameraLabel = document.getElementById('qt-rec-camera');
            if (cameraLabel && cameraLabel.closest('label'))
              cameraLabel.closest('label').lastChild.textContent = qtOpText('camera');
            const forceLabel = document.getElementById('qt-rec-force');
            if (forceLabel && forceLabel.closest('label'))
              forceLabel.closest('label').lastChild.textContent = qtOpText('force');
            qtOpSetText('#qt-record-dir','path');
            const recordPath = document.getElementById('qt-record-path');
            if (recordPath && !recordPath.title) recordPath.title = qtOpText('currentSavePath');
            if (typeof qtRecordingDirectorySelected !== 'undefined' &&
                !qtRecordingDirectorySelected && typeof qtShowRecordingDirectory === 'function')
              qtShowRecordingDirectory('', false);
            const viewNote = document.getElementById('view-selector-note');
            if (viewNote) viewNote.innerHTML = '<b>'+qtOpText('viewSelector')+'</b>　'+qtOpText('viewSelectorHelp');
            qtOpSetText('#viz-response-title','controlResponse');
            qtOpSetText('#viz-teach-label','originalTeaching');
            qtOpSetText('#viz-reference-label','controlReference');
            qtOpSetText('#viz-actual-label','actual');
            qtOpSetText('#viz-force-measured-label','measuredForce');
            qtOpSetText('#viz-force-predicted-label','predictedForce');
            const responseClean = document.getElementById('viz-response-clean');
            if (responseClean) {
              responseClean.textContent = qtOpText('cleanResponse');
              responseClean.title = qtOpText('cleanResponseHelp');
            }
            const responseMode = document.getElementById('viz-response-mode');
            if (responseMode) {
              const poseOption = responseMode.querySelector('option[value="pose"]');
              const forceOption = responseMode.querySelector('option[value="force"]');
              if (poseOption) poseOption.textContent = qtOpText('poseResponse');
              if (forceOption) forceOption.textContent = qtOpText('forceResponse');
              responseMode.title = qtOpText('controlResponse');
            }
            qtOpSetText('#camera-title','camera');
            qtOpSetText('#cam-open','openCamera');
            qtOpSetText('#cam-close','closeCamera');
            const cameraDevice = document.getElementById('cam-dev');
            if (cameraDevice) cameraDevice.title = qtOpText('selectCamera');
            const cameraRotate = document.getElementById('cam-rot');
            if (cameraRotate) cameraRotate.title = qtOpText('rotate90');
            const cameraMirror = document.getElementById('cam-mir');
            if (cameraMirror) cameraMirror.title = qtOpText('mirror');
            const cameraClose = document.getElementById('cam-close');
            if (cameraClose) cameraClose.title = qtOpText('closeCameraHelp');
            const stageText = document.getElementById('viz-stage');
            if (stageText && /^(当前阶段：待机|Current stage:\s*Standby)$/.test(stageText.textContent.trim()))
              stageText.textContent = qtOpText('currentStage')+qtOpText('standby');
            qtOpSetText('#joint-pose-title','jointPose');
            const st = typeof KIN !== 'undefined' && KIN === 'st';
            const q1 = document.getElementById('q1-lab');
            const q2 = document.getElementById('q2-lab');
            const q3 = document.getElementById('q3-lab');
            if (q1) q1.innerHTML = st ? '<i>s<sub>x</sub></i> '+qtOpText('swingX') : '<i>q₁</i> '+qtOpText('elevation');
            if (q2) q2.innerHTML = st ? '<i>s<sub>y</sub></i> '+qtOpText('swingY') : '<i>q₂</i> '+qtOpText('planeAngle');
            if (q3) q3.innerHTML = (st ? '<i>ψ</i> '+qtOpText('spinAngle') : '<i>q₃</i> '+qtOpText('axialAngle'))+
              ' <span id="q3-tag" style="font-size:10.5px;font-weight:600">'+
              qtOpText((typeof Q3_LOCK !== 'undefined' && Q3_LOCK) ? 'locked' : 'unlocked')+'</span>';
            if (window.__qtApplyPoseLanguage) window.__qtApplyPoseLanguage();
            if (window.__qtApplyMeasuredLanguage) window.__qtApplyMeasuredLanguage();
            if (window.__qtApplyNano25Language) window.__qtApplyNano25Language();
            if (window.__qtApplyForcePredictionLanguage) window.__qtApplyForcePredictionLanguage();
            if (window.__qtApplyModePanelLanguage) window.__qtApplyModePanelLanguage();
            if (typeof draw === 'function' && typeof geometry === 'function' && typeof state !== 'undefined')
              requestAnimationFrame(() => draw(geometry(state.q)));
          };
          if (qtOpLanguageSelect) {
            qtOpLanguageSelect.value = qtReadStoredLanguage();
            qtOpLanguageSelect.onchange = () => {
              try { localStorage.setItem('tro-operation-language', qtOpLanguageSelect.value); } catch (_) {}
              window.__qtApplyOperationLanguage();
            };
          }
          if (!window.__qtLanguageDebugHook && typeof setDebug === 'function') {
            window.__qtLanguageDebugHook = true;
            const rawSetDebugForLanguage = setDebug;
            setDebug = function(on) {
              const result = rawSetDebugForLanguage(on);
              setTimeout(window.__qtApplyOperationLanguage, 0);
              return result;
            };
          }
          setTimeout(window.__qtApplyOperationLanguage, 0);
          setTimeout(window.__qtApplyOperationLanguage, 600);
          // Qt 本地版默认每个示教段 2s；控制站会根据实测方向速度自动延长。
          const dwell = document.getElementById('tc-dwell');
          const dwellNum = document.getElementById('tc-dwelln');
          if (dwell) { dwell.max = '5'; dwell.value = '2'; }
          if (dwellNum) { dwellNum.max = '5'; dwellNum.value = '2'; }
          // Qt GUI 默认进入真机驱动状态；这里只打开使能，不会在连接或播放前主动输出。
          const driveButton = document.getElementById('tc-drv');
          if (driveButton && typeof tcDrive !== 'undefined') {
            tcDrive = true;
            driveButton.setAttribute('aria-pressed', 'true');
          }
          // 独立的真实肌肉长度 PI：不走角度 PI、DataModel 或 FEL。
          const controlMode = document.getElementById('tc-drvmode');
          if (controlMode && !controlMode.querySelector('option[value="mpc_ilc"]')) {
            const opt = document.createElement('option');
            opt.value = 'mpc_ilc'; opt.textContent = 'MPC + 约束 ILC（逐轮真机学习）';
            opt.title = '原始真正模型+LTV MPC固定；每轮只学习一条受限、平滑的任务前馈曲线，真实MSE变差自动回滚';
            const pureMpc = controlMode.querySelector('option[value="mpc"]');
            if (pureMpc) pureMpc.insertAdjacentElement('afterend', opt);
            else controlMode.insertBefore(opt, controlMode.firstChild);
          }
          if (controlMode && !controlMode.querySelector('option[value="mpc_codex"]')) {
            const opt = document.createElement('option');
            opt.value = 'mpc_codex'; opt.textContent = 'MPC + Codex 轮间补偿器';
            opt.title = '原始真正模型+LTV MPC固定；每轮结束后导出response，由当前Codex离线设计下一版受限补偿器；勾选循环时设计完成自动开始下一轮';
            const ilcOpt = controlMode.querySelector('option[value="mpc_ilc"]');
            if (ilcOpt) ilcOpt.insertAdjacentElement('afterend', opt);
            else controlMode.insertBefore(opt, controlMode.firstChild);
          }
          if (controlMode && !controlMode.querySelector('option[value="new_mpc_fel"]')) {
            const opt = document.createElement('option');
            opt.value = 'new_mpc_fel'; opt.textContent = 'NEW MPC + FEL（影子顺模型）';
            const oldFel = controlMode.querySelector('option[value="mpcfel"]');
            if (oldFel) oldFel.insertAdjacentElement('afterend', opt);
            else controlMode.appendChild(opt);
            controlMode.title += '\nNEW MPC+FEL：顺模型和逆模型均为活动/影子结构；每3轮合并训练，审核通过1次即在轮间替换。';
          }
          if (controlMode && !controlMode.querySelector('option[value="mpc_shadow"]')) {
            const opt = document.createElement('option');
            opt.value = 'mpc_shadow'; opt.textContent = 'MPC + 影子正模型（无逆模型）';
            const fullFel = controlMode.querySelector('option[value="new_mpc_fel"]');
            if (fullFel) fullFel.insertAdjacentElement('afterend', opt);
            else controlMode.appendChild(opt);
            opt.title = '真正模型用于MPC；影子正模型只学习和验证；不加载、不输出、不训练逆模型NN';
          }
          if (controlMode && !controlMode.querySelector('option[value="musclepi"]')) {
            const opt = document.createElement('option');
            opt.value = 'musclepi'; opt.textContent = '肌肉长度 PI（六组传感器）';
            const openOpt = controlMode.querySelector('option[value="open"]');
            controlMode.insertBefore(opt, openOpt || null);
          }
          // 闭环参考生成方式。Step 在每个规划阶段开始时直接给出目标值，并在该阶段
          // 保持；平滑则保留原五次 S 曲线。Qt GUI 默认使用平滑 S 曲线。
          if (controlMode && !document.getElementById('qt-reference-mode')) {
            const label = document.createElement('label');
            label.id = 'qt-reference-mode-wrap';
            label.className = 'tcopt';
            label.title = 'Step：每段开始直接跳到目标并保持；平滑：五次S曲线逐拍过渡';
            label.innerHTML = `参考
              <select id="qt-reference-mode" style="font-size:11px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:1px 3px">
                <option value="step">Step 阶跃</option>
                <option value="smooth" selected>平滑 S 曲线</option>
              </select>`;
            controlMode.insertAdjacentElement('afterend', label);
            const updateVisibility = () => {
              label.style.display = controlMode.value === 'open' ? 'none' : 'inline-flex';
            };
            controlMode.addEventListener('change', updateVisibility);
            updateVisibility();
          }
          // NEW MPC+FEL 的成对模型选择，以及无逆模型影子分支的独立重置按钮。
          // 重置只恢复 mpc_shadow 分支，不覆盖完整 NEW MPC+FEL 的正/逆模型。
          const installModelSource = () => {
            const mode = document.getElementById('tc-drvmode');
            if (!mode) { setTimeout(installModelSource, 200); return; }
            if (document.getElementById('qt-model-source-panel')) return;
            const panel = document.createElement('div');
            panel.id = 'qt-model-source-panel'; panel.className = 'btns';
            panel.style.cssText = 'margin-top:5px;align-items:center';
            panel.title = '完整FEL可选择初始化来源；无逆模型影子模式可单独恢复到基础拟合顺模型';
            panel.innerHTML = `<span class="aux" id="qt-model-source-label">正/逆模型</span>
              <select id="qt-model-source" style="font-size:11px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:2px 4px">
                <option value="original" selected>原始正模型＋零逆模型（generation 0）</option>
                <option value="latest">最新已验证（继续学习）</option>
              </select>
              <button class="btn" id="qt-reset-mpc-shadow"
                title="清除MPC+影子正模型（无逆模型）分支的已学习权重和审核历史，恢复到基础拟合顺模型 generation 0">影子模型从0训练</button>
              <button class="btn" id="qt-reset-mpc-ilc"
                title="只清除约束ILC的逐拍前馈曲线和审核历史，不改真正顺模型或MPC参数">ILC从0开始</button>
              <button class="btn" id="qt-reset-mpc-codex"
                title="清除Codex补偿参数、候选和报告；不改真正顺模型或MPC参数">Codex补偿器清零</button>
              <span class="aux" id="qt-model-source-note"></span>`;
            mode.closest('.btns').insertAdjacentElement('afterend', panel);
            const note = panel.querySelector('#qt-model-source-note');
            const select = panel.querySelector('#qt-model-source');
            const label = panel.querySelector('#qt-model-source-label');
            const resetShadow = panel.querySelector('#qt-reset-mpc-shadow');
            const resetIlc = panel.querySelector('#qt-reset-mpc-ilc');
            const resetCodex = panel.querySelector('#qt-reset-mpc-codex');
            const update = () => {
              const fullFel = mode.value === 'new_mpc_fel';
              const shadowOnly = mode.value === 'mpc_shadow';
              const ilc = mode.value === 'mpc_ilc';
              const codex = mode.value === 'mpc_codex';
              panel.style.display = (fullFel || shadowOnly || ilc || codex) ? 'flex' : 'none';
              select.style.display = fullFel ? '' : 'none';
              resetShadow.style.display = shadowOnly ? '' : 'none';
              resetIlc.style.display = ilc ? '' : 'none';
              resetCodex.style.display = codex ? '' : 'none';
              label.textContent = codex ? 'Codex补偿器' : ilc ? '约束ILC' : shadowOnly ? '影子正模型' : '正/逆模型';
              note.textContent = codex
                ? '每次完整跑一轮再设计；勾选“循环”后，候选就绪会自动开始下一轮'
                : ilc
                ? '同一轨迹自动续学；换轨迹自动回到generation 0'
                : shadowOnly
                ? '独立分支；每3轮合并训练并审核1次'
                : (select.value === 'original'
                  ? '每次播放先重置为原始正模型generation 0和零输出逆模型，再从第1轮学习'
                  : '最新活动正/逆模型，影子继续更新');
            };
            resetShadow.onclick = async () => {
              if (typeof tcPlaying !== 'undefined' && tcPlaying) {
                if (typeof tcSay === 'function') tcSay('请先停止控制，再重置影子模型。', true);
                return;
              }
              if (!window.confirm('确定清除无逆模型影子分支的学习结果，并从基础拟合顺模型重新训练吗？')) return;
              resetShadow.disabled = true;
              note.textContent = '正在重置…';
              try {
                const response = await fetch('http://127.0.0.1:8765/reset_mpc_shadow', {
                  method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'
                });
                const result = await response.json();
                if (result.err) throw new Error(result.err);
                note.textContent = '已回到 generation 0，等待第1/3轮';
                if (typeof tcSay === 'function') tcSay('<b>MPC+影子模型已从0开始</b>：'+result.message, false);
              } catch (error) {
                note.textContent = '重置失败';
                if (typeof tcSay === 'function') tcSay('影子模型重置失败：'+error.message, true);
              } finally {
                resetShadow.disabled = false;
              }
            };
            resetIlc.onclick = async () => {
              if (typeof tcPlaying !== 'undefined' && tcPlaying) {
                if (typeof tcSay === 'function') tcSay('请先停止控制，再重置ILC。', true);
                return;
              }
              if (!window.confirm('确定清除约束ILC已经学到的逐拍前馈曲线吗？真正顺模型和MPC参数不会改变。')) return;
              resetIlc.disabled = true;
              note.textContent = '正在清零…';
              try {
                const response = await fetch('http://127.0.0.1:8765/reset_mpc_ilc', {
                  method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'
                });
                const result = await response.json();
                if (result.err) throw new Error(result.err);
                note.textContent = '已回到 generation 0';
                if (typeof tcSay === 'function') tcSay('<b>MPC+ILC已从0开始</b>：'+result.message, false);
              } catch (error) {
                note.textContent = '重置失败';
                if (typeof tcSay === 'function') tcSay('ILC重置失败：'+error.message, true);
              } finally {
                resetIlc.disabled = false;
              }
            };
            const resetCodexCompensator = async () => {
              if (typeof tcPlaying !== 'undefined' && tcPlaying) {
                if (typeof tcSay === 'function') tcSay('请先停止控制，再清零Codex补偿器。', true);
                return false;
              }
              if (!window.confirm('确定清除Codex补偿器参数、候选和报告吗？真正顺模型和MPC参数不会改变。')) return false;
              resetCodex.disabled = true;
              note.textContent = '正在清零…';
              try {
                const response = await fetch('http://127.0.0.1:8765/reset_codex_compensator', {
                  method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'
                });
                const result = await response.json();
                if (result.err) throw new Error(result.err);
                note.textContent = '已清零；下一轮建立无补偿基线';
                if (typeof tcSay === 'function') tcSay('<b>Codex补偿器已清零</b>：'+result.message, false);
                return true;
              } catch (error) {
                note.textContent = '清零失败';
                if (typeof tcSay === 'function') tcSay('Codex补偿器清零失败：'+error.message, true);
                return false;
              } finally {
                resetCodex.disabled = false;
              }
            };
            window.__qtResetCodexCompensator = resetCodexCompensator;
            resetCodex.onclick = resetCodexCompensator;
            mode.addEventListener('change', update);
            select.addEventListener('change', update);
            update();

            // Codex 设计必须使用一整轮 response，因此控制站内部仍按单轮结束并放气。
            // “循环”由 GUI 在候选校验完成后重新启动下一轮，避免用户每轮再按一次播放。
            // 任意手动停止、取消循环、切换模式、IMU 断线或设计失败都会解除自动续跑。
            if (!window.__qtCodexCycleInstalled) {
              window.__qtCodexCycleInstalled = true;
              window.__qtCodexCycleWaiting = false;
              window.__qtCodexCycleToken = 0;
              const play = document.getElementById('tc-play');
              const loopBox = document.getElementById('tc-loop');
              const cancelCodexCycle = (message, bad=false) => {
                window.__qtCodexCycleToken = (window.__qtCodexCycleToken || 0) + 1;
                const wasWaiting = !!window.__qtCodexCycleWaiting;
                window.__qtCodexCycleWaiting = false;
                if (window.__qtTeachExecutionPlan) window.__qtTeachExecutionPlan.active = false;
                if (wasWaiting) {
                  try { tcPlaying = false; } catch (_) {}
                  if (play) play.textContent = '播放';
                  if (message && typeof tcSay === 'function') tcSay(message, bad);
                }
              };
              window.__qtCancelCodexCycle = cancelCodexCycle;

              if (typeof window.tcStop === 'function') {
                const rawTcStopForCodexCycle = window.tcStop;
                window.tcStop = function(...args) {
                  cancelCodexCycle();
                  return rawTcStopForCodexCycle.apply(this, args);
                };
              }
              if (loopBox) loopBox.addEventListener('change', () => {
                if (!loopBox.checked) cancelCodexCycle('Codex 自动循环已取消。', false);
              });
              mode.addEventListener('change', () => {
                if (mode.value !== 'mpc_codex') cancelCodexCycle();
              });

              window.__qtStartCodexCycleWait = async function(statusHint=null) {
                const loop = document.getElementById('tc-loop');
                if (mode.value !== 'mpc_codex' || !loop || !loop.checked ||
                    window.__qtCodexCycleWaiting) return;
                const executionPlan = window.__qtTeachExecutionPlan;
                if (executionPlan && executionPlan.active) {
                  // 每个完成轮次的request_id唯一。用它去重，避免 /step done回调与
                  // 后台ready恢复轮询同时到达时把同一轮计数两次，从而提前停止。
                  let autoStatus = statusHint && typeof statusHint === 'object'
                    ? statusHint : null;
                  if (!autoStatus || !autoStatus.request_id) {
                    try {
                      const status = await (await fetch('http://127.0.0.1:8765/status')).json();
                      autoStatus = status && status.codex_auto ? status.codex_auto : autoStatus;
                    } catch (_) {}
                  }
                  if (window.__qtCodexCycleWaiting) return;
                  const requestId = String(autoStatus && autoStatus.request_id || '');
                  let countThisRun = true;
                  if (requestId) {
                    if (executionPlan.lastCountedRequestId === requestId) countThisRun = false;
                    else if (executionPlan.lastCountedRequestId === '__pending__') {
                      executionPlan.lastCountedRequestId = requestId;
                      countThisRun = false;
                    } else executionPlan.lastCountedRequestId = requestId;
                  } else if (executionPlan.lastCountedRequestId === '__pending__') {
                    countThisRun = false;
                  } else executionPlan.lastCountedRequestId = '__pending__';
                  const invalidRetry = !!(autoStatus && autoStatus.state === 'retry');
                  if (countThisRun && !invalidRetry)
                    executionPlan.completedRuns = Number(executionPlan.completedRuns || 0) + 1;
                  const runLimitReached = executionPlan.limitUnit === 'runs' &&
                    executionPlan.limitValue != null &&
                    executionPlan.completedRuns >= Number(executionPlan.limitValue);
                  const minuteLimitReached = executionPlan.limitUnit === 'min' &&
                    executionPlan.deadlineMs != null && Date.now() >= executionPlan.deadlineMs;
                  if (runLimitReached || minuteLimitReached) {
                    executionPlan.active = false;
                    if (typeof tcSay === 'function') tcSay(
                      '<b>示教轨迹已按设定完成</b>：已完成 '+executionPlan.completedRuns+' 个完整轮次。', false);
                    return;
                  }
                }
                const token = (window.__qtCodexCycleToken || 0) + 1;
                window.__qtCodexCycleToken = token;
                window.__qtCodexCycleWaiting = true;
                // 原 tcMpcStart 的完成分支会把按钮恢复为“播放”，下一任务让本轮脚本
                // 先结束，再明确显示等待状态；此时按按钮就是停止整个自动循环。
                setTimeout(() => {
                  if (!window.__qtCodexCycleWaiting || token !== window.__qtCodexCycleToken) return;
                  try { tcPlaying = true; } catch (_) {}
                  if (play) play.textContent = '停止循环';
                  if (typeof tcSay === 'function')
                    tcSay(window.__qtMpcLlmReplayActive
                      ? '<b>本轮回放已完成并放气</b>，正在载入保存的下一轮控制器。'
                      : '<b>本轮已完成并放气</b>，正在等待 Codex 生成下一版补偿器；完成后自动开始下一轮。', false);
                }, 0);

                const pollAndRestart = async () => {
                  if (!window.__qtCodexCycleWaiting || token !== window.__qtCodexCycleToken) return;
                  const currentPlan = window.__qtTeachExecutionPlan;
                  if (currentPlan && currentPlan.active && currentPlan.limitUnit === 'min' &&
                      currentPlan.deadlineMs != null && Date.now() >= currentPlan.deadlineMs) {
                    currentPlan.active = false;
                    cancelCodexCycle('示教轨迹已达到设定分钟数；当前完整轮次已结束，循环停止。', false);
                    return;
                  }
                  const currentLoop = document.getElementById('tc-loop');
                  if (mode.value !== 'mpc_codex' || !currentLoop || !currentLoop.checked) {
                    cancelCodexCycle('Codex 自动循环已取消。', false); return;
                  }
                  if (typeof imuOn !== 'undefined' && !imuOn) {
                    cancelCodexCycle('IMU 已断开，Codex 自动循环停止。', true); return;
                  }
                  try {
                    const status = await (await fetch('http://127.0.0.1:8765/status')).json();
                    const auto = status && status.codex_auto ? status.codex_auto : {};
                    if (auto.state === 'ready' || auto.state === 'reused' ||
                        auto.state === 'retry' || auto.state === 'replay_ready') {
                      const reuseCurrent = auto.state === 'reused';
                      const retryStable = auto.state === 'retry';
                      const replayNext = auto.state === 'replay_ready';
                      note.textContent = replayNext
                        ? '正在载入保存的下一轮控制器…'
                        : retryStable
                        ? '本轮数据无效，正在自动重跑稳定补偿器…'
                        : reuseCurrent
                          ? '当前补偿器已达到MSE阈值，正在沿用并开始下一轮…'
                          : 'Codex候选已就绪，正在自动开始下一轮…';
                      // 自动续轮不清空已经显示的 response。tcMpcStart 是原页面的
                      // 普通启动函数，会调用 tcChartReset；这里启动后立即恢复历史，
                      // 并把新一轮时间接到上一点之后。控制响应独立窗口的保留标志
                      // 同时随 /start 传给 Python 控制站。
                      const chartHistory = (typeof tcPlot !== 'undefined' && Array.isArray(tcPlot))
                        ? tcPlot.slice() : null;
                      const chartNextT = chartHistory && chartHistory.length
                        ? Number(chartHistory[chartHistory.length - 1].t || 0) + 0.05 : 0;
                      window.__qtCodexPreserveResponse = true;
                      let ok = false;
                      try {
                        ok = await tcMpcStart();
                      } finally {
                        window.__qtCodexPreserveResponse = false;
                        if (chartHistory) {
                          try {
                            tcPlot = chartHistory;
                            tcPlotT0 = performance.now() - chartNextT * 1000;
                            if (typeof tcChartDraw === 'function') tcChartDraw();
                          } catch (_) {}
                        }
                      }
                      if (token !== window.__qtCodexCycleToken || !window.__qtCodexCycleWaiting) {
                        if (ok && typeof tcMpcStop === 'function') await tcMpcStop();
                        return;
                      }
                      window.__qtCodexCycleWaiting = false;
                      if (!ok) {
                        try { tcPlaying = false; } catch (_) {}
                        if (play) play.textContent = '播放';
                        if (typeof tcSay === 'function') tcSay('下一轮自动启动失败，循环已停止。', true);
                        return;
                      }
                      try { tcPlaying = true; } catch (_) {}
                      if (play) play.textContent = '停止循环';
                      if (typeof tcSay === 'function') tcSay(retryStable
                        ? '<b>本轮数据审核无效，不计入runs；已保留稳定补偿器并自动重跑。</b>'
                        : replayNext
                          ? '<b>保存的下一轮控制器已载入，过程回放继续。</b>'
                          : reuseCurrent
                          ? '<b>当前补偿器满足MSE阈值，已沿用并开始下一轮。</b>'
                          : '<b>Codex 新补偿器已载入，下一轮已自动开始。</b>', false);
                      return;
                    }
                    if (auto.state === 'replay_complete') {
                      cancelCodexCycle(auto.message || 'MPC+LLM过程回放已完成。', false);
                      return;
                    }
                    if (['error','skipped','disabled'].includes(auto.state)) {
                      cancelCodexCycle('Codex 未能生成可用候选，自动循环停止：' +
                        (auto.message || auto.state), true);
                      return;
                    }
                  } catch (_) {
                    // 短暂查询失败不立即停机；当前已经放气，下一秒继续查询即可。
                  }
                  setTimeout(pollAndRestart, 1000);
                };
                setTimeout(pollAndRestart, 100);
              };
            }
            if (!window.__qtCodexAutoPoll) {
              window.__qtCodexAutoPoll = setInterval(async () => {
                if (mode.value !== 'mpc_codex') return;
                try {
                  const r = await (await fetch('http://127.0.0.1:8765/status')).json();
                  const s = r && r.codex_auto;
                  if (!s || !s.state) return;
                  const prefix = {
                    running:'Codex分析中：', ready:'Codex候选已就绪：', reused:'沿用当前补偿器：',
                    retry:'数据无效，自动重跑稳定版：',
                    testing:'实机审核中：', error:'Codex自动分析失败：',
                    replay_loaded:'过程回放：', replay_testing:'过程回放中：',
                    replay_ready:'下一轮回放：', replay_wait:'模拟思考：', replay_complete:'过程回放完成：',
                    skipped:'未自动设计：', disabled:'自动设计已关闭：', idle:''
                  }[s.state] || '';
                  note.textContent = prefix + (s.message || s.state);
                  // 如果done回调因页面调度/切换恰好丢失，但后台候选已经ready，
                  // 只要原执行计划仍有效且尚未达到runs/mins，就恢复等待并续跑。
                  const plan = window.__qtTeachExecutionPlan;
                  const activeLoop = document.getElementById('tc-loop');
                  if ((s.state === 'ready' || s.state === 'reused' || s.state === 'retry' ||
                       s.state === 'replay_ready' || s.state === 'replay_wait') &&
                      plan && plan.active && plan.loop && activeLoop && activeLoop.checked &&
                      !window.__qtCodexCycleWaiting &&
                      !(typeof tcMpcTimer !== 'undefined' && tcMpcTimer) &&
                      typeof window.__qtStartCodexCycleWait === 'function') {
                    window.__qtStartCodexCycleWait(s);
                  }
                } catch (_) {}
              }, 1000);
            }
          };
          // 操作模式右栏顶部：连接 → 归零 → 姿态驱动。debug 的原六轴 IMU 面板保持原样。
          const installOperationPosePanel = () => {
            const qZero = document.getElementById('q-zero');
            const imuConnectButton = document.getElementById('imu-sw');
            const imuZeroButton = document.getElementById('imu-zero');
            const imuDriveButton = document.getElementById('imu-drive');
            if (!qZero || !imuConnectButton || !imuZeroButton || !imuDriveButton ||
                typeof imuTick !== 'function') {
              setTimeout(installOperationPosePanel, 200); return;
            }
            if (document.getElementById('qt-operation-pose')) return;
            if (!document.getElementById('qt-operation-pose-style')) {
              const style = document.createElement('style');
              style.id = 'qt-operation-pose-style';
              style.textContent =
                '#qt-operation-pose{padding-top:12px;padding-bottom:12px}'+
                '#qt-operation-pose .qt-op-actions{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}'+
                '#qt-operation-pose .qt-op-actions .btn{padding:7px 5px;font-weight:600}'+
                '#qt-operation-pose .qt-op-actions .btn[aria-pressed="true"]{background:var(--accent);'+
                'border-color:var(--accent);color:var(--on-accent)}'+
                '#qt-operation-pose .qt-op-status{margin-top:7px;font-size:11px;color:var(--muted)}'+
                '#qt-operation-pose .qt-op-status.qt-op-warning{color:var(--bad);font-weight:700;'+
                'padding:6px 8px;border:1px solid var(--bad);border-radius:5px;background:rgba(179,55,46,.08)}'+
                'body:not(.debug) #q-zero.qt-q-zero-head{padding:1px 8px;font-size:11px;font-weight:500;'+
                'text-transform:none;letter-spacing:0;white-space:nowrap}'+
                'body:not(.debug) #qt-q-zero-original-row{display:none}';
              document.head.appendChild(style);
            }
            const jointSection = qZero.closest('section');
            const jointHeading = jointSection ? jointSection.querySelector('h2') : null;
            const qZeroParent = qZero.parentNode;
            const qZeroNext = qZero.nextSibling;
            if (qZeroParent) qZeroParent.id = 'qt-q-zero-original-row';
            const panel = document.createElement('section');
            panel.id = 'qt-operation-pose';
            panel.className = 'op-only';
            panel.innerHTML =
              '<h2><span id="qt-op-pose-title">设备连接与姿态</span> <span class="sp"></span><span class="aux" id="qt-op-link-state">未连接</span></h2>'+ 
              '<div class="qt-op-actions">'+
                '<button class="btn" id="qt-op-connect">连接</button>'+ 
                '<button class="btn" id="qt-op-zero">归零</button>'+ 
                '<button class="btn" id="qt-op-drive" aria-pressed="true">姿态联动</button>'+ 
              '</div><div class="qt-op-status" id="qt-op-status">姿态联动默认开启；连接后点击“归零”。</div>';
            jointSection.parentNode.insertBefore(panel, jointSection);
            const connect = panel.querySelector('#qt-op-connect');
            const zero = panel.querySelector('#qt-op-zero');
            const drive = panel.querySelector('#qt-op-drive');
            const stateText = panel.querySelector('#qt-op-link-state');
            const detail = panel.querySelector('#qt-op-status');
            const debugQZeroText = qZero.textContent;
            const debugQZeroTitle = qZero.title;
            let serialNotice = '';
            const setDrive = enabled => {
              imuDrive = !!enabled;
              imuDriveButton.setAttribute('aria-pressed', String(imuDrive));
              drive.setAttribute('aria-pressed', String(imuDrive));
              drive.textContent = imuDrive ? qtOpText('poseDrive') : qtOpText('poseDriveOff');
            };
            const sync = () => {
              panel.querySelector('#qt-op-pose-title').textContent = qtOpText('devicePose');
              zero.textContent = qtOpText('zero');
              connect.textContent = imuOn ? qtOpText('disconnect') : qtOpText('connect');
              connect.setAttribute('aria-pressed', String(imuOn));
              zero.disabled = !imuOn;
              stateText.textContent = imuOn ? qtOpText('connected') : qtOpText('disconnected');
              detail.classList.toggle('qt-op-warning', !!serialNotice);
              if (serialNotice) {
                stateText.textContent = qtOpText('noData');
                detail.textContent = '⚠ ' + (qtOpEffectiveLanguage() === 'en'
                  ? qtOpText('serialResetNotice') : serialNotice);
              } else {
                detail.textContent = imuOn
                  ? (imuDrive ? qtOpText('driveEnabled') : qtOpText('driveDisabled'))
                  : qtOpText('driveDefault');
              }
              drive.textContent = imuDrive ? qtOpText('poseDrive') : qtOpText('poseDriveOff');
              drive.setAttribute('aria-pressed', String(!!imuDrive));
            };
            const syncMode = inDebug => {
              qZero.textContent = inDebug ? debugQZeroText : qtOpText('defaultZero');
              qZero.title = inDebug ? debugQZeroTitle : qtOpText('defaultPoseHelp');
              qZero.classList.toggle('qt-q-zero-head', !inDebug);
              if (inDebug && qZeroParent && qZero.parentNode !== qZeroParent) {
                if (qZeroNext && qZeroNext.parentNode === qZeroParent)
                  qZeroParent.insertBefore(qZero, qZeroNext);
                else qZeroParent.appendChild(qZero);
              } else if (!inDebug && jointHeading && qZero.parentNode !== jointHeading) {
                jointHeading.appendChild(qZero);
              }
              if (window.__qtSyncMeasuredMode) window.__qtSyncMeasuredMode();
            };
            connect.onclick = () => {
              serialNotice = '';
              imuConnectButton.click();
              setTimeout(sync, 0);
            };
            zero.onclick = () => {
              if (!imuOn) return;
              imuZeroButton.click();
              setDrive(true);
              qZero.click();
              if (typeof pyHost !== 'undefined' && pyHost &&
                  typeof pyHost.resetForcePrediction === 'function')
                pyHost.resetForcePrediction(()=>{});
              imuTick();
              detail.textContent = qtOpText('zeroed');
              sync();
            };
            drive.onclick = () => { imuDriveButton.click(); sync(); };

            const installSerialNoticeBridge = () => {
              if (typeof pyHost === 'undefined' || !pyHost || !pyHost.notice) {
                setTimeout(installSerialNoticeBridge, 200); return;
              }
              if (window.__qtSerialNoticeBridge) return;
              window.__qtSerialNoticeBridge = true;
              pyHost.notice.connect(message => {
                serialNotice = String(message || '串口没有返回数据；请按板上 RST 重新启动机器。');
                if (typeof imuSay === 'function') imuSay('<b>串口无数据</b>：'+serialNotice, true);
                sync();
              });
              if (pyHost.rx) pyHost.rx.connect(text => {
                if (serialNotice && String(text || '').trim()) {
                  serialNotice = '';
                  sync();
                }
              });
            };
            installSerialNoticeBridge();

            // 操作模式的默认状态；进入 debug 时恢复旧面板原来的“未启用驱动”状态。
            setDrive(true);
            syncMode(false);
            qZero.click();
            if (!window.__qtOperationModeHook && typeof setDebug === 'function') {
              window.__qtOperationModeHook = true;
              const rawSetDebug = setDebug;
              setDebug = function(on) {
                const result = rawSetDebug(on);
                setDrive(!on);
                syncMode(!!on);
                sync();
                return result;
              };
            }
            window.__qtSetOperationPoseDrive = setDrive;
            window.__qtSyncOperationPose = sync;
            window.__qtApplyPoseLanguage = () => {
              syncMode(typeof DEBUG !== 'undefined' && DEBUG);
              sync();
            };
            setInterval(sync, 250);
            sync();
            window.__qtApplyOperationLanguage();
          };
          installOperationPosePanel();

          // 操作界面的肌肉长度来自20路拉线编码器：主动ID1..10、被动ID11..20。
          // 原六路模型指令仍留在页面状态中兼容旧功能，但 Qt 界面不再显示这些假读数。
          const installMeasuredContraction = () => {
            const rows = document.getElementById('f-rows');
            const enc = document.getElementById('enc-lens');
            const title = document.getElementById('f-title');
            if (!rows || !enc || !title) { setTimeout(installMeasuredContraction, 200); return; }
            if (document.getElementById('qt-measured-contraction')) return;
            title.textContent = qtOpText('muscleParameters');
            title.title = qtOpText('muscleParametersHelp');
            const measured = document.createElement('div');
            measured.id = 'qt-measured-contraction';
            rows.parentNode.insertBefore(measured, rows);
            enc.title = '原长235mm；主动端ID=k，被动端ID=k+10。两者任一路掉线时该肌肉显示--';
            measured.appendChild(enc);
            const groups = [
              {name:'A1+A5', pressure:0}, {name:'A2+A3', pressure:2},
              {name:'A4+A8', pressure:1}, {name:'A6+A7', pressure:3},
              {name:'A9', pressure:4}, {name:'A10', pressure:5}
            ];
            window.__qtShowLengthPlaceholders = () => {
              enc.hidden = false;
              enc.innerHTML = groups.map(group =>
                `<span class="qt-sensor-pair qt-muscle-unknown" data-drive-channel="${group.pressure}">`+
                `<i>${group.name}</i> <b>--</b> mm<em class="qt-sensor-pressure">P -- kPa</em></span>`
              ).join('');
            };
            window.__qtSyncMeasuredMode = () => {
              const inDebug = typeof DEBUG !== 'undefined' && DEBUG;
              title.textContent = inDebug ? '真实肌肉长度（传感器）' : qtOpText('muscleParameters');
              title.title = inDebug
                ? '由20路编码器实时计算，不使用几何模型的收缩量指令'
                : qtOpText('muscleParametersHelp');
              if (!imuOn) {
                if (inDebug) enc.hidden = true;
                else window.__qtShowLengthPlaceholders();
              }
            };
            window.__qtApplyMeasuredLanguage = window.__qtSyncMeasuredMode;
            window.__qtShowLengthPlaceholders();
            if (!window.__qtLengthDisconnectHook && typeof imuDisconnect === 'function') {
              window.__qtLengthDisconnectHook = true;
              const rawImuDisconnectForLengths = imuDisconnect;
              imuDisconnect = async function(...args) {
                const result = await rawImuDisconnectForLengths.apply(this, args);
                if (typeof DEBUG === 'undefined' || !DEBUG) {
                  window.__qtShowLengthPlaceholders();
                  if (window.__qtSetOperationPoseDrive) window.__qtSetOperationPoseDrive(true);
                }
                if (window.__qtSyncOperationPose) window.__qtSyncOperationPose();
                return result;
              };
            }
            rows.hidden = true;
            rows.style.display = 'none';
            // 这些控件只服务原来的模型指令编辑；真实长度是只读传感器量，不应由滑块修改。
            ['fpre-row','f-btns','balance-note','preload-note',
             'couple-sw','npam-sw','aux-sw','in-sw','model-sw','f-sum'].forEach(id => {
              const node = document.getElementById(id);
              if (node) node.style.display = 'none';
            });
            window.__qtApplyOperationLanguage();
          };
          installMeasuredContraction();
          // 原页面只解析10路；Qt当前固件帧的第7段扩展为20路，并按主动/被动成对显示。
          const installTwentySensors = () => {
            if (typeof imuParse !== 'function' || typeof window.imuSend !== 'function') {
              setTimeout(installTwentySensors, 200); return;
            }
            if (window.__qtTwentySensors) return;
            window.__qtTwentySensors = true;
            // 颜色只表示六路比例阀当前收到的驱动指令，不再由长度传感器变化判断。
            // 通道顺序与发送给固件的 W/O 指令一致：V0..V5。
            window.__qtLastDriveU = Array(6).fill(0);
            const refreshDriveColors = () => {
              document.querySelectorAll('#enc-lens .qt-sensor-pair[data-drive-channel]').forEach(node => {
                const ch = Number(node.dataset.driveChannel);
                const u = Number(window.__qtLastDriveU[ch]) || 0;
                const driven = u > 0;
                node.classList.toggle('qt-muscle-driven', driven);
                node.classList.toggle('qt-muscle-idle', !driven);
                node.title = `${driven ? '正在驱动' : '未驱动'}：V${ch}=${Math.round(u)} mV；锁=${node.dataset.lock || '--'}`;
              });
            };
            const rawImuSendForDriveColor = window.imuSend;
            window.imuSend = function(command, ...args) {
              try {
                const text = String(command == null ? '' : command).trim();
                const kind = text.charAt(0).toUpperCase();
                if (kind === 'W' || kind === 'O') {
                  const values = text.slice(1).split(',').slice(0, 6).map(Number);
                  if (values.length === 6 && values.every(Number.isFinite))
                    window.__qtLastDriveU = values.map(v => Math.max(0, v));
                } else if (kind === 'S') {
                  window.__qtLastDriveU = Array(6).fill(0);
                }
                refreshDriveColors();
              } catch (_) {}
              return rawImuSendForDriveColor.call(this, command, ...args);
            };
            const rawImuParse = imuParse;
            imuParse = function(line) {
              const ok = rawImuParse(line);
              if (!ok) return ok;
              const parts = String(line).split('|').map(t => t.trim());
              if (parts.length < 7) return ok;
              const tokens = parts[6].split(/\s+/).filter(Boolean).slice(0, 20);
              if (tokens.length < 2) return ok;
              const got = tokens.map(t => {
                const v = Number.parseFloat(t);
                return Number.isFinite(v) ? v : null;
              });
              imuLensNum = Array.from({length:20}, (_, i) => i < got.length ? got[i] : null);
              const el = document.getElementById('enc-lens');
              if (!el) return ok;
              const lengths = Array.from({length:10}, (_, i) =>
                imuLensNum[i] != null && imuLensNum[i+10] != null
                  ? 235 - Math.abs(imuLensNum[i]) + Math.abs(imuLensNum[i+10]) : null);
              const fmt = v => v == null ? '--' : v.toFixed(1);
              const groups = [
                {name:'A1+A5', ids:[0,4], pressure:0},
                {name:'A2+A3', ids:[1,2], pressure:2},
                {name:'A4+A8', ids:[3,7], pressure:1},
                {name:'A6+A7', ids:[5,6], pressure:3},
                {name:'A9', ids:[8], pressure:4},
                {name:'A10', ids:[9], pressure:5}
              ];
              const avgs = groups.map(group => {
                const vals = group.ids.map(i => lengths[i]);
                return vals.every(v => v != null)
                  ? vals.reduce((a,b) => a+b, 0)/vals.length : null;
              });
              el.hidden = false;
              el.innerHTML = groups.map((group, i) => {
                const avg = avgs[i];
                const drive = Number(window.__qtLastDriveU[group.pressure]) || 0;
                const driven = drive > 0;
                const stateClass = driven ? 'qt-muscle-driven' : 'qt-muscle-idle';
                const pressure = Array.isArray(imuKpa) && Number.isFinite(Number(imuKpa[group.pressure]))
                  ? Number(imuKpa[group.pressure]).toFixed(0) : '--';
                const lock = typeof imuLocks === 'string' && imuLocks.length > group.pressure
                  ? imuLocks[group.pressure] : '--';
                return `<span class="qt-sensor-pair ${stateClass}" data-drive-channel="${group.pressure}" data-lock="${lock}" `+
                  `title="${driven?'正在驱动':'未驱动'}：V${group.pressure}=${Math.round(drive)} mV；锁=${lock}">`+
                  `<i>${group.name}</i> <b>${fmt(avg)}</b> mm`+
                  `<em class="qt-sensor-pressure">P ${pressure} kPa</em></span>`;
              }).join('');
              return ok;
            };
          };
          installTwentySensors();
          // Operation Mode stays in the main window; Debug uses a native,
          // independently movable window that proxies the same page state.
          const debugBtn = document.getElementById('mode-debug');
          if (debugBtn && typeof debugConsoleOpen === 'function') {
            debugBtn.onclick = () => {
              if (typeof debugConsoleIsOpen === 'function' && debugConsoleIsOpen()) debugConsoleClose();
              else debugConsoleOpen();
            };
            debugBtn.title = '打开可拖到另一块屏幕的 Debug Console；Operation Mode 保持在主窗口';
          }
          // 示教记录姿态死区：相对上一示教点不超过 2° 的分量视为手抖/漂移，沿用上一点。
          // 只修改记录进 teach 的参考姿态，不滤控制反馈、压力或 DataModel 标定数据。
          const installTeachDeadband = () => {
            const rec = document.getElementById('tc-rec');
            if (!rec || typeof rec.onclick !== 'function' ||
                typeof teach === 'undefined' || !Array.isArray(teach)) {
              setTimeout(installTeachDeadband, 200); return;
            }
            if (window.__qtTeachDeadband) return;
            window.__qtTeachDeadband = true;
            const rawRecord = rec.onclick;
            const deadbandDeg = 2.0;
            rec.title += '；Qt 本地版会忽略相对上一点不超过 2° 的姿态小变化';
            rec.onclick = function(...args) {
              const before = teach.length;
              rawRecord.apply(this, args);
              if (teach.length !== before + 1) return;
              const cur = teach[before];
              // 示教快照同时保存20路拉线编码器实测位移；保持 d 字段作为旧的六路指令，
              // 避免改变现有开环导出和控制协议的含义。
              cur.lens_sensor = (typeof imuLensNum !== 'undefined' && Array.isArray(imuLensNum))
                ? imuLensNum.slice(0, 20).map(v => v != null && Number.isFinite(Number(v)) ? Number(v) : null)
                : Array(20).fill(null);
              cur.muscle_length_mm = Array.from({length:10}, (_, i) => {
                const active = cur.lens_sensor[i], passive = cur.lens_sensor[i+10];
                return active != null && passive != null
                  ? 235 - Math.abs(active) + Math.abs(passive) : null;
              });
              const lengthGroups = [[0,4],[1,2],[3,7],[5,6],[8],[9]];
              cur.channel_length_mm = lengthGroups.map(ids => {
                const vals = ids.map(i=>cur.muscle_length_mm[i]);
                return vals.every(v=>v!=null) ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
              });
              // q/q_raw 始终保留传感器实际示教值，列表和 reference 曲线都画它；
              // q_control 单独保存经过2°记录死区后的控制用示教点。
              cur.q_raw = cur.q.slice();
              if (before < 1) { cur.q_control = cur.q.slice(); return; }
              const prev = teach[before-1];
              const prevControl = Array.isArray(prev.q_control) ? prev.q_control : prev.q;
              const held = [];
              cur.q_control = cur.q.map((v, i) => {
                if (Math.abs(v-prev.q[i]) <= deadbandDeg) { held.push(i); return prevControl[i]; }
                return v;
              });
              if (!held.length) return;
              if (typeof tcRender === 'function') tcRender(before);
              if (typeof tcSay === 'function') {
                const labels = (typeof KIN !== 'undefined' && KIN === 'st') ? ['s_x','s_y','ψ'] : ['q₁','q₂','q₃'];
                tcSay(`已记第 ${teach.length} 点：${held.map(i=>labels[i]).join('、')} 的变化≤${deadbandDeg}°，已按上一点处理。`, false);
              }
            };
          };
          // Reproducible benchmark trajectory recovered from the logged raw
          // reference.  It is loaded from a real JSON file, not reconstructed
          // from the measured response, so every controller sees the same task.
          const installFixedTask = () => {
            const rec = document.getElementById('tc-rec');
            if (!rec || typeof teach === 'undefined' || !Array.isArray(teach) ||
                typeof pyHost === 'undefined' || !pyHost || typeof pyHost.loadFixedTask !== 'function') {
              setTimeout(installFixedTask, 200); return;
            }
            if (document.getElementById('qt-load-fixed-task')) return;
            const btn = document.createElement('button');
            btn.className = 'btn';
            btn.id = 'qt-load-fixed-task';
            btn.textContent = '载入固定任务';
            btn.title = '载入重要测试任务的3个真实示教点；将替换当前示教列表';
            rec.parentNode.insertBefore(btn, rec.nextSibling);
            const loadTask = (automatic=false, done=null, reference='1') => {
              const referenceId = String(reference);
              const loader = referenceId === '3' && typeof pyHost.loadFixedTask3 === 'function'
                ? pyHost.loadFixedTask3
                : referenceId === '2' && typeof pyHost.loadFixedTask2 === 'function'
                  ? pyHost.loadFixedTask2 : pyHost.loadFixedTask;
              loader.call(pyHost, txt => {
                try {
                  const blob = JSON.parse(txt);
                  if (blob.error) throw new Error(blob.error);
                  if (!Array.isArray(blob.points) || !blob.points.length)
                    throw new Error('固定任务没有轨迹点');
                  if (typeof tcStop === 'function') tcStop();
                  teach = blob.points.map(src => ({
                    q: src.q.slice(0,3).map(Number),
                    q_raw: (src.q_raw || src.q).slice(0,3).map(Number),
                    q_control: (src.q_control || src.q).slice(0,3).map(Number),
                    t: Number(src.t), d: Array(6).fill(0),
                    lk: Array(6).fill(false), ll: Array(10).fill(null), err: 0
                  }));
                  teach.auto_hidden_return_to_zero = blob.auto_hidden_return_to_zero !== false;
                  // Keep the preset identity with the array so /start can
                  // enable narrowly scoped, repeatable experimental rules.
                  teach.fixed_task_id = referenceId;
                  tcRender(-1);
                  const fallbackNames = {'1':'Default 1','2':'Default 2','3':'Default 3'};
                  const taskName = String(blob.name || fallbackNames[referenceId] || 'Default 1');
                  const finishText = teach.auto_hidden_return_to_zero
                    ? '，末尾自动回零2.0s。' : '，末点后全机放气。';
                  tcSay(`${automatic?'已自动载入':'已载入'} ${taskName}：${teach.length}个示教点，`+
                    `规划 ${teach.reduce((s,p)=>s+p.t,0).toFixed(1)}s${finishText}`, false);
                  if (typeof done === 'function') done(true, blob);
                } catch (e) {
                  tcSay('固定任务载入失败：'+e.message, true);
                  if (typeof done === 'function') done(false, e);
                }
              });
            };
            window.__qtLoadFixedTask = loadTask;
            btn.onclick = () => loadTask(false, null, '1');
            if (teach.length === 0) loadTask(true);
          };
          if (!window.__qtNoMinVoltageText && typeof window.tcSay === 'function') {
            window.__qtNoMinVoltageText = true;
            const rawSay = window.tcSay;
            window.tcSay = function(msg, ...args) {
              msg = String(msg)
                .replace('（&lt;60 固件视为 0）', '')
                .replace('；&lt;60 视为 0', '')
                .split('≤2V').join('≤2.2V');
              return rawSay.call(this, msg, ...args);
            };
          }
          if (!window.__qtMpcMountFetch && typeof window.fetch === 'function') {
            window.__qtMpcMountFetch = true;
            const rawFetch = window.fetch.bind(window);
            window.fetch = function(input, init) {
              const url = String(input && input.url ? input.url : input);
              if (url.endsWith('/start') && init && typeof init.body === 'string') {
                try {
                  const body = JSON.parse(init.body);
                  const el = document.getElementById('imu-map');
                  body.mount = el ? el.value : 'ry-';
                  const modeEl = document.getElementById('tc-drvmode');
                  const referenceEl = document.getElementById('qt-reference-mode');
                  const modelSourceEl = document.getElementById('qt-model-source');
                  body.reference_mode = referenceEl ? referenceEl.value : 'smooth';
                  body.model_source = modelSourceEl ? modelSourceEl.value : 'latest';
                  body.new_mpc_fel = !!(modeEl && modeEl.value === 'new_mpc_fel');
                  body.shadow_forward_only = !!(modeEl && modeEl.value === 'mpc_shadow');
                  body.ilc_mpc = !!(modeEl && modeEl.value === 'mpc_ilc');
                  body.codex_compensator = !!(modeEl && modeEl.value === 'mpc_codex');
                  body.auto_hidden_return_to_zero = !(
                    typeof teach !== 'undefined' && teach.auto_hidden_return_to_zero === false);
                  body.fixed_task_id = (typeof teach !== 'undefined' &&
                    teach.fixed_task_id != null) ? String(teach.fixed_task_id) : '';
                  body.preserve_response = !!(body.codex_compensator &&
                    window.__qtCodexPreserveResponse);
                  const executionPlan = window.__qtTeachExecutionPlan;
                  body.execution_round = executionPlan && executionPlan.active
                    ? Number(executionPlan.completedRuns || 0) + 1 : 1;
                  if (executionPlan && executionPlan.active && executionPlan.loop) {
                    body.loop_limit_runs = executionPlan.limitUnit === 'runs'
                      ? executionPlan.limitValue : null;
                    body.loop_limit_min = executionPlan.limitUnit === 'min'
                      ? executionPlan.limitValue : null;
                  }
                  if (body.shadow_forward_only) {
                    body.ctrl = 'mpc';
                    body.fel = false;
                  }
                  if (body.ilc_mpc) {
                    body.ctrl = 'mpc';
                    body.fel = false;
                  }
                  if (body.codex_compensator) {
                    body.ctrl = 'mpc';
                    body.fel = false;
                    // Codex 模式每次只把一个完整轮次交给控制站；勾选循环时，
                    // GUI 会在轮后分析完成并得到候选后自动发起下一次 /start。
                    body.loop = false;
                  }
                  if (modeEl && modeEl.value === 'musclepi') {
                    body.ctrl = 'musclepi';
                    body.fel = false;
                  }
                  if (Array.isArray(body.points) && typeof teach !== 'undefined') {
                    body.points = body.points.map((p, i) => {
                      const tp = teach[i] || {};
                      const rawLens = Array.isArray(tp.lens_sensor) ? tp.lens_sensor.slice(0, 20) : null;
                      const recalculatedMuscle = rawLens && rawLens.length >= 20
                          ? Array.from({length:10}, (_, k) => {
                            const av = rawLens[k], bv = rawLens[k+10];
                            const a = Number(av), b = Number(bv);
                            return av != null && bv != null && Number.isFinite(a) && Number.isFinite(b)
                              ? 235 - Math.abs(a) + Math.abs(b) : null;
                          }) : null;
                      const lengthGroups = [[0,4],[1,2],[3,7],[5,6],[8],[9]];
                      const recalculatedChannel = recalculatedMuscle
                        ? lengthGroups.map(ids => {
                            const vals = ids.map(k => recalculatedMuscle[k]);
                            return vals.every(v => v != null)
                              ? vals.reduce((a,b) => a+b, 0)/vals.length : null;
                          }) : null;
                      return Object.assign({}, p, {
                        q: Array.isArray(tp.q_control) ? tp.q_control : p.q,
                        q_raw: Array.isArray(tp.q_raw) ? tp.q_raw : (Array.isArray(tp.q) ? tp.q : p.q),
                        lens_sensor: rawLens,
                        muscle_length_mm: recalculatedMuscle ||
                          (Array.isArray(tp.muscle_length_mm) ? tp.muscle_length_mm.slice(0, 10) : null),
                        channel_length_mm: recalculatedChannel ||
                          (Array.isArray(tp.channel_length_mm) ? tp.channel_length_mm.slice(0, 6) : null)
                      });
                    });
                  }
                  init = Object.assign({}, init, {body: JSON.stringify(body)});
                } catch (_) {}
              }
              return rawFetch(input, init);
            };
          }
          // Qt本地开环直连允许到3000mV；MPC/PI的W指令由gains.json:u_max限幅。
          if (!window.__qtOpenOutput3000 && typeof tcMvForPose === 'function') {
            window.__qtOpenOutput3000 = true;
            const rawTcMvForPose = tcMvForPose;
            tcMvForPose = function(...args) {
              return rawTcMvForPose(...args).map(v => Math.max(0, Math.min(3000, Math.round(Number(v)||0))));
            };
            const drv = document.getElementById('tc-drv');
            if (drv) drv.title = String(drv.title).replace('≤2V', '≤3V');
          }
          // Qt 本地版的四个固定开环模式。它们不读取示教点：
          //   1) 15(V0) 令 Y 到用户目标，再由 9(V4) 令 X 到用户目标，最后全机回零；
          //   2) 15(V0) 与 48(V1) 依次令 Y 到两个用户目标；
          //   3) 15→23→48→67：每路升压1.5s、减压1s；开始减压时启动下一路。
          //   4) 48(V1) 令 Y 到负向用户目标，再由 23(V2) 令 φ 到正向用户目标，最后全机回零。
          // 操作界面的每个模式可独立选择“单次/循环”；debug 保持原来的连续循环。
          // Mode 1/4 每阶段最长3s，Mode 2每阶段最长5s；停止/异常立即清零六路比例阀。
          const installFixedOpenLoop = () => {
            const play = document.getElementById('tc-play');
            const mode = document.getElementById('tc-drvmode');
            if (!play || !mode || typeof play.onclick !== 'function' ||
                typeof window.imuSend !== 'function') {
              setTimeout(installFixedOpenLoop, 200); return;
            }
            if (window.__qtFixedOpenLoop) return;
            window.__qtFixedOpenLoop = true;
            const openOpt = mode.querySelector('option[value="open"]');
            if (openOpt) {
              openOpt.textContent = '5 开环·固定模式';
              openOpt.title = '五个固定真机模式，不读取示教点；正式操作界面可分别选择单次或循环';
            }
            const anglePanel = document.createElement('div');
            anglePanel.id = 'qt-open-angle-panel';
            anglePanel.className = 'btns';
            anglePanel.style.cssText = 'margin-top:5px;align-items:center';
            anglePanel.title = '仅用于开环固定模式；选择后直接按播放，不需要示教点';
            anglePanel.innerHTML = `
              <span class="aux">固定模式</span>
              <select id="qt-open-fixed-mode" style="font-size:11px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:2px 4px">
                <option value="mode1">Mode 1｜Y +60° → X −20° → 全回0 循环</option>
                <option value="mode2">Mode 2｜Y +60° → Y −20° 循环</option>
                <option value="mode3">Mode 3｜15 → 23 → 48 → 67 循环</option>
                <option value="mode4">Mode 4｜Y −30° → φ +20° → 全回0 循环</option>
                <option value="mode5">Mode 5｜15 → 23 → 48 → 67，对侧延迟共充</option>
              </select>
              <span class="aux" id="qt-open-fixed-note"></span>`;
            mode.closest('.btns').insertAdjacentElement('afterend', anglePanel);
            const fixedMode = document.getElementById('qt-open-fixed-mode');
            const fixedNote = document.getElementById('qt-open-fixed-note');
            const modeNotes = {
              mode1: '15→Y+60°，9→X−20°，六路0mV保持1s后再循环；每阶段≤3s，3000mV',
              mode2: '15→Y+60°，48→Y−20°；每阶段≤5s，连续循环，3000mV',
              mode3: '每路2.5s：升压1.5s＋减压1s；前三次间隔1.5s，67→下一轮15间隔2s；一轮6.5s',
              mode4: '48→Y−30°，23→φ+20°，六路0mV保持1s后再循环；每阶段≤3s，3000mV',
              mode5: '主侧升压1.5s、降压1s；对侧0.5～1s升到3000mV、保持到1.5s，然后两侧一起降压；每轮均执行'
            };
            const showAnglePanel = () => {
              anglePanel.style.display = mode.value === 'open' ? 'flex' : 'none';
              fixedNote.textContent = modeNotes[fixedMode.value];
            };
            mode.addEventListener('change', showAnglePanel);
            fixedMode.addEventListener('change', showAnglePanel);
            showAnglePanel();
            const rawPlay = play.onclick;
            const rawStop = (typeof window.tcStop === 'function') ? window.tcStop : null;
            const fullMv = 3000;
            let targetYDeg = 60.0;
            let targetXDeg = -20.0;
            let targetPhiDeg = 20.0;
            let targetYNegDeg = -20.0;
            let activeLoop = true;
            let activeRunLimit = Infinity;
            let activeDurationMin = null;
            let activeDeadlineMs = Infinity;
            let mode3StopAfterLap = null;
            const mode1StageTimeoutS = 3.0;
            const mode2StageTimeoutS = 5.0;
            const mode1PrelockS = 0.100;
            const mode1InterLoopDelayS = 1.0;
            const cycleRiseS = 1.5;
            const cycleFallS = 1.0;
            const cycleStartStepS = cycleRiseS;
            const cycleActiveS = cycleRiseS + cycleFallS;
            const cycleBoundaryStepS = 2.0;
            const cycleChannels = [
              {name:'15', ch:0}, {name:'23', ch:2},
              {name:'48', ch:1}, {name:'67', ch:3}
            ];
            // Mode 5独立计算，保留Mode 3原有时序；跨轮保留67的末尾降压。
            const mode5OutputAt = (seconds, lapLimit=Infinity) => {
              const lapS = cycleStartStepS*(cycleChannels.length-1)+cycleBoundaryStepS;
              const lap = Math.floor(Math.max(0,seconds)/lapS);
              const result = [0,0,0,0,0,0];
              const rise = fraction => 0.5*(1-Math.cos(Math.PI*fraction));
              for (const previousLap of [lap-1,lap]) {
                if (previousLap < 0 || previousLap >= lapLimit) continue;
                cycleChannels.forEach((item,index) => {
                  const age = seconds-previousLap*lapS-index*cycleStartStepS;
                  if (age < 0 || age >= cycleActiveS) return;
                  const fall = age >= cycleRiseS
                    ? 0.5*(1+Math.cos(Math.PI*(age-cycleRiseS)/cycleFallS)) : 1;
                  const primary = age < cycleRiseS ? rise(age/cycleRiseS) : fall;
                  const opposite = age < 0.5 ? 0 : age < 1.0
                    ? rise((age-0.5)/0.5) : fall;
                  result[item.ch] = Math.max(result[item.ch],Math.round(fullMv*primary));
                  result[item.ch^1] = Math.max(result[item.ch^1],Math.round(fullMv*opposite));
                });
              }
              return result;
            };
            let timer = 0, running = false, phase = 0, phaseStartedAt = 0;
            let lastUiAt = 0, activeMode = 'mode1', cycleIndex = 0, cycleLap = 1;
            let cycleStartedAt = 0;
            // 供 IMU 零压漂移诊断判断是否正处于可安全归零的固定任务阶段。
            window.__qtFixedOpenIsRunning = () => running;

            const clearSequence = () => {
              if (timer) clearInterval(timer);
              timer = 0; running = false; phase = 0; phaseStartedAt = 0;
              lastUiAt = 0; cycleIndex = 0; cycleLap = 1; cycleStartedAt = 0;
              activeRunLimit = Infinity; activeDurationMin = null;
              activeDeadlineMs = Infinity; mode3StopAfterLap = null;
              fixedMode.disabled = false;
            };
            const writeOpen = (out) => {
              // O是teach_mode的开环专用流，允许3000mV；W由gains.json:u_max限幅。
              imuSend((activeMode === 'mode5' ? 'C' : 'O')+out.join(',')+'\n');
              // 保留开环的实时角度遥测，便于在“控制响应”窗口观察。
              if (typeof imuQ !== 'undefined' && typeof imuOn !== 'undefined' && imuOn) {
                try {
                  const qr=qMul(qConj(imuRef),imuQ);
                  fetch(TC_STATION+'/telem',{method:'POST',body:JSON.stringify({
                    qrel:Array.from(qr), ref:[0,0,0], u:out
                  })}).catch(()=>{});
                } catch (_) {}
              }
            };
            const writeExplicitPrelock = (channel) => {
              const locks = [0,0,0,0,0,0];
              if (channel >= 0 && channel < locks.length) locks[channel] = 1;
              // M命令把锁请求和零比例阀电压原子发送；预锁阶段绝不发送充气电压。
              imuSend('M'+locks.join(',')+';0,0,0,0,0,0\n');
            };
            const finishSequence = (message, bad=false, allowQueue=true) => {
              const completedMode = activeMode;
              clearSequence();
              writeOpen([0,0,0,0,0,0]);
              // 普通开环结束只清零比例阀，不发 s：有压通道继续自锁，
              // 固件确认压力回零后才自动开锁。“全机放气/急停”按钮仍会显式发 s。
              let queueResult = null;
              if (allowQueue && !bad && typeof window.__qtAdvanceFixedOpenQueue === 'function') {
                try { queueResult = window.__qtAdvanceFixedOpenQueue(completedMode, message); }
                catch (error) { console.error('fixed-open queue advance failed', error); }
              }
              if (queueResult && queueResult.continue) {
                // 队列交接期间保持一次连续任务：录制与 Nano25 不会在模式之间停止。
                tcPlaying = true;
                play.textContent = '停止顺序运行';
                return;
              }
              if (queueResult && queueResult.message) message = queueResult.message;
              if (typeof window.__qtCancelFixedOpenQueue === 'function')
                window.__qtCancelFixedOpenQueue(false);
              tcPlaying = false;
              play.textContent = '播放';
              if (typeof tcSay === 'function') tcSay(
                message || '<b>固定开环模式完成</b>，比例阀已清零；有压通道保持自锁，零压后自动开锁。', bad);
            };
            const tick = () => {
              if (!running) return;
              if (!tcDrive || mode.value !== 'open' ||
                  typeof imuOn === 'undefined' || !imuOn) {
                finishSequence('<b>固定开环模式已停止</b>，比例阀已清零；有压通道继续自锁。', true);
                return;
              }
              const xDeg = Number(imuMeasS && imuMeasS.sx);
              const yDeg = Number(imuMeasS && imuMeasS.sy);
              const phiDeg = Number(imuMeasS && imuMeasS.psi);
              if (activeMode !== 'mode3' && activeMode !== 'mode5' &&
                  (!Number.isFinite(xDeg) || !Number.isFinite(yDeg) ||
                   (activeMode === 'mode4' && !Number.isFinite(phiDeg)))) {
                finishSequence('<b>IMU X/Y/φ 角度无效</b>，试验中止并清零比例阀；有压通道继续自锁。', true);
                return;
              }
              const now = performance.now();
              const elapsed = (now-phaseStartedAt)/1000.0;
              let out = [0,0,0,0,0,0];
              const isTwoAxisMode = activeMode === 'mode1' || activeMode === 'mode4';
              const isMode4 = activeMode === 'mode4';
              const twoAxisModeLabel = isMode4 ? 'Mode 4' : 'Mode 1';

              if (activeMode === 'mode5') {
                const cycleElapsed = Math.max(0,(now-cycleStartedAt)/1000.0);
                const lapS = cycleStartStepS*(cycleChannels.length-1)+cycleBoundaryStepS;
                const rawLapIndex = Math.floor(cycleElapsed/lapS);
                if (!activeLoop) mode3StopAfterLap = 1;
                else if (Number.isFinite(activeRunLimit)) mode3StopAfterLap = activeRunLimit;
                else if (now >= activeDeadlineMs && mode3StopAfterLap == null)
                  mode3StopAfterLap = rawLapIndex+1;
                // 最后一轮67于4.5s启动，完整降压结束于7s；不再截断最后0.5s。
                const endS = mode3StopAfterLap == null ? Infinity
                  : (mode3StopAfterLap-1)*lapS
                    +cycleStartStepS*(cycleChannels.length-1)+cycleActiveS;
                if (cycleElapsed >= endS) {
                  finishSequence(`<b>Mode 5 已完成 ${mode3StopAfterLap} 轮</b>，六路比例阀已清零。`);
                  return;
                }
                cycleLap = Math.min(rawLapIndex+1,mode3StopAfterLap || Infinity);
                out = mode5OutputAt(cycleElapsed,mode3StopAfterLap || Infinity);
                writeOpen(out);
                if (now-lastUiAt >= 500) {
                  lastUiAt = now;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>Mode 5｜第${cycleLap}轮｜对侧延迟共充</b> u=[${out.join(', ')}]mV`,false);
                }
                return;
              }

              if (activeMode === 'mode3') {
                // 一个脉冲持续2.5s：0~1.5s升压、1.5~2.5s减压，不设保持段。
                // 同一轮内下一根在前一根开始减压时启动；67开始2s后启动下一轮15。
                const envelope = age => {
                  if (age < 0 || age >= cycleActiveS) return 0;
                  if (age < cycleRiseS)
                    return 0.5*(1-Math.cos(Math.PI*age/cycleRiseS));
                  const p = (age-cycleRiseS)/cycleFallS;
                  return 0.5*(1+Math.cos(Math.PI*p));
                };
                const cycleElapsed = Math.max(0, (now-cycleStartedAt)/1000.0);
                const cycleLapS = cycleStartStepS*(cycleChannels.length-1)+cycleBoundaryStepS;
                const rawLapIndex = Math.floor(cycleElapsed/cycleLapS);
                const lapTime = cycleElapsed-rawLapIndex*cycleLapS;
                if (!activeLoop) mode3StopAfterLap = 1;
                else if (Number.isFinite(activeRunLimit)) mode3StopAfterLap = activeRunLimit;
                else if (now >= activeDeadlineMs && mode3StopAfterLap == null)
                  mode3StopAfterLap = rawLapIndex+1;
                const scheduledRunS = mode3StopAfterLap == null
                  ? Infinity : cycleLapS*mode3StopAfterLap;
                if (cycleElapsed >= scheduledRunS) {
                  finishSequence(activeLoop
                    ? `<b>Mode 3 循环运行完成</b>，已完成 ${mode3StopAfterLap} 轮，六路比例阀已清零。`
                    : '<b>Mode 3 单次运行完成</b>，六路比例阀已清零。');
                  return;
                }
                cycleIndex = Math.min(cycleChannels.length-1,
                  Math.floor(lapTime/cycleStartStepS));
                const localTime = lapTime-cycleIndex*cycleStartStepS;
                cycleLap = rawLapIndex+1;
                const item = cycleChannels[cycleIndex];
                const currentMv = Math.round(fullMv*envelope(localTime));
                out[item.ch] = currentMv;
                let previousText = '';
                // 同一轮前三次交接保持1秒交叉渐变；67→下一轮15间隔2秒，交叉渐变0.5秒。
                if (cycleIndex > 0) {
                  const previous = cycleChannels[cycleIndex-1];
                  const previousMv = Math.round(fullMv*envelope(localTime+cycleStartStepS));
                  out[previous.ch] = Math.max(out[previous.ch], previousMv);
                  if (previousMv > 0) previousText = `，${previous.name}(V${previous.ch})降压=${previousMv}mV`;
                }
                writeOpen(out);
                if (now-lastUiAt >= 500) {
                  lastUiAt = now;
                  const action = '平滑升压';
                  if (typeof tcSay === 'function') tcSay(
                    `<b>Mode 3｜第${cycleLap}轮 ${item.name}(V${item.ch})</b>：${action}=${currentMv}mV${previousText}`, false);
                }
                return;
              }

              if (isTwoAxisMode && (phase === 0 || phase === 2)) {
                // Mode 1/4显式预锁：第一阶段分别锁15(V0)/48(V1)，第二阶段分别锁9(V4)/23(V2)。
                // 整整100ms内比例阀保持0；到时后的下一控制拍才进入对应充气阶段。
                const prelockChannel = phase === 0 ? (isMode4 ? 1 : 0) : (isMode4 ? 2 : 4);
                const prelockName = phase === 0
                  ? (isMode4 ? '48(V1)' : '15(V0)') : (isMode4 ? '23(V2)' : '9(V4)');
                writeExplicitPrelock(prelockChannel);
                if (elapsed >= mode1PrelockS) {
                  phase = phase === 0 ? 1 : 3;
                  phaseStartedAt = now; lastUiAt = 0;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>${twoAxisModeLabel}｜${prelockName}已先锁定${Math.round(mode1PrelockS*1000)}ms</b>，下一拍开始充气。`, false);
                } else if (now-lastUiAt >= 100) {
                  lastUiAt = now;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>${twoAxisModeLabel}｜正在预锁${prelockName}</b>：比例阀0mV，${Math.round(elapsed*1000)}/${Math.round(mode1PrelockS*1000)}ms`, false);
                }
                return;
              }

              if (isTwoAxisMode && phase === 4) {
                // X动作结束后六路保持0mV。若还有下一轮，固定等待1s后继续；
                // 轮次交接不再以X/Y姿态是否回到±3°作为判据。
                writeOpen([0,0,0,0,0,0]);
                const runLimitReached = Number.isFinite(activeRunLimit) &&
                  cycleLap >= activeRunLimit;
                if (!activeLoop || runLimitReached || now >= activeDeadlineMs) {
                  finishSequence(
                    `<b>${twoAxisModeLabel} ${activeLoop ? `已完成 ${cycleLap} 轮` : '单次运行完成'}</b>：六路比例阀已清零。`);
                  return;
                }
                if (elapsed >= mode1InterLoopDelayS) {
                  // 分钟限制也在等待结束时复核，避免到时后误启动新一轮。
                  if (now >= activeDeadlineMs) {
                    finishSequence(
                      `<b>${twoAxisModeLabel}已完成 ${cycleLap} 轮</b>：分钟运行时间已到，六路比例阀已清零。`);
                    return;
                  }
                  const completedLap = cycleLap;
                  cycleLap += 1; phase = 0; phaseStartedAt = now; lastUiAt = 0;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>${twoAxisModeLabel}｜第${completedLap}轮结束</b>：六路0mV已保持1s；开始第${cycleLap}轮。`, false);
                  return;
                }
                if (now-lastUiAt >= 250) {
                  lastUiAt = now;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>${twoAxisModeLabel}｜第${cycleLap}轮交接等待</b> — 六路0mV，${Math.min(1,elapsed).toFixed(1)}/1.0s`, false);
                }
                return;
              }

              const stageTimeoutS = isTwoAxisMode
                ? mode1StageTimeoutS : mode2StageTimeoutS;
              const yDrivePhase = isTwoAxisMode ? 1 : 0;
              const secondDrivePhase = isTwoAxisMode ? 3 : 1;
              if (phase === yDrivePhase) {
                const yTargetReached = isMode4 ? yDeg <= targetYDeg : yDeg >= targetYDeg;
                const yMuscleName = isMode4 ? '48(V1)' : '15(V0)';
                if (yTargetReached) {
                  // 明确停掉Y通道一拍，下一拍才启用第二根肌肉，避免命令重叠。
                  writeOpen([0,0,0,0,0,0]);
                  phase = isTwoAxisMode ? 2 : 1;
                  phaseStartedAt = now; lastUiAt = 0;
                  const next = isTwoAxisMode
                    ? (isMode4
                      ? `23(V2) 追 φ=${targetPhiDeg.toFixed(1)}°`
                      : `9(V4) 追 X=${targetXDeg.toFixed(1)}°`)
                    : `48(V1) 追 Y=${targetYNegDeg.toFixed(1)}°`;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>Y 已到 ${yDeg.toFixed(1)}°</b>：${yMuscleName}停止，下一拍开始${next}。`, false);
                  console.log(`# ${activeMode}：Y=${yDeg.toFixed(2)}° 到位，${yMuscleName}清零`);
                  return;
                }
                if (elapsed >= stageTimeoutS) {
                  // 第一阶段未到位也必须按固定时序继续；先清零一拍再进入第二阶段。
                  writeOpen([0,0,0,0,0,0]);
                  phase = isTwoAxisMode ? 2 : 1;
                  phaseStartedAt = now; lastUiAt = 0;
                  const next = isTwoAxisMode
                    ? (isMode4
                      ? `23(V2)追φ=${targetPhiDeg.toFixed(1)}°`
                      : `9(V4)追X=${targetXDeg.toFixed(1)}°`)
                    : `48(V1)追Y=${targetYNegDeg.toFixed(1)}°`;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>第一阶段${stageTimeoutS.toFixed(0)}s到时</b>：Y=${yDeg.toFixed(1)}°，虽未到${targetYDeg.toFixed(1)}°，仍继续${next}。`, false);
                  console.log(`# ${activeMode}：第一阶段到时未到位，${yMuscleName}清零后继续第二阶段`);
                  return;
                }
                out = isMode4 ? [0,fullMv,0,0,0,0] : [fullMv,0,0,0,0,0];
              } else if (phase === secondDrivePhase) {
                const reached = isMode4
                  ? phiDeg >= targetPhiDeg
                  : isTwoAxisMode ? xDeg <= targetXDeg : yDeg <= targetYNegDeg;
                if (reached) {
                  const result = isMode4
                    ? `φ已到${phiDeg.toFixed(1)}°（目标${targetPhiDeg.toFixed(1)}°）`
                    : isTwoAxisMode
                    ? `X已到${xDeg.toFixed(1)}°（目标${targetXDeg.toFixed(1)}°）`
                    : `Y已到${yDeg.toFixed(1)}°（目标${targetYNegDeg.toFixed(1)}°）`;
                  if (isTwoAxisMode) {
                    writeOpen([0,0,0,0,0,0]);
                    phase = 4; phaseStartedAt = now; lastUiAt = 0;
                    if (typeof tcSay === 'function') tcSay(
                      `<b>${twoAxisModeLabel}｜第${cycleLap}轮${isMode4 ? 'φ' : 'X'}动作完成</b>：${result}；六路已清零，若需循环则等待1s。`, false);
                    return;
                  }
                  const completedLap = cycleLap;
                  if (!activeLoop || (Number.isFinite(activeRunLimit) && cycleLap >= activeRunLimit) || now >= activeDeadlineMs) {
                    finishSequence(`<b>Mode 2 ${activeLoop ? `已完成 ${cycleLap} 轮` : '单次运行完成'}</b>：${result}。`);
                    return;
                  }
                  // 第二阶段完成后同样清零一拍，再回到第一阶段；循环只由手动停止、
                  // 模式切换或IMU异常终止，避免相邻两路命令直接重叠。
                  writeOpen([0,0,0,0,0,0]);
                  cycleLap += 1; phase = 0; phaseStartedAt = now; lastUiAt = 0;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>${isTwoAxisMode ? twoAxisModeLabel : 'Mode 2'}｜第${completedLap}轮完成</b>：${result}；下一拍开始第${cycleLap}轮。`, false);
                  return;
                }
                if (elapsed >= stageTimeoutS) {
                  const current = isMode4
                    ? `φ=${phiDeg.toFixed(1)}°`
                    : isTwoAxisMode ? `X=${xDeg.toFixed(1)}°` : `Y=${yDeg.toFixed(1)}°`;
                  if (isTwoAxisMode) {
                    writeOpen([0,0,0,0,0,0]);
                    phase = 4; phaseStartedAt = now; lastUiAt = 0;
                    if (typeof tcSay === 'function') tcSay(
                      `<b>${twoAxisModeLabel}｜第${cycleLap}轮${isMode4 ? 'φ' : 'X'}动作${stageTimeoutS.toFixed(0)}s到时</b>：${current}；六路已清零，若需循环则等待1s。`, false);
                    return;
                  }
                  const completedLap = cycleLap;
                  if (!activeLoop || (Number.isFinite(activeRunLimit) && cycleLap >= activeRunLimit) || now >= activeDeadlineMs) {
                    finishSequence(
                      `<b>Mode 2 ${activeLoop ? `已完成 ${cycleLap} 轮` : '单次运行结束'}</b>：${current}，第二阶段${stageTimeoutS.toFixed(0)}s到时。`);
                    return;
                  }
                  // 超时用于限制单路连续满电压时间，不再结束循环。
                  writeOpen([0,0,0,0,0,0]);
                  cycleLap += 1; phase = 0; phaseStartedAt = now; lastUiAt = 0;
                  if (typeof tcSay === 'function') tcSay(
                    `<b>${isTwoAxisMode ? twoAxisModeLabel : 'Mode 2'}｜第${completedLap}轮第二阶段${stageTimeoutS.toFixed(0)}s到时</b>：${current}，未到−20°；已清零，下一拍开始第${cycleLap}轮。`, false);
                  return;
                }
                out = isMode4
                  ? [0,0,fullMv,0,0,0]
                  : isTwoAxisMode ? [0,0,0,0,fullMv,0] : [0,fullMv,0,0,0,0];
              }
              writeOpen(out);
              if (now-lastUiAt >= 500) {
                lastUiAt = now;
                const text = phase === yDrivePhase
                  ? `阶段1/2：${isMode4 ? '48(V1)' : '15(V0)'}=${fullMv}mV，Y=${yDeg.toFixed(1)}° / ${targetYDeg.toFixed(1)}°`
                  : isTwoAxisMode
                  ? (isMode4
                    ? `阶段2/2：23(V2)=${fullMv}mV，φ=${phiDeg.toFixed(1)}° / ${targetPhiDeg.toFixed(1)}°`
                    : `阶段2/2：9(V4)=${fullMv}mV，X=${xDeg.toFixed(1)}° / ${targetXDeg.toFixed(1)}°`)
                  : `阶段2/2：48(V1)=${fullMv}mV，Y=${yDeg.toFixed(1)}° / ${targetYNegDeg.toFixed(1)}°`;
                if (typeof tcSay === 'function') tcSay(
                  `<b>${isTwoAxisMode ? twoAxisModeLabel : 'Mode 2'}｜第${cycleLap}轮</b> — ${text}`, false);
              }
            };
            play.onclick = function(...args) {
              // 只要选中了开环模式，播放键就直接执行固定试验；不再落回需要
              // 至少两个示教点的原回放逻辑。播放本身即视为本次真机驱动确认。
              if (mode.value !== 'open') return rawPlay.apply(this, args);
              if (running) { finishSequence('<b>固定开环模式已手动停止</b>，六路比例阀已清零。', false, false); return; }
              if (typeof imuOn === 'undefined' || !imuOn) {
                if (typeof tcSay === 'function') tcSay('先连接 IMU/串口，再运行固定开环试验。', true);
                return;
              }
              activeMode = fixedMode.value;
              targetYDeg = activeMode === 'mode4' ? -30.0 : 60.0;
              targetXDeg = -20.0; targetPhiDeg = 20.0; targetYNegDeg = -20.0;
              // debug 沿用原先“持续循环”；正式操作界面则读取每行自己的参数与循环框。
              activeLoop = (typeof DEBUG !== 'undefined' && DEBUG);
              activeRunLimit = Infinity;
              activeDurationMin = null;
              activeDeadlineMs = Infinity;
              mode3StopAfterLap = null;
              if ((!activeLoop || window.__qtTimedAutomaticActive) && typeof window.__qtGetFixedOpenConfig === 'function') {
                const cfg = window.__qtGetFixedOpenConfig(activeMode) || {};
                if (Number.isFinite(Number(cfg.targetYDeg))) targetYDeg = Number(cfg.targetYDeg);
                if (Number.isFinite(Number(cfg.targetXDeg))) targetXDeg = Number(cfg.targetXDeg);
                if (Number.isFinite(Number(cfg.targetPhiDeg))) targetPhiDeg = Number(cfg.targetPhiDeg);
                if (Number.isFinite(Number(cfg.targetYNegDeg))) targetYNegDeg = Number(cfg.targetYNegDeg);
                activeLoop = !!cfg.loop;
                const limitValue = Number(cfg.limitValue);
                if (activeLoop && cfg.limitValue != null && Number.isFinite(limitValue)) {
                  if (cfg.limitUnit === 'min') activeDurationMin = Math.max(0.1, limitValue);
                  else activeRunLimit = Math.max(1, Math.floor(limitValue));
                }
              }
              if (window.__qtTimedAutomaticActive) {
                activeLoop = false; activeRunLimit = 1; activeDurationMin = null;
              }
              if (!tcDrive) {
                tcDrive = true;
                const driveButton = document.getElementById('tc-drv');
                if (driveButton) driveButton.setAttribute('aria-pressed', 'true');
              }
              if (rawStop) rawStop();
              running = true; tcPlaying = true; phase = 0; cycleIndex = 0; cycleLap = 1;
              phaseStartedAt = performance.now(); cycleStartedAt = phaseStartedAt; lastUiAt = 0;
              activeDeadlineMs = activeLoop && activeDurationMin != null
                ? phaseStartedAt+activeDurationMin*60000 : Infinity;
              fixedMode.disabled = true;
              play.textContent = activeLoop ? '停止循环' : '停止运行';
              if (typeof tcChartReset === 'function') tcChartReset();
              if (typeof tcSay === 'function') {
                const task = activeMode === 'mode1'
                  ? `Mode 1｜Y ${targetYDeg.toFixed(1)}° → X ${targetXDeg.toFixed(1)}° → 回零`
                  : activeMode === 'mode2'
                  ? `Mode 2｜Y ${targetYDeg.toFixed(1)}° → Y ${targetYNegDeg.toFixed(1)}°`
                  : activeMode === 'mode4'
                  ? `Mode 4｜Y ${targetYDeg.toFixed(1)}° → φ ${targetPhiDeg.toFixed(1)}° → 回零`
                  : activeMode === 'mode5'
                  ? 'Mode 5｜15 → 23 → 48 → 67｜对侧0.5s后共充'
                  : 'Mode 3｜15 → 23 → 48 → 67';
                tcSay(`<b>${task}｜${activeLoop ? '循环' : '单次'}</b> 已启动。`, false);
              }
              tick();
              timer = setInterval(tick, 100);
            };
            if (rawStop) {
              window.tcStop = function(...args) {
                const wasRunning = running;
                clearSequence();
                if (typeof window.__qtCancelFixedOpenQueue === 'function')
                  window.__qtCancelFixedOpenQueue(false);
                const answer = rawStop.apply(this, args);
                if (wasRunning) {
                  writeOpen([0,0,0,0,0,0]);
                }
                return answer;
              };
            }
          };
          installFixedOpenLoop();
          // 正式操作界面的三级模式入口。原示教/控制面板保留在 debug 中，
          // 此处只代理它已有的按钮和选择器，避免出现两套控制状态。
          const installOperationModePanel = () => {
            const title = document.getElementById('f-title');
            const record = document.getElementById('tc-rec');
            const play = document.getElementById('tc-play');
            const clear = document.getElementById('tc-clear');
            const vent = document.getElementById('tc-vent');
            const legacyMode = document.getElementById('tc-drvmode');
            const fixedMode = document.getElementById('qt-open-fixed-mode');
            const teachCount = document.getElementById('tc-sum');
            if (!title || !record || !play || !clear || !vent || !legacyMode || !fixedMode) {
              setTimeout(installOperationModePanel, 200); return;
            }
            if (document.getElementById('qt-operation-modes')) return;
            const lengthSection = title.closest('section');
            const legacySection = record.closest('section');
            if (!lengthSection || !legacySection) return;
            let nanoPanel = document.getElementById('qt-operation-nano25');
            if (!nanoPanel) {
              nanoPanel = document.createElement('section');
              nanoPanel.id = 'qt-operation-nano25';
              nanoPanel.className = 'op-only';
              nanoPanel.innerHTML =
                '<h2><span id="qt-nano25-title">Nano25</span><span class="sp"></span>'+ 
                  '<span class="aux" id="qt-nano25-rate">@ 0 Hz</span>'+ 
                  '<label class="aux qt-force-mode"><span id="qt-force-prediction-mode-label">预测方式</span> '+
                    '<select id="qt-force-prediction-mode">'+
                      '<option value="fusion">Fusion单网络</option>'+
                      '<option value="separate">三轴分开模型</option>'+
                    '</select></label>'+ 
                  '<button class="btn" id="qt-nano25-toggle" type="button">启动 Nano25</button>'+ 
                  '<button class="btn" id="qt-force-prediction-collapse" type="button" aria-expanded="true">隐藏预测力</button></h2>'+ 
                '<div id="qt-nano25-body">'+
                '<div id="qt-nano25-values" title="F_GUI[Y,Z,X] = -F_Nano25">'+
                  '<span class="qt-nano25-value" data-nano-index="0"><i>F<sub>Y</sub></i><span class="qt-nano25-reading"><span class="qt-force-prediction-inline" data-force-axis="1"><small>预测</small><strong>--</strong></span><b>0.000</b></span><em>N</em></span>'+ 
                  '<span class="qt-nano25-value" data-nano-index="1"><i>F<sub>Z</sub></i><span class="qt-nano25-reading"><span class="qt-force-prediction-inline" data-force-axis="2"><small>预测</small><strong>--</strong></span><b>0.000</b></span><em>N</em></span>'+ 
                  '<span class="qt-nano25-value" data-nano-index="2"><i>F<sub>X</sub></i><span class="qt-nano25-reading"><span class="qt-force-prediction-inline" data-force-axis="0"><small>预测</small><strong>--</strong></span><b>0.000</b></span><em>N</em></span>'+ 
                  '<span class="qt-nano25-value" data-nano-index="3"><i>T<sub>X</sub></i><b>0.000</b><em>N·m</em></span>'+ 
                  '<span class="qt-nano25-value" data-nano-index="4"><i>T<sub>Y</sub></i><b>0.000</b><em>N·m</em></span>'+ 
                  '<span class="qt-nano25-value" data-nano-index="5"><i>T<sub>Z</sub></i><b>0.000</b><em>N·m</em></span>'+ 
                '</div>'+ 
                '<p class="note" id="qt-nano25-state">可选传感器；未启动也可执行轨迹。启动前保持无载荷，点击后自动完成约5秒归零。</p>'+ 
                '</div>';
              lengthSection.insertAdjacentElement('afterend', nanoPanel);
              const nanoButton = nanoPanel.querySelector('#qt-nano25-toggle');
              const predictionCollapse = nanoPanel.querySelector('#qt-force-prediction-collapse');
              const nanoRate = nanoPanel.querySelector('#qt-nano25-rate');
              const nanoValues = Array.from(nanoPanel.querySelectorAll('.qt-nano25-value'));
              const nanoStateText = nanoPanel.querySelector('#qt-nano25-state');
              let nanoState = 'stopped';
              let nanoPayload = {state:'stopped', hz:0, force_n:[0,0,0], torque_nm:[0,0,0]};
              let nanoRequirement = false;
              let predictionCollapsed = false;
              try { predictionCollapsed = localStorage.getItem('tro-force-prediction-collapsed') === '1'; }
              catch (_) {}
              const setPredictionCollapsed = (collapsed, persist=true) => {
                predictionCollapsed = !!collapsed;
                nanoPanel.classList.toggle('qt-force-prediction-hidden', predictionCollapsed);
                if (predictionCollapse) {
                  predictionCollapse.setAttribute('aria-expanded', String(!predictionCollapsed));
                  predictionCollapse.textContent = qtOpText(
                    predictionCollapsed ? 'forcePredictionShow' : 'forcePredictionHide');
                }
                if (persist) {
                  try { localStorage.setItem('tro-force-prediction-collapsed', predictionCollapsed ? '1' : '0'); }
                  catch (_) {}
                }
              };
              if (predictionCollapse) predictionCollapse.onclick = () =>
                setPredictionCollapsed(!predictionCollapsed);
              setPredictionCollapsed(predictionCollapsed, false);
              const nanoActive = state => ['starting','zeroing','running','reconnecting','stopping'].includes(state);
              const nanoControlRunning = () => {
                const fixed = typeof window.__qtFixedOpenIsRunning === 'function' &&
                  window.__qtFixedOpenIsRunning();
                return !!((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer) || fixed);
              };
              const nanoNumber = value => {
                const number = Number(value);
                return (Number.isFinite(number) ? number : 0).toFixed(3);
              };
              const setNanoRequirement = required => {
                required = !!required;
                nanoRequirement = required;
                nanoPayload.control_required = required;
                if (typeof pyHost !== 'undefined' && pyHost &&
                    typeof pyHost.setNano25ControlRequired === 'function') {
                  pyHost.setNano25ControlRequired(required, text => {
                    try { renderNano(JSON.parse(String(text || '{}'))); } catch (_) {}
                  });
                }
              };
              const renderNano = payload => {
                if (payload && typeof payload === 'object') nanoPayload = Object.assign({}, nanoPayload, payload);
                nanoState = String(nanoPayload.state || 'stopped');
                nanoRequirement = !!nanoPayload.control_required;
                const force = Array.isArray(nanoPayload.force_n) ? nanoPayload.force_n.slice(0,3) : [0,0,0];
                const torque = Array.isArray(nanoPayload.torque_nm) ? nanoPayload.torque_nm.slice(0,3) : [0,0,0];
                while (force.length < 3) force.push(0);
                while (torque.length < 3) torque.push(0);
                // Nano25 force component order corresponds to GUI [Y,Z,X], but
                // all three sensor-positive directions oppose the GUI axes.
                const guiForce = force.map(value => -(Number(value) || 0));
                window.__qtNano25Snapshot = {
                  valid:['running','reconnecting'].includes(nanoState),
                  force_gui_yzx_n:guiForce.slice(0,3),
                  torque_xyz_nm:torque.slice(0,3).map(value=>Number(value)||0)
                };
                if (typeof window.__qtPushForceResponse === 'function')
                  window.__qtPushForceResponse();
                const hz = Number(nanoPayload.hz);
                const sourceHz = Number(nanoPayload.source_hz);
                const averagedSamples = Number(nanoPayload.averaged_samples);
                const averaged = !!nanoPayload.averaged;
                nanoRate.textContent = `@ ${Math.round(Number.isFinite(hz)?hz:0)} Hz${averaged?' avg':''}`;
                nanoRate.title = averaged
                  ? `NI-DAQ ${Math.round(Number.isFinite(sourceHz)?sourceHz:0)} Hz; `+
                    `${Math.max(0,Math.round(Number.isFinite(averagedSamples)?averagedSamples:0))} samples averaged in this 50 ms window`
                  : '';
                guiForce.concat(torque).forEach((value, index) => {
                  const output = nanoValues[index] && nanoValues[index].querySelector('b');
                  if (output) output.textContent = nanoNumber(value);
                });
                nanoPanel.classList.toggle('qt-nano25-error', nanoState === 'error');
                nanoButton.disabled = nanoState === 'stopping' || nanoRequirement;
                if (nanoState === 'starting') nanoButton.textContent = qtOpText('nano25Starting');
                else if (nanoState === 'zeroing') nanoButton.textContent = qtOpText('nano25Zeroing');
                else if (nanoState === 'stopping') nanoButton.textContent = qtOpText('nano25Stopping');
                else if (nanoRequirement) nanoButton.textContent = qtOpText('nano25ControlRequired');
                else nanoButton.textContent = nanoActive(nanoState) ? qtOpText('nano25Stop') : qtOpText('nano25Start');
                const generic = nanoState === 'error' ? qtOpText('nano25Error') :
                  (nanoState === 'stopped' ? qtOpText('nano25Stopped') :
                   (nanoRequirement ? qtOpText('nano25ControlRequired') : qtOpText('nano25Help')));
                nanoStateText.textContent = nanoState === 'error' && nanoPayload.message
                  ? `${generic}: ${nanoPayload.message}` : generic;
              };
              window.__qtApplyNano25Language = () => {
                nanoPanel.querySelector('#qt-nano25-title').textContent = qtOpText('nano25Title');
                setPredictionCollapsed(predictionCollapsed, false);
                renderNano();
              };
              const requestNanoAction = method => {
                if (typeof pyHost === 'undefined' || !pyHost) {
                  renderNano({state:'error', message:'Qt host is unavailable.'}); return;
                }
                if (typeof pyHost[method] !== 'function') {
                  renderNano({state:'error', message:'Nano25 control is unavailable.'}); return;
                }
                nanoButton.disabled = true;
                pyHost[method](text => {
                  try { renderNano(JSON.parse(String(text || '{}'))); }
                  catch (error) { renderNano({state:'error', message:error.message}); }
                });
              };
              nanoButton.onclick = () => {
                if (nanoRequirement) return;
                requestNanoAction(nanoActive(nanoState) ? 'stopNano25' : 'startNano25');
              };
              const connectNanoBridge = () => {
                if (typeof pyHost === 'undefined' || !pyHost || !pyHost.nano25_status) {
                  setTimeout(connectNanoBridge, 200); return;
                }
                if (!window.__qtNano25Bridge) {
                  window.__qtNano25Bridge = true;
                  pyHost.nano25_status.connect(text => {
                    try { renderNano(JSON.parse(String(text || '{}'))); }
                    catch (error) { renderNano({state:'error', message:error.message}); }
                  });
                }
                if (typeof pyHost.nano25Status === 'function') pyHost.nano25Status(text => {
                  try { renderNano(JSON.parse(String(text || '{}'))); } catch (_) {}
                });
              };
              connectNanoBridge();
              renderNano();
              let lastNanoRequirement = false;
              setInterval(() => {
                // Nano25 is optional. If it was started, keep it continuous
                // during control; if it is absent or drops out, control continues.
                const required = nanoControlRunning() && nanoActive(nanoState);
                if (required !== lastNanoRequirement) {
                  lastNanoRequirement = required;
                  setNanoRequirement(required);
                }
                renderNano();
              }, 100);
            }
            if (!window.__qtForcePredictionUiInstalled) {
              window.__qtForcePredictionUiInstalled = true;
              const predictionValues = Array.from(
                nanoPanel.querySelectorAll('.qt-force-prediction-inline'));
              const predictionMode = nanoPanel.querySelector('#qt-force-prediction-mode');
              const predictionModeLabel = nanoPanel.querySelector(
                '#qt-force-prediction-mode-label');
              let predictionPayload = {
                state:'waiting', force_xyz_n:[0,0,0], hz:0, latency_ms:0,
                history_samples:0, window_size:1, prediction_mode:'fusion'
              };
              const predictionNumber = value => {
                const number = Number(value);
                return Number.isFinite(number) ? number.toFixed(3) : '--';
              };
              const renderForcePrediction = payload => {
                if (payload && typeof payload === 'object')
                  predictionPayload = Object.assign({}, predictionPayload, payload);
                const state = String(predictionPayload.state || 'waiting');
                const mode = predictionPayload.prediction_mode === 'separate'
                  ? 'separate' : 'fusion';
                if (predictionMode && document.activeElement !== predictionMode)
                  predictionMode.value = mode;
                const force = Array.isArray(predictionPayload.force_xyz_n)
                  ? predictionPayload.force_xyz_n.slice(0,3) : [0,0,0];
                while (force.length < 3) force.push(0);
                const valid = state === 'ready' && force.every(value=>Number.isFinite(Number(value)));
                window.__qtPredictedForceSnapshot = {
                  valid:valid,
                  force_xyz_n:force.map(value=>Number(value)||0),
                  model_version:String(predictionPayload.model_version||'force_estimator_v1')
                };
                if (typeof window.__qtPushForceResponse === 'function')
                  window.__qtPushForceResponse();
                predictionValues.forEach(element => {
                  const axis = Number(element.dataset.forceAxis);
                  const output = element.querySelector('strong');
                  if (output) output.textContent = valid ? predictionNumber(force[axis]) : '--';
                });
                const hz = Number(predictionPayload.hz);
                const latency = Number(predictionPayload.latency_ms);
                const featureMax = Number(predictionPayload.feature_max_abs_z);
                const outOfRange = valid && Number.isFinite(featureMax) && featureMax > 8;
                const windowSize = Math.max(1,
                  Math.round(Number(predictionPayload.window_size)||1));
                const historyCount = Math.max(0, Math.min(windowSize,
                  Math.round(Number(predictionPayload.history_samples)||0)));
                let statusText = qtOpText('forcePredictionWaiting');
                if (state === 'error') statusText =
                  `${qtOpText('forcePredictionError')}: ${predictionPayload.message||''}`;
                else if (state === 'loading') statusText = qtOpText('forcePredictionLoading');
                else if (state === 'ready') {
                  const warmup = historyCount < windowSize
                    ? ` · ${qtOpText('forcePredictionWarmup')
                        .replace('{n}',historyCount).replace('{window}',windowSize)}` : '';
                  const warning = outOfRange ? ` · ${qtOpText('forcePredictionOutOfRange')}` : '';
                  statusText = `${qtOpText('forcePredictionReady')} · `+
                    `${Math.round(Number.isFinite(hz)?hz:0)} Hz · `+
                    `${Number.isFinite(latency)?latency.toFixed(1):'--'} ms${warmup}${warning}`;
                }
                predictionValues.forEach(element => {
                  element.title = statusText;
                  element.classList.toggle('error', state === 'error');
                  element.classList.toggle('warning', outOfRange);
                });
              };
              window.__qtApplyForcePredictionLanguage = () => {
                if (predictionModeLabel)
                  predictionModeLabel.textContent = qtOpText('forcePredictionMode');
                if (predictionMode) {
                  const fusion = predictionMode.querySelector('option[value="fusion"]');
                  const separate = predictionMode.querySelector('option[value="separate"]');
                  if (fusion) fusion.textContent = qtOpText('forcePredictionFusion');
                  if (separate) separate.textContent = qtOpText('forcePredictionSeparate');
                }
                predictionValues.forEach(element => {
                  const label = element.querySelector('small');
                  if (label) label.textContent = qtOpText('forcePredictionInline');
                });
                renderForcePrediction();
              };
              if (predictionMode) predictionMode.onchange = () => {
                if (typeof pyHost === 'undefined' || !pyHost ||
                    typeof pyHost.setForcePredictionMode !== 'function') {
                  renderForcePrediction({state:'error',
                    message:'Force prediction mode control is unavailable.'});
                  return;
                }
                predictionMode.disabled = true;
                pyHost.setForcePredictionMode(predictionMode.value, text => {
                  predictionMode.disabled = false;
                  try { renderForcePrediction(JSON.parse(String(text||'{}'))); }
                  catch (error) { renderForcePrediction({state:'error',message:error.message}); }
                });
              };
              const connectForcePredictionBridge = () => {
                if (typeof pyHost === 'undefined' || !pyHost || !pyHost.force_prediction_status) {
                  setTimeout(connectForcePredictionBridge,200); return;
                }
                if (!window.__qtForcePredictionBridge) {
                  window.__qtForcePredictionBridge = true;
                  pyHost.force_prediction_status.connect(text => {
                    try { renderForcePrediction(JSON.parse(String(text||'{}'))); }
                    catch (error) { renderForcePrediction({state:'error',message:error.message}); }
                  });
                }
                if (typeof pyHost.forcePredictionStatus === 'function')
                  pyHost.forcePredictionStatus(text => {
                    try { renderForcePrediction(JSON.parse(String(text||'{}'))); } catch (_) {}
                  });
              };
              connectForcePredictionBridge();
              renderForcePrediction();
            }
            // 原面板仍是 debug 的完整控制台；正式操作区只显示下面这套精简入口。
            legacySection.classList.add('dbg');
            if (!document.getElementById('qt-operation-mode-style')) {
              const style = document.createElement('style');
              style.id = 'qt-operation-mode-style';
              style.textContent =
                '#qt-operation-nano25{padding-top:10px;padding-bottom:10px}'+
                '#qt-operation-nano25 #qt-nano25-toggle{padding:4px 8px;font-size:10.5px;font-weight:650}'+
                '#qt-operation-nano25 #qt-force-prediction-collapse{padding:4px 7px;font-size:10.5px;font-weight:650}'+
                '#qt-operation-nano25 #qt-nano25-rate{margin-right:7px;white-space:nowrap}'+
                '#qt-operation-nano25 .qt-force-mode{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}'+
                '#qt-operation-nano25 #qt-force-prediction-mode{max-width:118px;padding:2px 4px;font-size:10px}'+
                '#qt-operation-nano25 #qt-nano25-values{display:grid;grid-template-columns:minmax(0,1fr);gap:3px}'+
                '#qt-operation-nano25 .qt-nano25-value{display:grid;grid-template-columns:54px minmax(0,1fr) 34px;'+
                  'align-items:center;padding:4px 6px;border-radius:4px;background:var(--surface);'+
                  'border-left:3px solid var(--accent);font:11px var(--mono)}'+
                '#qt-operation-nano25 .qt-nano25-value i{font-style:normal;font-weight:650;color:var(--ink-2)}'+
                '#qt-operation-nano25 .qt-nano25-value b{text-align:right;color:var(--ink);font-weight:650}'+
                '#qt-operation-nano25 .qt-nano25-value em{text-align:right;font-style:normal;color:var(--muted)}'+
                '#qt-operation-nano25 .qt-nano25-reading{display:flex;justify-content:flex-end;align-items:baseline;min-width:0}'+
                '#qt-operation-nano25 #qt-nano25-state{margin:5px 0 0}'+
                '#qt-operation-nano25.qt-force-prediction-hidden .qt-force-mode,'+
                  '#qt-operation-nano25.qt-force-prediction-hidden .qt-force-prediction-inline{display:none!important}'+
                '#qt-operation-nano25.qt-nano25-error .qt-nano25-value,'+
                  '#qt-operation-nano25.qt-nano25-error #qt-nano25-state{color:var(--bad);border-color:var(--bad)}'+
                '#qt-operation-nano25 .qt-force-prediction-inline{display:inline-flex;justify-content:flex-end;align-items:baseline;'+
                  'gap:4px;margin-right:9px;color:var(--good);white-space:nowrap}'+
                '#qt-operation-nano25 .qt-force-prediction-inline small{font:9px var(--sans);color:var(--muted)}'+
                '#qt-operation-nano25 .qt-force-prediction-inline strong{min-width:40px;text-align:right;font-weight:650}'+
                '#qt-operation-nano25 .qt-force-prediction-inline.error strong{color:var(--bad)}'+
                '#qt-operation-nano25 .qt-force-prediction-inline.warning strong{color:var(--warn)}'+
                '#qt-operation-modes{padding-top:12px;padding-bottom:12px}'+
                '#qt-operation-modes .qt-op-mode-tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:9px}'+
                '#qt-operation-modes .qt-op-mode-tab{padding:7px 4px;font-weight:650}'+
                '#qt-operation-modes .qt-op-mode-tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}'+
                '#qt-operation-modes .qt-op-mode-pane[hidden]{display:none!important}'+
                '#qt-operation-modes .qt-op-auto-modes{display:grid;gap:6px}'+
                '#qt-operation-modes .qt-op-auto-mode{display:grid;grid-template-columns:58px minmax(0,1fr);align-items:center;'+
                'gap:5px;padding:5px;border:1px solid var(--line);border-radius:6px;background:var(--surface-2)}'+
                '#qt-operation-modes .qt-op-auto-mode[data-selected="true"]{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}'+
                '#qt-operation-modes .qt-op-auto-select{height:100%;min-height:34px;padding:5px;font-size:12px;font-weight:700}'+
                '#qt-operation-modes .qt-op-auto-mode[data-selected="true"] .qt-op-auto-select{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}'+
                '#qt-operation-modes .qt-op-auto-config{display:flex;align-items:center;justify-content:flex-end;gap:4px;min-width:0;flex-wrap:nowrap}'+
                '#qt-operation-modes .qt-op-param{display:inline-flex;align-items:center;gap:2px;font-size:10.5px;white-space:nowrap}'+
                '#qt-operation-modes .qt-op-param input[type="text"]{width:41px;padding:3px 2px;text-align:center;font-family:var(--mono);'+
                'font-size:10.5px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px}'+
                '#qt-operation-modes .qt-op-loop{display:inline-flex;align-items:center;gap:2px;font-size:10.5px;white-space:nowrap}'+
                '#qt-operation-modes .qt-op-loop input{accent-color:var(--accent);margin:0}'+
                '#qt-operation-modes .qt-op-limit{display:inline-flex;align-items:center;gap:2px;white-space:nowrap}'+
                '#qt-operation-modes .qt-op-limit input{width:38px;padding:3px 2px;text-align:center;font:10px var(--mono);'+
                'background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px}'+
                '#qt-operation-modes .qt-op-limit select{width:43px;padding:3px 1px;font:9.5px var(--mono);'+
                'background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px}'+
                '#qt-operation-modes #qt-op-llm-config-btn,#qt-operation-modes #qt-op-clear-memory,'+
                  '#qt-operation-modes #qt-op-save-process,#qt-operation-modes #qt-op-load-replay,'+
                  '#qt-operation-modes #qt-op-test-modified,#qt-operation-modes #qt-op-exit-replay{padding:4px 7px;font-size:10.5px;white-space:nowrap}'+
                '#qt-operation-modes #qt-op-llm-config-btn[hidden],#qt-operation-modes #qt-op-clear-memory[hidden],'+
                  '#qt-operation-modes #qt-op-save-process[hidden],#qt-operation-modes #qt-op-load-replay[hidden],'+
                  '#qt-operation-modes #qt-op-test-modified[hidden],#qt-operation-modes #qt-op-exit-replay[hidden],#qt-operation-modes #qt-op-replay-status[hidden]{display:none!important}'+
                '#qt-operation-modes .qt-op-replay-actions{margin-top:5px;gap:4px}'+
                '#qt-operation-modes #qt-op-replay-status{font-size:10px;color:var(--good);font-weight:650;line-height:1.3}'+
                '#qt-operation-modes .qt-op-sequence{font:11px var(--mono);color:var(--ink-2);white-space:nowrap}'+
                '#qt-operation-modes .qt-op-mode-actions{display:grid;grid-template-columns:1fr;gap:6px;margin-top:8px}'+
                '#qt-operation-modes .qt-op-mode-actions .btn{padding:8px 6px;font-weight:650}'+
                '#qt-op-llm-feedback[hidden]{display:none!important}'+
                '#qt-op-llm-feedback{display:grid;gap:5px;padding:7px;border:1px solid var(--line);border-radius:6px;background:var(--surface-2)}'+
                '#qt-op-llm-feedback-status{font-size:10.5px;line-height:1.35;color:var(--muted)}'+
                '#qt-op-llm-feedback-status.good{color:var(--good)}#qt-op-llm-feedback-status.bad{color:var(--bad)}'+
                '#qt-op-llm-feedback-audit:empty{display:none}'+
                '#qt-op-llm-feedback-audit{display:grid;gap:3px;max-height:92px;overflow-y:auto;padding:5px 6px;'+
                  'border:1px solid var(--line);border-radius:5px;background:var(--bg);font-size:9.5px;line-height:1.35}'+
                '#qt-op-llm-feedback-audit .applied{color:var(--good)}'+
                '#qt-op-llm-feedback-audit .queued{color:var(--muted)}'+
                '#qt-op-llm-feedback-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:5px}'+
                '#qt-op-llm-feedback-input{min-width:0;padding:6px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:5px;font-size:10.5px}'+
                '#qt-op-llm-feedback-send{padding:6px 8px;white-space:nowrap;font-size:10.5px}'+
                '#qt-operation-modes .qt-op-teach{display:grid;gap:7px}'+
                '#qt-operation-modes .qt-op-teach-row{display:flex;align-items:center;gap:7px;flex-wrap:wrap}'+
                '#qt-operation-modes .qt-op-teach-execution{gap:4px;flex-wrap:nowrap}'+
                '#qt-operation-modes .qt-op-teach-row .btn{padding:7px 10px;font-weight:650}'+
                '#qt-operation-modes #qt-op-default-select{padding:6px 7px;background:var(--bg);color:var(--ink);'+
                  'border:1px solid var(--line);border-radius:6px;font-size:11px}'+
                '#qt-operation-modes .qt-op-teach-count{margin-left:auto;font:11px var(--mono);color:var(--muted)}'+
                '#qt-operation-modes .qt-op-teach-list{display:grid;gap:4px;max-height:180px;overflow-y:auto;padding-right:2px}'+
                '#qt-operation-modes .qt-op-teach-empty{padding:10px;border:1px dashed var(--line);border-radius:5px;'+
                'text-align:center;font-size:11.5px;color:var(--muted)}'+
                '#qt-operation-modes .qt-op-teach-point{display:grid;grid-template-columns:28px minmax(0,1fr) auto 24px;align-items:center;gap:7px;'+
                'padding:6px 7px;border:1px solid var(--line);border-radius:5px;background:var(--surface-2);font-size:11px}'+
                '#qt-operation-modes .qt-op-teach-point-index{font:600 11px var(--mono);color:var(--accent-ink)}'+
                '#qt-operation-modes .qt-op-teach-point-pose{font:11px var(--mono);color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'+
                '#qt-operation-modes .qt-op-teach-point-time{font:10.5px var(--mono);color:var(--muted);white-space:nowrap}'+
                '#qt-operation-modes .qt-op-teach-delete{width:24px;height:24px;padding:0;border-radius:4px;font-size:13px;line-height:22px}'+
                '#qt-operation-modes .qt-op-teach-point.warn{border-color:var(--warn)}'+
                '#qt-operation-modes .qt-op-controller{display:grid;grid-template-columns:62px minmax(0,1fr);align-items:center;gap:7px;'+
                'font-size:12px;color:var(--ink-2)}'+
                '#qt-operation-modes .qt-op-controller select{min-width:0;padding:5px 6px;background:var(--bg);color:var(--ink);'+
                'border:1px solid var(--line);border-radius:5px}'+
                '#qt-operation-modes .qt-op-placeholder{min-height:62px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:6px;color:var(--muted);font-size:12px}'+
                '#qt-operation-modes .qt-op-test{display:grid;gap:8px}'+
                '#qt-operation-modes .qt-op-test-actions{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px}'+
                '#qt-operation-modes .qt-op-test-actions .btn{padding:8px 7px;font-weight:650}'+
                '#qt-operation-modes .qt-op-test-note{font-size:11px;color:var(--muted)}'+
                '#qt-operation-modes .qt-op-test-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}'+
                '#qt-operation-modes .qt-op-test-card{padding:7px;border:1px solid var(--line);border-radius:6px;background:var(--surface-2);min-width:0}'+
                '#qt-operation-modes .qt-op-test-card header{display:flex;padding:0;border:0;background:transparent;gap:5px;font-size:11.5px;font-weight:650}'+
                '#qt-operation-modes .qt-op-test-card header span{margin-left:auto;font-size:10px;color:var(--muted)}'+
                '#qt-operation-modes .qt-op-test-card small{display:block;margin-top:3px;color:var(--muted);font-size:10px;line-height:1.3;word-break:break-word}'+
                '#qt-operation-modes .qt-op-test-card.running{border-color:var(--accent)}'+
                '#qt-operation-modes .qt-op-test-card.pass{border-color:var(--good);background:color-mix(in srgb,var(--good) 8%,var(--surface-2))}'+
                '#qt-operation-modes .qt-op-test-card.pass header span{color:var(--good)}'+
                '#qt-operation-modes .qt-op-test-card.warn{border-color:var(--warn)}'+
                '#qt-operation-modes .qt-op-test-card.warn header span{color:var(--warn)}'+
                '#qt-operation-modes .qt-op-test-card.fail{border-color:var(--bad)}'+
                '#qt-operation-modes .qt-op-test-card.fail header span{color:var(--bad)}'+
                '#qt-operation-modes .qt-op-test-overall{padding:7px;border-radius:6px;border:1px solid var(--line);font-size:11.5px;color:var(--muted)}'+
                '#qt-operation-modes .qt-op-test-overall.pass{border-color:var(--good);color:var(--good);font-weight:700}'+
                '#qt-operation-modes .qt-op-test-overall.warn{border-color:var(--warn);color:var(--warn);font-weight:700}'+
                '#qt-operation-modes .qt-op-test-overall.fail{border-color:var(--bad);color:var(--bad);font-weight:700}'+
                '#qt-operation-modes .qt-op-mode-status{margin-top:7px;font-size:11px;color:var(--muted)}'+
                '#qt-operation-modes #qt-op-vent{padding:1px 8px;font-size:11px;font-weight:500;text-transform:none;letter-spacing:0}'+
                '#qt-op-llm-modal[hidden]{display:none!important}'+
                '#qt-op-llm-modal{position:fixed;inset:0;z-index:12000;display:grid;place-items:center;padding:20px;'+
                'background:rgba(8,14,24,.48);backdrop-filter:blur(2px)}'+
                '#qt-op-llm-dialog{width:min(430px,calc(100vw - 36px));padding:16px;border:1px solid var(--line);'+
                'border-radius:9px;background:var(--surface);box-shadow:0 18px 60px rgba(0,0,0,.28);color:var(--ink)}'+
                '#qt-op-llm-dialog h3{margin:0 0 12px;font-size:15px}'+
                '#qt-op-llm-dialog .qt-op-llm-grid{display:grid;grid-template-columns:132px minmax(0,1fr);gap:9px 10px;align-items:center}'+
                '#qt-op-llm-dialog label{font-size:11.5px;color:var(--ink-2)}'+
                '#qt-op-llm-dialog input[type="text"],#qt-op-llm-dialog input[type="password"],#qt-op-llm-dialog input[type="number"],#qt-op-llm-dialog select{'+
                'width:100%;box-sizing:border-box;padding:6px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:5px}'+
                '#qt-op-llm-dialog .qt-op-llm-auth-controls{display:flex;align-items:center;gap:6px;min-width:0}'+
                '#qt-op-llm-dialog .qt-op-llm-auth-controls select,#qt-op-llm-dialog .qt-op-llm-auth-controls input{min-width:0;flex:1}'+
                '#qt-op-llm-dialog .qt-op-llm-auth-controls .btn{padding:6px 8px;white-space:nowrap;font-size:10.5px}'+
                '#qt-op-llm-api-label[hidden],#qt-op-llm-api-controls[hidden]{display:none!important}'+
                '#qt-op-llm-auth-status{font-size:10.5px;color:var(--muted);line-height:1.3}'+
                '#qt-op-llm-auth-status-controls{flex-wrap:wrap}#qt-op-llm-auth-status-controls #qt-op-llm-auth-status{flex:1;min-width:120px}'+
                '#qt-op-llm-auth-status.good{color:var(--good)}#qt-op-llm-auth-status.bad{color:var(--bad)}'+
                '#qt-op-llm-dialog .qt-op-llm-auto{grid-column:1/-1;display:flex;align-items:center;gap:6px}'+
                '#qt-op-llm-status{margin-top:10px;font-size:10.5px;line-height:1.4;color:var(--muted)}'+
                '#qt-op-llm-status.good{color:var(--good)}#qt-op-llm-status.bad{color:var(--bad)}'+
                '#qt-op-llm-dialog .qt-op-llm-actions{display:flex;justify-content:flex-end;gap:7px;margin-top:12px}';
              document.head.appendChild(style);
            }
            const panel = document.createElement('section');
            panel.id = 'qt-operation-modes';
            panel.className = 'op-only';
            panel.innerHTML =
              '<h2><span id="qt-op-modes-title">模式</span> <span class="sp"></span><button class="btn" id="qt-op-vent">全机放气</button></h2>'+ 
              '<div class="qt-op-mode-tabs" role="tablist" aria-label="操作模式">'+
                '<button class="btn qt-op-mode-tab" role="tab" data-tab="auto" aria-selected="true">自动</button>'+ 
                '<button class="btn qt-op-mode-tab" role="tab" data-tab="teach" aria-selected="false">示教</button>'+ 
                '<button class="btn qt-op-mode-tab" role="tab" data-tab="test" aria-selected="false">测试</button>'+ 
              '</div>'+ 
              '<div class="qt-op-mode-pane" data-pane="auto">'+
                '<div class="qt-op-auto-modes">'+
                  '<div class="qt-op-auto-mode" data-fixed-mode="mode1" data-selected="true">'+
                    '<button class="btn qt-op-auto-select">模式 1</button><div class="qt-op-auto-config">'+
                      '<label class="qt-op-param">Y <input type="text" inputmode="decimal" id="qt-op-m1-y" value="+60" aria-label="模式1 Y目标角度">°</label>'+ 
                      '<label class="qt-op-param">X <input type="text" inputmode="decimal" id="qt-op-m1-x" value="-20" aria-label="模式1 X目标角度">°</label>'+ 
                      '<label class="qt-op-loop"><input type="checkbox" id="qt-op-m1-loop">循环</label>'+ 
                      '<label class="qt-op-limit"><input type="number" id="qt-op-m1-limit" min="1" max="999" step="1" value="1" placeholder="∞" title="留空表示无限运行"><select id="qt-op-m1-unit"><option value="runs">圈</option><option value="min">mins</option></select></label>'+ 
                    '</div></div>'+ 
                  '<div class="qt-op-auto-mode" data-fixed-mode="mode2" data-selected="false">'+
                    '<button class="btn qt-op-auto-select">模式 2</button><div class="qt-op-auto-config">'+
                      '<label class="qt-op-param">Y+ <input type="text" inputmode="decimal" id="qt-op-m2-y1" value="+60" aria-label="模式2 Y正向目标角度">°</label>'+ 
                      '<label class="qt-op-param">Y− <input type="text" inputmode="decimal" id="qt-op-m2-y2" value="-20" aria-label="模式2 Y反向目标角度">°</label>'+ 
                      '<label class="qt-op-loop"><input type="checkbox" id="qt-op-m2-loop">循环</label>'+ 
                      '<label class="qt-op-limit"><input type="number" id="qt-op-m2-limit" min="1" max="999" step="1" value="1" placeholder="∞" title="留空表示无限运行"><select id="qt-op-m2-unit"><option value="runs">圈</option><option value="min">mins</option></select></label>'+ 
                    '</div></div>'+ 
                  '<div class="qt-op-auto-mode" data-fixed-mode="mode3" data-selected="false">'+
                    '<button class="btn qt-op-auto-select">模式 3</button><div class="qt-op-auto-config">'+
                      '<span class="qt-op-sequence">15 → 23 → 48 → 67</span>'+ 
                      '<label class="qt-op-loop"><input type="checkbox" id="qt-op-m3-loop">循环</label>'+ 
                      '<label class="qt-op-limit"><input type="number" id="qt-op-m3-limit" min="1" max="999" step="1" value="1" placeholder="∞" title="留空表示无限运行"><select id="qt-op-m3-unit"><option value="runs">圈</option><option value="min">mins</option></select></label>'+ 
                    '</div></div>'+ 
                  '<div class="qt-op-auto-mode" data-fixed-mode="mode4" data-selected="false">'+
                    '<button class="btn qt-op-auto-select">模式 4</button><div class="qt-op-auto-config">'+
                      '<label class="qt-op-param">Y <input type="text" inputmode="decimal" id="qt-op-m4-y" value="-30" aria-label="模式4 Y目标角度">°</label>'+ 
                      '<label class="qt-op-param">φ <input type="text" inputmode="decimal" id="qt-op-m4-phi" value="+20" aria-label="模式4 φ目标角度">°</label>'+ 
                      '<label class="qt-op-loop"><input type="checkbox" id="qt-op-m4-loop">循环</label>'+ 
                      '<label class="qt-op-limit"><input type="number" id="qt-op-m4-limit" min="1" max="999" step="1" value="1" placeholder="∞" title="留空表示无限运行"><select id="qt-op-m4-unit"><option value="runs">圈</option><option value="min">mins</option></select></label>'+ 
                    '</div></div>'+ 
                  '<div class="qt-op-auto-mode" data-fixed-mode="mode5" data-selected="false">'+
                    '<button class="btn qt-op-auto-select">模式 5</button><div class="qt-op-auto-config">'+
                      '<span class="qt-op-sequence" title="主侧升压1.5s、降压1s；对侧0.5～1s升至3000mV，保持至1.5s后共同降压">15 → 23 → 48 → 67 ＋ 对侧</span>'+
                      '<label class="qt-op-loop"><input type="checkbox" id="qt-op-m5-loop">循环</label>'+
                      '<label class="qt-op-limit"><input type="number" id="qt-op-m5-limit" min="1" max="999" step="1" value="1" placeholder="∞" title="留空表示无限运行"><select id="qt-op-m5-unit"><option value="runs">圈</option><option value="min">mins</option></select></label>'+
                    '</div></div>'+
                '</div>'+ 
                '<div class="qt-op-mode-actions"><button class="btn primary" id="qt-op-auto-run">开始自动运行</button></div>'+ 
              '</div>'+ 
              '<div class="qt-op-mode-pane" data-pane="teach" hidden>'+
                '<div class="qt-op-teach">'+
                  '<div class="qt-op-teach-row">'+
                    '<button class="btn primary" id="qt-op-record">＋ 新增当前点</button>'+ 
                    '<select id="qt-op-default-select" title="选择预设示教轨迹">'+
                      '<option value="1" selected>Default 1</option>'+ 
                      '<option value="2">Default 2</option>'+ 
                      '<option value="3">Default 3</option>'+ 
                    '</select>'+ 
                    '<button class="btn" id="qt-op-load-default" type="button">载入</button>'+ 
                    '<button class="btn" id="qt-op-clear">清空全部</button>'+ 
                    '<span class="qt-op-teach-count" id="qt-op-teach-count">已记录 0 个点</span>'+ 
                  '</div>'+ 
                  '<div class="qt-op-teach-list" id="qt-op-teach-list">'+
                    '<div class="qt-op-teach-empty">还没有记录点</div>'+ 
                  '</div>'+ 
                  '<label class="qt-op-controller">控制器'+
                    '<select id="qt-op-teach-controller" aria-label="示教轨迹控制器">'+
                      '<option value="mpc" selected>MPC</option>'+ 
                      '<option value="pi">PID</option>'+ 
                      '<option value="mpc_codex">LLM + MPC</option>'+ 
                    '</select>'+ 
                  '</label>'+ 
                  '<div class="qt-op-teach-row qt-op-teach-execution">'+
                    '<label class="qt-op-loop"><input type="checkbox" id="qt-op-teach-loop">循环执行</label>'+ 
                    '<label class="qt-op-limit"><input type="number" id="qt-op-teach-limit" min="1" max="999" step="1" value="1" placeholder="∞" title="留空表示无限运行"><select id="qt-op-teach-unit"><option value="runs">圈</option><option value="min">mins</option></select></label>'+ 
                    '<button class="btn" id="qt-op-llm-config-btn" type="button" hidden>LLM配置</button>'+
                    '<button class="btn" id="qt-op-clear-memory" type="button" hidden>清除记忆</button>'+
                  '</div>'+ 
                  '<div class="qt-op-teach-row qt-op-replay-actions">'+
                    '<button class="btn" id="qt-op-save-process" type="button" hidden>保存10轮过程</button>'+
                    '<button class="btn" id="qt-op-load-replay" type="button" hidden>载入过程回放</button>'+
                    '<button class="btn" id="qt-op-load-replay2" type="button" hidden>Load 2</button>'+
                    '<button class="btn" id="qt-op-test-modified" type="button" hidden>测试修改版（单轮）</button>'+
                    '<button class="btn" id="qt-op-exit-replay" type="button" hidden>退出回放</button>'+
                    '<span id="qt-op-replay-status" hidden></span>'+
                  '</div>'+ 
                  '<div class="qt-op-mode-actions"><button class="btn" id="qt-op-teach-run">开始执行示教轨迹</button></div>'+ 
                  '<div id="qt-op-llm-feedback" hidden>'+ 
                    '<div id="qt-op-llm-feedback-status">仅在LLM推理时可提交</div>'+ 
                    '<div id="qt-op-llm-feedback-audit"></div>'+
                    '<div id="qt-op-llm-feedback-row"><input id="qt-op-llm-feedback-input" type="text" maxlength="1200" placeholder="例如：φ上升过快，降低上升段正向补偿并提前卸载">'+ 
                    '<button class="btn" id="qt-op-llm-feedback-send" type="button" disabled>提交意见</button></div>'+ 
                  '</div>'+ 
                '</div>'+ 
              '</div>'+ 
              '<div id="qt-op-llm-modal" hidden role="dialog" aria-modal="true" aria-labelledby="qt-op-llm-title">'+
                '<div id="qt-op-llm-dialog">'+
                  '<h3 id="qt-op-llm-title">LLM 配置</h3>'+ 
                  '<div class="qt-op-llm-grid">'+
                    '<label for="qt-op-llm-provider" id="qt-op-llm-provider-label">提供商</label>'+ 
                    '<select id="qt-op-llm-provider"><option value="gpt">GPT</option><option value="claude">Claude</option></select>'+ 
                    '<label for="qt-op-llm-auth-method" id="qt-op-llm-auth-method-label">认证方式</label>'+ 
                    '<div class="qt-op-llm-auth-controls"><select id="qt-op-llm-auth-method"><option value="account">ChatGPT账号</option><option value="api">API Key</option></select><button class="btn" id="qt-op-llm-auth-action" type="button">登录ChatGPT</button></div>'+ 
                    '<label for="qt-op-llm-api-key" id="qt-op-llm-api-label" hidden>API Key</label>'+ 
                    '<div class="qt-op-llm-auth-controls" id="qt-op-llm-api-controls" hidden><input type="password" id="qt-op-llm-api-key" autocomplete="off" spellcheck="false" placeholder="只用于本次登录，不写入项目"></div>'+ 
                    '<label id="qt-op-llm-auth-status-label">登录状态</label>'+ 
                    '<div class="qt-op-llm-auth-controls" id="qt-op-llm-auth-status-controls"><span id="qt-op-llm-auth-status">—</span><button class="btn" id="qt-op-llm-install-claude" type="button" hidden>安装Claude Code</button><button class="btn" id="qt-op-llm-auth-refresh" type="button">刷新状态</button><button class="btn" id="qt-op-llm-signout" type="button">退出登录</button></div>'+ 
                     '<label for="qt-op-llm-model" id="qt-op-llm-model-label">模型</label>'+ 
                     '<select id="qt-op-llm-model"></select>'+ 
                     '<label for="qt-op-llm-profile" id="qt-op-llm-profile-label">补偿器设计</label>'+ 
                     '<select id="qt-op-llm-profile"><option value="base">Base（矩阵 PID）</option><option value="plus">Plus（压力感知分层）</option><option value="pro">Pro（自由设计）</option></select>'+ 
                     '<label for="qt-op-llm-effort" id="qt-op-llm-effort-label">推理强度</label>'+ 
                    '<select id="qt-op-llm-effort"><option value="low">low</option><option value="medium" selected>medium</option><option value="high">high</option><option value="xhigh">xhigh</option></select>'+ 
                    '<label for="qt-op-llm-timeout" id="qt-op-llm-timeout-label">分析超时（秒）</label>'+ 
                    '<input type="number" id="qt-op-llm-timeout" min="30" max="900" step="10" value="300">'+ 
                    '<label for="qt-op-llm-reuse-mse" id="qt-op-llm-reuse-mse-label">沿用阈值（MSE deg²）</label>'+ 
                    '<input type="number" id="qt-op-llm-reuse-mse" min="0" max="10000" step="0.1" value="15.0">'+ 
                    '<label class="qt-op-llm-auto" id="qt-op-llm-fast-row"><input type="checkbox" id="qt-op-llm-fast"><span id="qt-op-llm-fast-label">Fast Mode（更多额度，更快）</span></label>'+ 
                    '<label class="qt-op-llm-auto"><input type="checkbox" id="qt-op-llm-auto" checked><span id="qt-op-llm-auto-label">每轮结束后自动生成下一版</span></label>'+ 
                  '</div>'+ 
                  '<div id="qt-op-llm-status">—</div>'+ 
                  '<div class="qt-op-llm-actions"><button class="btn" id="qt-op-llm-cancel" type="button">取消</button><button class="btn primary" id="qt-op-llm-save" type="button">保存配置</button></div>'+ 
                '</div>'+ 
              '</div>'+ 
              '<div class="qt-op-mode-pane" data-pane="test" hidden>'+ 
                '<div class="qt-op-test">'+
                  '<div class="qt-op-test-actions">'+
                    '<button class="btn primary" id="qt-op-quick-start">开始快速综合检测</button>'+ 
                    '<button class="btn" id="qt-op-quick-stop" disabled>停止检测</button>'+ 
                  '</div>'+ 
                  '<div class="qt-op-test-grid">'+
                    '<div class="qt-op-test-card" data-quick-key="environment"><header>软件环境<span>待检测</span></header><small>Python版本、运行组件与项目文件</small></div>'+ 
                    '<div class="qt-op-test-card" data-quick-key="pressure"><header>压力传感器<span>待检测</span></header><small>零漂、响应和六路一致性</small></div>'+ 
                    '<div class="qt-op-test-card" data-quick-key="muscle"><header>肌肉收缩<span>待检测</span></header><small>20路在线与六组长度变化</small></div>'+ 
                    '<div class="qt-op-test-card" data-quick-key="mapping"><header>阀/锁/映射<span>待检测</span></header><small>六路锁定和拮抗通道误锁</small></div>'+ 
                    '<div class="qt-op-test-card" data-quick-key="imu"><header>IMU<span>待检测</span></header><small>六方向运动响应与方向检查</small></div>'+ 
                    '<div class="qt-op-test-card" data-quick-key="speed"><header>最大速度<span>待检测</span></header><small>从同批扫描快速估计六方向P95</small></div>'+ 
                  '</div>'+ 
                  '<div class="qt-op-test-overall" id="qt-op-test-overall">尚未检测</div>'+ 
                '</div>'+ 
              '</div>'+ 
              '<div class="qt-op-mode-status" id="qt-op-mode-status">已选择模式 1</div>';
            nanoPanel.insertAdjacentElement('afterend', panel);

            const tabs = Array.from(panel.querySelectorAll('.qt-op-mode-tab'));
            const panes = Array.from(panel.querySelectorAll('.qt-op-mode-pane'));
            const autoModes = Array.from(panel.querySelectorAll('.qt-op-auto-mode'));
            const run = panel.querySelector('#qt-op-auto-run');
            const opRecord = panel.querySelector('#qt-op-record');
            const opDefaultSelect = panel.querySelector('#qt-op-default-select');
            const opLoadDefault = panel.querySelector('#qt-op-load-default');
            const opClear = panel.querySelector('#qt-op-clear');
            const teachController = panel.querySelector('#qt-op-teach-controller');
            const teachLoop = panel.querySelector('#qt-op-teach-loop');
            const teachLimit = panel.querySelector('#qt-op-teach-limit');
            const teachUnit = panel.querySelector('#qt-op-teach-unit');
            const teachRun = panel.querySelector('#qt-op-teach-run');
            const llmConfigButton = panel.querySelector('#qt-op-llm-config-btn');
            const clearMemoryButton = panel.querySelector('#qt-op-clear-memory');
            const saveProcessButton = panel.querySelector('#qt-op-save-process');
            const loadReplayButton = panel.querySelector('#qt-op-load-replay');
            const loadReplay2Button = panel.querySelector('#qt-op-load-replay2');
            const debugLoad2Timed = document.createElement('button');
            debugLoad2Timed.className = 'btn';
            debugLoad2Timed.type = 'button';
            debugLoad2Timed.id = 'qt-debug-auto123-30min';
            debugLoad2Timed.textContent = 'Automatic 1→2→3 循环30分钟';
            debugLoad2Timed.title = '按当前Automatic目标，各模式执行一次后切换，1→2→3重复；满30分钟完成当前模式后停止。';
            debugLoad2Timed.title = '1→2→3各执行一次算一轮；整套循环共计30分钟，完成当前整轮后停止。';
            legacySection.appendChild(debugLoad2Timed);
            loadReplayButton.title = 'Load 1：原对比过程；第5轮起V2在4～7秒+50mV';
            loadReplay2Button.title = 'Load 2：第5轮起V2在5～7.9秒按角度和速度落后程度补偿，最多+50mV';
            const testModifiedButton = panel.querySelector('#qt-op-test-modified');
            const exitReplayButton = panel.querySelector('#qt-op-exit-replay');
            const replayStatus = panel.querySelector('#qt-op-replay-status');
            const llmFeedbackPanel = panel.querySelector('#qt-op-llm-feedback');
            const llmFeedbackStatus = panel.querySelector('#qt-op-llm-feedback-status');
            const llmFeedbackAudit = panel.querySelector('#qt-op-llm-feedback-audit');
            const llmFeedbackInput = panel.querySelector('#qt-op-llm-feedback-input');
            const llmFeedbackSend = panel.querySelector('#qt-op-llm-feedback-send');
            const llmModal = panel.querySelector('#qt-op-llm-modal');
            const llmTitle = panel.querySelector('#qt-op-llm-title');
            const llmProvider = panel.querySelector('#qt-op-llm-provider');
            const llmModel = panel.querySelector('#qt-op-llm-model');
            const llmProfile = panel.querySelector('#qt-op-llm-profile');
            const llmEffort = panel.querySelector('#qt-op-llm-effort');
            const llmTimeout = panel.querySelector('#qt-op-llm-timeout');
            const llmReuseMse = panel.querySelector('#qt-op-llm-reuse-mse');
            const llmFast = panel.querySelector('#qt-op-llm-fast');
            const llmAuto = panel.querySelector('#qt-op-llm-auto');
            const llmStatus = panel.querySelector('#qt-op-llm-status');
            const llmAuthMethod = panel.querySelector('#qt-op-llm-auth-method');
            const llmApiLabel = panel.querySelector('#qt-op-llm-api-label');
            const llmApiControls = panel.querySelector('#qt-op-llm-api-controls');
            const llmApiKey = panel.querySelector('#qt-op-llm-api-key');
            const llmAuthStatus = panel.querySelector('#qt-op-llm-auth-status');
            const llmAuthAction = panel.querySelector('#qt-op-llm-auth-action');
            const llmAuthRefresh = panel.querySelector('#qt-op-llm-auth-refresh');
            const llmInstallClaude = panel.querySelector('#qt-op-llm-install-claude');
            const llmSignOut = panel.querySelector('#qt-op-llm-signout');
            const llmSave = panel.querySelector('#qt-op-llm-save');
            const llmCancel = panel.querySelector('#qt-op-llm-cancel');
            const opTeachCount = panel.querySelector('#qt-op-teach-count');
            const opTeachList = panel.querySelector('#qt-op-teach-list');
            const opVent = panel.querySelector('#qt-op-vent');
            const status = panel.querySelector('#qt-op-mode-status');
            const quickStart = panel.querySelector('#qt-op-quick-start');
            const quickStop = panel.querySelector('#qt-op-quick-stop');
            const quickOverall = panel.querySelector('#qt-op-test-overall');
            const quickCards = Array.from(panel.querySelectorAll('[data-quick-key]'));
            let activeTab = 'auto';
            let selectedMode = fixedMode.value || 'mode1';
            const automaticModeOrder = ['mode1','mode2','mode3','mode4','mode5'];
            const selectedModes = new Set();
            let autoQueue = null;
            let renderedTeachSignature = '';
            let quickRunning = false;
            let quickEnvironment = null;
            let quickEnvironmentAutoAttempted = false;
            const quickEnvironmentCacheKey = 'tro-software-environment-v0.1-passed';
            let recordingLaunchPending = false;
            let recordingStartGuardToken = 0;
            let previousExecutionRunning = false;
            let clearMemoryRunning = false;
            let llmReplayActive = false;
            let selectedReplayVariant = 1;
            let llmReplayInfo = null;
            let llmReplayBusy = false;
            let llmStatusState = 'idle';
            let llmStatusDetail = '';
            let llmFeedbackRunning = false;
            let llmFeedbackWaiting = false;
            let llmFeedbackHeld = false;
            let llmFeedbackRemaining = 0;
            let llmFeedbackQueued = 0;
            let llmFeedbackQueuedMessages = [];
            let llmFeedbackAppliedMessages = [];
            let llmFeedbackBusy = false;
            let codexAuthState = 'idle';
            let codexAuthDetail = '';
            let codexAuthSignedInMethod = '';
            const llmModels = {
              gpt:[['','defaultModel'],['gpt-6-astra','GPT-6 Astra'],['gpt-5.6','GPT-5.6'],
                ['gpt-5.6-sol','GPT-5.6 Sol'],['gpt-5.6-terra','GPT-5.6 Terra'],
                ['gpt-5.6-luna','GPT-5.6 Luna']],
              claude:[['claude-sonnet-5','Claude Sonnet 5'],['claude-opus-5','Claude Opus 5'],
                ['claude-haiku-4-5-20251001','Claude Haiku 4.5']]
            };
            const populateLlmModels = preferred => {
              const provider = llmProvider.value === 'claude' ? 'claude' : 'gpt';
              const previous = preferred == null ? llmModel.value : String(preferred || '');
              llmModel.replaceChildren();
              llmModels[provider].forEach(([value,label]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = label === 'defaultModel' ? qtOpText(label) : label;
                llmModel.appendChild(option);
              });
              if (previous && !llmModels[provider].some(item => item[0] === previous)) {
                const option = document.createElement('option');
                option.value = previous; option.textContent = previous + ' (saved)';
                llmModel.appendChild(option);
              }
              const fallback = provider === 'claude' ? 'claude-sonnet-5' : '';
              llmModel.value = Array.from(llmModel.options).some(option => option.value === previous)
                ? previous : fallback;
            };
            if (!automaticModeOrder.includes(selectedMode)) selectedMode = 'mode1';
            selectedModes.add(selectedMode);
            const quickLabel = stateName => ({
              running:qtOpText('testing'), pass:qtOpText('passed'),
              warn:qtOpText('retest'), fail:qtOpText('fault')
            })[stateName] || stateName;
            const renderLlmStatus = () => {
              llmStatus.classList.remove('good','bad');
              const text = ({loading:'llmLoading', ready:'llmReady', missing:'llmMissing',
                loadFailed:'llmLoadFailed', saved:'llmSaved', saveFailed:'llmSaveFailed'})[llmStatusState];
              llmStatus.textContent = text ? qtOpText(text) + (llmStatusDetail ? `：${llmStatusDetail}` : '') : '—';
              if (['ready','saved'].includes(llmStatusState)) llmStatus.classList.add('good');
              if (['missing','loadFailed','saveFailed'].includes(llmStatusState)) llmStatus.classList.add('bad');
            };
            const setLlmStatus = (stateName, detail='') => {
              llmStatusState = stateName;
              llmStatusDetail = String(detail || '');
              renderLlmStatus();
            };
            const renderCodexAuth = () => {
              const provider = llmProvider.value === 'claude' ? 'claude' : 'gpt';
              const accountOption = llmAuthMethod.querySelector('option[value="account"]');
              accountOption.textContent = qtOpText(provider === 'claude' ? 'claudeAccount' : 'chatgptAccount');
              const useApi = llmAuthMethod.value === 'api';
              llmApiLabel.hidden = !useApi;
              llmApiControls.hidden = !useApi;
              llmAuthAction.textContent = qtOpText(useApi ? 'useApiKey' :
                provider === 'claude' ? 'signInClaude' : 'signInChatGPT');
              const busy = ['checking','login_started','installing'].includes(codexAuthState);
              llmInstallClaude.hidden = !(provider === 'claude' && !useApi && codexAuthState === 'missing');
              llmAuthMethod.disabled = busy;
              llmAuthAction.disabled = busy;
              llmAuthRefresh.disabled = busy;
              llmInstallClaude.disabled = busy;
              llmSignOut.disabled = busy || codexAuthState !== 'signed_in';
              llmApiKey.disabled = busy;
              llmFast.disabled = provider !== 'gpt';
              const fastRow = panel.querySelector('#qt-op-llm-fast-row');
              if (fastRow) fastRow.style.opacity = provider === 'gpt' ? '1' : '0.48';
              llmAuthStatus.classList.remove('good','bad');
              let label = '—';
              if (codexAuthState === 'checking') label = qtOpText('authChecking');
              else if (codexAuthState === 'login_started') label = qtOpText('authStarting');
              else if (codexAuthState === 'installing') label = qtOpText('installingClaude');
              else if (codexAuthState === 'signed_in') {
                label = qtOpText(codexAuthSignedInMethod === 'api' ? 'signedInApi' :
                  provider === 'claude' ? 'signedInClaude' : 'signedInChatGPT');
                llmAuthStatus.classList.add('good');
              } else if (codexAuthState === 'signed_out') {
                label = qtOpText('signedOut'); llmAuthStatus.classList.add('bad');
              } else if (codexAuthState === 'missing') {
                label = qtOpText(provider === 'claude' ? 'claudeNotInstalled' : 'codexNotInstalled');
                llmAuthStatus.classList.add('bad');
              } else if (codexAuthState === 'error') {
                label = qtOpText('authFailed'); llmAuthStatus.classList.add('bad');
              }
              const showDetail = qtOpEffectiveLanguage() !== 'en' && codexAuthDetail &&
                ['signed_out','missing','error'].includes(codexAuthState);
              llmAuthStatus.textContent = label + (showDetail ? `：${codexAuthDetail}` : '');
            };
            const acceptCodexAuthStatus = raw => {
              let payload = raw;
              try { payload = typeof raw === 'string' ? JSON.parse(raw) : raw; } catch (_) { payload = null; }
              if (!payload || !payload.state) {
                codexAuthState = 'error'; codexAuthDetail = qtOpText('authFailed');
              } else {
                if (payload.provider && String(payload.provider) !== llmProvider.value) return;
                codexAuthState = String(payload.state);
                codexAuthDetail = String(payload.message || '');
                codexAuthSignedInMethod = String(payload.method || '');
                if (codexAuthState === 'signed_in') {
                  llmApiKey.value = '';
                }
              }
              renderCodexAuth();
            };
            const requestCodexAuthStatus = () => {
              const provider = llmProvider.value === 'claude' ? 'claude' : 'gpt';
              const checker = provider === 'claude' ? 'checkClaudeAuth' : 'checkCodexAuth';
              if (typeof pyHost === 'undefined' || !pyHost ||
                  typeof pyHost[checker] !== 'function') {
                acceptCodexAuthStatus({state:'missing',message:qtOpText('llmMissing')}); return;
              }
              codexAuthState = 'checking'; codexAuthDetail = ''; renderCodexAuth();
              const callback = started => {
                if (!started) acceptCodexAuthStatus({state:'error',message:qtOpText('authFailed')});
              };
              if (provider === 'claude') pyHost[checker](llmAuthMethod.value, callback);
              else pyHost[checker](callback);
            };
            const localizeQuickDetail = value => {
              let text = String(value || '—');
              if (qtOpEffectiveLanguage() !== 'en') return text;
              const direct = {
                '等待采样':'Waiting for samples', '检测中止':'Check aborted',
                '有效数据太少':'Insufficient valid data', '没有有效结果':'No valid result',
                '环境检查通过':'Environment check passed', '正在自动安装':'Installing automatically',
                '环境检测失败':'Environment check failed', '软件环境已经通过':'Software environment passed',
                '缺失组件已自动安装，软件环境通过':'Missing components installed automatically; software environment passed'
              };
              if (direct[text]) return direct[text];
              return text
                .replace(/^零压最大/, 'Maximum zero-pressure reading ')
                .replace(/；六路峰值 /, '; six-channel peaks ')
                .replace(/^20路完整帧/, 'Complete 20-channel frames ')
                .replace(/；六组长度变化 /, '; six-group length changes ')
                .replace(/^本路锁定率 /, 'Active-channel lock rates ')
                .replace(/；拮抗误锁最高/, '; maximum unintended antagonist locking ')
                .replace(/^六方向有效位移 /, 'Valid six-direction displacements ')
                .replace(/^已更新速度上限：/, 'Speed limits updated: ')
                .replace(/^速度未更新：/, 'Speed limits not updated: ')
                .replace(/；压力\/映射\/IMU检查未通过/, '; pressure/mapping/IMU checks did not pass')
                .replace(/^缺少或不兼容：/, 'Missing or incompatible: ')
                .replace(/^缺少：/, 'Missing: ')
                .replace(/^缺少模块 /, 'Missing module ')
                .replace(/^要求 /, 'Requires ');
            };
            const applyModePanelLanguage = () => {
              panel.querySelector('#qt-op-modes-title').textContent = qtOpText('modes');
              opVent.textContent = qtOpText('ventAll');
              const tabKeys = {auto:'automatic',teach:'teaching',test:'test'};
              tabs.forEach(button => { button.textContent = qtOpText(tabKeys[button.dataset.tab]); });
              autoModes.forEach(card => {
                card.querySelector('.qt-op-auto-select').textContent =
                  qtOpText('mode')+' '+card.dataset.fixedMode.slice(-1);
                const loopText = card.querySelector('.qt-op-loop');
                if (loopText && loopText.lastChild) loopText.lastChild.textContent = qtOpText('loop');
                const limit = card.querySelector('.qt-op-limit');
                if (limit) {
                  limit.querySelector('input').title = qtOpText('durationHelp');
                  const unit = limit.querySelector('select');
                  unit.querySelector('option[value="runs"]').textContent = qtOpText('runsUnit');
                  unit.querySelector('option[value="min"]').textContent = qtOpText('minutesUnit');
                }
              });
              opRecord.textContent = qtOpText('addPoint');
              opLoadDefault.textContent = qtOpText('loadDefault');
              opLoadDefault.title = qtOpText('loadDefaultHelp');
              opClear.textContent = qtOpText('clearAll');
              clearMemoryButton.textContent = qtOpText('clearMemory');
              clearMemoryButton.title = qtOpText('clearMemoryHelp');
              saveProcessButton.textContent = qtOpText('saveProcess');
              saveProcessButton.title = qtOpText('saveProcessHelp');
              loadReplayButton.textContent = 'Load 1';
              testModifiedButton.textContent = qtOpEffectiveLanguage() === 'en'
                ? `Test Load ${selectedReplayVariant} (1 Run)` : `测试 Load ${selectedReplayVariant}（单轮）`;
              loadReplayButton.title = qtOpText('loadReplayHelp');
              exitReplayButton.textContent = qtOpText('exitReplay');
              const controllerLabel = teachController.closest('label');
              if (controllerLabel && controllerLabel.firstChild)
                controllerLabel.firstChild.textContent = qtOpText('controller');
              const teachLoopLabel = teachLoop.closest('.qt-op-loop');
              if (teachLoopLabel && teachLoopLabel.lastChild)
                teachLoopLabel.lastChild.textContent = qtOpText('loopExecution');
              teachLimit.title = qtOpText('durationHelp');
              teachUnit.querySelector('option[value="runs"]').textContent = qtOpText('runsUnit');
              teachUnit.querySelector('option[value="min"]').textContent = qtOpText('minutesUnit');
              llmConfigButton.textContent = qtOpText('llmSettings');
              llmFeedbackInput.placeholder = qtOpText('llmInterventionPlaceholder');
              llmFeedbackInput.title = qtOpText('llmIntervention');
              renderLlmFeedback();
              llmTitle.textContent = qtOpText('llmConfiguration');
              panel.querySelector('#qt-op-llm-provider-label').textContent = qtOpText('llmProvider');
              llmProvider.querySelector('option[value="gpt"]').textContent = qtOpText('gptProvider');
              llmProvider.querySelector('option[value="claude"]').textContent = qtOpText('claudeProvider');
              panel.querySelector('#qt-op-llm-model-label').textContent = qtOpText('llmModel');
              populateLlmModels(llmModel.value);
              panel.querySelector('#qt-op-llm-profile-label').textContent = qtOpText('compensatorDesign');
              llmProfile.querySelector('option[value="base"]').textContent = qtOpText('designBase');
              llmProfile.querySelector('option[value="plus"]').textContent = qtOpText('designPlus');
              llmProfile.querySelector('option[value="pro"]').textContent = qtOpText('designPro');
              panel.querySelector('#qt-op-llm-effort-label').textContent = qtOpText('reasoningEffort');
              panel.querySelector('#qt-op-llm-timeout-label').textContent = qtOpText('analysisTimeout');
              panel.querySelector('#qt-op-llm-reuse-mse-label').textContent = qtOpText('reuseMse');
              llmReuseMse.title = qtOpText('reuseMseHelp');
              panel.querySelector('#qt-op-llm-fast-label').textContent = qtOpText('fastMode');
              panel.querySelector('#qt-op-llm-fast-row').title = qtOpText('fastModeHelp');
              panel.querySelector('#qt-op-llm-auto-label').textContent = qtOpText('automaticDesign');
              panel.querySelector('#qt-op-llm-auth-method-label').textContent = qtOpText('authMethod');
              llmAuthMethod.querySelector('option[value="account"]').textContent = qtOpText(
                llmProvider.value === 'claude' ? 'claudeAccount' : 'chatgptAccount');
              llmAuthMethod.querySelector('option[value="api"]').textContent = qtOpText('apiKeyAccess');
              panel.querySelector('#qt-op-llm-api-label').textContent = qtOpText('apiKey');
              llmApiKey.placeholder = qtOpText('apiKeyPlaceholder');
              panel.querySelector('#qt-op-llm-auth-status-label').textContent = qtOpText('authStatus');
              llmAuthRefresh.textContent = qtOpText('refreshStatus');
              llmInstallClaude.textContent = qtOpText('installClaude');
              llmSignOut.textContent = qtOpText('signOut');
              llmCancel.textContent = qtOpText('cancel');
              llmSave.textContent = qtOpText('saveSettings');
              renderLlmStatus();
              renderCodexAuth();
              quickStart.textContent = qtOpText('quickStart');
              quickStop.textContent = qtOpText('quickStop');
              const cardText = {
                environment:['pythonEnvironment','pythonDetail'], pressure:['pressureSensor','pressureDetail'],
                muscle:['muscleContraction','muscleDetail'], mapping:['valveLockMapping','mappingDetail'],
                imu:[null,'imuDetail'], speed:['maxSpeed','speedDetail']
              };
              quickCards.forEach(card => {
                const keys = cardText[card.dataset.quickKey];
                if (keys && keys[0]) card.querySelector('header').firstChild.textContent = qtOpText(keys[0]);
                if (keys) card.querySelector('small').textContent = card.dataset.quickDetail
                  ? localizeQuickDetail(card.dataset.quickDetail) : qtOpText(keys[1]);
                const stateNode = card.querySelector('header span');
                if (!card.classList.contains('running') && !card.classList.contains('pass') &&
                    !card.classList.contains('warn') && !card.classList.contains('fail'))
                  stateNode.textContent = qtOpText('pending');
                else for (const stateName of ['running','pass','warn','fail'])
                  if (card.classList.contains(stateName)) stateNode.textContent = quickLabel(stateName);
              });
              if (!quickRunning && !quickOverall.classList.contains('pass') &&
                  !quickOverall.classList.contains('warn') && !quickOverall.classList.contains('fail'))
                quickOverall.textContent = qtOpText('notTested');
              renderedTeachSignature = '';
              renderTeachPoints();
            };
            window.__qtApplyModePanelLanguage = applyModePanelLanguage;
            const signedNumber = (id, fallback) => {
              const input = document.getElementById(id);
              if (!input) return fallback;
              const cleaned = String(input.value).trim().replace(/°/g,'').replace(/−/g,'-');
              const value = Number(cleaned);
              return Number.isFinite(value) ? Math.max(-180, Math.min(180, value)) : fallback;
            };
            const formatTarget = (id, fallback) => {
              const input = document.getElementById(id);
              if (!input) return;
              const value = signedNumber(id, fallback);
              input.value = `${value > 0 ? '+' : ''}${Number(value.toFixed(1))}`;
            };
            const optionalLoopLimit = (inputId, unitId) => {
              const input = document.getElementById(inputId);
              const unit = document.getElementById(unitId);
              const raw = input ? String(input.value).trim() : '';
              const kind = unit && unit.value === 'min' ? 'min' : 'runs';
              if (!raw) return {limitValue:null,limitUnit:kind};
              let value = Number(raw);
              if (!Number.isFinite(value)) return {limitValue:null,limitUnit:kind};
              value = kind === 'runs' ? Math.floor(Math.max(1,Math.min(999,value)))
                : Math.max(0.1,Math.min(999,Number(value.toFixed(1))));
              return {limitValue:value,limitUnit:kind};
            };
            window.__qtGetFixedOpenConfig = modeName => {
              if (modeName === 'mode1') return Object.assign({
                targetYDeg:signedNumber('qt-op-m1-y',60),
                targetXDeg:signedNumber('qt-op-m1-x',-20),
                loop:document.getElementById('qt-op-m1-loop').checked
              }, optionalLoopLimit('qt-op-m1-limit','qt-op-m1-unit'));
              if (modeName === 'mode2') return Object.assign({
                targetYDeg:signedNumber('qt-op-m2-y1',60),
                targetYNegDeg:signedNumber('qt-op-m2-y2',-20),
                loop:document.getElementById('qt-op-m2-loop').checked
              }, optionalLoopLimit('qt-op-m2-limit','qt-op-m2-unit'));
              if (modeName === 'mode4') return Object.assign({
                targetYDeg:signedNumber('qt-op-m4-y',-30),
                targetPhiDeg:signedNumber('qt-op-m4-phi',20),
                loop:document.getElementById('qt-op-m4-loop').checked
              }, optionalLoopLimit('qt-op-m4-limit','qt-op-m4-unit'));
              if (modeName === 'mode5') return Object.assign({
                loop:document.getElementById('qt-op-m5-loop').checked
              }, optionalLoopLimit('qt-op-m5-limit','qt-op-m5-unit'));
              return Object.assign({loop:document.getElementById('qt-op-m3-loop').checked},
                optionalLoopLimit('qt-op-m3-limit','qt-op-m3-unit'));
            };
            ['m1','m2','m3','m4','m5','teach'].forEach(name => {
              const prefix = name === 'teach' ? 'qt-op-teach' : 'qt-op-'+name;
              const input = document.getElementById(prefix+'-limit');
              const unit = document.getElementById(prefix+'-unit');
              if (!input || !unit) return;
              const syncUnit = () => {
                const minutes = unit.value === 'min';
                input.step = minutes ? '0.1' : '1';
                input.min = minutes ? '0.1' : '1';
                if (input.value.trim()) {
                  const parsed = optionalLoopLimit(input.id,unit.id);
                  input.value = parsed.limitValue == null ? '' : String(parsed.limitValue);
                }
              };
              unit.addEventListener('change', syncUnit);
              input.addEventListener('blur', syncUnit);
              syncUnit();
            });

            const selectFixedMode = value => {
              if (!automaticModeOrder.includes(value)) return;
              selectedMode = value;
              legacyMode.value = 'open';
              legacyMode.dispatchEvent(new Event('change', {bubbles:true}));
              fixedMode.value = value;
              fixedMode.dispatchEvent(new Event('change', {bubbles:true}));
            };
            const renderAutoModeSelection = () => {
              autoModes.forEach(card => {
                const chosen = selectedModes.has(card.dataset.fixedMode);
                card.dataset.selected = String(chosen);
                const button = card.querySelector('.qt-op-auto-select');
                if (button) button.setAttribute('aria-pressed', String(chosen));
              });
            };
            const toggleAutoMode = value => {
              if (!automaticModeOrder.includes(value)) return;
              if (selectedModes.has(value)) selectedModes.delete(value);
              else selectedModes.add(value);
              if (selectedModes.has(value)) {
                selectFixedMode(value);
              } else if (selectedMode === value && selectedModes.size) {
                selectFixedMode(automaticModeOrder.find(name => selectedModes.has(name)));
              }
              renderAutoModeSelection();
            };
            renderAutoModeSelection();
            const cancelFixedOpenQueue = (announce=false) => {
              const queue = autoQueue;
              if (!queue) return false;
              queue.active = false;
              if (queue.transitionTimer) clearTimeout(queue.transitionTimer);
              autoQueue = null;
              window.__qtTimedAutomaticActive = false;
              if (announce && typeof tcSay === 'function')
                tcSay('<b>'+qtOpText('stopAuto')+'</b>', false);
              return true;
            };
            window.__qtCancelFixedOpenQueue = cancelFixedOpenQueue;
            window.__qtAdvanceFixedOpenQueue = completedMode => {
              const queue = autoQueue;
              if (!queue || !queue.active || queue.modes[queue.index] !== completedMode)
                return null;
              if (queue.deadlineMs && queue.index === queue.modes.length - 1 && Date.now() >= queue.deadlineMs) {
                cancelFixedOpenQueue(false);
                return {continue:false, message:'<b>Automatic 1→2→3：30分钟任务完成</b>，比例阀已清零。'};
              }
              if (queue.index + 1 >= queue.modes.length && !queue.deadlineMs) {
                queue.active = false;
                autoQueue = null;
                return {continue:false, message:'<b>'+qtOpText('queueComplete')+'</b>'};
              }
              const from = completedMode.slice(-1);
              queue.index = (queue.index + 1) % queue.modes.length;
              const nextMode = queue.modes[queue.index];
              const to = nextMode.slice(-1);
              selectFixedMode(nextMode);
              if (typeof tcSay === 'function')
                tcSay('<b>'+qtOpText('queueTransition',{from,to})+'</b>', false);
              queue.transitionTimer = setTimeout(() => {
                if (autoQueue !== queue || !queue.active) return;
                queue.transitionTimer = 0;
                if (queue.deadlineMs && queue.index === 0 && Date.now() >= queue.deadlineMs) {
                  cancelFixedOpenQueue(false);
                  if (typeof tcStop === 'function') tcStop();
                  stopSelectedRecordings();
                  if (typeof tcSay === 'function') tcSay('Automatic 30分钟任务完成。', false);
                  return;
                }
                play.click();
              }, 1000);
              return {continue:true};
            };
            const renderLlmFeedbackAudit = () => {
              if (!llmFeedbackAudit) return;
              llmFeedbackAudit.replaceChildren();
              const append = (kind,label,time,message) => {
                const row = document.createElement('div');
                row.className = kind;
                row.textContent = `${label}${time?' · '+time:''}: ${message}`;
                llmFeedbackAudit.appendChild(row);
              };
              for (const item of llmFeedbackAppliedMessages.slice(-5)) {
                const delivery = item && item.delivery === 'same_session_review'
                  ? qtOpText('llmFeedbackReview') : qtOpText('llmFeedbackInitial');
                append('applied',qtOpText('llmFeedbackAppliedLabel')+' · '+delivery,
                  String(item&&item.time||'').replace('T',' '),String(item&&item.message||''));
              }
              for (const message of llmFeedbackQueuedMessages.slice(-5))
                append('queued',qtOpText('llmFeedbackQueuedLabel'),' ',String(message||''));
            };
            const renderLlmFeedback = (message='') => {
              const selected = teachController.value === 'mpc_codex' && !llmReplayActive;
              llmFeedbackPanel.hidden = !selected;
              if (!selected) return;
              llmFeedbackStatus.classList.remove('good','bad');
              if (message) {
                llmFeedbackStatus.textContent = message;
                llmFeedbackStatus.classList.add('bad');
              } else if (llmFeedbackWaiting) {
                llmFeedbackStatus.textContent = llmFeedbackHeld
                  ? qtOpText('llmPromptHeld')
                  : qtOpText('llmPromptCountdown',{n:llmFeedbackRemaining});
              } else if (llmFeedbackRunning && llmFeedbackQueued > 0) {
                llmFeedbackStatus.textContent = qtOpText('llmInterventionQueued', {n:llmFeedbackQueued});
                llmFeedbackStatus.classList.add('good');
              } else if (llmFeedbackRunning) {
                llmFeedbackStatus.textContent = qtOpText('llmInterventionReady');
              } else {
                llmFeedbackStatus.textContent = qtOpText('llmInterventionIdle');
              }
              const canEdit = (llmFeedbackWaiting || llmFeedbackRunning) && !llmFeedbackBusy;
              llmFeedbackInput.disabled = !canEdit;
              if (llmFeedbackWaiting) {
                llmFeedbackSend.textContent = llmFeedbackInput.value.trim()
                  ? qtOpText('llmPromptSubmit') : qtOpText('llmPromptStartNow');
                llmFeedbackSend.disabled = !canEdit;
              } else {
                llmFeedbackSend.textContent = qtOpText('llmReasoningAdd');
                llmFeedbackSend.disabled = !canEdit || !llmFeedbackInput.value.trim();
              }
              renderLlmFeedbackAudit();
            };
            const refreshLlmFeedback = async () => {
              if (teachController.value !== 'mpc_codex') {
                llmFeedbackRunning = false; renderLlmFeedback(); return;
              }
              try {
                const response = await fetch('http://127.0.0.1:8765/llm_feedback');
                const result = await response.json();
                if (!response.ok || result.err) throw new Error(result.err || `HTTP ${response.status}`);
                llmFeedbackWaiting = !!result.waiting_for_prompt;
                llmFeedbackRunning = !!result.design_running && result.provider === 'gpt';
                llmFeedbackHeld = !!result.held;
                llmFeedbackRemaining = Number(result.remaining_s || 0);
                llmFeedbackQueued = Number(result.queued || 0);
                llmFeedbackQueuedMessages = Array.isArray(result.queued_feedback)
                  ? result.queued_feedback : [];
                llmFeedbackAppliedMessages = Array.isArray(result.applied_feedback)
                  ? result.applied_feedback : [];
                renderLlmFeedback();
              } catch (_) {
                llmFeedbackWaiting = false; llmFeedbackRunning = false;
                renderLlmFeedback();
              }
            };
            const holdLlmFeedback = async () => {
              if (!llmFeedbackWaiting || llmFeedbackHeld) return;
              llmFeedbackHeld = true; renderLlmFeedback();
              try {
                const response = await fetch('http://127.0.0.1:8765/llm_feedback', {
                  method:'POST', headers:{'Content-Type':'application/json'},
                  body:JSON.stringify({action:'hold'})
                });
                const result = await response.json();
                if (!response.ok || result.err) throw new Error(result.err || `HTTP ${response.status}`);
              } catch (_) {
                llmFeedbackHeld = false;
              }
              renderLlmFeedback();
            };
            const sendLlmFeedback = async () => {
              const message = llmFeedbackInput.value.trim();
              if ((!llmFeedbackWaiting && (!message || !llmFeedbackRunning)) || llmFeedbackBusy) return;
              llmFeedbackBusy = true; renderLlmFeedback();
              let failure = '';
              try {
                const response = await fetch('http://127.0.0.1:8765/llm_feedback', {
                  method:'POST', headers:{'Content-Type':'application/json'},
                  body:JSON.stringify({action:message?'submit':'skip', message})
                });
                const result = await response.json();
                if (!response.ok || result.err) throw new Error(result.err || `HTTP ${response.status}`);
                llmFeedbackInput.value = '';
                llmFeedbackQueued = Number(result.queued || llmFeedbackQueued + 1);
                if (result.started) {
                  llmFeedbackWaiting = false;
                  llmFeedbackHeld = false;
                }
                await refreshLlmFeedback();
              } catch (error) {
                failure = error && error.message ? error.message : qtOpText('llmInterventionIdle');
              } finally {
                llmFeedbackBusy = false;
                renderLlmFeedback(failure);
              }
            };
            const replayPost = async (path, body={}) => {
              const response = await fetch('http://127.0.0.1:8765'+path, {
                method:'POST', headers:{'Content-Type':'application/json'},
                body:JSON.stringify(body)
              });
              const result = await response.json();
              if (!response.ok || result.err) throw new Error(result.err || `HTTP ${response.status}`);
              return result;
            };
            const replayHostDialog = method => new Promise((resolve,reject) => {
              if (typeof pyHost === 'undefined' || !pyHost || typeof pyHost[method] !== 'function') {
                reject(new Error('当前本地宿主不支持MPC+LLM过程文件。')); return;
              }
              pyHost[method](raw => {
                try { resolve(typeof raw === 'string' ? JSON.parse(raw) : raw); }
                catch (error) { reject(error); }
              });
            });
            const renderReplayControls = () => {
              const selected = teachController.value === 'mpc_codex';
              testModifiedButton.textContent = qtOpEffectiveLanguage() === 'en'
                ? `Test Load ${selectedReplayVariant} (1 Run)` : `测试 Load ${selectedReplayVariant}（单轮）`;
              window.__qtMpcLlmReplayActive = !!llmReplayActive;
              // Debug-launched replay also needs reference overlays in Control Response.
              document.body.classList.toggle('control-reference-visible', !!llmReplayActive);
              saveProcessButton.hidden = !selected || llmReplayActive;
              loadReplayButton.hidden = !selected;
              loadReplay2Button.hidden = !selected;
              testModifiedButton.hidden = !selected;
              exitReplayButton.hidden = !selected || !llmReplayActive;
              replayStatus.hidden = !selected || !llmReplayActive;
              if (llmReplayActive) {
                const count = Number(llmReplayInfo&&llmReplayInfo.round_count||10);
                replayStatus.textContent = llmReplayInfo && llmReplayInfo.duration_min
                  ? 'Load 2 · 30 min · 固定修改版' : qtOpText('replayLoaded',{n:count});
              } else replayStatus.textContent = '';
            };
            const clearReplayMode = async (announce=false) => {
              const wasActive = llmReplayActive;
              llmReplayActive = false; llmReplayInfo = null;
              renderReplayControls();
              if (!wasActive) return;
              try { await replayPost('/codex_replay/clear'); } catch (_) {}
              if (announce && typeof tcSay === 'function')
                tcSay(qtOpText('replayExited'), false);
            };
            saveProcessButton.onclick = async () => {
              if (llmReplayBusy || llmReplayActive ||
                  (typeof tcPlaying !== 'undefined' && tcPlaying)) return;
              llmReplayBusy = true; sync();
              try {
                const picked = await replayHostDialog('chooseCodexReplaySavePath');
                if (!picked || !picked.ok) return;
                const points = teach.map(src => {
                  const point = {
                    q:(src.q||[0,0,0]).slice(0,3).map(Number),
                    q_raw:(src.q_raw||src.q||[0,0,0]).slice(0,3).map(Number),
                    q_control:(src.q_control||src.q||[0,0,0]).slice(0,3).map(Number),
                    t:Number(src.t)||0
                  };
                  if (Array.isArray(src.channel_length_mm))
                    point.channel_length_mm=src.channel_length_mm.slice(0,6).map(Number);
                  return point;
                });
                const result = await replayPost('/codex_replay/save', {
                  path:picked.path, lap_count:10, points,
                  auto_hidden_return_to_zero:teach.auto_hidden_return_to_zero!==false
                });
                if (typeof tcSay === 'function')
                  tcSay(`<b>${result.message}</b><br>${result.path}`, false);
              } catch (error) {
                if (typeof tcSay === 'function') tcSay('保存10轮过程失败：'+error.message, true);
              } finally { llmReplayBusy = false; sync(); }
            };
            const loadComparisonReplay = async (singleTest=false, variant=1, timed30=false) => {
              if (llmReplayBusy ||
                  (typeof tcPlaying !== 'undefined' && tcPlaying)) return;
              llmReplayBusy = true; sync();
              try {
                const preset = await new Promise((resolve, reject) => {
                  pyHost.loadFixedTask(txt => {
                    try { resolve(JSON.parse(txt)); } catch (error) { reject(error); }
                  });
                });
                if (preset.error) throw new Error(preset.error);
                const result = await replayPost('/codex_replay/load', {
                  comparison_preset:true, points:preset.points,
                  modified_single_test:singleTest,
                  comparison_variant:variant,
                  load2_30min:timed30,
                  auto_hidden_return_to_zero:preset.auto_hidden_return_to_zero !== false
                });
                if (!Array.isArray(result.points) || !result.points.length)
                  throw new Error('回放包没有示教轨迹');
                if (typeof tcStop === 'function') tcStop();
                teach = result.points.map(src => ({
                  ...src,
                  q:(src.q||[0,0,0]).slice(0,3).map(Number),
                  q_raw:(src.q_raw||src.q||[0,0,0]).slice(0,3).map(Number),
                  q_control:(src.q_control||src.q||[0,0,0]).slice(0,3).map(Number),
                  t:Number(src.t)||0,
                  d:Array.isArray(src.d)?src.d.slice(0,6):Array(6).fill(0),
                  lk:Array.isArray(src.lk)?src.lk.slice(0,6):Array(6).fill(false),
                  ll:Array.isArray(src.ll)?src.ll.slice(0,10):Array(10).fill(null),
                  err:0
                }));
                teach.auto_hidden_return_to_zero = result.auto_hidden_return_to_zero !== false;
                if (typeof tcRender === 'function') tcRender(-1);
                renderedTeachSignature = ''; renderTeachPoints();
                llmReplayActive = true; llmReplayInfo = result;
                selectedReplayVariant = variant;
                teachController.value = 'mpc_codex';
                teachLoop.checked = true;
                teachLimit.value = result.duration_min ? String(result.duration_min) : String(Number(result.round_count)||10);
                teachUnit.value = result.duration_min ? 'min' : 'runs';
                selectTeachController();
                if (typeof tcSay === 'function') tcSay(
                  `<b>Load ${variant} — ${qtOpText('replayLoaded',{n:Number(result.round_count)||10})}</b>：`+
                  (variant === 2 ? 'V2：第5～7.9秒仅角度与速度均落后时补偿，最多+50mV。' : 'V2：第4～7秒最多+50mV。')+
                  (timed30 ? '固定修改版重复运行30分钟；无前4轮、无模拟思考、不调用LLM；完成当前轮后停止。' : singleTest ? '直接测试修改版：第5轮主体＋第4轮φ下降；仅执行一轮，保留V3辅助，不调用LLM。' :
                  (variant === 2
                    ? 'Load 2：第1轮普通MPC原始输出，第2～4轮历史控制器，第5轮起修改版；不调用LLM。'
                    : '今日Default 1预设对比：第1轮纯MPC禁V3，第2～4轮今日MSE约100，第5轮起拼接版；MSE按实测记录，不调用LLM。')), false);
                return true;
              } catch (error) {
                if (typeof tcSay === 'function') tcSay('载入过程回放失败：'+error.message, true);
              } finally { llmReplayBusy = false; sync(); }
            };
            loadReplayButton.onclick = () => loadComparisonReplay(false);
            loadReplay2Button.onclick = () => loadComparisonReplay(false, 2);
            debugLoad2Timed.onclick = async () => {
              if (autoQueue && autoQueue.deadlineMs && autoQueue.active) {
                cancelFixedOpenQueue(false);
                if (typeof tcStop === 'function') tcStop();
                stopSelectedRecordings();
                return;
              }
              if (llmReplayBusy || recordingLaunchPending ||
                  (typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcStarting !== 'undefined' && tcMpcStarting)) return;
              if (typeof imuOn === 'undefined' || !imuOn) {
                if (typeof tcSay === 'function') tcSay('先连接GIGA/IMU，再执行Automatic 30分钟任务。', true);
                return;
              }
              if (llmReplayActive) await clearReplayMode(false);
              if (window.__qtTeachExecutionPlan) window.__qtTeachExecutionPlan.active = false;
              cancelFixedOpenQueue(false);
              selectedModes.clear();
              ['mode1','mode2','mode3'].forEach(name => selectedModes.add(name));
              renderAutoModeSelection();
              selectFixedMode('mode1');
              beginWithSelectedRecordings(() => {
                autoQueue = {modes:['mode1','mode2','mode3'], index:0, active:true,
                  transitionTimer:0, deadlineMs:Date.now()+30*60*1000};
                window.__qtTimedAutomaticActive = true;
                play.click();
              });
            };
            testModifiedButton.onclick = async () => {
              if (await loadComparisonReplay(true, selectedReplayVariant)) {
                // Use the existing Execute path, including recording and connection checks.
                panel.querySelector('#qt-op-teach-run').click();
              }
            };
            exitReplayButton.onclick = () => clearReplayMode(true);
            const selectTeachController = () => {
              const value = ['mpc','pi','mpc_codex'].includes(teachController.value)
                ? teachController.value : 'mpc';
              if (value !== 'mpc_codex' && llmReplayActive) clearReplayMode(false);
              llmConfigButton.hidden = value !== 'mpc_codex' || llmReplayActive;
              clearMemoryButton.hidden = value !== 'mpc_codex' || llmReplayActive;
              llmFeedbackPanel.hidden = value !== 'mpc_codex' || llmReplayActive;
              if (value !== 'mpc_codex') { llmApiKey.value = ''; llmModal.hidden = true; }
              legacyMode.value = value;
              legacyMode.dispatchEvent(new Event('change', {bubbles:true}));
              const legacyLoop = document.getElementById('tc-loop');
              if (legacyLoop) {
                legacyLoop.checked = !!teachLoop.checked;
                legacyLoop.dispatchEvent(new Event('change', {bubbles:true}));
              }
              renderReplayControls();
              refreshLlmFeedback();
            };
            const openLlmConfig = async () => {
              llmModal.hidden = false;
              setLlmStatus('loading');
              try {
                const response = await fetch('http://127.0.0.1:8765/llm_config');
                const result = await response.json();
                if (!response.ok || result.err) throw new Error(result.err || `HTTP ${response.status}`);
                llmProvider.value = result.provider === 'claude' ? 'claude' : 'gpt';
                llmAuthMethod.value = result.auth_method === 'api' ? 'api' : 'account';
                populateLlmModels(String(result.model || ''));
                llmProfile.value = ['base','plus','pro'].includes(result.prompt_profile)
                  ? result.prompt_profile : 'base';
                renderCodexAuth();
                requestCodexAuthStatus();
                const savedEffort = result.reasoning_effort === 'minimal'
                  ? 'low' : String(result.reasoning_effort || 'medium');
                llmEffort.value = ['low','medium','high','xhigh'].includes(savedEffort)
                  ? savedEffort : 'medium';
                llmTimeout.value = String(Math.max(30,Math.min(900,Number(result.timeout_s)||300)));
                llmReuseMse.value = String(Math.max(0,Math.min(10000,
                  Number.isFinite(Number(result.reuse_mse_threshold_deg2))
                    ? Number(result.reuse_mse_threshold_deg2) : 15)));
                llmFast.checked = result.fast_mode === true;
                llmAuto.checked = result.auto_design !== false;
                setLlmStatus('ready');
              } catch (error) {
                setLlmStatus('loadFailed', error && error.message);
              }
            };
            const saveLlmConfig = async () => {
              const timeout = Number(llmTimeout.value);
              const reuseMse = Number(llmReuseMse.value);
              if (!Number.isFinite(timeout) || timeout < 30 || timeout > 900) {
                setLlmStatus('saveFailed', '30–900 s'); return;
              }
              if (!Number.isFinite(reuseMse) || reuseMse < 0 || reuseMse > 10000) {
                setLlmStatus('saveFailed', 'MSE 0–10000 deg²'); return;
              }
              llmSave.disabled = true;
              try {
                const response = await fetch('http://127.0.0.1:8765/llm_config', {
                  method:'POST', headers:{'Content-Type':'application/json'},
                  body:JSON.stringify({provider:llmProvider.value, auth_method:llmAuthMethod.value,
                    model:llmModel.value,
                    prompt_profile:llmProfile.value,
                    reasoning_effort:llmEffort.value, timeout_s:timeout,
                    reuse_mse_threshold_deg2:reuseMse,
                    fast_mode:!!llmFast.checked,
                    auto_design:!!llmAuto.checked})
                });
                const result = await response.json();
                if (!response.ok || result.err) throw new Error(result.err || `HTTP ${response.status}`);
                setLlmStatus('saved', result.warn || '');
              } catch (error) {
                setLlmStatus('saveFailed', error && error.message);
              } finally {
                llmSave.disabled = false;
              }
            };
            const teachExecutionConfig = () => Object.assign({loop:!!teachLoop.checked},
              optionalLoopLimit('qt-op-teach-limit','qt-op-teach-unit'));
            const beginTeachExecutionPlan = () => {
              const cfg = teachExecutionConfig();
              const startedAtMs = Date.now();
              window.__qtTeachExecutionPlan = {
                active:true, loop:cfg.loop, limitValue:cfg.limitValue,
                limitUnit:cfg.limitUnit, completedRuns:0, startedAtMs,
                lastCountedRequestId:'',
                deadlineMs:cfg.loop && cfg.limitUnit === 'min' && cfg.limitValue != null
                  ? startedAtMs + Number(cfg.limitValue) * 60000 : null
              };
            };
            const renderTeachPoints = () => {
              const points = (typeof teach !== 'undefined' && Array.isArray(teach)) ? teach : [];
              const signature = JSON.stringify(points.map(point => [point.q,point.t,point.err]));
              if (signature === renderedTeachSignature) return;
              renderedTeachSignature = signature;
              if (!points.length) {
                opTeachList.innerHTML = '<div class="qt-op-teach-empty">'+qtOpText('noPoints')+'</div>';
                return;
              }
              opTeachList.innerHTML = points.map((point,index) => {
                const q = Array.isArray(point.q) ? point.q : [0,0,0];
                const dwell = Number.isFinite(Number(point.t)) ? Number(point.t).toFixed(1) : '--';
                const warn = Number(point.err) > 0.5;
                return `<div class="qt-op-teach-point${warn?' warn':''}" title="${warn?qtOpText('pointDeviation',{v:Number(point.err).toFixed(2)}):qtOpText('recordedPoint')}">`+
                  `<span class="qt-op-teach-point-index">#${index+1}</span>`+
                  `<span class="qt-op-teach-point-pose">X ${Number(q[0]||0).toFixed(1)}°　Y ${Number(q[1]||0).toFixed(1)}°　φ ${Number(q[2]||0).toFixed(1)}°</span>`+
                  `<span class="qt-op-teach-point-time">${dwell}s</span>`+
                  `<button class="btn qt-op-teach-delete" data-op-teach-delete="${index}" title="${qtOpText('deletePoint',{n:index+1})}" aria-label="${qtOpText('deletePoint',{n:index+1})}">×</button></div>`;
              }).join('');
              opTeachList.querySelectorAll('[data-op-teach-delete]').forEach(button => {
                button.onclick = () => {
                  if (typeof tcPlaying !== 'undefined' && tcPlaying) return;
                  const index = Number(button.dataset.opTeachDelete);
                  if (!Number.isInteger(index) || index < 0 || index >= teach.length) return;
                  teach.splice(index,1);
                  teach.fixed_task_id = '';
                  if (typeof tcRender === 'function') tcRender(-1);
                  renderedTeachSignature = '';
                  renderTeachPoints();
                };
              });
              opTeachList.scrollTop = opTeachList.scrollHeight;
            };
            const showTab = name => {
              activeTab = name;
              window.__qtOperationModeTab = name;
              document.body.classList.toggle('operation-teach-mode', name === 'teach');
              tabs.forEach(button =>
                button.setAttribute('aria-selected', String(button.dataset.tab === name)));
              panes.forEach(pane => { pane.hidden = pane.dataset.pane !== name; });
              if (name === 'auto') selectFixedMode(selectedMode);
              else if (name === 'teach') selectTeachController();
              else if (name === 'test') checkQuickEnvironment();
              if (typeof vizDraw === 'function') requestAnimationFrame(vizDraw);
            };
            tabs.forEach(button => button.onclick = () => showTab(button.dataset.tab));
            autoModes.forEach(card => {
              card.querySelector('.qt-op-auto-select').onclick = () => {
                if (!(typeof tcPlaying !== 'undefined' && tcPlaying))
                  toggleAutoMode(card.dataset.fixedMode);
              };
              card.querySelectorAll('input').forEach(input => {
                input.addEventListener('focus', () => {
                  if (!(typeof tcPlaying !== 'undefined' && tcPlaying)) selectFixedMode(card.dataset.fixedMode);
                });
              });
            });
            [['qt-op-m1-y',60],['qt-op-m1-x',-20],['qt-op-m2-y1',60],['qt-op-m2-y2',-20],
             ['qt-op-m4-y',-30],['qt-op-m4-phi',20]]
              .forEach(([id,fallback]) => {
                const input = document.getElementById(id);
                if (input) input.addEventListener('blur', () => formatTarget(id,fallback));
              });
            const stopSelectedRecordings = () => {
              // 取消仍在等待“控制真正启动”的保护计时器，避免停止并保存后旧计时器再次触发。
              recordingStartGuardToken += 1;
              if (typeof window.__qtStopSelectedRecordings === 'function')
                return window.__qtStopSelectedRecordings();
            };
            const beginWithSelectedRecordings = async action => {
              if (recordingLaunchPending) return;
              recordingLaunchPending = true;
              try {
                if (typeof window.__qtStartSelectedRecordings === 'function') {
                  const recordingReady = await window.__qtStartSelectedRecordings();
                  if (recordingReady === false) return;
                }
                const guardToken = ++recordingStartGuardToken;
                action();
                // /start、串口和控制器初始化可能超过750 ms。旧逻辑会在控制尚未真正
                // 启动时误判失败，提前关闭记录器，所以用户稍后按“停止执行”时已经无文件可存。
                // 现在最多等待15 s观察真实运行状态；一旦运行过，正常的sync下降沿负责保存。
                const startDeadline = Date.now() + 15000;
                const waitForExecutionStart = () => {
                  if (guardToken !== recordingStartGuardToken) return;
                  const active = (typeof tcPlaying !== 'undefined' && tcPlaying) ||
                    (typeof tcMpcStarting !== 'undefined' && tcMpcStarting) ||
                    (typeof tcMpcTimer !== 'undefined' && !!tcMpcTimer) || quickRunning ||
                    (typeof window.__qtFixedOpenIsRunning === 'function' &&
                     window.__qtFixedOpenIsRunning());
                  if (active) return;
                  if (Date.now() < startDeadline) {
                    setTimeout(waitForExecutionStart, 250);
                    return;
                  }
                  stopSelectedRecordings();
                  if (typeof tcSay === 'function')
                    tcSay('执行未在15秒内启动，已保存启动阶段的现有记录。', true);
                };
                setTimeout(waitForExecutionStart, 250);
              } finally {
                recordingLaunchPending = false;
              }
            };
            run.onclick = () => {
              if (window.__qtTeachExecutionPlan) window.__qtTeachExecutionPlan.active = false;
              if (autoQueue && autoQueue.active) {
                cancelFixedOpenQueue(true);
                if (typeof tcStop === 'function') tcStop();
                else if (typeof tcPlaying !== 'undefined' && tcPlaying) play.click();
                stopSelectedRecordings();
                return;
              }
              if (typeof tcPlaying !== 'undefined' && tcPlaying) {
                play.click(); stopSelectedRecordings(); return;
              }
              const modesToRun = automaticModeOrder.filter(modeName => selectedModes.has(modeName));
              if (!modesToRun.length) {
                if (typeof tcSay === 'function') tcSay(qtOpText('selectModeFirst'), true);
                return;
              }
              const infiniteBeforeLast = modesToRun.slice(0,-1).find(modeName => {
                const cfg = window.__qtGetFixedOpenConfig(modeName);
                return cfg.loop && cfg.limitValue == null;
              });
              if (infiniteBeforeLast) {
                if (typeof tcSay === 'function')
                  tcSay(qtOpText('finiteQueueRequired'), true);
                return;
              }
              cancelFixedOpenQueue(false);
              if (modesToRun.length > 1) {
                autoQueue = {modes:modesToRun, index:0, active:true, transitionTimer:0};
              }
              selectFixedMode(modesToRun[0]);
              beginWithSelectedRecordings(() => play.click());
            };
            opRecord.onclick = () => {
              if (llmReplayActive) clearReplayMode(false);
              if (typeof teach !== 'undefined' && Array.isArray(teach)) teach.fixed_task_id = '';
              record.click();
              renderTeachPoints();
            };
            opLoadDefault.onclick = () => {
              if (typeof tcPlaying !== 'undefined' && tcPlaying) return;
              if (llmReplayActive) clearReplayMode(false);
              if (typeof window.__qtLoadFixedTask !== 'function') {
                if (typeof tcSay === 'function') tcSay('默认示教轨迹尚未准备好。', true);
                return;
              }
              window.__qtLoadFixedTask(false, ok => {
                if (!ok) return;
                renderedTeachSignature = '';
                renderTeachPoints();
              }, opDefaultSelect.value);
            };
            opClear.onclick = () => {
              if (typeof tcPlaying !== 'undefined' && tcPlaying) return;
              if (llmReplayActive) clearReplayMode(false);
              clear.click();
              renderedTeachSignature = '';
              renderTeachPoints();
            };
            teachController.onchange = selectTeachController;
            teachLoop.onchange = selectTeachController;
            llmConfigButton.onclick = openLlmConfig;
            llmFeedbackSend.onclick = sendLlmFeedback;
            llmFeedbackInput.addEventListener('input', () => renderLlmFeedback());
            llmFeedbackInput.addEventListener('focus', holdLlmFeedback);
            llmFeedbackInput.addEventListener('keydown', event => {
              if (event.key === 'Enter') { event.preventDefault(); sendLlmFeedback(); }
            });
            setInterval(refreshLlmFeedback, 1000);
            clearMemoryButton.onclick = async () => {
              if (typeof tcPlaying !== 'undefined' && tcPlaying) {
                if (typeof tcSay === 'function') tcSay('请先停止控制，再清除LLM补偿器记忆。', true);
                return;
              }
              if (typeof window.__qtResetCodexCompensator !== 'function') {
                if (typeof tcSay === 'function') tcSay('补偿器清零接口尚未准备好，请稍后重试。', true);
                return;
              }
              clearMemoryRunning = true;
              sync();
              try {
                await window.__qtResetCodexCompensator();
              } finally {
                clearMemoryRunning = false;
                sync();
              }
            };
            llmProvider.onchange = () => {
              populateLlmModels();
              codexAuthState = 'idle'; codexAuthDetail = ''; codexAuthSignedInMethod = '';
              renderCodexAuth();
              requestCodexAuthStatus();
            };
            llmAuthMethod.onchange = () => { renderCodexAuth(); requestCodexAuthStatus(); };
            llmAuthRefresh.onclick = requestCodexAuthStatus;
            llmInstallClaude.onclick = () => {
              if (typeof pyHost === 'undefined' || !pyHost ||
                  typeof pyHost.installClaudeCode !== 'function') {
                acceptCodexAuthStatus({state:'error',message:qtOpText('claudeInstallFailed'),provider:'claude'});
                return;
              }
              codexAuthState = 'installing'; codexAuthDetail = ''; renderCodexAuth();
              pyHost.installClaudeCode(started => {
                if (!started) acceptCodexAuthStatus({state:'error',
                  message:qtOpText('claudeInstallFailed'),provider:'claude'});
              });
            };
            llmAuthAction.onclick = () => {
              if (typeof pyHost === 'undefined' || !pyHost) {
                acceptCodexAuthStatus({state:'missing',message:qtOpText('llmMissing')}); return;
              }
              if (llmAuthMethod.value === 'api') {
                const key = String(llmApiKey.value || '').trim();
                if (key.length < 20 || /\s/.test(key)) {
                  acceptCodexAuthStatus({state:'error',message:qtOpText('apiKeyRequired')}); return;
                }
                codexAuthState = 'login_started'; codexAuthDetail = ''; renderCodexAuth();
                const apiLogin = llmProvider.value === 'claude'
                  ? 'loginClaudeApiKey' : 'loginCodexApiKey';
                if (typeof pyHost[apiLogin] !== 'function') {
                  acceptCodexAuthStatus({state:'missing',message:qtOpText('llmMissing')}); return;
                }
                pyHost[apiLogin](key, started => {
                  llmApiKey.value = '';
                  if (!started) acceptCodexAuthStatus({state:'error',message:qtOpText('authFailed')});
                });
              } else {
                codexAuthState = 'login_started'; codexAuthDetail = ''; renderCodexAuth();
                const accountLogin = llmProvider.value === 'claude'
                  ? 'loginClaudeAccount' : 'loginCodexChatGPT';
                if (typeof pyHost[accountLogin] !== 'function') {
                  acceptCodexAuthStatus({state:'missing',message:qtOpText('llmMissing')}); return;
                }
                pyHost[accountLogin](started => {
                  if (!started) acceptCodexAuthStatus({state:'error',message:qtOpText('authFailed')});
                });
              }
            };
            llmSignOut.onclick = () => {
              if (typeof pyHost === 'undefined' || !pyHost ||
                  typeof pyHost.logoutLlm !== 'function') {
                acceptCodexAuthStatus({state:'error',message:qtOpText('authFailed')}); return;
              }
              codexAuthState = 'login_started'; codexAuthDetail = ''; renderCodexAuth();
              pyHost.logoutLlm(llmProvider.value, llmAuthMethod.value, started => {
                if (!started) acceptCodexAuthStatus({state:'error',message:qtOpText('authFailed')});
              });
            };
            if (typeof pyHost !== 'undefined' && pyHost && pyHost.codex_auth_status &&
                !window.__qtCodexAuthSignalConnected) {
              window.__qtCodexAuthSignalConnected = true;
              pyHost.codex_auth_status.connect(acceptCodexAuthStatus);
            }
            llmCancel.onclick = () => { llmApiKey.value = ''; llmModal.hidden = true; };
            llmSave.onclick = saveLlmConfig;
            llmModal.onclick = event => {
              if (event.target === llmModal) { llmApiKey.value = ''; llmModal.hidden = true; }
            };
            document.addEventListener('keydown', event => {
              if (event.key === 'Escape' && !llmModal.hidden) {
                llmApiKey.value = ''; llmModal.hidden = true;
              }
            });
            teachRun.onclick = async () => {
              if (window.__qtCodexCycleWaiting) {
                if (typeof window.__qtCancelCodexCycle === 'function')
                  window.__qtCancelCodexCycle('Codex 自动循环已由用户停止。', false);
                stopSelectedRecordings();
                sync();
                return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcStarting !== 'undefined' && tcMpcStarting)) {
                if (window.__qtTeachExecutionPlan) window.__qtTeachExecutionPlan.active = false;
                play.click(); stopSelectedRecordings(); return;
              }
              selectTeachController();
              if (teachController.value === 'mpc_codex') {
                try {
                  if (llmReplayActive) {
                    const replay = await replayPost('/codex_replay/reset');
                    teachLoop.checked = true;
                    teachLimit.value = replay.duration_min ? String(replay.duration_min) : String(Number(replay.round_count)||10);
                    teachUnit.value = replay.duration_min ? 'min' : 'runs';
                    selectTeachController();
                  } else {
                    await replayPost('/codex_replay/capture_start');
                  }
                } catch (error) {
                  if (typeof tcSay === 'function')
                    tcSay('MPC+LLM过程初始化失败：'+error.message, true);
                  return;
                }
              }
              beginTeachExecutionPlan();
              beginWithSelectedRecordings(() => play.click());
            };
            opVent.onclick = () => vent.click();

            const showQuickResult = payload => {
              const items = payload && payload.items ? payload.items : {};
              quickCards.forEach(card => {
                const result = items[card.dataset.quickKey];
                if (!result) return;
                card.classList.remove('running','pass','warn','fail');
                const stateName = ['running','pass','warn','fail'].includes(result.state)
                  ? result.state : 'warn';
                card.classList.add(stateName);
                card.querySelector('header span').textContent = quickLabel(stateName);
                card.dataset.quickDetail = String(result.detail || '');
                card.querySelector('small').textContent = localizeQuickDetail(result.detail);
              });
              if (payload && payload.message) quickOverall.textContent = localizeQuickDetail(payload.message);
              quickOverall.classList.remove('pass','warn','fail');
              if (payload && payload.state === 'complete') {
                const allPass = Object.values(items).length === 6 &&
                  Object.values(items).every(item => item.state === 'pass');
                quickOverall.classList.add(allPass ? 'pass' : 'warn');
                quickRunning = false;
              } else if (payload && payload.state === 'failed') {
                quickOverall.classList.add('fail'); quickRunning = false;
              }
              quickStart.disabled = quickRunning;
              quickStop.disabled = !quickRunning;
            };
            const requestEnvironmentInstall = () => {
              if (quickEnvironmentAutoAttempted || !pyHost ||
                  typeof pyHost.installMissingPythonPackages !== 'function') return;
              quickEnvironmentAutoAttempted = true;
              const pending = {state:'running',detail:qtOpText('autoInstall')};
              showQuickResult({state:'idle',items:{environment:pending}});
              pyHost.installMissingPythonPackages(raw => {
                let answer=null;
                try { answer=typeof raw==='string'?JSON.parse(raw):raw; } catch (_) {}
                if (!answer || !answer.ok) {
                  const failed={state:'fail',detail:answer&&answer.message||qtOpText('installStartFail')};
                  quickEnvironment=failed;
                  showQuickResult({state:'idle',message:failed.detail,items:{environment:failed}});
                } else if (answer.message) {
                  showQuickResult({state:'idle',message:answer.message,
                    items:{environment:{state:answer.started?'running':'pass',detail:answer.message}}});
                }
              });
            };
            const checkQuickEnvironment = () => {
              if (!pyHost || typeof pyHost.checkPythonEnvironment !== 'function') {
                quickEnvironment = {state:'fail', detail:qtOpText('noHostEnv')};
                showQuickResult({state:'idle',items:{environment:quickEnvironment}});
                return;
              }
              pyHost.checkPythonEnvironment(raw => {
                try {
                  quickEnvironment = typeof raw === 'string' ? JSON.parse(raw) : raw;
                  if (quickEnvironment && quickEnvironment.state === 'pass') {
                    try { localStorage.setItem(quickEnvironmentCacheKey, '1'); } catch (_) {}
                  }
                  const checks = quickEnvironment && Array.isArray(quickEnvironment.checks)
                    ? quickEnvironment.checks : [];
                  const failed = checks.filter(item => !item.ok);
                  const card = quickCards.find(item => item.dataset.quickKey === 'environment');
                  if (card) card.title = failed.length
                    ? failed.map(item => item.name+'：'+(item.detail||'未通过')).join('\n')
                    : qtOpText('envPass');
                  showQuickResult({state:'idle',items:{environment:quickEnvironment}});
                  if (quickEnvironment && quickEnvironment.state === 'fail')
                    requestEnvironmentInstall();
                } catch (_) {
                  quickEnvironment = {state:'fail',detail:qtOpText('envParseFail')};
                  showQuickResult({state:'idle',items:{environment:quickEnvironment}});
                }
              });
            };
            const startQuickUi = () => {
              const pending = {};
              quickCards.forEach(card => {
                pending[card.dataset.quickKey] = {state:'running',detail:qtOpText('waitSampling')};
              });
              if (quickEnvironment) pending.environment = quickEnvironment;
              quickRunning = true;
              showQuickResult({state:'running',message:qtOpText('testPreparing'),items:pending});
              if (typeof window.__qtStartSelectedRecordings === 'function')
                window.__qtStartSelectedRecordings();
            };
            quickStart.onclick = async () => {
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) {
                quickOverall.textContent = qtOpText('selectTaskFirst');
                quickOverall.className = 'qt-op-test-overall fail';
                return;
              }
              if (typeof imuOn === 'undefined' || !imuOn) {
                quickOverall.textContent = qtOpText('connectFirst');
                quickOverall.className = 'qt-op-test-overall fail';
                return;
              }
              if (!pyHost || typeof pyHost.startQuickHealthCheck !== 'function') {
                quickOverall.textContent = qtOpText('unsupportedTest');
                quickOverall.className = 'qt-op-test-overall fail';
                return;
              }
              if (typeof window.__qtEnsureRecordingDirectory === 'function' &&
                  !window.__qtEnsureRecordingDirectory()) return;
              checkQuickEnvironment();
              const mount = document.getElementById('imu-map');
              pyHost.startQuickHealthCheck(mount ? mount.value : 'ry-', message => {
                const text = String(message);
                if (text.indexOf('已启动') >= 0) startQuickUi();
                else {
                  quickRunning = false;
                  quickOverall.textContent = text;
                  quickOverall.className = 'qt-op-test-overall fail';
                  if (text.indexOf('Python运行环境未通过') >= 0)
                    requestEnvironmentInstall();
                }
              });
            };
            quickStop.onclick = () => {
              if (!pyHost || typeof pyHost.stopQuickHealthCheck !== 'function') return;
              pyHost.stopQuickHealthCheck(message => {
                quickRunning = false;
                stopSelectedRecordings();
                  quickOverall.textContent = localizeQuickDetail(message);
                quickOverall.className = 'qt-op-test-overall fail';
                quickStart.disabled = false; quickStop.disabled = true;
              });
            };
            if (pyHost && pyHost.rx && !window.__qtQuickHealthRx) {
              window.__qtQuickHealthRx = true;
              pyHost.rx.connect(text => {
                String(text).split(/\r?\n/).forEach(line => {
                  if (line.startsWith('[quickjson] ')) {
                    try { showQuickResult(JSON.parse(line.slice(12))); } catch (_) {}
                  } else if (line.startsWith('[envjson] ')) {
                    try {
                      const update=JSON.parse(line.slice(10));
                      quickEnvironment=update.environment||quickEnvironment;
                      const installing=update.state==='installing';
                      const shown=installing
                        ? {state:'running',detail:update.message||'正在自动安装'}
                        : Object.assign({},quickEnvironment||{state:'fail'},
                            {detail:update.message||quickEnvironment&&quickEnvironment.detail||'环境检测失败'});
                      showQuickResult({state:'idle',message:update.message||'',items:{environment:shown}});
                      if (update.state==='complete') checkQuickEnvironment();
                    } catch (_) {}
                  } else if (line.startsWith('[quick] ') && quickRunning) {
                    quickOverall.textContent = localizeQuickDetail(line.slice(8));
                  }
                });
              });
            }

            const sync = () => {
              const starting = typeof tcMpcStarting !== 'undefined' && tcMpcStarting;
              const running = (typeof tcPlaying !== 'undefined' && tcPlaying) || starting;
              const codexWaiting = !!window.__qtCodexCycleWaiting;
              const executionRunning = running || quickRunning || codexWaiting;
              if (previousExecutionRunning && !executionRunning) {
                stopSelectedRecordings();
                // A completed LLM+MPC lap briefly has neither tcPlaying nor
                // __qtCodexCycleWaiting set while the /step completion callback
                // and status poll hand control to the designer.  That gap is a
                // normal loop transition, not a user stop.  Keep the execution
                // plan alive so a ready candidate can automatically start the
                // next lap.  Explicit stop/cancel/mode-change paths already
                // deactivate the plan themselves.
                const plan = window.__qtTeachExecutionPlan;
                const keepCodexLoop = !!(plan && plan.active && plan.loop &&
                  teachController.value === 'mpc_codex' && teachLoop.checked);
                if (plan && !keepCodexLoop) plan.active = false;
              }
              previousExecutionRunning = executionRunning;
              const busy = executionRunning || recordingLaunchPending;
              run.textContent = running ? qtOpText('stopAuto') : qtOpText('startAuto');
              teachRun.textContent = executionRunning ? qtOpText('stopExecution') : qtOpText('startTeaching');
              autoModes.forEach(card => {
                card.querySelectorAll('button,input,select').forEach(control => { control.disabled = busy; });
              });
              opRecord.disabled = busy;
              opDefaultSelect.disabled = busy;
              opLoadDefault.disabled = busy;
              opClear.disabled = busy || !(typeof teach !== 'undefined' && teach.length);
              teachController.disabled = busy;
              teachLoop.disabled = busy || llmReplayActive;
              teachLimit.disabled = busy || llmReplayActive;
              teachUnit.disabled = busy || llmReplayActive;
              llmConfigButton.disabled = busy;
              clearMemoryButton.disabled = busy || clearMemoryRunning;
              saveProcessButton.disabled = busy || llmReplayBusy;
              loadReplayButton.disabled = busy || llmReplayBusy;
              loadReplay2Button.disabled = busy || llmReplayBusy;
              const timedAutomaticRunning = !!(autoQueue && autoQueue.active && autoQueue.deadlineMs);
              debugLoad2Timed.disabled = !timedAutomaticRunning && (busy || llmReplayBusy);
              debugLoad2Timed.textContent = timedAutomaticRunning
                ? '停止Automatic 30分钟任务' : 'Automatic 1→2→3 循环30分钟';
              testModifiedButton.disabled = busy || llmReplayBusy;
              exitReplayButton.disabled = busy || llmReplayBusy;
              renderReplayControls();
              opTeachList.querySelectorAll('[data-op-teach-delete]').forEach(button => {
                button.disabled = busy;
              });
              tabs.forEach(button => { button.disabled = busy; });
              quickStart.disabled = busy;
              quickStop.disabled = !quickRunning;
              const pointCount = (typeof teach !== 'undefined' && Array.isArray(teach)) ? teach.length : 0;
              const countText = qtOpEffectiveLanguage() === 'en'
                ? `${pointCount} ${qtOpText('points')}` : `${pointCount} ${qtOpText('points')}`;
              opTeachCount.textContent = `${qtOpText('recorded')} ${countText}`;
              renderTeachPoints();
              if (quickRunning) {
                status.textContent = qtOpText('quickRunning');
              } else if (running) {
                status.textContent = activeTab === 'teach'
                  ? `${qtOpText('runningTeaching')} | ${teachController.options[teachController.selectedIndex].text}`
                  : (autoQueue && autoQueue.active
                    ? `${qtOpText('runningQueue')} ${autoQueue.index+1}/${autoQueue.modes.length} | Mode ${selectedMode.slice(-1)}`
                    : `${qtOpText('runningMode')} ${selectedMode.slice(-1)}`);
              } else if (activeTab === 'auto') {
                const chosenModes = automaticModeOrder.filter(modeName => selectedModes.has(modeName));
                status.textContent = chosenModes.length
                  ? `${qtOpText('selectedMode')} ${chosenModes.map(modeName => modeName.slice(-1)).join(' → ')}`
                  : qtOpText('selectModeFirst');
              } else if (activeTab === 'teach') {
                const cfg = teachExecutionConfig();
                const loopDescription = cfg.loop
                  ? (cfg.limitValue != null && Number.isFinite(Number(cfg.limitValue))
                    ? `${cfg.limitValue} ${cfg.limitUnit === 'min' ? qtOpText('minutesUnit') : qtOpText('runsUnit')}`
                    : qtOpText('infiniteLoop'))
                  : qtOpText('singleRun');
                status.textContent = `${countText} | ${qtOpText('controller')}: ${teachController.options[teachController.selectedIndex].text} | ${loopDescription}`;
              } else {
                status.textContent = qtOpText('testStandby');
              }
            };
            window.__qtOperationExecutionIsRunning = () => {
              const fixedRunning = typeof window.__qtFixedOpenIsRunning === 'function' &&
                window.__qtFixedOpenIsRunning();
              return !!((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                (typeof tcMpcStarting !== 'undefined' && tcMpcStarting) ||
                (typeof tcMpcTimer !== 'undefined' && tcMpcTimer) || quickRunning ||
                window.__qtCodexCycleWaiting || fixedRunning);
            };
            showTab('auto');
            // 每台电脑第一次进入正式界面时自动检查一次；通过后本机记忆。
            // 用户以后打开“测试”页时仍会主动复查，以发现环境被改动的情况。
            let firstComputerEnvironmentCheck = true;
            try { firstComputerEnvironmentCheck = localStorage.getItem(quickEnvironmentCacheKey) !== '1'; }
            catch (_) {}
            if (firstComputerEnvironmentCheck) setTimeout(checkQuickEnvironment, 100);
            setInterval(sync, 250);
            sync();
            window.__qtApplyOperationLanguage();
          };
          installOperationModePanel();
          // 所有闭环控制的 /step 定时器统一从100ms提升到50ms（20Hz）。
          if (!window.__qtControl20Hz && typeof window.setInterval === 'function') {
            window.__qtControl20Hz = true;
            const rawSetInterval = window.setInterval.bind(window);
            window.setInterval = function(fn, delay, ...args) {
              const mode = document.getElementById('tc-drvmode');
              const fastModes = ['mpc','mpc_ilc','mpc_codex','mpcfel','new_mpc_fel','mpc_shadow',
                                  'pi','pifel','musclepi','open'];
              if (Number(delay) === 100 && mode && fastModes.includes(mode.value))
                delay = 50;
              return rawSetInterval(fn, delay, ...args);
            };
          }
          // MPC 的三轴角度/速度权重。界面只暴露两个标量，控制站自动复制到三个轴。
          const installMpcWeights = () => {
            const mode = document.getElementById('tc-drvmode');
            if (!mode || typeof window.fetch !== 'function') {
              setTimeout(installMpcWeights, 200); return;
            }
            if (document.getElementById('qt-mpc-weight-panel')) return;
            const panel = document.createElement('div');
            panel.id = 'qt-mpc-weight-panel'; panel.className = 'btns';
            panel.style.cssText = 'margin-top:5px;align-items:center';
            panel.title = 'Wq 和 Wv 分别自动应用到三个轴；所有MPC模式共用';
            panel.innerHTML = `
              <span class="aux">MPC 权重（三轴同值）</span>
              <label class="tcopt">Wq 角度 <input type="number" id="qt-mpc-wq" min="0" max="100000" step="100" value="5000" style="width:72px"></label>
              <label class="tcopt">Wv 速度 <input type="number" id="qt-mpc-wv" min="0" max="100000" step="10" value="300" style="width:68px"></label>
              <button class="btn" id="qt-mpc-weight-apply" title="运行中立即生效，并保存到 control/gains.json">应用 MPC 权重</button>
              <span class="aux" id="qt-mpc-weight-state">读取中…</span>`;
            mode.closest('.btns').insertAdjacentElement('afterend', panel);
            const wq = document.getElementById('qt-mpc-wq');
            const wv = document.getElementById('qt-mpc-wv');
            const btn = document.getElementById('qt-mpc-weight-apply');
            const state = document.getElementById('qt-mpc-weight-state');
            const url = 'http://127.0.0.1:8765/mpc_weights';
            let statusText = '读取中…';
            const showState = () => {
              const active = ['mpc','mpc_ilc','mpc_codex','mpcfel','new_mpc_fel','mpc_shadow'].includes(mode.value);
              panel.style.opacity = active ? '1' : '0.58';
              state.textContent = active ? statusText : '仅 MPC 系列有效';
            };
            const loadValues = async () => {
              try {
                const r = await (await fetch(url)).json();
                if (r.err) throw new Error(r.err);
                wq.value = r.wq; wv.value = r.wv; statusText = '已读取';
              } catch (_) { statusText = '控制站未连接'; }
              showState();
            };
            const applyValues = async () => {
              const payload = {wq:Number(wq.value), wv:Number(wv.value)};
              if (![payload.wq,payload.wv].every(v => Number.isFinite(v) && v >= 0 && v <= 100000)) {
                statusText = 'Wq/Wv 必须在 0～100000'; showState(); return;
              }
              btn.disabled = true; statusText = '应用中…'; showState();
              try {
                const r = await (await fetch(url, {method:'POST',
                  headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})).json();
                if (r.err) throw new Error(r.err);
                statusText = r.warn ? '已生效，保存失败' : (r.active ? '运行中已生效' : '已保存');
                if (typeof tcSay === 'function') tcSay('<b>MPC 权重已应用</b>：三轴 Wq='+r.wq+
                  '，三轴 Wv='+r.wv+(r.active?'（本次运行立即生效）':'（开始 MPC 后生效）')+
                  (r.warn?'；'+r.warn:''), !!r.warn);
              } catch (e) {
                statusText = '应用失败';
                if (typeof tcSay === 'function') tcSay('MPC 权重应用失败：'+e.message, true);
              } finally { btn.disabled = false; showState(); }
            };
            btn.onclick = applyValues;
            [wq,wv].forEach(el => el.addEventListener('keydown', e => {
              if (e.key === 'Enter') applyValues();
            }));
            mode.addEventListener('change', showState);
            loadValues();
          };
          // Qt 本地界面的在线 PID 面板。六路方向控制共用这一组增益；
          // 「应用」既更新正在运行的 PI/PI+FEL，也保存为下次启动值。
          const installPid = () => {
            const mode = document.getElementById('tc-drvmode');
            if (!mode || typeof window.fetch !== 'function') {
              setTimeout(installPid, 200); return;
            }
            if (document.getElementById('qt-pid-panel')) return;
            const panel = document.createElement('div');
            panel.id = 'qt-pid-panel'; panel.className = 'btns';
            panel.style.cssText = 'margin-top:5px;align-items:center';
            panel.title = '六路方向投影 PID 共用这一组增益；纯 PI 和 PI+FEL 模式有效';
            panel.innerHTML = `
              <span class="aux">PID</span>
              <label class="tcopt">Kp <input type="number" id="qt-pid-kp" min="0" max="100000" step="100" value="4500" style="width:68px"></label>
              <label class="tcopt">Ki <input type="number" id="qt-pid-ki" min="0" max="100000" step="100" value="1500" style="width:68px"></label>
              <label class="tcopt">Kd <input type="number" id="qt-pid-kd" min="0" max="100000" step="10" value="0" style="width:68px"></label>
              <label class="tcopt">Kv <input type="number" id="qt-pid-kv" min="0" max="100000" step="100" value="1500" style="width:68px"></label>
              <label class="tcopt">Vmax°/s <input type="number" id="qt-pid-vmax" min="1" max="180" step="1" value="30" style="width:58px"></label>
              <button class="btn" id="qt-pid-apply" title="运行中立即生效，并保存到 control/gains.json">应用 PID</button>
              <span class="aux" id="qt-pid-state">读取中…</span>`;
            mode.closest('.btns').insertAdjacentElement('afterend', panel);
            const kp = document.getElementById('qt-pid-kp');
            const ki = document.getElementById('qt-pid-ki');
            const kd = document.getElementById('qt-pid-kd');
            const kv = document.getElementById('qt-pid-kv');
            const vmax = document.getElementById('qt-pid-vmax');
            const btn = document.getElementById('qt-pid-apply');
            const state = document.getElementById('qt-pid-state');
            const pidUrl = 'http://127.0.0.1:8765/pid';
            let statusText = '读取中…';
            const showState = () => {
              const active = mode.value === 'pi' || mode.value === 'pifel';
              panel.style.opacity = active ? '1' : '0.58';
              state.textContent = active ? statusText : '仅 PI / PI+FEL 有效';
            };
            const loadPid = async () => {
              try {
                const r = await (await fetch(pidUrl)).json();
                if (r.err) throw new Error(r.err);
                kp.value = r.kp; ki.value = r.ki; kd.value = r.kd;
                kv.value = r.kv; vmax.value = r.vmax;
                statusText = '已读取';
              } catch (_) { statusText = '控制站未连接'; }
              showState();
            };
            const applyPid = async () => {
              const payload = {kp:Number(kp.value), ki:Number(ki.value), kd:Number(kd.value),
                kv:Number(kv.value), vmax:Number(vmax.value)};
              if ([payload.kp,payload.ki,payload.kd,payload.kv].some(v => !Number.isFinite(v) || v < 0 || v > 100000)
                  || !Number.isFinite(payload.vmax) || payload.vmax < 1 || payload.vmax > 180) {
                statusText = '增益0～100000，Vmax 1～180'; showState(); return;
              }
              btn.disabled = true; statusText = '应用中…'; showState();
              try {
                const r = await (await fetch(pidUrl, {method:'POST',
                  headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})).json();
                if (r.err) throw new Error(r.err);
                statusText = r.warn ? '已生效，保存失败' : (r.active ? '运行中已生效' : '已保存');
                if (typeof tcSay === 'function') tcSay('<b>PID 已应用</b>：Kp='+r.kp+'，Ki='+r.ki+'，Kd='+r.kd+
                  '，速度Kv='+r.kv+'，Vmax='+r.vmax+'°/s'+
                  (r.active?'（本次运行立即生效）':'（开始 PI 后生效）')+(r.warn?'；'+r.warn:''), !!r.warn);
              } catch (e) {
                statusText = '应用失败';
                if (typeof tcSay === 'function') tcSay('PID 应用失败：'+e.message, true);
              } finally { btn.disabled = false; showState(); }
            };
            btn.onclick = applyPid;
            [kp,ki,kd,kv,vmax].forEach(el => el.addEventListener('keydown', e => {
              if (e.key === 'Enter') applyPid();
            }));
            mode.addEventListener('change', showState);
            loadPid();
          };
          // 已完成轴的压力保持内环。与角度 PID 分开调整，运行中立即生效。
          const installPressurePid = () => {
            const mode = document.getElementById('tc-drvmode');
            const pidPanel = document.getElementById('qt-pid-panel');
            if (!mode || !pidPanel || typeof window.fetch !== 'function') {
              setTimeout(installPressurePid, 200); return;
            }
            if (document.getElementById('qt-pressure-pid-panel')) return;
            const panel = document.createElement('div');
            panel.id = 'qt-pressure-pid-panel'; panel.className = 'btns';
            panel.style.cssText = 'margin-top:5px;align-items:center';
            panel.title = '切换轴后，用 ADC 实测压力保持已完成轴；仅 PI / PI+FEL 有效';
            panel.innerHTML = `
              <span class="aux">保压内环</span>
              <label class="tcopt">Kp mV/kPa <input type="number" id="qt-pressure-kp" min="0" max="1000" step="1" value="35" style="width:62px"></label>
              <label class="tcopt">Ki mV/(kPa·s) <input type="number" id="qt-pressure-ki" min="0" max="1000" step="1" value="18" style="width:62px"></label>
              <label class="tcopt">死区 kPa <input type="number" id="qt-pressure-dead" min="0" max="30" step="0.1" value="1.5" style="width:58px"></label>
              <button class="btn" id="qt-pressure-apply" title="运行中立即生效，并保存到 control/gains.json">应用气压环</button>
              <span class="aux" id="qt-pressure-state">读取中…</span>`;
            pidPanel.insertAdjacentElement('afterend', panel);
            const kp = document.getElementById('qt-pressure-kp');
            const ki = document.getElementById('qt-pressure-ki');
            const dead = document.getElementById('qt-pressure-dead');
            const btn = document.getElementById('qt-pressure-apply');
            const state = document.getElementById('qt-pressure-state');
            const url = 'http://127.0.0.1:8765/pressure_pid';
            let statusText = '读取中…';
            const showState = () => {
              const active = mode.value === 'pi' || mode.value === 'pifel';
              panel.style.opacity = active ? '1' : '0.58';
              state.textContent = active ? statusText : '仅 PI / PI+FEL 有效';
            };
            const loadValues = async () => {
              try {
                const r = await (await fetch(url)).json();
                if (r.err) throw new Error(r.err);
                kp.value = r.kp; ki.value = r.ki; dead.value = r.deadband;
                statusText = '已读取';
              } catch (_) { statusText = '控制站未连接'; }
              showState();
            };
            const applyValues = async () => {
              const payload = {kp:Number(kp.value), ki:Number(ki.value), deadband:Number(dead.value)};
              if (!Number.isFinite(payload.kp) || payload.kp < 0 || payload.kp > 1000 ||
                  !Number.isFinite(payload.ki) || payload.ki < 0 || payload.ki > 1000 ||
                  !Number.isFinite(payload.deadband) || payload.deadband < 0 || payload.deadband > 30) {
                statusText = 'Kp/Ki 0～1000，死区 0～30'; showState(); return;
              }
              btn.disabled = true; statusText = '应用中…'; showState();
              try {
                const r = await (await fetch(url, {method:'POST',
                  headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})).json();
                if (r.err) throw new Error(r.err);
                statusText = r.warn ? '已生效，保存失败' : (r.active ? '运行中已生效' : '已保存');
                if (typeof tcSay === 'function') tcSay('<b>保压内环已应用</b>：Kp='+r.kp+
                  '，Ki='+r.ki+'，死区='+r.deadband+' kPa'+
                  (r.active?'（本次运行立即生效）':'（开始 PI 后生效）')+(r.warn?'；'+r.warn:''), !!r.warn);
              } catch (e) {
                statusText = '应用失败';
                if (typeof tcSay === 'function') tcSay('保压内环应用失败：'+e.message, true);
              } finally { btn.disabled = false; showState(); }
            };
            btn.onclick = applyValues;
            [kp,ki,dead].forEach(el => el.addEventListener('keydown', e => {
              if (e.key === 'Enter') applyValues();
            }));
            mode.addEventListener('change', showState);
            loadValues();
          };
          // 六组真实肌肉长度 PI 参数；示教点的 L 作为目标，当前20路传感器计算的 L 作为反馈。
          const installMusclePid = () => {
            const mode = document.getElementById('tc-drvmode');
            const pressurePanel = document.getElementById('qt-pressure-pid-panel');
            if (!mode || !pressurePanel || typeof window.fetch !== 'function') {
              setTimeout(installMusclePid, 200); return;
            }
            if (document.getElementById('qt-muscle-pid-panel')) return;
            const panel = document.createElement('div');
            panel.id = 'qt-muscle-pid-panel'; panel.className = 'btns';
            panel.style.cssText = 'margin-top:5px;align-items:center';
            panel.title = '六个物理阀组各自追踪示教时保存的真实平均肌肉长度；角度仅显示';
            panel.innerHTML = `
              <span class="aux">肌肉长度 PI</span>
              <label class="tcopt">Kp mV/mm <input type="number" id="qt-muscle-kp" min="0" max="2000" step="5" value="120" style="width:62px"></label>
              <label class="tcopt">Ki mV/(mm·s) <input type="number" id="qt-muscle-ki" min="0" max="2000" step="1" value="20" style="width:62px"></label>
              <label class="tcopt">死区 mm <input type="number" id="qt-muscle-dead" min="0" max="20" step="0.1" value="0.5" style="width:58px"></label>
              <button class="btn" id="qt-muscle-apply" title="运行中立即生效，并保存到 control/gains.json">应用长度 PI</button>
              <span class="aux" id="qt-muscle-state">读取中…</span>`;
            pressurePanel.insertAdjacentElement('afterend', panel);
            const kp = document.getElementById('qt-muscle-kp');
            const ki = document.getElementById('qt-muscle-ki');
            const dead = document.getElementById('qt-muscle-dead');
            const btn = document.getElementById('qt-muscle-apply');
            const state = document.getElementById('qt-muscle-state');
            const url = 'http://127.0.0.1:8765/muscle_pid';
            let statusText = '读取中…';
            const showState = () => {
              const active = mode.value === 'musclepi';
              panel.style.opacity = active ? '1' : '0.58';
              state.textContent = active ? statusText : '仅肌肉长度 PI 有效';
            };
            const loadValues = async () => {
              try {
                const r = await (await fetch(url)).json();
                if (r.err) throw new Error(r.err);
                kp.value = r.kp; ki.value = r.ki; dead.value = r.deadband;
                statusText = '已读取';
              } catch (_) { statusText = '控制站未连接'; }
              showState();
            };
            const applyValues = async () => {
              const payload = {kp:Number(kp.value), ki:Number(ki.value), deadband:Number(dead.value)};
              if (!Number.isFinite(payload.kp) || payload.kp < 0 || payload.kp > 2000 ||
                  !Number.isFinite(payload.ki) || payload.ki < 0 || payload.ki > 2000 ||
                  !Number.isFinite(payload.deadband) || payload.deadband < 0 || payload.deadband > 20) {
                statusText = 'Kp/Ki 0～2000，死区 0～20'; showState(); return;
              }
              btn.disabled = true; statusText = '应用中…'; showState();
              try {
                const r = await (await fetch(url, {method:'POST',
                  headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})).json();
                if (r.err) throw new Error(r.err);
                statusText = r.warn ? '已生效，保存失败' : (r.active ? '运行中已生效' : '已保存');
                if (typeof tcSay === 'function') tcSay('<b>肌肉长度 PI 已应用</b>：Kp='+r.kp+
                  ' mV/mm，Ki='+r.ki+' mV/(mm·s)，死区='+r.deadband+' mm'+
                  (r.active?'（本次运行立即生效）':'（选择肌肉长度 PI 后生效）')+
                  (r.warn?'；'+r.warn:''), !!r.warn);
              } catch (e) {
                statusText = '应用失败';
                if (typeof tcSay === 'function') tcSay('肌肉长度 PI 应用失败：'+e.message, true);
              } finally { btn.disabled = false; showState(); }
            };
            btn.onclick = applyValues;
            [kp,ki,dead].forEach(el => el.addEventListener('keydown', e => {
              if (e.key === 'Enter') applyValues();
            }));
            mode.addEventListener('change', showState);
            loadValues();
          };
          // 控制站的实时阶段指示：阶段编号按示教段计算，同一段内切轴只更新说明。
          // 同时让示教点列表跳到当前段，并用醒目的边框标出正在执行的段。
          const installStage = () => {
            const mode = document.getElementById('tc-drvmode');
            if (!mode || typeof window.fetch !== 'function') {
              setTimeout(installStage, 200); return;
            }
            if (document.getElementById('qt-stage-panel')) return;
            if (!document.getElementById('qt-stage-style')) {
              const style = document.createElement('style');
              style.id = 'qt-stage-style';
              style.textContent =
                '#qt-measured-contraction{margin:6px 0 8px;padding:8px 10px;border:1px solid #78aee6;'+
                'border-radius:6px;background:rgba(30,136,229,.08)}'+
                '#qt-measured-contraction>small{color:var(--muted);margin-left:6px}'+
                '#qt-measured-contraction #enc-lens{display:grid;grid-template-columns:minmax(0,1fr);'+
                'gap:5px 10px;justify-content:stretch;margin-top:7px;font-size:12px}'+
                '#qt-measured-contraction #enc-lens[hidden]{display:none}'+
                '#qt-measured-contraction #enc-lens span{padding:4px 6px;border-radius:4px;background:var(--surface);'+
                'text-align:left;white-space:nowrap;border-left:4px solid transparent;transition:color .15s,background .15s,border-color .15s}'+
                '#qt-measured-contraction #enc-lens .qt-sensor-pair i{display:inline-block;min-width:54px;'+
                'color:var(--accent);font-style:normal;font-weight:700}'+
                '#qt-measured-contraction #enc-lens .qt-sensor-pressure{font-style:normal;margin-left:12px;font-weight:700}'+
                '#qt-measured-contraction #enc-lens .qt-muscle-driven{color:#d32f2f;border-left-color:#d32f2f;background:rgba(211,47,47,.12)}'+
                '#qt-measured-contraction #enc-lens .qt-muscle-driven i,'+
                '#qt-measured-contraction #enc-lens .qt-muscle-driven b,'+
                '#qt-measured-contraction #enc-lens .qt-muscle-driven em{color:#d32f2f}'+
                '#qt-measured-contraction #enc-lens .qt-muscle-idle{color:#238636;border-left-color:#238636;background:rgba(35,134,54,.10)}'+
                '#qt-measured-contraction #enc-lens .qt-muscle-idle i,'+
                '#qt-measured-contraction #enc-lens .qt-muscle-idle b,'+
                '#qt-measured-contraction #enc-lens .qt-muscle-idle em{color:#238636}'+
                '#qt-measured-contraction #enc-lens .qt-muscle-unknown{color:var(--muted);border-left-color:var(--muted)}'+
                '#qt-measured-contraction #enc-lens .qt-sensor-status{grid-column:1/-1;text-align:center}'+
                '#tc-list .tcrow.on{border:3px solid var(--accent)!important;'+
                'background:rgba(30,136,229,.20)!important;'+
                'box-shadow:0 0 0 2px rgba(30,136,229,.22),0 3px 10px rgba(0,0,0,.18);'+
                'font-weight:700}'+
                '#tc-list .tcrow.on i{color:#fff;background:var(--accent);'+
                'border-radius:4px;padding:2px 4px;text-align:center}'+
                '#tc-list .tcrow{grid-template-columns:26px minmax(0,1fr) auto auto!important}'+
                '#tc-list .tcq{white-space:pre-line!important;overflow:visible!important;'+
                'text-overflow:clip!important;line-height:1.45}'+
                '#tc-list .tcrow>span:not(.tcq){display:none!important}'+
                '#tc-list .qt-plan-time{color:var(--accent);font-weight:800;margin-right:4px}'+
                '#tc-list .qt-plan-time.extended{color:var(--warn)}';
              document.head.appendChild(style);
            }
            // 原页面的“±6.13°”不是公差，而是 GUI 静力模型与记录角度的差值。
            // 只改显示名称，数值及示教数据保持不变。
            const clarifyPoseErrors = () => {
              document.querySelectorAll('#tc-list .tcrow span').forEach(el => {
                const m = el.textContent.trim().match(/^±([0-9.]+)°$/);
                if (!m) return;
                el.textContent = '模型差 ' + m[1] + '°';
                el.classList.add('qt-model-gap');
                el.title = 'GUI 静力模型预测姿态与记录角度的最大差值；仅供诊断，不参与 PI/MPC 控制或阶段规划';
              });
            };
            const refreshTeachRows = () => {
              clarifyPoseErrors();
              const timing = Array.isArray(window.__qtSegmentTiming) ? window.__qtSegmentTiming : [];
              document.querySelectorAll('#tc-list .tcrow').forEach((row, k) => {
                const cell = row.querySelector('.tcq');
                if (!cell) return;
                const p = (typeof teach !== 'undefined' && teach[k]) ? teach[k] : null;
                const q = p && Array.isArray(p.q_raw) ? p.q_raw : (p ? p.q : null);
                if (!Array.isArray(q) || q.length < 3) return;
                const angle = v => {
                  const n = Number(v) || 0;
                  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}°`;
                };
                const item = timing[k];
                const planned = item ? Number(item.planned_s) : Number(p.t);
                const timeText = Number.isFinite(planned) ? ` ｜ 规划 ${planned.toFixed(2)}s` : '';
                let lengths = p && Array.isArray(p.channel_length_mm) ? p.channel_length_mm : null;
                if (p && Array.isArray(p.lens_sensor) && p.lens_sensor.length >= 20) {
                  const lm = Array.from({length:10}, (_, j) => {
                    const av = p.lens_sensor[j], bv = p.lens_sensor[j+10];
                    const a = Number(av), b = Number(bv);
                    return av != null && bv != null && Number.isFinite(a) && Number.isFinite(b)
                      ? 235-Math.abs(a)+Math.abs(b) : null;
                  });
                  lengths = [[0,4],[1,2],[3,7],[5,6],[8],[9]].map(ids => {
                    const vals = ids.map(j => lm[j]);
                    return vals.every(v => v != null)
                      ? vals.reduce((a,b) => a+b,0)/vals.length : null;
                  });
                }
                const lengthText = mode.value === 'musclepi' && lengths && lengths.length >= 6
                  ? `\nL[15,23,48,67,9,10] = ${lengths.map(v => Number.isFinite(Number(v)) ? Number(v).toFixed(1) : '--').join('/') } mm`
                  : '';
                cell.textContent = `X ${angle(q[0])} ｜ Y ${angle(q[1])}\nφ ${angle(q[2])}${timeText}${lengthText}`;
              });
            };
            refreshTeachRows();
            const list = document.getElementById('tc-list');
            if (list) new MutationObserver(refreshTeachRows).observe(list, {childList:true});
            mode.addEventListener('change', refreshTeachRows);
            const panel = document.createElement('div');
            panel.id = 'qt-stage-panel';
            panel.textContent = '当前阶段：待机';
            panel.style.cssText = 'margin-top:6px;padding:7px 10px;border:1px solid #8bb9e8;'+
              'border-radius:6px;background:#eaf4ff;color:#174a7e;font-size:15px;font-weight:700;text-align:center';
            mode.closest('.btns').insertAdjacentElement('afterend', panel);
            const showStage = r => {
              if (!r || !r.stage) return;
              panel.textContent = '▶ 正在执行：' + r.stage;
              const point = Number(r.stage_point);
              const key = String(r.stage) + ':' + point;
              if (key === window.__qtLastStageKey) return;
              window.__qtLastStageKey = key;
              if (typeof tcRender === 'function') {
                tcRender(Number.isInteger(point) && point >= 0 ? point : -1);
                setTimeout(() => {
                  const active = document.querySelector('#tc-list .tcrow.on');
                  if (active) active.scrollIntoView({block:'nearest', behavior:'smooth'});
                }, 0);
              }
            };
            if (!window.__qtStageFetch) {
              window.__qtStageFetch = true;
              const rawFetch = window.fetch.bind(window);
              window.fetch = async function(input, init) {
                const response = await rawFetch(input, init);
                const url = String(input && input.url ? input.url : input);
                if (url.endsWith('/step')) {
                  response.clone().json().then(r => {
                    showStage(r);
                    if (r && r.mode === 'done' &&
                        typeof window.__qtStartCodexCycleWait === 'function')
                      window.__qtStartCodexCycleWait();
                  }).catch(() => {});
                } else if (url.endsWith('/start')) {
                  panel.textContent = '当前阶段：准备开始';
                  window.__qtLastStageKey = '';
                  response.clone().json().then(r => {
                    window.__qtSegmentTiming = Array.isArray(r.segment_timing) ? r.segment_timing : [];
                    refreshTeachRows();
                  }).catch(() => {});
                }
                return response;
              };
            }
          };
          // 复用原“跑 Demo”按钮：同一轮自动充放气同时辨识DataModel和V6几何。
          const installIdent = () => {
            const btn = document.getElementById('imu-demo');
            if (!btn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startDataModelIdent !== 'function' ||
                typeof pyHost.startV3RepeatTest !== 'function') {
              setTimeout(installIdent, 200);
              return;
            }
            const jointText = '辨识 DataModel + V6几何';
            btn.textContent = jointText;
            btn.title = '一键自动扫描V0～V5的600/1200/1800/1200/600mV充放气段；同一批姿态、压力、锁状态和20路编码器数据同时拟合DataModel与V6几何候选；急停随时有效';
            btn.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              if (!window.confirm(
                  '将自动依次扫描 V0～V5，每路执行 600/1200/1800/1200/600 mV，' +
                  '并在通道间完全放气。请确认机械臂周围无人、行程无遮挡。是否开始联合辨识？')) return;
              const me = document.getElementById('imu-map');
              pyHost.startDataModelIdent(me ? me.value : 'ry-', msg => {
                const ok = String(msg).indexOf('已启动') >= 0;
                if (ok) btn.textContent = '联合辨识中…';
                imuSay(String(msg), !ok);
              });
            };
            let v3btn = document.getElementById('qt-v3-repeat-test');
            if (!v3btn) {
              v3btn = document.createElement('button');
              v3btn.id = 'qt-v3-repeat-test';
              v3btn.className = 'btn';
              v3btn.textContent = '重复测试 V3（仅验证）';
              v3btn.title = '只扫描V3=6+7号肌肉；每次600/1200/1800/1200/600mV各5秒；完全放气后重复；只计算Kendall tau-b和直接RMSE，不更新任何模型或六路候选';
              (btn.closest('.btns') || btn.parentElement).appendChild(v3btn);
              v3btn.onclick = () => {
                if (typeof imuOn === 'undefined' || !imuOn) {
                  imuSay('先点「连接 IMU」。', true); return;
                }
                if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                    (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
                const raw = window.prompt('连续重复测试V3多少次？（1–10；推荐3次）', '3');
                if (raw === null) return;
                const repeats = Math.max(1, Math.min(10, parseInt(raw, 10) || 3));
                if (!window.confirm(
                    `将只驱动V3（6+7号肌肉）并自动重复${repeats}次，每次约25秒，` +
                    '重复间完全放气。结果不会写入任何模型。请确认运动范围无人，是否开始？')) return;
                const me = document.getElementById('imu-map');
                pyHost.startV3RepeatTest(me ? me.value : 'ry-', repeats, msg => {
                  const text = String(msg);
                  const ok = text.indexOf('已启动') >= 0;
                  if (ok) {
                    v3btn.disabled = true;
                    v3btn.textContent = 'V3重复测试中…';
                  }
                  imuSay('<b>V3重复测试：</b> ' + text, !ok);
                });
              };
            }
            if (!window.__qtIdentRx) {
              window.__qtIdentRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[ident]')) return;
                  const msg = line.slice(7).trim();
                  try { imuDemoMsg = msg; imuLastUi = 0; imuTick(); }
                  catch (_) { imuSay('<b>辨识：</b> ' + msg, /失败|中止/.test(msg)); }
                  if (/辨识完成|辨识失败|辨识已中止|V6几何拟合失败/.test(msg)) btn.textContent = jointText;
                });
              });
            }
            if (!window.__qtV3RepeatRx) {
              window.__qtV3RepeatRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[v3repeat]')) return;
                  const msg = line.slice(10).trim();
                  imuSay('<b>V3重复测试：</b> ' + msg,
                    /失败|中止|超时/.test(msg));
                  if (/分析完成|分析失败|测试已中止/.test(msg)) {
                    v3btn.disabled = false;
                    v3btn.textContent = '重复测试 V3（仅验证）';
                  }
                });
              });
            }
          };
          // 用已知 DAC 压力标定压力反馈 ADC。只使用 V0（肌肉 1、5）。
          const installAdcCal = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startAdcCalibration !== 'function') {
              setTimeout(installAdcCal, 200);
              return;
            }
            if (document.getElementById('qt-adc-cal')) return;
            const btn = document.createElement('button');
            btn.id = 'qt-adc-cal';
            btn.className = 'btn';
            btn.textContent = '标定压力 ADC（V0）';
            btn.title = 'V0 自动输出 250～2200 mV（最高约198kPa），每档连续稳定10秒后建立 ADC raw→kPa 分段表；急停随时有效';
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            row.appendChild(btn);
            btn.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              pyHost.startAdcCalibration(msg => {
                const ok = String(msg).indexOf('已启动') >= 0;
                if (ok) btn.textContent = 'ADC 标定中…';
                imuSay(String(msg), !ok);
              });
            };
            if (!window.__qtAdcCalRx) {
              window.__qtAdcCalRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[adccal]')) return;
                  const msg = line.slice(8).trim();
                  imuSay('<b>压力 ADC 标定：</b> ' + msg,
                    /失败|中止|超时|不可/.test(msg));
                  if (/完成|失败|中止|超时/.test(msg))
                    btn.textContent = '标定压力 ADC（V0）';
                });
              });
            }
          };
          // 六路手动目标压力：无预设档位/时长，ADC 闭环保压，直到用户停止。
          const installManualPressureTest = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.setManualPressureTargets !== 'function' ||
                typeof pyHost.stopManualPressureTest !== 'function') {
              setTimeout(installManualPressureTest, 200);
              return;
            }
            if (document.getElementById('qt-manual-pressure-panel')) return;
            const panel = document.createElement('div');
            panel.id = 'qt-manual-pressure-panel';
            panel.className = 'btns';
            panel.style.cssText = 'margin-top:5px;align-items:center;flex-wrap:wrap';
            panel.title = '输入不设预定义档位和时长；仅此手动压力功能允许对抗肌同时加压；仍保留3000mV硬上限和各通道100ms预锁';
            const defs = [
              ['V0｜1+5',0], ['V1｜4+8',1], ['V2｜2+3',2],
              ['V3｜6+7',3], ['V4｜9',4], ['V5｜10',5]
            ];
            panel.innerHTML = '<span class="aux"><b>六路手动压力 kPa</b></span>' +
              defs.map(([label,i]) =>
                `<label class="tcopt">${label} <input type="number" id="qt-manp-${i}" `+
                `min="0" step="5" value="0" style="width:64px"></label>`).join('') +
              '<button class="btn" id="qt-manp-apply">开始/更新压力</button>' +
              '<button class="btn" id="qt-manp-stop">停止并缓慢放气</button>' +
              '<span class="aux" id="qt-manp-state">未运行</span>' +
              '<span class="aux"><b>9号自锁：</b>V4比例阀始终0mV</span>' +
              '<button class="btn" id="qt-lock9-start">持续锁定9号</button>' +
              '<button class="btn" id="qt-lock9-stop" disabled>停止9号自锁</button>' +
              '<span class="aux" id="qt-lock9-state">未测试</span>' +
              '<span class="aux"><b>10号自锁：</b>V5比例阀始终0mV</span>' +
              '<button class="btn" id="qt-lock10-start">持续锁定10号</button>' +
              '<button class="btn" id="qt-lock10-stop" disabled>停止10号自锁</button>' +
              '<span class="aux" id="qt-lock10-state">未测试</span>';
            const anchor = document.getElementById('qt-pressure-pid-panel');
            if (anchor) anchor.insertAdjacentElement('afterend', panel);
            else (identBtn.closest('.btns') || identBtn.parentElement).insertAdjacentElement('afterend', panel);
            const apply = document.getElementById('qt-manp-apply');
            const stop = document.getElementById('qt-manp-stop');
            const state = document.getElementById('qt-manp-state');
            const lock9Start = document.getElementById('qt-lock9-start');
            const lock9Stop = document.getElementById('qt-lock9-stop');
            const lock9State = document.getElementById('qt-lock9-state');
            const lock10Start = document.getElementById('qt-lock10-start');
            const lock10Stop = document.getElementById('qt-lock10-stop');
            const lock10State = document.getElementById('qt-lock10-state');
            let active = false;
            let lock9Timer = 0;
            let lock9StartedAt = 0;
            let lock10Timer = 0;
            let lock10StartedAt = 0;
            const targets = () => defs.map((_,i) =>
              Number(document.getElementById('qt-manp-'+i).value));
            apply.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              const values = targets();
              if (values.some(v => !Number.isFinite(v) || v < 0)) {
                imuSay('六路目标压力必须是非负数。', true); return;
              }
              if (!active) {
                if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              }
              pyHost.setManualPressureTargets(JSON.stringify(values), msg => {
                const text = String(msg);
                if (/已启动|已更新/.test(text)) active = true;
                if (/缓慢放气/.test(text)) active = true;
                if (/已停止/.test(text)) active = false;
                state.textContent = text;
                const releasing = /缓慢放气/.test(text);
                apply.disabled = releasing;
                stop.disabled = releasing;
                apply.textContent = releasing ? '缓慢放气中' :
                  (active ? '更新六路压力' : '开始/更新压力');
                imuSay('<b>六路手动压力：</b> ' + text,
                  /无效|不能|缺少|请先|失败/.test(text));
              });
            };
            stop.onclick = () => {
              pyHost.stopManualPressureTest(msg => {
                const text = String(msg);
                const releasing = /缓慢放气/.test(text);
                active = releasing;
                apply.disabled = releasing;
                stop.disabled = releasing;
                apply.textContent = releasing ? '缓慢放气中' : '开始/更新压力';
                state.textContent = text;
                imuSay('<b>六路手动压力：</b> ' + text, false);
              });
            };
            const stopLock9Test = (message, sendVent=true) => {
              if (lock9Timer) clearInterval(lock9Timer);
              lock9Timer = 0;
              lock9Start.disabled = false;
              lock9Stop.disabled = true;
              if (sendVent && typeof imuSend === 'function') imuSend('s\n');
              lock9State.textContent = message || '已停止并全机放气';
            };
            lock9Start.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if (active) {
                imuSay('请先停止六路手动压力测试。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              if (lock9Timer) return;
              if (lock10Timer) stopLock10Test('已切换到9号测试', false);
              lock9StartedAt = performance.now();
              lock9Start.disabled = true;
              lock9Stop.disabled = false;
              const keepLock9 = () => {
                if (typeof imuOn === 'undefined' || !imuOn) {
                  stopLock9Test('串口已断开，测试停止', false); return;
                }
                // 锁请求位持续为1；六路比例阀始终为0。固件把逻辑1映射为锁阀GPIO持续LOW。
                imuSend('M0,0,0,0,1,0;0,0,0,0,0,0\n');
                const elapsedMs = Math.max(0, performance.now()-lock9StartedAt);
                const observed = typeof imuLocks === 'string' && imuLocks.length > 4
                  ? imuLocks[4] : '?';
                lock9State.textContent = elapsedMs < 100
                  ? `预锁中 ${Math.round(elapsedMs)}/100ms｜遥测=${observed}`
                  : `持续锁定｜锁请求=1｜GPIO=LOW｜遥测=${observed}`;
              };
              keepLock9();
              lock9Timer = setInterval(keepLock9, 50);
              imuSay('<b>9号自锁测试：</b>V4保持0mV，锁请求持续为1；100ms后应显示锁状态L。', false);
            };
            lock9Stop.onclick = () => stopLock9Test('已停止并全机放气', true);
            const stopLock10Test = (message, sendVent=true) => {
              if (lock10Timer) clearInterval(lock10Timer);
              lock10Timer = 0;
              lock10Start.disabled = false;
              lock10Stop.disabled = true;
              if (sendVent && typeof imuSend === 'function') imuSend('s\n');
              lock10State.textContent = message || '已停止并全机放气';
            };
            lock10Start.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if (active) {
                imuSay('请先停止六路手动压力测试。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              if (lock10Timer) return;
              if (lock9Timer) stopLock9Test('已切换到10号测试', false);
              lock10StartedAt = performance.now();
              lock10Start.disabled = true;
              lock10Stop.disabled = false;
              const keepLock10 = () => {
                if (typeof imuOn === 'undefined' || !imuOn) {
                  stopLock10Test('串口已断开，测试停止', false); return;
                }
                // 10号对应V5/D7：锁请求持续为1，六路比例阀始终保持0mV。
                imuSend('M0,0,0,0,0,1;0,0,0,0,0,0\n');
                const elapsedMs = Math.max(0, performance.now()-lock10StartedAt);
                const observed = typeof imuLocks === 'string' && imuLocks.length > 5
                  ? imuLocks[5] : '?';
                lock10State.textContent = elapsedMs < 100
                  ? `预锁中 ${Math.round(elapsedMs)}/100ms｜遥测=${observed}`
                  : `持续锁定｜锁请求=1｜GPIO=LOW｜遥测=${observed}`;
              };
              keepLock10();
              lock10Timer = setInterval(keepLock10, 50);
              imuSay('<b>10号自锁测试：</b>V5保持0mV，锁请求持续为1；100ms后应显示锁状态L。', false);
            };
            lock10Stop.onclick = () => stopLock10Test('已停止并全机放气', true);
            if (!window.__qtManualPressureRx) {
              window.__qtManualPressureRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[manp]')) return;
                  const msg = line.slice(6).trim();
                  state.textContent = msg;
                  imuSay('<b>六路手动压力：</b> ' + msg,
                    /失败|中断|中止/.test(msg));
                  if (/测试已停止/.test(msg)) {
                    active = false;
                    apply.disabled = false;
                    stop.disabled = false;
                    apply.textContent = '开始/更新压力';
                  } else if (/测试已启动|目标已更新/.test(msg)) {
                    active = true;
                    apply.disabled = false;
                    stop.disabled = false;
                    apply.textContent = '更新六路压力';
                  } else if (/缓慢放气/.test(msg)) {
                    active = true;
                    apply.disabled = true;
                    stop.disabled = true;
                    apply.textContent = '缓慢放气中';
                  }
                });
              });
            }
          };
          const installReachabilityMap = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startReachabilityMap !== 'function' ||
                typeof pyHost.startReachabilityExtrema !== 'function' ||
                typeof pyHost.startSlowRomSampling !== 'function' ||
                typeof pyHost.startSlowRomGradualSampling !== 'function' ||
                typeof pyHost.calculateTheoreticalReachability !== 'function') {
              setTimeout(installReachabilityMap, 200);
              return;
            }
            if (document.getElementById('qt-reachability-map')) return;
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            const theory = document.createElement('button');
            theory.id = 'qt-theoretical-reachability-map';
            theory.className = 'btn';
            theory.textContent = '计算 V6 理论可达图';
            theory.title = '纯离线逆静力学：随机候选姿态，求0–200kPa且对抗通道互斥的静力平衡压力；不会连接或控制真机';
            const start = document.createElement('button');
            start.id = 'qt-reachability-map';
            start.className = 'btn';
            start.textContent = '采样 Reachability（≤200 kPa）';
            start.title = '随机六路压力组合；每对拮抗肌只允许单边加压；逐点放气、预锁100ms、稳压后记录姿态并自动画图';
            const extrema = document.createElement('button');
            extrema.id = 'qt-reachability-extrema';
            extrema.className = 'btn';
            extrema.textContent = '实机极值采点（18点）';
            extrema.title = '只测6个单通道和12个非拮抗双通道；活动通道固定2200mV，测量标称200kPa ROM；不使用经验X/Y/φ范围截断';
            const slowRom = document.createElement('button');
            slowRom.id = 'qt-slow-rom-sampling';
            slowRom.className = 'btn';
            slowRom.textContent = '慢速全ROM采样（26点）';
            slowRom.title = '6个单通道、12个合法双通道和8个三轴组合；100mV/s缓慢升压；工况间全放气，若X/Y/φ未回到±3°则反向预锁并短促充气一次；不设软件压力或角度保护线；约20Hz response.xlsx保存到操作模式已选路径';
            const slowBoth = document.createElement('button');
            slowBoth.id = 'qt-slow-rom-gradual-sampling';
            slowBoth.className = 'btn';
            slowBoth.textContent = '缓充缓放ROM采样（26点）';
            slowBoth.title = '升压和降压均为100mV/s；降压时保持当前通道锁定，电压降到0后才全零放气/解锁；约20Hz保存response.xlsx';
            const stop = document.createElement('button');
            stop.id = 'qt-reachability-stop';
            stop.className = 'btn';
            stop.textContent = '停止 Reachability';
            stop.disabled = true;
            row.appendChild(theory);
            row.appendChild(start);
            row.appendChild(extrema);
            row.appendChild(slowRom);
            row.appendChild(slowBoth);
            row.appendChild(stop);
            const setSamplingUi = (active, mode) => {
              start.disabled = active;
              extrema.disabled = active;
              slowRom.disabled = active;
              slowBoth.disabled = active;
              start.textContent = active && mode === 'random'
                ? 'Reachability 采样中…' : '采样 Reachability（≤200 kPa）';
              extrema.textContent = active && mode === 'extrema'
                ? '实机极值采点中…' : '实机极值采点（18点）';
              slowRom.textContent = active && mode === 'slow_rom'
                ? '慢速全ROM采样中…' : '慢速全ROM采样（26点）';
              slowBoth.textContent = active && mode === 'slow_rom_gradual'
                ? '缓充缓放ROM采样中…' : '缓充缓放ROM采样（26点）';
              stop.disabled = !active;
            };
            theory.onclick = () => {
              const raw = window.prompt('理论图随机候选姿态数量（50–3000；600约需几十秒）', '600');
              if (raw === null) return;
              const count = Math.max(50, Math.min(3000, parseInt(raw, 10) || 600));
              pyHost.calculateTheoreticalReachability(count, msg => {
                const text = String(msg);
                const ok = text.indexOf('已开始') >= 0;
                if (ok) {
                  theory.disabled = true;
                  theory.textContent = 'V6 理论图计算中…';
                }
                imuSay('<b>V6 理论 Reachability：</b> ' + text,
                  /失败/.test(text));
              });
            };
            start.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              const raw = window.prompt('随机压力点数量（6–120；另自动记录零压原点）', '36');
              if (raw === null) return;
              const count = Math.max(6, Math.min(120, parseInt(raw, 10) || 36));
              const mount = document.getElementById('imu-map');
              pyHost.startReachabilityMap(mount ? mount.value : 'ry-', count, msg => {
                const text = String(msg);
                const ok = text.indexOf('已启动') >= 0;
                if (ok) {
                  setSamplingUi(true, 'random');
                }
                imuSay('<b>Reachability：</b> ' + text, !ok);
              });
            };
            extrema.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              const accepted = window.confirm(
                '将依次执行18个真机极值工况：6个单通道、12个非拮抗双通道；' +
                '每个活动通道固定输出2200 mV，用于测量标称200 kPa ROM，工况间完全放气。' +
                '该模式不按经验X/Y/φ范围提前停止，' +
                '但保留215kPa超压保护和V1的Y−底座机械限位。请确认运动范围无人且急停可用。');
              if (!accepted) return;
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              const mount = document.getElementById('imu-map');
              pyHost.startReachabilityExtrema(mount ? mount.value : 'ry-', msg => {
                const text = String(msg);
                const ok = text.indexOf('已启动') >= 0;
                if (ok) setSamplingUi(true, 'extrema');
                imuSay('<b>实机极值采点：</b> ' + text, !ok);
              });
            };
            slowRom.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              const accepted = window.confirm(
                '将依次执行26个慢速ROM工况：6个单通道、12个非拮抗双通道、8个三轴组合。' +
                '电压以100mV/s缓慢升压，姿态速度达到3°/s暂停升压、5°/s按300mV/s退压；' +
                '每个工况之间完全放气；若X/Y/φ未回到±3°，将选择相反方向通道预锁100ms并短促充气一次，再次放气。' +
                '预计需要约15–25分钟。本模式不设软件压力或角度保护线，请确认外部保护有效。' +
                '运行数据将以约20Hz保存为response.xlsx，使用操作模式已经选择的保存路径；完成或中止都会保存已有数据。' +
                '请确认机械臂全部运动空间无人且无遮挡。是否开始？');
              if (!accepted) return;
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              const mount = document.getElementById('imu-map');
              pyHost.startSlowRomSampling(mount ? mount.value : 'ry-', msg => {
                const text = String(msg);
                const ok = text.indexOf('已启动') >= 0;
                if (ok) setSamplingUi(true, 'slow_rom');
                imuSay('<b>慢速全ROM采样：</b> ' + text, !ok);
              });
            };
            slowBoth.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              const accepted = window.confirm(
                '将执行26个缓充缓放ROM工况。每点以100mV/s缓慢升压；结束后保持活动通道锁定，' +
                '以100mV/s缓慢降到0mV，姿态速度达到3°/s时暂停继续降压，然后才全零放气/解锁。' +
                '若回零后X/Y/φ不在±3°，反向短促充气一次，并同样缓慢降到0mV。' +
                '预计约25–40分钟；不设软件压力或角度保护线，请确认外部保护有效。' +
                '数据约20Hz保存为response.xlsx。是否开始？');
              if (!accepted) return;
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              const mount = document.getElementById('imu-map');
              pyHost.startSlowRomGradualSampling(mount ? mount.value : 'ry-', msg => {
                const text = String(msg);
                const ok = text.indexOf('已启动') >= 0;
                if (ok) setSamplingUi(true, 'slow_rom_gradual');
                imuSay('<b>缓充缓放ROM采样：</b> ' + text, !ok);
              });
            };
            stop.onclick = () => {
              pyHost.stopReachabilityMap(msg => {
                setSamplingUi(false, '');
                imuSay('<b>Reachability：</b> ' + String(msg), false);
              });
            };
            if (!window.__qtReachabilityRx) {
              window.__qtReachabilityRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[reach]')) return;
                  const msg = line.slice(7).trim();
                  imuSay('<b>Reachability：</b> ' + msg,
                    /失败|中止|超时|超过/.test(msg));
                  if (/采样完成|采样已中止|保存\/绘图失败/.test(msg)) {
                    setSamplingUi(false, '');
                  }
                });
              });
            }
            if (!window.__qtTheoryReachabilityRx) {
              window.__qtTheoryReachabilityRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[theoryreach]')) return;
                  const msg = line.slice(13).trim();
                  imuSay('<b>V6 理论 Reachability：</b> ' + msg,
                    /失败/.test(msg));
                  if (/计算完成|计算失败/.test(msg)) {
                    theory.disabled = false;
                    theory.textContent = '计算 V6 理论可达图';
                  }
                });
              });
            }
          };
          // 新采集已并入联合辨识按钮；此处仅保留旧日志只读分析入口。
          const installV6GeometryCal = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.analyzeV6GeometryLogs !== 'function') {
              setTimeout(installV6GeometryCal, 200);
              return;
            }
            if (document.getElementById('qt-v6-geometry-logs')) return;
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            const logBtn = document.createElement('button');
            logBtn.id = 'qt-v6-geometry-logs';
            logBtn.className = 'btn';
            logBtn.textContent = '分析 V6 旧日志';
            logBtn.title = '只读现有真机 CSV，对比 V6 理论长度与六路编码器长度；生成候选，不修改控制器';
            row.appendChild(logBtn);

            logBtn.onclick = () => {
              logBtn.disabled = true;
              logBtn.textContent = '旧日志分析中…';
              pyHost.analyzeV6GeometryLogs(msg => {
                const text = String(msg);
                const started = text.indexOf('已启动') >= 0;
                if (!started) {
                  logBtn.disabled = false;
                  logBtn.textContent = '分析 V6 旧日志';
                }
                imuSay('<b>V6 旧日志：</b> ' + text,
                  /失败|没有找到|正在后台/.test(text) && !started);
              });
            };
            if (!window.__qtV6GeometryRx) {
              window.__qtV6GeometryRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[v6geo]')) return;
                  const msg = line.slice(7).trim();
                  imuSay('<b>V6 几何：</b> ' + msg,
                    /失败|中止|不能应用|未通过/.test(msg));
                  if (/旧日志分析完成|旧日志分析失败/.test(msg)) {
                    logBtn.disabled = false;
                    logBtn.textContent = '分析 V6 旧日志';
                  }
                });
              });
            }
          };
          // 六个物理方向的实机最大角速度测试；结果写入 control/speed_limits.json。
          const installSpeedTest = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startSpeedTest !== 'function') {
              setTimeout(installSpeedTest, 200);
              return;
            }
            if (document.getElementById('qt-speed-test-y')) return;
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            const defs = [
              ['y', '测试 Y± 最大速度'],
              ['phi', '测试 φ± 最大速度'],
              ['x', '测试 X± 最大速度']
            ];
            const buttons = [];
            defs.forEach(([key, label]) => {
              const btn = document.createElement('button');
              btn.id = 'qt-speed-test-' + key;
              btn.className = 'btn';
              btn.textContent = label;
              btn.dataset.normalText = label;
              btn.title = '只测该轴正、负方向；电压取gains.json中的u_max，每方向2轮，每次固定记录5秒；不按运动角度提前停止';
              row.appendChild(btn);
              buttons.push(btn);
              btn.onclick = () => {
                if (typeof imuOn === 'undefined' || !imuOn) {
                  imuSay('先点「连接 IMU」。', true); return;
                }
                if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                    (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
                const me = document.getElementById('imu-map');
                pyHost.startSpeedTest(me ? me.value : 'ry-', key, msg => {
                  const ok = String(msg).indexOf('已启动') >= 0;
                  if (ok) {
                    buttons.forEach(b => b.disabled = true);
                    btn.textContent = '该轴测试中…';
                  }
                  imuSay(String(msg), !ok);
                });
              };
            });
            if (!window.__qtSpeedRx) {
              window.__qtSpeedRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[speed]')) return;
                  const msg = line.slice(7).trim();
                  try { imuDemoMsg = msg; imuLastUi = 0; imuTick(); }
                  catch (_) { imuSay('<b>最大速度测试：</b> ' + msg, /失败|中止|超时/.test(msg)); }
                  if (/轴测试完成|失败|中止|超时/.test(msg)) {
                    buttons.forEach(b => {
                      b.disabled = false;
                      b.textContent = b.dataset.normalText;
                    });
                  }
                });
              });
            }
          };
          // X 主肌与 φ 抵消肌的实机比例扫描：9+67、10+23 分开辨识。
          const installXCompTest = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startXCompTest !== 'function') {
              setTimeout(installXCompTest, 200);
              return;
            }
            if (document.getElementById('qt-xcomp-9-67')) return;
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            const defs = [
              ['9_67', '标定 9+67 补偿'],
              ['10_23', '标定 10+23 补偿']
            ];
            const buttons = [];
            defs.forEach(([key, label]) => {
              const btn = document.createElement('button');
              btn.id = 'qt-xcomp-' + key.replace('_', '-');
              btn.className = 'btn';
              btn.textContent = label;
              btn.dataset.normalText = label;
              btn.title = '主通道固定800mV，自动测试补偿比例0/0.25/0.5/0.75/1；每档2秒并完全放气后换档';
              row.appendChild(btn);
              buttons.push(btn);
              btn.onclick = () => {
                if (typeof imuOn === 'undefined' || !imuOn) {
                  imuSay('先点「连接 IMU」。', true); return;
                }
                if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                    (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
                const me = document.getElementById('imu-map');
                pyHost.startXCompTest(me ? me.value : 'ry-', key, msg => {
                  const ok = String(msg).indexOf('已启动') >= 0;
                  if (ok) {
                    buttons.forEach(b => b.disabled = true);
                    btn.textContent = '补偿标定中…';
                  }
                  imuSay(String(msg), !ok);
                });
              };
            });
            if (!window.__qtXCompRx) {
              window.__qtXCompRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[xcomp]')) return;
                  const msg = line.slice(7).trim();
                  try { imuDemoMsg = msg; imuLastUi = 0; imuTick(); }
                  catch (_) { imuSay('<b>X耦合补偿：</b> ' + msg,
                    /失败|中止|超时/.test(msg)); }
                  if (/测试完成|失败|中止|超时/.test(msg)) {
                    buttons.forEach(b => {
                      b.disabled = false;
                      b.textContent = b.dataset.normalText;
                    });
                  }
                });
              });
            }
          };
          const installPureMpcData = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startPureMpcDataCollection !== 'function') {
              setTimeout(installPureMpcData, 200);
              return;
            }
            if (document.getElementById('qt-pure-mpc-data')) return;
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            const btn = document.createElement('button');
            btn.id = 'qt-pure-mpc-data';
            btn.className = 'btn';
            btn.textContent = '采集纯数据 MPC 模型';
            btn.title = '每轮5种工况：单通道2 + 双通道2 + 三通道1；每种5档，工况间完全放气';
            row.appendChild(btn);
            btn.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              const me = document.getElementById('imu-map');
              pyHost.startPureMpcDataCollection(me ? me.value : 'ry-', msg => {
                const ok = String(msg).indexOf('已启动') >= 0;
                if (ok) {
                  btn.disabled = true;
                  btn.textContent = '纯数据采样中…';
                }
                imuSay(String(msg), !ok);
              });
            };
            if (!window.__qtPureMpcDataRx) {
              window.__qtPureMpcDataRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[mpcdata]')) return;
                  const msg = line.slice(9).trim();
                  try { imuDemoMsg = msg; imuLastUi = 0; imuTick(); }
                  catch (_) { imuSay('<b>纯数据MPC采样：</b> ' + msg,
                    /失败|中止|超时/.test(msg)); }
                  if (/采样完成|失败|中止|超时/.test(msg)) {
                    btn.disabled = false;
                    btn.textContent = '采集纯数据 MPC 模型';
                  }
                });
              });
            }
          };
          // 对比真正模型与真机的 u2/u3/u4/u5→φ 多拍输入梯度，并复核X补偿组合。
          const installPhiGradientTest = () => {
            const identBtn = document.getElementById('imu-demo');
            if (!identBtn || typeof pyHost === 'undefined' || !pyHost ||
                typeof pyHost.startPhiGradientTest !== 'function') {
              setTimeout(installPhiGradientTest, 200);
              return;
            }
            if (document.getElementById('qt-phi-gradient-test')) return;
            const row = identBtn.closest('.btns') || identBtn.parentElement;
            const btn = document.createElement('button');
            btn.id = 'qt-phi-gradient-test';
            btn.className = 'btn';
            btn.textContent = '验证 φ 模型梯度';
            btn.title = '自动测试23/67/9/10单路的600/1200/1800mV阶跃，并按标定比例复核9+67、10+23；比较真机与真正模型的5拍、10拍φ响应，不修改模型';
            row.appendChild(btn);
            btn.onclick = () => {
              if (typeof imuOn === 'undefined' || !imuOn) {
                imuSay('先点「连接 IMU」。', true); return;
              }
              if ((typeof tcPlaying !== 'undefined' && tcPlaying) ||
                  (typeof tcMpcTimer !== 'undefined' && tcMpcTimer)) tcStop();
              const mount = document.getElementById('imu-map');
              const mode = document.getElementById('tc-drvmode');
              const source = document.getElementById('qt-model-source');
              pyHost.startPhiGradientTest(
                mount ? mount.value : 'ry-',
                mode ? mode.value : 'mpc',
                source ? source.value : 'latest', msg => {
                  const ok = String(msg).indexOf('已启动') >= 0;
                  if (ok) {
                    btn.disabled = true;
                    btn.textContent = 'φ梯度验证中…';
                  }
                  imuSay(String(msg), !ok);
                });
            };
            if (!window.__qtPhiGradientRx) {
              window.__qtPhiGradientRx = true;
              pyHost.rx.connect(txt => {
                String(txt).split(/\r?\n/).forEach(line => {
                  if (!line.startsWith('[phigrad]')) return;
                  const msg = line.slice(9).trim();
                  try { imuDemoMsg = msg; imuLastUi = 0; imuTick(); }
                  catch (_) { imuSay('<b>φ模型梯度：</b> ' + msg,
                    /失败|中止|超时|✗/.test(msg)); }
                  if (/验证完成|失败|中止|超时/.test(msg)) {
                    btn.disabled = false;
                    btn.textContent = '验证 φ 模型梯度';
                  }
                });
              });
            }
          };
          installTeachDeadband();
          installModelSource();
          installFixedTask();
          installMpcWeights();
          installPid();
          installPressurePid();
          installMusclePid();
          installStage();
          installIdent();
          installAdcCal();
          installManualPressureTest();
          installReachabilityMap();
          installV6GeometryCal();
          installSpeedTest();
          installXCompTest();
          installPureMpcData();
          installPhiGradientTest();
        })();
        """)

    dash = DashWindow()                              # debug 模式保留原来的独立「控制响应」窗口
    dash.resize(820, 900)
    dash.move(60, 60)

    def _set_debug_dash(visible):
        if visible:
            dash.show()
            dash.raise_()
        else:
            dash.hide()

    bridge.debug_mode_changed.connect(_set_debug_dash)

    view.loadFinished.connect(_install_qt_overrides)
    view.load(QUrl.fromLocalFile(HTML))
    view.resize(1500, 950)
    # 关主窗 = 退出整个应用（否则曲线窗还开着时主窗只是隐藏，页面在后台继续控真机）
    view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    view.destroyed.connect(app.quit)
    view.show()
    # 操作模式使用嵌入式“数据可视化”；debug 模式由上面的信号恢复旧独立窗口。

    def _shutdown():
        # 顺序要紧：先停 MPC（触发 _finish → FEL 权重/CSV 落盘），再放气关串口
        gui_recorder.shutdown()
        try:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:8765/stop", data=b"{}", method="POST"), timeout=1.0)
        except Exception:
            pass
        # Nano25 与主控/IMU 串口解耦；只在整个应用退出时强制释放 NI-DAQ。
        bridge._stop_nano25(wait=True, force=True)
        if bridge._force_recording is not None:
            bridge.stopForceRecording()
        bridge.shutdownForcePrediction()
        bridge.close()
    app.aboutToQuit.connect(_shutdown)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

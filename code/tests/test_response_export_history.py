"""Exercise long-run export without importing or starting the hardware GUI."""
import ast
from collections import deque
import json
from pathlib import Path
import threading
import unittest


CONTROL = Path(__file__).resolve().parents[1] / "control"


def load_definition(filename, class_name, methods=None):
    tree = ast.parse((CONTROL / filename).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == class_name)
    if methods is not None:
        node.bases = []
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                     and n.name in methods]
        for n in node.body:
            n.decorator_list = []
    scope = {"deque": deque, "threading": threading, "json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 filename, "exec"), scope)
    return scope[class_name]


class ResponseExportTest(unittest.TestCase):
    def test_long_runs_survive_plot_window_eviction(self):
        buffer_type = load_definition("mpc_station.py", "ResponseTelemetry")
        for samples in (36000, 72000):
            with self.subTest(samples=samples):
                history = buffer_type()
                for i in range(samples):
                    history.append({"t": i / 20})
                self.assertEqual(len(history), 6000)
                exported = history.export_snapshot()
                self.assertEqual(len(exported), samples)
                self.assertEqual(exported[0]["t"], 0)
                self.assertEqual(exported[-1]["t"], (samples - 1) / 20)
                history.clear()
                self.assertEqual(history.export_snapshot(), [])
                self.assertEqual(len(exported), samples)

    def make_bridge(self, recording):
        bridge_type = load_definition("qt_gui.py", "SerialBridge",
                                      {"saveDataSnapshot", "stopDataRecording"})
        bridge = bridge_type()
        bridge._data_recording_lock = threading.Lock()
        bridge._data_recording = recording
        bridge._last_data_recording = None
        bridge._control_response_workbook_path = lambda: "response.xlsx"
        bridge._control_response_rows = lambda: [[i] for i in range(6000)]
        bridge.saved = None
        def queue(record):
            bridge.saved = record
            return {"ok": True, "rows": len(record["rows"])}
        bridge._queue_data_snapshot = queue
        return bridge

    def test_save_and_stop_keep_all_recorded_rows(self):
        record = {"rows": [[i] for i in range(72000)], "path": "response.xlsx"}
        bridge = self.make_bridge(record)
        self.assertEqual(json.loads(bridge.saveDataSnapshot())["rows"], 72000)
        self.assertIs(bridge._data_recording, record)
        self.assertIs(bridge.saved, record)
        self.assertEqual(json.loads(bridge.stopDataRecording())["rows"], 72000)
        self.assertIsNone(bridge._data_recording)
        self.assertIs(bridge.saved, record)

    def test_unarmed_save_uses_full_history(self):
        bridge = self.make_bridge(None)
        bridge._control_response_rows = lambda: [[i] for i in range(72000)]
        self.assertEqual(json.loads(bridge.saveDataSnapshot())["rows"], 72000)


if __name__ == "__main__":
    unittest.main()

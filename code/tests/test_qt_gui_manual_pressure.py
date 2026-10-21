import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook


CONTROL = Path(__file__).resolve().parents[1] / "control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from qt_gui import (MANUAL_PRESSURE_DEFLATE_SLEW_MV_PER_S,
                    OPEN_LOOP_MAX_OUTPUT_MV, REACHABILITY_MAX_KPA,
                    SerialBridge)
import qt_gui as qt_gui_module
from codex_compensator import (CodexCompensator, _build_design_prompt,
                               _append_llm_audit_events,
                               _llm_recording_paths,
                               PLUS_PROMPT_TEMPLATE_PATH,
                               PRO_PROMPT_TEMPLATE_PATH,
                               _probe_compensator_source,
                               _validate_compensator_source,
                               _write_llm_final_result,
                               normalize_prompt_profile,
                               normalize_reasoning_effort)


class _FakeSerial:
    def __init__(self):
        self.writes = []

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)


def _frame(pressure=None, locks="oooooo"):
    return {
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "kpa": np.zeros(6) if pressure is None else np.asarray(pressure, dtype=float),
        "locks": locks,
        "channel_length_mm": [235.0] * 6,
        "wall": time.time(),
    }


def test_manual_pressure_has_free_targets_but_keeps_hardware_output_ceiling():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()

    result = bridge.setManualPressureTargets(json.dumps([10000, 0, 80, 0, 50, 0]))
    assert "已启动" in result

    frame = _frame()
    frame["wall"] = bridge._manual_pressure_test["last_t"] + 0.06
    bridge._manual_pressure_on_frame(frame)
    command = bridge.ser.writes[-1].decode("ascii").strip()
    assert command.startswith("C")
    outputs = [int(value) for value in command[1:].split(",")]
    assert outputs[0] == OPEN_LOOP_MAX_OUTPUT_MV
    assert all(0 <= value <= OPEN_LOOP_MAX_OUTPUT_MV for value in outputs)
    assert bridge.write("W100,0,0,0,0,0\n") is False

    bridge._manual_pressure_test["rows"] = []
    write_count = len(bridge.ser.writes)
    stopped = bridge.stopManualPressureTest()
    assert "缓慢放气" in stopped
    assert bridge._manual_pressure_test["gradual_stop"] is True
    assert len(bridge.ser.writes) == write_count

    previous = bridge._manual_pressure_test["last_u"].copy()
    frame = _frame([50, 0, 0, 0, 0, 0], locks="Looooo")
    frame["wall"] = bridge._manual_pressure_test["last_t"] + 0.05
    bridge._manual_pressure_on_frame(frame)
    command = bridge.ser.writes[-1].decode("ascii").strip()
    released = np.asarray([int(value) for value in command[1:].split(",")])
    assert command.startswith("C")
    assert released[0] == pytest.approx(
        previous[0] - MANUAL_PRESSURE_DEFLATE_SLEW_MV_PER_S * 0.05,
        abs=1.0)
    assert bridge.ser.writes[-1] != b"s\n"

    bridge._manual_pressure_test["rows"] = []
    bridge._manual_pressure_stop("测试清理")
    assert bridge.ser.writes[-1] == b"s\n"


def test_manual_pressure_gradual_stop_waits_for_pressure_and_locks_before_final_vent():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    bridge.setManualPressureTargets(json.dumps([20, 0, 0, 0, 0, 0]))
    bridge.stopManualPressureTest()
    bridge._manual_pressure_test["last_u"] = np.array([4.0, 0, 0, 0, 0, 0])

    waiting = _frame([35, 0, 0, 0, 0, 0], locks="Looooo")
    waiting["wall"] = bridge._manual_pressure_test["last_t"] + 0.05
    bridge._manual_pressure_on_frame(waiting)
    assert bridge._manual_pressure_test is not None
    assert bridge.ser.writes[-1] == b"C0,0,0,0,0,0\n"

    ready = _frame([30, 0, 0, 0, 0, 0], locks="oooooo")
    ready["wall"] = bridge._manual_pressure_test["last_t"] + 0.05
    bridge._manual_pressure_test["rows"] = []
    bridge._manual_pressure_on_frame(ready)
    assert bridge._manual_pressure_test is None
    assert bridge.ser.writes[-1] == b"s\n"


def test_manual_pressure_allows_antagonist_pair_and_requires_unlock_safe_start():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    result = bridge.setManualPressureTargets(json.dumps([0, 0, 100, 100, 0, 0]))
    assert "已启动" in result
    frame = _frame()
    frame["wall"] = bridge._manual_pressure_test["last_t"] + 0.06
    bridge._manual_pressure_on_frame(frame)
    command = bridge.ser.writes[-1].decode("ascii").strip()
    assert command.startswith("C")
    outputs = [int(value) for value in command[1:].split(",")]
    assert outputs[2] > 0
    assert outputs[3] > 0
    bridge._manual_pressure_test["rows"] = []
    bridge.stopManualPressureTest()

    unsafe_bridge = SerialBridge()
    unsafe_bridge.ser = _FakeSerial()
    unsafe_bridge._latest_frame = _frame([31, 0, 0, 0, 0, 0])
    unsafe = unsafe_bridge.setManualPressureTargets(
        json.dumps([100, 0, 0, 0, 0, 0]))
    assert "≤30 kPa" in unsafe
    assert unsafe_bridge._manual_pressure_test is None


def test_manual_pressure_start_has_no_confirmation_popup():
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")
    start = source.index("apply.onclick = () =>", source.index("qt-manp-apply"))
    stop = source.index("stop.onclick = () =>", start)
    handler = source[start:stop]

    assert "setManualPressureTargets" in handler
    assert "window.confirm" not in handler


def test_reachability_targets_respect_pressure_and_antagonist_constraints():
    targets = SerialBridge._reachability_targets(36, seed=20260910)

    assert targets.shape == (36, 6)
    assert np.all(targets >= 0.0)
    assert np.all(targets <= REACHABILITY_MAX_KPA)
    for left, right in ((0, 1), (2, 3), (4, 5)):
        assert not np.any((targets[:, left] > 0.0) & (targets[:, right] > 0.0))
    for channel in range(6):
        assert np.any(np.all(targets == np.eye(6)[channel] * REACHABILITY_MAX_KPA,
                             axis=1))


def test_reachability_extrema_are_exactly_six_singles_and_twelve_legal_pairs():
    targets = SerialBridge._reachability_extrema_targets()

    assert targets.shape == (18, 6)
    assert len(np.unique(targets, axis=0)) == 18
    active_counts = np.sum(targets > 0.0, axis=1)
    assert np.count_nonzero(active_counts == 1) == 6
    assert np.count_nonzero(active_counts == 2) == 12
    assert np.all(targets[targets > 0.0] == REACHABILITY_MAX_KPA)
    for left, right in ((0, 1), (2, 3), (4, 5)):
        assert not np.any((targets[:, left] > 0.0) & (targets[:, right] > 0.0))


def test_slow_rom_has_26_antagonist_safe_boundary_cases():
    targets = SerialBridge._slow_rom_targets()

    assert targets.shape == (26, 6)
    assert len(np.unique(targets, axis=0)) == 26
    active_counts = np.sum(targets > 0.0, axis=1)
    assert np.count_nonzero(active_counts == 1) == 6
    assert np.count_nonzero(active_counts == 2) == 12
    assert np.count_nonzero(active_counts == 3) == 8
    for left, right in ((0, 1), (2, 3), (4, 5)):
        assert not np.any((targets[:, left] > 0.0) & (targets[:, right] > 0.0))


def test_slow_rom_recenter_uses_only_the_opposite_direction_channels():
    locks, output = SerialBridge._slow_rom_recenter_command([10.0, -20.0, 5.0])

    assert locks.tolist() == [True, False, False, True, True, False]
    assert output.tolist() == pytest.approx([350.0, 0.0, 0.0, 275.0, 300.0, 0.0])
    for left, right in ((0, 1), (2, 3), (4, 5)):
        assert not (locks[left] and locks[right])

    within_zero, within_output = SerialBridge._slow_rom_recenter_command(
        [3.0, -3.0, 0.0])
    assert not np.any(within_zero)
    assert not np.any(within_output)


def test_slow_rom_voltage_step_holds_and_backs_off_on_pose_speed():
    target = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    start = np.zeros(6)
    rising = SerialBridge._slow_rom_output_step(
        start, target, np.zeros(3), 0.05, 2200.0)
    held = SerialBridge._slow_rom_output_step(
        rising, target, np.array([0.0, 4.0, 0.0]), 0.05, 2200.0)
    backed_off = SerialBridge._slow_rom_output_step(
        np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]), target,
        np.array([0.0, 6.0, 0.0]), 0.05, 2200.0)

    assert rising.tolist() == pytest.approx([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert held.tolist() == pytest.approx(rising.tolist())
    assert backed_off.tolist() == pytest.approx([85.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_gradual_rom_release_is_100mv_per_second_and_pauses_on_motion():
    previous = np.array([100.0, 0.0, 80.0, 0.0, 0.0, 0.0])

    released = SerialBridge._slow_rom_release_step(
        previous, np.zeros(3), 0.05)
    held = SerialBridge._slow_rom_release_step(
        previous, np.array([0.0, 3.1, 0.0]), 0.05)

    assert released.tolist() == pytest.approx([95.0, 0.0, 75.0, 0.0, 0.0, 0.0])
    assert held.tolist() == pytest.approx(previous.tolist())


def test_slow_rom_requires_the_operation_mode_recording_directory():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    bridge._recording_directory_confirmed = False

    result = bridge.startSlowRomSampling("ry-")

    assert "操作模式" in result
    assert "路径" in result
    assert bridge._reachability_test is None


def test_reachability_endpoint_uses_hanging_zero_and_216mm_arm():
    origin = SerialBridge._reachability_endpoint_mm([0.0, 0.0, 70.0])
    x_positive = SerialBridge._reachability_endpoint_mm([20.0, 0.0, 0.0])
    y_positive = SerialBridge._reachability_endpoint_mm([0.0, 80.0, 0.0])

    assert np.allclose(origin, [0.0, 0.0, -216.0])
    assert np.allclose(np.linalg.norm(x_positive), 216.0)
    assert np.allclose(np.linalg.norm(y_positive), 216.0)
    assert x_positive[1] > 0.0
    assert y_positive[0] < 0.0


def test_extrema_does_not_stop_at_plot_envelope_but_keeps_v1_base_limit():
    free_case = {
        "sampling_mode": "extrema",
        "target": np.array([200.0, 0.0, 200.0, 0.0, 0.0, 0.0]),
    }
    v1_case = {
        "sampling_mode": "extrema",
        "target": np.array([0.0, 200.0, 0.0, 0.0, 0.0, 0.0]),
    }
    random_case = {"sampling_mode": "random", "target": np.zeros(6)}

    assert SerialBridge._reachability_pose_stop_reason(
        free_case, [31.0, 92.0, 88.0], q1_deg=95.0) is None
    assert "底座机械限位" in SerialBridge._reachability_pose_stop_reason(
        v1_case, [0.0, -38.1, 0.0], q1_deg=38.1)
    assert "经验姿态边界" in SerialBridge._reachability_pose_stop_reason(
        random_case, [21.0, 0.0, 0.0], q1_deg=21.0)


def test_reachability_start_is_gui_owned_and_stop_vents():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()

    result = bridge.startReachabilityMap("ry-", 12)

    assert "已启动" in result
    assert bridge._reachability_test is not None
    assert bridge.ser.writes[-1] == b"W0,0,0,0,0,0\n"
    assert bridge.write("W100,0,0,0,0,0\n") is False
    stopped = bridge.stopReachabilityMap()
    assert "已中止" in stopped
    assert bridge._reachability_test is None
    assert bridge.ser.writes[-1] == b"s\n"


def test_extrema_reachability_start_has_18_cases_and_preserves_safety_constraints():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()

    result = bridge.startReachabilityExtrema("ry-")

    assert "已启动" in result
    test = bridge._reachability_test
    assert test["sampling_mode"] == "extrema"
    assert test["targets"].shape == (18, 6)
    assert bridge.ser.writes[-1] == b"W0,0,0,0,0,0\n"
    bridge.stopReachabilityMap()


def test_extrema_run_commands_fixed_2200mv_instead_of_pressure_pi():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    assert "已启动" in bridge.startReachabilityExtrema("ry-")
    test = bridge._reachability_test
    test["stage"] = "run"
    test["target"] = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    test["requested_locks"] = test["target"] > 0.0
    test["qref"] = np.array([1.0, 0.0, 0.0, 0.0])
    frame = _frame(locks="Looooo")
    test["stage_t"] = frame["wall"] - 0.2
    test["last_t"] = frame["wall"] - 0.05

    bridge._reachability_on_frame(frame)

    command = bridge.ser.writes[-1].decode("ascii").strip()
    assert command == "M1,0,0,0,0,0;2200,0,0,0,0,0"
    bridge.stopReachabilityMap()


def test_slow_rom_start_and_first_run_tick_use_five_mv_step(tmp_path):
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    bridge._recording_base_dir = str(tmp_path)
    bridge._recording_directory_confirmed = True
    assert "已启动" in bridge.startSlowRomSampling("ry-")
    test = bridge._reachability_test
    assert test["sampling_mode"] == "slow_rom"
    assert test["targets"].shape == (26, 6)
    assert Path(test["response_path"]).parent.parent == tmp_path
    assert Path(test["response_path"]).name == "response.xlsx"
    test["stage"] = "run"
    test["target"] = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    test["requested_locks"] = test["target"] > 0.0
    test["qref"] = np.array([1.0, 0.0, 0.0, 0.0])
    frame = _frame(locks="Looooo")
    frame["lens"] = [0.0] * 20
    test["stage_t"] = frame["wall"] - 0.2
    test["last_t"] = frame["wall"] - 0.05

    bridge._reachability_on_frame(frame)

    assert bridge.ser.writes[-1].decode("ascii").strip() == \
        "M1,0,0,0,0,0;5,0,0,0,0,0"
    assert test["rows"][-1]["d_active_A1+5_mm"] == pytest.approx(0.0)
    assert test["rows"][-1]["d_passive_A1+5_mm"] == pytest.approx(0.0)
    too_soon = dict(frame)
    too_soon["wall"] = frame["wall"] + 0.02
    bridge._reachability_on_frame(too_soon)
    next_tick = dict(frame)
    next_tick["wall"] = frame["wall"] + 0.05
    bridge._reachability_on_frame(next_tick)
    assert len(test["response_rows"]) == 2
    response_path = Path(test["response_path"])
    assert "正在保存" in bridge.stopReachabilityMap()
    deadline = time.time() + 3.0
    while not response_path.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert response_path.exists()
    workbook = load_workbook(response_path, read_only=True, data_only=True)
    sheet = workbook["控制响应"]
    saved_rows = list(sheet.iter_rows(values_only=True))
    assert len(saved_rows) == 3
    assert saved_rows[0][0] == "response_time_s"
    assert saved_rows[1][0] == pytest.approx(0.0)
    assert saved_rows[2][0] == pytest.approx(0.05)
    workbook.close()


def test_gradual_rom_point_end_releases_to_zero_before_full_vent(tmp_path):
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    bridge._recording_base_dir = str(tmp_path)
    bridge._recording_directory_confirmed = True
    assert "已启动" in bridge.startSlowRomGradualSampling("ry-")
    test = bridge._reachability_test
    assert test["sampling_mode"] == "slow_rom_gradual"
    assert "slow_rom_gradual" in Path(test["response_path"]).parent.name
    test["stage"] = "run"
    test["target"] = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    test["requested_locks"] = test["target"] > 0.0
    test["last_u"] = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    test["qref"] = np.array([1.0, 0.0, 0.0, 0.0])
    test["pose_samples"] = [(0.0, np.zeros(3), np.zeros(6))]

    writes_before = len(bridge.ser.writes)
    bridge._reachability_end_point(test, "test", valid=True)
    assert test["stage"] == "slow_release"
    assert len(bridge.ser.writes) == writes_before

    release = _frame(locks="Looooo")
    release["lens"] = [0.0] * 20
    release["wall"] = test["last_t"] + 0.05
    bridge._reachability_on_frame(release)

    assert test["stage"] == "vent"
    assert bridge.ser.writes[-2] == b"M1,0,0,0,0,0;0,0,0,0,0,0\n"
    assert bridge.ser.writes[-1] == b"W0,0,0,0,0,0\n"
    response_path = Path(test["response_path"])
    bridge.stopReachabilityMap()
    deadline = time.time() + 3.0
    while not response_path.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert response_path.exists()


def test_slow_rom_has_no_software_pressure_or_pose_protection_line(tmp_path):
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    bridge._recording_base_dir = str(tmp_path)
    bridge._recording_directory_confirmed = True
    assert "已启动" in bridge.startSlowRomSampling("ry-")
    test = bridge._reachability_test
    test["stage"] = "run"
    test["target"] = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    test["requested_locks"] = test["target"] > 0.0
    test["qref"] = np.array([1.0, 0.0, 0.0, 0.0])
    frame = _frame(locks="Looooo")
    frame["lens"] = [0.0] * 20
    test["stage_t"] = frame["wall"] - 0.2
    test["last_t"] = frame["wall"] - 0.05
    bridge._reachability_on_frame(frame)
    assert len(test["response_rows"]) == 1

    response_path = Path(test["response_path"])
    outside_old_line = _frame(
        [216.0, 0.0, 0.0, 0.0, 0.0, 0.0], locks="Looooo")
    outside_old_line["lens"] = [0.0] * 20
    outside_old_line["wall"] = frame["wall"] + 0.05
    bridge._reachability_on_frame(outside_old_line)

    assert bridge._reachability_test is test
    assert len(test["response_rows"]) == 2
    assert SerialBridge._reachability_pose_stop_reason(
        test, [50.0, 120.0, 100.0], q1_deg=120.0) is None
    bridge.stopReachabilityMap()
    assert bridge.ser.writes[-1] == b"s\n"
    deadline = time.time() + 3.0
    while not response_path.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert response_path.exists()
    workbook = load_workbook(response_path, read_only=True, data_only=True)
    saved_rows = list(workbook["控制响应"].iter_rows(values_only=True))
    assert len(saved_rows) == 3
    workbook.close()


def test_slow_rom_vents_then_prelocks_and_pulses_once_for_recenter(tmp_path):
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    bridge._recording_base_dir = str(tmp_path)
    bridge._recording_directory_confirmed = True
    assert "已启动" in bridge.startSlowRomSampling("ry-")
    test = bridge._reachability_test
    test["qref"] = np.array([1.0, 0.0, 0.0, 0.0])
    test["case_i"] = 1
    frame = _frame(locks="oooooo")
    frame["lens"] = [0.0] * 20
    test["stage"] = "vent"
    test["stage_t"] = frame["wall"] - 2.0
    test["ready_t"] = frame["wall"] - 1.1
    test["last_send"] = frame["wall"] - 0.3
    bridge._reachability_pose_from_frame = lambda _test, _frame: (
        np.zeros(3), np.array([10.0, -20.0, 5.0]))

    bridge._reachability_on_frame(frame)

    assert test["stage"] == "recenter_prelock"
    assert test["requested_locks"].tolist() == [True, False, False, True, True, False]
    assert bridge.ser.writes[-1] == b"M1,0,0,1,1,0;0,0,0,0,0,0\n"

    locked = _frame(locks="LooLLo")
    locked["lens"] = [0.0] * 20
    locked["wall"] = test["stage_t"] + 0.11
    bridge._reachability_on_frame(locked)
    assert test["stage"] == "recenter_pulse"
    assert bridge.ser.writes[-1] == b"M1,0,0,1,1,0;350,0,0,275,300,0\n"

    pulse_done = _frame(locks="LooLLo")
    pulse_done["lens"] = [0.0] * 20
    pulse_done["wall"] = test["stage_t"] + 0.26
    bridge._reachability_on_frame(pulse_done)
    assert test["stage"] == "vent"
    assert test["recenter_done_for_case"] is True
    assert bridge.ser.writes[-1] == b"W0,0,0,0,0,0\n"

    response_path = Path(test["response_path"])
    bridge.stopReachabilityMap()
    deadline = time.time() + 3.0
    while not response_path.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert response_path.exists()


def test_reachability_plot_can_be_rendered(tmp_path):
    results = []
    for index, pose in enumerate(([0, 0, 0], [10, 5, 3], [-8, 12, -4])):
        results.append({
            "valid": True,
            "pose_deg": pose,
            "measured_kpa": [20.0 + 10.0 * index, 0, 0, 0, 0, 0],
        })
    target = tmp_path / "reachability.png"

    SerialBridge._save_reachability_plot(results, target)

    assert target.exists()
    assert target.stat().st_size > 1000


def test_reachability_uses_safe_vent_then_explicit_100ms_prelock():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    assert "已启动" in bridge.startReachabilityMap("ry-", 6)
    test = bridge._reachability_test
    test["targets"][0] = np.array([80.0, 0.0, 60.0, 0.0, 40.0, 0.0])

    vented = _frame()
    vented["wall"] = test["stage_t"] + 0.05
    bridge._reachability_on_frame(vented)
    vented["wall"] = test["stage_t"] + 1.1
    bridge._reachability_on_frame(vented)

    assert test["stage"] == "prelock"
    command = bridge.ser.writes[-1].decode("ascii").strip()
    assert command == "M1,0,1,0,1,0;0,0,0,0,0,0"

    locked = _frame(locks="LoLoLo")
    locked["wall"] = test["stage_t"] + 0.11
    bridge._reachability_on_frame(locked)
    assert test["stage"] == "run"


def test_v3_repeat_test_only_schedules_v3_and_stays_candidate_free():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()

    result = bridge.startV3RepeatTest("ry-", 3)

    assert "已启动" in result
    assert bridge._dm_ident["kind"] == "v3_repeat"
    assert bridge._dm_ident["channel_plan"] == [3, 3, 3]
    assert bridge._dm_ident["ch"] == 3
    assert bridge._dm_ident["levels"] == [600, 1200, 1800, 1200, 600]
    assert bridge.ser.writes[-1] == b"s\n"
    assert bridge.write("W0,0,0,100,0,0\n") is False

    bridge._ident_abort("test")
    assert bridge._dm_ident is None
    assert bridge.ser.writes[-1] == b"s\n"


def test_quick_health_uses_one_short_six_channel_scan_and_stop_vents():
    bridge = SerialBridge()
    bridge.ser = _FakeSerial()
    bridge._latest_frame = _frame()
    status_calls = []
    bridge._quick_health_status = lambda state, items=None, message="", report_path="": (
        status_calls.append((state, items or {}, message, report_path)))

    result = bridge.startQuickHealthCheck("ry-")

    assert "已启动" in result
    ident = bridge._dm_ident
    assert ident["kind"] == "quick_health"
    assert ident["channel_plan"] == list(range(6))
    assert ident["hold_s"] == 1.0
    assert len(ident["levels"]) == 5
    assert max(ident["levels"]) <= OPEN_LOOP_MAX_OUTPUT_MV
    assert set(status_calls[-1][1]) == {
        "environment", "pressure", "muscle", "mapping", "imu", "speed"}
    assert status_calls[-1][1]["environment"]["state"] == "pass"
    assert "model" not in status_calls[-1][1]
    assert bridge.ser.writes[-1] == b"s\n"

    stopped = bridge.stopQuickHealthCheck()
    assert "已停止" in stopped
    assert bridge._dm_ident is None
    assert bridge.ser.writes[-1] == b"s\n"


def test_python_environment_check_reports_runtime_and_project_files():
    bridge = SerialBridge()

    result = json.loads(bridge.checkPythonEnvironment())

    assert result["state"] == "pass"
    names = {item["name"] for item in result["checks"]}
    assert {"Python", "NumPy", "SciPy", "PyQt5", "Qt WebEngine",
            "pyserial", "openpyxl", "项目运行文件"} <= names
    assert all(item["ok"] for item in result["checks"])


def test_environment_auto_install_only_uses_allowlisted_missing_packages(monkeypatch):
    bridge = SerialBridge()
    report = {
        "state": "fail", "detail": "缺少 openpyxl",
        "checks": [
            {"name": "openpyxl", "ok": False},
            {"name": "项目运行文件", "ok": False},
        ],
    }
    started = []

    class _FakeThread:
        def __init__(self, target, args=(), daemon=None):
            self.target, self.args, self.daemon = target, args, daemon

        def start(self):
            started.append(self.args)

    monkeypatch.setattr(bridge, "_python_environment_report", lambda: report)
    monkeypatch.setattr(bridge, "_environment_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(qt_gui_module.threading, "Thread", _FakeThread)

    result = json.loads(bridge.installMissingPythonPackages())

    assert result["ok"] is True
    assert result["started"] is True
    assert result["packages"] == ["openpyxl"]
    assert started == [(["openpyxl"],)]


def test_gui_and_camera_video_sinks_are_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(qt_gui_module, "RECORDINGS_DIR", str(tmp_path))
    bridge = SerialBridge()
    bridge._recording_base_dir = str(tmp_path)

    gui = json.loads(bridge.startVideoRecording("gui", "video/webm;codecs=vp9"))
    camera = json.loads(bridge.startVideoRecording("camera", "video/webm"))
    assert gui["ok"] and camera["ok"]
    assert gui["path"] != camera["path"]

    gui_bytes = b"gui-webm-chunk"
    camera_bytes = b"camera-webm-chunk"
    assert bridge.appendVideoChunk(
        "gui", gui["token"], base64.b64encode(gui_bytes).decode("ascii"))
    assert bridge.appendVideoChunk(
        "camera", camera["token"], base64.b64encode(camera_bytes).decode("ascii"))
    gui_done = json.loads(bridge.finishVideoRecording("gui", gui["token"]))
    camera_done = json.loads(bridge.finishVideoRecording("camera", camera["token"]))

    assert Path(gui_done["path"]).read_bytes() == gui_bytes
    assert Path(camera_done["path"]).read_bytes() == camera_bytes


def test_recording_session_places_all_outputs_in_one_timestamp_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(qt_gui_module, "RECORDINGS_DIR", str(tmp_path))
    bridge = SerialBridge()
    bridge._recording_base_dir = str(tmp_path)

    rejected = json.loads(bridge.startRecordingSession())
    assert rejected["ok"] is False
    assert "选择保存文件夹" in rejected["error"]

    bridge._recording_directory_confirmed = True

    session = json.loads(bridge.startRecordingSession())
    gui_path = Path(bridge._recording_path("gui", "mp4"))
    response_path = Path(bridge._recording_path("response", "xlsx"))
    camera_path = Path(bridge._recording_path("camera", "webm"))

    assert session["ok"] is True
    assert (gui_path.parent == response_path.parent == camera_path.parent ==
            Path(session["folder"]))
    assert {gui_path.name, response_path.name, camera_path.name} == {
        "gui.mp4", "response.xlsx", "camera.webm"}
    assert os.environ[qt_gui_module.LLM_RECORDING_SESSION_ENV] == str(
        Path(session["folder"]).resolve())
    assert bridge.finishRecordingSession(session["token"]) is True
    assert qt_gui_module.LLM_RECORDING_SESSION_ENV not in os.environ


def test_llm_audit_files_are_saved_in_active_recording_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("TRO_RECORDING_SESSION_DIR", str(tmp_path))
    paths = _llm_recording_paths("round:01")

    _append_llm_audit_events(
        paths["reasoning"], "initial_design",
        stdout='{"type":"item.completed","item":{"type":"reasoning"}}\n',
        payload={"type": "provider_exit", "returncode": 0})
    _write_llm_final_result(
        paths["final"], {"request_id": "round:01", "rationale": "done"})

    reasoning_lines = Path(paths["reasoning"]).read_text(
        encoding="utf-8").splitlines()
    assert len(reasoning_lines) == 2
    assert json.loads(reasoning_lines[1])["event"]["type"] == "item.completed"
    assert json.loads(Path(paths["final"]).read_text(
        encoding="utf-8"))["rationale"] == "done"
    assert Path(paths["final"]).parent == tmp_path


def test_operation_gui_recording_omits_debug_frames_and_tracks_mode_state():
    bridge = SerialBridge()
    states = []
    bridge.debug_mode_changed.connect(states.append)

    assert bridge._debug_mode is False
    bridge.setDebugMode(True)
    assert bridge._debug_mode is True
    assert states == [True]

    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")
    assert 'if bool(getattr(self.bridge, "_debug_mode", False)):' in source
    assert '"请先切回操作模式再开始GUI录制"' in source

    html = Path(qt_gui_module.HTML).read_text(encoding="utf-8")
    assert "const codexBetweenRounds=!!window.__qtCodexCycleWaiting" in html
    assert "!qtDataRecording||!pyHost||codexBetweenRounds" in html


def test_operation_recording_replaces_view_readout_with_compact_task_checkboxes():
    html = Path(qt_gui_module.HTML).read_text(encoding="utf-8")

    assert 'id="cam-readout"' not in html
    assert 'id="task-record-options"' in html
    for element_id in ("qt-rec-gui", "qt-rec-response", "qt-rec-camera", "qt-rec-force"):
        assert f'type="checkbox" id="{element_id}"' in html
    assert 'type="checkbox" id="qt-rec-data"' not in html
    assert 'id="qt-record-dir"' in html
    assert 'id="qt-record-save"' in html
    assert "chooseRecordingDirectory" in html
    assert "recordingDirectoryState" in html
    assert "window.__qtEnsureRecordingDirectory" in html
    assert "window.__qtStartSelectedRecordings" in html
    assert "window.__qtStopSelectedRecordings" in html
    assert "function qtControlTaskIsRunning()" in html
    assert "saveDataSnapshot" in html
    assert "startForceRecording" in html
    assert "stopForceRecording" in html
    assert "saveForceSnapshot" in html
    assert "await window.__qtStopSelectedRecordings();" not in html[
        html.index("if(qtRecordSaveButton)qtRecordSaveButton.onclick"):]
    assert "nothingToSave" in html
    assert "qtRecordStatus(saveMessage,!saved.length);" in html
    assert "opUiText('nothingToSave','当前没有可保存的数据'),true" not in html
    assert "['qt-rec-response','qt-rec-force'].forEach(id=>" in html
    assert "input.checked&&!qtDataRecording" in html
    assert "input.checked&&!qtForceRecording" in html
    assert "response&&response.checked&&!qtDataRecording" in html
    assert "force&&force.checked&&!qtForceRecording" in html
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")
    assert "if (recordingReady === false) return;" in source
    assert "window.__qtOperationExecutionIsRunning = () =>" in source
    assert "recordingStartGuardToken" in source
    assert "const startDeadline = Date.now() + 15000" in source
    assert "setTimeout(waitForExecutionStart, 250)" in source
    assert "}, 750);" not in source
    assert "typeof tcMpcStarting !== 'undefined' && tcMpcStarting" in source
    assert "tcMpcStarting=false, tcMpcStartToken=0" in html
    assert "if(tcPlaying||tcMpcStarting){ tcStop(); return; }" in html


def test_manual_save_falls_back_to_last_completed_response(monkeypatch, tmp_path):
    bridge = SerialBridge()
    previous = {
        "rows": [[0.0, 1.0, 2.0, 3.0]],
        "path": str(tmp_path / "response.xlsx"),
    }
    bridge._last_data_recording = previous
    monkeypatch.setattr(bridge, "_control_response_rows", lambda: [])
    queued = []
    monkeypatch.setattr(
        bridge, "_queue_data_snapshot",
        lambda record: queued.append(record) or {
            "ok": True, "saving": True, "rows": len(record["rows"]),
            "path": record["path"],
        })

    result = json.loads(bridge.saveDataSnapshot())

    assert result["ok"] is True
    assert result["rows"] == 1
    assert queued == [previous]


def test_default_taught_trajectory_uses_five_second_phi_cycle():
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    task = json.loads((control_dir / "important_task_trajectory.json").read_text(
        encoding="utf-8"))
    points = task["points"]

    assert [point["t"] for point in points] == [2.0, 2.5, 1.0, 2.5]
    assert points[1]["t"] + points[3]["t"] == 5.0
    assert points[2]["t"] == 1.0
    assert points[2]["q"] == points[1]["q"]
    assert points[1]["q"][2] == 30.0
    first_phi_delta = abs(points[1]["q"][2] - points[0]["q"][2])
    second_phi_delta = abs(points[3]["q"][2] - points[2]["q"][2])
    assert 1.875 * first_phi_delta / points[1]["t"] < 22.6
    assert 1.875 * second_phi_delta / points[3]["t"] < 22.1


def test_default_2_changes_only_y_from_plus_30_to_minus_20():
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    task = json.loads((control_dir / "important_task_trajectory_2.json").read_text(
        encoding="utf-8"))

    assert task["name"] == "Default 2"
    assert task["auto_hidden_return_to_zero"] is True
    assert task["hidden_return_duration_s"] == 2.0
    assert [point["q"] for point in task["points"]] == [
        [0.0, 30.0, 0.0], [0.0, -20.0, 0.0]]
    assert [point["q_raw"] for point in task["points"]] == [
        [0.0, 30.0, 0.0], [0.0, -20.0, 0.0]]
    assert [point["q_control"] for point in task["points"]] == [
        [0.0, 30.0, 0.0], [0.0, -20.0, 0.0]]


def test_default_3_returns_x_then_y_to_zero_before_venting():
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    task = json.loads((control_dir / "important_task_trajectory_3.json").read_text(
        encoding="utf-8"))

    assert task["name"] == "Default 3"
    assert task["auto_hidden_return_to_zero"] is False
    assert "hidden_return_duration_s" not in task
    expected = [
        [0.0, 40.0, 0.0],
        [-10.0, 40.0, 0.0],
        [0.0, 40.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert [point["q"] for point in task["points"]] == expected
    assert [point["q_raw"] for point in task["points"]] == expected
    assert [point["q_control"] for point in task["points"]] == expected
    assert [point["t"] for point in task["points"]] == [2.0, 2.0, 2.0, 2.0]


def test_operation_mode_has_bilingual_selector_version_and_muscle_parameters():
    html = Path(qt_gui_module.HTML).read_text(encoding="utf-8")
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")
    station_source = (Path(qt_gui_module.__file__).with_name("mpc_station.py")
                      .read_text(encoding="utf-8"))

    assert html.index('id="op-language"') < html.index('id="mode-user"')
    assert '<option value="zh" selected>中文</option>' in html
    assert '<option value="en">English</option>' in html
    assert '<div id="app-version">Version 0.1</div>' in html
    assert "window.__qtApplyOperationLanguage" in source
    assert "(typeof DEBUG !== 'undefined' && DEBUG) ? 'zh'" in source
    assert "poseDrive:'姿态联动'" in source
    assert "poseDrive:'Pose Sync'" in source
    assert "jointPose:'Joint Pose'" in source
    assert "pythonEnvironment:'Software Environment'" in source
    assert "loadDefault:'载入'" in source
    assert "loadDefault:'Load'" in source
    assert 'id="qt-op-load-default"' in source
    assert 'id="qt-op-default-select"' in source
    assert '<option value="1" selected>Default 1</option>' in source
    assert '<option value="2">Default 2</option>' in source
    assert '<option value="3">Default 3</option>' in source
    assert "def loadFixedTask2" in source
    assert "def loadFixedTask3" in source
    assert "clearMemory:'清除记忆'" in source
    assert "clearMemory:'Clear Memory'" in source
    assert 'id="qt-op-clear-memory"' in source
    assert "window.__qtResetCodexCompensator" in source
    assert "reset_codex_compensator" in station_source
    assert "window.__qtLoadFixedTask = loadTask" in source
    assert "window.__qtLoadFixedTask(false" in source
    assert "tro-software-environment-v0.1-passed" in source
    for mode in (1, 2, 3):
        assert f'id="qt-op-m{mode}-limit"' in source
        assert f'id="qt-op-m{mode}-unit"' in source
        assert f'id="qt-op-m{mode}-limit" min="1" max="999" step="1" value="1"' in source
    assert '<option value="runs">圈</option><option value="min">mins</option>' in source
    assert 'id="qt-op-teach-limit" min="1" max="999" step="1" value="1"' in source
    assert 'id="qt-op-teach-unit"' in source
    assert "teachLoop.closest('.qt-op-loop')" in source
    assert 'id="qt-op-llm-config-btn"' in source
    assert 'id="qt-op-llm-modal"' in source
    assert 'id="qt-op-llm-provider"' in source
    assert '<option value="gpt">GPT</option><option value="claude">Claude</option>' in source
    assert '<select id="qt-op-llm-model"></select>' in source
    assert '<option value="pro">Pro（自由设计）</option>' in source
    assert "designPro:'Pro (Open Design)'" in source
    assert 'prompt_profile not in ("base", "plus", "pro")' in station_source
    assert 'id="qt-op-llm-reuse-mse"' in source
    assert 'id="qt-op-llm-fast"' in source
    assert "reuseMse:'沿用阈值（MSE deg²）'" in source
    assert "reuseMse:'Reuse threshold (MSE deg²)'" in source
    assert "fastMode:'Fast Mode（更多额度，更快）'" in source
    assert "fastMode:'Fast Mode (more credits, faster)'" in source
    assert "reuse_mse_threshold_deg2:reuseMse" in source
    assert "fast_mode:!!llmFast.checked" in source
    assert 'id="qt-op-llm-auth-method"' in source
    assert '<option value="account">ChatGPT账号</option><option value="api">API Key</option>' in source
    assert 'type="password" id="qt-op-llm-api-key"' in source
    assert 'id="qt-op-llm-auth-status"' in source
    assert "loginCodexChatGPT" in source
    assert "loginCodexApiKey" in source
    assert "checkCodexAuth" in source
    assert "checkClaudeAuth" in source
    assert "loginClaudeApiKey" in source
    assert "loginClaudeAccount" in source
    assert "installClaudeCode" in source
    assert 'id="qt-op-llm-install-claude"' in source
    assert "https://claude.ai/install.ps1" in source
    assert "logoutLlm" in source
    assert "codex_auth_status" in source
    assert 'id="qt-op-llm-signout"' in source
    assert "Sign in to Codex with a ChatGPT account or API key." not in source
    gains_text = Path(qt_gui_module.GAINS_PATH).read_text(encoding="utf-8").lower()
    assert "api_key" not in gains_text
    assert "value !== 'mpc_codex'" in source
    assert "http://127.0.0.1:8765/llm_config" in source
    assert "llmSettings:'LLM Settings'" in source
    assert '<option value="minimal">' not in source
    assert "['low','medium','high','xhigh'].includes(savedEffort)" in source
    assert json.loads(Path(qt_gui_module.GAINS_PATH).read_text(
        encoding="utf-8"))["codex_reasoning_effort"] in {
            "low", "medium", "high", "xhigh"}
    assert json.loads(Path(qt_gui_module.GAINS_PATH).read_text(
        encoding="utf-8"))["codex_fast_mode"] is True
    assert "beginTeachExecutionPlan" in source
    assert "loop_limit_runs" in source
    assert "loop_limit_min" in source
    assert 'BR["loop_limit_runs"]' in station_source
    assert 'BR["loop_deadline_monotonic"]' in station_source
    assert "completed_laps" in station_source
    assert '"codex_reuse_mse_threshold_deg2": 15.0' in station_source
    assert '"state": "reused"' in station_source
    assert 'stable_mse <= reuse_threshold' in station_source
    assert '"state": "retry"' in station_source
    assert '"retry_same_generation": True' in station_source
    assert "autoStatus.state === 'retry'" in source
    assert "s.state === 'retry'" in source
    assert "countThisRun && !invalidRetry" in source
    assert 'self.path == "/llm_config"' in station_source
    assert '"llm_provider": "gpt"' in station_source
    assert '"llm_auth_method": "account"' in station_source
    assert 'provider=design_provider' in station_source
    assert 'auth_method=design_auth_method' in station_source
    assert 'fast_mode=design_fast_mode' in station_source
    assert '"codex_reasoning_effort": "medium"' in station_source
    assert '"codex_fast_mode": False' in station_source
    compensator_source = (Path(qt_gui_module.__file__).with_name(
        "codex_compensator.py").read_text(encoding="utf-8"))
    assert "fast_mode: bool = False" in compensator_source
    assert "'service_tier=\"fast\"'" in compensator_source
    assert '"features.fast_mode=true"' in compensator_source
    assert "optionalLoopLimit" in source
    assert "activeRunLimit" in source
    assert "activeDeadlineMs" in source
    assert "window.__qtLocalizeOperationRuntime" in source
    assert "title.textContent = inDebug ? '真实肌肉长度（传感器）' : qtOpText('muscleParameters')" in source
    assert "当前肌肉长度与气压" not in source


def test_codex_reasoning_effort_migrates_legacy_minimal_and_rejects_unsafe_values():
    assert normalize_reasoning_effort("minimal") == "low"
    for effort in ("low", "medium", "high", "xhigh"):
        assert normalize_reasoning_effort(effort) == effort
    with pytest.raises(ValueError):
        normalize_reasoning_effort("none")


def test_codex_compensator_uses_a_signed_residual(tmp_path):
    reference = np.zeros((2, 3), dtype=float)
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(tmp_path / "request.json"), reference, reference,
        dt=0.05, u_max=2200.0, max_slew_mv_s=0.0)
    compensator.active["bias_mv"] = [-100.0, 0.0, 100.0, 0.0, 0.0, 0.0]

    total, applied = compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([1000.0, 0.0, 500.0, 0.0, 0.0, 0.0]),
        np.array([250.0, 0.0, 100.0, 0.0, 0.0, 0.0]),
        "LLLLLL", 0.0)

    assert total[[0, 2]].tolist() == [900.0, 600.0]
    assert applied[[0, 2]].tolist() == [-100.0, 100.0]

    compensator.active["bias_mv"] = [0.0] * 6
    kp = np.zeros((6, 3), dtype=float)
    kp[0, 0] = 30.0
    compensator.active["kp_mv_per_deg"] = kp.tolist()
    total, applied = compensator.apply(
        np.array([-20.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "LLLLLL", 0.05)
    assert total[0] == 400.0
    assert applied[0] == -600.0


def test_codex_llm_mpc_final_voltage_falls_slowly_and_holds_old_lock(tmp_path):
    reference = np.zeros((4, 3), dtype=float)
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(tmp_path / "request.json"), reference, reference,
        dt=0.05, u_max=2500.0, final_fall_slew_mv_s=1200.0)

    first, _ = compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([2500.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6),
        "Looooo", 0.0, [1, 0, 0, 0, 0, 0])
    switched, _ = compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6),
        "Looooo", 0.05, [0, 1, 0, 0, 0, 0])

    assert first[0] == pytest.approx(2500.0)
    assert switched.tolist() == pytest.approx([2440.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert compensator.last_guarded_lock_cmd[:2].tolist() == [1, 0]
    assert compensator.records[-1]["total_u_mv"][:2] == pytest.approx([2440.0, 0.0])


def test_codex_llm_mpc_phi_final_voltage_rises_at_most_30mv_per_tick(tmp_path):
    reference = np.zeros((3, 3), dtype=float)
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(tmp_path / "request.json"), reference, reference,
        dt=0.05, u_max=2500.0, final_fall_slew_mv_s=1200.0,
        final_phi_rise_slew_mv_s=600.0)

    first, _ = compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([1000.0, 0.0, 1000.0, 0.0, 0.0, 0.0]), np.zeros(6),
        "LoLooo", 0.0, [1, 0, 1, 0, 0, 0])
    second, _ = compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([1000.0, 0.0, 1000.0, 0.0, 0.0, 0.0]), np.zeros(6),
        "LoLooo", 0.05, [1, 0, 1, 0, 0, 0])

    # Y remains unrestricted on the rise; both phi channels use the final
    # MPC+LLM 600 mV/s guard, i.e. 30 mV per 50 ms control tick.
    assert first[[0, 2]].tolist() == pytest.approx([1000.0, 30.0])
    assert second[[0, 2]].tolist() == pytest.approx([1000.0, 60.0])


def test_codex_llm_mpc_latches_first_prelock_request(tmp_path):
    reference = np.zeros((3, 3), dtype=float)
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(tmp_path / "request.json"), reference, reference,
        dt=0.05, u_max=2500.0, final_fall_slew_mv_s=1200.0)

    compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 600.0, 0.0, 0.0, 0.0]), np.zeros(6),
        "oooooo", 0.0, [0, 0, 1, 0, 0, 0])
    total, _ = compensator.apply(
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 600.0, 0.0, 0.0]), np.zeros(6),
        "oooooo", 0.05, [0, 0, 0, 1, 0, 0])

    assert compensator.last_guarded_lock_cmd[2:4].tolist() == [1, 0]
    assert total[2:4].tolist() == pytest.approx([0.0, 0.0])


def test_codex_compensator_persists_compact_design_experience(tmp_path):
    reference = np.zeros((3, 3), dtype=float)
    state_path = tmp_path / "state.json"
    candidate_path = tmp_path / "candidate.json"
    request_path = tmp_path / "request.json"
    compensator = CodexCompensator(
        str(state_path), str(candidate_path), str(request_path),
        reference, reference, dt=0.05, u_max=2200.0,
        max_slew_mv_s=0.0)

    def record_lap(measured_value):
        for index in range(len(reference)):
            compensator.apply(
                np.zeros(3), np.full(3, measured_value),
                np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6),
                "LLLLLL", index*0.05)

    # Establish the measured stable baseline: 3 axes * 1 deg^2 = 3 deg^2.
    record_lap(1.0)
    baseline = compensator.finish_round(999.0)
    assert baseline["tracking_mse_deg2"] == pytest.approx(3.0)
    assert baseline["postmortem"]["outcome"] == "stable_baseline"

    parameters = json.loads(json.dumps(compensator.active))
    parameters["bias_mv"][0] = 100.0
    candidate_path.write_text(json.dumps({
        "schema": compensator.SCHEMA,
        "task_hash": compensator.task_hash,
        "base_generation": 0,
        "created_by": "test",
        "rationale": "V0 needs a small signed offset during the lap.",
        "design_strategy": "Raise only the locked V0 bias by 100 mV.",
        "expected_effect": "Reduce X-axis lag without changing other phases.",
        "failure_analysis": "Baseline only; no earlier candidate has failed.",
        "parameters": parameters,
    }), encoding="utf-8")

    started = compensator.begin_round()
    assert started["candidate"]["loaded"] is True
    record_lap(2.0)
    failed = compensator.finish_round(999.0)

    assert failed["rolled_back"] is True
    memory = failed["postmortem"]
    assert memory["outcome"] == "rolled_back"
    assert memory["hypothesis"].startswith("V0 needs")
    assert memory["design_strategy"].startswith("Raise only")
    assert memory["expected_effect"].startswith("Reduce X-axis")
    assert memory["measured_mse_deg2"] == pytest.approx(12.0)
    assert memory["comparison_mse_deg2"] == pytest.approx(3.0)
    assert memory["mse_delta_deg2"] == pytest.approx(9.0)
    assert memory["parameter_changes"][0]["parameter"] == "bias_mv[V0]"
    assert memory["audit_invalid_reasons"] == []

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["design_memory_last_5"][-1]["outcome"] == "rolled_back"
    assert len(request["design_memory_last_5"]) <= 5
    restored = CodexCompensator(
        str(state_path), str(candidate_path), str(request_path),
        reference, reference, dt=0.05, u_max=2200.0,
        max_slew_mv_s=0.0)
    assert restored.status()["design_memory_count"] == 2
    assert restored.status()["last_trial"]["outcome"] == "rolled_back"


def test_codex_candidate_schema_requires_strategy_effect_and_failure_memory():
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    schema = json.loads((control_dir / "codex_compensator_candidate_schema.json").read_text(
        encoding="utf-8"))
    required = set(schema["required"])
    assert {"rationale", "design_strategy", "expected_effect",
            "failure_analysis"} <= required
    assert "source_code" in required
    assert "parameters" not in schema["properties"]
    source = (control_dir / "codex_compensator.py").read_text(encoding="utf-8")
    prompt_data = json.loads((control_dir / "codex_compensator_prompt.json").read_text(
        encoding="utf-8"))
    prompt_text = "\n".join(str(value) for value in prompt_data.values())
    assert "Read design_memory_last_5" in prompt_text
    assert "rolled-back law without a different hypothesis" in prompt_text
    assert "OBJECTIVE, in order" in prompt_text
    assert "suggested_additional_positive_command_mv" in source
    assert "PID-ONLY COMPENSATOR" in prompt_text
    assert "different PID matrices and gains in deterministic time" in prompt_text
    assert "segment transitions must remain smooth" in prompt_text
    assert "In every segment calculate raw_output" in prompt_text
    assert "Optional deterministic segment-specific Kp, Kd, and Ki matrices" in prompt_text
    assert "Select the two axes with the largest reference peak-to-peak motion" in prompt_text
    assert "never select more than two" in prompt_text
    assert "reference peak-to-peak motion exceeds 1.0 degree" in prompt_text
    assert "Use 4.0 deg RMSE as the per-axis threshold" in prompt_text
    assert "If only one exceeds 4.0 deg RMSE, compensate only that axis" in prompt_text
    assert "If both exceed 4.0 deg RMSE, both may be compensated" in prompt_text
    assert "newest valid canary result overrides older claims" in prompt_text
    assert "never select or switch channels from reference sign" in prompt_text
    assert "DIRECTION-TO-CHANNEL RULE" not in prompt_text
    assert "direction_mask" not in prompt_text
    assert "output_limit_mv" in prompt_text
    assert "output_slew_mv_per_s" in prompt_text
    assert "kp_matrix (6,3)" in prompt_text
    assert "kd_matrix (6,3)" in prompt_text
    assert "ki_matrix (6,3)" in prompt_text
    assert "cross-axis entries zero unless repeated measured evidence" in prompt_text
    assert "applied_compensation = total_u - u_mpc" in prompt_text
    assert "electrical headroom [-u_mpc, u_max-u_mpc]" in prompt_text
    assert "further negative compensation cannot vent faster" in prompt_text
    assert "headroom_target = np.clip" in prompt_text
    assert "phase/time scheduling" not in prompt_text
    assert "kp_mv_per_deg" not in prompt_text
    assert '"candidate_template":' not in source
    assert "COMPLETED-LAP JSON REPORT" in prompt_text
    assert "stable_compensator_after_audit" in prompt_text
    assert '"--output-schema", schema_path, "-o", output_path, "-"' in source
    assert "input=prompt" in source
    assert "DESIGN STAGE: FIRST ROUND" in prompt_text
    assert "DESIGN STAGE: SECOND OR LATER ROUND" in prompt_text
    assert "DESIGN STAGE: FIRST ROUND" not in source
    assert "OUTPUT CONTRACT" not in source
    assert "COMPLETED-LAP JSON REPORT" not in source
    assert "First write failure_analysis separately for each selected axis" in prompt_text


def test_codex_uses_distinct_first_and_later_round_prompts():
    first = _build_design_prompt({
        "tracking_mse_deg2": 80.0,
        "last_design": {},
        "design_memory_last_5": [{"outcome": "stable_baseline"}],
    })
    assert "DESIGN STAGE: FIRST ROUND" in first
    assert "Select at most two axes from reference motion" in first
    assert "enable only those above 4.0 deg" in first

    poor_later = _build_design_prompt({
        "tracking_mse_deg2": 48.2,
        "last_design": {"generation": 1, "design_strategy": "earlier boost"},
        "design_memory_last_5": [{"outcome": "accepted", "measured_mse_deg2": 48.2}],
    })
    assert "DESIGN STAGE: SECOND OR LATER ROUND" in poor_later
    assert "48.200000 deg^2" in poor_later
    assert "Apply the 4.0 deg RMSE threshold separately" in poor_later
    assert ">30" not in poor_later
    assert "First write failure_analysis separately for each selected axis" in poor_later

    refining_later = _build_design_prompt({
        "tracking_mse_deg2": 22.0,
        "last_design": {"generation": 2},
        "design_memory_last_5": [{"outcome": "accepted", "measured_mse_deg2": 22.0}],
    })
    assert "22.000000 deg^2" in refining_later
    assert "Preserve useful behavior and apply the 4.0 deg RMSE threshold separately" in refining_later
    assert "<=30" not in refining_later


def test_codex_appends_operator_feedback_verbatim_to_prompt():
    feedback = "先解决Y过冲；φ暂时不要增加正向电压。"
    rendered = _build_design_prompt(
        {"tracking_mse_deg2": 20.0}, "{}", operator_feedback=feedback)

    assert rendered.endswith(feedback)


def test_codex_prompt_json_allows_literal_braces(tmp_path):
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    prompt_data = json.loads((control_dir / "codex_compensator_prompt.json").read_text(
        encoding="utf-8"))
    prompt_data["main_prompt"].append("Literal example: {'gain': 5}.")
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(json.dumps(prompt_data), encoding="utf-8")
    rendered = _build_design_prompt({}, "{}", prompt_path=str(prompt_path))
    assert "Literal example: {'gain': 5}." in rendered


def test_codex_plus_prompt_is_distinct_and_pressure_aware():
    rendered = _build_design_prompt(
        {"tracking_mse_deg2": 80.0}, "{}",
        prompt_path=PLUS_PROMPT_TEMPLATE_PATH)

    assert normalize_prompt_profile("base") == "base"
    assert normalize_prompt_profile("plus") == "plus"
    assert normalize_prompt_profile("pro") == "pro"
    with pytest.raises(ValueError):
        normalize_prompt_profile("unknown")
    assert "FIRST ADVANCED ROUND" in rendered
    assert "PRESSURE INNER LAYER" in rendered
    assert "Y+ -> V0, Y- -> V1" in rendered
    assert "pressure input must materially participate" in rendered
    assert "must not assume that PD" in rendered
    assert "Do not call self.reset()" in rendered


def test_codex_pro_prompt_is_open_but_keeps_physical_rules():
    rendered = _build_design_prompt(
        {"tracking_mse_deg2": 80.0}, "{}",
        prompt_path=PRO_PROMPT_TEMPLATE_PATH)

    assert "PRO STAGE: FIRST ROUND" in rendered
    assert "control at most two axes" in rendered
    assert "must never be driven positively at the same time" in rendered
    assert "PID, another feedback law, feedforward" in rendered
    assert "all optional rather than mandatory" in rendered
    assert "PRESSURE INNER LAYER" not in rendered
    assert "PID-ONLY COMPENSATOR" not in rendered


def test_plus_and_pro_keep_resumable_validator_repair_path():
    source = Path(qt_gui_module.__file__).with_name(
        "codex_compensator.py").read_text(encoding="utf-8")

    assert 'prompt_profile in ("plus", "pro")' in source
    assert '"validator_repair"' in source
    assert "repair_index < 2" in source


def test_codex_plus_prompt_template_passes_runtime_validator():
    prompt_data = json.loads(Path(PLUS_PROMPT_TEMPLATE_PATH).read_text(
        encoding="utf-8"))
    contract = next(item for item in prompt_data["main_prompt"]
                    if item.startswith("VALIDATED IMPLEMENTATION TEMPLATE"))
    source = "class Compensator:\n" + contract.split(
        "\n\nclass Compensator:\n", 1)[1]

    clean = _probe_compensator_source(source)
    assert "pressure_target" in clean
    assert "np.asarray(pressure)" in clean


def test_codex_prompt_class_template_passes_runtime_validator():
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    prompt_data = json.loads((control_dir / "codex_compensator_prompt.json").read_text(
        encoding="utf-8"))
    contract = next(item for item in prompt_data["main_prompt"]
                    if item.startswith("SOURCE-CODE CONTRACT"))
    source = "class Compensator:\n" + contract.split(
        "\n\nclass Compensator:\n", 1)[1]
    assert _probe_compensator_source(source).startswith("class Compensator:")


def test_codex_validator_strips_redundant_numpy_import():
    control_dir = Path(qt_gui_module.__file__).resolve().parent
    prompt_data = json.loads((control_dir / "codex_compensator_prompt.json").read_text(
        encoding="utf-8"))
    contract = next(item for item in prompt_data["main_prompt"]
                    if item.startswith("SOURCE-CODE CONTRACT"))
    source = "import numpy as np\n\nclass Compensator:\n" + contract.split(
        "\n\nclass Compensator:\n", 1)[1]
    clean = _probe_compensator_source(source)
    assert clean.startswith("class Compensator:")
    assert "import numpy" not in clean


def test_codex_validator_allows_local_array_item_assignment():
    source = """class Compensator:
    def __init__(self):
        self.kp_matrix = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.kd_matrix = np.zeros((6, 3))
        self.ki_matrix = np.zeros((6, 3))
        self.integral_limit_deg_s = np.zeros(3)
        self.output_limit_mv = np.ones(6) * 600.0
        self.output_slew_mv_per_s = np.ones(6) * 2000.0
        self.previous_output = np.zeros(6)
        self.integral_error = np.zeros(3)
        self.last_time = 0.0
    def reset(self, feedback):
        self.previous_output = np.zeros(6)
        self.integral_error = np.zeros(3)
        self.last_time = 0.0
    def compute(self, u_mpc, pose, velocity, reference_pose, reference_velocity, pressure, lock_state, trajectory_time):
        feature = np.asarray(reference_pose) - np.asarray(pose)
        velocity_error = np.asarray(reference_velocity) - np.asarray(velocity)
        self.integral_error = np.clip(self.integral_error, -self.integral_limit_deg_s, self.integral_limit_deg_s)
        raw = np.matmul(self.kp_matrix, feature) + np.matmul(self.kd_matrix, velocity_error) + np.matmul(self.ki_matrix, self.integral_error)
        output = np.zeros(6)
        output[0] = raw[0]
        lock_mask = np.asarray(lock_state) == "L"
        self.previous_output = self.previous_output * lock_mask
        target = np.clip(output, -self.output_limit_mv, self.output_limit_mv) * lock_mask
        dt = float(trajectory_time) - self.last_time
        if dt < 0.0:
            dt = 0.0
        step = np.clip(target - self.previous_output, -self.output_slew_mv_per_s * dt, self.output_slew_mv_per_s * dt)
        output = self.previous_output + step
        self.previous_output = output
        self.last_time = float(trajectory_time)
        return output
"""
    assert _probe_compensator_source(source).startswith("class Compensator:")


def test_codex_validator_accepts_angle_error_derivative_pid_without_gain_matrices():
    source = """class Compensator:
    def __init__(self):
        self.p_gain = np.array([0.0, 10.0, 0.0])
        self.d_gain = np.array([0.0, 8.0, 0.0])
        self.previous_error = np.zeros(3)
        self.output_limit_mv = np.array([400.0, 400.0, 0.0, 0.0, 0.0, 0.0])
        self.output_slew_mv_per_s = np.array([500.0, 500.0, 0.0, 0.0, 0.0, 0.0])
        self.u_max_mv = 2500.0
        self.previous_output = np.zeros(6)
        self.last_time = 0.0
    def reset(self, feedback):
        self.previous_error = np.zeros(3)
        self.previous_output = np.zeros(6)
        self.last_time = 0.0
    def compute(self, u_mpc, pose, velocity, reference_pose, reference_velocity, pressure, lock_state, trajectory_time):
        error = np.asarray(reference_pose) - np.asarray(pose)
        current_time = float(trajectory_time)
        dt = min(0.05, max(0.0, current_time - self.last_time))
        scheduled_p_gain = self.p_gain
        if current_time >= 2.0:
            scheduled_p_gain = 0.5 * self.p_gain
        derivative_error = np.zeros(3)
        if dt > 0.0:
            derivative_error = (error - self.previous_error) / dt
        self.previous_error = error
        self.last_time = current_time
        axis_output = scheduled_p_gain * error + self.d_gain * derivative_error
        raw_output = np.array([axis_output[1], -axis_output[1], 0.0, 0.0, 0.0, 0.0])
        base_u = np.clip(np.asarray(u_mpc), 0.0, self.u_max_mv)
        target = np.clip(raw_output, -self.output_limit_mv, self.output_limit_mv)
        target = np.clip(target, -base_u, self.u_max_mv - base_u)
        lock_mask = np.asarray(lock_state) == "L"
        self.previous_output = self.previous_output * lock_mask
        target = target * lock_mask
        maximum_step = self.output_slew_mv_per_s * dt
        output = self.previous_output + np.clip(target - self.previous_output, -maximum_step, maximum_step)
        self.previous_output = output
        return output
"""
    clean = _probe_compensator_source(source)
    _, instance = _validate_compensator_source(
        clean, require_design_contract=True)
    assert not hasattr(instance, "kp_matrix")
    assert instance.d_gain[1] == pytest.approx(8.0)


def test_codex_validator_requires_matrix_limits_and_slew_contract():
    source = """class Compensator:
    def __init__(self):
        self.kp_matrix = np.ones((6, 3))
        self.kd_matrix = np.zeros((6, 3))
        self.ki_matrix = np.zeros((6, 3))
        self.integral_limit_deg_s = np.zeros(3)
        self.output_limit_mv = np.ones(6) * 600.0
        self.output_slew_mv_per_s = np.ones(6) * 2000.0
        self.previous_output = np.zeros(6)
        self.integral_error = np.zeros(3)
        self.last_time = 0.0
    def reset(self, feedback):
        self.previous_output = np.zeros(6)
        self.integral_error = np.zeros(3)
        self.last_time = 0.0
    def compute(self, u_mpc, pose, velocity, reference_pose, reference_velocity, pressure, lock_state, trajectory_time):
        feature = np.asarray(reference_pose) - np.asarray(pose)
        velocity_error = np.asarray(reference_velocity) - np.asarray(velocity)
        self.integral_error = np.clip(self.integral_error, -self.integral_limit_deg_s, self.integral_limit_deg_s)
        lock_mask = np.asarray(lock_state) == "L"
        self.previous_output = self.previous_output * lock_mask
        raw = np.matmul(self.kp_matrix, feature) + np.matmul(self.kd_matrix, velocity_error) + np.matmul(self.ki_matrix, self.integral_error)
        target = np.clip(raw, -self.output_limit_mv, self.output_limit_mv) * lock_mask
        dt = float(trajectory_time) - self.last_time
        if dt < 0.0:
            dt = 0.0
        step = np.clip(target - self.previous_output, -self.output_slew_mv_per_s * dt, self.output_slew_mv_per_s * dt)
        output = self.previous_output + step
        self.previous_output = output
        self.last_time = float(trajectory_time)
        return output
"""
    assert _probe_compensator_source(source).startswith("class Compensator:")

    disabled_channels = source.replace(
        "self.output_limit_mv = np.ones(6) * 600.0",
        "self.output_limit_mv = np.array([600.0, 0.0, 400.0, 0.0, 0.0, 0.0])",
    ).replace(
        "self.output_slew_mv_per_s = np.ones(6) * 2000.0",
        "self.output_slew_mv_per_s = np.array([2000.0, 0.0, 1500.0, 0.0, 0.0, 0.0])",
    )
    assert _probe_compensator_source(disabled_channels).startswith("class Compensator:")

    all_limits_disabled = disabled_channels.replace(
        "np.array([600.0, 0.0, 400.0, 0.0, 0.0, 0.0])", "np.zeros(6)")
    assert _probe_compensator_source(all_limits_disabled).startswith("class Compensator:")

    active_channel_without_slew = disabled_channels.replace(
        "np.array([2000.0, 0.0, 1500.0, 0.0, 0.0, 0.0])", "np.zeros(6)")
    with pytest.raises(ValueError, match="wherever output_limit_mv is positive"):
        _probe_compensator_source(active_channel_without_slew)


def test_codex_complete_offline_multiround_flow(tmp_path):
    reference = np.zeros((4, 3), dtype=float)
    state_path = tmp_path / "state.json"
    candidate_path = tmp_path / "candidate.json"
    request_path = tmp_path / "request.json"
    compensator = CodexCompensator(
        str(state_path), str(candidate_path), str(request_path),
        reference, reference, dt=0.05, u_max=2200.0)

    def record_lap(measured_deg):
        for index in range(len(reference)):
            compensator.apply(
                np.zeros(3), np.full(3, measured_deg),
                np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6),
                "LLLLLL", index*0.05)

    def write_bias_candidate(base_generation, bias_mv, label):
        parameters = json.loads(json.dumps(compensator.active))
        parameters["bias_mv"][0] = float(bias_mv)
        candidate_path.write_text(json.dumps({
            "schema": compensator.SCHEMA,
            "task_hash": compensator.task_hash,
            "base_generation": base_generation,
            "created_by": "offline-flow-test",
            "rationale": f"{label} rationale",
            "design_strategy": f"{label} strategy: set V0 bias to {bias_mv} mV",
            "expected_effect": f"{label} expected lower lag and MSE",
            "failure_analysis": f"{label} analysis of the prior result",
            "parameters": parameters,
        }), encoding="utf-8")

    # Baseline: 3 axes * 4^2 = 48 deg^2. This must select the first-round prompt.
    record_lap(4.0)
    baseline = compensator.finish_round(48.0)
    first_report = json.loads(request_path.read_text(encoding="utf-8"))
    assert baseline["tracking_mse_deg2"] == pytest.approx(48.0)
    assert "DESIGN STAGE: FIRST ROUND" in _build_design_prompt(first_report)

    # Candidate 1 improves the complete lap; total MSE does not select an
    # aggressiveness tier or determine the next change magnitude.
    write_bias_candidate(0, 100.0, "round-one")
    started_one = compensator.begin_round()
    assert started_one["candidate"]["loaded"] is True
    record_lap(3.5)
    accepted_one = compensator.finish_round(36.75)
    poor_report = json.loads(request_path.read_text(encoding="utf-8"))
    poor_prompt = _build_design_prompt(poor_report)
    assert accepted_one["accepted"] is True
    assert accepted_one["tracking_mse_deg2"] == pytest.approx(36.75)
    assert poor_report["last_design"]["design_strategy"].startswith("round-one")
    assert poor_report["design_memory_last_5"][-1]["outcome"] == "accepted"
    assert "36.750000 deg^2" in poor_prompt
    assert "Apply the 4.0 deg RMSE threshold separately" in poor_prompt
    assert ">30" not in poor_prompt

    # Candidate 2 is treated by the same per-axis evidence rule.
    write_bias_candidate(1, 200.0, "round-two")
    started_two = compensator.begin_round()
    assert started_two["candidate"]["loaded"] is True
    record_lap(2.0)
    accepted_two = compensator.finish_round(12.0)
    refining_report = json.loads(request_path.read_text(encoding="utf-8"))
    refining_prompt = _build_design_prompt(refining_report)
    assert accepted_two["accepted"] is True
    assert "12.000000 deg^2" in refining_prompt
    assert "Preserve useful behavior and apply the 4.0 deg RMSE threshold separately" in refining_prompt
    assert "<=30" not in refining_prompt

    # A worse third candidate is rolled back. Its evidence remains available,
    # while the stable parameters remain those accepted in round two.
    write_bias_candidate(2, 300.0, "round-three-bad")
    started_three = compensator.begin_round()
    assert started_three["candidate"]["loaded"] is True
    record_lap(6.0)
    rejected_three = compensator.finish_round(108.0)
    rollback_report = json.loads(request_path.read_text(encoding="utf-8"))
    rollback_prompt = _build_design_prompt(rollback_report)
    assert rejected_three["rolled_back"] is True
    assert compensator.generation == 2
    assert compensator.active["bias_mv"][0] == pytest.approx(200.0)
    assert rollback_report["tested_parameters"]["bias_mv"][0] == pytest.approx(300.0)
    assert rollback_report["stable_parameters_after_audit"]["bias_mv"][0] == pytest.approx(200.0)
    assert rollback_report["design_memory_last_5"][-1]["outcome"] == "rolled_back"
    assert "Apply the 4.0 deg RMSE threshold separately" in rollback_prompt
    assert "stable_compensator_after_audit" in rollback_prompt


def test_codex_source_class_candidate_runs_behind_external_safety(tmp_path):
    reference = np.zeros((3, 3), dtype=float)
    state_path = tmp_path / "state.json"
    candidate_path = tmp_path / "candidate.json"
    request_path = tmp_path / "request.json"
    compensator = CodexCompensator(
        str(state_path), str(candidate_path), str(request_path),
        reference, reference, dt=0.05, u_max=2200.0)

    for index in range(3):
        compensator.apply(
            np.zeros(3), np.array([20.0, 0.0, 0.0]),
            np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6),
            "LLLLLL", index * 0.05)
    compensator.finish_round(400.0)

    source_code = """class Compensator:
    def __init__(self):
        self.kp_matrix = np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.kd_matrix = np.zeros((6, 3))
        self.ki_matrix = np.zeros((6, 3))
        self.integral_limit_deg_s = np.zeros(3)
        self.output_limit_mv = np.ones(6) * 600.0
        self.output_slew_mv_per_s = np.ones(6) * 2000.0
        self.previous_output = np.zeros(6)
        self.integral_error = np.zeros(3)
        self.last_time = 0.0
    def reset(self, feedback):
        self.previous_output = np.zeros(6)
        self.integral_error = np.zeros(3)
        self.last_time = 0.0
    def compute(self, u_mpc, pose, velocity, reference_pose, reference_velocity, pressure, lock_state, trajectory_time):
        feature = np.asarray(reference_pose) - np.asarray(pose)
        velocity_error = np.asarray(reference_velocity) - np.asarray(velocity)
        self.integral_error = np.clip(self.integral_error, -self.integral_limit_deg_s, self.integral_limit_deg_s)
        lock_mask = np.asarray(lock_state) == "L"
        self.previous_output = self.previous_output * lock_mask
        raw = np.matmul(self.kp_matrix, feature) + np.matmul(self.kd_matrix, velocity_error) + np.matmul(self.ki_matrix, self.integral_error)
        target = np.clip(raw, -self.output_limit_mv, self.output_limit_mv) * lock_mask
        dt = float(trajectory_time) - self.last_time
        if dt < 0.0:
            dt = 0.0
        step = np.clip(target - self.previous_output, -self.output_slew_mv_per_s * dt, self.output_slew_mv_per_s * dt)
        output = self.previous_output + step
        self.previous_output = output
        self.last_time = float(trajectory_time)
        return output
"""
    candidate_path.write_text(json.dumps({
        "schema": 1,
        "task_hash": compensator.task_hash,
        "base_generation": 0,
        "created_by": "source-test",
        "rationale": "nonlinear stateful correction",
        "design_strategy": "thresholded memory",
        "expected_effect": "earlier V0 drive",
        "failure_analysis": "baseline lag",
        "source_code": source_code,
    }), encoding="utf-8")

    started = compensator.begin_round()
    assert started["candidate"]["loaded"] is True
    assert compensator.active_kind == "python_class"
    locked_total, locked_comp = compensator.apply(
        np.array([10.0, 0.0, 0.0]), np.zeros(3),
        np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6),
        "LLLLLL", 0.25)
    unlocked_total, unlocked_comp = compensator.apply(
        np.array([10.0, 0.0, 0.0]), np.zeros(3),
        np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6),
        "oLLLLL", 0.30)
    relocked_total, relocked_comp = compensator.apply(
        np.array([10.0, 0.0, 0.0]), np.zeros(3),
        np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6),
        "LLLLLL", 0.35)
    assert locked_total[0] == pytest.approx(500.0)
    assert locked_comp[0] == pytest.approx(500.0)
    assert unlocked_total[0] == pytest.approx(0.0)
    assert unlocked_comp[0] == pytest.approx(0.0)
    assert relocked_total[0] == pytest.approx(100.0)
    assert relocked_comp[0] == pytest.approx(100.0)

    accepted = compensator.finish_round(100.0)
    assert accepted["accepted"] is True
    restored = CodexCompensator(
        str(state_path), str(candidate_path), str(request_path),
        reference, reference, dt=0.05, u_max=2200.0)
    assert restored.active_kind == "python_class"
    assert "class Compensator" in restored.active_source


def test_nano25_gui_stream_contract_and_operation_panel():
    payload = qt_gui_module.parse_nano25_stream_line(
        'NANO25_DATA {"state":"running","hz":999.5,'
        '"force_n":[-0.031,0.797,1.86],"torque_nm":[0.023,-0.046,-0.022]}')
    assert payload["state"] == "running"
    assert payload["hz"] == pytest.approx(999.5)
    assert payload["force_n"] == pytest.approx([-0.031, 0.797, 1.86])
    assert payload["torque_nm"] == pytest.approx([0.023, -0.046, -0.022])
    assert qt_gui_module.parse_nano25_stream_line("ordinary recorder output") is None

    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")
    assert "def startNano25" in source
    assert "def stopNano25" in source
    assert "id=\"qt-operation-nano25\"" not in source  # DOM id is assigned in JS.
    assert "nanoPanel.id = 'qt-operation-nano25'" in source
    assert "nano25Title:'Nano25'" in source
    assert "F_GUI[Y,Z,X]" in source
    assert "const guiForce = force.map(value => -(Number(value) || 0))" in source
    assert source.count('class=\"qt-nano25-value\" data-nano-index=') == 6
    assert "F<sub>Y</sub>" in source
    assert "F<sub>Z</sub>" in source
    assert "F<sub>X</sub>" in source
    assert "T<sub>X</sub>" in source
    assert "guiForce.concat(torque).forEach" in source

    recorder = (Path(qt_gui_module.__file__).resolve().parents[1] /
                "setup_for_shuai/src/nano25_acquisition_py/record_nano25.py")
    recorder_source = recorder.read_text(encoding="utf-8")
    assert 'p.add_argument("--gui-stream"' in recorder_source
    assert '"NANO25_DATA "' in recorder_source
    assert '"NANO25_EVENT "' in recorder_source
    assert "GUI_STREAM_HZ = 20.0" in recorder_source
    assert 'state["gui_ft_sum"] += ft' in recorder_source
    assert 'status_ft = state["gui_ft_sum"] / sample_count' in recorder_source
    assert '"source_hz": round(hz, 1)' in recorder_source
    assert '"averaged_samples": averaged_samples' in recorder_source
    assert 'process.stdin.write("stop\\n")' in source
    assert "Hz${averaged?' avg':''}" in source
    assert "def setNano25ControlRequired" in source
    assert "Nano25 must remain active while control is running." in source
    assert "window.__qtRequireNano25ForControl" not in source
    assert "abortControlForNanoLoss" not in source
    assert "__qtNano25ExternalControlRequired" not in source
    assert "nanoControlRunning() && nanoActive(nanoState)" in source
    assert "Nano25 is optional" in source
    assert "nano25Title:'Nano25'" in source
    assert "未启动也可执行轨迹" in source
    close_body = source.split("    def close(self):", 1)[1].split(
        "    @QtCore.pyqtSlot", 1)[0]
    assert "_stop_nano25" not in close_body

    html = (Path(qt_gui_module.__file__).resolve().parents[1] /
            "pam_shoulder_gui.html").read_text(encoding="utf-8")
    assert "now-lastHeavy>=50" in html
    assert "now-imuLastApply>=50" in html
    assert "now-lastHeavy>66" not in html


def test_codex_report_has_no_global_mse_aggressiveness(tmp_path):
    reference = np.zeros((1, 3), dtype=float)
    request_path = tmp_path / "request.json"
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(request_path), reference, reference,
        dt=0.05, u_max=2200.0, target_mse_deg2=15.0)
    compensator.apply(
        np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3),
        np.zeros(6), np.zeros(6), "LLLLLL", 0.0)
    compensator.finish_round(100.0)
    report = json.loads(request_path.read_text(encoding="utf-8"))

    assert not hasattr(compensator, "_design_aggressiveness")
    assert "design_aggressiveness" not in report
    assert "aggressive_tracking_mse_deg2" not in report["hard_constraints"]
    assert "very_aggressive_tracking_mse_deg2" not in report["hard_constraints"]


def test_codex_marks_repeated_axis_phase_errors_as_priority_hotspots(tmp_path):
    reference = np.zeros((24, 3), dtype=float)
    request_path = tmp_path / "request.json"
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(request_path), reference, reference,
        dt=0.05, u_max=2200.0, target_mse_deg2=15.0)

    def record_same_y_hotspot():
        compensator.begin_round()
        for index in range(len(reference)):
            measured = np.zeros(3)
            if 8 <= index < 12:
                measured[1] = -8.0
            compensator.apply(
                np.zeros(3), measured, np.zeros(3), np.zeros(3),
                np.zeros(6), np.zeros(6), "LLLLLL", index*0.05)
        compensator.finish_round(0.0)

    record_same_y_hotspot()
    record_same_y_hotspot()

    report = json.loads(request_path.read_text(encoding="utf-8"))
    recurrent = report["recurrent_phase_error_hotspots"]
    assert recurrent
    assert all(item["axis"] == "Y" for item in recurrent)
    assert all(item["occurrences_in_recent_valid_laps"] == 2
               for item in recurrent)
    assert all(item["priority"] == "repeated_large_error" for item in recurrent)
    assert all(item["signed_error_consistency"] == "consistent"
               for item in recurrent)


def test_codex_compensator_rejects_a_noop_candidate(tmp_path):
    reference = np.zeros((1, 3), dtype=float)
    candidate_path = tmp_path / "candidate.json"
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(candidate_path),
        str(tmp_path / "request.json"), reference, reference,
        dt=0.05, u_max=2200.0, max_slew_mv_s=0.0)
    compensator.apply(
        np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3),
        np.zeros(6), np.zeros(6), "LLLLLL", 0.0)
    compensator.finish_round(3.0)
    candidate_path.write_text(json.dumps({
        "schema": compensator.SCHEMA,
        "task_hash": compensator.task_hash,
        "base_generation": 0,
        "created_by": "test",
        "rationale": "Keep the controller unchanged.",
        "design_strategy": "Copy the current parameters.",
        "expected_effect": "No output change.",
        "failure_analysis": "No experiment proposed.",
        "parameters": compensator.active,
    }), encoding="utf-8")

    started = compensator.begin_round()

    assert started["candidate"]["loaded"] is False
    assert started["candidate"]["error"] is True
    assert "no-op or too small" in started["candidate"]["reason"]
    assert compensator.generation == 0
    assert compensator.canary_active is False


def test_codex_parameters_have_no_amplitude_or_per_round_change_cap(tmp_path):
    reference = np.zeros((1, 3), dtype=float)
    compensator = CodexCompensator(
        str(tmp_path / "state.json"), str(tmp_path / "candidate.json"),
        str(tmp_path / "request.json"), reference, reference,
        dt=0.05, u_max=2200.0)
    parameters = json.loads(json.dumps(compensator.active))
    parameters["kp_mv_per_deg"][0][0] = 1.0e6
    parameters["bias_mv"][0] = 1.0e6
    parameters["feedforward_knots"] = [{
        "time_s": 0.0, "delta_u_mv": [1.0e6, 0, 0, 0, 0, 0]}]

    validated = compensator._validate_parameters(parameters, compensator.active)
    compensator.active = validated
    total, _ = compensator.apply(
        np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3),
        np.zeros(6), np.zeros(6), "LLLLLL", 0.0)

    assert validated["kp_mv_per_deg"][0][0] == 1.0e6
    assert validated["bias_mv"][0] == 1.0e6
    assert total[0] == 2200.0
    assert np.all(total <= 2200.0)


def test_mode1_loop_uses_one_second_zero_output_handoff_without_pose_gate():
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")

    assert "const mode1InterLoopDelayS = 1.0;" in source
    assert "writeOpen([0,0,0,0,0,0]);" in source
    assert "if (elapsed >= mode1InterLoopDelayS)" in source
    assert "六路0mV已保持1s；开始第${cycleLap}轮" in source
    assert "returnPoseToleranceDeg" not in source
    assert "mode1ReturnTimeoutS" not in source


def test_debug_console_is_native_popup_and_keeps_operation_mode_in_main_window():
    html = (Path(qt_gui_module.__file__).resolve().parents[1] /
            "pam_shoulder_gui.html").read_text(encoding="utf-8")
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")

    assert "window.open('','tro-debug-console'" in html
    assert "const clone=source.cloneNode(true);" in html
    assert "node.dataset.debugSourceId=node.id;" in html
    assert "original.click()" in html
    assert "class NativePopupWebView(QWebEngineView):" in source
    assert "def createWindow(self, window_type):" in source
    assert "view = NativePopupWebView()" in source
    assert "QWebEngineSettings.JavascriptCanOpenWindows" in source
    assert "debugConsoleIsOpen()) debugConsoleClose();" in source
    assert "else debugConsoleOpen();" in source


def test_selected_automatic_modes_run_as_one_ordered_queue():
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")

    assert "const automaticModeOrder = ['mode1','mode2','mode3','mode4'];" in source
    assert "const selectedModes = new Set();" in source
    assert "toggleAutoMode(card.dataset.fixedMode);" in source
    assert "const modesToRun = automaticModeOrder.filter(modeName => selectedModes.has(modeName));" in source
    assert "autoQueue = {modes:modesToRun, index:0, active:true, transitionTimer:0};" in source
    assert "window.__qtAdvanceFixedOpenQueue = completedMode =>" in source
    assert "queue.transitionTimer = setTimeout(() =>" in source
    assert "}, 1000);" in source
    assert "writeOpen([0,0,0,0,0,0]);" in source
    assert "if (queueResult && queueResult.continue)" in source
    assert "tcPlaying = true;" in source


def test_mode4_drives_negative_y_then_positive_phi_with_prelocks():
    source = Path(qt_gui_module.__file__).read_text(encoding="utf-8")

    assert '<option value="mode4">Mode 4｜Y −30° → φ +20° → 全回0 循环</option>' in source
    assert 'data-fixed-mode="mode4"' in source
    assert 'id="qt-op-m4-y" value="-30"' in source
    assert 'id="qt-op-m4-phi" value="+20"' in source
    assert "targetYDeg:signedNumber('qt-op-m4-y',-30)" in source
    assert "targetPhiDeg:signedNumber('qt-op-m4-phi',20)" in source
    assert "const isTwoAxisMode = activeMode === 'mode1' || activeMode === 'mode4';" in source
    assert "const isMode4 = activeMode === 'mode4';" in source
    assert "const prelockChannel = phase === 0 ? (isMode4 ? 1 : 0) : (isMode4 ? 2 : 4);" in source
    assert "const yTargetReached = isMode4 ? yDeg <= targetYDeg : yDeg >= targetYDeg;" in source
    assert "? phiDeg >= targetPhiDeg" in source
    assert "out = isMode4 ? [0,fullMv,0,0,0,0] : [fullMv,0,0,0,0,0];" in source
    assert "? [0,0,fullMv,0,0,0]" in source
    assert "targetYDeg = activeMode === 'mode4' ? -30.0 : 60.0;" in source
    assert "Mode 4｜Y ${targetYDeg.toFixed(1)}° → φ ${targetPhiDeg.toFixed(1)}° → 回零" in source


def test_data_workbook_has_typed_flat_headers_and_frozen_header(tmp_path):
    row = [None] * len(SerialBridge.DATA_RECORD_HEADERS)
    row[0] = 12.05
    row[1:4] = [1.25, -2.5, 3.75]
    target = tmp_path / "experiment.xlsx"

    SerialBridge._write_data_workbook([row], str(target))

    workbook = load_workbook(target, read_only=False, data_only=True)
    sheet = workbook["控制响应"]
    assert sheet.freeze_panes == "A2"
    assert sheet.auto_filter.ref.endswith("2")
    assert sheet.max_column == 37
    assert [cell.value for cell in sheet[1]] == list(SerialBridge.DATA_RECORD_HEADERS)
    assert sheet["A1"].value == "response_time_s"
    assert sheet["B1"].value == "X_deg"
    assert sheet["E1"].value == "X_speed_deg_s"
    assert sheet["F1"].value == "Y_speed_deg_s"
    assert sheet["G1"].value == "phi_speed_deg_s"
    assert sheet["N1"].value == "L_A1+5_mm"
    assert sheet["O1"].value == "d_active_A1+5_mm"
    assert sheet["P1"].value == "d_passive_A1+5_mm"
    assert sheet["Q1"].value == "L_A2+3_mm"
    assert sheet["R1"].value == "d_active_A2+3_mm"
    assert sheet["S1"].value == "d_passive_A2+3_mm"
    assert sheet["AC1"].value == "L_A10_mm"
    assert sheet["AD1"].value == "d_active_A10_mm"
    assert sheet["AE1"].value == "d_passive_A10_mm"
    assert sheet["AF1"].value == "F_X_N"
    assert sheet["AK1"].value == "T_Z_Nm"
    assert sheet["A2"].value == pytest.approx(12.05)
    assert sheet["A2"].number_format == "0.000"
    assert sheet["B2"].value == 1.25


def test_response_sample_uses_requested_physical_order_and_gui_force_axes():
    bridge = SerialBridge()
    now = time.time()
    bridge._latest_frame = {
        "wall": now,
        "kpa": np.array([15.0, 48.0, 23.0, 67.0, 9.0, 10.0]),
        "channel_length_mm": [115.0, 148.0, 123.0, 167.0, 109.0, 110.0],
        "lens": [0.0] * 20,
    }
    bridge._nano25_latest.update({
        "state": "running", "received_wall": now,
        # Nano stream force order is GUI [Y,Z,X], opposite signs.
        "force_n": [2.0, 3.0, 1.0],
        "torque_nm": [4.0, 5.0, 6.0],
    })
    bridge._data_recording = {
        "path": "unused.xlsx", "t0": now, "last_wall": 0.0, "rows": [],
    }

    assert bridge.appendDataSample(json.dumps({
        "response_time_s": 4.25,
        "pose": [1.0, 2.0, 3.0],
    }))
    row = bridge._data_recording["rows"][0]

    assert len(row) == 37
    assert row[0] == pytest.approx(4.25)
    assert row[1:4] == pytest.approx([1.0, 2.0, 3.0])
    assert row[4:7] == pytest.approx([0.0, 0.0, 0.0])
    assert row[7:13] == pytest.approx([15.0, 23.0, 48.0, 67.0, 9.0, 10.0])
    assert row[13:31] == pytest.approx([
        115.0, 0.0, 0.0,
        123.0, 0.0, 0.0,
        148.0, 0.0, 0.0,
        167.0, 0.0, 0.0,
        109.0, 0.0, 0.0,
        110.0, 0.0, 0.0,
    ])
    assert row[31:34] == pytest.approx([-1.0, -2.0, -3.0])
    assert row[34:37] == pytest.approx([4.0, 5.0, 6.0])

    # The first valid sample is the zero. Subsequent A1/A5 active and passive
    # means preserve their signs instead of being converted with abs().
    bridge._data_recording["last_wall"] = 0.0
    bridge._latest_frame["lens"][0] = 2.0
    bridge._latest_frame["lens"][4] = -4.0
    bridge._latest_frame["lens"][10] = 3.0
    bridge._latest_frame["lens"][14] = 5.0
    bridge._latest_frame["lens"][1:4] = [6.0, 8.0, -3.0]
    bridge._latest_frame["lens"][5:10] = [10.0, 14.0, 5.0, 9.0, -10.0]
    bridge._latest_frame["lens"][11:14] = [-2.0, 4.0, 7.0]
    bridge._latest_frame["lens"][15:20] = [-8.0, -4.0, -1.0, -9.0, 10.0]
    assert bridge.appendDataSample(json.dumps({
        "response_time_s": 4.30,
        "pose": [2.0, 4.0, 2.0],
    }))
    assert bridge._data_recording["rows"][1][4:7] == pytest.approx([20.0, 40.0, -20.0])
    assert bridge._data_recording["rows"][1][14:16] == pytest.approx([-1.0, 4.0])
    assert bridge._data_recording["rows"][1][17:19] == pytest.approx([7.0, 1.0])
    assert bridge._data_recording["rows"][1][20:22] == pytest.approx([1.0, 3.0])
    assert bridge._data_recording["rows"][1][23:25] == pytest.approx([12.0, -6.0])
    assert bridge._data_recording["rows"][1][26:28] == pytest.approx([9.0, -9.0])
    assert bridge._data_recording["rows"][1][29:31] == pytest.approx([-10.0, 10.0])


def test_saved_control_response_zeroes_signed_l15_displacements(monkeypatch):
    import mpc_station as station

    monkeypatch.setattr(station, "TELEM", [
        {
            "t": 0.0, "q_gui": [0.0, 0.0, 0.0],
            "pressure_kpa": [0.0] * 6,
            "channel_length_mm": [235.0] * 6,
            "l15_displacement_raw_mm": [10.0, -4.0],
            "group_displacement_raw_mm": [
                [10.0, -4.0], [20.0, -10.0], [30.0, -20.0],
                [40.0, -30.0], [50.0, -40.0], [60.0, -50.0],
            ],
        },
        {
            "t": 0.05, "q_gui": [0.0, 1.0, 0.0],
            "pressure_kpa": [0.0] * 6,
            "channel_length_mm": [234.0] * 6,
            "l15_displacement_raw_mm": [12.0, -7.0],
            "group_displacement_raw_mm": [
                [12.0, -7.0], [24.0, -15.0], [36.0, -27.0],
                [48.0, -39.0], [60.0, -51.0], [72.0, -63.0],
            ],
        },
    ])

    rows = SerialBridge()._control_response_rows()

    assert rows[0][4:7] == pytest.approx([0.0, 0.0, 0.0])
    assert rows[1][4:7] == pytest.approx([0.0, 20.0, 0.0])
    assert rows[0][14:16] == pytest.approx([0.0, 0.0])
    assert rows[1][14:16] == pytest.approx([2.0, -3.0])
    assert rows[1][17:19] == pytest.approx([6.0, -7.0])
    assert rows[1][20:22] == pytest.approx([4.0, -5.0])
    assert rows[1][23:25] == pytest.approx([8.0, -9.0])
    assert rows[1][26:28] == pytest.approx([10.0, -11.0])
    assert rows[1][29:31] == pytest.approx([12.0, -13.0])


def test_response_sample_does_not_clear_previous_round_on_time_restart():
    bridge = SerialBridge()
    now = time.time()
    bridge._latest_frame = {
        "wall": now,
        "kpa": np.zeros(6),
        "channel_length_mm": [235.0] * 6,
    }
    bridge._data_recording = {
        "path": "unused.xlsx", "t0": now, "last_wall": 0.0, "rows": [],
    }

    assert bridge.appendDataSample(json.dumps({
        "response_time_s": 10.0, "pose": [9.0, 9.0, 9.0],
    }))
    bridge._data_recording["last_wall"] = 0.0
    assert bridge.appendDataSample(json.dumps({
        "response_time_s": 0.0, "pose": [1.0, 2.0, 3.0],
    }))

    assert len(bridge._data_recording["rows"]) == 2
    assert bridge._data_recording["rows"][0][:7] == pytest.approx(
        [10.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0])
    assert bridge._data_recording["rows"][1][:7] == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

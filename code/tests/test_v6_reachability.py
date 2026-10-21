import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from v6_reachability import (
    _dense_endpoint_envelope_mm,
    _sample_directional_swing_poses,
    _tip_positions_mm,
    measured_extrema_calibration,
)


def test_directional_swing_sampling_fills_closed_measured_envelope():
    poses = _sample_directional_swing_poses(
        np.random.default_rng(20260910), 100000,
        -20.0, 20.0, -38.0, 80.0, -70.0, 75.0)
    x_scale = np.where(poses[:, 0] < 0.0, 20.0, 20.0)
    y_scale = np.where(poses[:, 1] < 0.0, 38.0, 80.0)
    normalized_radius = np.sqrt(
        (poses[:, 0] / x_scale)**2 + (poses[:, 1] / y_scale)**2)

    assert np.max(normalized_radius) <= 1.0 + 1.0e-12
    assert np.min(poses[:, 0]) < -19.8
    assert np.max(poses[:, 0]) > 19.8
    assert np.min(poses[:, 1]) < -37.6
    assert np.max(poses[:, 1]) > 79.5
    assert np.all((poses[:, 2] >= -70.0) & (poses[:, 2] <= 75.0))


def test_dense_endpoint_envelope_matches_v6_forward_kinematics():
    blob = {
        "swing_x_range_deg": [-20.0, 20.0],
        "swing_y_range_deg": [-38.0, 80.0],
        "results": [{"pose_deg": [0.0, 0.0, 0.0]}],
    }
    dense = _dense_endpoint_envelope_mm(blob, point_count=1000)
    direct = _tip_positions_mm({"results": [
        {"pose_deg": [20.0, 0.0, 0.0]},
        {"pose_deg": [0.0, 80.0, 0.0]},
    ]})

    assert dense.shape == (1360, 3)
    assert np.allclose(np.linalg.norm(dense, axis=1), 216.0, atol=1.0e-10)
    assert np.any(np.linalg.norm(dense - direct[0], axis=1) < 1.0e-9)
    assert np.any(np.linalg.norm(dense - direct[1], axis=1) < 1.0e-9)


def test_physical_installation_places_both_y_limits_below_joint_centre():
    displayed = _tip_positions_mm({"results": [
        {"pose_deg": [0.0, 0.0, 0.0]},
        {"pose_deg": [0.0, 80.0, 0.0]},
        {"pose_deg": [0.0, -38.0, 0.0]},
    ]})

    assert np.allclose(displayed[0], [0.0, 0.0, -216.0], atol=1.0e-9)
    assert -40.0 < displayed[1, 2] < 0.0
    assert displayed[2, 2] < 0.0


def test_latest_extrema_calibration_uses_measured_convex_xy_envelope():
    measurement = {"created": "test", "seed": 7, "results": [
        {"valid": True, "pose_deg": [-42.0, 62.0, 30.0]},
        {"valid": True, "pose_deg": [41.0, 56.0, -35.0]},
        {"valid": True, "pose_deg": [0.0, -38.0, 5.0]},
        {"valid": True, "pose_deg": [-6.0, 67.0, 75.0]},
    ]}

    calibration = measured_extrema_calibration(measurement)
    blob = {"sample_seed": 7, **calibration,
            "results": [{"pose_deg": [0.0, 0.0, 0.0]}]}
    dense = _dense_endpoint_envelope_mm(blob, point_count=1000)

    assert calibration["valid_extreme_count"] == 4
    assert calibration["swing_x_range_deg"] == [-42.0, 41.0]
    assert calibration["swing_y_range_deg"] == [-38.0, 67.0]
    assert calibration["twist_range_deg"] == [-35.0, 75.0]
    assert dense.shape == (1360, 3)
    assert np.allclose(np.linalg.norm(dense, axis=1), 216.0, atol=1.0e-10)

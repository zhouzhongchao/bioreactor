import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import v6_geometry_calibration as geometry_cal


def _exciting_trajectory(samples=600):
    rng = np.random.default_rng(20260909)
    return rng.uniform([-28.0, -28.0, -35.0], [28.0, 28.0, 35.0],
                       size=(samples, 3))


def test_kendall_tau_b_corrects_for_tied_values():
    predicted = np.array([1.0, 1.0, 2.0, 3.0])
    measured = np.array([1.0, 2.0, 2.0, 3.0])

    assert np.isclose(geometry_cal._kendall_tau_b(predicted, measured), 0.8)


def test_reports_kendall_tau_b_and_unscaled_rmse_without_changing_mapping():
    q_deg = _exciting_trajectory()
    predicted = geometry_cal.predicted_channel_lengths_mm(q_deg)
    signed_scales = np.array([1.05, -0.93, 1.12, 0.88, -1.20, 0.78])
    measured = 247.0 + signed_scales * (predicted - predicted[0])

    result = geometry_cal.fit_geometry_samples(q_deg, measured, source="synthetic")

    fits = [channel["fixed_mapping_fit"] for channel in result["channels"]]
    tau_b = np.array([fit["kendall_tau_b"] for fit in fits])
    assert np.allclose(tau_b, np.sign(signed_scales), atol=1.0e-12)
    expected_rmse = np.sqrt(np.mean(
        ((predicted - predicted[0]) * (1.0 - signed_scales)) ** 2, axis=0))
    assert np.allclose([fit["rmse_mm"] for fit in fits], expected_rmse, atol=1.0e-10)
    assert all("signed_scale" not in fit for fit in fits)
    assert all("scale_abs" not in fit for fit in fits)
    assert result["ready_to_apply"] is True
    assert result["application_state"] == "not_applied"
    assert result["immutable_contract"]["valve_order"] == [
        "V0=15", "V1=48", "V2=23", "V3=67", "V4=9", "V5=10"]
    assert result["immutable_contract"]["lock_lead_ms"] == 100.0
    assert result["immutable_contract"]["unlock_pressure_kpa"] == 30.0


def test_rejects_channel_with_poor_monotonic_agreement():
    q_deg = _exciting_trajectory()
    predicted = geometry_cal.predicted_channel_lengths_mm(q_deg)
    measured = predicted - predicted[0]
    measured[:, 4] = np.linspace(-12.0, 12.0, measured.shape[0])

    result = geometry_cal.fit_geometry_samples(q_deg, measured, source="synthetic_bad")

    assert result["channels"][4]["fixed_mapping_fit"]["accepted"] is False
    assert result["ready_to_apply"] is False
    assert result["application_state"] == "not_applied"


def test_channel_masks_limit_each_fit_to_its_own_excitation_rows():
    q_deg = _exciting_trajectory(samples=240)
    predicted = geometry_cal.predicted_channel_lengths_mm(q_deg)
    measured = predicted - predicted[0]
    masks = np.zeros((q_deg.shape[0], 6), dtype=bool)
    for channel in range(6):
        masks[channel * 40:(channel + 1) * 40, channel] = True

    result = geometry_cal.fit_geometry_samples(
        q_deg, measured, channel_sample_masks=masks, source="masked")

    assert [c["fixed_mapping_fit"]["samples"] for c in result["channels"]] == [40] * 6
    assert result["ready_to_apply"] is False
    assert all("samples<120" in c["fixed_mapping_fit"]["reasons"]
               for c in result["channels"])

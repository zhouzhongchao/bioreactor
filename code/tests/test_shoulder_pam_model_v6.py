import unittest

import numpy as np

import shoulder_pam_model as legacy
import shoulder_pam_model_v6 as v6
from control import validate_v6_offline as validator


class TestV6Geometry(unittest.TestCase):
    def test_logged_q_columns_are_legacy_model_coordinates(self):
        validator._prepare_legacy()
        row = {"q1": "12.0", "q2": "-7.0", "q3": "9.0"}
        q_legacy = np.radians([12.0, -7.0, 9.0])
        lengths = legacy.pam_lengths_mm(q_legacy)
        expected = np.array([
            np.mean(lengths[list(rows)]) for rows in v6.VALVE_GEOMETRY_ROWS
        ])
        np.testing.assert_allclose(
            validator._geometry_channel_lengths("legacy", row), expected, atol=1.0e-12)
        np.testing.assert_allclose(
            validator._geometry_channel_lengths("v6", row), expected, atol=1.0e-10)

    def test_zero_pose_matches_v51_report(self):
        expected_lengths = np.array([300.541179]*8 + [238.272384]*2)
        np.testing.assert_allclose(v6.pam_lengths_mm(np.zeros(3)), expected_lengths,
                                   atol=1.0e-6)
        expected_h = np.array([
            [0.000040, -0.000040, 0.000040, -0.000040, -0.000040,
             0.000040, -0.000040, 0.000040, -0.057165, 0.057165],
            [0.010228, 0.025341, 0.025341, 0.010228, -0.010228,
             -0.025341, -0.025341, -0.010228, -0.033782, -0.033782],
            [-0.025341, -0.010228, 0.010228, 0.025341, -0.025341,
             -0.010228, 0.010228, 0.025341, 0.026826, -0.026826],
        ])
        np.testing.assert_allclose(v6.H_matrix(np.zeros(3)), expected_h, atol=5.0e-7)
        rank, singular_values, condition = v6.spatial_moment_arm_metrics(np.zeros(3))
        self.assertEqual(rank, 3)
        np.testing.assert_allclose(singular_values, [0.093405, 0.072592, 0.047305],
                                   atol=1.0e-6)
        self.assertAlmostEqual(condition, 1.97452, places=4)

    def test_length_jacobian_is_geometric_derivative(self):
        q = np.array([0.42, -0.31, 0.27])
        numeric = np.zeros((v6.N_PAM, 3))
        eps = 1.0e-7
        for column in range(3):
            step = np.zeros(3)
            step[column] = eps
            numeric[:, column] = (
                v6.pam_lengths_mm(q + step) - v6.pam_lengths_mm(q - step)
            )/(2.0*eps)*v6.MM_TO_M
        np.testing.assert_allclose(v6.length_jacobian(q), numeric, atol=2.0e-9)

    def test_generalized_force_identity(self):
        q = np.array([0.35, 0.18, -0.44])
        force = np.linspace(10.0, 100.0, v6.N_PAM)
        via_jacobian = -v6.length_jacobian(q).T @ force
        via_spatial_moment = v6.angular_velocity_matrix(q).T @ v6.H_matrix(q) @ force
        np.testing.assert_allclose(via_jacobian, via_spatial_moment, atol=1.0e-12)

    def test_gui_pose_roundtrip(self):
        q = v6.q_from_gui_degrees(12.0, -7.0, 9.0)
        sx, sy, phi = np.radians([12.0, -7.0, 9.0])
        theta = np.hypot(sx, sy)
        axis = np.array([sx/theta, sy/theta, 0.0])
        skew = np.array([[0.0, 0.0, axis[1]],
                         [0.0, 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        expected = (np.eye(3) + np.sin(theta)*skew
                    + (1.0-np.cos(theta))*(skew @ skew)) @ v6.rot_z(phi)
        np.testing.assert_allclose(v6.rotation_matrix(q), expected, atol=1.0e-12)


class TestV6HardwareConstraints(unittest.TestCase):
    def test_six_channel_mapping_is_unchanged(self):
        self.assertEqual(v6.VALVE_PHYSICAL_MUSCLES,
                         ((1, 5), (4, 8), (2, 3), (6, 7), (9,), (10,)))
        self.assertEqual(v6.VALVE_GEOMETRY_ROWS,
                         ((2, 1), (6, 5), (3, 7), (0, 4), (8,), (9,)))
        np.testing.assert_array_equal(
            v6.valve_to_geometry_rows([10, 20, 30, 40, 50, 60]),
            [40, 10, 10, 30, 40, 20, 20, 30, 50, 60])

    def test_antagonists_are_a_hard_constraint(self):
        projected = v6.project_antagonist_valves([10, 20, 40, 30, 50, 50])
        np.testing.assert_array_equal(projected, [0, 20, 40, 0, 50, 0])
        for left, right in v6.ANTAGONIST_PAIRS:
            self.assertFalse(projected[left] > 0 and projected[right] > 0)

    def test_lock_lead_is_two_50ms_ticks(self):
        self.assertEqual(v6.lock_lead_steps(0.05), 2)

    def test_30kpa_is_only_an_unlock_judgement(self):
        self.assertTrue(v6.unlock_permitted(30.0))
        self.assertFalse(v6.unlock_permitted(30.0001))
        params = v6.MuscleSystemParameters(
            f0_n=0.0, f1_n_per_kpa=1.0, k0_n_per_mm=0.0,
            b_inflate_n_s_per_mm=1.0, b_deflate_n_s_per_mm=1.0,
            b_inflate_slope_n_s_per_mm_kpa=0.0,
            b_deflate_slope_n_s_per_mm_kpa=0.0,
            k_ss_n_per_mm=0.0, k_sl_n_per_mm=0.0, max_force_n=500.0)
        force = v6.system_force(
            np.full(10, 10.0), np.zeros(10), np.zeros(10), np.zeros(10),
            np.ones(10, dtype=bool), np.ones(10, dtype=bool), params)
        np.testing.assert_allclose(force, 10.0)


class TestV6MuscleEquation(unittest.TestCase):
    def test_identified_three_element_relations_use_mm_units(self):
        params = v6.MuscleSystemParameters()
        pressure = np.full(10, 200.0)
        a = params.arrays()
        stiffness = a["k0_n_per_mm"] + a["k1_n_per_mm_kpa"]*pressure
        active_force = a["f0_n"] + a["f1_n_per_kpa"]*pressure
        b_inf = v6.damping_coefficient(
            pressure, np.ones(10, dtype=bool), params)
        b_def = v6.damping_coefficient(
            pressure, np.zeros(10, dtype=bool), params)
        np.testing.assert_allclose(stiffness, 3.46947)
        np.testing.assert_allclose(active_force, 106.722)
        np.testing.assert_allclose(b_inf, 1.91)
        np.testing.assert_allclose(b_def, 0.178)

    def test_corrected_locked_and_released_branches(self):
        params = v6.MuscleSystemParameters(
            f0_n=2.0, f1_n_per_kpa=0.5, k0_n_per_mm=3.0,
            k1_n_per_mm_kpa=0.0, b_inflate_n_s_per_mm=0.2,
            b_inflate_slope_n_s_per_mm_kpa=0.0,
            b_deflate_n_s_per_mm=0.4,
            b_deflate_slope_n_s_per_mm_kpa=0.0, k_ss_n_per_mm=4.0,
            k_sl_n_per_mm=5.0, x_ss_pre_mm=1.0, max_force_n=500.0)
        pressure = np.full(10, 20.0)
        x_ss = np.full(10, -0.5)
        x_rate = np.full(10, 2.0)
        x_sl = np.full(10, 1.5)
        locked = np.array([True, False]*5)
        result = v6.system_force(
            pressure, x_ss, x_rate, x_sl, locked, np.ones(10, dtype=bool), params)
        active = (2.0 + 0.5*20.0) + 0.2*2.0 + 3.0*(-0.5) + 4.0*(1.0-0.5)
        passive = 5.0*1.5
        np.testing.assert_allclose(result[locked], active)
        np.testing.assert_allclose(result[~locked], passive)


if __name__ == "__main__":
    unittest.main()

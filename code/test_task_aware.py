import unittest

import numpy as np

from dpspsr.core import residual_and_personal_structure
from dpspsr.fixed_privacy import symmetric_noise
from dpspsr.task_aware import (TaskAwareConfig, center_release, conditional_select,
                               intercept_structure, recover_task_aware)


class TaskAwareTests(unittest.TestCase):
    def test_clipping_sensitivity_and_public_intercept(self):
        rng = np.random.default_rng(81)
        for radius in [2, 4, 7]:
            for _ in range(25):
                x, y = rng.normal(size=(30, 8)) * 100, rng.integers(0, 2, 30)
                xp, yp = x.copy(), y.copy()
                xp[0], yp[0] = rng.normal(size=8) * 1e6, 1 - yp[0]
                a, b = [intercept_structure(xx, yy, radius) for xx, yy in [(x, y), (xp, yp)]]
                self.assertAlmostEqual(a[-1, -1], 1)
                self.assertLessEqual(np.linalg.norm(a - b, "fro"), 2 * radius**2 / len(y) + 1e-12)

    def test_centering_matches_clipped_population_covariance(self):
        rng = np.random.default_rng(72)
        x, y = rng.uniform(-1, 1, (50, 6)), rng.integers(0, 2, 50)
        h = intercept_structure(x, y, 7)
        c, variance = center_release(h, 0)
        z = np.column_stack([x, 2 * y - 1])
        np.testing.assert_allclose(c, np.cov(z, rowvar=False, bias=True), atol=1e-12)
        np.testing.assert_array_equal(variance, np.zeros(6))

    def test_centered_cross_block_noise_variance_matches_symmetric_release(self):
        rng = np.random.default_rng(172)
        x = rng.normal(loc=0.35, scale=0.4, size=25000)
        y = rng.choice([-1.0, 1.0], size=len(x), p=[0.3, 0.7])
        z = np.column_stack([x, y, np.ones(len(x))])
        h = z.T @ z / len(z)
        sigma = 0.035
        repetitions = 12000
        off_diag = np.triu_indices(3, 1)
        noise = np.zeros((repetitions, 3, 3))
        values = rng.normal(size=(repetitions, len(off_diag[0]))) * (sigma / np.sqrt(2))
        noise[:, off_diag[0], off_diag[1]] = values
        noise[:, off_diag[1], off_diag[0]] = values
        diag_values = rng.normal(size=(repetitions, 3)) * sigma
        noise[:, np.arange(3), np.arange(3)] = diag_values
        noisy = h[None, :, :] + noise
        centered_q = noisy[:, 0, 1] - noisy[:, 0, 2] * noisy[:, 1, 2]
        observed_variance = centered_q.var(ddof=1)
        mu_x, mu_y = h[0, 2], h[1, 2]
        v = sigma**2 / 2
        expected_variance = v * (1 + mu_x**2 + mu_y**2) + v**2
        self.assertAlmostEqual(observed_variance / expected_variance, 1.0, delta=0.05)

    def test_plugin_variance_formula_uses_off_diagonal_mean_noise_scale(self):
        h = np.eye(4)
        sigma = 0.2
        _, variance = center_release(h, sigma)
        v = sigma**2 / 2
        # Means are off-diagonal entries paired with the public intercept.
        np.testing.assert_allclose(variance, v + v**2, atol=1e-15)

    def test_label_cross_block_survives_new_recovery(self):
        # The shared basis can include the label axis. QHQ then deletes ALL q.
        h = np.eye(4)
        h[1, 2] = h[2, 1] = 0.7
        old_joint = h[:3, :3]
        p = np.array([[0.0], [0.0], [1.0]])
        _, hp, _ = residual_and_personal_structure(old_joint, p, 2, 0)
        np.testing.assert_allclose(hp[:-1, -1], 0)
        result = recover_task_aware([h], np.array([0.0]), np.array([1.0]),
                                   TaskAwareConfig(k=1), personal_weight=1)
        self.assertEqual(result["selected_sets"][0].tolist(), [1])
        self.assertGreater(result["recovered"][0][1, 2], 0.6)

    def test_personalization_controls_change_known_task_support(self):
        h0, h1 = np.eye(6), np.eye(6)
        h0[0, 4] = h0[4, 0] = 0.7
        h1[1, 4] = h1[4, 1] = 0.9
        args = ([h0, h1], np.array([0.001, 0.001]), np.array([0.5, 0.5]), TaskAwareConfig(k=1))
        shared = recover_task_aware(*args, personal_weight=0)
        local = recover_task_aware(*args, personal_weight=1)
        adaptive = recover_task_aware(*args)
        np.testing.assert_array_equal(shared["selected_sets"][0], shared["selected_sets"][1])
        self.assertEqual([s.tolist() for s in local["selected_sets"]], [[0], [1]])
        self.assertEqual([s.tolist() for s in adaptive["selected_sets"]], [[0], [1]])
        self.assertTrue(all(d["personal_weight"] > 0.99 for d in adaptive["diagnostics"]))

    def test_identical_clients_turn_off_personal_contrast(self):
        h = np.eye(7)
        h[1, 5] = h[5, 1] = 0.5
        result = recover_task_aware([h, h], np.array([0.1, 0.1]), np.array([0.5, 0.5]), TaskAwareConfig(k=3))
        self.assertEqual([d["personal_weight"] for d in result["diagnostics"]], [0, 0])

    def test_conditional_gain_matches_quadratic_objective(self):
        rng = np.random.default_rng(4)
        b = rng.normal(size=(8, 8))
        joint = b @ b.T
        subset, trace = conditional_select(joint, 5, 0.3)
        a, q = joint[:-1, :-1], joint[:-1, -1]
        previous = 0.0
        for size, row in enumerate(trace, 1):
            idx = subset[:size]
            objective = q[idx] @ np.linalg.solve(a[np.ix_(idx, idx)] + 0.3 * np.eye(size), q[idx])
            self.assertAlmostEqual(objective - previous, row["gain"], places=10)
            previous = objective

    def test_finite_psd_unique_and_invalid_inputs(self):
        rng = np.random.default_rng(23)
        releases = [np.eye(12) + symmetric_noise(rng, 12, 0.3) for _ in range(4)]
        result = recover_task_aware(releases, np.full(4, 0.3), np.full(4, 0.25), TaskAwareConfig(k=7))
        for h, s in zip(result["recovered"], result["selected_sets"]):
            self.assertGreaterEqual(np.linalg.eigvalsh(h).min(), -1e-12)
            self.assertEqual(len(set(s)), 7)
            self.assertTrue(np.all(s < 10))
        with self.assertRaises(ValueError):
            conditional_select(np.eye(4), 4, 0.1)
        with self.assertRaises(ValueError):
            intercept_structure(np.ones((3, 2)), np.ones(3), 1)
        with self.assertRaises(ValueError):
            recover_task_aware(releases, np.full(4, 0.3), np.full(4, 0.1), TaskAwareConfig(k=2))


if __name__ == "__main__":
    unittest.main()

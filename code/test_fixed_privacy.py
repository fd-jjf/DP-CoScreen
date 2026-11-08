import unittest

import numpy as np
import pandas as pd

from dpspsr.core import (dp_shared_operator, population_operator, leading_subspace,
                        shared_structure_from_subspace, residual_and_personal_structure,
                        greedy_select, conditional_score)
from dpspsr.fixed_privacy import (FixedFeatureMap, StructurePrivacyProtocol,
                                 empirical_structure, symmetric_noise)


class FixedPrivacyTests(unittest.TestCase):
    def test_record_replacement_does_not_change_other_rows_or_coordinates(self):
        mapper = FixedFeatureMap(["src_port", "duration"], ["proto"])
        a = pd.DataFrame({"src_port": [80, 443, None], "duration": [1, 2, np.inf],
                          "proto": ["tcp", "udp", "tcp"]})
        b = a.copy()
        b.loc[1] = [65535, 1e200, "previously-unseen-category"]
        xa, xb = mapper.transform(a), mapper.transform(b)
        np.testing.assert_array_equal(xa[[0, 2]], xb[[0, 2]])
        self.assertEqual(xa.shape, xb.shape)
        np.testing.assert_array_equal(mapper.transform(a.iloc[::-1])[::-1], xa)
        self.assertTrue(np.isfinite(xa).all())

    def test_raw_record_sensitivity_and_label_replacement(self):
        rng = np.random.default_rng(99)
        protocol = StructurePrivacyProtocol()
        mapper = FixedFeatureMap(["src_port", "duration"], ["proto"])
        for _ in range(40):
            frame = pd.DataFrame({"src_port": rng.integers(0, 65536, 30),
                                  "duration": rng.normal(0, 1e7, 30), "proto": ["tcp"] * 30})
            changed = frame.copy()
            changed.loc[0] = [None, -1e200, "unseen"]
            y = rng.integers(0, 2, 30)
            yp = y.copy()
            yp[0] = 1 - y[0]
            h = empirical_structure(mapper.transform(frame), y, protocol.clip_norm)
            hp = empirical_structure(mapper.transform(changed), yp, protocol.clip_norm)
            self.assertLessEqual(np.linalg.norm(h - hp, "fro"), 2 * protocol.clip_norm**2 / len(y) + 1e-12)

    def test_classical_calibration_scope(self):
        with self.assertRaises(ValueError):
            StructurePrivacyProtocol(epsilon=4)
        self.assertAlmostEqual(StructurePrivacyProtocol().sigma(20), 2 * StructurePrivacyProtocol().sigma(40))

    def test_correction_and_complete_downstream_invariance(self):
        rng = np.random.default_rng(73)
        h = []
        for _ in range(4):
            a = rng.normal(size=(12, 12))
            h.append(a @ a.T + symmetric_noise(rng, 12, 0.1))
        w, sigmas = np.full(4, 0.25), np.full(4, 0.1)
        raw = population_operator(h, w, 0.2)
        corrected = dp_shared_operator(h, w, sigmas, 0.2)
        d = raw - corrected
        np.testing.assert_allclose(d, np.eye(12) * np.trace(d) / 12, atol=1e-12)
        p0, p1 = leading_subspace(raw, 3), leading_subspace(corrected, 3)
        np.testing.assert_allclose(p0 @ p0.T, p1 @ p1.T, atol=1e-12)
        hs0, hs1 = [shared_structure_from_subspace(p, h, w) for p in [p0, p1]]
        for hi in h:
            hp0 = residual_and_personal_structure(hi, p0, 2, 0.1)[1]
            hp1 = residual_and_personal_structure(hi, p1, 2, 0.1)[1]
            s0 = greedy_select(hs0, hp0, hi, 5, 1, 1, 0.4)[0]
            s1 = greedy_select(hs1, hp1, hi, 5, 1, 1, 0.4)[0]
            np.testing.assert_array_equal(s0, s1)

    def test_score_bound_at_arbitrary_shared_prefix(self):
        rng = np.random.default_rng(1)
        for prefix in [[], [0], [0, 3]]:
            matrices = [rng.normal(size=(9, 9)) for _ in range(6)]
            hs, hp, h, hs0, hp0, h0 = matrices
            a = conditional_score(hs, hp, h, prefix, 1, 1, 0.4)
            b = conditional_score(hs0, hp0, h0, prefix, 1, 1, 0.4)
            bound = (np.max(np.abs(hs[:-1, -1] - hs0[:-1, -1]))
                     + np.max(np.abs(hp[:-1, -1] - hp0[:-1, -1]))
                     + (0.4 * np.max(np.abs(h[:-1, :-1] - h0[:-1, :-1])) if prefix else 0))
            keep = np.isfinite(a)
            self.assertLessEqual(np.max(np.abs(a[keep] - b[keep])), bound + 1e-12)

    def test_paired_variants_share_global_subset_and_have_fixed_budget(self):
        from run_fixed_protocol_pilot import variant_subsets
        rng = np.random.default_rng(17)
        h = []
        for _ in range(4):
            a = rng.normal(size=(13, 13))
            h.append(a @ a.T / 13)
        subsets, _, _ = variant_subsets(h, np.full(4, 0.25), np.full(4, 0.01))
        for variants in subsets.values():
            for selected in variants:
                self.assertEqual(len(selected), 10)
                self.assertEqual(len(set(selected)), 10)
                self.assertTrue(np.all(selected < 12))
        for i in range(4):
            np.testing.assert_array_equal(subsets["Full"][i], subsets["No-correction"][i])
            np.testing.assert_array_equal(subsets["DP-Global"][0], subsets["DP-Global"][i])


if __name__ == "__main__":
    unittest.main()

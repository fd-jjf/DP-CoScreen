import unittest

import numpy as np

from dpspsr.fixed_privacy import empirical_structure
from dpspsr.task_aware import TaskAwareConfig, recover_blockwise
from dpspsr.tree_personalization import recover_tree_hierarchical


def sparse_world(seed, strength, ntrain=500, ntest=100):
    fits, labels = [], []
    for client in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([seed, client]))
        x = rng.uniform(-.7, .7, size=(ntrain + ntest, 24))
        beta = np.zeros(24)
        beta[:3] = .8
        beta[np.arange(3 + 3 * client, 6 + 3 * client)] = strength
        y = (x @ beta + rng.normal(0, .6, len(x)) > 0).astype(int)
        fits.append(x[:ntrain])
        labels.append(y[:ntrain])
    return fits, labels


class TreePersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = sparse_world(812, 1, ntrain=500, ntest=100)
        self.h = [empirical_structure(x, y, 4) for x, y in zip(self.x, self.y)]
        self.weights = np.full(4, .25)
        self.config = TaskAwareConfig(k=4)

    def test_nested_tree_preserves_common_prefix_and_feature_budget(self):
        result = recover_tree_hierarchical(self.h, np.full(4, .001), self.weights,
            self.config, shared_fraction=.5, fixed_ridge=.1)
        prefix = result["diagnostics"][0]["shared_prefix_length"]
        self.assertEqual(prefix, 2)
        for selected in result["selected_sets"]:
            self.assertEqual(len(selected), 4)
            self.assertEqual(len(set(selected.tolist())), 4)
            np.testing.assert_array_equal(selected[:prefix], result["selected_sets"][0][:prefix])
        counts = result["cluster_counts"]
        self.assertTrue(all(b >= a and b-a <= 1 for a, b in zip(counts, counts[1:])))
        self.assertLessEqual(max(counts), 3)
        for diag in result["diagnostics"]:
            self.assertLessEqual(diag["personal_weight"], .25)
            self.assertLessEqual(diag["shared_prefix_length"], 4)

    def test_branch_weight_cap_is_a_configurable_convex_mixture(self):
        result = recover_tree_hierarchical(self.h, np.full(4, .001), self.weights,
            self.config, shared_fraction=.5, max_edge_weight=.75, fixed_ridge=.1)
        self.assertTrue(all(0 <= d["personal_weight"] <= .75
                            for d in result["diagnostics"]))

    def test_identical_releases_revert_to_shared(self):
        same = [self.h[0]] * 4
        result = recover_tree_hierarchical(same, np.zeros(4), self.weights,
            self.config, fixed_ridge=.1)
        shared = recover_blockwise(same, np.zeros(4), self.weights, self.config,
            personal_weight=0, fixed_ridge=.1)
        for selected, baseline in zip(result["selected_sets"], shared["selected_sets"]):
            np.testing.assert_array_equal(selected, baseline)
        self.assertEqual(set(result["cluster_counts"]), {1})

    def test_tree_returns_shared_covariance_for_every_client(self):
        result = recover_tree_hierarchical(self.h, np.full(4, .001), self.weights,
            self.config, fixed_ridge=.1)
        first_a = result["recovered"][0][:-1, :-1]
        for recovered in result["recovered"][1:]:
            np.testing.assert_allclose(recovered[:-1, :-1], first_a)

    def test_unconditional_relevance_ablation_is_supported(self):
        result = recover_tree_hierarchical(
            self.h, np.full(4, .001), self.weights, self.config,
            shared_fraction=.5, fixed_ridge=.1, conditional=False,
        )
        for selected, diag in zip(result["selected_sets"], result["diagnostics"]):
            self.assertEqual(len(selected), self.config.k)
            self.assertEqual(len(set(selected.tolist())), self.config.k)
            self.assertFalse(diag["conditional_redundancy"])

    def test_invalid_tree_parameters(self):
        for kwargs in [{"shared_fraction": 0}, {"split_window": 0},
                       {"silhouette_threshold": 1}, {"max_edge_weight": 1.01},
                       {"conditional": 1}]:
            with self.assertRaises(ValueError):
                recover_tree_hierarchical(self.h, np.full(4, .001), self.weights,
                    self.config, fixed_ridge=.1, **kwargs)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from mllms.interpretability.activation_patching.statistics import (
    holm_adjust,
    stratified_paired_bootstrap,
)


class PatchingStatisticsTest(unittest.TestCase):
    def test_stratified_bootstrap_is_reproducible(self):
        observed = np.array([2.0, 4.0, 3.0, 5.0])
        control = np.array([1.0, 1.0, 2.0, 2.0])
        strata = np.array(["a", "a", "b", "b"])
        first = stratified_paired_bootstrap(
            observed, control, strata, iterations=500, seed=7
        )
        second = stratified_paired_bootstrap(
            observed, control, strata, iterations=500, seed=7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], 2.0)
        self.assertEqual(first["num_examples"], 4)

    def test_holm_adjustment_is_monotone_in_rank_order(self):
        adjusted = holm_adjust([0.01, 0.03, 0.04])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()

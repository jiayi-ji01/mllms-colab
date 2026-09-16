import unittest

import torch

from mllms.interpretability.activation_patching.metrics import site_scores


class PatchingMetricsTest(unittest.TestCase):
    def test_near_zero_denominator_produces_nan_recovery(self):
        delta, recovery = site_scores(
            torch.tensor([2.0, 3.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.0, 2.0]),
            denominator_epsilon=1e-6,
        )
        self.assertTrue(torch.isnan(recovery[0]))
        self.assertAlmostEqual(float(delta[1]), 2.0)
        self.assertAlmostEqual(float(recovery[1]), 1.0)


if __name__ == "__main__":
    unittest.main()

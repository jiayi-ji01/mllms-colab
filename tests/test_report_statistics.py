import unittest

import numpy as np

from mllms.interpretability.activation_patching.report_statistics import (
    assert_same_ids,
    paired_intervals,
)


class ReportStatisticsTest(unittest.TestCase):
    def test_constant_paired_change_has_zero_uncertainty(self):
        baseline = np.array([1., 8., 2., 9.])
        values = np.column_stack([baseline, baseline + 3, baseline + 3 - baseline])
        result = paired_intervals(values, ['a', 'a', 'b', 'b'], iterations=500)
        self.assertAlmostEqual(result['ci_low'][2], 3)
        self.assertAlmostEqual(result['ci_high'][2], 3)
        self.assertAlmostEqual(result['ci_low'][1] - result['ci_low'][0], 3)

    def test_cluster_bootstrap_keeps_repeated_prompts_together(self):
        values = np.array([0.] * 8 + [10.] * 8)
        strata = ['a'] * 16
        clustered = paired_intervals(values, strata, ['p'] * 8 + ['q'] * 8, iterations=1000)
        ordinary = paired_intervals(values, strata, iterations=1000)
        self.assertEqual(clustered['num_clusters'], 2)
        self.assertEqual(clustered['mean'][0], 5)
        self.assertLess(clustered['ci_low'][0], ordinary['ci_low'][0])
        self.assertGreater(clustered['ci_high'][0], ordinary['ci_high'][0])

    def test_fixed_task_weights_and_reproducibility(self):
        values = [1., 1., 7.]
        first = paired_intervals(values, ['a', 'a', 'b'], ['p', 'p', 'q'], iterations=100)
        second = paired_intervals(values, ['a', 'a', 'b'], ['p', 'p', 'q'], iterations=100)
        np.testing.assert_allclose(first['ci_low'], [3])
        np.testing.assert_array_equal(first['ci_low'], second['ci_low'])

    def test_missing_values_and_misalignment_fail(self):
        with self.assertRaises(ValueError):
            paired_intervals([1, np.nan], ['a', 'a'])
        for ids in (['b', 'a'], ['a'], ['a', 'a']):
            with self.assertRaises(ValueError):
                assert_same_ids(['a', 'b'], ids, 'fixture')


if __name__ == '__main__':
    unittest.main()

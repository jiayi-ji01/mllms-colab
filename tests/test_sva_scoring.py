import unittest

import torch

from mllms.evaluation.sva.scoring import final_logit_difference, summarize_scores


class SVAScoringTest(unittest.TestCase):
    def test_logit_difference_and_corrupted_orientation(self):
        logits = torch.zeros(2, 3, 8)
        logits[0, -1, 2] = 4.0
        logits[0, -1, 5] = 1.0
        logits[1, -1, 2] = 1.0
        logits[1, -1, 5] = 6.0
        difference = final_logit_difference(
            logits,
            torch.tensor([2, 2]),
            torch.tensor([5, 5]),
        )
        self.assertTrue(torch.equal(difference, torch.tensor([3.0, -5.0])))

        rows = [
            {
                "original": {
                    "clean_ld": 3.0,
                    "corrupted_ld": -5.0,
                    "clean_correct": True,
                    "corrupted_correct": True,
                    "pair_correct": True,
                }
            }
        ]
        summary = summarize_scores(rows, "original")
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["mean_logit_difference"], 4.0)

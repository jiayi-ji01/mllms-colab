import unittest

import torch

from mllms.evaluation.sva.scoring import (
    final_logit_difference,
    sequence_log_probability_difference,
    summarize_scores,
)
from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT


class SVAScoringTest(unittest.TestCase):
    def test_sequence_score_supports_multi_token_answers(self):
        torch.manual_seed(3)
        model = GPT(
            GPTConfig(
                vocab_size=16,
                block_size=8,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                dropout=0.0,
            )
        ).eval()
        prompts = torch.tensor([[1, 2, 3], [1, 4, 3]])
        clean_answers = torch.tensor([[5, 6], [5, 6]])
        corrupted_answers = torch.tensor([[7, 8], [7, 8]])
        difference = sequence_log_probability_difference(
            model, prompts, clean_answers, corrupted_answers
        )
        self.assertEqual(difference.shape, (2,))
        self.assertTrue(torch.isfinite(difference).all())

    def test_single_token_sequence_score_matches_logit_difference(self):
        torch.manual_seed(5)
        model = GPT(
            GPTConfig(
                vocab_size=16,
                block_size=8,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                dropout=0.0,
            )
        ).eval()
        prompts = torch.tensor([[1, 2, 3], [1, 4, 3]])
        clean_answers = torch.tensor([[5], [5]])
        corrupted_answers = torch.tensor([[7], [7]])
        logits, _ = model(prompts)
        old = final_logit_difference(
            logits, clean_answers[:, 0], corrupted_answers[:, 0]
        )
        new = sequence_log_probability_difference(
            model, prompts, clean_answers, corrupted_answers
        )
        self.assertTrue(torch.allclose(old, new, atol=1e-6))

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

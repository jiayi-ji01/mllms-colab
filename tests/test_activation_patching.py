import unittest

import torch

from mllms.interpretability.activation_patching.interventions import patch_batch
from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT


class ActivationPatchingTest(unittest.TestCase):
    def test_final_residual_patch_recovers_clean_output(self):
        torch.manual_seed(7)
        model = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=8,
                d_model=16,
                n_heads=2,
                n_layers=2,
                d_ff=32,
                dropout=0.0,
            )
        ).eval()
        examples = [
            {
                "sample_id": "example",
                "clean_ids": [1, 2, 3, 4],
                "corrupted_ids": [1, 5, 3, 4],
                "clean_answer_id": 6,
                "corrupted_answer_id": 7,
            }
        ]
        result = patch_batch(
            model,
            examples,
            torch.device("cpu"),
            max_positions=None,
            intervention_batch_size=8,
        )[0]
        recovery = result["scores"]["recovery"]
        self.assertEqual(recovery["resid_post"].shape, (2, 4))
        self.assertEqual(recovery["head_out"].shape, (2, 4, 2))
        self.assertLess(abs(float(recovery["resid_post"][-1, -1]) - 1.0), 1e-4)

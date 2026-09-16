import unittest

import torch

from mllms.interpretability.activation_patching.interventions import patch_batch
from mllms.interpretability.activation_patching.runner import build_control_sources
from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT


class ActivationPatchingTest(unittest.TestCase):
    def test_opposite_number_control_uses_corrupted_source_activation(self):
        examples = [{
            "sample_id": "example",
            "task": "simple",
            "clean_type": "singular",
            "clean_ids": [1, 2, 3],
            "corrupted_ids": [1, 4, 3],
            "clean_answer_ids": [5],
            "corrupted_answer_ids": [6],
        }]
        controlled = build_control_sources(examples, "opposite-number")
        self.assertEqual(controlled[0]["clean_ids"], [1, 4, 3])
        self.assertEqual(controlled[0]["sample_id"], "example")
        self.assertEqual(controlled[0]["source_sample_id"], "example")

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

    def test_cross_language_patch_uses_source_clean_and_target_denominator(self):
        torch.manual_seed(11)
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
        with torch.no_grad():
            model.token_embedding.weight[16:].copy_(
                model.token_embedding.weight[:16]
            )
        source_examples = [
            {
                "sample_id": "example",
                "clean_ids": [1, 2, 3, 4],
                "corrupted_ids": [1, 5, 3, 4],
                "clean_answer_ids": [6],
                "corrupted_answer_ids": [7],
            }
        ]
        target_examples = [
            {
                "sample_id": "example",
                "clean_ids": [17, 18, 19, 20],
                "corrupted_ids": [17, 21, 19, 20],
                "clean_answer_ids": [22],
                "corrupted_answer_ids": [23],
            }
        ]
        result = patch_batch(
            model,
            source_examples,
            target_examples=target_examples,
            device=torch.device("cpu"),
            max_positions=None,
            intervention_batch_size=8,
        )[0]
        self.assertEqual(result["sample_id"], "example")
        self.assertAlmostEqual(
            result["target_clean_ld"] - result["target_corrupted_ld"],
            result["denominator"],
            places=6,
        )
        patched = result["scores"]["patched_ld"]["resid_post"][-1, -1]
        delta = result["scores"]["delta_ld"]["resid_post"][-1, -1]
        self.assertAlmostEqual(
            float(patched) - result["target_corrupted_ld"],
            float(delta),
            places=5,
        )
        self.assertLess(
            abs(float(result["scores"]["recovery"]["resid_post"][-1, -1]) - 1.0),
            1e-4,
        )

    def test_cross_language_patch_rejects_misaligned_examples(self):
        model = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=8,
                d_model=16,
                n_heads=2,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
            )
        ).eval()
        source = [{
            "sample_id": "source",
            "clean_ids": [1, 2, 3],
            "corrupted_ids": [1, 4, 3],
            "clean_answer_ids": [5],
            "corrupted_answer_ids": [6],
        }]
        target = [{
            "sample_id": "target",
            "clean_ids": [17, 18, 19],
            "corrupted_ids": [17, 20, 19],
            "clean_answer_ids": [21],
            "corrupted_answer_ids": [22],
        }]
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            patch_batch(
                model,
                source,
                target_examples=target,
                device=torch.device("cpu"),
                max_positions=None,
                intervention_batch_size=8,
            )

    def test_component_selection_limits_the_public_result_schema(self):
        model = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=8,
                d_model=16,
                n_heads=2,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
            )
        ).eval()
        examples = [{
            "sample_id": "example",
            "clean_ids": [1, 2, 3],
            "corrupted_ids": [1, 4, 3],
            "clean_answer_ids": [5],
            "corrupted_answer_ids": [6],
        }]
        result = patch_batch(
            model,
            examples,
            device=torch.device("cpu"),
            max_positions=1,
            intervention_batch_size=8,
            components=("head_out",),
        )[0]
        self.assertEqual(set(result["scores"]["recovery"]), {"head_out"})
        self.assertEqual(
            result["scores"]["recovery"]["head_out"].shape,
            (1, 1, 2),
        )

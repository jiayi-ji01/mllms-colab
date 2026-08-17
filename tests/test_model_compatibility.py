from dataclasses import asdict
import unittest

import torch

from mllms.model.config import GPTConfig
from mllms.model.loading import load_model_checkpoint
from mllms.model.transformer import GPT


class ModelCompatibilityTest(unittest.TestCase):
    def test_checkpoint_state_dict_schema_is_stable(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            dropout=0.0,
        )
        model = GPT(config)
        expected_keys = {
            "token_embedding.weight", "position_embedding.weight",
            "blocks.0.ln_1.weight", "blocks.0.attn.qkv.weight",
            "blocks.0.attn.out_proj.weight", "blocks.0.mlp.fc_in.weight",
            "blocks.0.mlp.fc_out.weight", "final_norm.weight", "lm_head.weight",
        }
        self.assertTrue(expected_keys <= set(model.state_dict()))

        with self.subTest("legacy checkpoint dictionary"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "legacy-schema.pt"
                torch.save(
                    {"model": model.state_dict(), "model_config": asdict(config), "step": 3},
                    path,
                )
                checkpoint, loaded, loaded_config = load_model_checkpoint(
                    path, torch.device("cpu")
                )
                self.assertEqual(checkpoint["step"], 3)
                self.assertEqual(loaded_config, config)
                self.assertEqual(list(loaded.state_dict()), list(model.state_dict()))

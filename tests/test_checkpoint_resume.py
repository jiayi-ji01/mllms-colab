from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT
from mllms.runtime import make_grad_scaler
from mllms.training.checkpoint import load_checkpoint, save_checkpoint
from mllms.training.config import TrainingConfig


class CheckpointResumeTest(unittest.TestCase):
    def test_checkpoint_contains_and_restores_complete_training_state(self):
        model_config = GPTConfig(
            vocab_size=32,
            block_size=8,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            dropout=0.0,
        )
        training_config = TrainingConfig(
            vocab_size=16,
            block_size=8,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            target_seen_tokens=None,
            max_steps=10,
            warmup_ratio=0.0,
            warmup_steps=1,
        )
        model = GPT(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = make_grad_scaler(False)
        generator = torch.Generator(device="cpu").manual_seed(42)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_000005.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                scaler,
                5,
                0.4,
                model_config,
                training_config,
                2.1,
                2.2,
                4096,
                2048,
                2048,
                1,
                generator,
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(raw["epoch"], 0.4)
            self.assertEqual(raw["validation_loss"], 2.2)
            self.assertEqual(raw["model_config"], asdict(model_config))
            self.assertEqual(raw["training_config"], asdict(training_config))
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

            restored = load_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                scaler,
                generator,
                model_config,
                torch.device("cpu"),
            )
            self.assertEqual(restored, (5, 2.1, 2.2, 4096, 2048, 2048, 1))


if __name__ == "__main__":
    unittest.main()

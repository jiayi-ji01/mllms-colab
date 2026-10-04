from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from model import GPTConfig
from model import GPT
from runtime import make_grad_scaler
from checkpoint import TrainingState, load_checkpoint, save_checkpoint
from config import TrainingConfig


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
            progress = TrainingState(
                step=5, best_validation_loss=2.1, validation_loss=2.2,
                tokens_seen=4096, original_tokens_seen=2048, clone_tokens_seen=2048,
                evaluations_without_improvement=1,
            )
            x = torch.tensor([[1, 2, 3, 4]])
            _, loss = model(x, x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            expected_weights = {name: value.clone() for name, value in model.state_dict().items()}
            save_checkpoint(
                path, model, optimizer, scheduler, scaler, progress,
                training_config, generator, epoch=0.4,
            )
            expected_random = (random.random(), np.random.random(),
                               torch.rand(3), torch.rand(3, generator=generator))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
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
                torch.device("cpu"),
            )
            self.assertEqual(restored, progress)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.random(), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
            torch.testing.assert_close(torch.rand(3, generator=generator),
                                       expected_random[3], rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
            self.assertNotIn("progress", raw)
            self.assertNotIn("state", raw)
            for name, value in asdict(progress).items():
                self.assertEqual(raw[name], value)

            # Older checkpoint dictionaries may omit the last validation loss.
            del raw["validation_loss"]
            torch.save(raw, path)
            restored = load_checkpoint(
                path, model, optimizer, scheduler, scaler, generator, torch.device("cpu"),
            )
            self.assertEqual(restored.validation_loss, float("inf"))
            self.assertEqual(restored.step, 5)
            raw["model_config"]["d_model"] = 8
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, "model_config"):
                load_checkpoint(
                    path, model, optimizer, scheduler, scaler, generator, torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()

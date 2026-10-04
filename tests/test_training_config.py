from dataclasses import asdict
from pathlib import Path
import unittest

from config import TrainingConfig, load_training_config


class TrainingConfigTest(unittest.TestCase):
    def test_experiment_configs_are_complete_and_relative(self):
        for path in sorted(Path("configs/experiments").glob("*.yaml")):
            config = load_training_config(path)
            self.assertEqual(config.n_layers, 12)
            self.assertFalse(Path(config.output_dir).is_absolute())
            self.assertEqual(config.tokens_per_step, 8192)

    def test_wikipedia_config_matches_the_recorded_experiment(self):
        config = load_training_config(
            Path("configs/experiments/gpt12_wikipedia_clone.yaml")
        )
        self.assertEqual(config.vocab_size, 16000)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.d_ff, 2048)
        self.assertEqual(config.target_epochs, 4.0)
        self.assertIsNone(config.target_seen_tokens)
        self.assertEqual(config.warmup_steps, 500)
        self.assertEqual(config.checkpoint_interval, 5000)
        self.assertEqual(config.output_dir, "outputs/runs/gpt12_wikipedia_clone")

    def test_model_config_uses_both_token_spaces_and_architecture_settings(self):
        training = TrainingConfig(
            vocab_size=17, block_size=8, d_model=24, n_heads=3,
            n_layers=2, d_ff=48, dropout=0.2, bias=False,
        )
        self.assertEqual(asdict(training.model_config()), {
            "vocab_size": 34, "block_size": 8, "d_model": 24, "n_heads": 3,
            "n_layers": 2, "d_ff": 48, "dropout": 0.2, "bias": False,
        })
        self.assertNotIn("model_config", asdict(training))

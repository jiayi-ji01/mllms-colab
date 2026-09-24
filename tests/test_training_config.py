from pathlib import Path
import unittest

from mllms.training.config import load_training_config


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

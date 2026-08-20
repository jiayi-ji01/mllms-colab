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

    def test_wikipedia_keeps_babylm_model_and_optimizer_settings(self):
        wikipedia = load_training_config(
            Path("configs/experiments/gpt12_wikipedia_clone.yaml")
        )
        babylm = load_training_config(
            Path("configs/experiments/gpt12_babylm_clone.yaml")
        )
        unchanged = (
            "vocab_size", "block_size", "d_model", "n_heads", "n_layers",
            "d_ff", "dropout", "bias", "micro_batch_size",
            "gradient_accumulation_steps", "learning_rate",
            "min_learning_rate", "weight_decay", "beta1", "beta2",
            "warmup_ratio", "warmup_steps", "grad_clip", "log_interval",
            "eval_interval", "eval_batches", "early_stopping_patience",
            "seed", "p_clone",
        )
        for name in unchanged:
            self.assertEqual(getattr(wikipedia, name), getattr(babylm, name), name)
        self.assertEqual(babylm.target_epochs, 2.0)
        self.assertEqual(wikipedia.target_epochs, 4.0)

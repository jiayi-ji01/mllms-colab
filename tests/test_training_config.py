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

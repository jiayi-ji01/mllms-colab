from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from mllms.evaluation.sva.prepare import parse_args as parse_sva_prepare
from mllms.evaluation.sva.controlled import parse_args as parse_controlled_sva
from mllms.interpretability.activation_patching.runner import (
    parse_args as parse_patching,
)


class StageConfigTest(unittest.TestCase):
    def test_sva_defaults_come_from_yaml(self):
        with patch.object(sys, "argv", ["prepare-sva"]):
            args = parse_sva_prepare()
        self.assertEqual(args.num_pairs, 1000)
        self.assertEqual(args.seed, 42)
        self.assertEqual(
            args.tokenizer,
            Path("artifacts/babylm_tokenizer/tokenizer.model"),
        )

    def test_cli_overrides_patching_yaml(self):
        with patch.object(
            sys,
            "argv",
            [
                "activation-patching",
                "--checkpoint",
                "checkpoint.pt",
                "--max-examples",
                "25",
            ],
        ):
            args = parse_patching()
        self.assertEqual(args.checkpoint, Path("checkpoint.pt"))
        self.assertEqual(args.max_examples, 25)
        self.assertEqual(args.intervention_batch_size, 64)

    def test_controlled_sva_defaults_come_from_yaml(self):
        with patch.object(sys, "argv", ["build-controlled-sva"]):
            args = parse_controlled_sva()
        self.assertEqual(args.test_pairs_per_condition, 800)
        self.assertEqual(args.dev_pairs_per_condition, 100)
        self.assertEqual(args.output_dir, Path("data/sva/controlled_v1"))

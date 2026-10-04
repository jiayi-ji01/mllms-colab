from dataclasses import asdict
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import sentencepiece as spm
import torch

from config import TrainingConfig
from evaluate_lm import evaluate_languages, parse_args
from model import GPT, GPTConfig
from token_data import ClonedMapper, TokenStream


class LanguageModelCliTest(unittest.TestCase):
    def test_accepts_held_out_test_split(self):
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_lm.py",
                "--checkpoint",
                "best.pt",
                "--split",
                "test",
                "--full-validation",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.checkpoint, Path("best.pt"))
        self.assertEqual(args.split, "test")
        self.assertTrue(args.full_validation)

    def test_full_evaluation_counts_partial_tail_for_both_languages(self):
        config = GPTConfig(vocab_size=32, block_size=4, d_model=8,
                           n_heads=2, n_layers=1, d_ff=16, dropout=0.0)
        model = GPT(config).eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        with tempfile.TemporaryDirectory() as directory:
            np.arange(1, 8, dtype=np.uint16).tofile(Path(directory) / "test.bin")
            rows = evaluate_languages(
                model, "trained", TokenStream(directory, "test"),
                ClonedMapper(16, pad_id=0), ["original", "clone"],
                2, torch.device("cpu"), "fp32", None, 42,
            )
        self.assertEqual([row["valid_next_token_positions"] for row in rows], [6, 6, 12])
        for row in rows:
            self.assertAlmostEqual(row["average_cross_entropy"], math.log(32), places=5)
            self.assertAlmostEqual(row["perplexity"], 32, places=4)

    def test_direct_script_writes_trained_metrics_without_debug_outputs(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_lm.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("the cat sits here\nthe dogs sit there\n" * 10)
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(root / "tokenizer"),
                vocab_size=32, model_type="bpe", pad_id=0, unk_id=1,
                bos_id=2, eos_id=3, minloglevel=2,
            )
            tokenizer_path = root / "tokenizer.model"
            tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            tokens = np.array(tokenizer.encode("the cat sits here"), dtype=np.uint16)
            tokens.tofile(root / "test.bin")
            training = TrainingConfig(
                vocab_size=32, block_size=4, d_model=8, n_heads=2,
                n_layers=1, d_ff=16, dropout=0.0, data_dir=str(root),
                tokenizer_path=str(tokenizer_path), micro_batch_size=2,
            )
            config = GPTConfig(vocab_size=64, block_size=4, d_model=8,
                               n_heads=2, n_layers=1, d_ff=16, dropout=0.0)
            checkpoint = root / "model.pt"
            torch.save({"model": GPT(config).state_dict(), "model_config": asdict(config),
                        "training_config": asdict(training), "step": 7}, checkpoint)
            environment = {**os.environ, "OMP_NUM_THREADS": "1"}
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, str(script), "--checkpoint", str(checkpoint),
                 "--split", "test", "--full-validation", "--device", "cpu",
                 "--output-dir", str(root / "evaluation")],
                cwd=root, env=environment, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / "evaluation"
            with (output / "test_loss_perplexity.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["language"] for row in rows], ["original", "clone", "average"])
            self.assertEqual({row["model"] for row in rows}, {"trained"})
            self.assertEqual(int(rows[0]["valid_next_token_positions"]), len(tokens) - 1)
            summary = json.loads((output / "sanity_check_summary.json").read_text())
            self.assertEqual(summary["checkpoint_step"], 7)
            self.assertEqual(summary["evaluation_mode"], "full")
            self.assertNotIn("trained_minus_random_loss", summary)
            self.assertFalse((output / "next_token_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()

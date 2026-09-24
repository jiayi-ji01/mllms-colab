from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

from mllms.cli import COMMANDS, main
from mllms.reporting.evidence import (
    STEPS,
    _assert_hash,
    _assert_provenance,
    _portable_candidates,
    _portable_value,
)
from mllms.training.cli import parse_args as parse_training_args


class PublicInterfaceTest(unittest.TestCase):
    def test_cli_exposes_only_the_current_pipeline(self):
        self.assertEqual(
            set(COMMANDS),
            {
                ("data", "prepare-wikipedia"),
                ("data", "tokenize"),
                ("tokenizer", "train"),
                ("train",),
                ("analyze", "build-controlled-sva"),
                ("analyze", "evaluate-sva"),
                ("analyze", "activation-patching"),
                ("analyze", "patching-statistics"),
                ("analyze", "sanity-check"),
                ("plot",),
            },
        )
        output = StringIO()
        with patch.object(sys, "argv", ["mllms", "--help"]), redirect_stdout(output):
            main()
        help_text = output.getvalue()
        self.assertIn("prepare-wikipedia", help_text)
        retired_commands = ("baby" + "lm", "b" + "limp", "tiny" + "stories")
        for command in retired_commands:
            self.assertNotIn(command, help_text.lower())

    def test_training_requires_an_explicit_config(self):
        with patch.object(sys, "argv", ["train"]), self.assertRaises(SystemExit):
            parse_training_args()

    def test_provenance_mismatches_fail(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            _assert_hash(Path(__file__), "not-a-hash", "fixture")
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            _assert_provenance(
                {"checkpoint_sha256": "expected"},
                {"checkpoint_sha256": "actual"},
                ("checkpoint_sha256",),
                Path("fixture"),
            )

    def test_portable_metadata_removes_machine_and_commit_identifiers(self):
        value = _portable_value(
            {"output_dir": "/workspace/account/project/outputs/run"}
        )
        self.assertEqual(value["output_dir"], "outputs/run")
        candidates = _portable_candidates(
            {"status": "frozen_before_test", "selection_code_commit": "abc"}
        )
        self.assertNotIn("selection_code_commit", candidates)
        self.assertIn("selection_provenance", candidates)

    def test_tracked_manifest_is_machine_portable(self):
        report_dir = Path("reports/cross_language_sva/data")
        for path in (report_dir / "manifest.json", report_dir / "candidate_manifest.json"):
            text = path.read_text()
            self.assertNotIn("/workspace/", text)
            self.assertNotIn("/home/", text)
            self.assertNotIn("selection_code_commit", text)
            self.assertNotRegex(text, r"\b(?:job|array)[ _-]?\d+\b")
            self.assertIsInstance(json.loads(text), dict)

        manifest = json.loads((report_dir / "manifest.json").read_text())
        checkpoint_order = [
            (item["checkpoint"], item["step"])
            for item in manifest["checkpoints"]
        ]
        self.assertEqual(checkpoint_order, list(STEPS))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

from train import parse_args as parse_training_args


class PublicInterfaceTest(unittest.TestCase):
    def test_training_requires_an_explicit_config(self):
        with patch.object(sys, "argv", ["train"]), self.assertRaises(SystemExit):
            parse_training_args()

    def test_tracked_manifest_is_machine_portable(self):
        evidence_dir = Path("evidence/cross_language_sva/data")
        for path in (evidence_dir / "manifest.json", evidence_dir / "candidate_manifest.json"):
            text = path.read_text()
            self.assertNotIn("/workspace/", text)
            self.assertNotIn("/home/", text)
            self.assertNotIn("selection_code_commit", text)
            self.assertNotRegex(text, r"\b(?:job|array)[ _-]?\d+\b")
            self.assertIsInstance(json.loads(text), dict)

        manifest = json.loads((evidence_dir / "manifest.json").read_text())
        checkpoint_order = [
            (item["checkpoint"], item["step"])
            for item in manifest["checkpoints"]
        ]
        self.assertEqual(checkpoint_order, [
            ("step_005000", 5000), ("step_015000", 15000),
            ("step_030000", 30000), ("step_050000", 50000),
            ("step_065000", 65000), ("best", 77500),
        ])


if __name__ == "__main__":
    unittest.main()

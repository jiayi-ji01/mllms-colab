from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from mllms.evaluation.sanity import parse_args


class SanityCliTest(unittest.TestCase):
    def test_accepts_held_out_test_split(self):
        with patch.object(
            sys,
            "argv",
            [
                "sanity-check",
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


if __name__ == "__main__":
    unittest.main()

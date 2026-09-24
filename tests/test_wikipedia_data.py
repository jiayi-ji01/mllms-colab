from pathlib import Path
import tempfile
import unittest

from mllms.data.wikipedia import choose_split, normalize_article, prepare_rows


class WikipediaDataTest(unittest.TestCase):
    def test_normalization_and_split_are_deterministic(self):
        self.assertEqual(normalize_article("A\n  short\tarticle"), "A short article")
        targets = {"train": 80, "validation": 10, "test": 10}
        self.assertEqual(
            choose_split("article-17", 42, targets),
            choose_split("article-17", 42, targets),
        )

    def test_prepare_rows_writes_disjoint_complete_splits(self):
        rows = (
            {"id": f"article-{index}", "text": "one two three four five"}
            for index in range(10_000)
        )
        targets = {"train": 20, "validation": 10, "test": 10}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            statistics = prepare_rows(rows, output_dir, targets, seed=42)
            for split, target in targets.items():
                self.assertGreaterEqual(statistics[split]["actual_words"], target)
                self.assertTrue((output_dir / f"{split}.txt").is_file())
                self.assertFalse((output_dir / f"{split}.txt.partial").exists())

            with self.assertRaises(FileExistsError):
                prepare_rows([], output_dir, targets, seed=42)


if __name__ == "__main__":
    unittest.main()

import unittest

from mllms.evaluation.sva.controlled import generate_split, make_record


class FakeTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, out_type=int):
        del out_type
        ids = []
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary) + 2
            ids.append(self.vocabulary[word])
        return ids

    @staticmethod
    def eos_id():
        return 1


class ControlledSVATest(unittest.TestCase):
    def test_counterfactual_changes_only_subject_number(self):
        tokenizer = FakeTokenizer()
        record = make_record(
            tokenizer,
            sample_id="test:pp:0000",
            split="test",
            condition="pp_attractor",
            clean_number="singular",
            attractor_number="plural",
            subject_lemma="author",
            attractor_lemma="guard",
            main_verb_lemma="walk",
            modifier="young",
            relation="near",
        )
        self.assertEqual(record["clean_prompt"], "The young author near the guards")
        self.assertEqual(
            record["corrupted_prompt"],
            "The young authors near the guards",
        )
        self.assertEqual(record["clean_answer"], "walks")
        self.assertEqual(record["corrupted_answer"], "walk")
        self.assertEqual(record["clean_attractor_relation"], "mismatched")
        self.assertEqual(
            len(record["clean_input_ids"]),
            len(record["corrupted_input_ids"]),
        )
        self.assertEqual(
            record["prediction_position"],
            len(record["clean_input_ids"]) - 1,
        )

    def test_generation_is_balanced_and_duplicate_free(self):
        tokenizer = FakeTokenizer()
        records = generate_split(
            tokenizer,
            split="test",
            lexicon={
                "nouns": ["author", "doctor", "guard", "teacher"],
                "main_verbs": ["walk", "run"],
                "past_verbs": ["called", "helped"],
            },
            pairs_per_condition=8,
            rng=__import__("random").Random(7),
        )
        self.assertEqual(len(records), 32)
        self.assertEqual(len({row["sample_id"] for row in records}), 32)
        for task in {row["task"] for row in records}:
            group = [row for row in records if row["task"] == task]
            self.assertEqual(len(group), 8)
            self.assertEqual(
                sum(row["clean_type"] == "singular" for row in group),
                4,
            )


if __name__ == "__main__":
    unittest.main()

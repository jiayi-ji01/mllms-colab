import unittest

import torch

from mllms.data.cloned_language import ClonedMapper


class ClonedLanguageTest(unittest.TestCase):
    def test_clone_mapping_offsets_non_pad_tokens_only(self):
        mapper = ClonedMapper(original_vocab_size=8, pad_id=0)
        ids = torch.tensor([[0, 1, 7], [2, 0, 3]])
        original = mapper.map_to_language(ids, 0)
        clone = mapper.map_to_language(ids, 1)
        self.assertTrue(torch.equal(original, ids))
        self.assertTrue(torch.equal(clone, torch.tensor([[0, 9, 15], [10, 0, 11]])))
        self.assertTrue(torch.equal(mapper.recover(clone), ids))

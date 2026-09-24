"""Memory-mapped next-token dataset and batch construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mllms.data.cloned_language import ClonedMapper


class TokenStream:
    """Sample fixed-length batches from a flat uint16 token stream."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        self.path = Path(data_dir) / f"{split}.bin"
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing token file: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

    def get_batch(
        self,
        *,
        batch_size: int,
        block_size: int,
        device: torch.device | str,
        cloned_mapper: ClonedMapper,
        language: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample windows, map one language per sequence, and create x/y."""
        if batch_size <= 0 or block_size <= 0:
            raise ValueError("batch_size and block_size must be positive")

        max_start = len(self.tokens) - block_size - 1
        if max_start < 0:
            raise ValueError(f"{self.path} is too short for block_size={block_size}")

        starts = torch.randint(
            max_start + 1,
            (batch_size,),
            generator=generator,
        ).tolist()
        windows = np.stack(
            [
                np.array(
                    self.tokens[start : start + block_size + 1],
                    dtype=np.int64,
                    copy=True,
                )
                for start in starts
            ]
        )
        sequence = torch.from_numpy(windows)

        if language is None:
            sequence, language_ids = cloned_mapper.random_map(
                sequence,
                generator=generator,
            )
        else:
            sequence = cloned_mapper.map_to_language(sequence, language)
            language_ids = torch.full((batch_size,), language, dtype=torch.long)

        # Mapping before the shift keeps every input/target pair in one language.
        return (
            sequence[:, :-1].to(device),
            sequence[:, 1:].to(device),
            language_ids.to(device),
        )

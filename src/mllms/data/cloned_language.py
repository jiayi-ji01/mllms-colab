"""Original-to-cloned token-ID mapping."""

# Clone-language mapping adapted from Schäfer et al. (2024).
# The implementation was extracted and simplified for this project.

import torch


class ClonedMapper:
    """Map SentencePiece IDs into original or cloned vocabulary space."""

    def __init__(
        self,
        original_vocab_size: int,
        p_clone: float = 0.5,
        pad_id: int = 0,
        generator: torch.Generator | None = None,
    ) -> None:
        if original_vocab_size <= 0:
            raise ValueError("original_vocab_size must be positive")
        if not 0 <= pad_id < original_vocab_size:
            raise ValueError("pad_id must be inside the original vocabulary")
        if not 0.0 <= p_clone <= 1.0:
            raise ValueError("p_clone must be between 0 and 1")

        self.original_vocab_size = original_vocab_size
        self.model_vocab_size = 2 * original_vocab_size
        self.p_clone = p_clone
        self.pad_id = pad_id
        self.generator = generator

    def sample_language(
        self,
        batch_size: int | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> int | torch.Tensor:
        """Sample one language per sequence."""
        return self._sample(batch_size, device, self.generator)

    def map_to_language(
        self,
        token_ids: torch.Tensor,
        language_id: int | torch.Tensor,
    ) -> torch.Tensor:
        """Map base IDs to language 0 (original) or 1 (clone)."""
        ids = self._check_ids(token_ids, self.original_vocab_size)
        languages = self._language_tensor(language_id, ids)
        offsets = languages * self.original_vocab_size
        return torch.where(ids == self.pad_id, ids, ids + offsets)

    def random_map(
        self,
        token_ids: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int | torch.Tensor]:
        """Sample languages and map a sequence or batch."""
        if token_ids.ndim not in (1, 2):
            raise ValueError("token_ids must have shape [T] or [B, T]")

        batch_size = token_ids.size(0) if token_ids.ndim == 2 else None
        languages = self._sample(
            batch_size,
            token_ids.device,
            generator if generator is not None else self.generator,
        )
        return self.map_to_language(token_ids, languages), languages

    def recover(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Recover original SentencePiece IDs from mapped IDs."""
        ids = self._check_ids(token_ids, self.model_vocab_size)
        unused_id = self.original_vocab_size + self.pad_id
        if (ids == unused_id).any().item():
            raise ValueError(f"token ID {unused_id} is the unused cloned-PAD ID")
        return ids % self.original_vocab_size

    def _sample(
        self,
        batch_size: int | None,
        device: torch.device | str | None,
        generator: torch.Generator | None,
    ) -> int | torch.Tensor:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        shape = () if batch_size is None else (batch_size,)
        sample_device = generator.device if generator is not None else "cpu"
        languages = (
            torch.rand(shape, generator=generator, device=sample_device)
            < self.p_clone
        ).long()

        if batch_size is None:
            return int(languages.item())
        return languages.to(device or "cpu")

    def _language_tensor(
        self,
        language_id: int | torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(language_id, int):
            languages = torch.tensor(language_id, device=token_ids.device)
        elif isinstance(language_id, torch.Tensor):
            if language_id.is_floating_point() or language_id.dtype == torch.bool:
                raise TypeError("language_id must use an integer dtype")
            languages = language_id.to(token_ids.device, dtype=torch.long)
        else:
            raise TypeError("language_id must be an int or torch.Tensor")

        if token_ids.ndim == 1:
            if languages.numel() != 1:
                raise ValueError("one sequence requires one language_id")
            languages = languages.reshape(())
        elif languages.numel() == 1:
            languages = languages.reshape(())
        elif languages.shape == (token_ids.size(0),):
            languages = languages[:, None]
        else:
            raise ValueError("batch language_ids must have shape [batch_size]")

        if ((languages < 0) | (languages > 1)).any().item():
            raise ValueError("language_id must be 0 (original) or 1 (clone)")
        return languages.long()

    @staticmethod
    def _check_ids(token_ids: torch.Tensor, upper_bound: int) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim not in (1, 2) or token_ids.numel() == 0:
            raise ValueError("token_ids must have non-empty shape [T] or [B, T]")
        if token_ids.is_floating_point() or token_ids.dtype == torch.bool:
            raise TypeError("token_ids must use an integer dtype")

        ids = token_ids.long()
        if ((ids < 0) | (ids >= upper_bound)).any().item():
            raise ValueError(f"token IDs must satisfy 0 <= id < {upper_bound}")
        return ids

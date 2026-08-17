from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from mllms.model.components import HookPoint, TransformerBlock
from mllms.model.config import GPTConfig


class GPT(nn.Module):
    """Decoder-only Transformer for next-token prediction."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def activation_points(self) -> dict[str, HookPoint]:
        """Return the named activation sites exposed by every layer."""
        points: dict[str, HookPoint] = {}
        for layer, block in enumerate(self.blocks):
            prefix = f"blocks.{layer}"
            points[f"{prefix}.resid_post"] = block.hook_resid_post
            points[f"{prefix}.attn_out"] = block.hook_attn_out
            points[f"{prefix}.mlp_out"] = block.hook_mlp_out
            points[f"{prefix}.head_out"] = block.attn.hook_head_out
        return points
    @contextmanager
    def hooks(
        self,
        hook_functions: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    ) -> Iterator[None]:
        points = self.activation_points()
        unknown = set(hook_functions) - set(points)
        if unknown:
            raise KeyError(f"Unknown activation points: {sorted(unknown)}")

        handles = []
        for name, function in hook_functions.items():
            def forward_hook(
                _module: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                function: Callable[[torch.Tensor], torch.Tensor] = function,
            ) -> torch.Tensor:
                return function(output)

            handles.append(points[name].register_forward_hook(forward_hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names: set[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        selected = names or set(self.activation_points())
        cache: dict[str, torch.Tensor] = {}

        def save(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
            def save_activation(activation: torch.Tensor) -> torch.Tensor:
                cache[name] = activation.detach().clone()
                return activation

            return save_activation

        with self.hooks({name: save(name) for name in selected}):
            logits, _ = self(input_ids)
        return logits, cache

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token logits and optional cross-entropy loss."""
        self._validate_ids(input_ids, "input_ids")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"block_size {self.config.block_size}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            self._validate_ids(targets, "targets")
            if targets.shape != (batch_size, sequence_length):
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, self.config.vocab_size)
                threshold = torch.topk(next_logits, k).values[:, [-1]]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold,
                    float("-inf"),
                )

            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    def _validate_ids(self, token_ids: torch.Tensor, name: str) -> None:
        if token_ids.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, sequence]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use torch.int32 or torch.int64")
        if token_ids.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if ((token_ids < 0) | (token_ids >= self.config.vocab_size)).any().item():
            raise ValueError(
                f"{name} must satisfy 0 <= id < {self.config.vocab_size}"
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

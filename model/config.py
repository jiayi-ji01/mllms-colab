"""Model configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for the decoder-only GPT model."""

    vocab_size: int
    block_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

"""Named Transformer components exposed to interpretability experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mllms.model.config import GPTConfig


class HookPoint(nn.Module):
    """Identity module that provides a stable activation-hook location."""

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return activation


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with exposed per-head outputs."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )
        self.residual_dropout = nn.Dropout(config.dropout)
        self.hook_head_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = self.hook_head_out(attention)
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            d_model,
        )
        return self.residual_dropout(self.out_proj(attention))


class FeedForward(nn.Module):
    """GPT MLP with GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.activation(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block with stable attention, MLP, and residual hooks."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.hook_attn_out(self.attn(self.ln_1(x)))
        x = x + attention_output
        mlp_output = self.hook_mlp_out(self.mlp(self.ln_2(x)))
        return self.hook_resid_post(x + mlp_output)

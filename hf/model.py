"""Standalone ChudGPT-Public architecture for Hugging Face downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 1024
    embedding_dim: int = 384
    num_layers: int = 9
    num_heads: int = 6
    feed_forward_dim: int = 1808
    dropout: float = 0.1
    rope_base: float = 10000.0


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, length: int, base: float) -> None:
        super().__init__()
        inverse = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        frequencies = torch.outer(torch.arange(length).float(), inverse)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.size(-2)
        cosine = self.cos[:, :, :length].to(dtype=query.dtype)
        sine = self.sin[:, :, :length].to(dtype=query.dtype)
        return query * cosine + rotate_half(query) * sine, key * cosine + rotate_half(key) * sine


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.qkv = nn.Linear(config.embedding_dim, config.embedding_dim * 3)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.context_length, config.rope_base)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        reshape = lambda tensor: tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        query, key, value = reshape(query), reshape(key), reshape(value)
        query, key = self.rope(query, key)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.projection(attended.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = Attention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim), nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim), nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class ChudGPTPublic(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.token_embedding(token_ids))
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: str = "cpu") -> "ChudGPTPublic":
        from safetensors.torch import load_file

        directory = Path(model_dir)
        values = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config = ModelConfig(**values["model_config"])
        model = cls(config).to(device)
        model.load_state_dict(load_file(directory / "model.safetensors", device=device))
        model.eval()
        return model

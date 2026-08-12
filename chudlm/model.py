from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    embedding_dim: int = 288
    num_layers: int = 12
    num_heads: int = 6
    feed_forward_dim: int = 1104
    dropout: float = 0.1
    rope_base: float = 10000.0

    def validate(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if (self.embedding_dim // self.num_heads) % 2 != 0:
            raise ValueError("Each attention head dimension must be even for RoPE")
        if min(self.vocab_size, self.context_length, self.num_layers) <= 0:
            raise ValueError("Vocabulary, context length, and layer count must be positive")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_length).float()
        frequencies = torch.outer(positions, inverse_frequency)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        cos = self.cos[:, :, :length].to(dtype=q.dtype)
        sin = self.sin[:, :, :length].to(dtype=q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.context_length, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda value: value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        q, k = self.rope(q, k)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.projection(output.transpose(1, 2).contiguous().view(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.size(1) > self.config.context_length:
            raise ValueError(f"Sequence length exceeds context length {self.config.context_length}")
        hidden = self.dropout(self.token_embedding(token_ids))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

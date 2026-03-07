#!/usr/bin/env python3
"""Per-position temporal model for commaVQ compression.

Predicts each token from K previous tokens at the SAME spatial position.
Shared weights across all 128 positions, with a position embedding.

Key property: encoder and decoder feed identical integer inputs,
so probabilities match exactly. No float mismatch possible.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalConfig:
    vocab_size: int = 1024       # VQ tokens 0-1023
    n_positions: int = 128       # spatial positions (8x16 grid)
    context_len: int = 20        # K previous frames to condition on
    dim: int = 128               # model dimension
    n_layer: int = 4             # transformer layers
    n_head: int = 4              # attention heads
    intermediate_size: int = 512  # MLP hidden dim
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.dim // self.n_head


class TemporalAttention(nn.Module):
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=True)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=True)

    def forward(self, x, mask):
        bsz, seqlen, _ = x.shape
        q, k, v = self.c_attn(x).split(self.config.dim, dim=-1)
        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)
        return self.c_proj(y)


class TemporalMLP(nn.Module):
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class TemporalBlock(nn.Module):
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = TemporalAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = TemporalMLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModel(nn.Module):
    """Predicts next token from K previous tokens at same spatial position.

    Input: (batch, K) int tokens — temporal history at one position
    Output: (batch, K, vocab_size) logits — next-token predictions

    The model is autoregressive over the temporal dimension:
    logits[:, t, :] predicts the token at time t+1 given tokens 0..t.

    For compression, we use logits[:, -1, :] to predict the current frame's token
    given the previous K frames.
    """

    def __init__(self, config: TemporalConfig = TemporalConfig()):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_emb = nn.Embedding(config.context_len, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)
        self.layers = nn.ModuleList([TemporalBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Causal mask: each position can only see itself and earlier positions
        mask = torch.tril(torch.ones(config.context_len, config.context_len, dtype=torch.bool))
        self.register_buffer('causal_mask', mask.view(1, 1, config.context_len, config.context_len))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, positions=None):
        """
        tokens: (batch, seq_len) — token IDs (0..1023)
        positions: (batch,) — spatial position IDs (0..127), or None
        Returns: (batch, seq_len, vocab_size) logits
        """
        bsz, seq_len = tokens.shape
        assert seq_len <= self.config.context_len

        time_ids = torch.arange(seq_len, device=tokens.device)
        x = self.token_emb(tokens) + self.time_emb(time_ids)

        if positions is not None:
            # Add spatial position embedding (broadcast across time)
            x = x + self.pos_emb(positions).unsqueeze(1)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Model size presets
CONFIGS = {
    "tiny": TemporalConfig(dim=64, n_layer=2, n_head=2, intermediate_size=256),
    "small": TemporalConfig(dim=128, n_layer=4, n_head=4, intermediate_size=512),
    "medium": TemporalConfig(dim=256, n_layer=6, n_head=4, intermediate_size=1024),
    "large": TemporalConfig(dim=384, n_layer=8, n_head=6, intermediate_size=1536),
}

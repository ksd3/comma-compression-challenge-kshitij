#!/usr/bin/env python3
"""Per-position temporal model v2: includes spatial context from previous frames.

For each position p at frame t, the model sees:
  - K previous tokens at position p (temporal history)
  - K previous tokens at N spatial neighbors (spatial-temporal context)

Both encoder and decoder know previous frames completely, so inputs are identical.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalV2Config:
    vocab_size: int = 1024
    n_positions: int = 128       # spatial positions (8x16 grid)
    context_len: int = 20        # K previous frames
    n_neighbors: int = 4         # number of spatial neighbors (left, right, above, below)
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.dim // self.n_head

    @property
    def n_channels(self):
        return 1 + self.n_neighbors  # center + neighbors


# Precompute neighbor indices for 8x16 grid
def build_neighbor_map(rows=8, cols=16, n_neighbors=4):
    """Returns neighbor_idx: (128, n_neighbors) array of neighbor indices.
    4 neighbors: left, right, above, below.
    8 neighbors: + NW, NE, SW, SE.
    Clamp at borders.
    """
    import numpy as np
    neighbor_idx = np.zeros((rows * cols, n_neighbors), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            pos = r * cols + c
            left = r * cols + max(0, c - 1)
            right = r * cols + min(cols - 1, c + 1)
            above = max(0, r - 1) * cols + c
            below = min(rows - 1, r + 1) * cols + c
            if n_neighbors == 4:
                neighbor_idx[pos] = [left, right, above, below]
            elif n_neighbors == 8:
                nw = max(0, r - 1) * cols + max(0, c - 1)
                ne = max(0, r - 1) * cols + min(cols - 1, c + 1)
                sw = min(rows - 1, r + 1) * cols + max(0, c - 1)
                se = min(rows - 1, r + 1) * cols + min(cols - 1, c + 1)
                neighbor_idx[pos] = [left, right, above, below, nw, ne, sw, se]
    return neighbor_idx


NEIGHBOR_MAP_4 = build_neighbor_map(n_neighbors=4)
NEIGHBOR_MAP_8 = build_neighbor_map(n_neighbors=8)
NEIGHBOR_MAP = NEIGHBOR_MAP_4  # default


class TemporalV2Attention(nn.Module):
    def __init__(self, config: TemporalV2Config):
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


class TemporalV2MLP(nn.Module):
    def __init__(self, config: TemporalV2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class TemporalV2Block(nn.Module):
    def __init__(self, config: TemporalV2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = TemporalV2Attention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = TemporalV2MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelV2(nn.Module):
    """Temporal model with spatial neighbor context.

    Input per timestep: (center_token, left, right, above, below) = 5 tokens
    These are embedded separately and summed to create a single representation.

    Input shape: (batch, K, 1+n_neighbors) — K timesteps, each with center + neighbors
    Output shape: (batch, K, vocab_size) — next-token predictions
    """

    def __init__(self, config: TemporalV2Config = TemporalV2Config()):
        super().__init__()
        self.config = config

        # Separate embedding for center vs each neighbor type
        self.center_emb = nn.Embedding(config.vocab_size, config.dim)
        self.neighbor_emb = nn.Embedding(config.vocab_size, config.dim)
        self.neighbor_weight = nn.Parameter(torch.ones(config.n_neighbors) / config.n_neighbors)

        self.time_emb = nn.Embedding(config.context_len, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)

        self.layers = nn.ModuleList([TemporalV2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

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
        tokens: (batch, seq_len, 1+n_neighbors) — center + neighbor tokens per timestep
        positions: (batch,) — spatial position IDs (0..127), or None
        Returns: (batch, seq_len, vocab_size) logits
        """
        bsz, seq_len, n_ch = tokens.shape
        assert seq_len <= self.config.context_len

        # Embed center token
        center = self.center_emb(tokens[:, :, 0])  # (batch, K, dim)

        # Embed and aggregate neighbor tokens
        if n_ch > 1:
            neighbor_tokens = tokens[:, :, 1:]  # (batch, K, n_neighbors)
            neighbor_embeds = self.neighbor_emb(neighbor_tokens)  # (batch, K, n_neighbors, dim)
            w = torch.softmax(self.neighbor_weight, dim=0).view(1, 1, -1, 1)
            neighbors = (neighbor_embeds * w).sum(dim=2)  # (batch, K, dim)
            x = center + neighbors
        else:
            x = center

        time_ids = torch.arange(seq_len, device=tokens.device)
        x = x + self.time_emb(time_ids)

        if positions is not None:
            x = x + self.pos_emb(positions).unsqueeze(1)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CONFIGS_V2 = {
    "tiny": TemporalV2Config(dim=64, n_layer=2, n_head=2, intermediate_size=256),
    "small": TemporalV2Config(dim=128, n_layer=4, n_head=4, intermediate_size=512),
    "small8": TemporalV2Config(dim=128, n_layer=4, n_head=4, intermediate_size=512, n_neighbors=8),
    "medium": TemporalV2Config(dim=256, n_layer=6, n_head=4, intermediate_size=1024),
    "medium8": TemporalV2Config(dim=256, n_layer=6, n_head=4, intermediate_size=1024, n_neighbors=8),
    "large": TemporalV2Config(dim=384, n_layer=8, n_head=6, intermediate_size=1536),
}

#!/usr/bin/env python3
"""Per-position temporal model v4: 4 raster-order spatial neighbors from current frame.

Extends v3 by conditioning on 4 already-decoded neighbors from current frame:
  - left (same row, col-1)
  - above (row-1, same col)
  - above-left (row-1, col-1)
  - above-right (row-1, col+1)

All 4 are available during raster-order decoding (left-to-right, top-to-bottom).
Uses separate embedding per neighbor slot + learned weighting.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalV4Config:
    vocab_size: int = 1024
    n_positions: int = 128       # spatial positions (8x16 grid)
    context_len: int = 20        # K previous frames
    n_temporal_neighbors: int = 4  # temporal neighbors from prev frames
    n_spatial_neighbors: int = 3   # current-frame above-row neighbors
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.dim // self.n_head


def build_temporal_neighbor_map(rows=8, cols=16):
    """4 neighbors: left, right, above, below from previous frames."""
    n = rows * cols
    neighbor_idx = np.zeros((n, 4), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            pos = r * cols + c
            left = r * cols + max(0, c - 1)
            right = r * cols + min(cols - 1, c + 1)
            above = max(0, r - 1) * cols + c
            below = min(rows - 1, r + 1) * cols + c
            neighbor_idx[pos] = [left, right, above, below]
    return neighbor_idx


def build_spatial_neighbor_map(rows=8, cols=16):
    """3 above-row current-frame neighbors: above, above-left, above-right.
    All from previous row — available when batching entire row.
    Returns (128, 3) array. -1 means not available."""
    n = rows * cols
    neighbor_idx = np.full((n, 3), -1, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            pos = r * cols + c
            if r > 0:
                neighbor_idx[pos, 0] = (r - 1) * cols + c       # above
            if r > 0 and c > 0:
                neighbor_idx[pos, 1] = (r - 1) * cols + (c - 1) # above-left
            if r > 0 and c < cols - 1:
                neighbor_idx[pos, 2] = (r - 1) * cols + (c + 1) # above-right
    return neighbor_idx


TEMPORAL_NEIGHBOR_MAP = build_temporal_neighbor_map()
SPATIAL_NEIGHBOR_MAP = build_spatial_neighbor_map()


class V4Attention(nn.Module):
    def __init__(self, config: TemporalV4Config):
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


class V4MLP(nn.Module):
    def __init__(self, config: TemporalV4Config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class V4Block(nn.Module):
    def __init__(self, config: TemporalV4Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = V4Attention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = V4MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelV4(nn.Module):
    """Temporal model v4: temporal context + 4 raster spatial neighbors.

    Input per timestep: (center_token, temp_neighbor1..4) from prev frames
    Additional input: 4 spatial neighbors from current frame (raster-available)

    Each spatial neighbor gets its own embedding. Spatial embeddings are
    weighted-summed and added as bias to all timestep representations.
    """

    def __init__(self, config: TemporalV4Config = TemporalV4Config()):
        super().__init__()
        self.config = config

        # Temporal embeddings
        self.center_emb = nn.Embedding(config.vocab_size, config.dim)
        self.temporal_neighbor_emb = nn.Embedding(config.vocab_size, config.dim)
        self.temporal_neighbor_weight = nn.Parameter(
            torch.ones(config.n_temporal_neighbors) / config.n_temporal_neighbors
        )

        # Spatial current-frame embeddings (one embedding table per slot + shared)
        # +1 for "not available" sentinel
        self.spatial_emb = nn.Embedding(config.vocab_size + 1, config.dim)
        self.spatial_weight = nn.Parameter(
            torch.ones(config.n_spatial_neighbors) / config.n_spatial_neighbors
        )

        self.time_emb = nn.Embedding(config.context_len, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)

        self.layers = nn.ModuleList([V4Block(config) for _ in range(config.n_layer)])
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

    def forward(self, temporal_tokens, spatial_tokens, positions=None):
        """
        temporal_tokens: (batch, seq_len, 1+n_temporal_neighbors)
        spatial_tokens: (batch, n_spatial_neighbors) — use vocab_size for unavailable
        positions: (batch,) — spatial position IDs
        Returns: (batch, seq_len, vocab_size) logits
        """
        bsz, seq_len, n_ch = temporal_tokens.shape

        # Embed temporal center
        center = self.center_emb(temporal_tokens[:, :, 0])

        # Embed and aggregate temporal neighbors
        if n_ch > 1:
            neighbor_tokens = temporal_tokens[:, :, 1:]
            neighbor_embeds = self.temporal_neighbor_emb(neighbor_tokens)
            w = torch.softmax(self.temporal_neighbor_weight, dim=0).view(1, 1, -1, 1)
            neighbors = (neighbor_embeds * w).sum(dim=2)
            x = center + neighbors
        else:
            x = center

        # Add time embeddings
        time_ids = torch.arange(seq_len, device=temporal_tokens.device)
        x = x + self.time_emb(time_ids)

        # Add position embeddings
        if positions is not None:
            x = x + self.pos_emb(positions).unsqueeze(1)

        # Add spatial context (broadcast to all timesteps)
        spatial_embeds = self.spatial_emb(spatial_tokens)  # (batch, n_spatial, dim)
        sw = torch.softmax(self.spatial_weight, dim=0).view(1, -1, 1)
        spatial_ctx = (spatial_embeds * sw).sum(dim=1)  # (batch, dim)
        x = x + spatial_ctx.unsqueeze(1)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CONFIGS_V4 = {
    "tiny": TemporalV4Config(dim=64, n_layer=2, n_head=2, intermediate_size=256),
    "small": TemporalV4Config(dim=128, n_layer=4, n_head=4, intermediate_size=512),
    "medium": TemporalV4Config(dim=192, n_layer=6, n_head=4, intermediate_size=768),
    "large": TemporalV4Config(dim=256, n_layer=8, n_head=8, intermediate_size=1024),
}

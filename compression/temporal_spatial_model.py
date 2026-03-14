#!/usr/bin/env python3
"""Temporal-spatial model: predicts each token using temporal context + current-frame spatial context.

For each position p at frame t (decoded in raster order):
  - Temporal context: K previous tokens at position p from frames t-K..t-1
  - Spatial-temporal: K previous tokens at neighbors from frames t-K..t-1
  - Current-frame spatial: already-decoded positions in raster order (left, above, above-left, above-right)

The current-frame spatial tokens are available because we decode in raster order.
Both encoder and decoder use the same raster order, so inputs are identical.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TSConfig:
    vocab_size: int = 1024
    n_positions: int = 128       # spatial positions (8x16 grid)
    context_len: int = 20        # K previous frames
    n_temporal_neighbors: int = 4  # temporal neighbors (from prev frames)
    n_spatial_neighbors: int = 4   # current-frame neighbors (raster-available)
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.dim // self.n_head


# Precompute temporal neighbor indices (from previous frames — all available)
def build_temporal_neighbor_map(rows=8, cols=16):
    """4 neighbors: left, right, above, below. Clamp at borders."""
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


# Precompute current-frame spatial neighbor indices (raster-order available)
def build_spatial_neighbor_map(rows=8, cols=16):
    """For raster order decoding, which neighbors are available at each position?
    Returns: (128, 4) array. -1 means not available.
    Neighbors: left, above, above-left, above-right
    """
    n = rows * cols
    neighbor_idx = np.full((n, 4), -1, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            pos = r * cols + c
            # Left (same row, c-1) — available if c > 0
            if c > 0:
                neighbor_idx[pos, 0] = r * cols + (c - 1)
            # Above (prev row, same col) — available if r > 0
            if r > 0:
                neighbor_idx[pos, 1] = (r - 1) * cols + c
            # Above-left
            if r > 0 and c > 0:
                neighbor_idx[pos, 2] = (r - 1) * cols + (c - 1)
            # Above-right
            if r > 0 and c < cols - 1:
                neighbor_idx[pos, 3] = (r - 1) * cols + (c + 1)
    return neighbor_idx


TEMPORAL_NEIGHBOR_MAP = build_temporal_neighbor_map()
SPATIAL_NEIGHBOR_MAP = build_spatial_neighbor_map()


class TSAttention(nn.Module):
    def __init__(self, config: TSConfig):
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


class TSMLP(nn.Module):
    def __init__(self, config: TSConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class TSBlock(nn.Module):
    def __init__(self, config: TSConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = TSAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = TSMLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalSpatialModel(nn.Module):
    """Model that combines temporal and current-frame spatial context.

    Input: temporal context (batch, K, 1+4) + spatial context (batch, n_spatial)
    The temporal part is K timesteps of (center + 4 temporal neighbors).
    The spatial part is current-frame raster-available neighbors.

    We concatenate: [spatial_tokens..., temporal_tokens...]
    and use causal attention where spatial tokens can attend to each other
    and temporal tokens can attend to everything before them.

    Actually simpler approach: embed everything into a single vector per timestep,
    plus add spatial context as additional input features.

    Simplest approach: same as TemporalModelV2 but with additional spatial input.
    - Temporal: (K, 1+4) embedded and attended causally
    - Spatial: 4 current-frame neighbors embedded and summed into the last position's representation
    - Output: predict next token at this position

    This way, the model architecture is almost identical to v2, just with an extra
    spatial embedding added to the query at the last timestep.
    """

    def __init__(self, config: TSConfig = TSConfig()):
        super().__init__()
        self.config = config

        # Temporal embeddings (same as v2)
        self.center_emb = nn.Embedding(config.vocab_size, config.dim)
        self.temporal_neighbor_emb = nn.Embedding(config.vocab_size, config.dim)
        self.temporal_neighbor_weight = nn.Parameter(
            torch.ones(config.n_temporal_neighbors) / config.n_temporal_neighbors
        )

        # Spatial current-frame embeddings
        self.spatial_emb = nn.Embedding(config.vocab_size + 1, config.dim)  # +1 for "not available" token
        self.spatial_weight = nn.Parameter(
            torch.ones(config.n_spatial_neighbors) / config.n_spatial_neighbors
        )

        self.time_emb = nn.Embedding(config.context_len, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)

        self.layers = nn.ModuleList([TSBlock(config) for _ in range(config.n_layer)])
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
        temporal_tokens: (batch, K, 1+n_temporal_neighbors) — center + temporal neighbors
        spatial_tokens: (batch, n_spatial_neighbors) — current-frame raster neighbors
                        Use vocab_size (1024) for unavailable neighbors
        positions: (batch,) — spatial position IDs (0..127)
        Returns: (batch, K, vocab_size) logits
        """
        bsz, seq_len, n_ch = temporal_tokens.shape

        # Embed temporal center
        center = self.center_emb(temporal_tokens[:, :, 0])  # (batch, K, dim)

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

        # Add spatial context to ALL timesteps (since spatial neighbors at prev frames
        # would also be available — but we only have current frame spatial)
        # Actually, only add to the representation that makes the prediction
        # We add spatial info as a bias to the last position's embedding
        spatial_embeds = self.spatial_emb(spatial_tokens)  # (batch, n_spatial, dim)
        sw = torch.softmax(self.spatial_weight, dim=0).view(1, -1, 1)
        spatial_ctx = (spatial_embeds * sw).sum(dim=1)  # (batch, dim)

        # Add spatial context to all positions (it's a global bias)
        x = x + spatial_ctx.unsqueeze(1)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


TS_CONFIGS = {
    "tiny": TSConfig(dim=64, n_layer=2, n_head=2, intermediate_size=256),
    "small": TSConfig(dim=128, n_layer=4, n_head=4, intermediate_size=512),
    "medium": TSConfig(dim=256, n_layer=6, n_head=4, intermediate_size=1024),
}

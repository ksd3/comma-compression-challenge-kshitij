#!/usr/bin/env python3
"""Per-position temporal model v3: includes current-frame spatial context.

Like v2 (temporal + prev-frame neighbors) but also conditions on the token
at the position directly above in the CURRENT frame (available because we
decode row-by-row, top to bottom).

Both encoder and decoder decode in the same row-by-row order, so the
"above" token is always available during both encoding and decoding.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalV3Config:
    vocab_size: int = 1024
    n_positions: int = 128       # spatial positions (8x16 grid)
    context_len: int = 20        # K previous frames
    n_neighbors: int = 4         # temporal neighbors from prev frames
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0
    use_above: bool = True       # condition on current-frame above token

    @property
    def head_dim(self):
        return self.dim // self.n_head


def build_neighbor_map(rows=8, cols=16, n_neighbors=4):
    """Returns (128, n_neighbors) array of temporal neighbor indices."""
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


def build_above_map(rows=8, cols=16):
    """Returns (128,) array: above_map[pos] = position above, or -1 if row 0."""
    above_map = np.full(rows * cols, -1, dtype=np.int32)
    for r in range(1, rows):
        for c in range(cols):
            pos = r * cols + c
            above_map[pos] = (r - 1) * cols + c
    return above_map


NEIGHBOR_MAP_4 = build_neighbor_map(n_neighbors=4)
NEIGHBOR_MAP_8 = build_neighbor_map(n_neighbors=8)
ABOVE_MAP = build_above_map()


class V3Attention(nn.Module):
    def __init__(self, config: TemporalV3Config):
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


class V3MLP(nn.Module):
    def __init__(self, config: TemporalV3Config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class V3Block(nn.Module):
    def __init__(self, config: TemporalV3Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = V3Attention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = V3MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelV3(nn.Module):
    """Temporal model v3: temporal + prev-frame neighbors + current-frame above.

    Input per timestep: (center_token, neighbor1, ..., neighborN) = 1+N tokens from prev frames
    Additional input: above_token from current frame (or special "unavailable" token for row 0)

    The above token is embedded separately and added to all timestep representations.
    """

    def __init__(self, config: TemporalV3Config = TemporalV3Config()):
        super().__init__()
        self.config = config

        # Temporal embeddings
        self.center_emb = nn.Embedding(config.vocab_size, config.dim)
        self.neighbor_emb = nn.Embedding(config.vocab_size, config.dim)
        self.neighbor_weight = nn.Parameter(torch.ones(config.n_neighbors) / config.n_neighbors)

        # Current-frame above embedding (+1 for "not available" sentinel)
        if config.use_above:
            self.above_emb = nn.Embedding(config.vocab_size + 1, config.dim)

        self.time_emb = nn.Embedding(config.context_len, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)

        self.layers = nn.ModuleList([V3Block(config) for _ in range(config.n_layer)])
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

    def forward(self, tokens, positions=None, above_tokens=None):
        """
        tokens: (batch, seq_len, 1+n_neighbors) — center + temporal neighbor tokens
        positions: (batch,) — spatial position IDs (0..127)
        above_tokens: (batch,) — current-frame above token (vocab_size=1024 if unavailable)
        Returns: (batch, seq_len, vocab_size) logits
        """
        bsz, seq_len, n_ch = tokens.shape

        # Embed temporal center
        center = self.center_emb(tokens[:, :, 0])  # (batch, K, dim)

        # Embed and aggregate temporal neighbors
        if n_ch > 1:
            neighbor_tokens = tokens[:, :, 1:]
            neighbor_embeds = self.neighbor_emb(neighbor_tokens)
            w = torch.softmax(self.neighbor_weight, dim=0).view(1, 1, -1, 1)
            neighbors = (neighbor_embeds * w).sum(dim=2)
            x = center + neighbors
        else:
            x = center

        # Add time embeddings
        time_ids = torch.arange(seq_len, device=tokens.device)
        x = x + self.time_emb(time_ids)

        # Add position embeddings
        if positions is not None:
            x = x + self.pos_emb(positions).unsqueeze(1)

        # Add current-frame above token embedding (broadcast to all timesteps)
        if self.config.use_above and above_tokens is not None:
            above_embed = self.above_emb(above_tokens)  # (batch, dim)
            x = x + above_embed.unsqueeze(1)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CONFIGS_V3 = {
    "tiny": TemporalV3Config(dim=64, n_layer=2, n_head=2, intermediate_size=256),
    "small": TemporalV3Config(dim=128, n_layer=4, n_head=4, intermediate_size=512),
    "medium": TemporalV3Config(dim=256, n_layer=6, n_head=4, intermediate_size=1024),
}

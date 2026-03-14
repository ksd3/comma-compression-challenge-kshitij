#!/usr/bin/env python3
"""Temporal-frame model: combines temporal context with frame-level spatial context.

For each position in the current frame:
  - Temporal context: K previous tokens at that position (via per-position temporal encoding)
  - Full previous frame: cross-attention to all 128 positions from frame t-1
  - Current-frame spatial: causal self-attention to already-decoded positions (raster order)

Architecture:
  1. Per-position temporal encoder: embeds K previous tokens into a summary per position
  2. Previous frame context: 128 position embeddings from frame t-1
  3. Current frame decoder: raster-order causal attention + cross-attention to temporal summaries
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TFConfig:
    vocab_size: int = 1024
    n_positions: int = 128       # 8x16 grid
    context_len: int = 10        # K previous frames for temporal summary
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.dim // self.n_head


class TFAttention(nn.Module):
    def __init__(self, config: TFConfig, is_cross=False):
        super().__init__()
        self.config = config
        self.is_cross = is_cross
        self.q_proj = nn.Linear(config.dim, config.dim, bias=True)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=True)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=True)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=True)

    def forward(self, x, context=None, mask=None):
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x)
        if self.is_cross and context is not None:
            k = self.k_proj(context)
            v = self.v_proj(context)
            kv_len = context.shape[1]
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)
            kv_len = seqlen
        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.config.n_head, self.config.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.config.n_head, self.config.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)
        return self.out_proj(y)


class TFMLP(nn.Module):
    def __init__(self, config: TFConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class TFBlock(nn.Module):
    def __init__(self, config: TFConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.self_attn = TFAttention(config, is_cross=False)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.cross_attn = TFAttention(config, is_cross=True)
        self.ln_3 = nn.LayerNorm(config.dim)
        self.mlp = TFMLP(config)

    def forward(self, x, context, causal_mask):
        x = x + self.self_attn(self.ln_1(x), mask=causal_mask)
        x = x + self.cross_attn(self.ln_2(x), context=context)
        x = x + self.mlp(self.ln_3(x))
        return x


class TemporalFrameModel(nn.Module):
    """Temporal-frame model combining temporal and spatial context.

    For efficient temporal encoding, we embed K previous tokens at each position
    and create a summary via mean-pooling (simple but effective).

    Input:
      - prev_frames: (batch, K, 128) — K previous frames
      - curr_tokens: (batch, 128) — current frame tokens (for teacher-forced training)
    Output:
      - logits: (batch, 128, vocab_size)
    """

    def __init__(self, config: TFConfig = TFConfig()):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)

        # Temporal encoding: project K-length token sequences to summary
        self.temporal_emb = nn.Embedding(config.vocab_size, config.dim)
        self.temporal_time_emb = nn.Embedding(config.context_len, config.dim)
        # Simple MLP to compress temporal info
        self.temporal_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(approximate='tanh'),
        )

        # Decoder layers
        self.layers = nn.ModuleList([TFBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Causal mask for raster-order decoding
        mask = torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool))
        self.register_buffer('causal_mask', mask.view(1, 1, config.n_positions, config.n_positions))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_temporal(self, prev_frames):
        """Encode K previous frames into per-position temporal summaries.

        prev_frames: (batch, K, 128) — K previous frames of tokens
        Returns: (batch, 128, dim) — per-position temporal summaries
        """
        bsz, K, n_pos = prev_frames.shape

        # Embed tokens: (batch, K, 128, dim)
        tok_emb = self.temporal_emb(prev_frames)

        # Add time embeddings (most recent = index K-1)
        time_ids = torch.arange(K, device=prev_frames.device)
        tok_emb = tok_emb + self.temporal_time_emb(time_ids).view(1, K, 1, self.config.dim)

        # Mean-pool across time: (batch, 128, dim)
        summary = tok_emb.mean(dim=1)

        # Add position embeddings
        pos_ids = torch.arange(n_pos, device=prev_frames.device)
        summary = summary + self.pos_emb(pos_ids)

        # Project
        summary = self.temporal_proj(summary)

        return summary

    def forward(self, prev_frames, curr_tokens):
        """
        prev_frames: (batch, K, 128) — K previous frames
        curr_tokens: (batch, 128) — current frame tokens
        Returns: (batch, 128, vocab_size) logits
        """
        bsz = prev_frames.shape[0]

        # Encode temporal context: (batch, 128, dim)
        temporal_ctx = self.encode_temporal(prev_frames)

        # Encode current frame (shifted for AR prediction)
        pos_ids = torch.arange(self.config.n_positions, device=curr_tokens.device)
        shifted = torch.zeros_like(curr_tokens)
        shifted[:, 1:] = curr_tokens[:, :-1]

        x = self.token_emb(shifted) + self.pos_emb(pos_ids)

        for layer in self.layers:
            x = layer(x, temporal_ctx, self.causal_mask)

        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


TF_CONFIGS = {
    "tiny": TFConfig(dim=64, n_layer=2, n_head=2, intermediate_size=256, context_len=5),
    "small": TFConfig(dim=128, n_layer=4, n_head=4, intermediate_size=512, context_len=10),
    "medium": TFConfig(dim=192, n_layer=6, n_head=4, intermediate_size=768, context_len=10),
    "large": TFConfig(dim=256, n_layer=8, n_head=8, intermediate_size=1024, context_len=10),
}

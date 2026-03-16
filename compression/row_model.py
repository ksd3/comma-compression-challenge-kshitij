#!/usr/bin/env python3
"""Row-level frame model for commaVQ compression.

For each frame, decode 8 rows (top to bottom). Within each row, all 16
positions are predicted in parallel — they share the same context:
  - K previous frames (all 128 positions) via cross-attention
  - Previously decoded rows in current frame via cross-attention
  - Position-specific learned queries

This allows efficient decompression: 8 forward passes per frame.

During training, teacher forcing is used. During inference, decoded
rows are fed back as context for subsequent rows.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RowModelConfig:
    vocab_size: int = 1024
    n_positions: int = 128       # 8x16 grid
    n_rows: int = 8
    n_cols: int = 16
    n_prev_frames: int = 3      # K previous frames
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.dim // self.n_head


class RowAttention(nn.Module):
    def __init__(self, config: RowModelConfig, is_cross=False):
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


class RowMLP(nn.Module):
    def __init__(self, config: RowModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class RowBlock(nn.Module):
    def __init__(self, config: RowModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.self_attn = RowAttention(config, is_cross=False)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.cross_attn = RowAttention(config, is_cross=True)
        self.ln_3 = nn.LayerNorm(config.dim)
        self.mlp = RowMLP(config)

    def forward(self, x, context):
        # Self-attention within the 16 query positions (no mask needed — all attend to all)
        x = x + self.self_attn(self.ln_1(x))
        # Cross-attention to context (previous frames + decoded rows)
        x = x + self.cross_attn(self.ln_2(x), context=context)
        # MLP
        x = x + self.mlp(self.ln_3(x))
        return x


class RowModel(nn.Module):
    """Row-level frame model.

    For each row r (0..7), predict 16 tokens simultaneously.
    Context: K previous frames (K×128 tokens) + rows 0..r-1 of current frame.

    Forward pass processes ALL 8 rows with teacher forcing for training efficiency.
    For inference, rows are processed sequentially.
    """

    def __init__(self, config: RowModelConfig = RowModelConfig()):
        super().__init__()
        self.config = config

        # Token embedding (shared)
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)

        # Position embedding for all 128 positions
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)

        # Row query embedding: 16 learned queries per row
        # These are the "query" positions for each row
        self.row_query = nn.Embedding(config.n_positions, config.dim)

        # Frame time embedding (distinguish prev frames from current)
        self.time_emb = nn.Embedding(config.n_prev_frames + 1, config.dim)

        # Transformer blocks
        self.layers = nn.ModuleList([RowBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def build_context(self, prev_frames, decoded_rows, device):
        """Build context from previous frames and already-decoded rows.

        prev_frames: (batch, K, 128) previous frame tokens
        decoded_rows: list of (batch, 16) tensors for rows 0..r-1
        Returns: (batch, K*128 + r*16, dim) context embeddings
        """
        bsz = prev_frames.shape[0]
        K = prev_frames.shape[1]
        pos_ids = torch.arange(self.config.n_positions, device=device)

        contexts = []

        # Encode previous frames
        for k in range(K):
            frame_emb = self.token_emb(prev_frames[:, k]) + self.pos_emb(pos_ids)
            frame_emb = frame_emb + self.time_emb(torch.tensor(k, device=device))
            contexts.append(frame_emb)

        # Encode already-decoded rows from current frame
        if decoded_rows:
            n_decoded = len(decoded_rows) * self.config.n_cols
            decoded_tokens = torch.cat(decoded_rows, dim=1)  # (batch, r*16)
            decoded_pos = torch.arange(n_decoded, device=device)
            decoded_emb = self.token_emb(decoded_tokens) + self.pos_emb(decoded_pos)
            decoded_emb = decoded_emb + self.time_emb(torch.tensor(K, device=device))
            contexts.append(decoded_emb)

        return torch.cat(contexts, dim=1)

    def predict_row(self, context, row_idx, device):
        """Predict tokens for one row given context.

        context: (batch, ctx_len, dim) — previous frames + decoded rows
        row_idx: which row (0..7)
        Returns: (batch, 16, vocab_size) logits
        """
        cols = self.config.n_cols
        pos_start = row_idx * cols
        pos_ids = torch.arange(pos_start, pos_start + cols, device=device)

        # Query for this row
        x = self.row_query(pos_ids).unsqueeze(0).expand(context.shape[0], -1, -1)

        for layer in self.layers:
            x = layer(x, context)

        x = self.ln_f(x)
        return self.head(x)

    def forward(self, prev_frames, curr_tokens):
        """Training forward pass with teacher forcing.

        prev_frames: (batch, K, 128)
        curr_tokens: (batch, 128)
        Returns: (batch, 128, vocab_size) logits
        """
        bsz = prev_frames.shape[0]
        device = prev_frames.device
        cols = self.config.n_cols
        rows = self.config.n_rows

        all_logits = []
        decoded_rows = []

        for row_idx in range(rows):
            context = self.build_context(prev_frames, decoded_rows, device)
            logits = self.predict_row(context, row_idx, device)  # (batch, 16, vocab)
            all_logits.append(logits)

            # Teacher forcing: use ground truth for next row's context
            row_tokens = curr_tokens[:, row_idx * cols:(row_idx + 1) * cols]
            decoded_rows.append(row_tokens)

        return torch.cat(all_logits, dim=1)  # (batch, 128, vocab)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


ROW_CONFIGS = {
    "tiny": RowModelConfig(dim=64, n_layer=2, n_head=2, intermediate_size=256, n_prev_frames=3),
    "small": RowModelConfig(dim=128, n_layer=4, n_head=4, intermediate_size=512, n_prev_frames=3),
    "medium": RowModelConfig(dim=192, n_layer=6, n_head=4, intermediate_size=768, n_prev_frames=3),
    "large": RowModelConfig(dim=256, n_layer=8, n_head=8, intermediate_size=1024, n_prev_frames=3),
}

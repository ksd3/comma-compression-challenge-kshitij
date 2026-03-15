#!/usr/bin/env python3
"""Frame-level autoregressive model for commaVQ compression.

For each frame at time t, the model predicts all 128 tokens autoregressively
in raster order, conditioned on:
  - The full previous frame (128 tokens from t-1) via cross-attention
  - Already-decoded tokens in the current frame via causal self-attention

Architecture:
  - Token embedding: shared for all tokens (vocab_size=1024)
  - Position embedding: 128 positions for spatial layout
  - Cross-attention layers: attend to previous frame
  - Causal self-attention: attend to already-decoded current-frame tokens
  - Output: per-position logits over 1024 tokens
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrameModelConfig:
    vocab_size: int = 1024
    n_positions: int = 128       # 8x16 grid
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0
    n_prev_frames: int = 1      # number of previous frames to condition on

    @property
    def head_dim(self):
        return self.dim // self.n_head


class FrameAttention(nn.Module):
    """Self-attention with optional cross-attention to context."""
    def __init__(self, config: FrameModelConfig, is_cross=False):
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


class FrameMLP(nn.Module):
    def __init__(self, config: FrameModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class FrameBlock(nn.Module):
    def __init__(self, config: FrameModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.self_attn = FrameAttention(config, is_cross=False)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.cross_attn = FrameAttention(config, is_cross=True)
        self.ln_3 = nn.LayerNorm(config.dim)
        self.mlp = FrameMLP(config)

    def forward(self, x, context, causal_mask):
        # Causal self-attention on current frame tokens
        x = x + self.self_attn(self.ln_1(x), mask=causal_mask)
        # Cross-attention to previous frame
        x = x + self.cross_attn(self.ln_2(x), context=context)
        # MLP
        x = x + self.mlp(self.ln_3(x))
        return x


class FrameModel(nn.Module):
    """Frame-level autoregressive model.

    Predicts current frame tokens in raster order, conditioned on previous frame.

    Input:
      - prev_frame: (batch, 128) int tokens from previous frame
      - curr_tokens: (batch, 128) int tokens from current frame (teacher-forced during training)
    Output:
      - logits: (batch, 128, vocab_size) predictions for each position
    """

    def __init__(self, config: FrameModelConfig = FrameModelConfig()):
        super().__init__()
        self.config = config

        # Shared token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        # Position embedding (128 spatial positions)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)
        # Frame type embedding (prev vs current)
        self.frame_type_emb = nn.Embedding(2, config.dim)

        self.layers = nn.ModuleList([FrameBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Causal mask for self-attention (raster order)
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

    def encode_prev_frames(self, prev_frames):
        """Encode previous frame(s) tokens as context.

        prev_frames: (batch, K, 128) or (batch, 128) — K previous frames
        Returns: (batch, K*128, dim) context
        """
        if prev_frames.dim() == 2:
            prev_frames = prev_frames.unsqueeze(1)  # (batch, 1, 128)

        bsz, K, n_pos = prev_frames.shape
        pos_ids = torch.arange(n_pos, device=prev_frames.device)

        contexts = []
        for k in range(K):
            x = self.token_emb(prev_frames[:, k]) + self.pos_emb(pos_ids)
            x = x + self.frame_type_emb(torch.zeros(1, dtype=torch.long, device=prev_frames.device))
            contexts.append(x)

        return torch.cat(contexts, dim=1)  # (batch, K*128, dim)

    def forward(self, prev_frames, curr_tokens):
        """
        prev_frames: (batch, K, 128) or (batch, 128) — previous frame(s) tokens
        curr_tokens: (batch, 128) — current frame tokens (shifted right for AR)
        Returns: (batch, 128, vocab_size) logits
        """
        bsz = prev_frames.shape[0]

        # Encode previous frame(s) as context
        context = self.encode_prev_frames(prev_frames)  # (batch, K*128, dim)

        # Encode current frame tokens (shifted: position i predicts token i)
        # Input at position i is token at position i-1 (for AR prediction)
        # Position 0 gets a special "start" signal (we use the zero embedding)
        pos_ids = torch.arange(self.config.n_positions, device=curr_tokens.device)

        # Shift tokens right: input[0] = 0 (start), input[i] = curr_tokens[i-1]
        shifted = torch.zeros_like(curr_tokens)
        shifted[:, 1:] = curr_tokens[:, :-1]
        # Position 0 input is a learned "start of frame" token — we use token 0 shifted in

        x = self.token_emb(shifted) + self.pos_emb(pos_ids)
        x = x + self.frame_type_emb(torch.ones(1, dtype=torch.long, device=curr_tokens.device))

        for layer in self.layers:
            x = layer(x, context, self.causal_mask)

        x = self.ln_f(x)
        return self.head(x)

    def predict_position(self, prev_context, decoded_tokens, pos):
        """Predict a single position during inference.

        prev_context: (1, 128, dim) — encoded previous frame
        decoded_tokens: (1, pos) — already decoded tokens in current frame
        pos: position to predict (0..127)
        Returns: (1, vocab_size) logits
        """
        # Build input for positions 0..pos
        n = pos + 1
        pos_ids = torch.arange(n, device=prev_context.device)

        # Shifted input
        if pos == 0:
            input_tokens = torch.zeros(1, 1, dtype=torch.long, device=prev_context.device)
        else:
            input_tokens = torch.zeros(1, n, dtype=torch.long, device=prev_context.device)
            input_tokens[0, 1:] = decoded_tokens[0, :pos]

        x = self.token_emb(input_tokens) + self.pos_emb(pos_ids)
        x = x + self.frame_type_emb(torch.ones(1, dtype=torch.long, device=prev_context.device))

        mask = self.causal_mask[:, :, :n, :n]
        for layer in self.layers:
            x = layer(x, prev_context, mask)

        x = self.ln_f(x)
        logits = self.head(x[:, -1:])  # Only last position
        return logits[:, 0]

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


FRAME_CONFIGS = {
    "tiny": FrameModelConfig(dim=64, n_layer=2, n_head=2, intermediate_size=256),
    "small": FrameModelConfig(dim=128, n_layer=4, n_head=4, intermediate_size=512),
    "small3": FrameModelConfig(dim=128, n_layer=4, n_head=4, intermediate_size=512, n_prev_frames=3),
    "medium": FrameModelConfig(dim=192, n_layer=6, n_head=4, intermediate_size=768),
    "medium3": FrameModelConfig(dim=192, n_layer=6, n_head=4, intermediate_size=768, n_prev_frames=3),
    "large": FrameModelConfig(dim=256, n_layer=8, n_head=8, intermediate_size=1024),
}

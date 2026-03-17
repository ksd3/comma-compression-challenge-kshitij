#!/usr/bin/env python3
"""
Decompress commaVQ submission using frame-level autoregressive model + ANS.

This file is self-contained and included in the submission zip.
Dependencies: torch, numpy, constriction (all available via pip/PyPI).
"""
import os
import sys
import io
import struct
import json
import lzma
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Install constriction if needed
try:
    import constriction
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'constriction'])
    import constriction

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get('OUTPUT_DIR', HERE / 'compression_challenge_submission_decompressed'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================
# Frame Model Architecture (must match training exactly)
# =====================================================================

@dataclass
class FrameModelConfig:
    vocab_size: int = 1024
    n_positions: int = 128
    dim: int = 128
    n_layer: int = 4
    n_head: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0
    n_prev_frames: int = 1

    @property
    def head_dim(self):
        return self.dim // self.n_head


class FrameAttention(nn.Module):
    def __init__(self, config, is_cross=False):
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
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class FrameBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.self_attn = FrameAttention(config, is_cross=False)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.cross_attn = FrameAttention(config, is_cross=True)
        self.ln_3 = nn.LayerNorm(config.dim)
        self.mlp = FrameMLP(config)

    def forward(self, x, context, causal_mask):
        x = x + self.self_attn(self.ln_1(x), mask=causal_mask)
        x = x + self.cross_attn(self.ln_2(x), context=context)
        x = x + self.mlp(self.ln_3(x))
        return x


class FrameModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.n_positions, config.dim)
        self.frame_type_emb = nn.Embedding(2, config.dim)
        self.layers = nn.ModuleList([FrameBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        mask = torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool))
        self.register_buffer('causal_mask', mask.view(1, 1, config.n_positions, config.n_positions))

    def encode_prev_frames(self, prev_frames):
        if prev_frames.dim() == 2:
            prev_frames = prev_frames.unsqueeze(1)
        bsz, K, n_pos = prev_frames.shape
        pos_ids = torch.arange(n_pos, device=prev_frames.device)
        contexts = []
        for k in range(K):
            x = self.token_emb(prev_frames[:, k]) + self.pos_emb(pos_ids)
            x = x + self.frame_type_emb(torch.zeros(1, dtype=torch.long, device=prev_frames.device))
            contexts.append(x)
        return torch.cat(contexts, dim=1)

    def forward(self, prev_frames, curr_tokens):
        context = self.encode_prev_frames(prev_frames)
        pos_ids = torch.arange(self.config.n_positions, device=curr_tokens.device)
        shifted = torch.zeros_like(curr_tokens)
        shifted[:, 1:] = curr_tokens[:, :-1]
        x = self.token_emb(shifted) + self.pos_emb(pos_ids)
        x = x + self.frame_type_emb(torch.ones(1, dtype=torch.long, device=curr_tokens.device))
        for layer in self.layers:
            x = layer(x, context, self.causal_mask)
        x = self.ln_f(x)
        return self.head(x)


# =====================================================================
# Model Loading
# =====================================================================

def load_model_compact(compact_path):
    with open(compact_path, 'rb') as f:
        raw = lzma.decompress(f.read())

    offset = 0
    meta_len = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    meta = json.loads(raw[offset:offset + meta_len]); offset += meta_len

    config = FrameModelConfig(
        dim=meta['dim'],
        n_layer=meta['n_layer'],
        n_head=meta['n_head'],
        intermediate_size=meta['intermediate_size'],
        n_prev_frames=meta.get('n_prev_frames', 1),
    )
    model = FrameModel(config)

    state = {}
    for key, shape in zip(meta['keys'], meta['shapes']):
        numel = 1
        for s in shape:
            numel *= s
        data = np.frombuffer(raw[offset:offset + numel * 2], dtype=np.float16).copy()
        offset += numel * 2
        state[key] = torch.from_numpy(data).float().reshape(shape)

    n_pos = config.n_positions
    mask = torch.tril(torch.ones(n_pos, n_pos, dtype=torch.bool))
    state['causal_mask'] = mask.view(1, 1, n_pos, n_pos)

    model.load_state_dict(state)
    model = model.to(device=DEVICE)
    model.eval()
    return model, config


# =====================================================================
# Decompression
# =====================================================================

def decompress_segment(model, config, compressed_bytes):
    """Decompress one segment using frame model + ANS."""
    compressed_u32 = np.frombuffer(compressed_bytes, dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed_u32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    K = config.n_prev_frames
    n_frames, n_pos = 1200, 128
    tokens = np.zeros((n_frames, n_pos), dtype=np.int16)

    for frame_idx in range(n_frames):
        # Build previous frames context
        if frame_idx < K:
            prev = np.zeros((K, 128), dtype=np.int64)
            if frame_idx > 0:
                prev[K - frame_idx:] = tokens[0:frame_idx]
        else:
            prev = tokens[frame_idx - K:frame_idx].astype(np.int64)

        prev_t = torch.tensor(prev, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Decode position by position (autoregressive)
        decoded = np.zeros(128, dtype=np.int64)
        for pos in range(n_pos):
            curr_t = torch.tensor(decoded, dtype=torch.long, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                    logits = model(prev_t, curr_t)
            pos_logits = logits[0, pos, :config.vocab_size].float()
            probs = torch.softmax(pos_logits, dim=-1).cpu().numpy().astype(np.float32)
            token = coder.decode(model_family, probs.reshape(1, -1))[0]
            decoded[pos] = token
            tokens[frame_idx, pos] = token

    return tokens.reshape(-1, 8, 16)


def read_data_bin(data_bin_path):
    """Parse the single-blob format."""
    with open(data_bin_path, 'rb') as f:
        raw = f.read()

    offset = 0
    backend_name_len, = struct.unpack_from('<B', raw, offset); offset += 1
    backend_name = raw[offset:offset + backend_name_len].decode('utf-8'); offset += backend_name_len
    num_segments, = struct.unpack_from('<I', raw, offset); offset += 4

    names = []
    for _ in range(num_segments):
        name_len, = struct.unpack_from('<H', raw, offset); offset += 2
        name = raw[offset:offset + name_len].decode('utf-8'); offset += name_len
        names.append(name)

    sizes = []
    for _ in range(num_segments):
        size, = struct.unpack_from('<I', raw, offset); offset += 4
        sizes.append(size)

    for name, size in zip(names, sizes):
        blob = raw[offset:offset + size]
        offset += size
        yield name, blob


def decompress_all():
    """Decompress all segments."""
    model_path = HERE / 'model.bin'
    data_path = HERE / 'data.bin'

    print(f"Loading model from {model_path}...", flush=True)
    model, config = load_model_compact(model_path)
    print(f"  Model loaded on {DEVICE}, K={config.n_prev_frames}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for name, blob in read_data_bin(data_path):
        tokens = decompress_segment(model, config, blob)
        np.save(output_dir / name, tokens)
        count += 1
        if count % 100 == 0:
            print(f"  Decompressed {count} segments", flush=True)

    print(f"Decompressed {count} segments total", flush=True)


if __name__ == '__main__':
    decompress_all()

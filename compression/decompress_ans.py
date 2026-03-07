#!/usr/bin/env python3
"""
Decompress commaVQ submission using trained GPT model + ANS coding.

This file is included in the submission zip and run by evaluate.sh.
Dependencies are installed from PyPI (free, only zip size counts).
"""
import os
import sys
import io
import json
import lzma
import struct
import subprocess

# Install dependencies if needed
def ensure_deps():
    for pkg in ['torch', 'constriction', 'numpy']:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get('OUTPUT_DIR', HERE / 'compression_challenge_submission_decompressed'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---- Inline GPT model ----

@dataclass
class GPTConfig:
    block_size: int = 20*129
    vocab_size: int = 1025
    n_layer: int = 24
    n_head: int = 16
    dim: int = 1024
    intermediate_size: int = 4*1024
    tokens_per_frame: int = 129

    @property
    def bos_token(self):
        return self.vocab_size - 1

    @property
    def head_dim(self):
        return self.dim // self.n_head

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.dim)
        self.ln_2 = nn.LayerNorm(config.dim)

    def forward(self, x, input_pos, mask):
        h = x + self.attn(self.ln_1(x), mask, input_pos)
        return h + self.mlp(self.ln_2(h))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.dim, 3*config.dim, bias=True)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=True)
        self.kv_cache = None

    def forward(self, x, mask, input_pos=None):
        bsz, seqlen, _ = x.shape
        q, k, v = self.c_attn(x).split([self.config.dim]*3, dim=-1)
        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.config.n_head, self.config.head_dim).transpose(1, 2)
        if self.kv_cache is None:
            mask = mask[:, :, :, :seqlen]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))

class GPT(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config
        transformer = {
            'wte': nn.Embedding(config.vocab_size, config.dim),
            'wpe': nn.Embedding(config.block_size, config.dim),
            'h': nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
            'ln_f': nn.LayerNorm(config.dim)
        }
        self.transformer = nn.ModuleDict(transformer)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, idx, input_pos=None):
        if input_pos is None:
            input_pos = torch.arange(idx.shape[1], device=idx.device)
        mask = self.causal_mask[:, :, input_pos]
        x = self.transformer.wte(idx) + self.transformer.wpe(input_pos)
        for layer in self.transformer.h:
            x = layer(x, input_pos, mask)
        return self.lm_head(self.transformer.ln_f(x))


class FrameIndependentGPT(GPT):
    """GPT with frame-independent attention mask for compression.
    Tokens within a frame can't see each other, only previous frames + own BOS."""

    _mask_cache = {}

    def forward_with_frame_mask(self, idx, context_frames=20):
        bsz, seq_len = idx.shape
        tpf = self.config.tokens_per_frame  # 129

        if seq_len not in self._mask_cache or self._mask_cache[seq_len].device != idx.device:
            positions = torch.arange(seq_len, device=idx.device)
            frame_ids = positions // tpf
            bos_positions = frame_ids * tpf

            q_frames = frame_ids.unsqueeze(1)
            k_frames = frame_ids.unsqueeze(0)
            k_pos = positions.unsqueeze(0)
            q_bos = bos_positions.unsqueeze(1)

            prev_frame_mask = k_frames < q_frames
            own_bos_mask = k_pos == q_bos
            mask = (prev_frame_mask | own_bos_mask).view(1, 1, seq_len, seq_len)
            self._mask_cache[seq_len] = mask
        else:
            mask = self._mask_cache[seq_len]

        input_pos = torch.arange(seq_len, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(input_pos)
        for layer in self.transformer.h:
            x = layer(x, input_pos, mask)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


MODEL_CONFIGS = {
    "tiny": GPTConfig(n_layer=4, n_head=4, dim=128, intermediate_size=512),
    "small": GPTConfig(n_layer=6, n_head=4, dim=256, intermediate_size=1024),
    "medium": GPTConfig(n_layer=8, n_head=8, dim=512, intermediate_size=2048),
}


# ---- Model loading ----

def load_model_compact(compact_path):
    """Load model from LZMA-compressed fp16 format."""
    with open(compact_path, 'rb') as f:
        raw = lzma.decompress(f.read())

    offset = 0
    meta_len = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    meta = json.loads(raw[offset:offset + meta_len]); offset += meta_len

    config = MODEL_CONFIGS[meta['config_size']]
    model_version = meta.get('version', 1)
    if model_version == 2:
        model = FrameIndependentGPT(config)
    else:
        model = GPT(config)

    state = {}
    for key, shape in zip(meta['keys'], meta['shapes']):
        numel = 1
        for s in shape:
            numel *= s
        data = np.frombuffer(raw[offset:offset + numel * 2], dtype=np.float16).copy()
        offset += numel * 2
        state[key] = torch.from_numpy(data).float().reshape(shape)

    state['causal_mask'] = torch.tril(
        torch.ones(config.block_size, config.block_size, dtype=torch.bool)
    ).view(1, 1, config.block_size, config.block_size)

    model.load_state_dict(state)
    model = model.to(device=DEVICE)
    model.eval()
    return model, config, model_version


# ---- Data format ----

def read_data_bin(data_bin_path):
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
        blob = raw[offset:offset + size]; offset += size
        yield backend_name, name, blob


# ---- ANS decompression ----

def decompress_segment(model, config, compressed_bytes):
    """Decompress one segment by recomputing exact same probabilities as encoder."""
    import constriction

    compressed_u32 = np.frombuffer(compressed_bytes, dtype=np.uint32)

    n_frames, n_tokens = 1200, 128
    bos = config.bos_token

    # Step 1: We need to reproduce exact same probs as encoder.
    # The encoder feeds complete frames (context + current frame with all tokens)
    # and uses causal masking. During decoding we must be autoregressive.
    #
    # HOWEVER: we can decode ALL 1200*128 tokens at once if we have the same
    # probability table. The trick: decode by reconstructing probs frame-by-frame.

    # Decode token-by-token within each frame
    coder = constriction.stream.stack.AnsCoder(compressed_u32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    all_tokens = []

    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - 19)

            # Build the same context window as encoder (with BOS tokens)
            if len(all_tokens) == 0:
                # First frame: just BOS
                bos_arr = np.array([[bos]], dtype=np.int64)
                full_seq_np = bos_arr.ravel()
            else:
                # Previous decoded frames + current BOS
                prev = np.array(all_tokens[start_frame:frame_idx])  # (K, 128)
                bos_col = np.full((prev.shape[0], 1), bos, dtype=np.int64)
                prev_with_bos = np.concatenate([bos_col, prev], axis=1)  # (K, 129)
                full_seq_np = np.append(prev_with_bos.ravel(), bos)

            seq = torch.tensor(full_seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)

            # Decode 128 tokens autoregressively
            frame_tokens = []
            for j in range(n_tokens):
                if DEVICE == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(seq)
                else:
                    logits = model(seq)

                token_logits = logits[0, -1, :1024].float()
                probs = torch.softmax(token_logits, dim=-1).cpu().numpy()

                token = coder.decode(model_family, probs.astype(np.float32).reshape(1, -1))
                frame_tokens.append(int(token[0]))

                if j < n_tokens - 1:
                    next_tok = torch.tensor([[frame_tokens[-1]]], dtype=torch.long, device=DEVICE)
                    seq = torch.cat([seq, next_tok], dim=1)

            all_tokens.append(frame_tokens)

            if (frame_idx + 1) % 100 == 0:
                print(f"  Frame {frame_idx+1}/{n_frames}", flush=True)

    return np.array(all_tokens, dtype=np.int16).reshape(-1, 8, 16)


def decompress_segment_v2(model, config, compressed_bytes):
    """Decompress using frame-independent model: 1 forward pass per frame.

    Feed [prev_decoded_frames, BOS, zeros] and get 128 predictions at once.
    The zeros don't matter because data tokens are masked from each other.
    """
    import constriction

    compressed_u32 = np.frombuffer(compressed_bytes, dtype=np.uint32)

    n_frames, n_tokens = 1200, 128
    bos = config.bos_token

    coder = constriction.stream.stack.AnsCoder(compressed_u32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    all_tokens = []

    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - 19)

            # Build context: prev frames + current frame (BOS + dummy data)
            if len(all_tokens) == 0:
                # First frame: [BOS, 0, 0, ..., 0] (129 tokens)
                frame_seq = np.zeros(129, dtype=np.int64)
                frame_seq[0] = bos
                full_seq_np = frame_seq
            else:
                # Previous decoded frames with BOS + current frame with dummy data
                prev = np.array(all_tokens[start_frame:frame_idx])  # (K, 128)
                bos_col = np.full((prev.shape[0], 1), bos, dtype=np.int64)
                prev_with_bos = np.concatenate([bos_col, prev], axis=1)  # (K, 129)
                cur_frame = np.zeros(129, dtype=np.int64)
                cur_frame[0] = bos
                full_seq_np = np.concatenate([prev_with_bos.ravel(), cur_frame])

            seq = torch.tensor(full_seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)

            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model.forward_with_frame_mask(seq)
            else:
                logits = model.forward_with_frame_mask(seq)

            # Get predictions for all 128 data tokens at once
            frame_offset = (frame_idx - start_frame) * 129
            frame_logits = logits[0, frame_offset:frame_offset + 128, :1024].float()
            frame_probs = torch.softmax(frame_logits, dim=-1).cpu().numpy()

            # Decode all 128 tokens
            tokens = coder.decode(model_family, frame_probs.astype(np.float32))
            all_tokens.append(tokens.tolist())

            if (frame_idx + 1) % 100 == 0:
                print(f"  Frame {frame_idx+1}/{n_frames}", flush=True)

    return np.array(all_tokens, dtype=np.int16).reshape(-1, 8, 16)


def decompress_all():
    data_bin = HERE / 'data.bin'
    model_path = HERE / 'model.bin'

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    model, config, model_version = load_model_compact(model_path)
    print(f"Model loaded on {DEVICE} (version={model_version})", flush=True)

    decomp_fn = decompress_segment_v2 if model_version == 2 else decompress_segment

    count = 0
    for backend_name, name, blob in read_data_bin(data_bin):
        print(f"Decompressing {name}...", flush=True)
        tokens = decomp_fn(model, config, blob)
        np.save(output_dir / name, tokens)
        count += 1
        print(f"  Done ({count} segments)", flush=True)

    print(f"Decompressed {count} segments total", flush=True)


if __name__ == '__main__':
    decompress_all()

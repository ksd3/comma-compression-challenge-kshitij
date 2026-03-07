#!/usr/bin/env python3
"""
GPT + ANS compression for commaVQ challenge.
Uses the provided GPT-2 medium world model to predict token probabilities,
then encodes with ANS (constriction library).

The model is ~1.3GB so it can't be shipped in the zip.
This module is used for analysis and as a reference for training smaller models.
"""
import sys
import os
import time
import struct
import numpy as np
import torch
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.gpt import GPT, GPTConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16


def load_model():
    model = GPT()
    model.load_state_dict_from_url()
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def tokens_to_gpt_input(tokens_2d):
    """Convert (N, 128) tokens to GPT input format with BOS tokens.
    Returns (N, 129) with BOS prepended to each frame.
    """
    config = GPTConfig()
    n_frames, n_tokens = tokens_2d.shape
    assert n_tokens == 128
    bos = np.full((n_frames, 1), config.bos_token, dtype=tokens_2d.dtype)
    return np.concatenate([bos, tokens_2d], axis=1)  # (N, 129)


def get_probabilities_for_segment(model, tokens_2d, context_frames=20):
    """Get probability distributions for each token in the segment.

    Uses sliding windows of `context_frames` frames.
    Returns: probs array of shape (1200*128, 1024).
    """
    config = GPTConfig()
    n_frames = tokens_2d.shape[0]
    gpt_input = tokens_to_gpt_input(tokens_2d)  # (1200, 129)

    all_probs = []

    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - context_frames + 1)
            end_frame = frame_idx + 1

            window = gpt_input[start_frame:end_frame]  # (num_ctx_frames, 129)
            seq = torch.tensor(window.ravel(), dtype=torch.long, device=DEVICE).unsqueeze(0)

            logits = model(seq)  # (1, seq_len, vocab_size)

            # logits[t] predicts token[t+1]
            # Current frame offset in window:
            frame_offset = (frame_idx - start_frame) * 129
            # logits[frame_offset + j] predicts token[frame_offset + 1 + j] for j=0..127
            frame_logits = logits[0, frame_offset:frame_offset + 128, :1024]
            frame_probs = torch.softmax(frame_logits.float(), dim=-1).cpu().numpy()

            all_probs.append(frame_probs)

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{n_frames}", end='\r')

    print()
    return np.concatenate(all_probs, axis=0)  # (1200*128, 1024)


def compress_segment(model, tokens_2d):
    """Compress a single segment using GPT + ANS.
    Returns compressed uint32 array.
    """
    n_frames, n_tokens = tokens_2d.shape
    assert n_tokens == 128

    probs = get_probabilities_for_segment(model, tokens_2d)
    symbols = tokens_2d.ravel().astype(np.int32)

    coder = constriction.stream.stack.AnsCoder()
    model_family = constriction.stream.model.Categorical(perfect=False)
    coder.encode_reverse(symbols, model_family, probs.astype(np.float32))

    return coder.get_compressed()  # uint32 array


def decompress_segment(model, compressed_data, n_frames=1200, n_tokens=128):
    """Decompress a single segment using GPT + ANS.

    Key insight: during encoding, the probabilities for frame F depend only on
    frames 0..F (via causal attention). During decoding, we process frame by frame:
    for frame F, we build context from already-decoded frames 0..F-1, plus the
    BOS token for frame F. The causal mask ensures logits for frame F's tokens
    only depend on prior tokens, so the probabilities match encoding exactly.
    """
    config = GPTConfig()
    coder = constriction.stream.stack.AnsCoder(compressed_data)
    model_family = constriction.stream.model.Categorical(perfect=False)

    decoded_frames = []

    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - 19)

            # Build context window from decoded frames + current BOS
            if frame_idx == 0:
                # Just BOS token
                seq_np = np.array([config.bos_token], dtype=np.int64)
            else:
                prev_tokens = np.array(decoded_frames[start_frame:frame_idx])  # (K, 128)
                prev_gpt = tokens_to_gpt_input(prev_tokens)  # (K, 129)
                # Add BOS for current frame
                seq_np = np.append(prev_gpt.ravel(), config.bos_token)

            seq = torch.tensor(seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
            logits = model(seq)  # (1, seq_len, vocab_size)

            # We need probs for 128 tokens in current frame.
            # The last 128 logits correspond to:
            #   logits[-128] predicts first data token (after BOS)
            #   logits[-127] predicts second data token
            #   ...
            #   logits[-1] predicts last data token
            # Wait, that's only true if we include the full current frame in the input.
            # But we only have BOS for the current frame, not its data tokens.
            #
            # Actually for the FIRST data token, logits[-1] (after BOS) predicts it.
            # For subsequent tokens, we need to feed them autoregressively.
            #
            # BUT: in the encoder, we fed the ENTIRE frame at once and used causal masking.
            # The causal mask means logits[t] only depends on tokens 0..t.
            # So logits at position (BOS of current frame) depends on all prior frames + BOS.
            # That gives us P(first_data_token | context, BOS).
            #
            # For P(second_data_token | context, BOS, first_data_token), we need to
            # feed first_data_token too. So we must decode autoregressively within each frame.

            frame_tokens = []
            for j in range(n_tokens):
                # Get prob distribution for next token
                frame_logits = logits[0, -1, :1024]  # (1024,)
                frame_probs = torch.softmax(frame_logits.float(), dim=-1).cpu().numpy()

                # Decode one token
                token = coder.decode(model_family, frame_probs.astype(np.float32).reshape(1, -1))
                frame_tokens.append(int(token[0]))

                if j < n_tokens - 1:
                    # Feed decoded token for next step
                    next_input = torch.tensor([[frame_tokens[-1]]], dtype=torch.long, device=DEVICE)
                    seq = torch.cat([seq, next_input], dim=1)
                    logits = model(seq)

            decoded_frames.append(frame_tokens)

            if frame_idx % 100 == 0:
                print(f"  Decode frame {frame_idx}/{n_frames}", end='\r')

    print()
    return np.array(decoded_frames, dtype=np.int16).reshape(-1, 8, 16)


def test_single_segment():
    """Test compression on a single segment to measure bits/token."""
    from datasets import load_dataset

    print("Loading model...")
    model = load_model()

    print("Loading data...")
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=2, data_files=data_files)

    tokens = np.array(ds['train'][0]['token.npy'])  # (1200, 8, 16)
    tokens_2d = tokens.reshape(1200, 128)

    print("Computing probabilities...")
    t0 = time.time()
    probs = get_probabilities_for_segment(model, tokens_2d)
    t1 = time.time()
    print(f"Probability computation: {t1-t0:.1f}s")

    # Measure cross-entropy (bits per token)
    symbols = tokens_2d.ravel().astype(np.int64)
    token_probs = probs[np.arange(len(symbols)), symbols]
    bits_per_token = -np.log2(token_probs + 1e-10).mean()
    print(f"Cross-entropy: {bits_per_token:.3f} bits/token")
    print(f"Theoretical compression ratio: {10/bits_per_token:.2f}x")

    # Actually compress with ANS
    print("Compressing with ANS...")
    compressed = compress_segment(model, tokens_2d)
    compressed_bytes = len(compressed) * 4
    original_bytes = 1200 * 128 * 10 / 8
    ratio = original_bytes / compressed_bytes
    print(f"Compressed: {compressed_bytes} bytes, Original: {int(original_bytes)} bytes")
    print(f"Actual compression ratio: {ratio:.2f}x")

    # Verify round-trip using probs (fast path)
    print("Verifying round-trip (fast, using stored probs)...")
    coder = constriction.stream.stack.AnsCoder(compressed)
    model_family = constriction.stream.model.Categorical(perfect=False)
    decoded = coder.decode(model_family, probs.astype(np.float32))
    assert np.all(decoded == symbols.astype(np.int32)), "Round-trip FAILED!"
    print("Round-trip OK!")


if __name__ == '__main__':
    test_single_segment()

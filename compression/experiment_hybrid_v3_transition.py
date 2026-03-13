#!/usr/bin/env python3
"""Test hybrid approach: v3 model probs mixed with transition table probs.

The idea is that the transition table captures long-range positional patterns
that the model misses, while the model captures temporal dynamics.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v3 import (
    TemporalModelV3, TemporalV3Config, CONFIGS_V3,
    NEIGHBOR_MAP_4, NEIGHBOR_MAP_8, ABOVE_MAP
)
from compression.temporal_v3_compress import get_probs_for_row

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_transition_tables(segments, n_positions=128, vocab_size=1024, smoothing=1.0):
    """Build per-position transition probability tables P(token_t | token_{t-1})."""
    tables = np.zeros((n_positions, vocab_size, vocab_size), dtype=np.float64)
    for seg in segments:
        for pos in range(n_positions):
            for t in range(1, seg.shape[0]):
                prev_token = seg[t-1, pos]
                curr_token = seg[t, pos]
                tables[pos, prev_token, curr_token] += 1

    # Add smoothing and normalize
    tables += smoothing
    tables /= tables.sum(axis=2, keepdims=True)
    return tables.astype(np.float32)


def build_above_tables(segments, n_positions=128, vocab_size=1024, smoothing=1.0):
    """Build per-position tables P(token | above_token_same_frame)."""
    tables = np.zeros((n_positions, vocab_size + 1, vocab_size), dtype=np.float64)
    for seg in segments:
        for pos in range(n_positions):
            above_pos = ABOVE_MAP[pos]
            for t in range(seg.shape[0]):
                if above_pos >= 0:
                    above_token = seg[t, above_pos]
                else:
                    above_token = vocab_size  # unavailable
                curr_token = seg[t, pos]
                tables[pos, above_token, curr_token] += 1

    tables += smoothing
    tables /= tables.sum(axis=2, keepdims=True)
    return tables.astype(np.float32)


def main():
    # Load model
    cp = torch.load('compression/temporal_v3_small.pt', map_location='cpu', weights_only=False)
    config = cp['config']
    model = TemporalModelV3(config).to(DEVICE)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4
    K = config.context_len

    # Load segments
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    segments = []
    for i, ex in enumerate(ds['train']):
        if i >= 500:
            break
        segments.append(np.array(ex['token.npy']).reshape(1200, 128))

    # Build transition tables from first 200 segments (held-out test on 200-210)
    print("Building transition tables from 200 segments...", flush=True)
    trans_tables = build_transition_tables(segments[:200])
    above_tables = build_above_tables(segments[:200])

    # Test on segments 200-210 (not used for tables)
    test_segs = segments[200:210]

    # Test different mixing strategies
    alphas = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]

    for alpha in alphas:
        total_nll = 0.0
        total_tokens = 0

        for seg in test_segs[:3]:  # 3 segments for speed
            for frame_idx in range(200):  # All frames including early ones
                decoded_frame = np.zeros(128, dtype=np.int16)
                for row in range(8):
                    # Model probs
                    model_probs = get_probs_for_row(
                        model, config, seg, frame_idx, row, decoded_frame, neighbor_map
                    )

                    for c in range(16):
                        pos = row * 16 + c
                        token = seg[frame_idx, pos]

                        # Transition table probs
                        if frame_idx > 0:
                            prev_token = seg[frame_idx - 1, pos]
                            t_probs = trans_tables[pos, prev_token]
                        else:
                            t_probs = np.ones(1024, dtype=np.float32) / 1024

                        # Above table probs
                        above_pos = ABOVE_MAP[pos]
                        if above_pos >= 0:
                            above_token = decoded_frame[above_pos]
                        else:
                            above_token = config.vocab_size
                        a_probs = above_tables[pos, above_token]

                        # Mix: model * alpha + tables * (1-alpha)
                        # Use geometric mean for tables
                        if alpha == 1.0:
                            final_probs = model_probs[c]
                        elif alpha == 0.0:
                            combined_table = (t_probs * a_probs)
                            combined_table /= combined_table.sum()
                            final_probs = combined_table
                        else:
                            combined_table = (t_probs * a_probs)
                            combined_table /= combined_table.sum()
                            final_probs = model_probs[c] ** alpha * combined_table ** (1 - alpha)
                            final_probs /= final_probs.sum()

                        p = max(final_probs[token], 1e-10)
                        total_nll -= math.log(p)
                        total_tokens += 1
                        decoded_frame[pos] = token

        bits = total_nll / total_tokens / math.log(2)
        print(f"alpha={alpha:.2f}: {bits:.3f} bits/token ({total_tokens} tokens)", flush=True)


if __name__ == '__main__':
    main()

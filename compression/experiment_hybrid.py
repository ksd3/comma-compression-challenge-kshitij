#!/usr/bin/env python3
"""Test hybrid neural + transition table with better transition tables."""
import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v2 import NEIGHBOR_MAP_4, NEIGHBOR_MAP_8
from compression.temporal_compress import load_model_compact, get_probs_for_frame

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_transition_tables(segments):
    marginal = np.zeros(1024, dtype=np.float64)
    transition = np.zeros((1024, 1024), dtype=np.float64)
    for seg in segments:
        for token in seg[0]:
            marginal[token] += 1
        for t in range(1, seg.shape[0]):
            for pos in range(128):
                prev = seg[t-1, pos]
                cur = seg[t, pos]
                transition[prev, cur] += 1
    marginal = (marginal + 1) / (marginal.sum() + 1024)
    for i in range(1024):
        row = transition[i]
        transition[i] = (row + 0.01) / (row.sum() + 0.01 * 1024)
    return marginal.astype(np.float32), transition.astype(np.float32)


def eval_bits(model, config, eval_segments, neighbor_map, marginal, transition, alpha):
    total_nll = 0.0
    total_tokens = 0
    for seg in eval_segments:
        for frame_idx in range(1, min(200, seg.shape[0])):
            neural_probs = get_probs_for_frame(model, config, seg, frame_idx, neighbor_map)
            # transition probs
            prev = seg[frame_idx - 1]
            trans_probs = transition[prev]  # (128, 1024)

            hybrid = alpha * neural_probs + (1 - alpha) * trans_probs
            hybrid = hybrid / hybrid.sum(axis=1, keepdims=True)

            targets = seg[frame_idx]
            for pos in range(128):
                p = max(hybrid[pos, targets[pos]], 1e-10)
                total_nll += -math.log(p)
                total_tokens += 1
    return total_nll / total_tokens / math.log(2)


def main():
    compact_path = os.path.join(HERE, 'temporal_v2_small_compact.bin')
    model, config = load_model_compact(compact_path)
    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4

    print("Loading data...")
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    # Use 500 segments for transition tables, 3 for eval
    all_segments = []
    for i, example in enumerate(ds['train']):
        if i >= 500:
            break
        all_segments.append(np.array(example['token.npy']).reshape(1200, 128))
    print(f"Loaded {len(all_segments)} segments")

    print("Building transition tables from 500 segments...")
    t0 = time.time()
    marginal, transition = build_transition_tables(all_segments)
    print(f"  Built in {time.time()-t0:.0f}s")

    # Eval on first 3 segments (200 frames each)
    eval_segs = all_segments[:3]
    print(f"Evaluating on {len(eval_segs)} segments x 200 frames...")

    # Test alphas
    for alpha in [1.0, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]:
        t0 = time.time()
        bits = eval_bits(model, config, eval_segs, neighbor_map, marginal, transition, alpha)
        elapsed = time.time() - t0
        print(f"  alpha={alpha:.2f}: {bits:.4f} bits/token ({elapsed:.0f}s)")

    # Also test: what if we use 2nd-order transition (prev 2 frames)?
    print("\n=== 2nd-order transition table ===")
    # P(token | prev_same_pos, prev2_same_pos) — 1024*1024*1024 is too large
    # Instead, try: P(token | prev_same_pos, prev_neighbor_avg)
    # Or: product of experts: P_neural * P_transition
    print("Testing product of experts (geometric mean)...")
    for alpha in [0.9, 0.85, 0.8, 0.7]:
        total_nll = 0.0
        total_tokens = 0
        for seg in eval_segs:
            for frame_idx in range(1, min(200, seg.shape[0])):
                neural_probs = get_probs_for_frame(model, config, seg, frame_idx, neighbor_map)
                prev = seg[frame_idx - 1]
                trans_probs = transition[prev]

                # Geometric mean (product of experts)
                log_hybrid = alpha * np.log(np.maximum(neural_probs, 1e-10)) + \
                             (1 - alpha) * np.log(np.maximum(trans_probs, 1e-10))
                hybrid = np.exp(log_hybrid)
                hybrid = hybrid / hybrid.sum(axis=1, keepdims=True)

                targets = seg[frame_idx]
                for pos in range(128):
                    p = max(hybrid[pos, targets[pos]], 1e-10)
                    total_nll += -math.log(p)
                    total_tokens += 1
        bits = total_nll / total_tokens / math.log(2)
        print(f"  geometric alpha={alpha:.2f}: {bits:.4f} bits/token")


if __name__ == '__main__':
    main()

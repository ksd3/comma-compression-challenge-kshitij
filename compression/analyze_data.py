#!/usr/bin/env python3
"""Analyze commaVQ token statistics to guide compression strategy."""
import os, sys, math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_segments(n_segments=100):
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    segments = []
    for i, example in enumerate(ds['train']):
        if i >= n_segments:
            break
        segments.append(np.array(example['token.npy']).reshape(1200, 128))
    return segments


def entropy(counts_or_probs):
    """Shannon entropy in bits."""
    p = np.array(counts_or_probs, dtype=np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return -np.sum(p * np.log2(p))


def analyze(segments):
    all_tokens = np.concatenate([s.ravel() for s in segments])
    n_total = len(all_tokens)
    print(f"=== Data Statistics ({len(segments)} segments, {n_total:,} tokens) ===\n")

    # --- 1. Zero-order entropy ---
    counts = np.bincount(all_tokens, minlength=1024)
    H0 = entropy(counts)
    print(f"1. ZERO-ORDER ENTROPY")
    print(f"   H0 = {H0:.4f} bits/token")
    print(f"   Theoretical limit at H0: {10/H0:.3f}x compression")
    n_used = np.sum(counts > 0)
    print(f"   Codebook usage: {n_used}/1024 ({100*n_used/1024:.1f}%)")
    top20 = np.argsort(counts)[::-1][:20]
    print(f"   Top 20 tokens: {list(top20)} (cover {100*counts[top20].sum()/n_total:.1f}%)")
    print(f"   Bottom 50% tokens cover: {100*np.sort(counts)[:n_used//2].sum()/n_total:.1f}% of data")
    print()

    # --- 2. Per-position entropy (spatial) ---
    # Each frame is 8x16; tokens are stored as 128-flat
    print(f"2. PER-POSITION ENTROPY (spatial structure)")
    pos_entropies = []
    for pos in range(128):
        pos_tokens = np.concatenate([s[:, pos] for s in segments])
        pos_counts = np.bincount(pos_tokens, minlength=1024)
        pos_entropies.append(entropy(pos_counts))
    pos_entropies = np.array(pos_entropies)
    print(f"   Mean per-position H = {pos_entropies.mean():.4f} bits")
    print(f"   Min/Max per-position H = {pos_entropies.min():.4f} / {pos_entropies.max():.4f}")
    # Reshape to 8x16 to see spatial pattern
    pos_ent_2d = pos_entropies.reshape(8, 16)
    print(f"   Entropy by row (8 rows):")
    for r in range(8):
        print(f"     Row {r}: {pos_ent_2d[r].mean():.3f} (min={pos_ent_2d[r].min():.3f}, max={pos_ent_2d[r].max():.3f})")
    print()

    # --- 3. Conditional entropy: H(token | prev_token_in_frame) ---
    print(f"3. SPATIAL CONDITIONAL ENTROPY")
    # H(t_j | t_{j-1}) within each frame (sequential)
    joint_counts = np.zeros((1024, 1024), dtype=np.int64)
    for s in segments:
        for j in range(1, 128):
            prev = s[:, j-1]
            cur = s[:, j]
            for p, c in zip(prev, cur):
                joint_counts[p, c] += 1
    # H(cur | prev) = H(prev, cur) - H(prev)
    H_joint = entropy(joint_counts.ravel())
    H_prev = entropy(joint_counts.sum(axis=1))
    H_cond_seq = H_joint - H_prev
    print(f"   H(t_j | t_{{j-1}}) sequential = {H_cond_seq:.4f} bits")
    print(f"   Mutual info I(t_j; t_{{j-1}}) = {H0 - H_cond_seq:.4f} bits")

    # H(token | left neighbor) in 8x16 grid
    joint_left = np.zeros((1024, 1024), dtype=np.int64)
    for s in segments:
        grid = s.reshape(-1, 8, 16)
        for row in range(8):
            for col in range(1, 16):
                prev = grid[:, row, col-1]
                cur = grid[:, row, col]
                for p, c in zip(prev, cur):
                    joint_left[p, c] += 1
    H_joint_l = entropy(joint_left.ravel())
    H_prev_l = entropy(joint_left.sum(axis=1))
    H_cond_left = H_joint_l - H_prev_l
    print(f"   H(t | left_neighbor) = {H_cond_left:.4f} bits")
    print(f"   MI with left = {H0 - H_cond_left:.4f} bits")

    # H(token | above neighbor) in 8x16 grid
    joint_above = np.zeros((1024, 1024), dtype=np.int64)
    for s in segments:
        grid = s.reshape(-1, 8, 16)
        for row in range(1, 8):
            for col in range(16):
                above = grid[:, row-1, col]
                cur = grid[:, row, col]
                for a, c in zip(above, cur):
                    joint_above[a, c] += 1
    H_joint_a = entropy(joint_above.ravel())
    H_prev_a = entropy(joint_above.sum(axis=1))
    H_cond_above = H_joint_a - H_prev_a
    print(f"   H(t | above_neighbor) = {H_cond_above:.4f} bits")
    print(f"   MI with above = {H0 - H_cond_above:.4f} bits")
    print()

    # --- 4. Temporal conditional entropy ---
    print(f"4. TEMPORAL CONDITIONAL ENTROPY")
    # H(token_t | same_position_t-1) across frames
    joint_temp = np.zeros((1024, 1024), dtype=np.int64)
    for s in segments:
        for pos in range(128):
            col = s[:, pos]
            for t in range(1, len(col)):
                joint_temp[col[t-1], col[t]] += 1
    H_joint_t = entropy(joint_temp.ravel())
    H_prev_t = entropy(joint_temp.sum(axis=1))
    H_cond_temp = H_joint_t - H_prev_t
    print(f"   H(t_pos | t_pos_prev_frame) = {H_cond_temp:.4f} bits")
    print(f"   Temporal MI (same pos) = {H0 - H_cond_temp:.4f} bits")

    # H(token_t | same_position, 2 frames back)
    joint_temp2 = np.zeros((1024, 1024), dtype=np.int64)
    for s in segments:
        for pos in range(128):
            col = s[:, pos]
            for t in range(2, len(col)):
                joint_temp2[col[t-2], col[t]] += 1
    H_joint_t2 = entropy(joint_temp2.ravel())
    H_prev_t2 = entropy(joint_temp2.sum(axis=1))
    H_cond_temp2 = H_joint_t2 - H_prev_t2
    print(f"   H(t_pos | t_pos_2_frames_back) = {H_cond_temp2:.4f} bits")
    print(f"   Temporal MI (2 frames) = {H0 - H_cond_temp2:.4f} bits")
    print()

    # --- 5. Run-length behavior ---
    print(f"5. RUN-LENGTH BEHAVIOR")
    # Temporal runs: how often does same position keep same value
    run_lengths = []
    for s in segments[:20]:  # limit for speed
        for pos in range(128):
            col = s[:, pos]
            rl = 1
            for t in range(1, len(col)):
                if col[t] == col[t-1]:
                    rl += 1
                else:
                    run_lengths.append(rl)
                    rl = 1
            run_lengths.append(rl)
    rl_arr = np.array(run_lengths)
    print(f"   Temporal run lengths (same pos across frames):")
    print(f"     Mean={rl_arr.mean():.2f}, Median={np.median(rl_arr):.0f}, Max={rl_arr.max()}")
    for thresh in [1, 2, 5, 10, 20, 50]:
        pct = 100 * np.sum(rl_arr >= thresh) / len(rl_arr)
        print(f"     Runs >= {thresh:3d}: {pct:.1f}%")

    # Spatial runs within frame (horizontal)
    spatial_runs = []
    for s in segments[:20]:
        grid = s.reshape(-1, 8, 16)
        for f in range(grid.shape[0]):
            for row in range(8):
                rl = 1
                for col in range(1, 16):
                    if grid[f, row, col] == grid[f, row, col-1]:
                        rl += 1
                    else:
                        spatial_runs.append(rl)
                        rl = 1
                spatial_runs.append(rl)
    sr_arr = np.array(spatial_runs)
    print(f"   Horizontal spatial run lengths:")
    print(f"     Mean={sr_arr.mean():.2f}, Median={np.median(sr_arr):.0f}, Max={sr_arr.max()}")
    print()

    # --- 6. Code usage sparsity ---
    print(f"6. CODE USAGE SPARSITY")
    # Per-segment codebook usage
    seg_usage = []
    for s in segments:
        seg_usage.append(len(np.unique(s)))
    seg_usage = np.array(seg_usage)
    print(f"   Unique tokens per segment: mean={seg_usage.mean():.0f}, min={seg_usage.min()}, max={seg_usage.max()}")

    # Per-frame codebook usage
    frame_usage = []
    for s in segments[:20]:
        for f in range(s.shape[0]):
            frame_usage.append(len(np.unique(s[f])))
    frame_usage = np.array(frame_usage)
    print(f"   Unique tokens per frame: mean={frame_usage.mean():.1f}, min={frame_usage.min()}, max={frame_usage.max()}")

    # Per-position across segments (how many unique values does each position take?)
    pos_unique = []
    for pos in range(128):
        all_pos = np.concatenate([s[:, pos] for s in segments])
        pos_unique.append(len(np.unique(all_pos)))
    pos_unique = np.array(pos_unique)
    print(f"   Unique tokens per position: mean={pos_unique.mean():.0f}, min={pos_unique.min()}, max={pos_unique.max()}")
    print()

    # --- 7. Delta coding potential ---
    print(f"7. DELTA CODING (temporal)")
    deltas = []
    zero_deltas = 0
    total_deltas = 0
    for s in segments[:20]:
        for t in range(1, s.shape[0]):
            d = s[t].astype(np.int32) - s[t-1].astype(np.int32)
            deltas.extend(d.tolist())
            zero_deltas += np.sum(d == 0)
            total_deltas += len(d)
    delta_arr = np.array(deltas)
    H_delta = entropy(np.bincount(delta_arr - delta_arr.min()))
    print(f"   H(delta) = {H_delta:.4f} bits")
    print(f"   Zero deltas: {100*zero_deltas/total_deltas:.1f}%")
    print(f"   Delta range: [{delta_arr.min()}, {delta_arr.max()}]")
    print(f"   |delta| <= 10: {100*np.sum(np.abs(delta_arr) <= 10)/len(delta_arr):.1f}%")
    print(f"   |delta| <= 50: {100*np.sum(np.abs(delta_arr) <= 50)/len(delta_arr):.1f}%")
    print()

    # --- 8. Theoretical ceiling estimate ---
    print(f"8. THEORETICAL CEILING ESTIMATE")
    print(f"   Raw: 10 bits/token")
    print(f"   H0 (zero-order): {H0:.3f} bits → {10/H0:.3f}x")
    print(f"   H(t|left): {H_cond_left:.3f} bits → {10/H_cond_left:.3f}x")
    print(f"   H(t|above): {H_cond_above:.3f} bits → {10/H_cond_above:.3f}x")
    print(f"   H(t|prev_frame): {H_cond_temp:.3f} bits → {10/H_cond_temp:.3f}x")
    print(f"   H(delta): {H_delta:.3f} bits → {10/H_delta:.3f}x")
    # Combined: H(t | left, above, prev_frame) — estimated
    print(f"   (Combined spatial+temporal conditioning would be even lower)")
    print()

    # --- 9. Cross-segment vs within-segment ---
    print(f"9. CROSS-SEGMENT VARIATION")
    seg_entropies = []
    for s in segments:
        c = np.bincount(s.ravel(), minlength=1024)
        seg_entropies.append(entropy(c))
    seg_entropies = np.array(seg_entropies)
    print(f"   Per-segment H0: mean={seg_entropies.mean():.3f}, std={seg_entropies.std():.3f}")
    print(f"   Range: [{seg_entropies.min():.3f}, {seg_entropies.max():.3f}]")
    avg_seg_H = seg_entropies.mean()
    print(f"   Global H0={H0:.3f} vs avg segment H0={avg_seg_H:.3f}")
    print(f"   Cross-segment overhead: {H0 - avg_seg_H:.3f} bits")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, help='Number of segments to analyze')
    args = parser.parse_args()

    print("Loading data...", flush=True)
    segments = load_segments(args.n)
    print(f"Loaded {len(segments)} segments\n", flush=True)
    analyze(segments)

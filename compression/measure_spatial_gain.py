#!/usr/bin/env python3
"""Measure how much spatial context from CURRENT frame could help.

For raster-order decoding of frame t:
- We know all prev frames
- We know positions 0..p-1 of current frame
- How much does knowing the current-frame neighbors reduce entropy?
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("Loading data...")
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    segments = []
    for i, example in enumerate(ds['train']):
        if i >= 100:
            break
        segments.append(np.array(example['token.npy']).reshape(1200, 128))

    # For raster order in 8x16 grid:
    # pos = r*16 + c
    # Available neighbors when decoding pos p in raster order:
    # - left (same row, c-1): available if c > 0
    # - above (prev row, same col): available if r > 0
    # - above-left: available if r > 0 and c > 0
    # - above-right: available if r > 0 and c < 15
    # Previous frame: all neighbors available

    # Measure: H(token_t_p | token_{t-1}_p) vs H(token_t_p | token_{t-1}_p, token_t_{left}, token_t_{above})
    # This tells us how much spatial context from the current frame helps

    # 1) H(token | prev_frame_same_pos)
    joint_prev = np.zeros((1024, 1024), dtype=np.float64)
    for seg in segments:
        for t in range(1, seg.shape[0]):
            for pos in range(128):
                prev = seg[t-1, pos]
                cur = seg[t, pos]
                joint_prev[prev, cur] += 1

    # H(cur|prev)
    h_cond_prev = 0.0
    total = joint_prev.sum()
    for i in range(1024):
        row_sum = joint_prev[i].sum()
        if row_sum == 0:
            continue
        for j in range(1024):
            if joint_prev[i, j] > 0:
                p_joint = joint_prev[i, j] / total
                p_cond = joint_prev[i, j] / row_sum
                h_cond_prev -= p_joint * math.log2(p_cond)
    print(f"H(token | prev_frame_same_pos) = {h_cond_prev:.4f} bits")

    # 2) H(token | prev_frame_same_pos, current_frame_left)
    # For interior positions (c > 0):
    # Joint: (prev, left, cur) — 1024^3 too large
    # Instead, compute conditional entropy empirically

    # More practical: just measure correlation
    # H(token | left_in_current_frame)
    joint_left = np.zeros((1024, 1024), dtype=np.float64)
    count = 0
    for seg in segments:
        for t in range(seg.shape[0]):
            for r in range(8):
                for c in range(1, 16):  # skip c=0
                    pos = r * 16 + c
                    left_pos = r * 16 + (c - 1)
                    left = seg[t, left_pos]
                    cur = seg[t, pos]
                    joint_left[left, cur] += 1
                    count += 1

    h_cond_left = 0.0
    total = joint_left.sum()
    for i in range(1024):
        row_sum = joint_left[i].sum()
        if row_sum == 0:
            continue
        for j in range(1024):
            if joint_left[i, j] > 0:
                p_joint = joint_left[i, j] / total
                p_cond = joint_left[i, j] / row_sum
                h_cond_left -= p_joint * math.log2(p_cond)
    print(f"H(token | left_same_frame) = {h_cond_left:.4f} bits")

    # H(token | above_in_current_frame)
    joint_above = np.zeros((1024, 1024), dtype=np.float64)
    for seg in segments:
        for t in range(seg.shape[0]):
            for r in range(1, 8):  # skip r=0
                for c in range(16):
                    pos = r * 16 + c
                    above_pos = (r - 1) * 16 + c
                    above = seg[t, above_pos]
                    cur = seg[t, pos]
                    joint_above[above, cur] += 1

    h_cond_above = 0.0
    total = joint_above.sum()
    for i in range(1024):
        row_sum = joint_above[i].sum()
        if row_sum == 0:
            continue
        for j in range(1024):
            if joint_above[i, j] > 0:
                p_joint = joint_above[i, j] / total
                p_cond = joint_above[i, j] / row_sum
                h_cond_above -= p_joint * math.log2(p_cond)
    print(f"H(token | above_same_frame) = {h_cond_above:.4f} bits")

    # Marginal entropy
    marginal = np.zeros(1024, dtype=np.float64)
    for seg in segments:
        for token in seg.ravel():
            marginal[token] += 1
    marginal = marginal / marginal.sum()
    h0 = -sum(p * math.log2(p) for p in marginal if p > 0)
    print(f"\nH(token) = {h0:.4f} bits")
    print(f"MI(token; prev_frame_same_pos) = {h0 - h_cond_prev:.4f} bits")
    print(f"MI(token; left_same_frame) = {h0 - h_cond_left:.4f} bits")
    print(f"MI(token; above_same_frame) = {h0 - h_cond_above:.4f} bits")

    # The question: how much ADDITIONAL info does left/above give beyond prev_frame?
    # Need H(token | prev, left) which requires 1024^3 table...
    # Let's sample it empirically instead
    print("\n=== Additional gain from current-frame neighbors ===")
    print("Sampling H(token | prev_frame, left_same_frame)...")

    # Use a dict to count (prev, left) -> cur distribution
    from collections import defaultdict
    counts = defaultdict(lambda: np.zeros(1024, dtype=np.float64))
    for seg in segments[:20]:  # Use fewer to keep memory manageable
        for t in range(1, seg.shape[0]):
            for r in range(8):
                for c in range(1, 16):
                    pos = r * 16 + c
                    left_pos = r * 16 + (c - 1)
                    prev = seg[t-1, pos]
                    left = seg[t, left_pos]
                    cur = seg[t, pos]
                    counts[(prev, left)][cur] += 1

    # Compute H(token | prev, left)
    total_nll = 0.0
    total_n = 0
    for key, dist in counts.items():
        n = dist.sum()
        if n == 0:
            continue
        probs = dist / n
        for j in range(1024):
            if probs[j] > 0:
                total_nll -= dist[j] * math.log2(probs[j])
        total_n += n
    h_cond_prev_left = total_nll / total_n
    print(f"H(token | prev_frame, left_same_frame) = {h_cond_prev_left:.4f} bits")
    print(f"Additional info from left: {h_cond_prev - h_cond_prev_left:.4f} bits")

    # Similarly for above
    print("Sampling H(token | prev_frame, above_same_frame)...")
    counts2 = defaultdict(lambda: np.zeros(1024, dtype=np.float64))
    for seg in segments[:20]:
        for t in range(1, seg.shape[0]):
            for r in range(1, 8):
                for c in range(16):
                    pos = r * 16 + c
                    above_pos = (r - 1) * 16 + c
                    prev = seg[t-1, pos]
                    above = seg[t, above_pos]
                    cur = seg[t, pos]
                    counts2[(prev, above)][cur] += 1

    total_nll = 0.0
    total_n = 0
    for key, dist in counts2.items():
        n = dist.sum()
        if n == 0:
            continue
        probs = dist / n
        for j in range(1024):
            if probs[j] > 0:
                total_nll -= dist[j] * math.log2(probs[j])
        total_n += n
    h_cond_prev_above = total_nll / total_n
    print(f"H(token | prev_frame, above_same_frame) = {h_cond_prev_above:.4f} bits")
    print(f"Additional info from above: {h_cond_prev - h_cond_prev_above:.4f} bits")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Test what context length gives the best bits/token on eval."""
import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v2 import TemporalModelV2, TemporalV2Config, NEIGHBOR_MAP_4

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_context_length(segments, K_values):
    """Measure empirical conditional entropy H(token|prev K tokens at same pos)."""
    print(f"Testing context lengths: {K_values}")
    print(f"Using {len(segments)} segments")

    for K in K_values:
        # Build frequency table: count occurrences of each token given context
        # For small K, we can do exact counting
        # For large K, we need a model — so let's just measure with the trained model
        # Instead, let's measure: for each (prev_token, cur_token) pair at distance d,
        # what's the mutual information?

        # Simple approach: for each distance d from 1 to K, compute H(token|prev_at_distance_d)
        pass

    # Actually, let's measure how much information each additional frame adds
    # by computing MI(token_t; token_{t-d}) for d=1,2,...,30
    print("\n=== Mutual Information by Distance ===")
    print("MI(token_t; token_{t-d}) for each distance d")

    max_d = 40
    for d in range(1, max_d + 1):
        # Build joint distribution
        joint = np.zeros((1024, 1024), dtype=np.float64)
        for seg in segments:
            for t in range(d, seg.shape[0]):
                for pos in range(128):
                    prev = seg[t-d, pos]
                    cur = seg[t, pos]
                    joint[prev, cur] += 1

        # Normalize
        joint = joint / joint.sum()
        marginal_prev = joint.sum(axis=1)
        marginal_cur = joint.sum(axis=0)

        # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for i in range(1024):
            for j in range(1024):
                if joint[i, j] > 0 and marginal_prev[i] > 0 and marginal_cur[j] > 0:
                    mi += joint[i, j] * math.log2(joint[i, j] / (marginal_prev[i] * marginal_cur[j]))

        print(f"  d={d:2d}: MI={mi:.4f} bits")

        if d == 5:
            # Compute cumulative info: H(token) - H(token|prev_1,...,prev_5)
            pass


def main():
    print("Loading data...")
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    segments = []
    for i, example in enumerate(ds['train']):
        if i >= 20:  # Just 20 segments for fast computation
            break
        segments.append(np.array(example['token.npy']).reshape(1200, 128))

    eval_context_length(segments, [5, 10, 15, 20, 30, 40])


if __name__ == '__main__':
    main()

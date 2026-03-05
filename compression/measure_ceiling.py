#!/usr/bin/env python3
"""Measure conditional entropy H(token | prev_1,...,prev_K) at same spatial position.

This tells us the theoretical ceiling for per-position temporal models.
"""
import sys, os, math, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_segments(n_segments=500):
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


def compute_conditional_entropy(segments, K, n_positions=128):
    """Compute H(token | prev_1, ..., prev_K) at same spatial position.

    Uses empirical counting with tuple contexts.
    """
    # For each context tuple, count the next-token distribution
    context_counts = defaultdict(lambda: np.zeros(1024, dtype=np.int32))
    total_tokens = 0

    for seg in segments:
        for pos in range(n_positions):
            col = seg[:, pos]  # temporal sequence at this position
            for t in range(K, len(col)):
                context = tuple(col[t-K:t].tolist())
                context_counts[context][col[t]] += 1
                total_tokens += 1

    # H(token | context) = sum_context P(context) * H(token | context=c)
    total_nll = 0.0
    for context, counts in context_counts.items():
        n = counts.sum()
        if n == 0:
            continue
        p = counts / n
        p = p[p > 0]
        h = -np.sum(p * np.log2(p))
        total_nll += n * h

    return total_nll / total_tokens, len(context_counts), total_tokens


def compute_conditional_entropy_fast(segments, K, n_positions=128):
    """Faster version using numpy vectorization for small K."""
    if K > 3:
        # For large K, use hash-based approach
        return compute_conditional_entropy_hashed(segments, K, n_positions)
    return compute_conditional_entropy(segments, K, n_positions)


def compute_conditional_entropy_hashed(segments, K, n_positions=128):
    """Hash-based conditional entropy for large K.

    Instead of storing full tuple keys, hash them to reduce memory.
    But this introduces hash collisions. Use a large hash space.

    Actually, let's just use a dict with tuple keys but limit the data
    to keep memory reasonable.
    """
    context_counts = defaultdict(lambda: defaultdict(int))
    total_tokens = 0

    for seg in segments:
        for pos in range(n_positions):
            col = seg[:, pos]
            for t in range(K, len(col)):
                context = tuple(col[t-K:t].tolist())
                context_counts[context][col[t]] += 1
                total_tokens += 1

    # Compute conditional entropy
    total_nll = 0.0
    n_contexts = len(context_counts)
    for context, token_counts in context_counts.items():
        n = sum(token_counts.values())
        h = 0.0
        for c in token_counts.values():
            p = c / n
            h -= p * math.log2(p)
        total_nll += n * h

    return total_nll / total_tokens, n_contexts, total_tokens


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, help='Number of segments')
    args = parser.parse_args()

    print("Loading data...", flush=True)
    segments = load_segments(args.n)
    print(f"Loaded {len(segments)} segments\n", flush=True)

    print(f"{'K':>3} | {'H(t|ctx)':>10} | {'Compress':>10} | {'Contexts':>12} | {'Tokens':>12} | {'Time':>6}")
    print("-" * 70)

    for K in [0, 1, 2, 3, 5, 10, 20]:
        t0 = time.time()
        if K == 0:
            # Marginal entropy
            all_tokens = np.concatenate([s.ravel() for s in segments])
            counts = np.bincount(all_tokens, minlength=1024)
            p = counts / counts.sum()
            p = p[p > 0]
            h = -np.sum(p * np.log2(p))
            n_ctx = 1
            n_tok = len(all_tokens)
        else:
            h, n_ctx, n_tok = compute_conditional_entropy_hashed(segments, K)
        elapsed = time.time() - t0
        rate = 10.0 / h if h > 0 else float('inf')
        print(f"{K:>3} | {h:>10.4f} | {rate:>9.3f}x | {n_ctx:>12,} | {n_tok:>12,} | {elapsed:>5.0f}s", flush=True)


if __name__ == '__main__':
    main()

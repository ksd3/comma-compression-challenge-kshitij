#!/usr/bin/env python3
"""Compress using per-position conditional probability tables + ANS.

For each position p, build P(token_t | token_{t-1}, above_token_t) where:
- token_{t-1} is the previous frame's token at the same position
- above_token_t is the current frame's token directly above (row-1, same col)

For row 0 positions (no above), use P(token_t | token_{t-1}).
For frame 0 (no previous), use P(token_t | above_token_t) or uniform.

Tables are built from the FULL dataset (5000 segments) since both encoder
and decoder have access to the same data for training.

Size analysis:
- Per position: 1024 × 1025 × (stored as counts, then normalized on the fly)
- We only need to store the tables. With LZMA compression, sparse tables compress well.
"""
import os, sys, io, struct, time, math, lzma, zipfile, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

HERE = os.path.dirname(os.path.abspath(__file__))

# Above map: position -> position above (-1 if row 0)
def build_above_map(rows=8, cols=16):
    above_map = np.full(rows * cols, -1, dtype=np.int32)
    for r in range(1, rows):
        for c in range(cols):
            above_map[r * cols + c] = (r - 1) * cols + c
    return above_map

ABOVE_MAP = build_above_map()


def build_tables(segments, smoothing=0.5):
    """Build conditional probability tables from data.

    Returns:
        tables: (128, 1024, 1025, 1024) is too big.
        Instead, build tables as:
          - prev_above_tables: (128, 1024, 1025) -> distributions over 1024
            where dim 1 = prev_token (0..1023), dim 2 = above_token (0..1024, 1024=unavailable)
        This is 128 * 1024 * 1025 * 1024 * 4 bytes = 549GB. Way too big.

        Need a different approach. Use factored tables:
        - prev_table: (128, 1024) -> P(token | prev_token, position) -- 1024 distributions
        - above_table: (128, 1025) -> P(token | above_token, position) -- 1025 distributions
        Then combine: P(token | prev, above) ∝ P(token|prev) * P(token|above) / P(token)
    """
    n_pos = 128
    vocab = 1024

    # Count tables
    prev_counts = np.zeros((n_pos, vocab, vocab), dtype=np.float64)
    above_counts = np.zeros((n_pos, vocab + 1, vocab), dtype=np.float64)
    marginal_counts = np.zeros((n_pos, vocab), dtype=np.float64)

    for seg in segments:
        for pos in range(n_pos):
            above_pos = ABOVE_MAP[pos]
            for t in range(seg.shape[0]):
                token = seg[t, pos]
                marginal_counts[pos, token] += 1

                if t > 0:
                    prev_token = seg[t-1, pos]
                    prev_counts[pos, prev_token, token] += 1

                if above_pos >= 0:
                    above_token = seg[t, above_pos]
                else:
                    above_token = vocab
                above_counts[pos, above_token, token] += 1

    # Normalize with smoothing
    prev_probs = prev_counts + smoothing
    prev_probs /= prev_probs.sum(axis=2, keepdims=True)

    above_probs = above_counts + smoothing
    above_probs /= above_probs.sum(axis=2, keepdims=True)

    marginal_probs = marginal_counts + smoothing
    marginal_probs /= marginal_probs.sum(axis=1, keepdims=True)

    return prev_probs.astype(np.float32), above_probs.astype(np.float32), marginal_probs.astype(np.float32)


def get_combined_probs(prev_probs, above_probs, marginal_probs, pos, prev_token, above_token):
    """Combine prev and above probs using product of experts / marginal."""
    p_prev = prev_probs[pos, prev_token] if prev_token >= 0 else np.ones(1024, dtype=np.float32) / 1024
    p_above = above_probs[pos, above_token]
    p_marginal = marginal_probs[pos]

    # Product of experts: P(t|prev,above) ∝ P(t|prev) * P(t|above) / P(t)
    combined = p_prev * p_above / np.maximum(p_marginal, 1e-10)
    combined = np.maximum(combined, 1e-10)
    combined /= combined.sum()
    return combined


def compress_segment(prev_probs, above_probs, marginal_probs, tokens_2d):
    """Compress one segment using table-based predictions + ANS."""
    import constriction

    n_frames = tokens_2d.shape[0]
    n_pos = tokens_2d.shape[1]
    all_probs = []

    for frame_idx in range(n_frames):
        decoded_frame = np.zeros(128, dtype=np.int16)
        for row in range(8):
            for c in range(16):
                pos = row * 16 + c
                above_pos = ABOVE_MAP[pos]

                if frame_idx > 0:
                    prev_token = tokens_2d[frame_idx - 1, pos]
                else:
                    prev_token = -1  # no previous

                if above_pos >= 0 and row > 0:
                    above_token = decoded_frame[above_pos]
                else:
                    above_token = 1024  # unavailable

                probs = get_combined_probs(prev_probs, above_probs, marginal_probs,
                                          pos, prev_token, above_token)
                all_probs.append(probs)
                decoded_frame[pos] = tokens_2d[frame_idx, pos]

    all_probs = np.array(all_probs, dtype=np.float32)

    symbols = []
    for frame_idx in range(n_frames):
        for row in range(8):
            for c in range(16):
                symbols.append(tokens_2d[frame_idx, row * 16 + c])
    symbols = np.array(symbols, dtype=np.int32)

    coder = constriction.stream.stack.AnsCoder()
    model_family = constriction.stream.model.Categorical(perfect=False)
    coder.encode_reverse(symbols, model_family, all_probs)

    return coder.get_compressed().tobytes()


def measure_bits(prev_probs, above_probs, marginal_probs, tokens_2d):
    """Measure bits/token without actual ANS coding."""
    n_frames = tokens_2d.shape[0]
    total_nll = 0.0
    total_tokens = 0

    for frame_idx in range(n_frames):
        decoded_frame = np.zeros(128, dtype=np.int16)
        for row in range(8):
            for c in range(16):
                pos = row * 16 + c
                above_pos = ABOVE_MAP[pos]

                if frame_idx > 0:
                    prev_token = tokens_2d[frame_idx - 1, pos]
                else:
                    prev_token = -1

                if above_pos >= 0 and row > 0:
                    above_token = decoded_frame[above_pos]
                else:
                    above_token = 1024

                probs = get_combined_probs(prev_probs, above_probs, marginal_probs,
                                          pos, prev_token, above_token)
                token = tokens_2d[frame_idx, pos]
                p = max(probs[token], 1e-10)
                total_nll -= math.log(p)
                total_tokens += 1
                decoded_frame[pos] = token

    return total_nll / total_tokens / math.log(2)


def main():
    from datasets import load_dataset
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-segments", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--smoothing", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading dataset...", flush=True)
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    segments = []
    for i, ex in enumerate(ds['train']):
        if i >= args.n_segments:
            break
        segments.append(np.array(ex['token.npy']).reshape(1200, 128))
    print(f"Loaded {len(segments)} segments", flush=True)

    print("Building tables...", flush=True)
    prev_probs, above_probs, marginal_probs = build_tables(segments, smoothing=args.smoothing)

    # Table size
    table_bytes = prev_probs.nbytes + above_probs.nbytes + marginal_probs.nbytes
    print(f"Table size (raw): {table_bytes/1024/1024:.1f} MB", flush=True)

    # Compress tables with LZMA
    table_buf = io.BytesIO()
    # Quantize to uint16 for better compression
    for tbl in [prev_probs, above_probs, marginal_probs]:
        # Convert to log2 scale, quantize to uint16
        log_probs = np.log2(np.maximum(tbl, 1e-10))
        # Scale to uint16 range
        min_val = log_probs.min()
        max_val = 0.0
        scale = 65535.0 / (max_val - min_val) if max_val > min_val else 1.0
        quantized = ((log_probs - min_val) * scale).clip(0, 65535).astype(np.uint16)
        table_buf.write(struct.pack('<ff', min_val, scale))
        table_buf.write(quantized.tobytes())

    compressed_tables = lzma.compress(table_buf.getvalue(), preset=9)
    print(f"Table size (LZMA): {len(compressed_tables)/1024/1024:.1f} MB", flush=True)

    # Measure bits/token on test segments
    print(f"\nMeasuring bits/token on {args.n_test} segments (first 100 frames)...", flush=True)
    total_nll = 0.0
    total_tokens = 0

    for i in range(args.n_test):
        seg = segments[i][:100]  # First 100 frames for speed
        decoded_frame = np.zeros(128, dtype=np.int16)
        seg_nll = 0.0
        seg_tokens = 0

        for frame_idx in range(seg.shape[0]):
            decoded_frame[:] = 0
            for row in range(8):
                for c in range(16):
                    pos = row * 16 + c
                    above_pos = ABOVE_MAP[pos]

                    if frame_idx > 0:
                        prev_token = segments[i][frame_idx - 1, pos]
                    else:
                        prev_token = -1

                    if above_pos >= 0 and row > 0:
                        above_token = decoded_frame[above_pos]
                    else:
                        above_token = 1024

                    probs = get_combined_probs(prev_probs, above_probs, marginal_probs,
                                              pos, prev_token, above_token)
                    token = seg[frame_idx, pos]
                    p = max(probs[token], 1e-10)
                    seg_nll -= math.log(p)
                    seg_tokens += 1
                    decoded_frame[pos] = token

        bits = seg_nll / seg_tokens / math.log(2)
        total_nll += seg_nll
        total_tokens += seg_tokens
        print(f"  Segment {i}: {bits:.3f} bits/token", flush=True)

    overall_bits = total_nll / total_tokens / math.log(2)
    print(f"\nOverall: {overall_bits:.3f} bits/token", flush=True)

    # Project compression rate
    data_bits = 5000 * 1200 * 128 * overall_bits
    data_bytes = data_bits / 8
    total_size = data_bytes + len(compressed_tables)
    original_size = 5000 * 1200 * 128 * 10 / 8
    rate = original_size / total_size
    print(f"Table size: {len(compressed_tables)/1024/1024:.1f} MB", flush=True)
    print(f"Projected data: {data_bytes/1024/1024:.1f} MB", flush=True)
    print(f"Projected total: {total_size/1024/1024:.1f} MB", flush=True)
    print(f"Projected rate: {rate:.3f}x", flush=True)


if __name__ == '__main__':
    main()

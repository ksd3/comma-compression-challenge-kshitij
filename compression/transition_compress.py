#!/usr/bin/env python3
"""Transition table compression: P(token | prev_frame_same_pos) + ANS.

This is the simplest possible approach with no neural network.
Model: 1024x1024 conditional probability table + 1024 marginal (for frame 0).
"""
import os, sys, io, struct, time, math, json, lzma, zipfile, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

HERE = os.path.dirname(os.path.abspath(__file__))


def build_tables(segments):
    """Build transition table from data."""
    # Marginal distribution (for frame 0)
    marginal = np.zeros(1024, dtype=np.float64)
    # Conditional distribution P(cur | prev) at same position
    transition = np.zeros((1024, 1024), dtype=np.float64)

    for seg in segments:
        # Frame 0: marginal
        for token in seg[0]:
            marginal[token] += 1
        # Frames 1-1199: conditional on previous frame same position
        for t in range(1, seg.shape[0]):
            for pos in range(128):
                prev = seg[t-1, pos]
                cur = seg[t, pos]
                transition[prev, cur] += 1

    # Normalize with Laplace smoothing to avoid zeros
    marginal = (marginal + 1) / (marginal.sum() + 1024)
    for i in range(1024):
        row = transition[i]
        transition[i] = (row + 0.01) / (row.sum() + 0.01 * 1024)

    return marginal, transition


def save_tables(marginal, transition, path):
    """Save tables as LZMA-compressed binary."""
    buf = io.BytesIO()
    # Quantize to uint16 for compact storage
    # Store as log-probabilities quantized to 16 bits
    marginal_q = (marginal * 65535).clip(1, 65535).astype(np.uint16)
    transition_q = (transition * 65535).clip(1, 65535).astype(np.uint16)
    buf.write(marginal_q.tobytes())
    buf.write(transition_q.tobytes())
    compressed = lzma.compress(buf.getvalue(), preset=9)
    with open(path, 'wb') as f:
        f.write(compressed)
    return len(compressed)


def load_tables(path):
    """Load tables from LZMA-compressed binary."""
    with open(path, 'rb') as f:
        raw = lzma.decompress(f.read())
    offset = 0
    marginal_q = np.frombuffer(raw[offset:offset + 1024*2], dtype=np.uint16).copy()
    offset += 1024 * 2
    transition_q = np.frombuffer(raw[offset:offset + 1024*1024*2], dtype=np.uint16).copy()
    # Normalize back to probabilities (all float32)
    marginal = marginal_q.astype(np.float32)
    marginal = marginal / marginal.sum()
    transition = transition_q.reshape(1024, 1024).astype(np.float32)
    transition = transition / transition.sum(axis=1, keepdims=True)
    return marginal, transition


def compress_segment(marginal, transition, tokens_2d):
    """Compress one segment using transition table + ANS."""
    import constriction
    n_frames, n_pos = tokens_2d.shape
    symbols = []
    probs = []

    for t in range(n_frames):
        for pos in range(n_pos):
            token = tokens_2d[t, pos]
            if t == 0:
                probs.append(marginal)
            else:
                prev = tokens_2d[t-1, pos]
                probs.append(transition[prev])
            symbols.append(token)

    symbols = np.array(symbols, dtype=np.int32)
    probs = np.array(probs, dtype=np.float32)

    coder = constriction.stream.stack.AnsCoder()
    model_family = constriction.stream.model.Categorical(perfect=False)
    coder.encode_reverse(symbols, model_family, probs)
    return coder.get_compressed().tobytes()


def decompress_segment(marginal, transition, compressed_bytes):
    """Decompress one segment."""
    import constriction
    compressed_u32 = np.frombuffer(compressed_bytes, dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed_u32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    n_frames, n_pos = 1200, 128
    tokens = np.zeros((n_frames, n_pos), dtype=np.int16)

    for t in range(n_frames):
        for pos in range(n_pos):
            if t == 0:
                p = marginal.astype(np.float32).reshape(1, -1)
            else:
                prev = int(tokens[t-1, pos])
                p = transition[prev].astype(np.float32).reshape(1, -1)
            token = coder.decode(model_family, p)
            tokens[t, pos] = token[0]

    return tokens.reshape(-1, 8, 16)


def compress_all(args):
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    examples = list(ds['train'])
    if args.quick:
        examples = examples[:100]

    # Build tables from all data
    print("Building transition tables...", flush=True)
    segments = [np.array(ex['token.npy']).reshape(1200, 128) for ex in examples]
    marginal, transition = build_tables(segments)

    # Save tables
    table_path = os.path.join(HERE, 'transition_tables.bin')
    table_size = save_tables(marginal, transition, table_path)
    print(f"Table size: {table_size:,} bytes ({table_size/1024:.0f} KB)", flush=True)

    # CRITICAL: reload from file so encoder uses identical quantized tables as decoder
    marginal, transition = load_tables(table_path)
    print("Loaded quantized tables for encoding", flush=True)

    # Compress
    print(f"Compressing {len(segments)} segments...", flush=True)
    t0 = time.time()
    names = []
    blobs = []
    total_bits = 0
    total_tokens = 0

    for i, (example, seg) in enumerate(zip(examples, segments)):
        name = example['json']['file_name']
        compressed = compress_segment(marginal, transition, seg)
        names.append(name)
        blobs.append(compressed)
        total_bits += len(compressed) * 8
        total_tokens += seg.size

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_bits = total_bits / total_tokens
            print(f"  [{i+1}/{len(segments)}] {avg_bits:.3f} bits/token, "
                  f"{elapsed:.0f}s ({elapsed/(i+1):.1f}s/seg)", flush=True)

    compress_time = time.time() - t0
    avg_bits = total_bits / total_tokens

    # Verify round-trip on first segment
    if not args.skip_verify:
        print("Verifying round-trip...", flush=True)
        m2, t2 = load_tables(table_path)
        decoded = decompress_segment(m2, t2, blobs[0])
        original = segments[0].reshape(-1, 8, 16)
        if np.array_equal(decoded, original):
            print("  Round-trip OK!", flush=True)
        else:
            match = (decoded == original).sum()
            total = original.size
            print(f"  MISMATCH: {match}/{total} tokens match", flush=True)

    # Build zip
    print("Building zip...", flush=True)
    backend_name = "transition"
    backend_bytes = backend_name.encode('utf-8')
    header = struct.pack('<B', len(backend_bytes)) + backend_bytes
    header += struct.pack('<I', len(names))

    name_table = b''
    for name in names:
        nb = name.encode('utf-8')
        name_table += struct.pack('<H', len(nb)) + nb

    size_table = b''
    for blob in blobs:
        size_table += struct.pack('<I', len(blob))

    data_blob = header + name_table + size_table + b''.join(blobs)

    zip_path = os.path.join(HERE, 'transition_submission.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('data.bin', data_blob)
        zf.write(table_path, 'tables.bin')
        # Would also need decompress.py here

    zip_size = os.path.getsize(zip_path)
    data_size = sum(len(b) for b in blobs)
    original_size = total_tokens * 10 / 8

    print(f"\n{'='*50}", flush=True)
    print(f"Table size:       {table_size:>12,} bytes ({table_size/1024:.0f} KB)", flush=True)
    print(f"Compressed data:  {data_size:>12,} bytes ({data_size/1024/1024:.1f} MB)", flush=True)
    print(f"Zip file size:    {zip_size:>12,} bytes ({zip_size/1024/1024:.1f} MB)", flush=True)
    print(f"Original size:    {int(original_size):>12,} bytes ({original_size/1024/1024:.1f} MB)", flush=True)
    print(f"Compression rate: {original_size/zip_size:.3f}x", flush=True)
    print(f"Bits/token:       {avg_bits:.3f}", flush=True)
    print(f"Compress time:    {compress_time:.0f}s", flush=True)
    if len(segments) < 5000:
        proj_data = data_size * (5000 / len(segments))
        proj_total = proj_data + table_size + 1000
        proj_rate = (5000 * 1200 * 128 * 10 / 8) / proj_total
        print(f"Projected 5k:     {proj_rate:.3f}x", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()
    compress_all(args)

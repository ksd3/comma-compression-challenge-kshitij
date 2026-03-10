#!/usr/bin/env python3
"""Compress commaVQ data using temporal model v3 (with current-frame above) + ANS.

Row-by-row processing: for each frame, decode rows 0-7 top to bottom.
Within each row, all 16 positions are processed in parallel.
The "above" token from the already-decoded row above is used as context.
"""
import os, sys, io, struct, time, math, json, lzma, zipfile, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v3 import (
    TemporalModelV3, TemporalV3Config, CONFIGS_V3,
    NEIGHBOR_MAP_4, NEIGHBOR_MAP_8, ABOVE_MAP
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_model_compact(checkpoint_path, output_path):
    """Save model as LZMA-compressed fp16 weights."""
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = cp['model_state_dict']
    state = {k: v for k, v in state.items() if k != 'causal_mask'}

    config = cp['config']
    meta = {
        'version': 3,
        'dim': config.dim,
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'intermediate_size': config.intermediate_size,
        'context_len': config.context_len,
        'n_neighbors': config.n_neighbors,
        'use_above': config.use_above,
        'keys': list(state.keys()),
        'shapes': [list(v.shape) for v in state.values()],
    }

    buf = io.BytesIO()
    meta_json = json.dumps(meta).encode()
    buf.write(struct.pack('<I', len(meta_json)))
    buf.write(meta_json)
    for v in state.values():
        buf.write(v.half().numpy().tobytes())

    compressed = lzma.compress(buf.getvalue(), preset=9)
    with open(output_path, 'wb') as f:
        f.write(compressed)
    return len(compressed)


def load_model_compact(compact_path, device=DEVICE):
    """Load model from LZMA-compressed fp16 format."""
    with open(compact_path, 'rb') as f:
        raw = lzma.decompress(f.read())

    offset = 0
    meta_len = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    meta = json.loads(raw[offset:offset + meta_len]); offset += meta_len

    config = TemporalV3Config(
        dim=meta['dim'],
        n_layer=meta['n_layer'],
        n_head=meta['n_head'],
        intermediate_size=meta['intermediate_size'],
        context_len=meta['context_len'],
        n_neighbors=meta['n_neighbors'],
        use_above=meta.get('use_above', True),
    )
    model = TemporalModelV3(config)

    state = {}
    for key, shape in zip(meta['keys'], meta['shapes']):
        numel = 1
        for s in shape:
            numel *= s
        data = np.frombuffer(raw[offset:offset + numel * 2], dtype=np.float16).copy()
        offset += numel * 2
        state[key] = torch.from_numpy(data).float().reshape(shape)

    mask = torch.tril(torch.ones(config.context_len, config.context_len, dtype=torch.bool))
    state['causal_mask'] = mask.view(1, 1, config.context_len, config.context_len)

    model.load_state_dict(state)
    model = model.to(device=device)
    model.eval()
    return model, config


def get_probs_for_row(model, config, seg, frame_idx, row, decoded_frame, neighbor_map, device=DEVICE):
    """Get probability distributions for 16 positions in a row.

    Uses temporal context from previous frames + above token from current frame.
    Returns (16, 1024) float32 probs.
    """
    K = config.context_len
    cols = 16
    positions = [row * cols + c for c in range(cols)]

    start = max(0, frame_idx - K)
    actual_K = frame_idx - start

    if actual_K == 0:
        return np.full((cols, 1024), 1.0 / 1024, dtype=np.float32)

    # Build temporal context for all 16 positions
    contexts = np.zeros((cols, actual_K, 1 + config.n_neighbors), dtype=np.int64)
    for i, pos in enumerate(positions):
        neighbors = neighbor_map[pos]
        center = seg[start:frame_idx, pos].reshape(actual_K, 1)
        neigh = seg[start:frame_idx][:, neighbors]
        contexts[i] = np.concatenate([center, neigh], axis=1)

    if actual_K < K:
        padded = np.zeros((cols, K, 1 + config.n_neighbors), dtype=np.int64)
        padded[:, K - actual_K:] = contexts
        contexts = padded

    ctx_t = torch.tensor(contexts, dtype=torch.long, device=device)
    pos_t = torch.tensor(positions, dtype=torch.long, device=device)

    # Above tokens from current frame
    above_tokens = np.full(cols, config.vocab_size, dtype=np.int64)  # default: unavailable
    if row > 0:
        for c in range(cols):
            above_pos = ABOVE_MAP[row * cols + c]
            if above_pos >= 0:
                above_tokens[c] = decoded_frame[above_pos]
    above_t = torch.tensor(above_tokens, dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(ctx_t, pos_t, above_t)

    frame_logits = logits[:, -1, :config.vocab_size].float()
    probs = torch.softmax(frame_logits, dim=-1).cpu().numpy()
    return probs


def compress_segment(model, config, tokens_2d, neighbor_map):
    """Compress one segment using temporal v3 model + ANS."""
    import constriction

    n_frames = tokens_2d.shape[0]
    all_probs = []

    for frame_idx in range(n_frames):
        decoded_frame = np.zeros(128, dtype=np.int16)
        for row in range(8):
            probs = get_probs_for_row(model, config, tokens_2d, frame_idx, row, decoded_frame, neighbor_map)
            all_probs.append(probs)  # (16, 1024)
            # Update decoded frame with actual tokens
            for c in range(16):
                decoded_frame[row * 16 + c] = tokens_2d[frame_idx, row * 16 + c]

    # Stack: (1200*8*16, 1024) = (153600, 1024)
    all_probs = np.concatenate(all_probs, axis=0).astype(np.float32)

    # Symbols in row-major order within each frame
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


def decompress_segment(model, config, compressed_bytes, neighbor_map):
    """Decompress one segment using temporal v3 model + ANS."""
    import constriction

    compressed_u32 = np.frombuffer(compressed_bytes, dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed_u32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    n_frames, n_pos = 1200, 128
    tokens = np.zeros((n_frames, n_pos), dtype=np.int16)

    for frame_idx in range(n_frames):
        decoded_frame = np.zeros(128, dtype=np.int16)
        for row in range(8):
            probs = get_probs_for_row(model, config, tokens, frame_idx, row, decoded_frame, neighbor_map)
            # Decode 16 positions
            row_tokens = coder.decode(model_family, probs.astype(np.float32))
            for c in range(16):
                pos = row * 16 + c
                decoded_frame[pos] = row_tokens[c]
                tokens[frame_idx, pos] = row_tokens[c]

    return tokens.reshape(-1, 8, 16)


def compress_all(args):
    from datasets import load_dataset
    import multiprocessing

    # Save compact model
    compact_path = args.model.replace('.pt', '_compact.bin')
    print("Saving compact model...", flush=True)
    model_size = save_model_compact(args.model, compact_path)
    print(f"  Compact: {model_size:,} bytes ({model_size/1024/1024:.1f} MB)", flush=True)

    # Load from compact model
    print("Loading model from compact format...", flush=True)
    model, config = load_model_compact(compact_path)
    print(f"  Loaded on {DEVICE}", flush=True)

    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4

    # Load dataset
    print("Loading dataset...", flush=True)
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    examples = list(ds['train'])
    if args.quick:
        examples = examples[:args.n_quick]
        print(f"  Quick mode: {len(examples)} segments", flush=True)

    # Verify round-trip on first segment
    if not args.skip_verify:
        print("Verifying round-trip on segment 0...", flush=True)
        seg0 = np.array(examples[0]['token.npy']).reshape(1200, 128)
        compressed0 = compress_segment(model, config, seg0, neighbor_map)
        decoded0 = decompress_segment(model, config, compressed0, neighbor_map)
        original0 = seg0.reshape(-1, 8, 16)
        match = np.array_equal(decoded0, original0)
        if match:
            print("  Round-trip OK!", flush=True)
        else:
            n_match = (decoded0 == original0).sum()
            n_total = original0.size
            print(f"  MISMATCH: {n_match}/{n_total} tokens", flush=True)
            if not args.force:
                print("  Aborting. Use --force to continue anyway.", flush=True)
                return

    # Compress all segments
    print(f"Compressing {len(examples)} segments...", flush=True)
    t0 = time.time()
    names = []
    blobs = []
    total_bits = 0
    total_tokens = 0

    for i, example in enumerate(examples):
        tokens = np.array(example['token.npy']).reshape(1200, 128)
        name = example['json']['file_name']

        compressed = compress_segment(model, config, tokens, neighbor_map)
        names.append(name)
        blobs.append(compressed)
        total_bits += len(compressed) * 8
        total_tokens += tokens.size

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_bits = total_bits / total_tokens
            print(f"  [{i+1}/{len(examples)}] {avg_bits:.3f} bits/token, "
                  f"{elapsed:.0f}s ({elapsed/(i+1):.1f}s/seg)", flush=True)

    compress_time = time.time() - t0
    avg_bits = total_bits / total_tokens

    # Build zip
    print("Building zip...", flush=True)
    backend_name = "temporal_v3"
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

    zip_path = args.output
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('data.bin', data_blob)
        zf.write(compact_path, 'model.bin')

    zip_size = os.path.getsize(zip_path)
    data_size = sum(len(b) for b in blobs)
    original_size = total_tokens * 10 / 8

    print(f"\n{'='*50}", flush=True)
    print(f"Model size:       {model_size:>12,} bytes ({model_size/1024/1024:.1f} MB)", flush=True)
    print(f"Compressed data:  {data_size:>12,} bytes ({data_size/1024/1024:.1f} MB)", flush=True)
    print(f"Zip file size:    {zip_size:>12,} bytes ({zip_size/1024/1024:.1f} MB)", flush=True)
    print(f"Original size:    {int(original_size):>12,} bytes ({original_size/1024/1024:.1f} MB)", flush=True)
    print(f"Compression rate: {original_size/zip_size:.3f}x", flush=True)
    print(f"Bits/token:       {avg_bits:.3f}", flush=True)
    print(f"Compress time:    {compress_time:.0f}s", flush=True)
    if len(examples) < 5000:
        proj_data = data_size * (5000 / len(examples))
        proj_total = proj_data + model_size + 1000
        proj_rate = (5000 * 1200 * 128 * 10 / 8) / proj_total
        print(f"Projected 5k:     {proj_rate:.3f}x", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default=os.path.join(HERE, "temporal_v3_submission.zip"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-quick", type=int, default=100)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    compress_all(args)

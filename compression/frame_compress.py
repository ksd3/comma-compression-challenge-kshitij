#!/usr/bin/env python3
"""Compress commaVQ data using frame model + ANS.

The frame model predicts all 128 positions in one forward pass using
teacher forcing (causal mask ensures each position only sees previous).
This makes compression very efficient — 1 forward pass per frame.

For decompression, we need to decode sequentially (position by position)
since each position depends on already-decoded ones.
"""
import os, sys, io, struct, time, math, json, lzma, zipfile, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.frame_model import FrameModel, FrameModelConfig, FRAME_CONFIGS

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_model_compact(checkpoint_path, output_path):
    """Save model as LZMA-compressed fp16 weights."""
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = cp['model_state_dict']
    state = {k: v for k, v in state.items() if k != 'causal_mask'}

    config = cp['config']
    meta = {
        'version': 5,
        'type': 'frame',
        'dim': config.dim,
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'intermediate_size': config.intermediate_size,
        'n_prev_frames': config.n_prev_frames,
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

    config = FrameModelConfig(
        dim=meta['dim'],
        n_layer=meta['n_layer'],
        n_head=meta['n_head'],
        intermediate_size=meta['intermediate_size'],
        n_prev_frames=meta.get('n_prev_frames', 1),
    )
    model = FrameModel(config)

    state = {}
    for key, shape in zip(meta['keys'], meta['shapes']):
        numel = 1
        for s in shape:
            numel *= s
        data = np.frombuffer(raw[offset:offset + numel * 2], dtype=np.float16).copy()
        offset += numel * 2
        state[key] = torch.from_numpy(data).float().reshape(shape)

    n_pos = config.n_positions
    mask = torch.tril(torch.ones(n_pos, n_pos, dtype=torch.bool))
    state['causal_mask'] = mask.view(1, 1, n_pos, n_pos)

    model.load_state_dict(state)
    model = model.to(device=device)
    model.eval()
    return model, config


def get_frame_probs(model, config, seg, frame_idx, device=DEVICE):
    """Get probability distributions for all 128 positions in a frame.

    Uses teacher forcing — one forward pass gives all 128 predictions.
    Returns (128, 1024) float32 probs.
    """
    K = config.n_prev_frames

    if frame_idx < K:
        # Not enough previous frames; use what we have, pad with zeros
        if frame_idx == 0:
            prev = np.zeros((K, 128), dtype=np.int64)
        else:
            prev = np.zeros((K, 128), dtype=np.int64)
            prev[K - frame_idx:] = seg[0:frame_idx]
    else:
        prev = seg[frame_idx - K:frame_idx].astype(np.int64)

    curr = seg[frame_idx].astype(np.int64)

    prev_t = torch.tensor(prev, dtype=torch.long, device=device).unsqueeze(0)  # (1, K, 128)
    curr_t = torch.tensor(curr, dtype=torch.long, device=device).unsqueeze(0)  # (1, 128)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(prev_t, curr_t)  # (1, 128, vocab)

    logits = logits[0, :, :config.vocab_size].float()  # (128, 1024)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def compress_segment(model, config, tokens_2d):
    """Compress one segment. One forward pass per frame (very efficient!)."""
    import constriction

    n_frames = tokens_2d.shape[0]
    all_probs = []

    for frame_idx in range(n_frames):
        probs = get_frame_probs(model, config, tokens_2d, frame_idx)
        all_probs.append(probs)  # (128, 1024)

    all_probs = np.concatenate(all_probs, axis=0).astype(np.float32)  # (1200*128, 1024)

    # Symbols in raster order per frame
    symbols = tokens_2d.reshape(-1).astype(np.int32)

    coder = constriction.stream.stack.AnsCoder()
    model_family = constriction.stream.model.Categorical(perfect=False)
    coder.encode_reverse(symbols, model_family, all_probs)

    return coder.get_compressed().tobytes()


def decompress_segment(model, config, compressed_bytes):
    """Decompress one segment.

    Must decode position-by-position since each depends on previous.
    Uses incremental forward passes through the model.
    """
    import constriction

    compressed_u32 = np.frombuffer(compressed_bytes, dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed_u32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    K = config.n_prev_frames
    n_frames, n_pos = 1200, 128
    tokens = np.zeros((n_frames, n_pos), dtype=np.int16)

    for frame_idx in range(n_frames):
        # Build previous frames context
        if frame_idx < K:
            prev = np.zeros((K, 128), dtype=np.int64)
            if frame_idx > 0:
                prev[K - frame_idx:] = tokens[0:frame_idx]
        else:
            prev = tokens[frame_idx - K:frame_idx].astype(np.int64)

        prev_t = torch.tensor(prev, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Decode position by position
        decoded = np.zeros(128, dtype=np.int64)
        for pos in range(n_pos):
            # Build current tokens (decoded so far)
            curr_t = torch.tensor(decoded, dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                    logits = model(prev_t, curr_t)

            # Get probs for this position
            pos_logits = logits[0, pos, :config.vocab_size].float()
            probs = torch.softmax(pos_logits, dim=-1).cpu().numpy().astype(np.float32)

            # Decode this position
            token = coder.decode(model_family, probs.reshape(1, -1))[0]
            decoded[pos] = token
            tokens[frame_idx, pos] = token

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
    print(f"  Loaded on {DEVICE}, K={config.n_prev_frames}", flush=True)

    # Load dataset
    print("Loading dataset...", flush=True)
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    examples = list(ds['train'])
    if args.quick:
        examples = examples[:args.n_quick]
        print(f"  Quick mode: {len(examples)} segments", flush=True)

    # Verify round-trip
    if not args.skip_verify:
        print("Verifying round-trip on segment 0...", flush=True)
        seg0 = np.array(examples[0]['token.npy']).reshape(1200, 128)
        compressed0 = compress_segment(model, config, seg0)
        decoded0 = decompress_segment(model, config, compressed0)
        original0 = seg0.reshape(-1, 8, 16)
        match = np.array_equal(decoded0, original0)
        if match:
            print("  Round-trip OK!", flush=True)
        else:
            n_match = (decoded0 == original0).sum()
            n_total = original0.size
            print(f"  MISMATCH: {n_match}/{n_total} tokens", flush=True)
            if not args.force:
                print("  Aborting.", flush=True)
                return

    # Compress
    print(f"Compressing {len(examples)} segments...", flush=True)
    t0 = time.time()
    names, blobs = [], []
    total_bits, total_tokens = 0, 0

    for i, example in enumerate(examples):
        tokens = np.array(example['token.npy']).reshape(1200, 128)
        name = example['json']['file_name']

        compressed = compress_segment(model, config, tokens)
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
    backend_name = "frame_model"
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
    decompress_script = os.path.join(HERE, 'decompress_submission.py')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('data.bin', data_blob)
        zf.write(compact_path, 'model.bin')
        zf.write(decompress_script, 'decompress.py')

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
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=os.path.join(HERE, "frame_submission.zip"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-quick", type=int, default=100)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    compress_all(args)

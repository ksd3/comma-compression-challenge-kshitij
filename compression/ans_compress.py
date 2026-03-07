#!/usr/bin/env python3
"""
Compress commaVQ data using a small trained GPT model + ANS coding.

Two encoding modes:
- "full": Uses full autoregressive probs (1 fwd pass per frame for encoding,
          128 fwd passes per frame for decoding — slow decode)
- "parallel": Uses only inter-frame probs (1 fwd pass per frame for both
             encoding AND decoding — fast but slightly worse compression)

Default is "full" since it gives better compression and encoding is one-time.
"""
import os
import sys
import io
import time
import json
import lzma
import struct
import zipfile
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import constriction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.gpt import GPT, GPTConfig
from compression.train_model_v2 import FrameIndependentGPT
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_CONFIGS = {
    "tiny": GPTConfig(n_layer=4, n_head=4, dim=128, intermediate_size=512),
    "small": GPTConfig(n_layer=6, n_head=4, dim=256, intermediate_size=1024),
    "medium": GPTConfig(n_layer=8, n_head=8, dim=512, intermediate_size=2048),
}


def save_model_compact(checkpoint_path, output_path):
    """Save model as LZMA-compressed fp16."""
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = {k: v for k, v in cp['model_state_dict'].items() if k != 'causal_mask'}

    meta = {
        'config_size': cp['config_size'],
        'version': cp.get('version', 1),
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


def load_model_from_checkpoint(checkpoint_path, device=DEVICE):
    """Load model from standard .pt checkpoint."""
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = MODEL_CONFIGS[cp['config_size']]
    model = GPT(config)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(device=device)
    model.eval()
    return model, config, cp['config_size']


def load_model_compact(compact_path, device=DEVICE, model_version=1):
    """Load model from LZMA-compressed fp16 format (same as decoder uses)."""
    with open(compact_path, 'rb') as f:
        raw = lzma.decompress(f.read())

    offset = 0
    meta_len = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    meta = json.loads(raw[offset:offset + meta_len]); offset += meta_len

    config = MODEL_CONFIGS[meta['config_size']]
    if model_version == 2:
        model = FrameIndependentGPT(config)
    else:
        model = GPT(config)

    state = {}
    for key, shape in zip(meta['keys'], meta['shapes']):
        numel = 1
        for s in shape:
            numel *= s
        data = np.frombuffer(raw[offset:offset + numel * 2], dtype=np.float16).copy()
        offset += numel * 2
        state[key] = torch.from_numpy(data).float().reshape(shape)

    state['causal_mask'] = torch.tril(
        torch.ones(config.block_size, config.block_size, dtype=torch.bool)
    ).view(1, 1, config.block_size, config.block_size)

    model.load_state_dict(state)
    model = model.to(device=device)
    model.eval()
    return model, config


def get_probs_full(model, config, tokens_2d, context_frames=20):
    """Full autoregressive probs: uses causal masking within frame.
    One forward pass per frame. Returns (1200*128, 1024) probs.

    During encoding we feed the COMPLETE frame so causal attention gives:
      P(t_j | prev_frames, BOS, t_0, ..., t_{j-1})
    """
    n_frames = tokens_2d.shape[0]
    bos = config.bos_token
    bos_col = np.full((n_frames, 1), bos, dtype=tokens_2d.dtype)
    frames = np.concatenate([bos_col, tokens_2d], axis=1)

    all_probs = []
    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - context_frames + 1)
            window = frames[start_frame:frame_idx + 1]
            seq = torch.tensor(window.ravel(), dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seq)

            frame_offset = (frame_idx - start_frame) * 129
            frame_logits = logits[0, frame_offset:frame_offset + 128, :1024].float()
            frame_probs = torch.softmax(frame_logits, dim=-1).cpu().numpy()
            all_probs.append(frame_probs)

    return np.concatenate(all_probs, axis=0)


def get_probs_parallel(model, config, tokens_2d, context_frames=20):
    """Parallel-decodable probs: only use inter-frame context.
    Each token is predicted as P(t_j | prev_frames, BOS).
    All 128 tokens get the SAME probability distribution.

    Actually wait — with causal masking, we can still get different probs
    per position by feeding prev_frames + BOS only (no current frame data tokens).
    Then logits[-1] gives P(t_0 | context, BOS).
    But for t_1..t_127, we only have P(t_j | context, BOS) which is the SAME
    distribution since the model has no info about t_0.

    Better approach: feed prev_frames + BOS + token positions via positional encoding.
    But that's not how the model works — it needs actual token values.

    So in parallel mode, all 128 tokens in a frame share the SAME probability
    distribution: P(t | context, BOS). This gives weaker compression but allows
    O(1) fwd passes per frame for both encode and decode.
    """
    n_frames = tokens_2d.shape[0]
    bos = config.bos_token
    bos_col = np.full((n_frames, 1), bos, dtype=tokens_2d.dtype)
    frames = np.concatenate([bos_col, tokens_2d], axis=1)

    all_probs = []
    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - context_frames + 1)
            # Only feed context frames + current BOS (no data tokens)
            if frame_idx == 0:
                window = np.array([[bos]], dtype=tokens_2d.dtype)
            else:
                prev = frames[start_frame:frame_idx]  # previous frames with BOS+data
                cur_bos = np.array([[bos]], dtype=tokens_2d.dtype)
                window = np.concatenate([prev.reshape(1, -1), cur_bos], axis=1)

            seq = torch.tensor(window.ravel(), dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seq)

            # logits[-1] predicts next token = first data token
            token_logits = logits[0, -1, :1024].float()
            probs = torch.softmax(token_logits, dim=-1).cpu().numpy()

            # Same distribution for all 128 tokens in this frame
            frame_probs = np.tile(probs, (128, 1))
            all_probs.append(frame_probs)

    return np.concatenate(all_probs, axis=0)


def get_probs_v2(model, config, tokens_2d, context_frames=20):
    """Frame-independent probs using v2 model with custom attention mask.

    Each data token can only see previous frames + own BOS (not other data tokens
    in same frame). This means the actual data token values DON'T affect predictions,
    so encoder and decoder get IDENTICAL probabilities with 1 fwd pass per frame.

    Encoder feeds: [prev_frames_with_data, current_BOS, current_data]
    Decoder feeds: [prev_frames_with_data, current_BOS, zeros]
    Both get same predictions at data positions (data tokens are masked out).
    """
    n_frames = tokens_2d.shape[0]
    bos = config.bos_token
    bos_col = np.full((n_frames, 1), bos, dtype=tokens_2d.dtype)
    frames = np.concatenate([bos_col, tokens_2d], axis=1)  # (1200, 129)

    all_probs = []
    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - context_frames + 1)
            window = frames[start_frame:frame_idx + 1]
            seq = torch.tensor(window.ravel(), dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model.forward_with_frame_mask(seq, context_frames=context_frames)

            # Get logits for current frame's data token positions
            # Position frame_offset is BOS, frame_offset+1..frame_offset+128 are data positions
            # Logits at position p predict token at position p+1 (next-token prediction)
            # Actually no — with frame-independent mask, logits at position [frame_offset+j]
            # for j in 0..127 predict the data token at that position.
            # Wait, let me think again. The input is [BOS, t0, t1, ..., t127] for each frame.
            # logits[pos] predicts the NEXT token after pos. So:
            # logits[BOS_pos] predicts t0, logits[t0_pos] predicts t1, etc.
            # But with frame-independent mask, t0_pos can't see t0 (only prev frames + BOS).
            # So logits[t0_pos] = f(prev_frames, BOS, pos_embed[t0_pos]) which predicts t1
            # but with NO info about t0.
            #
            # Actually in the training code, input is flat[:-1] and target is flat[1:].
            # The loss is computed on "data token positions" which are positions where
            # the target is a data token (not BOS).
            #
            # For frame F at positions [F*129, F*129+1, ..., F*129+128]:
            #   pos F*129 = BOS position, logits here predict t0 (using prev frames + BOS)
            #   pos F*129+1 = t0 position, logits here predict t1 (using prev frames + BOS, NOT t0)
            #   ...
            #   pos F*129+127 = t126 position, logits here predict t127
            #
            # So we need logits at positions [frame_offset, frame_offset+127] to predict t0..t127
            frame_offset = (frame_idx - start_frame) * 129
            frame_logits = logits[0, frame_offset:frame_offset + 128, :1024].float()
            frame_probs = torch.softmax(frame_logits, dim=-1).cpu().numpy()
            all_probs.append(frame_probs)

    return np.concatenate(all_probs, axis=0)


def compress_segment(model, config, tokens_2d, mode="full"):
    """Compress one segment with ANS coding."""
    if mode == "full":
        probs = get_probs_full(model, config, tokens_2d)
    elif mode == "v2":
        probs = get_probs_v2(model, config, tokens_2d)
    else:
        probs = get_probs_parallel(model, config, tokens_2d)

    symbols = tokens_2d.ravel().astype(np.int32)

    coder = constriction.stream.stack.AnsCoder()
    model_family = constriction.stream.model.Categorical(perfect=False)
    coder.encode_reverse(symbols, model_family, probs.astype(np.float32))

    return coder.get_compressed().tobytes()


def compress_all(model_path, output_zip_path, quick=False, mode="full"):
    import multiprocessing
    from datasets import load_dataset

    # Save compact model first, then load FROM compact model
    # This ensures encoder uses identical weights to decoder (fp16→fp32)
    compact_path = str(model_path).replace('.pt', '_compact.bin')
    print(f"Saving compact model...", flush=True)
    model_size = save_model_compact(model_path, compact_path)
    print(f"  Compact: {model_size/1024/1024:.1f} MB", flush=True)

    # Load from compact model to ensure bit-exact match with decoder
    print(f"Loading model from compact format...", flush=True)
    cp = torch.load(model_path, map_location='cpu', weights_only=True)
    config_name = cp['config_size']
    model_version = cp.get('version', 1)
    if model_version == 2:
        mode = "v2"  # Force v2 mode for v2 models
        print(f"  Detected v2 model, using frame-independent mode", flush=True)
    model, config = load_model_compact(compact_path, model_version=model_version)

    print("Loading dataset...", flush=True)
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    examples = list(ds['train'])
    if quick:
        examples = examples[:100]
        print(f"  Quick mode: {len(examples)} segments", flush=True)

    print(f"Compressing (mode={mode})...", flush=True)
    t0 = time.time()
    names = []
    compressed_blobs = []
    total_bits = 0
    total_tokens = 0

    for i, example in enumerate(examples):
        tokens = np.array(example['token.npy']).reshape(1200, 128)
        name = example['json']['file_name']

        compressed = compress_segment(model, config, tokens, mode=mode)

        names.append(name)
        compressed_blobs.append(compressed)
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

    # Encode mode in backend name so decompressor knows which mode to use
    backend_name = f"ans_{config_name}_{mode}"
    backend_bytes = backend_name.encode('utf-8')
    header = struct.pack('<B', len(backend_bytes)) + backend_bytes
    header += struct.pack('<I', len(names))

    name_table = b''
    for name in names:
        nb = name.encode('utf-8')
        name_table += struct.pack('<H', len(nb)) + nb

    size_table = b''
    for blob in compressed_blobs:
        size_table += struct.pack('<I', len(blob))

    data_blob = header + name_table + size_table + b''.join(compressed_blobs)

    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('data.bin', data_blob)
        zf.write(compact_path, 'model.bin')
        zf.write(HERE / 'decompress_ans.py', 'decompress.py')

    zip_size = os.path.getsize(output_zip_path)
    data_size = sum(len(b) for b in compressed_blobs)
    original_size = total_tokens * 10 / 8
    rate = original_size / zip_size

    print(f"\n{'='*50}", flush=True)
    print(f"Mode:             {mode}", flush=True)
    print(f"Model size:       {model_size:>12,} bytes ({model_size/1024/1024:.1f} MB)", flush=True)
    print(f"Compressed data:  {data_size:>12,} bytes ({data_size/1024/1024:.1f} MB)", flush=True)
    print(f"Zip file size:    {zip_size:>12,} bytes ({zip_size/1024/1024:.1f} MB)", flush=True)
    print(f"Original size:    {int(original_size):>12,} bytes ({original_size/1024/1024:.1f} MB)", flush=True)
    print(f"Compression rate: {rate:.3f}x", flush=True)
    print(f"Bits/token:       {avg_bits:.3f}", flush=True)
    print(f"Compress time:    {compress_time:.0f}s", flush=True)
    if len(examples) < 5000:
        proj_data = data_size * (5000 / len(examples))
        proj_total = proj_data + model_size + 1000  # zip overhead
        proj_rate = (5000 * 1200 * 128 * 10 / 8) / proj_total
        print(f"Projected 5k:     {proj_rate:.3f}x", flush=True)
    print(f"{'='*50}", flush=True)

    return rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=str(HERE / "compression_challenge_submission.zip"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mode", default="full", choices=["full", "parallel", "v2"])
    args = parser.parse_args()

    compress_all(args.model, args.output, args.quick, args.mode)

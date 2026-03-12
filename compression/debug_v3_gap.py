#!/usr/bin/env python3
"""Debug the gap between v3 eval bits (3.086) and ANS bits (3.825).

Hypothesis: early frames (0 to K-1) with limited temporal context get poor predictions.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v3 import (
    TemporalModelV3, TemporalV3Config, CONFIGS_V3,
    NEIGHBOR_MAP_4, NEIGHBOR_MAP_8, ABOVE_MAP
)
from compression.temporal_v3_compress import get_probs_for_row

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Load model
    cp = torch.load('compression/temporal_v3_small.pt', map_location='cpu', weights_only=False)
    config = cp['config']
    model = TemporalModelV3(config).to(DEVICE)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4
    K = config.context_len

    # Load 3 segments
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    segments = []
    for i, ex in enumerate(ds['train']):
        if i >= 3:
            break
        segments.append(np.array(ex['token.npy']).reshape(1200, 128))

    # Measure bits/token by frame range
    ranges = [
        (0, 1, "frame 0 (no context)"),
        (1, 5, "frames 1-4"),
        (5, 20, "frames 5-19"),
        (20, 100, "frames 20-99"),
        (100, 400, "frames 100-399"),
        (400, 1200, "frames 400-1199"),
    ]

    for seg_idx, seg in enumerate(segments[:1]):  # Just 1 segment for speed
        print(f"\nSegment {seg_idx}:")
        for start, end, label in ranges:
            total_nll = 0.0
            total_tokens = 0
            for frame_idx in range(start, min(end, seg.shape[0])):
                decoded_frame = np.zeros(128, dtype=np.int16)
                for row in range(8):
                    probs = get_probs_for_row(model, config, seg, frame_idx, row, decoded_frame, neighbor_map)
                    # Compute cross-entropy
                    for c in range(16):
                        pos = row * 16 + c
                        token = seg[frame_idx, pos]
                        p = max(probs[c, token], 1e-10)
                        total_nll -= math.log(p)
                        total_tokens += 1
                        decoded_frame[pos] = token

            if total_tokens > 0:
                bits = total_nll / total_tokens / math.log(2)
                print(f"  {label}: {bits:.3f} bits/token ({total_tokens} tokens)")

    # Also compute overall for full 1200 frames
    print("\n\nFull segment breakdown:")
    seg = segments[0]
    cumulative_nll = 0.0
    cumulative_tokens = 0
    checkpoints = [1, 5, 10, 20, 50, 100, 200, 500, 1200]

    for frame_idx in range(min(200, seg.shape[0])):  # Just 200 frames for speed
        decoded_frame = np.zeros(128, dtype=np.int16)
        for row in range(8):
            probs = get_probs_for_row(model, config, seg, frame_idx, row, decoded_frame, neighbor_map)
            for c in range(16):
                pos = row * 16 + c
                token = seg[frame_idx, pos]
                p = max(probs[c, token], 1e-10)
                cumulative_nll -= math.log(p)
                cumulative_tokens += 1
                decoded_frame[pos] = token

        if (frame_idx + 1) in checkpoints:
            bits = cumulative_nll / cumulative_tokens / math.log(2)
            print(f"  After {frame_idx+1} frames: {bits:.3f} bits/token (cumulative)")


if __name__ == '__main__':
    main()

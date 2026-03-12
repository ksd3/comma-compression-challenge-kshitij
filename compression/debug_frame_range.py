#!/usr/bin/env python3
"""Analyze per-frame-range compression quality across multiple segments."""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v3 import (
    TemporalModelV3, NEIGHBOR_MAP_4, NEIGHBOR_MAP_8, ABOVE_MAP
)
from compression.temporal_v3_compress import get_probs_for_row

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    cp = torch.load('compression/temporal_v3_small.pt', map_location='cpu', weights_only=False)
    config = cp['config']
    model = TemporalModelV3(config).to(DEVICE)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4

    from datasets import load_dataset
    import multiprocessing
    ds = load_dataset('commaai/commavq', num_proc=multiprocessing.cpu_count(),
                      data_files={'train': ['data-0000.tar.gz', 'data-0001.tar.gz']})

    # Test 5 segments, sample frames from different ranges
    ranges = [(0, 20), (20, 100), (100, 400), (400, 800), (800, 1200)]
    n_frames_per_range = 20  # Sample 20 frames per range for speed

    results = {r: {'nll': 0.0, 'tokens': 0} for r in ranges}

    for seg_idx in range(5):
        seg = np.array(list(ds['train'])[seg_idx + 200]['token.npy']).reshape(1200, 128)
        print(f"\nSegment {seg_idx + 200}:", flush=True)

        for start, end in ranges:
            # Sample frames evenly
            frame_indices = np.linspace(start, end - 1, n_frames_per_range, dtype=int)
            seg_nll = 0.0
            seg_tokens = 0

            for frame_idx in frame_indices:
                decoded_frame = np.zeros(128, dtype=np.int16)
                for row in range(8):
                    probs = get_probs_for_row(model, config, seg, frame_idx, row, decoded_frame, neighbor_map)
                    for c in range(16):
                        pos = row * 16 + c
                        token = seg[frame_idx, pos]
                        p = max(probs[c, token], 1e-10)
                        seg_nll -= math.log(p)
                        seg_tokens += 1
                        decoded_frame[pos] = token

            bits = seg_nll / seg_tokens / math.log(2)
            results[(start, end)]['nll'] += seg_nll
            results[(start, end)]['tokens'] += seg_tokens
            print(f"  frames {start}-{end}: {bits:.3f} bits/token", flush=True)

    print("\n\nOverall averages:", flush=True)
    for (start, end), data in results.items():
        bits = data['nll'] / data['tokens'] / math.log(2)
        print(f"  frames {start}-{end}: {bits:.3f} bits/token", flush=True)


if __name__ == '__main__':
    main()

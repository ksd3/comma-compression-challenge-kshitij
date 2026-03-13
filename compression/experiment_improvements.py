#!/usr/bin/env python3
"""Test improvements to temporal v2 compression: temperature, hybrid, etc."""
import os, sys, time, math, struct, io
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v2 import TemporalModelV2, TemporalV2Config, NEIGHBOR_MAP_4, NEIGHBOR_MAP_8
from compression.temporal_compress import load_model_compact, get_probs_for_frame

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_probs_with_temp(model, config, seg, frame_idx, neighbor_map, temperature=1.0, device=DEVICE):
    """Get probs with temperature scaling."""
    K = config.context_len
    n_pos = 128
    start = max(0, frame_idx - K)
    actual_K = frame_idx - start

    if actual_K == 0:
        return np.full((n_pos, 1024), 1.0 / 1024, dtype=np.float32)

    contexts = np.zeros((n_pos, actual_K, 1 + config.n_neighbors), dtype=np.int64)
    for pos in range(n_pos):
        neighbors = neighbor_map[pos]
        center = seg[start:frame_idx, pos].reshape(actual_K, 1)
        neigh = seg[start:frame_idx][:, neighbors]
        contexts[pos] = np.concatenate([center, neigh], axis=1)

    if actual_K < K:
        padded = np.zeros((n_pos, K, 1 + config.n_neighbors), dtype=np.int64)
        padded[:, K - actual_K:] = contexts
        contexts = padded

    ctx_t = torch.tensor(contexts, dtype=torch.long, device=device)
    pos_t = torch.arange(n_pos, dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(ctx_t, pos_t)

    frame_logits = logits[:, -1, :config.vocab_size].float()
    # Apply temperature
    frame_logits = frame_logits / temperature
    probs = torch.softmax(frame_logits, dim=-1).cpu().numpy()
    return probs


def get_transition_probs(seg, frame_idx, marginal, transition):
    """Get transition table probs for a frame."""
    n_pos = 128
    if frame_idx == 0:
        return np.tile(marginal, (n_pos, 1))
    else:
        prev = seg[frame_idx - 1]  # (128,)
        return transition[prev]  # (128, 1024)


def eval_bits_per_token(all_probs, all_targets):
    """Compute bits/token from probability arrays."""
    total_nll = 0.0
    total_tokens = 0
    for probs, targets in zip(all_probs, all_targets):
        # probs: (128, 1024), targets: (128,)
        for pos in range(128):
            p = probs[pos, targets[pos]]
            p = max(p, 1e-10)
            total_nll += -math.log(p)
            total_tokens += 1
    return total_nll / total_tokens / math.log(2)


def test_temperature(model, config, segments, neighbor_map, temperatures):
    """Test different temperature values."""
    print("\n=== Temperature Scaling ===")
    for temp in temperatures:
        all_probs = []
        all_targets = []
        for seg in segments:
            for frame_idx in range(1, min(100, seg.shape[0])):  # Skip frame 0, test 99 frames
                probs = get_probs_with_temp(model, config, seg, frame_idx, neighbor_map, temperature=temp)
                all_probs.append(probs)
                all_targets.append(seg[frame_idx])
        bits = eval_bits_per_token(all_probs, all_targets)
        print(f"  temp={temp:.2f}: {bits:.4f} bits/token")


def test_hybrid(model, config, segments, neighbor_map, marginal, transition, alphas):
    """Test hybrid of neural + transition table."""
    print("\n=== Hybrid Neural + Transition ===")
    for alpha in alphas:
        all_probs = []
        all_targets = []
        for seg in segments:
            for frame_idx in range(1, min(100, seg.shape[0])):
                neural_probs = get_probs_for_frame(model, config, seg, frame_idx, neighbor_map)
                trans_probs = get_transition_probs(seg, frame_idx, marginal, transition)
                # Blend
                hybrid = alpha * neural_probs + (1 - alpha) * trans_probs
                # Re-normalize
                hybrid = hybrid / hybrid.sum(axis=1, keepdims=True)
                all_probs.append(hybrid)
                all_targets.append(seg[frame_idx])
        bits = eval_bits_per_token(all_probs, all_targets)
        print(f"  alpha={alpha:.2f} (neural): {bits:.4f} bits/token")


def build_transition_tables(segments):
    """Build transition table from segments."""
    marginal = np.zeros(1024, dtype=np.float64)
    transition = np.zeros((1024, 1024), dtype=np.float64)
    for seg in segments:
        for token in seg[0]:
            marginal[token] += 1
        for t in range(1, seg.shape[0]):
            for pos in range(128):
                prev = seg[t-1, pos]
                cur = seg[t, pos]
                transition[prev, cur] += 1
    marginal = (marginal + 1) / (marginal.sum() + 1024)
    for i in range(1024):
        row = transition[i]
        transition[i] = (row + 0.01) / (row.sum() + 0.01 * 1024)
    return marginal.astype(np.float32), transition.astype(np.float32)


def main():
    # Load model
    compact_path = os.path.join(HERE, 'temporal_v2_small_compact.bin')
    if not os.path.exists(compact_path):
        # Create compact from .pt
        from compression.temporal_compress import save_model_compact
        save_model_compact(os.path.join(HERE, 'temporal_v2_small.pt'), compact_path)

    model, config = load_model_compact(compact_path)
    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4
    print(f"Model: dim={config.dim}, layers={config.n_layer}, neighbors={config.n_neighbors}")

    # Load a few segments
    print("Loading data...")
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    segments = []
    for i, example in enumerate(ds['train']):
        if i >= 5:
            break
        segments.append(np.array(example['token.npy']).reshape(1200, 128))
    print(f"Loaded {len(segments)} segments")

    # Baseline (temp=1.0)
    print("\n=== Baseline (temp=1.0) ===")
    all_probs = []
    all_targets = []
    for seg in segments:
        for frame_idx in range(1, min(100, seg.shape[0])):
            probs = get_probs_for_frame(model, config, seg, frame_idx, neighbor_map)
            all_probs.append(probs)
            all_targets.append(seg[frame_idx])
    bits = eval_bits_per_token(all_probs, all_targets)
    print(f"  Baseline: {bits:.4f} bits/token")

    # Test temperatures
    test_temperature(model, config, segments, neighbor_map,
                     [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

    # Build transition tables and test hybrid
    print("\nBuilding transition tables from 5 segments...")
    marginal, transition = build_transition_tables(segments)
    test_hybrid(model, config, segments, neighbor_map, marginal, transition,
                [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5])


if __name__ == '__main__':
    main()

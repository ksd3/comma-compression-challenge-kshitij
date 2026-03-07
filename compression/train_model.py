#!/usr/bin/env python3
"""
Train a small GPT model on commaVQ data for entropy-based compression.

Uses mixed precision (bf16) for speed and memory efficiency.
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.gpt import GPT, GPTConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = Path(__file__).resolve().parent


def make_small_config(size="small"):
    configs = {
        "tiny": GPTConfig(n_layer=4, n_head=4, dim=128, intermediate_size=512),
        "small": GPTConfig(n_layer=6, n_head=4, dim=256, intermediate_size=1024),
        "medium": GPTConfig(n_layer=8, n_head=8, dim=512, intermediate_size=2048),
    }
    return configs[size]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_all_segments():
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    segments = []
    names = []
    for example in ds['train']:
        tokens = np.array(example['token.npy']).reshape(1200, 128)
        segments.append(tokens)
        names.append(example['json']['file_name'])
    return segments, names


class SegmentDataset(torch.utils.data.Dataset):
    """Yields random windows of context_frames from random segments."""

    def __init__(self, segments, config, context_frames=20, samples_per_epoch=50000):
        self.segments = segments
        self.config = config
        self.context_frames = context_frames
        self.samples_per_epoch = samples_per_epoch
        self.bos = config.bos_token

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Pick random segment and random window
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]
        n_frames = seg.shape[0]

        start = np.random.randint(0, n_frames - self.context_frames + 1)
        window = seg[start:start + self.context_frames]  # (context_frames, 128)

        # Add BOS tokens: (context_frames, 129)
        bos_col = np.full((self.context_frames, 1), self.bos, dtype=window.dtype)
        frames = np.concatenate([bos_col, window], axis=1)
        flat = frames.ravel().astype(np.int64)

        return torch.from_numpy(flat[:-1]), torch.from_numpy(flat[1:])


def train(args):
    config = make_small_config(args.size)
    print(f"Model: {args.size} (n_layer={config.n_layer}, dim={config.dim})")

    model = GPT(config).to(DEVICE)
    n_params = count_params(model)
    print(f"Params: {n_params:,} ({n_params/1e6:.1f}M, {n_params/1024/1024:.1f} MB int8)")

    # Load data
    print("Loading data...", flush=True)
    t0 = time.time()
    segments, names = load_all_segments()
    print(f"Loaded {len(segments)} segments in {time.time()-t0:.0f}s", flush=True)

    # Dataset: random windows, 50k samples per epoch
    dataset = SegmentDataset(segments, config, context_frames=20,
                             samples_per_epoch=args.samples_per_epoch)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    # Optimizer with mixed precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    total_steps = len(loader) * args.epochs
    warmup_steps = min(200, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"Training: {args.epochs} epochs × {len(loader)} steps = {total_steps} total", flush=True)
    model.train()
    best_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_tokens = 0
        t0 = time.time()

        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                                       targets.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * targets.numel()
            epoch_tokens += targets.numel()
            global_step += 1

            if global_step % 200 == 0:
                avg = epoch_loss / epoch_tokens
                bits = avg / math.log(2)
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0
                tps = epoch_tokens / elapsed
                print(f"  step {global_step}/{total_steps} loss={avg:.4f} "
                      f"bits={bits:.3f} lr={lr:.2e} tok/s={tps:.0f}", flush=True)

        avg_loss = epoch_loss / epoch_tokens
        bits = avg_loss / math.log(2)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
              f"bits/token={bits:.3f} time={elapsed:.0f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = HERE / f"model_{args.size}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_size': args.size,
                'loss': avg_loss,
                'bits_per_token': bits,
                'epoch': epoch,
            }, save_path)
            print(f"  Saved best model ({bits:.3f} bits/token)", flush=True)

    # Evaluate on first segment
    print("\nEvaluating on first segment...", flush=True)
    model.eval()
    evaluate_segment(model, config, segments[0])

    return model, config


def evaluate_segment(model, config, tokens_2d):
    """Compute per-token cross-entropy for a single segment."""
    n_frames = tokens_2d.shape[0]
    bos = config.bos_token

    bos_col = np.full((n_frames, 1), bos, dtype=tokens_2d.dtype)
    frames = np.concatenate([bos_col, tokens_2d], axis=1)

    total_nll = 0
    total_tokens = 0

    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - 19)
            window = frames[start_frame:frame_idx + 1].ravel()
            seq = torch.tensor(window, dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seq)

            frame_offset = (frame_idx - start_frame) * 129
            pred_logits = logits[0, frame_offset:frame_offset + 128, :1024].float()
            targets = torch.tensor(tokens_2d[frame_idx], dtype=torch.long, device=DEVICE)

            loss = F.cross_entropy(pred_logits, targets, reduction='sum')
            total_nll += loss.item()
            total_tokens += 128

    bits = total_nll / total_tokens / math.log(2)
    print(f"  Cross-entropy: {bits:.3f} bits/token")
    print(f"  Theoretical compression: {10/bits:.2f}x")
    return bits


def quantize_model(model_path, output_path=None):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    config = make_small_config(checkpoint['config_size'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    if output_path is None:
        output_path = str(model_path).replace('.pt', '_int8.pt')

    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    torch.save({
        'model_state_dict': quantized.state_dict(),
        'config_size': checkpoint['config_size'],
        'bits_per_token': checkpoint['bits_per_token'],
    }, output_path)

    orig_size = os.path.getsize(model_path)
    quant_size = os.path.getsize(output_path)
    print(f"Original: {orig_size/1024/1024:.1f} MB")
    print(f"Quantized: {quant_size/1024/1024:.1f} MB")
    print(f"Reduction: {orig_size/quant_size:.1f}x")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--quantize", type=str, help="Path to model to quantize")
    args = parser.parse_args()

    if args.quantize:
        quantize_model(args.quantize)
    else:
        train(args)

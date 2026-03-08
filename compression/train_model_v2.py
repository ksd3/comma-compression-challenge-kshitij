#!/usr/bin/env python3
"""
Train a small GPT model with frame-independent attention masking.

Within each frame, tokens cannot attend to other tokens in the same frame.
They can only attend to tokens from PREVIOUS frames + their own BOS token.
This enables 1 forward pass per frame for both encoding and decoding.
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


def make_frame_independent_mask(n_frames, tokens_per_frame=129):
    """Create attention mask where tokens within a frame can't see each other,
    but CAN see all tokens from previous frames and their own BOS token.

    Frame structure: [BOS, t0, t1, ..., t127] × n_frames
    For data token t_j in frame F:
      - Can attend to ALL tokens in frames 0..F-1 (BOS + data)
      - Can attend to BOS of frame F
      - CANNOT attend to t_0..t_127 of frame F (including self)

    This means all 128 data tokens in a frame get DIFFERENT predictions
    (due to positional embeddings) but independent of each other.
    """
    seq_len = n_frames * tokens_per_frame
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for f in range(n_frames):
        frame_start = f * tokens_per_frame
        bos_pos = frame_start
        data_start = frame_start + 1
        data_end = frame_start + tokens_per_frame

        for pos in range(frame_start, data_end):
            # Can attend to all positions in previous frames
            if frame_start > 0:
                mask[pos, :frame_start] = True
            # Can attend to own BOS
            mask[pos, bos_pos] = True
            # BOS can attend to itself (already set above)
            # Data tokens CANNOT attend to other data tokens in same frame

    return mask.view(1, 1, seq_len, seq_len)


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, segments, config, context_frames=20, samples_per_epoch=50000):
        self.segments = segments
        self.config = config
        self.context_frames = context_frames
        self.samples_per_epoch = samples_per_epoch
        self.bos = config.bos_token

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]
        n_frames = seg.shape[0]

        start = np.random.randint(0, n_frames - self.context_frames + 1)
        window = seg[start:start + self.context_frames]

        # Target: [BOS, t0, t1, ..., t127] per frame (actual tokens)
        bos_col = np.full((self.context_frames, 1), self.bos, dtype=window.dtype)
        target_frames = np.concatenate([bos_col, window], axis=1)
        target_flat = target_frames.ravel().astype(np.int64)

        # Input: replace data token positions with BOS so model can't use token embeddings.
        # This ensures encode and decode see identical inputs.
        input_frames = np.full_like(target_frames, self.bos)  # all BOS
        # Keep BOS positions as BOS (already done), keep previous frame data
        # Actually we need previous frames to have real data tokens (the model
        # attends to them), but current frame data should be BOS.
        # However, the mask blocks within-frame data attention, so previous frame
        # data tokens ARE visible via attention. We need their actual values.
        # So: input has real tokens for ALL positions, but...
        # Wait - the issue is the residual connection, not attention.
        # Even for PREVIOUS frames, the residual carries token info at each position.
        # But for previous frames, that's fine since we DO know those tokens.
        # Only for the CURRENT frame do we not know the data tokens at decode time.
        #
        # Solution: in input, replace ONLY current-frame data positions with BOS.
        # But "current frame" differs per token in the sequence.
        #
        # Simpler: ALL data token positions get BOS in the input.
        # Previous frame data is still accessible via attention to those positions,
        # where the key/value come from BOS embedding + pos_embed. Hmm, that's wrong
        # because then the model can't distinguish different tokens in prev frames.
        #
        # We need previous frame tokens to be real, current frame tokens to be BOS.
        # In a single training example with 20 frames, EVERY frame is a "current frame"
        # at different positions. So we can't make just one frame's data BOS.
        #
        # The only way: the model input has ALL real tokens. But the mask ensures
        # data tokens only attend to prev frames + BOS. The residual carries the
        # token embedding, but we must ensure the model doesn't rely on it.
        #
        # Alternative: use a SEPARATE embedding for "current frame data" positions.
        # Or: just use a single learnable "data placeholder" token at all data positions.
        input_frames = np.concatenate([bos_col, np.full_like(window, self.bos)], axis=1)
        input_flat = input_frames.ravel().astype(np.int64)

        return torch.from_numpy(input_flat[:-1]), torch.from_numpy(target_flat[1:])


class FrameIndependentGPT(GPT):
    """GPT with frame-independent attention mask for compression."""

    def __init__(self, config=GPTConfig()):
        super().__init__(config)
        # We'll set the mask dynamically based on context_frames

    _mask_cache = {}

    def forward_with_frame_mask(self, idx, context_frames=20):
        """Forward pass with frame-independent masking."""
        bsz, seq_len = idx.shape
        tpf = self.config.tokens_per_frame  # 129

        # Cache the mask (it's the same for all inputs of same seq_len)
        if seq_len not in self._mask_cache or self._mask_cache[seq_len].device != idx.device:
            positions = torch.arange(seq_len, device=idx.device)
            frame_ids = positions // tpf
            pos_in_frame = positions % tpf
            bos_positions = frame_ids * tpf

            # Each position can see:
            # 1. All positions in previous frames (frame_id[key] < frame_id[query])
            # 2. Its own BOS token (key == bos_position of query's frame)
            q_frames = frame_ids.unsqueeze(1)  # (seq_len, 1)
            k_frames = frame_ids.unsqueeze(0)  # (1, seq_len)
            k_pos = positions.unsqueeze(0)      # (1, seq_len)
            q_bos = bos_positions.unsqueeze(1)  # (seq_len, 1)

            prev_frame_mask = k_frames < q_frames           # previous frames
            own_bos_mask = k_pos == q_bos                    # own BOS
            mask = (prev_frame_mask | own_bos_mask).view(1, 1, seq_len, seq_len)
            self._mask_cache[seq_len] = mask
        else:
            mask = self._mask_cache[seq_len]

        input_pos = torch.arange(seq_len, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(input_pos)

        for layer in self.transformer.h:
            x = layer(x, input_pos, mask)

        x = self.transformer.ln_f(x)
        return self.lm_head(x)


def make_loss_mask(seq_len, tokens_per_frame=129):
    """Create mask for loss computation.
    We want loss on data token positions only (not BOS).
    Also, with frame-independent mask, we only get valid predictions
    for data tokens (not BOS) since BOS is predicted from previous frame.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for f in range(seq_len // tokens_per_frame):
        # Data positions in this frame: [f*129 + 1, ..., f*129 + 128]
        data_start = f * tokens_per_frame + 1
        data_end = data_start + tokens_per_frame - 1
        mask[data_start:data_end] = True
    return mask


def train(args):
    config = make_small_config(args.size)
    print(f"Model: {args.size} (n_layer={config.n_layer}, dim={config.dim})", flush=True)

    model = FrameIndependentGPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)

    # Load data
    print("Loading data...", flush=True)
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    segments = []
    for example in ds['train']:
        segments.append(np.array(example['token.npy']).reshape(1200, 128))
    print(f"Loaded {len(segments)} segments", flush=True)

    # Dataset
    dataset = SegmentDataset(segments, config, context_frames=20,
                             samples_per_epoch=args.samples_per_epoch)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    # Precompute loss mask for target sequence
    target_len = 20 * 129 - 1  # target length
    loss_mask = make_loss_mask(target_len, 129).to(DEVICE)

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

    print(f"Training: {args.epochs} epochs × {len(loader)} steps", flush=True)
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
                logits = model.forward_with_frame_mask(inputs)
                # Apply loss mask: only compute loss on data tokens
                logits_flat = logits.reshape(-1, config.vocab_size)
                targets_flat = targets.reshape(-1)
                mask = loss_mask[:logits_flat.shape[0] // inputs.shape[0]].repeat(inputs.shape[0])
                masked_logits = logits_flat[mask]
                masked_targets = targets_flat[mask]
                loss = F.cross_entropy(masked_logits, masked_targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n_tokens = masked_targets.numel()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            global_step += 1

            if global_step % 200 == 0:
                avg = epoch_loss / epoch_tokens
                bits = avg / math.log(2)
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0
                print(f"  step {global_step}/{total_steps} loss={avg:.4f} "
                      f"bits={bits:.3f} lr={lr:.2e}", flush=True)

        avg_loss = epoch_loss / epoch_tokens
        bits = avg_loss / math.log(2)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
              f"bits/token={bits:.3f} time={elapsed:.0f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = HERE / f"model_{args.size}_v2.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_size': args.size,
                'loss': avg_loss,
                'bits_per_token': bits,
                'epoch': epoch,
                'version': 2,
            }, save_path)
            print(f"  Saved best model ({bits:.3f} bits/token)", flush=True)

    # Evaluate
    print("\nEvaluating...", flush=True)
    model.eval()
    evaluate_v2(model, config, segments[0])


def evaluate_v2(model, config, tokens_2d):
    """Evaluate with frame-independent masking."""
    n_frames = tokens_2d.shape[0]
    bos = config.bos_token

    bos_col = np.full((n_frames, 1), bos, dtype=tokens_2d.dtype)
    frames = np.concatenate([bos_col, tokens_2d], axis=1)

    total_nll = 0
    total_tokens = 0

    with torch.no_grad():
        for frame_idx in range(n_frames):
            start_frame = max(0, frame_idx - 19)
            window = frames[start_frame:frame_idx + 1]
            seq = torch.tensor(window.ravel(), dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model.forward_with_frame_mask(seq, context_frames=20)

            # Get logits for current frame's data tokens
            frame_offset = (frame_idx - start_frame) * 129
            # With frame-independent mask, logits at positions [frame_offset+1 ... frame_offset+128]
            # are the predictions for data tokens, conditioned only on prev frames + BOS
            pred_logits = logits[0, frame_offset:frame_offset + 128, :1024].float()
            targets = torch.tensor(tokens_2d[frame_idx], dtype=torch.long, device=DEVICE)

            loss = F.cross_entropy(pred_logits, targets, reduction='sum')
            total_nll += loss.item()
            total_tokens += 128

    bits = total_nll / total_tokens / math.log(2)
    print(f"  Cross-entropy: {bits:.3f} bits/token", flush=True)
    print(f"  Theoretical compression: {10/bits:.2f}x", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    args = parser.parse_args()
    train(args)

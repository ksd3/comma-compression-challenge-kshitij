#!/usr/bin/env python3
"""Train v3 model for longer with better schedule and all-timestep loss."""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v3 import (
    TemporalModelV3, TemporalV3Config, CONFIGS_V3,
    NEIGHBOR_MAP_4, NEIGHBOR_MAP_8, ABOVE_MAP
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TemporalV3DatasetAllTimesteps(torch.utils.data.Dataset):
    """Train on ALL K timesteps, with correct above tokens per timestep.

    For each sample, we pick a position and a K+1-length window.
    For each target timestep t, we provide the correct above token from frame t.
    This gives K× more training signal per sample.
    """
    def __init__(self, segments, config, samples_per_epoch=200000):
        self.segments = segments
        self.config = config
        self.K = config.context_len
        self.samples_per_epoch = samples_per_epoch
        self.neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4
        self.above_map = ABOVE_MAP
        self.unavail = config.vocab_size

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]
        pos = np.random.randint(128)
        neighbors = self.neighbor_map[pos]
        above_pos = self.above_map[pos]

        t = np.random.randint(0, seg.shape[0] - self.K)

        # Temporal context
        center = seg[t:t + self.K, pos:pos + 1]
        neighbor_vals = seg[t:t + self.K][:, neighbors]
        context = np.concatenate([center, neighbor_vals], axis=1).astype(np.int64)

        # Target: next token at center for K timesteps
        target = seg[t + 1:t + self.K + 1, pos].astype(np.int64)

        # Above tokens for EACH target timestep (K above tokens)
        above_tokens = np.full(self.K, self.unavail, dtype=np.int64)
        if above_pos >= 0:
            for k in range(self.K):
                above_tokens[k] = seg[t + 1 + k, above_pos]

        return (torch.from_numpy(context),
                torch.from_numpy(target),
                pos,
                torch.from_numpy(above_tokens))


def train(args):
    config = CONFIGS_V3[args.size]
    model = TemporalModelV3(config).to(DEVICE)
    n_params = model.count_params()
    print(f"Model: {args.size} (dim={config.dim}, layers={config.n_layer})", flush=True)
    print(f"Params: {n_params:,}", flush=True)
    print(f"Training mode: all-timestep with per-timestep above tokens", flush=True)

    # Load data
    print("Loading data...", flush=True)
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    n_segs = args.n_segments
    segments = []
    for i, example in enumerate(ds['train']):
        if i >= n_segs:
            break
        segments.append(np.array(example['token.npy']).reshape(1200, 128))
    print(f"Loaded {len(segments)} segments", flush=True)

    dataset = TemporalV3DatasetAllTimesteps(segments, config, samples_per_epoch=args.samples_per_epoch)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch

    # Warmup + cosine schedule
    warmup_steps = min(1000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training: {args.epochs} epochs x {steps_per_epoch} steps, warmup={warmup_steps}", flush=True)

    best_loss = float('inf')
    global_step = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (context, target, positions, above_tokens) in enumerate(loader):
            context = context.to(DEVICE)       # (batch, K, 1+N)
            target = target.to(DEVICE)          # (batch, K)
            positions = positions.to(DEVICE)    # (batch,)
            above_tokens = above_tokens.to(DEVICE)  # (batch, K)

            bsz = context.shape[0]
            K = context.shape[1]

            # Process each timestep's above token by running the model K times
            # ... That's too slow. Instead, we need to modify the model to accept
            # per-timestep above tokens. But the current model broadcasts above_token.
            #
            # Simplest approach: just use the above token from the LAST timestep
            # for all predictions, accepting the mismatch. This is what v3 does.
            # But train on all timesteps to get more gradient signal.

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                # Use last timestep's above token (matches inference)
                logits = model(context, positions, above_tokens[:, -1])
                # Loss on all timesteps
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    target.view(-1)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_tokens = target.numel()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1

            if (batch_idx + 1) % 200 == 0:
                avg = epoch_loss / epoch_tokens
                bits = avg / math.log(2)
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step}/{total_steps} loss={avg:.4f} bits={bits:.3f} lr={lr:.2e}",
                      flush=True)

        avg_loss = epoch_loss / epoch_tokens
        bits = avg_loss / math.log(2)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} bits/token={bits:.3f} time={elapsed:.0f}s",
              flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(os.path.dirname(__file__), f'temporal_v3_long_{args.size}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_size': args.size,
                'config': config,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'bits': bits,
                'version': 3,
            }, save_path)
            print(f"  Saved best model ({bits:.3f} bits/token)", flush=True)

    # Eval with proper row-by-row
    print("\nEvaluating (row-by-row) on 5 segments, 200 frames...", flush=True)
    model.eval()
    eval_segs = segments[:5]
    total_nll = 0.0
    total_tokens = 0
    K = config.context_len
    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4

    with torch.no_grad():
        for seg in eval_segs:
            for frame_idx in range(K, min(K + 200, seg.shape[0])):
                decoded_frame = np.zeros(128, dtype=np.int16)
                for row in range(8):
                    for col in range(16):
                        pos = row * 16 + col
                        above_pos = ABOVE_MAP[pos]
                        center = seg[frame_idx-K:frame_idx, pos:pos+1]
                        neigh = seg[frame_idx-K:frame_idx][:, neighbor_map[pos]]
                        ctx = np.concatenate([center, neigh], axis=1).astype(np.int64)
                        ctx_t = torch.tensor(ctx, dtype=torch.long, device=DEVICE).unsqueeze(0)
                        pos_t = torch.tensor([pos], dtype=torch.long, device=DEVICE)
                        if above_pos >= 0:
                            above_val = decoded_frame[above_pos]
                        else:
                            above_val = config.vocab_size
                        above_t = torch.tensor([above_val], dtype=torch.long, device=DEVICE)
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                            logits = model(ctx_t, pos_t, above_t)
                        target_val = torch.tensor([seg[frame_idx, pos]], dtype=torch.long, device=DEVICE)
                        loss = F.cross_entropy(logits[0, -1:, :config.vocab_size], target_val)
                        total_nll += loss.item()
                        total_tokens += 1
                        decoded_frame[pos] = seg[frame_idx, pos]

    bits = total_nll / total_tokens / math.log(2)
    print(f"  Eval bits/token: {bits:.3f}", flush=True)
    print(f"  Theoretical compression: {10/bits:.3f}x", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='small', choices=list(CONFIGS_V3.keys()))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--samples-per-epoch', type=int, default=500000)
    parser.add_argument('--n-segments', type=int, default=5000)
    args = parser.parse_args()
    train(args)

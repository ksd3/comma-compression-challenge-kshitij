#!/usr/bin/env python3
"""Train per-position temporal model v2 (with spatial neighbors) for commaVQ compression."""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v2 import TemporalModelV2, TemporalV2Config, CONFIGS_V2, NEIGHBOR_MAP_4, NEIGHBOR_MAP_8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TemporalV2Dataset(torch.utils.data.Dataset):
    """Yields (context_with_neighbors, target, position) samples.

    context_with_neighbors: (K, 1+4) — center + 4 neighbors for each timestep
    target: (K,) — next token at center position
    position: spatial position ID
    """
    def __init__(self, segments, config, samples_per_epoch=200000):
        self.segments = segments
        self.config = config
        self.K = config.context_len
        self.samples_per_epoch = samples_per_epoch
        self.neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]  # (1200, 128)
        pos = np.random.randint(128)
        neighbors = self.neighbor_map[pos]  # (4,) neighbor position indices

        t = np.random.randint(0, seg.shape[0] - self.K)

        # Context: center + 4 neighbors for K timesteps
        center = seg[t:t + self.K, pos:pos + 1]  # (K, 1)
        neighbor_vals = seg[t:t + self.K][:, neighbors]  # (K, 4)
        context = np.concatenate([center, neighbor_vals], axis=1).astype(np.int64)  # (K, 5)

        # Target: next token at center position
        target = seg[t + 1:t + self.K + 1, pos].astype(np.int64)  # (K,)

        return torch.from_numpy(context), torch.from_numpy(target), pos


def train(args):
    config = CONFIGS_V2[args.size]
    model = TemporalModelV2(config).to(DEVICE)
    n_params = model.count_params()
    print(f"Model: {args.size} (dim={config.dim}, layers={config.n_layer}, neighbors={config.n_neighbors})", flush=True)
    print(f"Params: {n_params:,}", flush=True)

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

    dataset = TemporalV2Dataset(segments, config, samples_per_epoch=args.samples_per_epoch)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0)

    print(f"Training: {args.epochs} epochs x {steps_per_epoch} steps", flush=True)

    best_loss = float('inf')
    global_step = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (context, target, positions) in enumerate(loader):
            context = context.to(DEVICE)
            target = target.to(DEVICE)
            positions = positions.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                logits = model(context, positions)
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
            save_path = os.path.join(os.path.dirname(__file__), f'temporal_v2_{args.size}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_size': args.size,
                'config': config,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'bits': bits,
                'version': 2,
            }, save_path)
            print(f"  Saved best model ({bits:.3f} bits/token)", flush=True)

    # Quick eval on 3 segments
    print("\nEvaluating on 3 segments...", flush=True)
    model.eval()
    eval_segs = segments[:3]
    total_nll = 0.0
    total_tokens = 0
    K = config.context_len

    neighbor_map = NEIGHBOR_MAP_8 if config.n_neighbors == 8 else NEIGHBOR_MAP_4
    with torch.no_grad():
        for seg in eval_segs:
            for pos in range(128):
                neighbors = neighbor_map[pos]
                col = seg[:, pos]
                neighbor_cols = seg[:, neighbors]  # (1200, 4)

                for t in range(K, len(col)):
                    center = col[t-K:t].reshape(K, 1)
                    neigh = neighbor_cols[t-K:t]  # (K, 4)
                    ctx = np.concatenate([center, neigh], axis=1).astype(np.int64)  # (K, 5)
                    ctx_t = torch.tensor(ctx, dtype=torch.long, device=DEVICE).unsqueeze(0)
                    pos_t = torch.tensor([pos], dtype=torch.long, device=DEVICE)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                        logits = model(ctx_t, pos_t)

                    target = torch.tensor([col[t]], dtype=torch.long, device=DEVICE)
                    loss = F.cross_entropy(logits[0, -1:, :config.vocab_size], target)
                    total_nll += loss.item()
                    total_tokens += 1

    bits = total_nll / total_tokens / math.log(2)
    print(f"  Eval bits/token: {bits:.3f}", flush=True)
    print(f"  Theoretical compression: {10/bits:.3f}x", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='small', choices=list(CONFIGS_V2.keys()))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--samples-per-epoch', type=int, default=500000)
    parser.add_argument('--n-segments', type=int, default=5000)
    args = parser.parse_args()
    train(args)

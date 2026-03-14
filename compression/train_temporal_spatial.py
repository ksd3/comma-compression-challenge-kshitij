#!/usr/bin/env python3
"""Train temporal-spatial model for commaVQ compression."""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_spatial_model import (
    TemporalSpatialModel, TSConfig, TS_CONFIGS,
    TEMPORAL_NEIGHBOR_MAP, SPATIAL_NEIGHBOR_MAP
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TSDataset(torch.utils.data.Dataset):
    """Yields (temporal_context, spatial_context, target, position) samples.

    temporal_context: (K, 1+4) — center + 4 temporal neighbors for K timesteps
    spatial_context: (4,) — current-frame raster-available neighbors (or 1024 if unavailable)
    target: (K,) — next token at center position for K timesteps
    position: spatial position ID
    """
    def __init__(self, segments, config, samples_per_epoch=200000):
        self.segments = segments
        self.config = config
        self.K = config.context_len
        self.samples_per_epoch = samples_per_epoch
        self.temporal_neighbors = TEMPORAL_NEIGHBOR_MAP
        self.spatial_neighbors = SPATIAL_NEIGHBOR_MAP
        self.n_spatial = config.n_spatial_neighbors
        self.unavail_token = config.vocab_size  # 1024

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]  # (1200, 128)
        pos = np.random.randint(128)
        t_neighbors = self.temporal_neighbors[pos]
        s_neighbors = self.spatial_neighbors[pos]

        t = np.random.randint(0, seg.shape[0] - self.K)

        # Temporal context: center + 4 temporal neighbors for K timesteps
        center = seg[t:t + self.K, pos:pos + 1]  # (K, 1)
        temporal_neigh = seg[t:t + self.K][:, t_neighbors]  # (K, 4)
        temporal_ctx = np.concatenate([center, temporal_neigh], axis=1).astype(np.int64)  # (K, 5)

        # Spatial context: current-frame raster neighbors at each target timestep
        # For training, we use the REAL current-frame values (teacher forcing)
        # Target timestep is t+1, t+2, ..., t+K
        # But we need to be careful: the spatial context should be from the TARGET frame
        # Actually for the autoregressive setup, during both training and inference,
        # we condition on the actual decoded values at the target frame.
        # So spatial_context[i] = tokens at spatial neighbors of pos in frame t+1+i
        # We use K spatial contexts, one per target timestep

        spatial_ctx = np.full((self.K, self.n_spatial), self.unavail_token, dtype=np.int64)
        for k in range(self.K):
            frame = t + 1 + k  # target frame
            for j in range(self.n_spatial):
                if s_neighbors[j] >= 0:
                    spatial_ctx[k, j] = seg[frame, s_neighbors[j]]

        # Target: next token at center position
        target = seg[t + 1:t + self.K + 1, pos].astype(np.int64)  # (K,)

        return (torch.from_numpy(temporal_ctx),
                torch.from_numpy(spatial_ctx),
                torch.from_numpy(target),
                pos)


def train(args):
    config = TS_CONFIGS[args.size]
    model = TemporalSpatialModel(config).to(DEVICE)
    n_params = model.count_params()
    print(f"Model: {args.size} (dim={config.dim}, layers={config.n_layer})", flush=True)
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

    dataset = TSDataset(segments, config, samples_per_epoch=args.samples_per_epoch)
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

        for batch_idx, (temporal_ctx, spatial_ctx, target, positions) in enumerate(loader):
            temporal_ctx = temporal_ctx.to(DEVICE)
            spatial_ctx = spatial_ctx.to(DEVICE)
            target = target.to(DEVICE)
            positions = positions.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                # spatial_ctx is (batch, K, n_spatial) — we need to pick per-timestep
                # For the model, we pass spatial context for each timestep
                # But the model currently takes (batch, n_spatial) as a single context
                # We need to modify to handle per-timestep spatial context

                # For now, use spatial context from the LAST target timestep
                # This is a simplification — we could do better with per-timestep spatial
                logits = model(temporal_ctx, spatial_ctx[:, -1, :], positions)
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
            save_path = os.path.join(os.path.dirname(__file__), f'ts_{args.size}.pt')
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

    # Quick eval
    print("\nEvaluating on 3 segments...", flush=True)
    model.eval()
    eval_segs = segments[:3]
    total_nll = 0.0
    total_tokens = 0
    K = config.context_len

    with torch.no_grad():
        for seg in eval_segs:
            for pos in range(128):
                t_neighbors = TEMPORAL_NEIGHBOR_MAP[pos]
                s_neighbors = SPATIAL_NEIGHBOR_MAP[pos]

                for t in range(K, min(K + 200, seg.shape[0])):  # Test 200 frames
                    # Temporal context
                    center = seg[t-K:t, pos:pos+1]
                    temp_neigh = seg[t-K:t][:, t_neighbors]
                    ctx = np.concatenate([center, temp_neigh], axis=1).astype(np.int64)
                    ctx_t = torch.tensor(ctx, dtype=torch.long, device=DEVICE).unsqueeze(0)

                    # Spatial context (current frame)
                    spatial = np.full(config.n_spatial_neighbors, config.vocab_size, dtype=np.int64)
                    for j in range(config.n_spatial_neighbors):
                        if s_neighbors[j] >= 0:
                            spatial[j] = seg[t, s_neighbors[j]]
                    spatial_t = torch.tensor(spatial, dtype=torch.long, device=DEVICE).unsqueeze(0)

                    pos_t = torch.tensor([pos], dtype=torch.long, device=DEVICE)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                        logits = model(ctx_t, spatial_t, pos_t)

                    target_val = torch.tensor([seg[t, pos]], dtype=torch.long, device=DEVICE)
                    loss = F.cross_entropy(logits[0, -1:, :config.vocab_size], target_val)
                    total_nll += loss.item()
                    total_tokens += 1

    bits = total_nll / total_tokens / math.log(2)
    print(f"  Eval bits/token: {bits:.3f}", flush=True)
    print(f"  Theoretical compression: {10/bits:.3f}x", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='small', choices=list(TS_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--samples-per-epoch', type=int, default=500000)
    parser.add_argument('--n-segments', type=int, default=5000)
    args = parser.parse_args()
    train(args)

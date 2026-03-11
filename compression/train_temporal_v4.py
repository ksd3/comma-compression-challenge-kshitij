#!/usr/bin/env python3
"""Train per-position temporal model v4 (4 raster spatial neighbors) for commaVQ compression."""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.temporal_model_v4 import (
    TemporalModelV4, TemporalV4Config, CONFIGS_V4,
    TEMPORAL_NEIGHBOR_MAP, SPATIAL_NEIGHBOR_MAP
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TemporalV4Dataset(torch.utils.data.Dataset):
    """Yields (temporal_context, spatial_context, target, position) samples.

    Only the LAST timestep's spatial context is correct (matches above_token logic).
    Loss should only be computed on the last timestep.
    """
    def __init__(self, segments, config, samples_per_epoch=200000):
        self.segments = segments
        self.config = config
        self.K = config.context_len
        self.samples_per_epoch = samples_per_epoch
        self.t_neighbors = TEMPORAL_NEIGHBOR_MAP
        self.s_neighbors = SPATIAL_NEIGHBOR_MAP
        self.unavail = config.vocab_size  # 1024

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]
        pos = np.random.randint(128)
        t_neigh = self.t_neighbors[pos]
        s_neigh = self.s_neighbors[pos]

        t = np.random.randint(0, seg.shape[0] - self.K)

        # Temporal context: center + 4 temporal neighbors for K timesteps
        center = seg[t:t + self.K, pos:pos + 1]
        temp_neigh = seg[t:t + self.K][:, t_neigh]
        temporal_ctx = np.concatenate([center, temp_neigh], axis=1).astype(np.int64)

        # Target: next token at center for K timesteps
        target = seg[t + 1:t + self.K + 1, pos].astype(np.int64)

        # Spatial context: 3 above-row neighbors in the LAST target frame (t+K)
        n_spatial = self.config.n_spatial_neighbors
        spatial_ctx = np.full(n_spatial, self.unavail, dtype=np.int64)
        target_frame = t + self.K
        for j in range(n_spatial):
            if s_neigh[j] >= 0:
                spatial_ctx[j] = seg[target_frame, s_neigh[j]]

        return (torch.from_numpy(temporal_ctx),
                torch.from_numpy(spatial_ctx),
                torch.from_numpy(target),
                pos)


def train(args):
    config = CONFIGS_V4[args.size]
    model = TemporalModelV4(config).to(DEVICE)
    n_params = model.count_params()
    print(f"Model: {args.size} (dim={config.dim}, layers={config.n_layer}, "
          f"spatial={config.n_spatial_neighbors})", flush=True)
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

    dataset = TemporalV4Dataset(segments, config, samples_per_epoch=args.samples_per_epoch)
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
                logits = model(temporal_ctx, spatial_ctx, positions)
                # Only last timestep where spatial context is correct
                loss = F.cross_entropy(
                    logits[:, -1, :config.vocab_size],
                    target[:, -1]
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_tokens = target.shape[0]
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
            save_path = os.path.join(os.path.dirname(__file__), f'temporal_v4_{args.size}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_size': args.size,
                'config': config,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'bits': bits,
                'version': 4,
            }, save_path)
            print(f"  Saved best model ({bits:.3f} bits/token)", flush=True)

    # Eval with proper raster-order decoding
    print("\nEvaluating (raster-order) on 3 segments, 200 frames...", flush=True)
    model.eval()
    eval_segs = segments[:3]
    total_nll = 0.0
    total_tokens = 0
    K = config.context_len

    with torch.no_grad():
        for seg in eval_segs:
            for frame_idx in range(K, min(K + 200, seg.shape[0])):
                decoded_frame = np.zeros(128, dtype=np.int16)
                for row in range(8):
                    for col in range(16):
                        pos = row * 16 + col
                        s_neigh = SPATIAL_NEIGHBOR_MAP[pos]

                        # Temporal context
                        center = seg[frame_idx-K:frame_idx, pos:pos+1]
                        t_neigh = seg[frame_idx-K:frame_idx][:, TEMPORAL_NEIGHBOR_MAP[pos]]
                        ctx = np.concatenate([center, t_neigh], axis=1).astype(np.int64)
                        ctx_t = torch.tensor(ctx, dtype=torch.long, device=DEVICE).unsqueeze(0)

                        # Spatial context (above-row neighbors, already decoded)
                        spatial = np.full(config.n_spatial_neighbors, config.vocab_size, dtype=np.int64)
                        for j in range(config.n_spatial_neighbors):
                            if s_neigh[j] >= 0:
                                spatial[j] = decoded_frame[s_neigh[j]]
                        spatial_t = torch.tensor(spatial, dtype=torch.long, device=DEVICE).unsqueeze(0)

                        pos_t = torch.tensor([pos], dtype=torch.long, device=DEVICE)

                        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                            logits = model(ctx_t, spatial_t, pos_t)

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
    parser.add_argument('--size', default='small', choices=list(CONFIGS_V4.keys()))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--samples-per-epoch', type=int, default=500000)
    parser.add_argument('--n-segments', type=int, default=5000)
    args = parser.parse_args()
    train(args)

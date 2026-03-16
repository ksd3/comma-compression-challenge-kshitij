#!/usr/bin/env python3
"""Train row-level model for commaVQ compression."""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.row_model import RowModel, RowModelConfig, ROW_CONFIGS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RowDataset(torch.utils.data.Dataset):
    def __init__(self, segments, config, samples_per_epoch=200000):
        self.segments = segments
        self.K = config.n_prev_frames
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seg_idx = np.random.randint(len(self.segments))
        seg = self.segments[seg_idx]
        t = np.random.randint(self.K, seg.shape[0])

        prev_frames = seg[t - self.K:t].astype(np.int64)  # (K, 128)
        curr_frame = seg[t].astype(np.int64)                # (128,)

        return torch.from_numpy(prev_frames), torch.from_numpy(curr_frame)


def train(args):
    config = ROW_CONFIGS[args.size]
    model = RowModel(config).to(DEVICE)
    n_params = model.count_params()
    print(f"Model: {args.size} (dim={config.dim}, layers={config.n_layer}, K={config.n_prev_frames})", flush=True)
    print(f"Params: {n_params:,}", flush=True)

    print("Loading data...", flush=True)
    from datasets import load_dataset
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    segments = []
    for i, example in enumerate(ds['train']):
        if i >= args.n_segments:
            break
        segments.append(np.array(example['token.npy']).reshape(1200, 128))
    print(f"Loaded {len(segments)} segments", flush=True)

    dataset = RowDataset(segments, config, samples_per_epoch=args.samples_per_epoch)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch

    warmup_steps = min(1000, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Training: {args.epochs} epochs x {steps_per_epoch} steps", flush=True)

    best_loss = float('inf')
    global_step = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (prev_frames, curr_frame) in enumerate(loader):
            prev_frames = prev_frames.to(DEVICE)
            curr_frame = curr_frame.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                logits = model(prev_frames, curr_frame)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    curr_frame.view(-1)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_tokens = curr_frame.numel()
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
            save_path = os.path.join(os.path.dirname(__file__), f'row_{args.size}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_size': args.size,
                'config': config,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'bits': bits,
            }, save_path)
            print(f"  Saved best model ({bits:.3f} bits/token)", flush=True)

    # Eval
    K = config.n_prev_frames
    print(f"\nEvaluating on 5 segments, 200 frames...", flush=True)
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seg_idx in range(min(5, len(segments))):
            seg = segments[seg_idx]
            for t in range(K, min(K + 200, seg.shape[0])):
                prev = torch.tensor(seg[t-K:t].astype(np.int64), device=DEVICE).unsqueeze(0)
                curr = torch.tensor(seg[t:t+1].astype(np.int64), device=DEVICE)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                    logits = model(prev, curr)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    curr.view(-1)
                )
                total_nll += loss.item() * 128
                total_tokens += 128

    bits = total_nll / total_tokens / math.log(2)
    print(f"  Eval bits/token: {bits:.3f}", flush=True)
    print(f"  Theoretical compression: {10/bits:.3f}x", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='small', choices=list(ROW_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--samples-per-epoch', type=int, default=500000)
    parser.add_argument('--n-segments', type=int, default=5000)
    args = parser.parse_args()
    train(args)

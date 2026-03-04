#!/usr/bin/env python3
"""
End-to-end evaluation harness for commaVQ compression.

1. Compresses the dataset using the specified backend
2. Creates the submission zip (ZIP_STORED, single-blob format)
3. Extracts and decompresses to verify round-trip
4. Computes and prints compression ratio

Usage:
  python compression/run_eval.py                     # default lzma backend
  python compression/run_eval.py --backend lzma_opt  # optimized lzma
  python compression/run_eval.py --backend delta_lzma
  python compression/run_eval.py --quick             # test on first 100 segments
"""
import os
import sys
import time
import shutil
import struct
import zipfile
import argparse
import tempfile
import multiprocessing
import numpy as np

from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from backends import get_backend, BACKENDS

HERE = Path(__file__).resolve().parent


def run_eval():
    parser = argparse.ArgumentParser(description="commaVQ compression evaluation harness")
    parser.add_argument("--backend", default="lzma", choices=list(BACKENDS.keys()),
                        help="Compression backend to use")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test on first 100 segments only")
    parser.add_argument("--output", default=str(HERE / "compression_challenge_submission.zip"),
                        help="Output zip path")
    args = parser.parse_args()

    backend = get_backend(args.backend)
    output_zip = Path(args.output)
    num_proc = multiprocessing.cpu_count()

    print(f"=== commaVQ Compression Eval ===")
    print(f"Backend: {backend.name}")
    print()

    # ---- Step 1: Load data ----
    print("[1/4] Loading dataset...")
    t0 = time.time()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    examples = list(ds['train'])
    if args.quick:
        examples = examples[:100]
        print(f"  Quick mode: using {len(examples)} segments")
    print(f"  Loaded {len(examples)} segments in {time.time()-t0:.1f}s")
    print()

    # ---- Step 2: Compress ----
    print("[2/4] Compressing...")
    t0 = time.time()

    names = []
    compressed_blobs = []
    total_tokens = 0

    for i, example in enumerate(examples):
        tokens = np.array(example['token.npy'])  # (1200, 8, 16)
        name = example['json']['file_name']
        compressed = backend.compress(tokens)

        names.append(name)
        compressed_blobs.append(compressed)
        total_tokens += tokens.size

        if (i + 1) % 500 == 0:
            print(f"  Compressed {i+1}/{len(examples)}")

    compress_time = time.time() - t0
    print(f"  Compressed {len(names)} segments in {compress_time:.1f}s")
    print()

    # ---- Step 3: Build zip ----
    print("[3/4] Building zip...")
    t0 = time.time()

    # Build single-blob format
    backend_bytes = args.backend.encode('utf-8')
    header = struct.pack('<B', len(backend_bytes)) + backend_bytes
    header += struct.pack('<I', len(names))

    name_table = b''
    for name in names:
        nb = name.encode('utf-8')
        name_table += struct.pack('<H', len(nb)) + nb

    size_table = b''
    for blob in compressed_blobs:
        size_table += struct.pack('<I', len(blob))

    data_blob = header + name_table + size_table + b''.join(compressed_blobs)

    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('data.bin', data_blob)
        zf.write(HERE / 'decompress.py', 'decompress.py')

    zip_size = output_zip.stat().st_size
    print(f"  Zip size: {zip_size:,} bytes ({zip_size/1024/1024:.1f} MB)")
    print(f"  Built in {time.time()-t0:.1f}s")
    print()

    # ---- Step 4: Verify round-trip ----
    print("[4/4] Verifying round-trip...")
    t0 = time.time()

    # Extract zip to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(output_zip, 'r') as zf:
            zf.extractall(tmpdir)

        # Import and run the decompress.py from the extracted zip
        # (simulates what evaluate.sh does)
        decompress_dir = tmpdir / 'decompressed'
        os.makedirs(decompress_dir, exist_ok=True)

        # Parse data.bin directly to verify
        from decompress import read_data_bin, DECOMPRESS_FNS

        errors = 0
        count = 0
        for backend_name, name, blob in read_data_bin(tmpdir / 'data.bin'):
            decompress_fn = DECOMPRESS_FNS[backend_name]
            tokens = decompress_fn(blob)

            # Verify against original
            gt_tokens = np.array(examples[count]['token.npy'])
            if not np.all(tokens == gt_tokens):
                print(f"  MISMATCH: {name}")
                errors += 1
            count += 1

        if errors == 0:
            print(f"  All {count} segments verified OK in {time.time()-t0:.1f}s")
        else:
            print(f"  FAILED: {errors}/{count} segments had mismatches!")
            sys.exit(1)

    print()

    # ---- Results ----
    original_size = total_tokens * 10 / 8
    rate = original_size / zip_size
    raw_compressed_size = sum(len(b) for b in compressed_blobs)

    print("=" * 50)
    print(f"Backend:          {backend.name}")
    print(f"Segments:         {len(names)}")
    print(f"Original size:    {int(original_size):>12,} bytes ({original_size/1024/1024:.1f} MB)")
    print(f"Compressed data:  {raw_compressed_size:>12,} bytes ({raw_compressed_size/1024/1024:.1f} MB)")
    print(f"Zip file size:    {zip_size:>12,} bytes ({zip_size/1024/1024:.1f} MB)")
    print(f"Compression rate: {rate:.3f}x")
    print(f"Bits/token:       {10/rate:.2f}")
    print(f"Compress time:    {compress_time:.1f}s")
    print("=" * 50)


if __name__ == '__main__':
    run_eval()

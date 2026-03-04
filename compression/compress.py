#!/usr/bin/env python3
"""
Compress commaVQ dataset into a submission zip.

Uses a single-blob format to avoid zip metadata overhead for 5000 separate files.
Format: [4-byte num_segments] [4-byte offset per segment] [compressed data blobs]

Usage:
  python compress.py                    # default lzma backend
  python compress.py --backend lzma_opt # optimized lzma
  python compress.py --backend delta_lzma
"""
import os
import sys
import argparse
import struct
import zipfile
import multiprocessing
import numpy as np

from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from backends import get_backend, BACKENDS

HERE = Path(__file__).resolve().parent


def compress_all(backend_name="lzma"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=backend_name, choices=list(BACKENDS.keys()))
    parser.add_argument("--output", default=str(HERE / "compression_challenge_submission.zip"))
    args = parser.parse_args()

    backend = get_backend(args.backend)
    output_zip = Path(args.output)
    num_proc = multiprocessing.cpu_count()

    print(f"Backend: {backend.name}")

    # Load data
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

    # Collect all segments with their names
    names = []
    compressed_blobs = []
    total_tokens = 0

    for i, example in enumerate(ds['train']):
        tokens = np.array(example['token.npy'])  # (1200, 8, 16)
        name = example['json']['file_name']
        compressed = backend.compress(tokens)

        names.append(name)
        compressed_blobs.append(compressed)
        total_tokens += tokens.size

        if (i + 1) % 500 == 0:
            print(f"  Compressed {i+1}/{len(ds['train'])} segments")

    print(f"Compressed {len(names)} segments")

    # Build single-blob format:
    # [1 byte: backend_name_len] [backend_name bytes]
    # [4 bytes: num_segments]
    # [num_segments * name entries: 2-byte name_len + name_bytes]
    # [num_segments * 4-byte blob_size]
    # [concatenated blobs]
    num_segments = len(names)
    backend_bytes = args.backend.encode('utf-8')
    header = struct.pack('<B', len(backend_bytes)) + backend_bytes
    header += struct.pack('<I', num_segments)

    # Name table
    name_table = b''
    for name in names:
        name_bytes = name.encode('utf-8')
        name_table += struct.pack('<H', len(name_bytes)) + name_bytes

    # Size table
    size_table = b''
    for blob in compressed_blobs:
        size_table += struct.pack('<I', len(blob))

    # Concatenate everything
    data_blob = header + name_table + size_table + b''.join(compressed_blobs)

    # Create zip with ZIP_STORED (no double compression)
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('data.bin', data_blob)
        # Include decompress.py
        zf.write(HERE / 'decompress.py', 'decompress.py')

    zip_size = output_zip.stat().st_size
    original_size = total_tokens * 10 / 8  # 10 bits per token
    rate = original_size / zip_size

    print(f"Zip size: {zip_size:,} bytes")
    print(f"Original size: {int(original_size):,} bytes")
    print(f"Compression rate: {rate:.2f}x")

    return rate


if __name__ == '__main__':
    compress_all()

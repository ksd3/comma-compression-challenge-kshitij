#!/usr/bin/env python3
"""
Decompress commaVQ submission.
Reads single-blob format from data.bin and writes .npy files.

This file is included in the submission zip and run by evaluate.sh.
"""
import os
import sys
import lzma
import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get('OUTPUT_DIR', HERE / 'compression_challenge_submission_decompressed'))


# ---- Backend decompression functions ----
# The backend used is encoded in data.bin header (or we default to lzma)

def decompress_lzma(data: bytes) -> np.ndarray:
    tokens = np.frombuffer(lzma.decompress(data), dtype=np.int16)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)

def decompress_lzma_opt(data: bytes) -> np.ndarray:
    filters = [
        {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 23}
    ]
    tokens = np.frombuffer(lzma.decompress(data, format=lzma.FORMAT_RAW, filters=filters), dtype=np.int16)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)

def decompress_delta_lzma(data: bytes) -> np.ndarray:
    delta = np.frombuffer(lzma.decompress(data), dtype=np.int16)
    delta = delta.reshape(128, -1).T
    flat = np.cumsum(delta, axis=0).astype(np.int16)
    return flat.reshape(-1, 8, 16)

def decompress_zstd(data: bytes) -> np.ndarray:
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    tokens = np.frombuffer(dctx.decompress(data), dtype=np.int16)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)

def _unpack10(data):
    """Unpack 10-bit values from bytes."""
    n = struct.unpack_from('<I', data, 0)[0]
    packed = data[4:]
    result = []
    for i in range(0, len(packed), 5):
        if i + 4 >= len(packed):
            break
        b0, b1, b2, b3, b4 = packed[i], packed[i+1], packed[i+2], packed[i+3], packed[i+4]
        a = b0 | ((b1 & 0x03) << 8)
        b = ((b1 >> 2) & 0x3F) | ((b2 & 0x0F) << 6)
        c = ((b2 >> 4) & 0x0F) | ((b3 & 0x3F) << 4)
        d = ((b3 >> 6) & 0x03) | (b4 << 2)
        result.extend([a, b, c, d])
    return np.array(result[:n], dtype=np.int16)

def decompress_bitpack_lzma(data: bytes) -> np.ndarray:
    packed = lzma.decompress(data)
    flat = _unpack10(packed)
    return flat.reshape(128, -1).T.reshape(-1, 8, 16)


# Map backend name to decompress function
DECOMPRESS_FNS = {
    "lzma": decompress_lzma,
    "lzma_opt": decompress_lzma_opt,
    "delta_lzma": decompress_delta_lzma,
    "zstd": decompress_zstd,
    "bitpack_lzma": decompress_bitpack_lzma,
}


def read_data_bin(data_bin_path: Path):
    """Parse the single-blob format and yield (backend_name, name, raw_bytes) tuples."""
    with open(data_bin_path, 'rb') as f:
        raw = f.read()

    offset = 0

    # Read backend name
    backend_name_len, = struct.unpack_from('<B', raw, offset)
    offset += 1
    backend_name = raw[offset:offset + backend_name_len].decode('utf-8')
    offset += backend_name_len

    # Read num_segments
    num_segments, = struct.unpack_from('<I', raw, offset)
    offset += 4

    # Read name table
    names = []
    for _ in range(num_segments):
        name_len, = struct.unpack_from('<H', raw, offset)
        offset += 2
        name = raw[offset:offset + name_len].decode('utf-8')
        offset += name_len
        names.append(name)

    # Read size table
    sizes = []
    for _ in range(num_segments):
        size, = struct.unpack_from('<I', raw, offset)
        offset += 4
        sizes.append(size)

    # Read blobs
    for name, size in zip(names, sizes):
        blob = raw[offset:offset + size]
        offset += size
        yield backend_name, name, blob


def decompress_all():
    """Decompress all segments from data.bin and save as .npy files."""
    data_bin = HERE / 'data.bin'

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    decompress_fn = None
    for backend_name, name, blob in read_data_bin(data_bin):
        if decompress_fn is None:
            decompress_fn = DECOMPRESS_FNS[backend_name]
            print(f"Backend: {backend_name}")
        tokens = decompress_fn(blob)
        np.save(output_dir / name, tokens)
        count += 1
        if count % 500 == 0:
            print(f"  Decompressed {count} segments")

    print(f"Decompressed {count} segments total")


if __name__ == '__main__':
    decompress_all()

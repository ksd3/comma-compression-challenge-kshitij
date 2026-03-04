#!/usr/bin/env python3
"""
Pluggable compression backends for commaVQ challenge.

Each backend implements:
  compress(tokens: np.ndarray) -> bytes     # tokens shape: (1200, 8, 16) int16
  decompress(data: bytes) -> np.ndarray     # returns (1200, 8, 16) int16
"""
import lzma
import struct
import numpy as np


class LZMABackend:
    """Baseline LZMA compression (~1.6x). Transpose trick for better compression."""

    name = "lzma"

    @staticmethod
    def compress(tokens: np.ndarray) -> bytes:
        flat = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
        return lzma.compress(flat)

    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        tokens = np.frombuffer(lzma.decompress(data), dtype=np.int16)
        return tokens.reshape(128, -1).T.reshape(-1, 8, 16)


class LZMAOptBackend:
    """Optimized LZMA with tuned parameters."""

    name = "lzma_opt"

    @staticmethod
    def compress(tokens: np.ndarray) -> bytes:
        flat = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
        # Use extreme compression with large dictionary
        filters = [
            {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 23}
        ]
        return lzma.compress(flat, format=lzma.FORMAT_RAW, filters=filters)

    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        filters = [
            {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 23}
        ]
        tokens = np.frombuffer(lzma.decompress(data, format=lzma.FORMAT_RAW, filters=filters), dtype=np.int16)
        return tokens.reshape(128, -1).T.reshape(-1, 8, 16)


class DeltaLZMABackend:
    """Delta encoding + LZMA. Exploits frame-to-frame correlation."""

    name = "delta_lzma"

    @staticmethod
    def compress(tokens: np.ndarray) -> bytes:
        # Shape: (1200, 8, 16) -> (1200, 128)
        flat = tokens.astype(np.int16).reshape(-1, 128)
        # Delta encode along frames (temporal prediction)
        delta = np.zeros_like(flat)
        delta[0] = flat[0]
        delta[1:] = flat[1:] - flat[:-1]
        # Transpose and compress
        data = delta.T.ravel().tobytes()
        return lzma.compress(data)

    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        delta = np.frombuffer(lzma.decompress(data), dtype=np.int16)
        delta = delta.reshape(128, -1).T  # (1200, 128)
        # Undo delta
        flat = np.cumsum(delta, axis=0).astype(np.int16)
        return flat.reshape(-1, 8, 16)


class ZstdBackend:
    """Zstandard compression with max level."""

    name = "zstd"

    @staticmethod
    def compress(tokens: np.ndarray) -> bytes:
        import zstandard as zstd
        flat = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
        cctx = zstd.ZstdCompressor(level=22)
        return cctx.compress(flat)

    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        tokens = np.frombuffer(dctx.decompress(data), dtype=np.int16)
        return tokens.reshape(128, -1).T.reshape(-1, 8, 16)


class BitpackLZMABackend:
    """Pack 10-bit tokens tightly before LZMA compression."""

    name = "bitpack_lzma"

    @staticmethod
    def compress(tokens: np.ndarray) -> bytes:
        flat = tokens.astype(np.int16).reshape(-1, 128).T.ravel()
        # Pack 10-bit values: 4 tokens = 40 bits = 5 bytes
        packed = BitpackLZMABackend._pack10(flat)
        return lzma.compress(packed)

    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        packed = lzma.decompress(data)
        flat = BitpackLZMABackend._unpack10(packed)
        return flat.reshape(128, -1).T.reshape(-1, 8, 16)

    @staticmethod
    def _pack10(values):
        """Pack array of 10-bit values into bytes."""
        n = len(values)
        # Pad to multiple of 4
        pad = (4 - n % 4) % 4
        if pad:
            values = np.concatenate([values, np.zeros(pad, dtype=np.int16)])
        values = values.astype(np.uint16)
        result = bytearray()
        for i in range(0, len(values), 4):
            a, b, c, d = int(values[i]), int(values[i+1]), int(values[i+2]), int(values[i+3])
            # 4 x 10 bits = 40 bits = 5 bytes
            result.append(a & 0xFF)
            result.append(((a >> 8) & 0x03) | ((b & 0x3F) << 2))
            result.append(((b >> 6) & 0x0F) | ((c & 0x0F) << 4))
            result.append(((c >> 4) & 0x3F) | ((d & 0x03) << 6))
            result.append((d >> 2) & 0xFF)
        # Prepend original length
        return struct.pack('<I', n) + bytes(result)

    @staticmethod
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


# Registry of available backends
BACKENDS = {
    "lzma": LZMABackend,
    "lzma_opt": LZMAOptBackend,
    "delta_lzma": DeltaLZMABackend,
    "zstd": ZstdBackend,
    "bitpack_lzma": BitpackLZMABackend,
}

def get_backend(name: str):
    return BACKENDS[name]

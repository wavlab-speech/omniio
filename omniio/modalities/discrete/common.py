"""Byte layout and bit-packing for the ``discrete`` modality.

    header  (little-endian)
        b"ODSQ"        magic
        u8   version   (1)
        u8   flags     bit0: payload is zstd-compressed
        u16  n_streams
        per stream:  u32 length | u8 bits | u32 vocab (0 = unknown) | f32 rate (0 = none)
    payload
        stream 0 bit-packed (little-endian bit order), byte-aligned, then stream 1, ...
        (per-stream alignment lets a reader unpack a subset of streams from header offsets)

Streams are stored in the order given (codebook-major for RVQ). Values are unsigned;
``bits`` is the narrowest width holding ``vocab - 1``.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MAGIC = b"ODSQ"
VERSION = 1
FLAG_ZSTD = 1
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_HEAD = struct.Struct("<4sBBH")
_STREAM = struct.Struct("<IBIf")


@dataclass
class StreamInfo:
    length: int
    bits: int
    vocab: int              # 0 = unknown
    rate: float             # 0.0 = none

    @property
    def packed_bytes(self) -> int:
        return (self.length * self.bits + 7) // 8


def dtype_for(bits: int) -> np.dtype:
    return np.dtype(np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32 if bits <= 32 else np.uint64)


MAX_PACKED_BITS = 56    # the unpacker reads 8-byte windows starting mid-byte; wider values store as u64


def bits_for(vocab: int) -> int:
    b = max(1, int(math.ceil(math.log2(int(vocab)))) if vocab > 1 else 1)
    return 64 if b > MAX_PACKED_BITS else b


# ------------------------------------------------------------------ bit packing
def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Little-endian bit stream of ``bits`` per value (values must fit)."""
    a = np.ascontiguousarray(values, dtype=np.uint64).ravel()
    n = int(a.size)
    if n == 0:
        return b""
    if bits < 64 and int(a.max()) >= (1 << bits):
        raise ValueError(f"value {int(a.max())} does not fit in {bits} bits")
    if bits in (8, 16, 32, 64):                        # byte-aligned widths: a plain cast
        return a.astype(f"<u{bits // 8}").tobytes()
    if not 1 <= bits <= MAX_PACKED_BITS:
        raise ValueError(f"bits must be in [1, {MAX_PACKED_BITS}] or 64, got {bits}")
    out = np.zeros((n * bits + 7) // 8, dtype=np.uint8)
    bitpos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    for b in range(bits):                              # scatter bit b of every value
        pos = bitpos + np.uint64(b)
        np.bitwise_or.at(out, (pos >> np.uint64(3)).astype(np.int64),
                         (((a >> np.uint64(b)) & np.uint64(1)) << (pos & np.uint64(7))).astype(np.uint8))
    return out.tobytes()


def unpack_bits(buf: bytes, bits: int, n: int) -> np.ndarray:
    """Inverse of `pack_bits` -> the narrowest unsigned dtype for ``bits``."""
    if n == 0:
        return np.zeros(0, dtype=dtype_for(bits))
    if bits in (8, 16, 32, 64):
        return np.frombuffer(buf, dtype=f"<u{bits // 8}", count=n).astype(dtype_for(bits), copy=False)
    if not 1 <= bits <= MAX_PACKED_BITS:
        raise ValueError(f"bits must be in [1, {MAX_PACKED_BITS}] or 64, got {bits}")
    need = (n * bits + 7) // 8
    if len(buf) < need:
        raise ValueError(f"packed stream is {len(buf)} bytes, need {need}")
    raw = np.frombuffer(buf, dtype=np.uint8, count=need)
    # every value lies within an 8-byte little-endian window starting at its first byte
    padded = np.concatenate([raw, np.zeros(8, dtype=np.uint8)])
    bitpos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    first = (bitpos >> np.uint64(3)).astype(np.int64)
    win = np.lib.stride_tricks.as_strided(padded, shape=(need + 1, 8), strides=(1, 1))[first]     # (n, 8) bytes
    words = win.copy().view("<u8").ravel()
    vals = (words >> (bitpos & np.uint64(7))) & np.uint64((1 << bits) - 1)
    return vals.astype(dtype_for(bits))


# ------------------------------------------------------------------ encode / decode
def encode(streams: Sequence[np.ndarray], vocab_sizes: Sequence[int], rates: Sequence[float], *,
           compress: bool = False, compression_level: int = 3) -> tuple[bytes, list[StreamInfo]]:
    if not (len(streams) == len(vocab_sizes) == len(rates)):
        raise ValueError("streams, vocab_sizes and rates must have the same length")
    infos, chunks = [], []
    for s, v, r in zip(streams, vocab_sizes, rates):
        a = np.asarray(s)
        if a.ndim != 1:
            raise ValueError(f"each stream must be 1-D, got shape {a.shape}")
        if a.size and (not np.issubdtype(a.dtype, np.integer) or int(a.min()) < 0):
            raise ValueError("streams must hold non-negative integers")
        vmax = int(a.max()) if a.size else 0
        vocab = int(v) if v else vmax + 1
        if vmax >= vocab:
            raise ValueError(f"value {vmax} outside the alphabet of size {vocab}")
        bits = bits_for(vocab)
        infos.append(StreamInfo(length=int(a.size), bits=bits, vocab=vocab, rate=float(r or 0.0)))
        chunks.append(pack_bits(a, bits))
    payload = b"".join(chunks)
    flags = 0
    if compress:
        import zstandard as zstd
        payload = zstd.ZstdCompressor(level=compression_level).compress(payload)
        flags |= FLAG_ZSTD
    head = _HEAD.pack(MAGIC, VERSION, flags, len(infos)) + b"".join(
        _STREAM.pack(i.length, i.bits, i.vocab, i.rate) for i in infos)
    return head + payload, infos


def read_header(blob: bytes) -> tuple[list[StreamInfo], int, int]:
    """-> (stream infos, payload offset, flags)."""
    if len(blob) < _HEAD.size or blob[:4] != MAGIC:
        raise ValueError("not an omniio discrete-sequence entry (bad magic)")
    _, version, flags, n = _HEAD.unpack_from(blob, 0)
    if version != VERSION:
        raise ValueError(f"unsupported discrete format version {version}")
    infos, off = [], _HEAD.size
    for _ in range(n):
        length, bits, vocab, rate = _STREAM.unpack_from(blob, off)
        infos.append(StreamInfo(length=length, bits=bits, vocab=vocab, rate=rate))
        off += _STREAM.size
    return infos, off, flags


def decode(blob: bytes, streams: Optional[Sequence[int]] = None) -> tuple[list[StreamInfo], list[Optional[np.ndarray]]]:
    """Unpack the requested stream indices (all by default). Unrequested streams come
    back as None so positions stay aligned with the header."""
    infos, off, flags = read_header(blob)
    payload = blob[off:]
    if flags & FLAG_ZSTD:
        import zstandard as zstd
        payload = zstd.ZstdDecompressor().decompress(payload)
    want = set(range(len(infos))) if streams is None else set(int(i) for i in streams)
    out: list[Optional[np.ndarray]] = []
    pos = 0
    for k, info in enumerate(infos):
        nbytes = info.packed_bytes
        out.append(unpack_bits(payload[pos: pos + nbytes], info.bits, info.length) if k in want else None)
        pos += nbytes
    return infos, out


def slice_indices(info: StreamInfo, start_time: Optional[float], end_time: Optional[float],
                  start_frame: Optional[int], end_frame: Optional[int]) -> tuple[int, int]:
    """[start, end) element range of a stream. Frames (elements) take priority over
    seconds, as in the video reader; seconds map through the stream's own rate (floor
    start, ceil end)."""
    lo, hi = 0, info.length
    if start_frame is not None:
        lo = max(lo, int(start_frame))
    elif start_time is not None:
        if info.rate <= 0:
            raise ValueError("time-based slicing needs a per-stream rate (units per second)")
        lo = max(lo, int(math.floor(start_time * info.rate)))
    if end_frame is not None:
        hi = min(hi, int(end_frame))
    elif end_time is not None:
        if info.rate <= 0:
            raise ValueError("time-based slicing needs a per-stream rate (units per second)")
        hi = min(hi, int(math.ceil(end_time * info.rate)))
    return lo, max(lo, hi)

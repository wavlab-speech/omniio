"""Byte layout of a ``track`` entry.

    header (little-endian, 24 bytes)
        b"OTRK"  magic
        u8  version (1)
        u8  dtype   1 = float16, 2 = float32
        u16 reserved
        u32 bin     (bp per bin)
        u64 n_bins
        u32 reserved
    payload
        n_bins values, bin k = mean of the signal over bases [k*bin, (k+1)*bin)
        (NaN where the track has no coverage over the whole bin)

Binning is done exactly from the base-pair values at ingestion (``nanmean`` per bin), not
from the bigWig's zoom levels. A window ``[start, end)`` in bp maps to bins
``[start // bin, ceil(end / bin))`` = one contiguous read.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"OTRK"
VERSION = 1
_HEAD = struct.Struct("<4sBBHIQI")
HEADER_SIZE = _HEAD.size     # 24
DTYPES = {1: np.dtype("<f2"), 2: np.dtype("<f4")}
_CODES = {"float16": 1, "f2": 1, "float32": 2, "f4": 2}


@dataclass
class TrackHeader:
    version: int
    dtype_code: int
    bin: int
    n_bins: int

    @property
    def dtype(self) -> np.dtype:
        return DTYPES[self.dtype_code]

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize


def parse_header(buf: bytes) -> TrackHeader:
    if len(buf) < HEADER_SIZE:
        raise ValueError("track entry too short for a header")
    magic, version, dt, _r, bin_, n_bins, _r2 = _HEAD.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"not a track entry (magic {magic!r})")
    if version != VERSION or dt not in DTYPES:
        raise ValueError(f"unsupported track entry (version {version}, dtype {dt})")
    return TrackHeader(version, dt, bin_, n_bins)


def bin_values(values: np.ndarray, bin: int) -> np.ndarray:
    """Exact per-bin ``nanmean`` of a base-resolution signal (``values`` may hold NaN for
    uncovered bases); the last partial bin averages the bases it has. -> float64."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    n_bins = (n + bin - 1) // bin
    pad = n_bins * bin - n
    if pad:
        v = np.concatenate([v, np.full(pad, np.nan)])
    v = v.reshape(n_bins, bin)
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(v), axis=1)
        s = np.nansum(v, axis=1)
        out = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def encode(binned: np.ndarray, bin: int, dtype: str = "float16") -> tuple[bytes, TrackHeader]:
    code = _CODES[dtype] if isinstance(dtype, str) else int(dtype)
    a = np.ascontiguousarray(np.asarray(binned), dtype=DTYPES[code])
    head = TrackHeader(VERSION, code, int(bin), int(a.size))
    return _HEAD.pack(MAGIC, VERSION, code, 0, int(bin), int(a.size), 0) + a.tobytes(), head

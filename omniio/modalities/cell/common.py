"""Byte layout and codec for the ``cell`` modality: one sparse row per entry.

    header (little-endian, 15 bytes)
        b"OCEL"  magic
        u8   version      (1)
        u8   value_dtype  code from VALUE_DTYPES (1 u8, 2 u16, 3 u32, 4 f32, 5 f64)
        u8   flags        bit0 RAW_INDICES  indices are plain u32 (not delta varints)
                          bit1 ZSTD         payload (indices + values) is zstd-compressed
                          bit2 DENSE        no indices; values holds every column
        u32  nnz          stored entries (== n_cols for a dense entry)
        u32  n_cols       width of the row (the archive's gene / peak / marker count)
    payload
        indices  nnz x column index, strictly increasing, as delta-coded LEB128 varints
                 (first index as is, then the gaps) -- or nnz x u32 with RAW_INDICES
        values   nnz x value_dtype

Explicit zeros are dropped at write time, so ``nnz`` counts true non-zeros. A row is
stored dense when the writer is told to (``layout="dense"``) or, under ``layout="auto"``,
when more than half of its columns are non-zero (protein / ADT panels).

Everything here is numpy-vectorised: a 2-4 KB row decodes in tens of microseconds, which
is what makes per-cell random access worthwhile against an HDF5 chunk read.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

MAGIC = b"OCEL"
VERSION = 1
FLAG_RAW_INDICES = 1
FLAG_ZSTD = 2
FLAG_DENSE = 4
_HEAD = struct.Struct("<4sBBBII")
HEADER_SIZE = _HEAD.size          # 15

VALUE_DTYPES = {1: np.dtype("<u1"), 2: np.dtype("<u2"), 3: np.dtype("<u4"), 4: np.dtype("<f4"), 5: np.dtype("<f8")}
_DTYPE_CODES = {np.dtype(v).str: k for k, v in VALUE_DTYPES.items()}
_DTYPE_NAMES = {"uint8": 1, "uint16": 2, "uint32": 3, "float32": 4, "float64": 5,
                "u1": 1, "u2": 2, "u4": 3, "f4": 4, "f8": 5}


def auto_dtype_code(values: np.ndarray) -> int:
    """Narrowest unsigned integer code holding integral, non-negative ``values``
    (u8 / u16 / u32); float32 for anything else."""
    v = np.asarray(values)
    if v.size == 0:
        return 2
    if v.dtype.kind == "f":
        if not (np.all(np.isfinite(v)) and np.array_equal(v, np.floor(v))):
            return 4
    if v.dtype.kind in "biu" or v.dtype.kind == "f":
        mn, mx = float(v.min()), float(v.max())
        if mn < 0:
            return 4
        return 1 if mx <= 255 else 2 if mx <= 65535 else 3 if mx <= 4294967295 else 5
    return 4


def dtype_code(value_dtype: Union[str, np.dtype, int]) -> int:
    if isinstance(value_dtype, (int, np.integer)):
        if int(value_dtype) not in VALUE_DTYPES:
            raise ValueError(f"unknown value_dtype code {value_dtype}")
        return int(value_dtype)
    if isinstance(value_dtype, str) and value_dtype in _DTYPE_NAMES:
        return _DTYPE_NAMES[value_dtype]
    key = np.dtype(value_dtype).newbyteorder("<").str
    if key not in _DTYPE_CODES:
        raise ValueError(f"unsupported value_dtype {value_dtype!r}; use one of {sorted(_DTYPE_NAMES)[:5]}")
    return _DTYPE_CODES[key]


@dataclass
class RowHeader:
    version: int
    value_dtype: int
    flags: int
    nnz: int
    n_cols: int

    @property
    def dtype(self) -> np.dtype:
        return VALUE_DTYPES[self.value_dtype]

    @property
    def dense(self) -> bool:
        return bool(self.flags & FLAG_DENSE)

    @property
    def zstd(self) -> bool:
        return bool(self.flags & FLAG_ZSTD)

    @property
    def raw_indices(self) -> bool:
        return bool(self.flags & FLAG_RAW_INDICES)


def parse_header(buf: bytes) -> RowHeader:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"cell entry too short for a header ({len(buf)} bytes)")
    magic, version, vd, flags, nnz, n_cols = _HEAD.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"not a cell entry (magic {magic!r})")
    if version != VERSION:
        raise ValueError(f"unsupported cell entry version {version}")
    if vd not in VALUE_DTYPES:
        raise ValueError(f"unknown value_dtype code {vd}")
    return RowHeader(version, vd, flags, nnz, n_cols)


# ------------------------------------------------------------------ varints
def varint_encode(values: np.ndarray) -> bytes:
    """LEB128 of each non-negative value (<= 2**35), vectorised."""
    v = np.ascontiguousarray(values, dtype=np.uint64).ravel()
    n = int(v.size)
    if n == 0:
        return b""
    # bytes per value: 1 + number of 7-bit thresholds the value reaches
    nb = np.ones(n, dtype=np.int64)
    for k in range(1, 5):
        nb += (v >= np.uint64(1 << (7 * k))).astype(np.int64)
    if int(v.max()) >= (1 << 35):
        raise ValueError("varint_encode: value too large (max 2**35 - 1)")
    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    np.cumsum(nb[:-1], out=starts[1:])
    out = np.zeros(int(starts[-1] + nb[-1]), dtype=np.uint8)
    for k in range(5):
        sel = nb > k
        if not sel.any():
            break
        chunk = ((v[sel] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        more = (nb[sel] > k + 1)
        chunk[more] |= 0x80
        out[starts[sel] + k] = chunk
    return out.tobytes()


def varint_decode(buf: Union[bytes, np.ndarray], n: int) -> np.ndarray:
    """Inverse of `varint_encode`: the first ``n`` varints of ``buf`` -> uint64 array.

    Gene-index deltas are almost all one byte (a 2 000-nnz row over 30 k genes has ~1 %
    two-byte gaps), so the first byte of every value is gathered in one pass and only the
    values that continue are patched, byte position by byte position (at most 5)."""
    if n == 0:
        return np.zeros(0, dtype=np.uint64)
    b = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    cont = b >= 128
    ends = np.flatnonzero(~cont)
    if ends.size < n:
        raise ValueError(f"varint stream truncated: {ends.size} values, expected {n}")
    last = int(ends[n - 1]) + 1
    if last == n:                                        # every value is a single byte
        return b[:n].astype(np.uint64)
    low = (b[:last] & 0x7F).astype(np.uint64)
    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[: n - 1] + 1
    out = low[starts]
    multi = np.flatnonzero(ends[:n] != starts)
    width = ends[multi] - starts[multi]
    if width.size and int(width.max()) > 4:
        raise ValueError("varint wider than 35 bits")
    for k in range(1, 5):
        sel = multi[width >= k]
        if sel.size == 0:
            break
        out[sel] |= low[starts[sel] + k] << np.uint64(7 * k)
    return out


# ------------------------------------------------------------------ encode / decode
def _coerce_values(values: np.ndarray, code: int) -> np.ndarray:
    """Cast to the declared dtype, refusing casts that would change a value."""
    target = VALUE_DTYPES[code]
    v = np.asarray(values)
    if v.dtype == target:
        return v
    if target.kind == "u":
        if v.dtype.kind == "f":
            if not np.all(np.isfinite(v)):
                raise ValueError("non-finite value cannot be stored as an integer")
            if not np.array_equal(v, np.floor(v)):
                raise ValueError(f"non-integral value cannot be stored as {target.name}; use value_dtype='float32'")
        if v.size and (int(v.min()) < 0):
            raise ValueError(f"negative value cannot be stored as {target.name}")
        if v.size and int(v.max()) > int(np.iinfo(target).max):
            raise ValueError(f"value {int(v.max())} overflows {target.name}; widen value_dtype")
        return v.astype(target)
    return v.astype(target)        # float targets: plain cast


def encode_row(indices: np.ndarray, values: np.ndarray, n_cols: int, *, value_dtype="uint16",
               layout: str = "auto", raw_indices: bool = False, compress: bool = False,
               compression_level: int = 3) -> tuple[bytes, RowHeader]:
    """Pack one sparse row. ``indices`` need not be sorted; duplicates are summed; explicit
    zeros are dropped. ``layout``: 'sparse' | 'dense' | 'auto' (dense when nnz > n_cols / 2)."""
    idx = np.asarray(indices)
    val = np.asarray(values)
    code = auto_dtype_code(val) if value_dtype == "auto" else dtype_code(value_dtype)
    if idx.shape != val.shape or idx.ndim != 1:
        raise ValueError("indices and values must be 1-D and the same length")
    n_cols = int(n_cols)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_cols):
        raise ValueError(f"column index out of range for n_cols={n_cols}")
    val = _coerce_values(val, code)
    if idx.size:
        order = np.argsort(idx, kind="stable")
        idx, val = idx[order].astype(np.int64), val[order]
        if idx.size > 1 and np.any(idx[1:] == idx[:-1]):          # sum duplicates
            uniq, inv = np.unique(idx, return_inverse=True)
            summed = np.zeros(uniq.size, dtype=np.float64 if val.dtype.kind == "f" else np.uint64)
            np.add.at(summed, inv, val)
            idx, val = uniq, _coerce_values(summed, code)
        keep = val != 0
        if not keep.all():
            idx, val = idx[keep], val[keep]
    nnz = int(idx.size)
    flags = 0
    dense = layout == "dense" or (layout == "auto" and n_cols > 0 and nnz > n_cols // 2)
    if layout not in ("auto", "dense", "sparse"):
        raise ValueError("layout must be 'auto', 'sparse' or 'dense'")
    if dense:
        row = np.zeros(n_cols, dtype=VALUE_DTYPES[code])
        row[idx] = val
        payload = row.tobytes()
        flags |= FLAG_DENSE
        stored = n_cols
    else:
        if raw_indices:
            ib = idx.astype("<u4").tobytes()
            flags |= FLAG_RAW_INDICES
        else:
            deltas = np.empty(nnz, dtype=np.uint64)
            if nnz:
                deltas[0] = idx[0]
                deltas[1:] = np.diff(idx)
            ib = varint_encode(deltas)
        payload = ib + val.astype(VALUE_DTYPES[code]).tobytes()
        stored = nnz
    if compress:
        import zstandard as zstd
        payload = zstd.ZstdCompressor(level=compression_level).compress(payload)
        flags |= FLAG_ZSTD
    head = RowHeader(VERSION, code, flags, stored, n_cols)
    return _HEAD.pack(MAGIC, VERSION, code, flags, stored, n_cols) + payload, head


def decode_row(buf: bytes) -> tuple[RowHeader, np.ndarray, np.ndarray]:
    """-> (header, indices uint32 sorted, values) of a whole entry. A dense entry comes back
    as its non-zero positions (so the result is the same as for a sparse one)."""
    head = parse_header(buf)
    payload = memoryview(buf)[HEADER_SIZE:]
    if head.zstd:
        import zstandard as zstd
        payload = memoryview(zstd.ZstdDecompressor().decompress(bytes(payload)))
    if head.dense:
        row = np.frombuffer(payload, dtype=head.dtype, count=head.n_cols)
        idx = np.flatnonzero(row).astype(np.uint32)
        return head, idx, row[idx]
    nnz = head.nnz
    if head.raw_indices:
        idx = np.frombuffer(payload, dtype="<u4", count=nnz).astype(np.uint32, copy=False)
        off = 4 * nnz
    else:
        if nnz:
            b = np.frombuffer(payload, dtype=np.uint8)
            deltas = varint_decode(b, nnz)
            idx = np.cumsum(deltas, dtype=np.uint64).astype(np.uint32)
            off = int(np.flatnonzero((b & 0x80) == 0)[nnz - 1]) + 1
        else:
            idx, off = np.zeros(0, dtype=np.uint32), 0
    val = np.frombuffer(payload, dtype=head.dtype, count=nnz, offset=off)
    return head, idx, val.astype(head.dtype.newbyteorder("="), copy=False)

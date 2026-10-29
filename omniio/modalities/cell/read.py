"""Read per-cell sparse rows from an omniio archive (local pread or HTTP range)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import requests

from omniio.blob.handles import archive_fd, pread_all
from omniio.definitions import CellRead
from omniio.modalities.cell.common import decode_row


class RangeNotSupportedError(RuntimeError):
    """The server answered a ranged GET with the whole file (200) or the wrong length."""


def _make(buf: bytes, entry_size: int, bytes_read: int) -> CellRead:
    head, idx, val = decode_row(buf)
    return CellRead(file_type="cell.zst" if head.zstd else "cell", modality="cell", indices=idx, values=val,
                    n_genes=int(head.n_cols), nnz=int(idx.size), value_dtype=head.dtype.newbyteorder("=").name,
                    stored_dense=head.dense, bytes_read=bytes_read, entry_size=entry_size)


def _fetch_remote(url: str, start: int, size: int) -> bytes:
    if size <= 0:
        return b""
    resp = requests.get(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    resp.raise_for_status()
    if resp.status_code != 206:
        raise RangeNotSupportedError(f"{url}: server ignored the Range header (HTTP {resp.status_code}); "
                                     "cell entries need a range-capable server")
    if len(resp.content) != size:
        raise RangeNotSupportedError(f"{url}: asked for {size} bytes, got {len(resp.content)}")
    return resp.content


def cell_read_local(archive_path: str, start_offset: int, file_size: int) -> CellRead:
    """One cell: a single ``pread`` of the entry, decoded."""
    buf = pread_all(archive_fd(archive_path), file_size, start_offset)
    return _make(buf, file_size, len(buf))


def cell_read_remote(archive_url: str, start_offset: int, file_size: int) -> CellRead:
    """Same over one HTTP range request; a server that ignores ``Range`` raises
    `RangeNotSupportedError`."""
    buf = _fetch_remote(archive_url, start_offset, file_size)
    return _make(buf, file_size, len(buf))


# ------------------------------------------------------------------ batch
def coalesce(ranges: Sequence[tuple], max_gap: int = 64 * 1024) -> list:
    """Sort ``(start, size)`` ranges and merge those closer than ``max_gap`` bytes ->
    ``[(start, size, [original positions...]), ...]``."""
    order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
    groups = []
    for i in order:
        s, n = int(ranges[i][0]), int(ranges[i][1])
        if groups and s <= groups[-1][0] + groups[-1][1] + max_gap:
            g = groups[-1]
            g[1] = max(g[1], s + n - g[0])
            g[2].append(i)
        else:
            groups.append([s, n, [i]])
    return groups


def _batch(fetch, ranges, max_gap: int):
    import scipy.sparse as sp
    B = len(ranges)
    idx_parts: list = [None] * B
    val_parts: list = [None] * B
    n_cols = None
    dtype = None
    bytes_read = 0
    for gs, gn, members in coalesce(ranges, max_gap):
        buf = fetch(gs, gn)
        bytes_read += len(buf)
        for i in members:
            s, n = int(ranges[i][0]), int(ranges[i][1])
            head, idx, val = decode_row(buf[s - gs: s - gs + n])
            if n_cols is None:
                n_cols = head.n_cols
            elif head.n_cols != n_cols:
                raise ValueError(f"entries with different panel widths in one batch ({n_cols} and {head.n_cols})")
            dtype = val.dtype if dtype is None else np.result_type(dtype, val.dtype)
            idx_parts[i], val_parts[i] = idx, val
    if B == 0:
        return sp.csr_matrix((0, n_cols or 0)), 0
    lengths = np.fromiter((p.size for p in idx_parts), dtype=np.int64, count=B)
    indptr = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.concatenate(idx_parts).astype(np.int32) if lengths.sum() else np.zeros(0, dtype=np.int32)
    data = np.concatenate(val_parts).astype(dtype) if lengths.sum() else np.zeros(0, dtype=dtype or np.uint16)
    return sp.csr_matrix((data, indices, indptr), shape=(B, int(n_cols))), bytes_read


def cell_read_batch_local(archive_path: str, ranges: Sequence[tuple], max_gap: int = 64 * 1024):
    """``ranges`` = ``[(start_byte, size), ...]`` of one bin -> ``scipy.sparse.csr_matrix``
    ``(len(ranges), n_genes)`` in the given order. Ranges closer than ``max_gap`` share a pread."""
    fd = archive_fd(archive_path)
    X, _ = _batch(lambda s, n: pread_all(fd, n, s), ranges, max_gap)
    return X


def cell_read_batch_remote(archive_url: str, ranges: Sequence[tuple], max_gap: int = 64 * 1024):
    X, _ = _batch(lambda s, n: _fetch_remote(archive_url, s, n), ranges, max_gap)
    return X

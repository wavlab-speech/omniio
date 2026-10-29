"""Encode per-cell sparse rows for an omniio archive.

Two kinds of item go through ``cell_write``:

* **one row** -- a 1 x G ``scipy.sparse`` matrix, a 1-D dense array, an ``(indices,
  values, n_cols)`` tuple, or a dict with those keys plus extra metadata -> one entry;
* **many rows** -- a `CellRows` item, a generator or any iterator yielding ``(entry_id, row)``
  or ``(entry_id, row, metadata)`` -> one entry per row, returned as `Entries`, which
  ``Blob.append`` unrolls into the worker's shard.

Where the rows come from (an ``.h5ad`` file, a Census query, a VCF) and what the columns
mean is the caller's: see ``examples/cell_h5ad.py``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple, Union

import numpy as np

from omniio.blob.entries import Entries
from omniio.modalities.cell.common import VALUE_DTYPES, encode_row


class CellRows:
    """An item that yields many cells. Subclass and implement ``__iter__`` to yield
    ``(entry_id, row)`` or ``(entry_id, row, metadata)``, with ``row`` in any single-row
    form ``cell_write`` accepts. Keep the instance picklable (paths, ranges, a seed: not
    open files), since ``Blob.append`` sends items to worker processes and the rows are
    read there."""

    def __iter__(self) -> Iterator:
        raise NotImplementedError


def _single_row(source) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    extra: Dict[str, Any] = {}
    if isinstance(source, dict):
        extra = {k: v for k, v in source.items() if k not in ("indices", "values", "n_cols", "n_genes")}
        n = source.get("n_cols", source.get("n_genes"))
        return np.asarray(source["indices"]), np.asarray(source["values"]), int(n), extra
    if isinstance(source, tuple) and len(source) == 3:
        idx, val, n = source
        return np.asarray(idx), np.asarray(val), int(n), extra
    if hasattr(source, "tocsr") and hasattr(source, "shape"):        # scipy sparse
        m = source.tocsr()
        if m.shape[0] != 1:
            raise ValueError(f"a cell entry is one row; got a {m.shape} matrix")
        return m.indices, m.data, int(m.shape[1]), extra
    a = np.asarray(source)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 1:
        raise ValueError(f"a cell entry is one row; got shape {a.shape}")
    idx = np.flatnonzero(a)
    return idx, a[idx], int(a.size), extra


def _encode(source, value_dtype, layout, compress, compression_level, raw_indices) -> Tuple[bytes, dict]:
    idx, val, n_cols, extra = _single_row(source)
    raw, head = encode_row(idx, val, n_cols, value_dtype=value_dtype, layout=layout, raw_indices=raw_indices,
                           compress=compress, compression_level=compression_level)
    nz = val[val != 0] if val.size else val
    meta = {
        "nnz": int(np.count_nonzero(val)),
        "total_counts": float(nz.sum()) if nz.size else 0.0,
        "n_genes": int(n_cols),
        "value_dtype": VALUE_DTYPES[head.value_dtype].name,
        "format": "cell.zst" if compress else "cell",
        "stored_size": len(raw),
    }
    meta.update(extra)
    return raw, meta


def _many(rows, kw) -> Iterator[Tuple[str, bytes, dict]]:
    for item in rows:
        if len(item) == 2:
            entry_id, row = item
            extra = None
        else:
            entry_id, row, extra = item
        raw, meta = _encode(row, **kw)
        if extra:
            meta.update(extra)
        yield str(entry_id), raw, meta


def _is_many(source) -> bool:
    if isinstance(source, CellRows) or hasattr(source, "__next__"):
        return True
    return not isinstance(source, (dict, tuple, np.ndarray, list)) and not hasattr(source, "shape") and hasattr(source, "__iter__")


def cell_write(
    source: Any,
    item_id: str,
    value_dtype: Union[str, np.dtype] = "uint16",
    layout: str = "auto",
    compress: bool = False,
    compression_level: int = 3,
    raw_indices: bool = False,
):
    """Encode one cell, or every cell of a `CellRows` / iterator item.

    Args:
        source:        One row -- ``scipy.sparse`` (1 x G), 1-D array, ``(indices, values,
                       n_cols)`` or a dict with those keys (other keys become metadata) --
                       or a `CellRows` / generator / iterator yielding ``(entry_id, row)`` or
                       ``(entry_id, row, metadata)`` (one entry per row, via `Entries`).
        item_id:       Entry id for a single row; ignored for many rows (they carry their own).
        value_dtype:   'uint8' | 'uint16' (default) | 'uint32' | 'float32' | 'float64', or
                       'auto' = per entry, the narrowest unsigned width that holds the row
                       (u8 / u16 / u32; float32 for fractional or negative values). A value
                       that does not survive a declared cast (overflow, fraction, sign) raises.
        layout:        'auto' (dense when > half the columns are non-zero) | 'sparse' | 'dense'.
        compress:      zstd over indices + values (counts compress ~2x; costs a decompress per read).
        raw_indices:   store indices as u32 instead of delta varints.

    Returns:
        ``(raw_bytes, metadata)`` with ``nnz``, ``total_counts``, ``n_genes`` (the panel
        width), ``value_dtype``, ``format`` ('cell' | 'cell.zst'), ``stored_size`` and the
        extra keys; or an `Entries` iterator of ``(entry_id, raw_bytes, metadata)``.
    """
    kw = dict(value_dtype=value_dtype, layout=layout, compress=compress, compression_level=compression_level,
              raw_indices=raw_indices)
    if _is_many(source):
        return Entries(_many(source, kw))
    return _encode(source, **kw)

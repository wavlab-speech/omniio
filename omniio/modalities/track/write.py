"""Store genomic signal tracks: a bigWig / bedGraph path (one entry per chromosome, via
`Entries`), ``(bigwig_path, chrom)``, or an already binned array."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from omniio.blob.entries import Entries
from omniio.modalities.track.common import bin_values, encode

_EXTRA = "bigWig ingestion needs pyBigWig: pip install 'omniio[genome]'"


def _bw():
    try:
        import pyBigWig
    except ImportError as e:  # pragma: no cover
        raise ImportError(_EXTRA) from e
    return pyBigWig


def bin_chromosome(bw, chrom: str, length: int, bin: int, chunk_bp: int = 8_000_000) -> np.ndarray:
    """Exact per-bin means of one chromosome, read ``chunk_bp`` bases at a time
    (``chunk_bp`` is a multiple of ``bin`` so bins never straddle chunks)."""
    chunk_bp = max(bin, (chunk_bp // bin) * bin)
    parts = []
    for s in range(0, length, chunk_bp):
        e = min(s + chunk_bp, length)
        vals = bw.values(chrom, s, e, numpy=True)
        parts.append(bin_values(vals, bin))
    return np.concatenate(parts) if parts else np.zeros(0)


def _meta(name, head, raw, length_bp, source, extra, binned):
    with np.errstate(invalid="ignore"):
        finite = binned[np.isfinite(binned)]
    meta = {"name": name, "length": int(length_bp), "bin": head.bin, "n_bins": head.n_bins,
            "coverage_fraction": float(finite.size / max(binned.size, 1)),
            "mean": float(finite.mean()) if finite.size else None, "max": float(finite.max()) if finite.size else None,
            "format": "track", "dtype": "float16" if head.dtype_code == 1 else "float32", "stored_size": len(raw)}
    if source is not None:
        meta["source_file"] = source
    meta.update(extra)
    return meta


def track_write(source: Any, item_id: str, bin: int = 128, dtype: str = "float16", id_prefix: str = "",
                chroms: Optional[Sequence[str]] = None, min_length: int = 0):
    """Encode a binned track (or every chromosome of a bigWig).

    Args:
        source:  a bigWig / bedGraph path -> one entry per chromosome (`Entries`, ids
                 ``id_prefix + chrom``); ``(path, chrom)`` for one chromosome; or a dict
                 ``{"values": <already binned array>, "length": <bp>, "name": ..., extra...}``.
        bin:     bp per bin (fixed for the archive).
        dtype:   'float16' (default) or 'float32'.
        chroms:  (paths) only these chromosomes; ``min_length`` skips short contigs.

    Returns:
        ``(raw_bytes, metadata)`` -- ``name``, ``length`` (bp), ``bin``, ``n_bins``,
        ``coverage_fraction``, ``mean``, ``max``, ``format``, ``dtype``, ``stored_size`` --
        or `Entries` for a whole file (+ ``source_file``).
    """
    if isinstance(source, dict):
        binned = np.asarray(source["values"], dtype=np.float64)
        raw, head = encode(binned, bin, dtype)
        extra = {k: v for k, v in source.items() if k not in ("values", "length", "name")}
        return raw, _meta(source.get("name", item_id), head, raw, source.get("length", binned.size * bin), None, extra, binned)
    if isinstance(source, tuple) and len(source) == 2:
        path, chrom = source
        bw = _bw().open(os.fspath(path))
        try:
            length = bw.chroms(chrom)
            if length is None:
                raise KeyError(f"{chrom!r} not in {path}")
            binned = bin_chromosome(bw, chrom, length, bin)
        finally:
            bw.close()
        raw, head = encode(binned, bin, dtype)
        return raw, _meta(chrom, head, raw, length, os.fspath(path), {}, binned)
    if isinstance(source, (str, Path)):
        path = os.fspath(source)
        want = set(chroms) if chroms else None

        def gen():
            bw = _bw().open(path)
            try:
                for chrom, length in bw.chroms().items():
                    if (want is not None and chrom not in want) or length < min_length:
                        continue
                    binned = bin_chromosome(bw, chrom, length, bin)
                    raw, head = encode(binned, bin, dtype)
                    yield id_prefix + chrom, raw, _meta(chrom, head, raw, length, path, {}, binned)
            finally:
                bw.close()
        return Entries(gen())
    raise TypeError(f"unsupported track source {type(source).__name__}")

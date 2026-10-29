"""Store nucleotide sequences: one entry per record, from a string / bytes, a ``(name,
sequence)`` pair, or a FASTA path (plain or gzip; one entry per record via `Entries`)."""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from omniio.blob.entries import Entries
from omniio.modalities.seq.common import encode


def iter_fasta(path: str) -> Iterator[Tuple[str, bytes]]:
    """Yield ``(record name, sequence bytes)`` from a FASTA file, streaming line by line
    (one record held in memory at a time)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    name, chunks = None, []
    with opener(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if name is not None:
                    yield name, b"".join(chunks)
                name = line[1:].split()[0].decode() if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line.rstrip(b"\r\n"))
        if name is not None:
            yield name, b"".join(chunks)


def _meta(name: Optional[str], head, stats: dict, raw: bytes, source: Optional[str], extra: dict) -> dict:
    meta = {"name": name, "length": head.length, "n_count": stats["n_count"], "lower_count": stats["lower_count"],
            "format": "seq", "stored_size": len(raw)}
    if source is not None:
        meta["source_file"] = source
    meta.update(extra)
    return meta


def seq_write(source: Any, item_id: str, keep_case: bool = True, id_prefix: str = "", records: Optional[list] = None):
    """Encode a sequence (or every record of a FASTA file).

    Args:
        source:    a ``str`` / ``bytes`` sequence; ``(name, sequence)``; a dict with
                   ``sequence`` (+ ``name`` and extra metadata keys); or a FASTA path
                   (``.fa``, ``.fasta``, ``.fna``, optionally ``.gz``) -> one entry per record,
                   ids ``id_prefix + record name``.
        keep_case: store the soft-mask (lowercase) bitmap; ``False`` drops case.
        records:   (FASTA) only these record names.

    Returns:
        ``(raw_bytes, metadata)`` with ``name``, ``length``, ``n_count``, ``lower_count``,
        ``format``, ``stored_size``; or `Entries` for a FASTA path (+ ``source_file``).
    """
    if isinstance(source, (str, Path)) and str(source).lower().endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")):
        path = os.fspath(source)
        want = set(records) if records else None

        def gen():
            for name, seq in iter_fasta(path):
                if want is not None and name not in want:
                    continue
                raw, head, stats = encode(seq, keep_case=keep_case)
                yield id_prefix + name, raw, _meta(name, head, stats, raw, path, {})
        return Entries(gen())
    extra, name = {}, None
    if isinstance(source, dict):
        extra = {k: v for k, v in source.items() if k not in ("sequence", "name")}
        name = source.get("name", item_id)
        seq = source["sequence"]
    elif isinstance(source, tuple) and len(source) == 2:
        name, seq = source
    else:
        seq, name = source, item_id
    raw, head, stats = encode(seq, keep_case=keep_case)
    return raw, _meta(name, head, stats, raw, None, extra)

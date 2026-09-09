"""Read integer sequences from an omniio archive (local file or HTTP range)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import requests

from omniio.definitions import DiscreteRead
from omniio.modalities.discrete.common import decode, slice_indices


def _decode(blob: bytes, streams: Optional[Sequence[int]], start_time, end_time, start_frame, end_frame) -> DiscreteRead:
    infos, arrays = decode(blob, streams)
    keep = [k for k, a in enumerate(arrays) if a is not None]
    out, lengths = [], []
    for k in keep:
        lo, hi = slice_indices(infos[k], start_time, end_time, start_frame, end_frame)
        a = arrays[k][lo:hi]
        out.append(a); lengths.append(int(a.size))
    return DiscreteRead(
        file_type="discrete", modality="discrete",
        streams=out, stream_indices=keep, lengths=lengths,
        vocab_sizes=[infos[k].vocab for k in keep],
        rates=[infos[k].rate for k in keep] if any(infos[k].rate > 0 for k in keep) else None,
        n_streams=len(infos),
        start_time=start_time, end_time=end_time, start_frame=start_frame, end_frame=end_frame,
    )


def discrete_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
    streams: Optional[Sequence[int]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> DiscreteRead:
    """
    Read one discrete-sequence entry from a binary archive blob.

    Args:
        archive_path: Path to the .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.
        streams:      Stream indices to unpack (default all) — e.g. ``[0, 1]`` for the
                      two coarsest RVQ codebooks; others are skipped, not decoded.
        start_frame / end_frame: Window in elements (frames), applied to every stream;
                      takes priority over the time window, as in `video_read`.
        start_time / end_time:   Window in seconds, mapped through each stream's own
                      rate (floor / ceil); needs rates.

    Returns:
        DiscreteRead: ``streams`` (list of 1-D unsigned arrays in the narrowest dtype),
        ``array`` (``(n, T)`` when lengths agree), ``lengths``, ``vocab_sizes``, ``rates``.
    """
    with open(archive_path, "rb") as f:
        f.seek(start_offset)
        blob = f.read(file_size)
    return _decode(blob, streams, start_time, end_time, start_frame, end_frame)


def discrete_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
    streams: Optional[Sequence[int]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> DiscreteRead:
    """Same as `discrete_read_local`, over an HTTP range request."""
    end_byte = start_offset + file_size - 1
    resp = requests.get(archive_url, headers={"Range": f"bytes={start_offset}-{end_byte}"})
    resp.raise_for_status()
    return _decode(resp.content, streams, start_time, end_time, start_frame, end_frame)

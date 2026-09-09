"""Read integer sequences from an omniio archive (local file or HTTP range)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import requests

from omniio.definitions import DiscreteRead
from omniio.modalities.discrete.common import decode_partial


def _finish(entry_size: int, infos, idx, arrays, windows, start_time, end_time, start_frame, end_frame, n_read) -> DiscreteRead:
    return DiscreteRead(
        file_type="discrete", modality="discrete",
        streams=arrays, stream_indices=idx, lengths=[int(a.size) for a in arrays],
        vocab_sizes=[infos[k].vocab for k in idx],
        rates=[infos[k].rate for k in idx] if any(infos[k].rate > 0 for k in idx) else None,
        n_streams=len(infos), frame_windows=windows,
        start_time=start_time, end_time=end_time, start_frame=start_frame, end_frame=end_frame,
        bytes_read=n_read, entry_size=entry_size,
    )


def _read_with(read, file_size: int, streams, start_level, end_level, start_time, end_time, start_frame, end_frame) -> DiscreteRead:
    infos, idx, arrays, windows, n_read = decode_partial(
        read, streams=streams, start_level=start_level, end_level=end_level,
        start_time=start_time, end_time=end_time, start_frame=start_frame, end_frame=end_frame)
    return _finish(file_size, infos, idx, arrays, windows, start_time, end_time, start_frame, end_frame, n_read)


def discrete_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
    streams: Optional[Sequence[int]] = None,
    start_level: Optional[int] = None,
    end_level: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> DiscreteRead:
    """
    Read one discrete-sequence entry from a binary archive blob, fetching only the
    bytes the requested streams / frames occupy.

    Args:
        archive_path: Path to the .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.
        streams:      Explicit stream indices to decode (default all); or
        start_level / end_level: a window of stream (RVQ level) indices, [start, end).
        start_frame / end_frame: Window in elements (frames), applied to every stream;
                      takes priority over the time window, as in `video_read`.
        start_time / end_time:   Window in seconds, mapped through each stream's own
                      rate (floor / ceil); needs rates.

    Returns:
        DiscreteRead: ``streams`` (list of 1-D unsigned arrays in the narrowest dtype),
        ``array`` (``(n, T)`` when lengths agree), ``lengths``, ``vocab_sizes``, ``rates``,
        ``frame_windows``, ``bytes_read`` (the I/O actually done).
    """
    with open(archive_path, "rb") as f:
        def read(off: int, size: int) -> bytes:
            f.seek(start_offset + off)
            return f.read(min(size, file_size - off))
        return _read_with(read, file_size, streams, start_level, end_level, start_time, end_time, start_frame, end_frame)


def discrete_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
    streams: Optional[Sequence[int]] = None,
    start_level: Optional[int] = None,
    end_level: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> DiscreteRead:
    """Same as `discrete_read_local`, over HTTP range requests — one per fetched byte
    range (the header, then one per stream window, or one for a contiguous level window)."""
    def read(off: int, size: int) -> bytes:
        size = min(size, file_size - off)
        if size <= 0:
            return b""
        resp = requests.get(archive_url, headers={"Range": f"bytes={start_offset + off}-{start_offset + off + size - 1}"})
        resp.raise_for_status()
        return resp.content
    return _read_with(read, file_size, streams, start_level, end_level, start_time, end_time, start_frame, end_frame)

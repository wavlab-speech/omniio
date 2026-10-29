"""Read a genomic interval of a binned track: the header, then one contiguous read."""
from __future__ import annotations

from typing import Optional

import numpy as np
import requests

from omniio.blob.handles import archive_fd, pread_all
from omniio.definitions import TrackRead
from omniio.modalities.track.common import HEADER_SIZE, parse_header


def _read_with(read, file_size: int, start, end) -> TrackRead:
    head = parse_header(read(0, HEADER_SIZE))
    bin_ = head.bin
    s = 0 if start is None else max(0, int(start))
    e = head.n_bins * bin_ if end is None else int(end)
    k0 = min(s // bin_, head.n_bins)
    k1 = min((e + bin_ - 1) // bin_, head.n_bins)
    k1 = max(k1, k0)
    buf = read(HEADER_SIZE + k0 * head.itemsize, (k1 - k0) * head.itemsize) if k1 > k0 else b""
    arr = np.frombuffer(buf, dtype=head.dtype, count=k1 - k0).astype(np.float32)
    return TrackRead(file_type="track", modality="track", array=arr, bin=bin_, start=k0 * bin_, end=k1 * bin_,
                     n_bins=head.n_bins, length=head.n_bins * bin_, bytes_read=HEADER_SIZE + len(buf), entry_size=file_size)


def track_read_local(archive_path: str, start_offset: int, file_size: int, start: Optional[int] = None,
                     end: Optional[int] = None) -> TrackRead:
    """Bins covering bp ``[start, end)`` (default the whole chromosome). ``array`` is float32
    (NaN = no coverage); ``result.start`` / ``result.end`` are the bp bounds actually
    covered, snapped outwards to the bin grid."""
    fd = archive_fd(archive_path)

    def read(off, size):
        return pread_all(fd, min(size, file_size - off), start_offset + off) if size > 0 else b""
    return _read_with(read, file_size, start, end)


def track_read_remote(archive_url: str, start_offset: int, file_size: int, start: Optional[int] = None,
                      end: Optional[int] = None) -> TrackRead:
    def read(off, size):
        size = min(size, file_size - off)
        if size <= 0:
            return b""
        resp = requests.get(archive_url, headers={"Range": f"bytes={start_offset + off}-{start_offset + off + size - 1}"})
        resp.raise_for_status()
        return resp.content
    return _read_with(read, file_size, start, end)

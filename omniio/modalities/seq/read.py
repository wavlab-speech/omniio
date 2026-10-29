"""Read a sequence window from an archive: three byte-range reads (bases, N bitmap,
soft-mask bitmap), or one for the whole entry."""
from __future__ import annotations

from typing import Optional

import requests

from omniio.blob.handles import archive_fd, pread_all
from omniio.definitions import SeqRead
from omniio.modalities.seq.common import HEADER_SIZE, decode_window, parse_header


def _read_with(read, file_size: int, start, end, reverse_complement, soft_mask) -> SeqRead:
    head = parse_header(read(0, HEADER_SIZE))
    n_read = HEADER_SIZE
    s = 0 if start is None else max(0, int(start))
    e = head.length if end is None else min(head.length, int(end))
    if e < s:
        e = s
    b0, b1 = s // 4, (e + 3) // 4
    bases = read(HEADER_SIZE + b0, b1 - b0) if e > s else b""
    n_read += len(bases)
    m0, m1 = s // 8, (e + 7) // 8
    nmask = lower = None
    if head.has_n and e > s:
        nmask = read(head.n_offset + m0, m1 - m0); n_read += len(nmask)
    if head.has_lower and soft_mask and e > s:
        lower = read(head.lower_offset + m0, m1 - m0); n_read += len(lower)
    seq = decode_window(head, bases, s, e, nmask, lower, reverse_complement=reverse_complement, soft_mask=soft_mask)
    return SeqRead(file_type="seq", modality="seq", sequence=seq, start=s, end=e, length=head.length,
                   reverse_complement=reverse_complement, bytes_read=n_read, entry_size=file_size)


def seq_read_local(archive_path: str, start_offset: int, file_size: int, start: Optional[int] = None,
                   end: Optional[int] = None, reverse_complement: bool = False, soft_mask: bool = True) -> SeqRead:
    """Bases ``[start, end)`` (0-based, half-open; default the whole record). ``soft_mask``
    keeps lowercase for repeat-masked bases; ``reverse_complement`` returns the other strand
    (so ``sequence[0]`` is the complement of base ``end - 1``)."""
    fd = archive_fd(archive_path)

    def read(off, size):
        return pread_all(fd, min(size, file_size - off), start_offset + off) if size > 0 else b""
    return _read_with(read, file_size, start, end, reverse_complement, soft_mask)


def seq_read_remote(archive_url: str, start_offset: int, file_size: int, start: Optional[int] = None,
                    end: Optional[int] = None, reverse_complement: bool = False, soft_mask: bool = True) -> SeqRead:
    def read(off, size):
        size = min(size, file_size - off)
        if size <= 0:
            return b""
        resp = requests.get(archive_url, headers={"Range": f"bytes={start_offset + off}-{start_offset + off + size - 1}"})
        resp.raise_for_status()
        return resp.content
    return _read_with(read, file_size, start, end, reverse_complement, soft_mask)

"""Byte-range reads from archive blobs with a per-process file-descriptor cache.

Every modality reader used to `open(archive_path)` for each entry. On a networked file
system that is a metadata round trip per read: on Delta's hdd Lustre pool a random 150 KB
read cost 245 ms with open-per-read versus 60-85 ms through a cached descriptor with
`os.pread` (measured Nov 6 2026 on a GH200 node; a 2-turn TTS document needs two reads,
so a 4-worker DataLoader was capped at ~1.5 s/step). `read_range` keeps up to
`OMNIIO_FD_CACHE` (default 256) read-only descriptors open per process, evicting the least
recently used. `os.pread` is positional, so threads sharing a descriptor never race on an
offset; the cache is dropped in a forked child (descriptors stay valid but are re-opened
lazily, keeping the parent's and child's eviction independent). Set `OMNIIO_FD_CACHE=0`
to restore open-per-read.
"""
from __future__ import annotations

import os
from collections import OrderedDict

_LIMIT = int(os.environ.get("OMNIIO_FD_CACHE", "256"))
_fds: "OrderedDict[str, int]" = OrderedDict()


def _forget_all() -> None:
    _fds.clear()            # inherited fds are left to the parent; the child re-opens lazily


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_all)


def read_range(archive_path: str, start_offset: int, size: int) -> bytes:
    """`size` bytes of `archive_path` starting at `start_offset` (short at EOF, like read)."""
    if _LIMIT <= 0:
        with open(archive_path, "rb") as f:
            f.seek(start_offset)
            return f.read(size)
    fd = _fds.get(archive_path)
    if fd is None:
        fd = os.open(archive_path, os.O_RDONLY)
        _fds[archive_path] = fd
        while len(_fds) > _LIMIT:
            _, old = _fds.popitem(last=False)
            try:
                os.close(old)
            except OSError:
                pass
    else:
        _fds.move_to_end(archive_path)
    chunks = []
    off, left = int(start_offset), int(size)
    while left > 0:
        chunk = os.pread(fd, left, off)
        if not chunk:
            break
        chunks.append(chunk)
        off += len(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def clear_cache() -> None:
    """Close every cached descriptor (tests, or before deleting archives)."""
    for fd in _fds.values():
        try:
            os.close(fd)
        except OSError:
            pass
    _fds.clear()

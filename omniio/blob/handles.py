"""
Per-process cache of open archive file descriptors.

A reader that opens the archive on every call pays an `openat` and several
`fstat`s per entry (3.5 and 8.8 per read in the Phase 4 profile), which on a
parallel filesystem is the dominant cost of a small read. Archives are
append-only files, so a descriptor stays valid; the cache keys on (pid, path)
so that forked or spawned DataLoader workers never share a descriptor, and it
is bounded (LRU) so a process that touches thousands of bins does not run out
of file descriptors.

    fd = archive_fd(path)
    data = os.pread(fd, size, offset)
"""
import os
import threading
from collections import OrderedDict

_MAX_OPEN = int(os.environ.get("OMNIIO_MAX_OPEN_ARCHIVES", "64"))
_lock = threading.Lock()
_cache: "OrderedDict[tuple, int]" = OrderedDict()


def archive_fd(path: str) -> int:
    """Open (or reuse) a read-only descriptor for `path` in this process."""
    key = (os.getpid(), os.fspath(path))
    with _lock:
        fd = _cache.get(key)
        if fd is not None:
            _cache.move_to_end(key)
            return fd
        fd = os.open(key[1], os.O_RDONLY)
        _cache[key] = fd
        while len(_cache) > _MAX_OPEN:
            _, old = _cache.popitem(last=False)
            try:
                os.close(old)
            except OSError:
                pass
        return fd


def pread_all(fd: int, size: int, offset: int) -> bytes:
    """os.pread that keeps reading until `size` bytes or EOF."""
    chunks, got = [], 0
    while got < size:
        b = os.pread(fd, size - got, offset + got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks) if len(chunks) != 1 else chunks[0]


def close_all() -> None:
    with _lock:
        for fd in _cache.values():
            try:
                os.close(fd)
            except OSError:
                pass
        _cache.clear()

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
import io
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


class ArchiveWindow(io.RawIOBase):
    """Bytes `[offset, offset + size)` of an open archive as a seekable read-only file.

    Lets a decoder seek inside one entry and pull only the bytes a time window needs,
    instead of the whole entry being read first: a 3 s crop of a 15 min FLAC is a few
    hundred KB of I/O rather than ~40 MB, and a 1 s window of a 1 GB MP4 a few MB
    (the index atom plus one GOP) rather than the whole film, which is what dominates
    random-window loading from a parallel filesystem. Wrap it in `io.BufferedReader`
    with a buffer matched to the container's read pattern."""

    def __init__(self, fd: int, offset: int, size: int):
        self._fd, self._offset, self._size, self._pos = fd, int(offset), int(size), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = min(max(base + int(pos), 0), self._size)
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        data = os.pread(self._fd, n, self._offset + self._pos)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)

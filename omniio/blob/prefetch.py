"""
Lookahead reads: hide per-request latency when entries are read in an order the caller
already knows (a sampler's index sequence, a shuffled epoch, a list of windows).

Random access costs one request per entry, so throughput is in-flight requests divided by
latency: ~25 µs on a warm local archive, 0.1 to 2 ms on a parallel filesystem, 150 to 400 ms
over HTTP. A sequential stream hides that latency because the kernel or the HTTP client
reads ahead. `Lookahead` does the same for random order: it keeps `depth` reads in flight
on a thread pool and yields the results in the requested order, so a consumer that decodes
one entry sees the next ones arriving meanwhile. Works for any fetch function:

    from omniio.blob.handles import archive_fd, pread_all
    reads = [(path, off, size) for ...]                       # the epoch's order
    for raw in Lookahead(lambda p, o, s: pread_all(archive_fd(p), s, o), depth=64)(reads):
        decode(raw)

    # remote: one HTTP range request per entry, 64 in flight, same call shape
    store = Lookahead(lambda url, o, s: http_range(url, o, s), depth=64)

`read_batch` is the batch-shaped variant for DataLoader `__getitems__`: it sorts a batch's
reads by (file, offset), fetches them concurrently and returns them in the batch's order.
"""
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")
Request = Tuple[str, int, int]          # (path or URL, offset, size)


class Lookahead:
    def __init__(self, fetch: Callable[[str, int, int], T], depth: int = 64, threads: Optional[int] = None):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.fetch, self.depth = fetch, depth
        self.threads = threads or depth
        self._pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    def pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.threads)
            return self._pool

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None

    def __call__(self, requests: Iterable[Request]) -> Iterator[T]:
        """Yield fetch(*req) for every request, in order, keeping `depth` requests in flight."""
        pool = self.pool()
        pending: deque = deque()
        it = iter(requests)
        try:
            for req in it:
                pending.append(pool.submit(self.fetch, *req))
                if len(pending) >= self.depth:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for f in pending:
                f.cancel()

    def read_batch(self, requests: Sequence[Request]) -> List[T]:
        """All of a batch's reads at once, sorted by (file, offset) for the storage, results in
        the batch's own order."""
        order = sorted(range(len(requests)), key=lambda k: (requests[k][0], requests[k][1]))
        got = list(self.pool().map(lambda k: self.fetch(*requests[k]), order))
        out: List[T] = [None] * len(requests)  # type: ignore[list-item]
        for k, v in zip(order, got):
            out[k] = v
        return out


def local_fetch(path: str, offset: int, size: int) -> bytes:
    """The archive read used by the local readers (cached descriptor, one pread)."""
    from omniio.blob.handles import archive_fd, pread_all
    return pread_all(archive_fd(path), size, offset)

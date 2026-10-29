"""
Index orders produced by each loader under test, as generators of index
batches over a corpus in storage order. Composition-only experiments use the
orders directly; training experiments build the same orders into samplers.

  random_batches            uniform random access: a fresh global permutation
                            per epoch (omniio_random)
  buffered_batches          WebDataset order: shards of `shard_size`
                            consecutive entries in a seeded random order,
                            streamed through a shuffle buffer of `k` items
                            (emit uniform from buffer, insert next): wds_buf<k>
  preshuffled_batches       the corpus globally permuted once with
                            PRESHUFFLE_SEED before sharding, then the same
                            WebDataset pipeline: wds_preshuffled_buf<k>.
                            epoch=1 reuses the same pre-shuffled shards with a
                            new shard order and buffer stream (what a second
                            epoch of training does)
"""
from typing import Iterator

import numpy as np

PRESHUFFLE_SEED = 12345


def random_batches(n: int, B: int, n_batches: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    perm, pos = rng.permutation(n), 0
    for _ in range(n_batches):
        if pos + B > n:
            perm, pos = rng.permutation(n), 0
        yield perm[pos:pos + B]
        pos += B


def _shard_stream(order: np.ndarray, shard_size: int, rng: np.random.Generator) -> Iterator[int]:
    """Endless stream of corpus indices: storage order inside shards, shard
    order reshuffled every pass (WebDataset's shardshuffle)."""
    n = len(order)
    starts = np.arange(0, n, shard_size)
    while True:
        for s in rng.permutation(starts):
            yield from order[s:min(s + shard_size, n)].tolist()


def buffered_batches(n: int, B: int, n_batches: int, k: int, shard_size: int, rng: np.random.Generator,
                     order: np.ndarray = None) -> Iterator[np.ndarray]:
    """WebDataset's shuffle: fill k, then per incoming item emit one uniformly
    chosen buffered item and put the new one in its place."""
    order = np.arange(n) if order is None else order
    st = _shard_stream(order, shard_size, rng)
    buf = np.fromiter((next(st) for _ in range(min(k, n))), dtype=np.int64)
    L = len(buf)
    for _ in range(n_batches):
        out = np.empty(B, dtype=np.int64)
        for i in range(B):  # one emission at a time: a slot is refilled before the next draw
            j = int(rng.integers(L))
            out[i] = buf[j]
            buf[j] = next(st)
        yield out


def preshuffled_order(n: int, seed: int = PRESHUFFLE_SEED) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def preshuffled_batches(n: int, B: int, n_batches: int, k: int, shard_size: int, rng: np.random.Generator,
                        seed: int = PRESHUFFLE_SEED) -> Iterator[np.ndarray]:
    return buffered_batches(n, B, n_batches, k, shard_size, rng, order=preshuffled_order(n, seed))


def loader_batches(name: str, n: int, B: int, n_batches: int, shard_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    if name.startswith("block"):
        k, block = parse_block(name)
        return block_batches(n, B, n_batches, k, block, rng)
    """Dispatch on the loader names used in the tables."""
    if name == "omniio_random":
        return random_batches(n, B, n_batches, rng)
    if name.startswith("wds_preshuffled_buf"):
        k = parse_k(name.split("wds_preshuffled_buf")[1].split("_")[0])
        return preshuffled_batches(n, B, n_batches, k, shard_size, rng)
    if name.startswith("wds_buf"):
        return buffered_batches(n, B, n_batches, parse_k(name[len("wds_buf"):]), shard_size, rng)
    raise ValueError(name)


def parse_k(s: str) -> int:
    s = s.lower()
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


def epoch_stream(order: np.ndarray, shard_size: int, rng: np.random.Generator) -> np.ndarray:
    """One epoch of the stream as an array: shards of `shard_size` consecutive
    entries of `order`, in a random shard order."""
    n = len(order)
    starts = rng.permutation(np.arange(0, n, shard_size))
    return np.concatenate([order[s:min(s + shard_size, n)] for s in starts])


def independent_buffered_batches(n: int, B: int, M: int, k: int, shard_size: int, rng: np.random.Generator,
                                 order: np.ndarray = None) -> Iterator[np.ndarray]:
    """M independent batches of the WebDataset order: for each, a fresh epoch
    stream, the buffer filled with its first k items, then advanced by a
    uniformly random number of emissions U in [0, n - k - B] (a random
    position within the epoch), then B one-at-a-time emissions. The advance is
    vectorised: an emission puts the next stream item into a uniformly chosen
    slot, so the buffer after U emissions is the result of U slot assignments
    with the last write to a slot winning."""
    order = np.arange(n) if order is None else order
    k = min(k, n)
    for _ in range(M):
        stream = epoch_stream(order, shard_size, rng)
        buf = stream[:k].copy()
        U = int(rng.integers(0, max(1, n - k - B)))
        pos = k
        CH = 1 << 17
        while pos < k + U:
            c = min(CH, k + U - pos)
            buf[rng.integers(k, size=c)] = stream[pos:pos + c]   # sequential semantics: last write wins
            pos += c
        out = np.empty(B, dtype=np.int64)
        for i in range(B):
            j = int(rng.integers(k))
            out[i] = buf[j]
            buf[j] = stream[pos] if pos < n else stream[pos % n]
            pos += 1
        yield out


def independent_batches(name: str, n: int, B: int, M: int, shard_size: int, rng: np.random.Generator,
                        preshuffle_seed: int = PRESHUFFLE_SEED) -> Iterator[np.ndarray]:
    if name == "omniio_random":
        return random_batches(n, B, M, rng)
    if name.startswith("block"):
        k, block = parse_block(name)
        return independent_block_batches(n, B, M, k, block, rng)
    if name.startswith("wds_preshuffled_buf"):
        k = parse_k(name.split("wds_preshuffled_buf")[1].split("_")[0])
        return independent_buffered_batches(n, B, M, k, shard_size, rng, order=preshuffled_order(n, preshuffle_seed))
    if name.startswith("wds_buf"):
        return independent_buffered_batches(n, B, M, parse_k(name[len("wds_buf"):]), shard_size, rng)
    raise ValueError(name)


def block_batches(n: int, B: int, n_batches: int, k: int, block_size: int, rng: np.random.Generator,
                  order: np.ndarray = None) -> Iterator[np.ndarray]:
    """CorgiPile-style block sampling: the buffer holds k // block_size randomly chosen
    whole blocks (block_size consecutive items of the storage order), shuffled uniformly;
    each emission takes the next item of the shuffled buffer; when the buffer is used up it
    is refilled with the next k // block_size blocks of a random block order (one pass over
    the corpus per epoch)."""
    order = np.arange(n) if order is None else order
    n_blocks = int(np.ceil(n / block_size))
    per_buf = max(1, k // block_size)
    block_order = rng.permutation(n_blocks)
    pos = 0
    buf = np.empty(0, dtype=np.int64)
    bpos = 0
    emitted = 0
    while emitted < n_batches:
        out = np.empty(B, dtype=np.int64)
        for i in range(B):
            if bpos >= len(buf):
                if pos >= n_blocks:
                    block_order = rng.permutation(n_blocks); pos = 0
                blocks = block_order[pos:pos + per_buf]; pos += per_buf
                buf = np.concatenate([order[b * block_size:min((b + 1) * block_size, n)] for b in blocks])
                rng.shuffle(buf); bpos = 0
            out[i] = buf[bpos]; bpos += 1
        emitted += 1
        yield out


def independent_block_batches(n: int, B: int, M: int, k: int, block_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """M independent batches of block sampling: a fresh block order, the buffer at a
    uniformly random refill of the epoch, advanced by a random number of emissions."""
    n_blocks = int(np.ceil(n / block_size))
    per_buf = max(1, k // block_size)
    for _ in range(M):
        blocks = rng.permutation(n_blocks)[:per_buf]
        buf = np.concatenate([np.arange(b * block_size, min((b + 1) * block_size, n)) for b in blocks])
        rng.shuffle(buf)
        start = int(rng.integers(0, max(1, len(buf) - B)))
        if len(buf) - start < B:          # buffer smaller than a batch: continue into the next refill
            nxt = rng.permutation(n_blocks)[:per_buf]
            buf2 = np.concatenate([np.arange(b * block_size, min((b + 1) * block_size, n)) for b in nxt]); rng.shuffle(buf2)
            buf = np.concatenate([buf, buf2])
        yield buf[start:start + B].astype(np.int64)


def parse_block(name: str):
    """'block_buf10k' -> (k=10000, block=1000); 'block100_buf10k' -> (10000, 100)."""
    head, kk = name.split("_buf")
    block = int(head[len("block"):]) if head != "block" else 1000
    return parse_k(kk), block

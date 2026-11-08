"""OMAC entries: writing (with ``length_mismatch``) and indexed reading.

OMAC (Omniio Multichannel Audio Codec, ``docs/omac.md``) is a lossless codec
for any number of channels whose block index lets a reader fetch just the
bytes of a time window, and decode just the channels it asks for. The codec
itself is the ``omac`` extension module, built from ``rust/omac`` (distribution
``omniio-omac``) and imported lazily, so the rest of omniio works without it.
"""
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf

from omniio.definitions import AudioRead

MAGIC = b"OMAC"
FIXED_HEADER = 40

# Bytes fetched first when reading an entry: the fixed header, the channel
# lengths and the block index of entries up to 65536 blocks (8 bytes each),
# i.e. about 4.6 h at 16 kHz or 1.5 h at 48 kHz with 4096-frame blocks. Paid
# once per entry and process; the index is cached after that.
HEAD_FETCH = 512 * 1024

# Per-process cache of entry headers and block indexes, so later windows of
# the same entry cost one read. Index arrays are 8 bytes per block, so 256
# two-hour 16 kHz entries are about 60 MB.
_CACHE_ENTRIES = 256
_MAX_CACHED_INDEX = 4 * 1024 * 1024

LENGTH_MISMATCH = ("error", "truncate", "pad")

_SUBTYPE_BITS = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}


def _omac():
    try:
        import omac
    except ImportError as e:  # pragma: no cover - exercised without the extension
        raise ImportError(
            "OMAC needs the omac extension: pip install ./rust/omac "
            "(Rust toolchain required), or the omniio-omac wheel"
        ) from e
    return omac


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

Source = Union[str, Path, Tuple[np.ndarray, int]]


def _load_source(src: Source, bit_depth: Optional[int]) -> Tuple[np.ndarray, int, int, str]:
    """(int64 samples (frames, channels), sample_rate, bits, label)."""
    if isinstance(src, tuple):
        data, sr = src
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        if np.issubdtype(data.dtype, np.integer):
            bits = bit_depth or 8 * data.dtype.itemsize
            return data.astype(np.int64), int(sr), bits, "array"
        bits = bit_depth or 16
        scale = 2 ** (bits - 1)
        q = np.clip(np.round(data.astype(np.float64) * scale), -scale, scale - 1)
        return q.astype(np.int64), int(sr), bits, "array"
    path = str(src)
    info = sf.info(path)
    bits = _SUBTYPE_BITS.get(info.subtype)
    if bits is None:
        raise ValueError(
            f"{path}: OMAC is lossless integer PCM; subtype {info.subtype} is not supported"
        )
    dtype = "int16" if bits <= 16 else "int32"
    x, sr = sf.read(path, dtype=dtype, always_2d=True)
    # soundfile left-justifies samples in the requested dtype.
    x = x.astype(np.int64) >> (8 * np.dtype(dtype).itemsize - bits)
    return x, sr, bits, path


def _seconds(frames: int, sr: int) -> str:
    return f"{frames:,} frames ({frames / sr:.2f} s)"


def omac_write(
    sources: Union[Source, Sequence[Source]],
    item_id: str,
    target_bit_depth: Optional[int] = None,
    length_mismatch: str = "error",
    level: int = 5,
    block_size: int = 4096,
    threads: int = 1,
) -> Tuple[bytes, dict]:
    """Encode one or more sources as a single OMAC entry.

    Args:
        sources:          A path, an ``(array, sample_rate)`` tuple, or a list
                          of them. Their channels are concatenated in order:
                          four mono WAVs make one 4-channel entry.
        item_id:          Archive id (unused here; part of the writer interface).
        target_bit_depth: Coded bit depth. Default: the largest of the sources
                          (16 for float arrays). Narrower sources are shifted
                          up, which the codec stores at no cost (wasted bits).
        length_mismatch:  What to do when the sources differ in length:
                          ``"error"`` (default) raises and lists every source;
                          ``"truncate"`` cuts to the shortest; ``"pad"``
                          zero-fills to the longest and records each
                          channel's real length (``valid_frames``).
        level:            Compression level, 0 (fastest) to 8 (smallest).
        block_size:       Frames per block (the unit of random access).
        threads:          Encoder threads; the output does not depend on it.

    Returns:
        (raw_bytes, metadata). ``source_frames`` lists the original length of
        each source.
    """
    if length_mismatch not in LENGTH_MISMATCH:
        raise ValueError(f"length_mismatch must be one of {LENGTH_MISMATCH}, not {length_mismatch!r}")
    if isinstance(sources, (str, Path, tuple)):
        sources = [sources]
    if not sources:
        raise ValueError("no audio sources given")
    loaded = [_load_source(s, target_bit_depth) for s in sources]

    rates = {sr for _, sr, _, _ in loaded}
    if len(rates) > 1:
        detail = ", ".join(f"{label}: {sr} Hz" for _, sr, _, label in loaded)
        raise ValueError(f"sources have different sample rates: {detail}")
    sr = rates.pop()
    bits = target_bit_depth or max(b for _, _, b, _ in loaded)
    for x, _, b, label in loaded:
        if b > bits:
            raise ValueError(f"{label} is {b}-bit, more than target_bit_depth={bits}")

    lengths = [len(x) for x, _, _, _ in loaded]
    mismatch = len(set(lengths)) > 1
    if mismatch and length_mismatch == "error":
        detail = "\n".join(f"  {label}: {_seconds(len(x), sr)}" for x, _, _, label in loaded)
        raise ValueError(
            "audio sources differ in length; pass length_mismatch='truncate' or "
            f"'pad' to combine them anyway:\n{detail}"
        )
    frames = min(lengths) if length_mismatch == "truncate" else max(lengths)

    channels: List[np.ndarray] = []
    valid: List[int] = []
    for x, _, b, _ in loaded:
        x = x << (bits - b)
        n = min(len(x), frames)
        for c in range(x.shape[1]):
            col = np.zeros(frames, dtype=np.int32)
            col[:n] = x[:n, c]
            channels.append(col)
            valid.append(n)
    samples = np.stack(channels, 1)

    raw = _omac().encode(
        samples, sr, bits, block_size=block_size, level=level,
        valid_frames=valid, threads=threads,
    )
    metadata = {
        "sample_rate": sr,
        "channels": samples.shape[1],
        "samples": frames,
        "format": "omac",
        "bit_depth": bits,
        "duration": frames / sr,
        "block_size": block_size,
        # Always present: Blob builds the parquet schema from the first row.
        "source_frames": lengths,
    }
    return raw, metadata


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

Fetch = Callable[[int, int], bytes]  # entry-relative [lo, hi) -> bytes


class _Index:
    """Header fields and (part of) the block index of one entry."""

    def __init__(self, fixed: bytes, info: dict, valid: np.ndarray,
                 offsets: Optional[np.ndarray]):
        self.fixed = fixed
        self.info = info
        self.valid_frames = valid
        self.offsets = offsets  # full block index, or None if not cached

    def index_pos(self, k: int) -> int:
        return FIXED_HEADER + 8 * self.info["channels"] + 8 * k


_index_cache: "OrderedDict[tuple, _Index]" = OrderedDict()
_cache_lock = threading.Lock()  # readers may share a process across threads


def is_cached(key: tuple) -> bool:
    """Whether the header and index of the entry ``key`` are cached."""
    with _cache_lock:
        return key in _index_cache


def _load_index(fetch: Fetch, entry_size: int, key: Optional[tuple]) -> _Index:
    if key is not None:
        with _cache_lock:
            if key in _index_cache:
                _index_cache.move_to_end(key)
                return _index_cache[key]
    head = fetch(0, min(entry_size, HEAD_FETCH))
    info = _omac().read_info(head[:FIXED_HEADER])
    c, nb = info["channels"], info["num_blocks"]
    idx_lo = FIXED_HEADER + 8 * c
    idx_hi = idx_lo + 8 * (nb + 1)
    if len(head) < idx_lo:
        head += fetch(len(head), idx_lo)
    valid = np.frombuffer(head, "<u8", c, FIXED_HEADER).copy()
    offsets = None
    if idx_hi <= len(head):
        offsets = np.frombuffer(head, "<u8", nb + 1, idx_lo).copy()
    elif 8 * (nb + 1) <= _MAX_CACHED_INDEX:
        offsets = np.frombuffer(fetch(idx_lo, idx_hi), "<u8").copy()
    entry = _Index(bytes(head[:FIXED_HEADER]), info, valid, offsets)
    if key is not None:
        with _cache_lock:
            _index_cache[key] = entry
            while len(_index_cache) > _CACHE_ENTRIES:
                _index_cache.popitem(last=False)
    return entry


def omac_read(
    fetch: Fetch,
    entry_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    channels: Optional[Sequence[int]] = None,
    cache_key: Optional[tuple] = None,
    threads: int = 1,
) -> AudioRead:
    """Read a window of an OMAC entry through ``fetch``.

    ``fetch(lo, hi)`` returns bytes ``[lo, hi)`` of the entry (a local seek or
    an HTTP range request). A cold read costs a header read and a data read
    (plus an index read for very long entries); with ``cache_key`` the header
    and index are kept, and later windows of the entry cost the data read only.
    """
    idx = _load_index(fetch, entry_size, cache_key)
    info = idx.info
    sr, b, n = info["sample_rate"], info["block_size"], info["num_frames"]
    lo = 0 if start_time is None else round(start_time * sr)
    hi = n if end_time is None else round(end_time * sr)
    lo, hi = max(0, min(lo, n)), max(0, min(hi, n))
    hi = max(hi, lo)
    sel = list(range(info["channels"])) if channels is None else [int(c) for c in channels]
    if lo == hi:
        array = np.zeros((0, len(sel)), dtype=np.float32)
    else:
        k0, k1 = lo // b, (hi - 1) // b
        if idx.offsets is not None:
            offs = idx.offsets[k0 : k1 + 2]
        else:
            offs = np.frombuffer(fetch(idx.index_pos(k0), idx.index_pos(k1 + 2)), "<u8")
        offs = [int(o) for o in offs]
        data = fetch(offs[0], offs[-1])
        array = _omac().decode_range(
            idx.fixed, data, offs, k0, lo, hi, sel, "float32", True, threads
        )
    return AudioRead(
        file_type="omac",
        modality="audio",
        sample_rate=sr,
        array=array,
        valid_frames=[int(idx.valid_frames[c]) for c in sel],
    )

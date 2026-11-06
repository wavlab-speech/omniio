"""Store integer sequences (VQ / RVQ codes, token ids, ...) in an omniio archive."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from omniio.modalities.discrete.common import encode


def _as_streams(source: Any) -> tuple[list[np.ndarray], Optional[list], Optional[list], Optional[str]]:
    """-> (streams, vocab_sizes, rates, codec) from the accepted source forms."""
    vocab = rates = codec = None
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                if "streams" in z:                          # object-free: stream_0, stream_1, ... or one 2-D array
                    source = z["streams"]
                else:
                    keys = sorted(k for k in z.files if k.startswith("stream_"))
                    source = [z[k] for k in keys] if keys else z[z.files[0]]
                if "vocab_sizes" in z: vocab = z["vocab_sizes"].tolist()
                if "rates" in z: rates = z["rates"].tolist()
        else:
            source = np.load(p, allow_pickle=False)
    if isinstance(source, dict):
        vocab = source.get("vocab_sizes", source.get("vocab_size", vocab))
        rates = source.get("rates", source.get("rate", rates))
        codec = source.get("codec")
        source = source["streams"]
    if isinstance(source, np.ndarray):
        if source.ndim == 1:
            streams = [source]
        elif source.ndim == 2:
            streams = [source[i] for i in range(source.shape[0])]   # (n_streams, T): stream-major
        else:
            raise ValueError(f"expected a 1-D or 2-D array, got shape {source.shape}")
    elif isinstance(source, (list, tuple)):
        streams = [np.asarray(s) for s in source]
    else:
        try:                                                     # torch tensors and the like
            return _as_streams(np.asarray(source.detach().cpu() if hasattr(source, "detach") else source))
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"unsupported discrete source {type(source).__name__}") from e
    return [np.ascontiguousarray(s).astype(np.int64, copy=False) for s in streams], vocab, rates, codec


def _broadcast(x, n: int, name: str) -> list:
    if x is None:
        return [0] * n
    if np.isscalar(x):
        return [x] * n
    x = list(x)
    if len(x) != n:
        raise ValueError(f"{name}: expected {n} values (one per stream), got {len(x)}")
    return x


def discrete_write(
    source: Any,
    item_id: str,
    vocab_size: Union[int, Sequence[int], None] = None,
    rate: Union[float, Sequence[float], None] = None,
    codec: Optional[str] = None,
    compress: bool = False,
    compression_level: int = 3,
) -> Tuple[bytes, dict]:
    """Encode integer streams and return raw bytes + metadata dict.

    Args:
        source:     A 1-D array (one stream), a 2-D ``(n_streams, T)`` array (RVQ codes,
                    codebook-major), a list of 1-D arrays (ragged lengths allowed), a
                    ``.npy`` / ``.npz`` path, a torch tensor, or a dict
                    ``{"streams": ..., "vocab_sizes": ..., "rates": ..., "codec": ...}``.
        item_id:    Unique identifier for this sample.
        vocab_size: Alphabet size, one for all streams or one per stream (default:
                    ``max + 1`` per stream — pass it explicitly so the bit width is
                    stable across entries).
        rate:       Units per second, one for all streams or one per stream; enables
                    time-based slicing on read. None = no time axis.
        codec:      Free-form provenance (e.g. ``"encodec_24khz_6kbps"``); stored in the
                    metadata, never interpreted.
        compress:   Also zstd-wrap the bit-packed payload (rarely smaller: codes are
                    near-uniform; useful for silence-heavy or highly repetitive data).
        compression_level: zstandard level when ``compress`` is set.

    Returns:
        (raw_bytes, metadata) with ``n_streams``, ``lengths``, ``vocab_sizes``, ``bits``,
        ``rates`` (None when rate-less), ``duration`` (s, from the rates), ``n_tokens``,
        ``ragged``, ``codec``, ``format`` ('discrete' | 'discrete.zst') and sizes.
    """
    streams, v_src, r_src, c_src = _as_streams(source)
    n = len(streams)
    if n == 0:
        raise ValueError("no streams to write")
    vocab = _broadcast(vocab_size if vocab_size is not None else v_src, n, "vocab_size")
    rates = _broadcast(rate if rate is not None else r_src, n, "rate")
    codec = codec if codec is not None else c_src
    raw, infos = encode(streams, vocab, rates, compress=compress, compression_level=compression_level)
    lengths = [i.length for i in infos]
    has_rate = any(i.rate > 0 for i in infos)
    metadata = {
        "n_streams": n,
        "lengths": lengths,
        "vocab_sizes": [i.vocab for i in infos],
        "bits": [i.bits for i in infos],
        "rates": [i.rate for i in infos] if has_rate else None,
        "duration": (max(i.length / i.rate for i in infos if i.rate > 0) if has_rate else None),
        "n_tokens": int(sum(lengths)),
        "ragged": len(set(lengths)) > 1,
        "codec": codec,
        "format": "discrete.zst" if compress else "discrete",
        "original_size": int(sum(i.packed_bytes for i in infos)),
        "stored_size": len(raw),
    }
    return raw, metadata

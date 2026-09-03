"""``ReadHelper`` / ``WriteHelper``: rspecifier- and wspecifier-driven I/O."""

import numpy as np

from omniio.tools.kaldi.matio import (
    ReadError,
    _ArkWriter,
    load_ark,
    load_mat,
    load_scp_lines,
)
from omniio.tools.kaldi.specifier import parse_rspecifier, parse_wspecifier
from omniio.tools.kaldi.stream import open_like_kaldi


def load_segments(path):
    """Parse a Kaldi ``segments`` file into ``(utt, rec, start, end)`` tuples."""
    segments = []
    with open_like_kaldi(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ReadError(
                    "{}:{}: expected '<utt> <rec> <start> <end>', got "
                    "{!r}".format(path, lineno, line)
                )
            utt, rec, start, end = fields
            segments.append((utt, rec, float(start), float(end)))
    return segments


class _SegmentedReader:
    """Yield per-segment slices of the recordings named by an rspecifier."""

    def __init__(self, rspecifier, segments, endian="<"):
        kind, path = parse_rspecifier(rspecifier)
        if kind != "scp":
            raise ValueError(
                "segments= requires an scp rspecifier so that recordings can "
                "be looked up by id, got {!r}".format(rspecifier)
            )
        self._index = dict(load_scp_lines(path))
        self._segments = load_segments(segments)
        self._endian = endian

    def __iter__(self):
        cached_rec = None
        cached = None
        for utt, rec, start, end in self._segments:
            if rec != cached_rec:
                if rec not in self._index:
                    raise ReadError(
                        "Recording {!r} required by segment {!r} is not in the "
                        "scp".format(rec, utt)
                    )
                value = load_mat(self._index[rec], self._endian)
                if not isinstance(value, tuple):
                    raise ReadError(
                        "segments= can only be applied to audio entries, but "
                        "{!r} is a plain array".format(rec)
                    )
                cached_rec, cached = rec, value
            rate, array = cached
            begin = int(start * rate)
            stop = int(end * rate)
            yield utt, (rate, array[begin:stop])


class ReadHelper:
    """Iterate ``(key, value)`` over an rspecifier.

    >>> with ReadHelper("scp:feats.scp") as reader:  # doctest: +SKIP
    ...     for key, array in reader:
    ...         ...

    With ``segments`` the rspecifier must point at audio, and each item is a
    ``(utt_id, (rate, array))`` slice of the corresponding recording.
    """

    def __init__(self, rspecifier, segments=None, endian="<"):
        self.rspecifier = rspecifier
        self.endian = endian
        self._closed = False
        if segments is not None:
            self._iterable = _SegmentedReader(rspecifier, segments, endian)
        else:
            kind, path = parse_rspecifier(rspecifier)
            if kind == "scp":
                self._iterable = self._iter_scp(path)
            else:
                self._iterable = load_ark(path, endian)

    def _iter_scp(self, path):
        for key, name in load_scp_lines(path):
            yield key, load_mat(name, self.endian)

    def __iter__(self):
        return iter(self._iterable)

    def close(self):
        self._closed = True
        close = getattr(self._iterable, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class WriteHelper:
    """Write ``(key, value)`` pairs to the destinations named by a wspecifier.

    >>> with WriteHelper("ark,scp:feats.ark,feats.scp") as writer:  # doctest: +SKIP
    ...     writer["utt1"] = array
    """

    def __init__(
        self,
        wspecifier,
        compression_method=None,
        write_function=None,
        write_kwargs=None,
        endian="<",
    ):
        spec = parse_wspecifier(wspecifier)
        self.wspecifier = wspecifier
        self._flush = "f" in spec
        self._writer = _ArkWriter(
            spec["ark"],
            scp=spec.get("scp"),
            text="t" in spec,
            endian=endian,
            compression_method=compression_method,
            write_function=write_function,
            write_kwargs=write_kwargs,
        )
        self._closed = False

    def __setitem__(self, key, value):
        if self._closed:
            raise ValueError("Cannot write to a closed WriteHelper")
        if not isinstance(value, (np.ndarray, tuple, list)):
            raise TypeError(
                "Values must be arrays or (rate, array) tuples, got "
                "{}".format(type(value).__name__)
            )
        self._writer.write(key, value)
        if self._flush:
            self._writer._ark.flush()

    def close(self):
        if not self._closed:
            self._closed = True
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

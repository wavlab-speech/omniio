"""Reading and writing Kaldi ``ark`` / ``scp`` objects.

The binary layout implemented here is the one produced by Kaldi's table
writers (``src/util/kaldi-table-inl.h``, ``src/matrix/kaldi-matrix.cc``,
Apache-2.0)::

    <key> <space> <object>

A ``KaldiObjectHolder`` object starts with the binary marker ``\\0B`` followed
by a token naming the type:

    ``FM ``/``DM ``  float32 / float64 matrix
    ``FV ``/``DV ``  float32 / float64 vector
    ``CM ``/``CM2 ``/``CM3 ``  compressed matrix (see :mod:`.compression`)

A ``std::vector<int32>`` (alignments) has no token: ``\\0B`` is followed
directly by the ``\\4``-prefixed length.

``WaveHolder`` objects are not preceded by ``\\0B``; the payload is a complete
RIFF file.  ESPnet additionally uses the "extended" archive layout introduced
by ``kaldiio``, in which the payload is any container ``soundfile`` can decode,
framed as ``AUDIO`` + a ``\\N``-prefixed little-endian length + the blob.  Such
archives are readable by this module and by ``kaldiio`` but not by Kaldi
itself.
"""

import collections
import collections.abc
import io
import re

import numpy as np

from omniio.tools.kaldi import compression
from omniio.tools.kaldi.stream import open_like_kaldi, parse_extended_filename

BINARY_MARKER = b"\x00B"
AUDIO_MARKER = b"AUDIO"

_SOUND_MAGIC = (b"RIFF", b"fLaC", b"OggS", b"\x1aE\xdf\xa3")

#: A text token with no decimal point or exponent.
_INTEGER = re.compile(r"^[+-]?\d+$")

_MATRIX_TOKENS = {"FM": "f4", "DM": "f8"}
_VECTOR_TOKENS = {"FV": "f4", "DV": "f8"}


class ReadError(RuntimeError):
    """Raised when an archive cannot be parsed."""


def _peek(fd, n):
    if hasattr(fd, "peek"):
        buf = fd.peek(n)
        if len(buf) >= n:
            return buf[:n]
    pos = fd.tell()
    buf = fd.read(n)
    fd.seek(pos)
    return buf


#: Upper bound on a single ``fd.read``. Sizes come off the wire, so a stale scp
#: offset can ask for an absurd one; reading up to this much at a time lets the
#: first short read reject it instead of allocating first.
_READ_CHUNK = 1 << 23  # 8 MiB


def _read_exact(fd, n, what):
    if n <= _READ_CHUNK:
        buf = fd.read(n)
    else:
        parts = []
        got = 0
        while got < n:
            chunk = fd.read(min(_READ_CHUNK, n - got))
            if not chunk:
                break
            parts.append(chunk)
            got += len(chunk)
        buf = b"".join(parts)
    if len(buf) != n:
        raise ReadError(
            "Unexpected end of archive while reading {} "
            "({} of {} bytes)".format(what, len(buf), n)
        )
    return buf


def _read_sized_int(fd, endian):
    """Read Kaldi's ``\\N``-prefixed integer; returns ``(value, bytes_read)``.

    The width byte says how many bytes follow, so the value has to be widened
    on the side the endianness puts the low-order bytes -- padding on the wrong
    side scales it by 2**(8*pad).
    """
    width = _read_exact(fd, 1, "integer size")[0]
    raw = _read_exact(fd, width, "integer")
    return int.from_bytes(raw, "big" if endian == ">" else "little"), 1 + width


def _read_token(fd):
    """Read a space-terminated token; returns ``(token, bytes_read)``."""
    chars = []
    while True:
        c = fd.read(1)
        if not c:
            raise ReadError("Unexpected end of archive while reading a type token")
        if c == b" ":
            return "".join(chars), len(chars) + 1
        chars.append(c.decode("ascii", "replace"))
        if len(chars) > 8:
            raise ReadError("Unrecognised type token {!r}".format("".join(chars)))


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def _read_binary_object(fd, endian):
    _read_exact(fd, 2, "binary marker")
    size = 2

    head = _peek(fd, 1)
    if not head:
        raise ReadError("Unexpected end of archive after the binary marker")
    if not head.isalpha():
        # ``std::vector<int32>``: no token, just the \4-prefixed length.
        dim, n = _read_sized_int(fd, endian)
        size += n
        # Collected rather than preallocated: ``dim`` comes straight off the
        # wire, and np.empty would honour a corrupt one before the first read
        # could fail.
        values = []
        for _ in range(dim):
            value, n = _read_sized_int(fd, endian)
            values.append(value)
            size += n
        return np.array(values, dtype=np.int32), size

    token, n = _read_token(fd)
    size += n

    if token in _MATRIX_TOKENS:
        rows, n = _read_sized_int(fd, endian)
        size += n
        cols, n = _read_sized_int(fd, endian)
        size += n
        dtype = np.dtype(endian + _MATRIX_TOKENS[token])
        nbytes = rows * cols * dtype.itemsize
        buf = _read_exact(fd, nbytes, "matrix data")
        return np.frombuffer(buf, dtype=dtype).reshape(rows, cols), size + nbytes

    if token in _VECTOR_TOKENS:
        dim, n = _read_sized_int(fd, endian)
        size += n
        dtype = np.dtype(endian + _VECTOR_TOKENS[token])
        nbytes = dim * dtype.itemsize
        buf = _read_exact(fd, nbytes, "vector data")
        return np.frombuffer(buf, dtype=dtype), size + nbytes

    if token in compression.TOKEN_TO_FORMAT:
        header = _read_exact(fd, 16, "compressed matrix header")
        rows, cols = compression.header_shape(header, endian)
        nbytes = compression.data_size(compression.TOKEN_TO_FORMAT[token], rows, cols)
        body = _read_exact(fd, nbytes, "compressed matrix data")
        return (
            compression.decompress(header + body, token, endian),
            size + 16 + nbytes,
        )

    raise ReadError("Unsupported Kaldi object type token {!r}".format(token))


# Kaldi's WaveHolder keeps the on-disk PCM width, so a 16-bit ark decodes to
# int16.  The "extended" AUDIO framing has no such contract and decodes to
# float64, which is what ``soundfile`` returns by default.
_DTYPE_FOR_SUBTYPE = {
    "PCM_S8": "int16",
    "PCM_U8": "int16",
    "PCM_16": "int16",
    "PCM_24": "int32",
    "PCM_32": "int32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}


def _decode_sound(blob, native_dtype=False):
    import soundfile

    with io.BytesIO(blob) as f:
        if native_dtype:
            with soundfile.SoundFile(f) as sf:
                dtype = _DTYPE_FOR_SUBTYPE.get(sf.subtype, "float64")
            f.seek(0)
            array, rate = soundfile.read(f, dtype=dtype)
        else:
            array, rate = soundfile.read(f)
    return int(rate), array


def _read_extended_audio(fd):
    _read_exact(fd, len(AUDIO_MARKER), "AUDIO marker")
    width = _read_exact(fd, 1, "AUDIO length size")[0]
    raw = _read_exact(fd, width, "AUDIO length")
    length = int.from_bytes(raw, "little")
    blob = _read_exact(fd, length, "AUDIO payload")
    return _decode_sound(blob), len(AUDIO_MARKER) + 1 + width + length


def _read_riff(fd):
    """Read one RIFF chunk written by Kaldi's ``WaveHolder``."""
    head = _read_exact(fd, 8, "RIFF header")
    # The RIFF size field counts everything after itself.
    length = int.from_bytes(head[4:8], "little")
    rest = _read_exact(fd, length, "RIFF payload")
    return _decode_sound(head + rest, native_dtype=True), 8 + length


def _read_bare_sound(fd):
    """Read a non-RIFF audio container that runs to the end of the stream."""
    blob = fd.read()
    return _decode_sound(blob, native_dtype=True), len(blob)


def _read_text_object(fd, endian):
    chunks = []
    while True:
        line = fd.readline()
        if not line:
            break
        chunks.append(line)
        if b"]" in line:
            break
    raw = b"".join(chunks)
    try:
        text = raw.decode()
    except UnicodeDecodeError as e:
        raise ReadError(
            "Not a Kaldi object: no binary marker, no audio magic, and not "
            "decodable as text. An scp offset left over from an earlier "
            "version of the archive looks exactly like this."
        ) from e
    if "[" not in text or "]" not in text:
        raise ReadError("Malformed text-format object: {!r}".format(text[:80]))

    start = text.index("[")
    body = text[start + 1 : text.rindex("]")]
    is_matrix = body.lstrip(" ").startswith("\n")
    if is_matrix:
        rows = [r.split() for r in body.split("\n") if r.strip()]
        array = np.array(rows, dtype=np.float32)
    else:
        tokens = body.split()
        # A vector of whole numbers is an alignment or a label sequence, and
        # readers parse those with int(); the binary format keeps them integral
        # too, as std::vector<int32>. Matrices are always float, as in Kaldi.
        if tokens and all(_INTEGER.match(t) for t in tokens):
            array = np.array(tokens, dtype=np.int32)
        else:
            array = np.array(tokens, dtype=np.float32)
    return array, len(raw)


def read_object(fd, endian="<", return_size=False):
    """Read one object from ``fd``, positioned at the start of the payload.

    Returns the object, or ``(object, n_bytes)`` when ``return_size`` is set.
    Audio payloads come back as ``(sample_rate, array)``.
    """
    head = _peek(fd, len(AUDIO_MARKER))
    if not head:
        raise ReadError("Unexpected end of archive")

    if head[:2] == BINARY_MARKER:
        obj, size = _read_binary_object(fd, endian)
    elif head[: len(AUDIO_MARKER)] == AUDIO_MARKER:
        obj, size = _read_extended_audio(fd)
    elif head[:4] == b"RIFF":
        obj, size = _read_riff(fd)
    elif head[:4] in _SOUND_MAGIC:
        obj, size = _read_bare_sound(fd)
    else:
        obj, size = _read_text_object(fd, endian)

    return (obj, size) if return_size else obj


def _read_key(fd):
    """Read a space-terminated archive key; ``None`` at end of stream."""
    chars = []
    while True:
        c = fd.read(1)
        if not c:
            if chars:
                raise ReadError(
                    "Archive ended in the middle of the key {!r}".format("".join(chars))
                )
            return None
        if c in (b" ", b"\t"):
            return "".join(chars)
        if c in (b"\n", b"\r"):
            # Tolerate archives concatenated with stray newlines.
            if not chars:
                continue
            raise ReadError("Archive key {!r} is not followed by a space".format("".join(chars)))
        chars.append(c.decode("utf-8", "replace"))


def load_ark(file_or_fd, endian="<", return_position=False):
    """Iterate ``(key, object)`` over an ark.

    With ``return_position`` each item is ``(key, object, offset)`` where the
    offset is the value an scp entry would carry.
    """
    should_close = isinstance(file_or_fd, str)
    fd = open_like_kaldi(file_or_fd, "rb") if should_close else file_or_fd
    try:
        position = 0
        while True:
            key = _read_key(fd)
            if key is None:
                return
            position += len(key.encode()) + 1
            obj, size = read_object(fd, endian, return_size=True)
            if return_position:
                yield key, obj, position
            else:
                yield key, obj
            position += size
    finally:
        if should_close:
            fd.close()


def load_mat(name, endian="<", fd_dict=None):
    """Load a single object named by a Kaldi extended filename.

    ``name`` may be ``file.ark:1234``, a plain path holding one object, or a
    pipe such as ``gunzip -c foo.gz |``.

    ``fd_dict`` is a caller-owned ``{path: file}`` cache. Passing the same one
    across calls keeps each archive open, which is worth it when reading many
    entries out of a few arks; the caller closes them.
    """
    path, offset = parse_extended_filename(name)
    if fd_dict is not None and offset is not None and isinstance(path, str):
        fd = fd_dict.get(path)
        if fd is None:
            fd = fd_dict[path] = open_like_kaldi(path, "rb")
        fd.seek(offset)
        return read_object(fd, endian)

    with open_like_kaldi(path, "rb") as fd:
        if offset is not None:
            fd.seek(offset)
        return read_object(fd, endian)


def load_scp_lines(path, separator=None):
    """Yield ``(key, extended_filename)`` for each line of an scp file.

    ``separator`` splits the key from the file; the default splits on the first
    run of whitespace, which is what Kaldi writes. Pass one explicitly for an
    scp whose keys may themselves contain spaces.
    """
    with open_like_kaldi(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(separator, 1) if separator else line.split(None, 1)
            if len(parts) != 2:
                raise ReadError(
                    "{}:{}: expected '<key> <file>', got {!r}".format(path, lineno, line)
                )
            yield parts[0].strip(), parts[1].strip()


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


def slice_segment(value, start, end):
    """Cut ``[start, end)`` seconds out of an audio entry.

    A *negative* ``end`` is Kaldi's "to the end of the recording" sentinel,
    written as ``-1``; passing it through as a slice bound would index from the
    end instead and silently drop audio. An ``end`` of exactly ``0`` is not
    that sentinel -- Kaldi's ``extract-segments`` rejects such a line as an
    invalid segment -- and yields the empty slice ordinary indexing gives.
    """
    if not isinstance(value, tuple):
        raise ReadError(
            "segments can only be applied to audio entries, but this one is a " "plain array"
        )
    rate, array = value
    begin = int(start * rate)
    stop = len(array) if end < 0 else int(end * rate)
    return rate, array[begin:stop]


class LazyLoader(collections.abc.Mapping):
    """Dict-like view over an scp file; objects are read on ``__getitem__``."""

    def __init__(self, path, endian="<", max_cache_fd=0, separator=None):
        self._path = path
        self._endian = endian
        self._max_cache_fd = max_cache_fd
        self._cache = collections.OrderedDict()
        self._dict = collections.OrderedDict(load_scp_lines(path, separator))

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._path)

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()

    def __getitem__(self, key):
        name = self._dict[key]
        if self._max_cache_fd <= 0:
            return load_mat(name, self._endian)

        path, offset = parse_extended_filename(name)
        if offset is None:
            return load_mat(name, self._endian)

        fd = self._cache.get(path)
        if fd is None:
            fd = open_like_kaldi(path, "rb")
            self._cache[path] = fd
            while len(self._cache) > self._max_cache_fd:
                _, stale = self._cache.popitem(last=False)
                stale.close()
        else:
            self._cache.move_to_end(path)
        fd.seek(offset)
        return read_object(fd, self._endian)

    def close(self):
        for fd in self._cache.values():
            fd.close()
        self._cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SegmentedLoader(collections.abc.Mapping):
    """Dict-like view keyed by the utterances of a ``segments`` file.

    Each value is the ``(rate, array)`` slice of the recording the segment
    names. Recordings are decoded on demand and the most recent one is kept,
    so reading a segments file in its own order costs one decode per recording
    rather than one per segment.
    """

    def __init__(self, path, segments, endian="<", separator=None):
        self._index = dict(load_scp_lines(path, separator))
        self._dict = collections.OrderedDict(
            (utt, (rec, start, end)) for utt, rec, start, end in load_segments(segments)
        )
        self._endian = endian
        self._cached_rec = None
        self._cached = None

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()

    def __getitem__(self, key):
        rec, start, end = self._dict[key]
        if rec != self._cached_rec:
            if rec not in self._index:
                raise ReadError(
                    "Recording {!r} required by segment {!r} is not in the " "scp".format(rec, key)
                )
            self._cached = load_mat(self._index[rec], self._endian)
            self._cached_rec = rec
        return slice_segment(self._cached, start, end)


def load_scp(fname, endian="<", separator=None, segments=None, max_cache_fd=0):
    """Return a lazy mapping from utterance id to array for an scp file.

    With ``segments`` the mapping is keyed by the segments file's utterances
    instead, and each value is a slice of the recording it names.
    """
    if segments is not None:
        return SegmentedLoader(fname, segments, endian=endian, separator=separator)
    return LazyLoader(fname, endian=endian, max_cache_fd=max_cache_fd, separator=separator)


def load_wav_scp(fname, segments=None, separator=None, endian="<"):
    """``load_scp`` for an scp of audio; every value is ``(rate, array)``.

    Provided for compatibility only. :func:`load_scp` already returns
    ``(rate, array)`` for audio entries, so there is nothing this adds;
    ``kaldiio`` deprecated its own copy for the same reason.
    """
    return load_scp(fname, endian=endian, separator=separator, segments=segments)


def load_scp_sequential(fname, endian="<", separator=None, segments=None):
    """Iterate ``(key, object)`` over an scp file in file order.

    With ``segments`` the iteration follows the segments file instead, and each
    recording is decoded once for as long as consecutive segments share it.
    """
    if segments is None:
        for key, name in load_scp_lines(fname, separator):
            yield key, load_mat(name, endian)
        return

    index = dict(load_scp_lines(fname, separator))
    cached_rec, cached = None, None
    for utt, rec, start, end in load_segments(segments):
        if rec != cached_rec:
            if rec not in index:
                raise ReadError(
                    "Recording {!r} required by segment {!r} is not in the " "scp".format(rec, utt)
                )
            cached, cached_rec = load_mat(index[rec], endian), rec
        yield utt, slice_segment(cached, start, end)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _sized_int(value, endian):
    return b"\x04" + np.array(value, dtype=endian + "i4").tobytes()


def _as_rate_and_array(value):
    """Accept both ``(rate, array)`` and ``(array, rate)``."""
    first, second = value
    if isinstance(first, (int, np.integer)) and not isinstance(first, np.ndarray):
        return int(first), np.asarray(second)
    if isinstance(second, (int, np.integer)) and not isinstance(second, np.ndarray):
        return int(second), np.asarray(first)
    raise ValueError(
        "A sound entry must be (rate, array) or (array, rate), got "
        "({}, {})".format(type(first).__name__, type(second).__name__)
    )


_SUBTYPE_FOR_DTYPE = {
    "int16": "PCM_16",
    "int32": "PCM_32",
    "float32": "FLOAT",
    "float64": "DOUBLE",
}


def _encode_sound(array, rate, fmt="wav", subtype=None):
    import soundfile

    if subtype is None:
        subtype = _SUBTYPE_FOR_DTYPE.get(array.dtype.name)
        if fmt.lower() != "wav":
            # Only WAV carries every PCM width; let soundfile pick otherwise.
            subtype = None
    with io.BytesIO() as f:
        soundfile.write(f, array, rate, format=fmt, subtype=subtype)
        return f.getvalue()


def _encode_object(value, endian, compression_method, write_function, write_kwargs):
    """Serialise one archive value to bytes."""
    if isinstance(value, tuple):
        rate, array = _as_rate_and_array(value)
        if write_function is None:
            # Kaldi's WaveHolder: a bare RIFF file, no binary marker.
            return _encode_sound(array, rate, "wav")
        if write_function == "soundfile":
            kwargs = dict(write_kwargs or {})
            fmt = kwargs.pop("format", "wav")
            blob = _encode_sound(array, rate, fmt, **kwargs)
        elif callable(write_function):
            with io.BytesIO() as f:
                write_function(f, (rate, array), **(write_kwargs or {}))
                blob = f.getvalue()
        else:
            raise ValueError(
                "write_function must be None, 'soundfile' or a callable, got "
                "{!r}".format(write_function)
            )
        width = max(1, (len(blob).bit_length() + 7) // 8)
        return AUDIO_MARKER + bytes([width]) + len(blob).to_bytes(width, "little") + blob

    array = np.asarray(value)
    if array.dtype.kind in "iu" and array.ndim == 2:
        # Kaldi has no integer matrix type, and float32 cannot hold int32
        # exactly above 2**24, so writing one as FM would corrupt it silently.
        raise ValueError(
            "Kaldi archives have no integer matrix type, so a 2-dim {} array "
            "cannot be written. Cast it to float32/float64 if the loss of "
            "exactness is acceptable, or write one row per key.".format(array.dtype)
        )

    # After the guard: compression casts to float32, so it would lose the
    # same exactness silently.
    if compression_method is not None and array.ndim == 2:
        token, payload = compression.compress(array, compression_method, endian)
        return BINARY_MARKER + token.encode() + b" " + payload

    if array.dtype.kind in "iu" and array.ndim == 1:
        out = [BINARY_MARKER, _sized_int(array.shape[0], endian)]
        out.extend(_sized_int(int(v), endian) for v in array)
        return b"".join(out)

    if array.dtype == np.float64:
        token = "DM" if array.ndim == 2 else "DV"
        dtype = endian + "f8"
    else:
        token = "FM" if array.ndim == 2 else "FV"
        dtype = endian + "f4"
        array = array.astype(np.float32, copy=False)

    if array.ndim == 2:
        shape = _sized_int(array.shape[0], endian) + _sized_int(array.shape[1], endian)
    elif array.ndim == 1:
        shape = _sized_int(array.shape[0], endian)
    else:
        raise ValueError(
            "Only 1- and 2-dim arrays can be written to an ark, got " "ndim={}".format(array.ndim)
        )
    return (
        BINARY_MARKER
        + token.encode()
        + b" "
        + shape
        + np.ascontiguousarray(array, dtype=dtype).tobytes()
    )


def _text_values(array):
    """Format a row for a text ark, keeping integers integral.

    Writing an integer array as ``5.0`` rather than ``5`` breaks every reader
    that parses the token back with ``int()`` -- which is what a k-means label
    or an alignment is.
    """
    if array.dtype.kind in "iub":
        return " ".join(str(int(v)) for v in array)
    return " ".join(repr(float(v)) for v in array)


def _encode_text_object(value):
    array = np.asarray(value)
    if array.ndim == 2:
        rows = "".join("\n  " + _text_values(row) + " " for row in array)
        return (" [" + rows + "]\n").encode()
    if array.ndim == 1:
        return (" [ " + _text_values(array) + " ]\n").encode()
    raise ValueError(
        "Only 1- and 2-dim arrays can be written to a text ark, got " "ndim={}".format(array.ndim)
    )


def _check_scp_target(name):
    """An scp entry is only usable if ``load_mat`` can reopen and seek it.

    That rules out stdin/stdout and pipe destinations as well as unnamed
    streams: each would produce lines such as ``utt -:123`` that look
    well-formed and only fail much later, at read time.
    """
    if not isinstance(name, str):
        raise ValueError(
            "Cannot write an scp for an archive object with no file name: "
            "every line would point at an unopenable path. Pass a path for "
            "the ark, or drop the scp."
        )
    stripped = name.strip()
    if stripped == "-" or stripped.startswith("|") or stripped.endswith("|"):
        raise ValueError(
            "Cannot write an scp for the non-seekable archive target {!r}: an "
            "scp entry is '<path>:<offset>' and is read back by reopening the "
            "path and seeking, which a stream or a pipe cannot support. Write "
            "the ark to a file, or drop the scp.".format(name)
        )


class _ArkWriter:
    """Writes ``<key> <object>`` records and, optionally, the matching scp."""

    def __init__(
        self,
        ark,
        scp=None,
        append=False,
        text=False,
        endian="<",
        compression_method=None,
        write_function=None,
        write_kwargs=None,
    ):
        self._own_ark = isinstance(ark, str)
        self._own_scp = isinstance(scp, str)
        self._ark_name = ark if self._own_ark else getattr(ark, "name", None)
        # Before opening anything: for a pipe target that would already have
        # started the subprocess.
        if scp is not None:
            _check_scp_target(self._ark_name)

        mode = "ab" if append else "wb"
        self._ark = open_like_kaldi(ark, mode) if self._own_ark else ark
        self._scp = open_like_kaldi(scp, "a" if append else "w") if self._own_scp else scp
        self._text = text
        self._endian = endian
        self._compression_method = compression_method
        self._write_function = write_function
        self._write_kwargs = write_kwargs

        try:
            self._pos = self._ark.tell()
        except (AttributeError, OSError, io.UnsupportedOperation):
            self._pos = 0

    def write(self, key, value):
        if not isinstance(key, str) or not key or any(c.isspace() for c in key):
            raise ValueError(
                "Archive keys must be non-empty strings without whitespace, " "got {!r}".format(key)
            )
        head = (key + " ").encode()
        if self._text:
            body = _encode_text_object(value)
        else:
            body = _encode_object(
                value,
                self._endian,
                self._compression_method,
                self._write_function,
                self._write_kwargs,
            )
        self._ark.write(head + body)
        offset = self._pos + len(head)
        self._pos += len(head) + len(body)
        if self._scp is not None:
            self._scp.write("{} {}:{}\n".format(key, self._ark_name, offset))
        return offset

    def close(self):
        if self._own_ark:
            self._ark.close()
        if self._own_scp and self._scp is not None:
            self._scp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def save_ark(
    ark,
    array_dict,
    scp=None,
    append=False,
    text=False,
    endian="<",
    compression_method=None,
    write_function=None,
    write_kwargs=None,
):
    """Write ``array_dict`` to an ark, optionally emitting the matching scp.

    ``ark`` and ``scp`` may each be a path or an already-open file object; when
    they are file objects the caller keeps ownership and ``append`` only
    affects how offsets are seeded.
    """
    writer = _ArkWriter(
        ark,
        scp=scp,
        append=append,
        text=text,
        endian=endian,
        compression_method=compression_method,
        write_function=write_function,
        write_kwargs=write_kwargs,
    )
    try:
        for key, value in array_dict.items():
            writer.write(key, value)
    finally:
        writer.close()


def save_mat(path, array, endian="<", compression_method=None):
    """Write a single object (no key) to ``path``."""
    body = _encode_object(array, endian, compression_method, None, None)
    with open_like_kaldi(path, "wb") as f:
        f.write(body)

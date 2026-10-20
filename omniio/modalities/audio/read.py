import io
import os
import requests
import warnings
from typing import Optional, Tuple

import av
import numpy as np
import soundfile as sf

from omniio.definitions import AudioRead
from omniio.blob.handles import archive_fd, pread_all


class PyAVCompatibilityWarning(UserWarning):
    """The installed PyAV / ffmpeg is outside the range WebM/Opus windowed reads were
    validated on. Full reads are unaffected; windows decode but their alignment has not
    been checked on this build (see README, "Windowed reads of WebM/Opus")."""


# The range swept by tests/tools/sweep_pyav_webm.sh: PyAV 9.2-18.1 on ffmpeg 4.4-8.1. The
# package's dependency floor is av>=10.0.0 (the whole test suite passes there); 9.2 is
# validated for windowed *reads* only, from a conda-forge build without an Opus encoder.
# The upper edge is deliberately not a dependency cap, because a newer PyAV is untested
# rather than known bad, and only WebM/Opus windows are affected. The timestamps a seek produces changed
# twice inside the range (PyAV 17 moved stream.start_time by the codec delay; every build
# moves the trimmed head frame's pts), so a build outside it may have changed them
# again. libavcodec cannot be expressed as a dependency at all: it is inside the PyAV
# wheel, and conda-forge builds pair one PyAV with several ffmpegs. Widen after
# rerunning the sweep, not before.
_TESTED_PYAV = ((9, 2), (18, 1))
_TESTED_LIBAVCODEC = ((58, 134), (62, 28))


def _version_tuple(text: str) -> Tuple[int, ...]:
    """``"18.1.0rc1"`` -> ``(18, 1, 0)``; anything with fewer than two numeric components
    is unknown and returns ``()``, so a bare ``"18"`` cannot pass a range check."""
    out = []
    for part in text.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out) if len(out) >= 2 else ()


def pyav_compatibility() -> Optional[str]:
    """``None`` when the installed PyAV / libavcodec lie inside the validated range, else
    a message saying which one does not."""
    pyav = _version_tuple(getattr(av, "__version__", ""))[:2]
    lavc = tuple(getattr(av, "library_versions", {}).get("libavcodec", ())[:2])
    problems = []
    if not pyav or not (_TESTED_PYAV[0] <= pyav <= _TESTED_PYAV[1]):
        problems.append(f"PyAV {getattr(av, '__version__', 'of unknown version')}")
    if not lavc or not (_TESTED_LIBAVCODEC[0] <= lavc <= _TESTED_LIBAVCODEC[1]):
        problems.append(
            "libavcodec " + ".".join(map(str, lavc)) if lavc else "an unknown libavcodec"
        )
    if not problems:
        return None
    verb = "are" if len(problems) > 1 else "is"
    return (
        f"omniio: {' and '.join(problems)} {verb} outside the range WebM/Opus windowed reads "
        f"were validated on (PyAV {_TESTED_PYAV[0][0]}.{_TESTED_PYAV[0][1]}-"
        f"{_TESTED_PYAV[1][0]}.{_TESTED_PYAV[1][1]}, libavcodec "
        f"{_TESTED_LIBAVCODEC[0][0]}.{_TESTED_LIBAVCODEC[0][1]}-"
        f"{_TESTED_LIBAVCODEC[1][0]}.{_TESTED_LIBAVCODEC[1][1]}). Windows of WebM/Opus "
        "entries may be misaligned on this build; full reads are unaffected. Run "
        "tests/tools/sweep_pyav_webm.sh to check, or read from 0 and slice."
    )


_warned_untested_build = False


def _warn_if_untested_build() -> None:
    """Once per process, on the first windowed WebM/Opus read only: the check is about
    window alignment, so readers of FLAC/WAV/OGG or of whole WebM entries never see it."""
    global _warned_untested_build
    if _warned_untested_build:
        return
    _warned_untested_build = True
    message = pyav_compatibility()
    if message is not None:
        # _read_webm <- audio_read_local/_remote <- interface.audio_read <- the caller
        warnings.warn(message, PyAVCompatibilityWarning, stacklevel=4)

# Magic bytes for format detection
_MAGIC = {
    b"fLaC":        "flac",
    b"RIFF":        "wav",
    b"\x1aE\xdf\xa3": "webm",  # EBML header (Matroska/WebM)
    b"OggS":        "ogg",
}

def _detect_format(header: bytes) -> str:
    for magic, fmt in _MAGIC.items():
        if header[: len(magic)] == magic:
            return fmt
    raise ValueError(f"Unknown audio format (header bytes: {header[:8].hex()})")

class _ArchiveWindow(io.RawIOBase):
    """Bytes `[offset, offset + size)` of an open archive as a seekable read-only file.

    Lets the decoder seek inside one entry and pull only the frames a time window needs,
    instead of the whole entry being read first: a 3 s crop of a 15 min FLAC is a few
    hundred KB of I/O rather than ~40 MB, which is what dominates random-window loading
    from a parallel filesystem."""

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


def _read_pcm(
    blob,
    fmt: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read FLAC/WAV/OGG via soundfile, with optional time slicing. `blob` is the entry's
    bytes or a seekable file object positioned over it."""
    buf = io.BytesIO(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else blob
    # One header parse: open the stream once, seek, read. (sf.info + sf.read parsed it twice.)
    with sf.SoundFile(buf) as snd:
        sr = snd.samplerate
        start_frame = 0 if start_time is None else min(int(start_time * sr), snd.frames)
        end_frame = snd.frames if end_time is None else min(int(end_time * sr), snd.frames)
        if start_frame:                       # a window past the end returns an empty array, as sf.read did
            snd.seek(start_frame)
        data = snd.read(max(end_frame - start_frame, 0), dtype="float32", always_2d=True)

    return AudioRead(
        file_type=fmt,
        modality="audio",
        sample_rate=sr,
        array=data,
    )


# Opus keeps decoder state across frames, and WebM can only be seeked to a cluster
# boundary, so a seek lands at or before the target time. The reader seeks early by this
# margin and drops whatever lands before the window: the extra audio both covers the
# codec's own 80 ms pre-roll and pushes the seek onto an earlier cluster, so the decoder
# is warm by the time the window starts. Measured on a 10 s Opus file, the first samples
# of a window differ from the same samples of a full decode by 0.09 at 80 ms of margin
# and by 2e-4 at 250 ms; the extra quarter second of decoding costs ~1 ms.
_OPUS_SEEK_MARGIN_S = 0.25

# Window reads: entries up to this size are read whole with one pread; larger ones are
# decoded in place through a buffered view with this buffer (both overridable for tests
# and benchmarks).
_WINDOW_BUFFER_BYTES = int(os.environ.get("OMNIIO_WINDOW_BUFFER_BYTES", 1 << 18))
_SMALL_ENTRY_BYTES = int(os.environ.get("OMNIIO_SMALL_ENTRY_BYTES", 4 * (1 << 18)))


def _frame_samples(frame) -> np.ndarray:
    """A decoded audio frame as (channels, samples), planar or packed."""
    arr = frame.to_ndarray()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if not frame.format.is_planar and arr.shape[0] == 1 and frame.samples:
        arr = arr.reshape(frame.samples, -1).T  # packed: one plane, channels interleaved
    return arr


def _snap_to_grid(pos: int, frame_samples: int, tol: int) -> int:
    """Correct a pts-derived sample position onto the frame grid.

    Matroska timestamps are milliseconds, so a position read off a frame's pts is out by
    up to half a millisecond (+-24 samples at 48 kHz) — enough for a window to miss the
    samples a full decode has at that offset. A decode that starts at a seek begins on a
    frame boundary (libavformat re-applies the codec delay there, so the run starts a
    whole number of frames into the stream), and only one such boundary can be within a
    timestamp tick of the estimate. Snapping is capped at `tol`, so on a stream that does
    not hold to the grid it can never move a position further than the quantisation had.
    """
    if frame_samples <= 0:
        return pos
    snapped = frame_samples * round(pos / frame_samples)
    return snapped if abs(snapped - pos) <= tol else pos


def _is_opus_frame_size(n: int, sr: int) -> bool:
    """Opus codes 2.5, 5, 10, 20, 40 or 60 ms per frame — nothing else is a frame grid."""
    return any(abs(n - round(sr * ms / 1000)) <= 1 for ms in (2.5, 5, 10, 20, 40, 60))


def _decode_packets(container, stream):
    """Decode `stream` from the container's current position, yielding (packet, frame).

    The packet itself, because the caller needs both its pts and -- to tell a seek that
    came back to the start of the stream from one that did not -- the byte position it was
    read from.

    The packet's pts rather than the frame's. After a seek the decoder discards the codec
    delay from the first frame again, and a PyAV that sets the decoder's packet time base
    (13.0 and later, on ffmpeg 7) moves that frame's pts past the discarded samples,
    rounded to the container tick: a frame trimmed by 312 samples reports 7 ms after its
    packet, 336 samples at 48 kHz, which is not where it sits. Older builds left the
    frame's pts equal to the packet's. The packet's pts is the same in both and is what
    the anchor below is defined on.
    """
    for packet in container.demux(stream):
        for frame in packet.decode():
            yield packet, frame


def _first_output_packet(container, stream) -> Tuple[Optional[int], Optional[int]]:
    """(pts, byte position) of the stream's first packet that carries output.

    Its pts is the origin of the timeline the anchor is on, i.e. where a full decode's
    sample 0 sits. Its byte position identifies the packet itself, which is how
    `_webm_frames` recognises a seek that has come back to the start of the stream even
    when the demuxer hands that packet back under a different timestamp.

    Not ``stream.start_time``. PyAV 16 and earlier report the first packet's pts there;
    PyAV 17 and later (newer libavformat) report the first *frame's* presentation time,
    which is the packet's pts plus the codec delay, rounded to the tick -- so an anchor
    measured from ``start_time`` is off by the delay on those builds. A packet's own pts
    is the same number everywhere.

    Leading packets that end at or before pts 0 are skipped. They are priming: their
    content lies entirely before the stream's own zero, so the decoder consumes them and
    emits nothing, and a full decode's sample 0 belongs to the packet after them. AAC in
    MP4 writes exactly one such packet, at pts -1024 with duration 1024 (the encoder
    delay ``codec_context.delay`` also reports), and anchoring on it puts every seeked
    window a whole AAC frame -- 21 ms at 48 kHz -- early. WebM/Opus has no such packet:
    its head packet starts before zero (pts -7 ms here) but runs past it, so it is
    trimmed rather than dropped and stays the anchor, which is why this reads exactly as
    it did before for WebM.

    The test is on the packets alone. Decoding one to see whether it yields a frame would
    answer the same question directly, but it leaves the decoder warm, and a warm decoder
    does not re-apply the codec delay to the first frame after the seek below -- which is
    the very thing the anchor assumes it does.

    The pts returned is the earlier of that packet's own pts and where the packet after it
    says its content must begin (that packet's pts less one packet's span). The two agree
    on an ordinary file. They disagree where a muxer has written a head packet that starts
    before the stream's zero as starting at zero -- libavformat says so, `failed to avoid
    negative pts`, and writes it anyway; ffmpeg 5.1 does it to WebM/Opus with a video
    track. The recorded pts is then later than the audio it carries by exactly the amount
    clamped away (13 ms on the file this was measured on), and every window read through
    that anchor is that much late. Taking the minimum can only move the origin earlier,
    and only to where the following packet already places it, so a stream with a real gap
    after its first packet keeps the pts it declares.
    """
    first = None
    for packet in container.demux(stream):
        if packet.pts is None:
            if packet.size == 0:        # the flush packet: no more real packets
                break
            continue
        if first is None:
            if packet.duration and packet.pts + packet.duration <= 0:
                continue                # priming: consumed whole, decodes to nothing
            first = packet
            continue
        pts = first.pts
        if first.duration:
            pts = min(pts, packet.pts - first.duration)
        return pts, getattr(first, "pos", None)
    if first is not None:
        return first.pts, getattr(first, "pos", None)
    return None, None


def _webm_frames(container, stream, seek_sample: Optional[int]):
    """Decode a WebM audio stream, yielding (index of the frame's first sample, frame).

    Indices count from the stream's first sample, so they address the same array a full
    read returns. With `seek_sample` None the container is decoded from where it is — the
    first sample of a fresh container is sample 0 by construction, no timestamps involved.
    An int seeks (to a cluster at or before it, minus the seek margin), anchors the run on
    the first packet's pts, measured from the pts of the stream's first packet, and counts
    samples from there; the caller trims what lands before the window.

    Why the packet's pts is the anchor: a packet at pts P decodes to a whole frame whose
    samples sit at [P, P + frame) on the stream's timeline, and a full decode discards the
    first `delay` samples of the stream, so that frame's first sample is output sample
    P - delay. After a seek the decoder discards `delay` samples again, from the head
    frame, so the head's first surviving sample is stream sample P + delay, i.e. output
    sample P. The trimmed head therefore starts exactly at its packet's pts, whatever the
    codec delay is.

    One case the pts cannot carry: a seek that falls back to the start of the stream may
    hand the first packet back stamped with the seek target rather than its own pts. That
    is recognised by the packet's byte position and the run is placed at sample 0, which
    is where it really is; see the comment below.
    """
    sr = stream.rate
    time_base = stream.time_base

    if seek_sample is None:
        pos = 0
        for _, frame in _decode_packets(container, stream):
            arr = _frame_samples(frame)
            yield pos, arr
            pos += arr.shape[1]
        return

    start_pts, start_pos = _first_output_packet(container, stream)
    if start_pts is None:
        start_pts = stream.start_time or 0
    seek_s = max(0.0, seek_sample / sr - _OPUS_SEEK_MARGIN_S)
    container.seek(start_pts + int(seek_s / time_base), stream=stream)

    frames = _decode_packets(container, stream)
    first = next(frames, None)
    if first is None:
        return
    head_packet, head = first
    head_arr = _frame_samples(head)
    # The frame after the first is a whole frame, which is the grid the run sits on. The
    # first frame is not: the decoder trims the codec delay off it after a seek.
    nxt = next(frames, None)
    nxt_arr = None if nxt is None else _frame_samples(nxt[1])
    grid = (nxt_arr if nxt_arr is not None else head_arr).shape[1]

    head_pos = getattr(head_packet, "pos", None)
    if start_pos is not None and start_pos >= 0 and head_pos == start_pos:
        # Back at the stream's first packet, so the run starts at sample 0 whatever
        # timestamp the packet arrived with -- a seek that falls back to the start of the
        # file can restamp it with the seek target. See CLAUDE.md, "Common Implementation
        # Notes".
        pos = 0
    else:
        head_pts = head_packet.pts
        pts = start_pts if head_pts is None else head_pts
        pos = int(round(float((pts - start_pts) * time_base) * sr))
        if _is_opus_frame_size(grid, sr):
            pos = _snap_to_grid(pos, grid, int(sr * time_base / 2) + 1)

    yield pos, head_arr
    pos += head_arr.shape[1]
    if nxt_arr is not None:
        yield pos, nxt_arr
        pos += nxt_arr.shape[1]
    for _, frame in frames:
        arr = _frame_samples(frame)
        yield pos, arr
        pos += arr.shape[1]


def _gather_window(
    frames, start_sample: int, end_sample: Optional[int], detect_overshoot: bool = True
):
    """Trim decoded frames to [start_sample, end_sample).

    Returns (chunks, overshot). `overshot` marks a seek that landed after the requested
    start — the window cannot be built from these frames and the caller decodes from the
    top instead. A run that did not seek starts at the stream's first sample, so there is
    nothing earlier to recover and `detect_overshoot` is off: whatever it yields is the
    beginning of the stream.
    """
    chunks = []
    for i, (pos, arr) in enumerate(frames):
        if i == 0 and detect_overshoot and pos > start_sample:
            return [], True
        n = arr.shape[1]
        if pos + n <= start_sample:                  # entirely before the window
            continue
        if end_sample is not None and pos >= end_sample:
            break
        lo = max(0, start_sample - pos)
        hi = n if end_sample is None else min(n, end_sample - pos)
        if hi > lo:
            chunks.append(arr[:, lo:hi])
        if end_sample is not None and pos + hi >= end_sample:
            break
    return chunks, False


def _read_webm(
    blob: bytes,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read WebM/Opus via PyAV, with optional time slicing."""
    if start_time is not None or end_time is not None:
        _warn_if_untested_build()
    with av.open(io.BytesIO(blob), mode="r") as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        channels = stream.channels

        start_sample = 0 if start_time is None else max(0, int(round(start_time * sr)))
        end_sample = (
            None if end_time is None else max(start_sample, int(round(end_time * sr)))
        )

        # Seek only when it can actually skip work and there is a timeline to seek
        # against: a window inside the first `margin` seconds would seek to sample 0
        # anyway, and decoding from the top is both exact and cheaper than a seek that
        # makes libavformat re-apply the codec delay.
        seeking = (
            start_sample > _OPUS_SEEK_MARGIN_S * sr and stream.start_time is not None
        )
        chunks, overshot = _gather_window(
            _webm_frames(container, stream, start_sample if seeking else None),
            start_sample,
            end_sample,
            detect_overshoot=seeking,
        )

    if overshot:
        # The seek landed after the window and nothing before it can be recovered from
        # that container: decode the stream from the top in a fresh one, where the first
        # decoded sample is sample 0 by construction.
        with av.open(io.BytesIO(blob), mode="r") as container:
            stream = container.streams.audio[0]
            chunks, _ = _gather_window(
                _webm_frames(container, stream, None),
                start_sample,
                end_sample,
                detect_overshoot=False,
            )

    if not chunks:
        return AudioRead(
            file_type="webm",
            modality="audio",
            sample_rate=sr,
            array=np.empty((0, channels), dtype=np.float32),
        )

    raw = np.concatenate(chunks, axis=1)  # (channels, window)
    data = raw.T.astype(np.float32)       # (frames, channels)

    # Normalize integer formats to float
    if np.issubdtype(raw.dtype, np.integer):
        data /= float(np.iinfo(raw.dtype).max)

    return AudioRead(
        file_type="webm",
        modality="audio",
        sample_rate=sr,
        array=data,
    )


def audio_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> AudioRead:
    """
    Read a single audio entry from a binary archive blob.

    Args:
        archive_path: Path to the .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.
        start_time:   Start time in seconds (None = beginning).
        end_time:     End time in seconds (None = end of file).

    Returns:
        AudioRead with sample_rate and float32 array (frames, channels).
    """
    fd = archive_fd(archive_path)
    windowed = start_time is not None or end_time is not None
    if windowed and file_size > _SMALL_ENTRY_BYTES:
        fmt = _detect_format(os.pread(fd, 16, start_offset))
        if fmt != "webm":
            # A time window of a large seekable PCM/FLAC entry: decode in place, reading
            # only what the decoder asks for (a few hundred KB of a 40 MB recording).
            window = io.BufferedReader(_ArchiveWindow(fd, start_offset, file_size), _WINDOW_BUFFER_BYTES)
            return _read_pcm(window, fmt, start_time, end_time)
    # Whole entry in one pread: for an entry of a few window buffers the in-place path
    # costs more in Python callbacks than the bytes it saves (0.312 vs 0.278 ms for a
    # 333 KB entry), and the decoder then seeks in memory.
    blob = pread_all(fd, file_size, start_offset)
    fmt = _detect_format(blob[:16])
    if fmt == "webm":
        return _read_webm(blob, start_time, end_time)
    return _read_pcm(blob, fmt, start_time, end_time)

def audio_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> AudioRead:
    """
    Read a single audio entry from a remote binary archive via HTTP range request.

    Args:
        archive_url: URL to the remote .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.
        start_time:   Start time in seconds (None = beginning).
        end_time:     End time in seconds (None = end of file).

    Returns:
        AudioRead with sample_rate and float32 array (frames, channels).
    """
    end_byte = start_offset + file_size - 1
    headers = {"Range": f"bytes={start_offset}-{end_byte}"}

    resp = requests.get(archive_url, headers=headers)
    resp.raise_for_status()

    blob = resp.content

    header = blob[:16]
    fmt = _detect_format(header)

    if fmt == "webm":
        return _read_webm(blob, start_time, end_time)
    else:
        return _read_pcm(blob, fmt, start_time, end_time)
import io
import os
import requests
from typing import Optional

import av
import numpy as np
import soundfile as sf

from omniio.definitions import AudioRead

# Magic bytes for format detection
_MAGIC = {
    b"fLaC":        "flac",
    b"RIFF":        "wav",
    b"\x1aE\xdf\xa3": "webm",  # EBML header (Matroska/WebM)
    b"OggS":        "ogg",
    b"ID3":         "mp3",   # ID3v2 tag ahead of the first MPEG frame
}


def _is_mpeg_frame_sync(header: bytes) -> bool:
    """A tagless mp3 starts directly on an MPEG audio frame: 11 sync bits, then a
    non-zero layer field. The layer check keeps AAC ADTS (0xFFF1/0xFFF9, layer 00) out."""
    return (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
            and (header[1] >> 1) & 0x3 != 0)


def _detect_format(header: bytes) -> str:
    for magic, fmt in _MAGIC.items():
        if header[: len(magic)] == magic:
            return fmt
    if _is_mpeg_frame_sync(header):
        return "mp3"
    raise ValueError(f"Unknown audio format (header bytes: {header[:8].hex()})")

def _read_pcm(
    blob: bytes,
    fmt: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read FLAC/WAV/OGG/MP3 via soundfile, with optional time slicing. `blob` is the
    entry's bytes, or an already-seekable file object (a lazy byte-range view)."""
    buf = blob if hasattr(blob, "seek") else io.BytesIO(blob)
    info = sf.info(buf)
    sr = info.samplerate

    start_frame = 0 if start_time is None else int(start_time * sr)
    end_frame = info.frames if end_time is None else int(end_time * sr)
    num_frames = end_frame - start_frame
    # MP3 windows are never seeked into. libsndfile/mpg123 lands on the right sample,
    # but the frames after it can decode wrong: each draws on a bit reservoir (up to
    # 511/255 bytes back) and overlap state from frames the decoder never saw, and how
    # many frames that spans depends on the bitrate, with no safe fixed bound (up to
    # ~2 s for 8 kbps stereo). Decoding from the first sample makes the window a slice
    # of the full decode by construction; cost is O(end_frame) rather than O(window).
    read_from = 0 if fmt == "mp3" else start_frame

    buf.seek(0)
    data, sr = sf.read(
        buf,
        start=read_from,
        stop=end_frame,
        dtype="float32",
        always_2d=True,
    )
    if read_from != start_frame:
        data = data[start_frame:]

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


def _webm_frames(container, stream, seek_sample: Optional[int]):
    """Decode a WebM audio stream, yielding (index of the frame's first sample, frame).

    Indices count from the stream's first sample, so they address the same array a full
    read returns. With `seek_sample` None the container is decoded from where it is — the
    first sample of a fresh container is sample 0 by construction, no timestamps involved.
    An int seeks (to a cluster at or before it, minus the seek margin), anchors the run on
    the first frame's pts and counts samples from there; the caller trims what lands
    before the window.
    """
    sr = stream.rate
    time_base = stream.time_base

    if seek_sample is None:
        pos = 0
        for frame in container.decode(audio=0):
            arr = _frame_samples(frame)
            yield pos, arr
            pos += arr.shape[1]
        return

    start_pts = stream.start_time or 0
    seek_s = max(0.0, seek_sample / sr - _OPUS_SEEK_MARGIN_S)
    container.seek(start_pts + int(seek_s / time_base), stream=stream)

    frames = container.decode(audio=0)
    head = next(frames, None)
    if head is None:
        return
    head_arr = _frame_samples(head)
    # The frame after the first is a whole frame, which is the grid the run sits on. The
    # first frame is not: libavformat trims the codec delay off it after a seek.
    nxt = next(frames, None)
    nxt_arr = None if nxt is None else _frame_samples(nxt)
    grid = (nxt_arr if nxt_arr is not None else head_arr).shape[1]

    pts = start_pts if head.pts is None else head.pts
    pos = int(round(float((pts - start_pts) * time_base) * sr))
    if _is_opus_frame_size(grid, sr):
        pos = _snap_to_grid(pos, grid, int(sr * time_base / 2) + 1)

    yield pos, head_arr
    pos += head_arr.shape[1]
    if nxt_arr is not None:
        yield pos, nxt_arr
        pos += nxt_arr.shape[1]
    for frame in frames:
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
    fd = _archive_fd(archive_path)
    header = os.pread(fd, 16, start_offset)
    fmt = _detect_format(header)

    if fmt == "webm":
        return _read_webm(os.pread(fd, file_size, start_offset), start_time, end_time)
    if file_size > _LAZY_MIN_BYTES and (start_time is not None or end_time is not None):
        # A big entry with a time window (e.g. a 7 s crop of a 22-minute ambience file):
        # let libsndfile seek inside a lazy byte-range view instead of reading the whole
        # entry — only the pages the decoder touches are fetched from disk/NFS.
        return _read_pcm(_RangeFile(fd, start_offset, file_size), fmt, start_time, end_time)
    return _read_pcm(os.pread(fd, file_size, start_offset), fmt, start_time, end_time)


# ---- per-process archive handle cache + lazy byte-range file ----------------------------
# Each entry read used to open() the multi-GB .bin, seek, read the WHOLE entry and close.
# On NFS the open/close alone is a metadata round trip (~30 ms), which dominated small
# entries (0.2 MB read at ~7 MB/s effective). Handles are now kept per process (loader
# workers are processes) and read with os.pread (stateless, thread-safe).
_LAZY_MIN_BYTES = 8 * 1024 * 1024
_FDS: dict[str, int] = {}
_FDS_MAX = 64


def _archive_fd(path: str) -> int:
    fd = _FDS.get(path)
    if fd is None:
        if len(_FDS) >= _FDS_MAX:                     # bounded: drop the oldest handle
            old, ofd = next(iter(_FDS.items()))
            _FDS.pop(old); os.close(ofd)
        fd = os.open(path, os.O_RDONLY)
        _FDS[path] = fd
    return fd


class _RangeFile(io.RawIOBase):
    """Read-only, seekable file object over bytes [start, start+size) of an open fd."""

    def __init__(self, fd: int, start: int, size: int):
        self._fd, self._start, self._size, self._pos = fd, int(start), int(size), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = int(offset)
        elif whence == io.SEEK_CUR:
            self._pos += int(offset)
        else:
            self._pos = self._size + int(offset)
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        data = os.pread(self._fd, n, self._start + self._pos)
        k = len(data)
        b[:k] = data
        self._pos += k
        return k

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""
        data = os.pread(self._fd, n, self._start + self._pos)
        self._pos += len(data)
        return data

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
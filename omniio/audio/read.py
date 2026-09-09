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
}

def _detect_format(header: bytes) -> str:
    for magic, fmt in _MAGIC.items():
        if header[: len(magic)] == magic:
            return fmt
    raise ValueError(f"Unknown audio format (header bytes: {header[:8].hex()})")

def _read_pcm(
    blob: bytes,
    fmt: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read FLAC/WAV/OGG via soundfile, with optional time slicing. `blob` is the
    entry's bytes, or an already-seekable file object (a lazy byte-range view)."""
    buf = blob if hasattr(blob, "seek") else io.BytesIO(blob)
    info = sf.info(buf)
    sr = info.samplerate

    start_frame = 0 if start_time is None else int(start_time * sr)
    end_frame = info.frames if end_time is None else int(end_time * sr)
    num_frames = end_frame - start_frame

    buf.seek(0)
    data, sr = sf.read(
        buf,
        start=start_frame,
        stop=end_frame,
        dtype="float32",
        always_2d=True,
    )

    return AudioRead(
        file_type=fmt,
        modality="audio",
        sample_rate=sr,
        array=data,
    )


def _read_webm(
    blob: bytes,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read WebM/Opus via PyAV, with optional time slicing."""
    buf = io.BytesIO(blob)

    with av.open(buf, mode="r") as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        time_base = stream.time_base

        # Seek to start if requested
        if start_time is not None and start_time > 0:
            # av.open seeks in time_base units; use the stream's time_base
            target_pts = int(start_time / time_base)
            container.seek(target_pts, stream=stream)

        start_sample = 0 if start_time is None else int(start_time * sr)
        end_sample = None if end_time is None else int(end_time * sr)

        chunks = []
        total_samples = 0

        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()  # (channels, samples)
            frame_samples = arr.shape[1]
            frame_start_pts = frame.pts * float(time_base) * sr if frame.pts else total_samples

            chunks.append(arr)
            total_samples += frame_samples

            if end_sample is not None and total_samples >= (end_sample - start_sample):
                break

        if not chunks:
            return AudioRead(
                file_type="webm",
                modality="audio",
                sample_rate=sr,
                array=np.empty((0, stream.channels), dtype=np.float32),
            )

        raw = np.concatenate(chunks, axis=1)  # (channels, total)
        data = raw.T.astype(np.float32)       # (frames, channels)

        # Normalize integer formats to float
        if np.issubdtype(raw.dtype, np.integer):
            data /= float(np.iinfo(raw.dtype).max)

        # Trim to exact sample boundaries
        # After seeking, PyAV may decode a few extra frames before/after
        if end_sample is not None:
            keep = end_sample - start_sample
            data = data[:keep]

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
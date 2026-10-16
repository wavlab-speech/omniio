import io
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

class _RangeReader(io.RawIOBase):
    """Seekable read-only view of ``[start_offset, start_offset + size)`` in a
    local file or, over HTTP range requests, a remote one.

    Handed to soundfile so that a time slice reads only the header and the
    bytes it needs, instead of the whole entry -- an AMI headset recording is
    ~95 MB, a two-second segment of it 64 KB.

    Data is fetched in aligned blocks and kept for the life of the reader:
    libsndfile jumps between the header, the trailing chunks and the samples,
    and a plain ``io.BufferedReader`` would drop its buffer on every one of
    those seeks.  Each read fetches its run of missing blocks with one request.
    ``bytes_read`` counts what was actually fetched.
    """

    def __init__(
        self, archive_path: str, start_offset: int, size: int, remote: bool, block_size: int
    ):
        self._start = start_offset
        self._size = size
        self._pos = 0
        self._block = block_size
        self._blocks = {}
        self._remote = remote
        self.bytes_read = 0
        if remote:
            self._url = archive_path
            self._session = requests.Session()
            self._file = None
        else:
            self._file = open(archive_path, "rb")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def _fetch(self, lo: int, hi: int) -> bytes:
        """Bytes ``[lo, hi)`` of the entry, from the backing file or URL."""
        if self._remote:
            a = self._start + lo
            resp = self._session.get(self._url, headers={"Range": f"bytes={a}-{a + hi - lo - 1}"})
            resp.raise_for_status()
            data = resp.content[: hi - lo]
        else:
            self._file.seek(self._start + lo)
            data = self._file.read(hi - lo)
        self.bytes_read += len(data)
        return data

    def readinto(self, b):
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        bs = self._block
        first, last = self._pos // bs, (self._pos + n - 1) // bs
        missing = [i for i in range(first, last + 1) if i not in self._blocks]
        if missing:
            lo, hi = missing[0] * bs, min((missing[-1] + 1) * bs, self._size)
            data = self._fetch(lo, hi)
            for i in range(missing[0], missing[-1] + 1):
                if i not in self._blocks:
                    self._blocks[i] = data[i * bs - lo : (i + 1) * bs - lo]
        out = b"".join(self._blocks[i] for i in range(first, last + 1))
        skip = self._pos - first * bs
        k = len(out[skip : skip + n])
        b[:k] = out[skip : skip + n]
        self._pos += k
        return k

    def close(self):
        if not self.closed:
            if self._file is not None:
                self._file.close()
            if self._remote:
                self._session.close()
        super().close()


# Remote reads pay a round trip per request, so they fetch more per block.
_LOCAL_BLOCK = 64 * 1024
_REMOTE_BLOCK = 256 * 1024


def _read_sliced(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_time: Optional[float],
    end_time: Optional[float],
    remote: bool,
) -> AudioRead:
    """Time-sliced read that fetches only the bytes the slice needs.

    WebM is not seekable through soundfile, so it still reads the whole entry.
    """
    block = _REMOTE_BLOCK if remote else _LOCAL_BLOCK
    with _RangeReader(archive_path, start_offset, file_size, remote, block) as raw:
        fmt = _detect_format(raw.read(16))
        raw.seek(0)
        if fmt == "webm":
            return _read_webm(raw.read(), start_time, end_time)
        return _read_pcm(raw, fmt, start_time, end_time)


def _read_pcm(
    blob,
    fmt: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read FLAC/WAV/OGG via soundfile, with optional time slicing.

    ``blob`` is the entry's bytes, or a seekable file-like object positioned at
    its start.
    """
    buf = io.BytesIO(blob) if isinstance(blob, (bytes, bytearray)) else blob

    # One open: each one parses the header (and, for WAV, the trailing chunks),
    # which is a round trip per open on a remote entry.
    with sf.SoundFile(buf) as f:
        sr = f.samplerate
        start_frame = 0 if start_time is None else int(start_time * sr)
        end_frame = f.frames if end_time is None else int(end_time * sr)
        start_frame, end_frame, _ = slice(start_frame, end_frame).indices(f.frames)
        f.seek(start_frame)
        data = f.read(max(end_frame - start_frame, 0), dtype="float32", always_2d=True)

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
    if start_time is not None or end_time is not None:
        return _read_sliced(archive_path, start_offset, file_size, start_time, end_time, False)

    with open(archive_path, "rb") as f:
        f.seek(start_offset)
        blob = f.read(file_size)

    header = blob[:16]
    fmt = _detect_format(header)

    if fmt == "webm":
        return _read_webm(blob, start_time, end_time)
    else:
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
    if start_time is not None or end_time is not None:
        return _read_sliced(archive_url, start_offset, file_size, start_time, end_time, True)

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
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

def _read_pcm(
    blob: bytes,
    fmt: str,
    start_time: Optional[float],
    end_time: Optional[float],
) -> AudioRead:
    """Read FLAC/WAV/OGG via soundfile, with optional time slicing."""
    buf = io.BytesIO(blob)
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


def _decode_packets(container, stream):
    """Decode `stream` from the container's current position, yielding (packet pts, frame).

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
            yield packet.pts, frame


def _webm_frames(container, stream, seek_sample: Optional[int]):
    """Decode a WebM audio stream, yielding (index of the frame's first sample, frame).

    Indices count from the stream's first sample, so they address the same array a full
    read returns. With `seek_sample` None the container is decoded from where it is — the
    first sample of a fresh container is sample 0 by construction, no timestamps involved.
    An int seeks (to a cluster at or before it, minus the seek margin), anchors the run on
    the first packet's pts and counts samples from there; the caller trims what lands
    before the window.

    Why the packet's pts is the anchor: a packet at pts P decodes to a whole frame whose
    samples sit at [P, P + frame) on the stream's timeline, and a full decode discards the
    first `delay` samples of the stream, so that frame's first sample is output sample
    P - delay. After a seek the decoder discards `delay` samples again, from the head
    frame, so the head's first surviving sample is stream sample P + delay, i.e. output
    sample P. The trimmed head therefore starts exactly at its packet's pts, whatever the
    codec delay is.
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

    start_pts = stream.start_time or 0
    seek_s = max(0.0, seek_sample / sr - _OPUS_SEEK_MARGIN_S)
    container.seek(start_pts + int(seek_s / time_base), stream=stream)

    frames = _decode_packets(container, stream)
    first = next(frames, None)
    if first is None:
        return
    head_pts, head = first
    head_arr = _frame_samples(head)
    # The frame after the first is a whole frame, which is the grid the run sits on. The
    # first frame is not: the decoder trims the codec delay off it after a seek.
    nxt = next(frames, None)
    nxt_arr = None if nxt is None else _frame_samples(nxt[1])
    grid = (nxt_arr if nxt_arr is not None else head_arr).shape[1]

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
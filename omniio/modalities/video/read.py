"""
video/read.py — Read video (+ audio) from binary archive blobs.
"""

import io
from typing import Optional

import av
import numpy as np
import requests

from omniio.definitions import VideoRead

# The windowed-read machinery the audio reader already has: seek a little early, anchor
# the decoded run on the landing packet's pts measured from the first packet's pts, place
# every frame on the stream's own sample timeline, trim to the window, and decode from the
# top instead when the seek overshoots. Imported rather than lifted into a shared module
# so that the definitions, the comments recording which PyAV builds the timestamps hold
# on, and the tests that pin them (tests/test_webm_opus_windows.py) all stay in one place.
# `_webm_frames` is named for the container that forced it into existence, but nothing in
# it is WebM-only: the one Opus-shaped step, snapping a position onto the codec's frame
# grid, checks the frame size first and is a no-op for AAC's 1024.
from omniio.modalities.audio.read import (
    _OPUS_SEEK_MARGIN_S,
    _gather_window,
    _warn_if_untested_build,
    _webm_frames,
)


def _decode_audio_window(
    buf: io.BytesIO,
    start_sample: int,
    end_sample: Optional[int],
    channels: int,
) -> np.ndarray:
    """Decode [start_sample, end_sample) of the container's first audio stream.

    A pass of its own over a second container: the video pass has already seeked the
    first one to a keyframe before the requested frame, which is not where the audio
    window begins.

    This used to append every frame the decoder produced from wherever its seek landed and
    then keep the first (end - start) samples of that, so the audio began at the seek
    landing rather than at the requested time -- on WebM, where a seek lands on a cluster
    boundary, seconds away from it. Going through _webm_frames/_gather_window places each
    decoded frame on the stream's sample timeline and trims the lead-in against it.

    Returns (samples, channels) float32, integer sample formats scaled by their dtype max.
    """
    buf.seek(0)
    with av.open(buf, mode="r") as container:
        stream = container.streams.audio[0]
        # Seek only where it can skip real work and there is a timeline to seek against:
        # a window inside the first margin would seek to the start anyway, and decoding
        # from the top is both exact and cheaper than a seek that makes libavformat
        # re-apply the codec delay.
        seeking = (
            start_sample > _OPUS_SEEK_MARGIN_S * stream.rate
            and stream.start_time is not None
        )
        if seeking and "matroska" in (container.format.name or ""):
            # A window of a Matroska/WebM entry now rides on exactly the seek timestamps
            # the audio reader's windows are anchored on, so the same build check applies.
            _warn_if_untested_build()
        chunks, overshot = _gather_window(
            _webm_frames(container, stream, start_sample if seeking else None),
            start_sample,
            end_sample,
            detect_overshoot=seeking,
        )

    if overshot:
        # The seek landed after the window and nothing before it can be recovered from
        # that container: decode from the top in a fresh one, where the first decoded
        # sample is sample 0 by construction.
        buf.seek(0)
        with av.open(buf, mode="r") as container:
            stream = container.streams.audio[0]
            chunks, _ = _gather_window(
                _webm_frames(container, stream, None),
                start_sample,
                end_sample,
                detect_overshoot=False,
            )

    if not chunks:
        return np.empty((0, channels), dtype=np.float32)

    raw = np.concatenate(chunks, axis=1)  # (channels, window)
    audio_array = raw.T.astype(np.float32)  # (samples, channels)
    if np.issubdtype(raw.dtype, np.integer):
        audio_array /= float(np.iinfo(raw.dtype).max)
    return audio_array


def _decode_video(
    blob: bytes,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> VideoRead:
    """
    Decode MP4 bytes into numpy arrays for video and audio.

    Seeking can be specified either by frame index or by time in seconds.
    If both are provided, frame indices take priority.

    Args:
        blob:        Raw MP4 bytes.
        start_frame: First frame to include (None = 0).
        end_frame:   Frame to stop at, exclusive (None = last).
        start_time:  Start time in seconds (used if start_frame is None).
        end_time:    End time in seconds (used if end_frame is None).

    Returns:
        VideoRead with video_array (F, H, W, 3) uint8 and
        audio_array (samples, channels) float32.
    """
    buf = io.BytesIO(blob)

    with av.open(buf, mode="r") as container:
        video_stream = container.streams.video[0] if container.streams.video else None
        audio_stream = container.streams.audio[0] if container.streams.audio else None

        fps = None
        height = None
        width = None
        sample_rate = None

        if video_stream:
            fps = float(video_stream.average_rate) if video_stream.average_rate else None
            height = video_stream.height
            width = video_stream.width
            total_frames = video_stream.frames or None

        if audio_stream:
            sample_rate = audio_stream.rate

        # --- Resolve frame / time boundaries -------------------------
        if video_stream and fps:
            if start_frame is None and start_time is not None:
                start_frame = int(round(start_time * fps))     # nearest frame, not floor: a window
            if end_frame is None and end_time is not None:     # just under a boundary began a frame early
                end_frame = int(round(end_time * fps))

        if start_frame is None:
            start_frame = 0

        # Compute time boundaries for audio slicing.
        #
        # The audio window is cut on the video frame grid, not at the raw
        # start_time/end_time. video_array begins at start_frame, so audio beginning at
        # start_time instead would be out of step with it by up to one frame period
        # (40 ms at 25 fps, more wherever the frame rate is fractional). The two arrays
        # describing the same span is what a video reader owes its caller; a caller who
        # wants sample-exact audio bounds has omniio.audio.read for that. With no video
        # stream, or none that declares a frame rate, there is no grid to snap to and the
        # requested times are used as given -- such a container used to be handed
        # t_start = 0 and silently return audio from the beginning of the file.
        audio_start_sample = None
        audio_end_sample = None
        if audio_stream and sample_rate:
            if fps:
                t_start = start_frame / fps
                t_end = end_frame / fps if end_frame is not None else None
            else:
                t_start = start_time or 0.0
                t_end = end_time
            audio_start_sample = max(0, int(round(t_start * sample_rate)))
            audio_end_sample = (
                None
                if t_end is None
                else max(audio_start_sample, int(round(t_end * sample_rate)))
            )

        # --- Seek to nearest keyframe before start_frame -------------
        if start_frame > 0 and video_stream and fps:
            target_sec = start_frame / fps
            time_base = video_stream.time_base
            target_pts = int(target_sec / time_base)
            container.seek(target_pts, stream=video_stream)

        # --- Decode video frames -------------------------------------
        video_frames = []
        if video_stream:
            frame_idx = 0
            # After seeking, we need to count from decoded frames
            # and skip until we reach start_frame
            started = (start_frame == 0)

            for frame in container.decode(video=0):
                # Estimate frame index from pts
                if frame.pts is not None and fps and video_stream.time_base:
                    frame_idx = int(
                        float(frame.pts * video_stream.time_base) * fps + 0.5
                    )
                
                if not started:
                    if frame_idx >= start_frame:
                        started = True
                    else:
                        continue

                if end_frame is not None and frame_idx >= end_frame:
                    break

                arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                video_frames.append(arr)

        if video_frames:
            video_array = np.stack(video_frames, axis=0)  # (F, H, W, 3)
        else:
            h = height or 0
            w = width or 0
            video_array = np.empty((0, h, w, 3), dtype=np.uint8)

        # --- Decode audio (separate pass) ----------------------------
        audio_array = None
        if audio_stream:
            audio_array = _decode_audio_window(
                buf,
                audio_start_sample or 0,
                audio_end_sample,
                audio_stream.channels,
            )

    return VideoRead(
        file_type="mp4",
        modality="video",
        sample_rate=sample_rate,
        fps=fps,
        height=height,
        width=width,
        video_array=video_array,
        audio_array=audio_array,
    )


def video_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> VideoRead:
    """
    Read a single video entry from a local binary archive blob.

    Seeking can be by frame index or time in seconds (frames take priority).

    Args:
        archive_path: Path to the .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.
        start_frame:  First frame to include (None = 0).
        end_frame:    Frame to stop at, exclusive (None = last).
        start_time:   Start time in seconds (alternative to start_frame).
        end_time:     End time in seconds (alternative to end_frame).

    Returns:
        VideoRead with video_array (F,H,W,3) uint8,
        audio_array (samples, channels) float32.
    """
    with open(archive_path, "rb") as f:
        f.seek(start_offset)
        blob = f.read(file_size)

    return _decode_video(blob, start_frame, end_frame, start_time, end_time)


def video_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> VideoRead:
    """
    Read a single video entry from a remote binary archive via HTTP range request.

    Args:
        archive_url:  URL to the remote .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.
        start_frame:  First frame to include (None = 0).
        end_frame:    Frame to stop at, exclusive (None = last).
        start_time:   Start time in seconds (alternative to start_frame).
        end_time:     End time in seconds (alternative to end_frame).

    Returns:
        VideoRead with video_array (F,H,W,3) uint8,
        audio_array (samples, channels) float32.
    """
    end_byte = start_offset + file_size - 1
    headers = {"Range": f"bytes={start_offset}-{end_byte}"}

    resp = requests.get(archive_url, headers=headers)
    resp.raise_for_status()

    return _decode_video(resp.content, start_frame, end_frame, start_time, end_time)
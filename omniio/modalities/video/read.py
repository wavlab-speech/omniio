"""
video/read.py — Read video (+ audio) from binary archive blobs.
"""

import io
from typing import Optional

import av
import numpy as np
import requests

from omniio.definitions import VideoRead

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
                start_frame = int(start_time * fps)
            if end_frame is None and end_time is not None:
                end_frame = int(end_time * fps)

        if start_frame is None:
            start_frame = 0

        # Compute time boundaries for audio slicing
        audio_start_sample = None
        audio_end_sample = None
        if audio_stream and sample_rate:
            t_start = start_frame / fps if fps else 0
            t_end = end_frame / fps if (end_frame and fps) else None
            audio_start_sample = int(t_start * sample_rate)
            audio_end_sample = int(t_end * sample_rate) if t_end else None

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
            buf.seek(0)
            with av.open(buf, mode="r") as audio_container:
                a_stream = audio_container.streams.audio[0]

                # Seek for audio if needed
                if audio_start_sample and audio_start_sample > 0 and sample_rate:
                    target_sec = audio_start_sample / sample_rate
                    target_pts = int(target_sec / a_stream.time_base)
                    audio_container.seek(target_pts, stream=a_stream)

                audio_chunks = []
                total_audio_samples = 0

                for frame in audio_container.decode(audio=0):
                    arr = frame.to_ndarray()  # (channels, samples)
                    audio_chunks.append(arr)
                    total_audio_samples += arr.shape[1]

                    if (audio_end_sample is not None
                            and total_audio_samples >= audio_end_sample):
                        break

                if audio_chunks:
                    raw_audio = np.concatenate(audio_chunks, axis=1)  # (ch, samples)
                    audio_array = raw_audio.T.astype(np.float32)      # (samples, ch)

                    if np.issubdtype(raw_audio.dtype, np.integer):
                        audio_array /= float(np.iinfo(raw_audio.dtype).max)

                    if audio_end_sample is not None:
                        keep = audio_end_sample - (audio_start_sample or 0)
                        audio_array = audio_array[:keep]
                else:
                    audio_array = np.empty(
                        (0, audio_stream.channels), dtype=np.float32
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
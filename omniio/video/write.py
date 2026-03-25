"""
video/write.py — Encode video (+ optional audio) to MP4 H.264+AAC bytes.
"""

import io
from typing import Optional, Tuple, Union
from pathlib import Path

import av
import numpy as np


# Maps source container format to whether we can fast-path copy
_FASTPATH_FORMATS = {"mp4", "mov"}


def _detect_container(header: bytes) -> Optional[str]:
    """Sniff container from first bytes."""
    # ftyp box = MP4/MOV family
    if len(header) >= 8 and header[4:8] in (b"ftyp", b"moov", b"mdat"):
        return "mp4"
    if len(header) >= 4 and header[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    return None


def _probe_file(path: str) -> dict:
    """Return basic metadata from an existing video file."""
    with av.open(path, mode="r") as c:
        vs = c.streams.video[0] if c.streams.video else None
        aus = c.streams.audio[0] if c.streams.audio else None

        info = {}
        if vs:
            info["width"] = vs.width
            info["height"] = vs.height
            info["fps"] = float(vs.average_rate) if vs.average_rate else None
            info["video_codec"] = vs.codec_context.name
            # Note: gop_size is only available on encoders, not decoders
        if aus:
            info["sample_rate"] = aus.rate
            info["audio_channels"] = aus.codec_context.channels
            info["audio_codec"] = aus.codec_context.name

        # Detect container from first bytes
        c.seek(0)
        info["format"] = c.format.name  # e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    return info


def _is_mp4_h264_aac(info: dict) -> bool:
    """Check if the file is already MP4 with H.264 video and AAC audio."""
    fmt = info.get("format", "")
    is_mp4 = any(f in fmt for f in ("mp4", "mov"))
    is_h264 = info.get("video_codec") == "h264"
    has_audio = "audio_codec" in info
    is_aac = info.get("audio_codec") == "aac" if has_audio else True
    return is_mp4 and is_h264 and is_aac


def video_write(
    video_path: Union[str, Path],
    item_id: str,
    target_fps: Optional[int] = None,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    gop_size: int = 30,
    video_bitrate: int = 2_000_000,
    audio_bitrate: int = 128_000,
    crf: int = 23,
) -> Tuple[bytes, dict]:
    """
    Read a video file, optionally re-encode to MP4 H.264+AAC, and return
    raw bytes + metadata dict.

    Fast path: if the source is already MP4/H.264+AAC and no resolution or
    fps change is requested, return the file bytes as-is.

    Args:
        video_path:    Path to source video file (str or Path).
        item_id:       Unique identifier for this sample.
        target_fps:    Target frame rate (None = keep original).
        target_height: Target height in pixels (None = keep original).
        target_width:  Target width in pixels (None = keep original).
        gop_size:      Keyframe interval (frames). Lower = faster seeks, larger files.
        video_bitrate: Target video bitrate (used as maxrate with CRF).
        audio_bitrate: Target AAC audio bitrate.
        crf:           Constant rate factor for H.264 (0-51, lower = better quality).

    Returns:
        (raw_bytes, metadata_dict)
    """
    # Convert Path to string
    video_path = str(video_path)

    info = _probe_file(video_path)

    src_fps = info.get("fps")
    src_height = info.get("height")
    src_width = info.get("width")
    src_sample_rate = info.get("sample_rate")

    if target_fps is None:
        target_fps = src_fps
    if target_height is None:
        target_height = src_height
    if target_width is None:
        target_width = src_width

    # --- Fast path: already MP4 H.264+AAC, no transform needed -------
    no_resize = (target_height == src_height and target_width == src_width)
    no_fps_change = (target_fps == src_fps)

    if _is_mp4_h264_aac(info) and no_resize and no_fps_change:
        with open(video_path, "rb") as f:
            raw_bytes = f.read()

        metadata = {
            "fps": src_fps,
            "height": src_height,
            "width": src_width,
            "sample_rate": src_sample_rate,
            "format": "mp4",
            "video_codec": "h264",
            "audio_codec": info.get("audio_codec"),
            "gop_size": None,  # Cannot determine from decoder
        }
        return raw_bytes, metadata

    # --- Slow path: decode and re-encode to MP4 ----------------------
    buf = io.BytesIO()

    with av.open(video_path, mode="r") as in_container:
        in_video = in_container.streams.video[0] if in_container.streams.video else None
        in_audio = in_container.streams.audio[0] if in_container.streams.audio else None

        with av.open(buf, mode="w", format="mp4") as out_container:
            # -- set up video stream --
            out_video = None
            if in_video:
                out_video = out_container.add_stream("libx264", rate=int(target_fps))
                out_video.width = target_width
                out_video.height = target_height
                out_video.pix_fmt = "yuv420p"
                out_video.gop_size = gop_size
                out_video.options = {
                    "crf": str(crf),
                    "preset": "medium",
                    "maxrate": str(video_bitrate),
                    "bufsize": str(video_bitrate * 2),
                }

            # -- set up audio stream --
            out_audio = None
            if in_audio:
                out_audio = out_container.add_stream("aac", rate=in_audio.rate)
                out_audio.layout = in_audio.layout.name
                out_audio.bit_rate = audio_bitrate

            # -- decode / encode loop --
            streams_to_decode = []
            if in_video:
                streams_to_decode.append(in_video)
            if in_audio:
                streams_to_decode.append(in_audio)

            for packet in in_container.demux(streams_to_decode):
                if packet.stream == in_video and out_video:
                    for frame in packet.decode():
                        # Rescale if resolution changed
                        if (frame.width != target_width or
                                frame.height != target_height):
                            frame = frame.reformat(
                                width=target_width,
                                height=target_height,
                                format="yuv420p",
                            )
                        elif frame.format.name != "yuv420p":
                            frame = frame.reformat(format="yuv420p")

                        for out_packet in out_video.encode(frame):
                            out_container.mux(out_packet)

                elif packet.stream == in_audio and out_audio:
                    for frame in packet.decode():
                        frame.pts = None  # let encoder set pts
                        for out_packet in out_audio.encode(frame):
                            out_container.mux(out_packet)

            # Flush encoders
            if out_video:
                for out_packet in out_video.encode(None):
                    out_container.mux(out_packet)
            if out_audio:
                for out_packet in out_audio.encode(None):
                    out_container.mux(out_packet)

    raw_bytes = buf.getvalue()

    metadata = {
        "fps": target_fps,
        "height": target_height,
        "width": target_width,
        "sample_rate": src_sample_rate,
        "format": "mp4",
        "video_codec": "h264",
        "audio_codec": "aac" if in_audio else None,
        "gop_size": gop_size,
    }

    return raw_bytes, metadata
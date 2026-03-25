import io
import av
import numpy as np
import soundfile as sf
from typing import Optional, Tuple


BIT_DEPTH_TO_SUBTYPE = {
    8:  "PCM_S8",
    16: "PCM_16",
    24: "PCM_24",
    32: "PCM_32",
}

SUBTYPE_TO_BIT_DEPTH = {v: k for k, v in BIT_DEPTH_TO_SUBTYPE.items()}

PYAV_FORMATS = {"webm", "opus"}


def _read_webm(audio_path: str) -> Tuple[np.ndarray, int, int, int]:
    """
    Read a WebM/Opus file via PyAV.

    Returns:
        (samples_float64 [frames, channels], sample_rate, channels, num_frames)
    """
    with av.open(audio_path, mode="r") as container:
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        channels = stream.channels

        chunks = []
        for frame in container.decode(audio=0):
            # frame.to_ndarray() shape: (channels, samples) for planar,
            # or (1, samples * channels) for interleaved — reformat first
            frame = frame.to_ndarray()
            chunks.append(frame)

        raw = np.concatenate(chunks, axis=1)  # (channels, total_samples)
        # Transpose to (frames, channels) and normalize to float64
        data = raw.T.astype(np.float64)
        if np.issubdtype(raw.dtype, np.integer):
            max_val = float(np.iinfo(raw.dtype).max)
            data /= max_val

    return data, sample_rate, data.shape[1], data.shape[0]


def _write_webm(data: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode float64 audio (frames, channels) to WebM/Opus bytes via PyAV.
    Opus always uses 48kHz internally; PyAV handles resampling.
    """
    buf = io.BytesIO()

    with av.open(buf, mode="w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.channels = data.shape[1]

        # Convert float64 -> s16 interleaved for the encoder
        samples_s16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)

        frame = av.AudioFrame.from_ndarray(
            samples_s16.T,  # (channels, frames)
            format="s16",
            layout="stereo" if data.shape[1] == 2 else "mono",
        )
        frame.rate = sample_rate
        frame.pts = 0

        for packet in stream.encode(frame):
            container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return buf.getvalue()


def _get_webm_info(audio_path: str) -> dict:
    """Read metadata from a WebM/Opus file."""
    with av.open(audio_path, mode="r") as container:
        stream = container.streams.audio[0]
        return {
            "sample_rate": stream.rate,
            "channels": stream.channels,
            "samples": stream.frames if stream.frames > 0 else None,
            "format": "webm",
            "bit_depth": None,  # Opus is not PCM-based
        }


def audio_write(
    audio_path: str,
    item_id: str,
    target_format: Optional[str] = None,
    target_bit_depth: Optional[int] = None,
) -> Tuple[bytes, dict]:
    """
    Read an audio file, optionally convert format/bit depth, and return
    raw bytes + metadata dict.

    Uses soundfile for FLAC/WAV and PyAV for WebM/Opus.

    Args:
        audio_path:       Path to the source audio file.
        item_id:          Unique identifier for this sample.
        target_format:    Desired output format ('flac', 'wav', 'webm').
                          If None, keeps the original format.
        target_bit_depth: Desired bit depth (e.g. 16, 24, 32).
                          Ignored when target format is webm/opus.

    Returns:
        (raw_bytes, metadata_dict)
    """

    # --- Detect source format ----------------------------------------
    src_is_webm = audio_path.lower().endswith((".webm", ".opus"))

    if src_is_webm:
        webm_info = _get_webm_info(audio_path)
        src_format = "webm"
        src_bit_depth = None
        src_sample_rate = webm_info["sample_rate"]
        src_channels = webm_info["channels"]
        src_frames = webm_info["samples"]
    else:
        info = sf.info(audio_path)
        src_format = info.format.lower()
        src_subtype = info.subtype
        src_sample_rate = info.samplerate
        src_channels = info.channels
        src_frames = info.frames
        src_bit_depth = SUBTYPE_TO_BIT_DEPTH.get(src_subtype)

    if target_format is None:
        target_format = src_format
    if target_bit_depth is None:
        target_bit_depth = src_bit_depth  # None for Opus, int for PCM

    target_format = target_format.lower()
    target_is_webm = target_format in PYAV_FORMATS

    # --- Fast path: no conversion needed -----------------------------
    needs_conversion = True
    if target_format == src_format:
        if target_is_webm:
            # Opus is not PCM — nothing to convert if format matches
            needs_conversion = False
        elif (
            src_bit_depth is not None
            and target_bit_depth is not None
            and target_bit_depth >= src_bit_depth
        ):
            needs_conversion = False

    if not needs_conversion:
        with open(audio_path, "rb") as f:
            raw_bytes = f.read()

        metadata = {
            "sample_rate": src_sample_rate,
            "channels": src_channels,
            "samples": src_frames,
            "format": src_format,
            "bit_depth": src_bit_depth,
        }
        return raw_bytes, metadata

    # --- Slow path: decode then re-encode ----------------------------

    # Decode
    if src_is_webm:
        data, sr, channels, frames = _read_webm(audio_path)
    else:
        data, sr = sf.read(audio_path, dtype="float64", always_2d=True)

    # Encode
    if target_is_webm:
        raw_bytes = _write_webm(data, sr)
        metadata = {
            "sample_rate": 48000,  # Opus always outputs 48kHz
            "channels": data.shape[1],
            "samples": data.shape[0],
            "format": "webm",
            "bit_depth": None,
            "duration": data.shape[0] / 48000
        }
    else:
        target_subtype = BIT_DEPTH_TO_SUBTYPE.get(target_bit_depth)
        if target_subtype is None:
            raise ValueError(
                f"Unsupported target bit depth: {target_bit_depth}. "
                f"Supported: {sorted(BIT_DEPTH_TO_SUBTYPE.keys())}"
            )

        buf = io.BytesIO()
        sf.write(
            buf,
            data,
            sr,
            format=target_format.upper(),
            subtype=target_subtype,
        )
        raw_bytes = buf.getvalue()

        metadata = {
            "sample_rate": sr,
            "channels": data.shape[1],
            "samples": data.shape[0],
            "format": target_format,
            "bit_depth": target_bit_depth,
            "duration": data.shape[0] / sr
        }

    return raw_bytes, metadata
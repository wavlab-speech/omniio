"""Pytest configuration and fixtures for omniio tests."""

import io
import os
import tempfile
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf
import zstandard as zstd


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_wav(temp_dir):
    """Create a sample WAV audio file."""
    audio_path = temp_dir / "sample.wav"
    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)

    # Generate stereo sine wave
    t = np.linspace(0, duration, num_samples)
    freq_left = 440.0  # A4
    freq_right = 554.37  # C#5
    left = np.sin(2 * np.pi * freq_left * t).astype(np.float32)
    right = np.sin(2 * np.pi * freq_right * t).astype(np.float32)
    audio = np.stack([left, right], axis=1)

    sf.write(audio_path, audio, sample_rate, subtype='PCM_16')
    return audio_path, sample_rate, audio.shape[0]


@pytest.fixture
def sample_audio_flac(temp_dir):
    """Create a sample FLAC audio file."""
    audio_path = temp_dir / "sample.flac"
    sample_rate = 44100
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Generate mono sine wave
    t = np.linspace(0, duration, num_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32).reshape(-1, 1)

    sf.write(audio_path, audio, sample_rate, format='FLAC')
    return audio_path, sample_rate, audio.shape[0]


@pytest.fixture
def sample_audio_webm(temp_dir):
    """Create a sample WebM/Opus audio file."""
    audio_path = temp_dir / "sample.webm"
    sample_rate = 48000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Generate stereo sine wave
    t = np.linspace(0, duration, num_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32).reshape(-1, 1)
    audio = np.tile(audio, (1, 2))  # Make stereo

    # Convert to int16 for encoding
    audio_int16 = (audio * 32767).astype(np.int16)

    with av.open(str(audio_path), mode='w', format='webm') as container:
        stream = container.add_stream('libopus', rate=sample_rate)
        stream.layout = 'stereo'

        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(audio_int16.T),
            format='s16p',  # Use planar format for multi-channel
            layout='stereo'
        )
        frame.rate = sample_rate
        frame.pts = 0

        for packet in stream.encode(frame):
            container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return audio_path, 48000, num_samples


@pytest.fixture
def sample_video_mp4(temp_dir):
    """Create a sample MP4 video file with audio."""
    video_path = temp_dir / "sample.mp4"

    width, height = 320, 240
    fps = 30
    duration = 2.0
    num_frames = int(fps * duration)

    with av.open(str(video_path), mode='w') as container:
        # Video stream
        video_stream = container.add_stream('h264', rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = 'yuv420p'

        # Audio stream
        audio_stream = container.add_stream('aac', rate=44100)
        audio_stream.layout = 'stereo'

        # Generate video frames
        for i in range(num_frames):
            # Create a simple gradient pattern
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = int(255 * i / num_frames)  # Red gradient
            img[:, :, 1] = 128  # Fixed green
            img[:, :, 2] = 255 - int(255 * i / num_frames)  # Blue gradient

            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in video_stream.encode(frame):
                container.mux(packet)

        # Flush video
        for packet in video_stream.encode():
            container.mux(packet)

        # Generate audio
        audio_samples = int(44100 * duration)
        t = np.linspace(0, duration, audio_samples)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio = np.stack([audio, audio], axis=1)  # Stereo
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(audio_int16.T),
            format='s16p',  # Use planar format for multi-channel
            layout='stereo'
        )
        audio_frame.rate = 44100

        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)

        for packet in audio_stream.encode():
            container.mux(packet)

    return video_path, fps, width, height, num_frames


@pytest.fixture
def sample_text(temp_dir):
    """Create a sample text file."""
    text_path = temp_dir / "sample.txt"
    text_content = "This is a sample text for testing.\n" * 100
    text_path.write_text(text_content, encoding='utf-8')
    return text_path, text_content


@pytest.fixture
def compressed_text():
    """Create compressed text data."""
    text = "This is a test text for compression." * 10
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(text.encode('utf-8'))
    return compressed, text


@pytest.fixture
def sample_binary_archive(temp_dir, sample_audio_wav):
    """Create a sample binary archive with audio data."""
    archive_path = temp_dir / "archive.bin"
    audio_path, _, _ = sample_audio_wav

    # Read audio file
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    # Write to archive
    with open(archive_path, 'wb') as f:
        f.write(audio_bytes)

    return archive_path, 0, len(audio_bytes)

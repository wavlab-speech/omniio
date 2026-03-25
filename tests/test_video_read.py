"""Tests for video reading functionality."""

import pytest
import numpy as np

from omniio.video.read import (
    _decode_video,
    video_read_local,
)
from omniio.definitions import VideoRead


class TestVideoReadLocal:
    """Test local video reading."""

    def test_read_full_video(self, temp_dir, sample_video_mp4):
        """Test reading full video from archive."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        # Create archive with video
        archive_path = temp_dir / "video_archive.bin"
        with open(video_path, 'rb') as src, open(archive_path, 'wb') as dst:
            video_bytes = src.read()
            dst.write(video_bytes)

        result = video_read_local(archive_path, 0, len(video_bytes))

        assert isinstance(result, VideoRead)
        assert result.file_type == "mp4"
        assert result.modality == "video"
        assert result.fps == fps
        assert result.width == width
        assert result.height == height
        assert result.video_array is not None
        assert result.audio_array is not None

    def test_read_video_frame_shape(self, temp_dir, sample_video_mp4):
        """Test video array has correct shape."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        archive_path = temp_dir / "video_archive.bin"
        with open(video_path, 'rb') as src, open(archive_path, 'wb') as dst:
            video_bytes = src.read()
            dst.write(video_bytes)

        result = video_read_local(archive_path, 0, len(video_bytes))

        # Check video array shape: (frames, height, width, 3)
        assert result.video_array.ndim == 4
        assert result.video_array.shape[2] == width
        assert result.video_array.shape[1] == height
        assert result.video_array.shape[3] == 3  # RGB
        assert result.video_array.dtype == np.uint8

    def test_read_video_audio_shape(self, temp_dir, sample_video_mp4):
        """Test audio array has correct shape."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        archive_path = temp_dir / "video_archive.bin"
        with open(video_path, 'rb') as src, open(archive_path, 'wb') as dst:
            video_bytes = src.read()
            dst.write(video_bytes)

        result = video_read_local(archive_path, 0, len(video_bytes))

        # Check audio array shape: (samples, channels)
        assert result.audio_array.ndim == 2
        assert result.audio_array.shape[1] == 2  # Stereo
        assert result.audio_array.dtype == np.float32
        assert result.sample_rate == 44100

    def test_read_video_with_frame_slice(self, temp_dir, sample_video_mp4):
        """Test reading video with frame-based slicing."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        archive_path = temp_dir / "video_archive.bin"
        with open(video_path, 'rb') as src, open(archive_path, 'wb') as dst:
            video_bytes = src.read()
            dst.write(video_bytes)

        start_frame = 10
        end_frame = 30

        result = video_read_local(
            archive_path, 0, len(video_bytes),
            start_frame=start_frame,
            end_frame=end_frame
        )

        # Should have approximately end_frame - start_frame frames
        expected_frames = end_frame - start_frame
        assert abs(result.video_array.shape[0] - expected_frames) <= 2

    def test_read_video_with_time_slice(self, temp_dir, sample_video_mp4):
        """Test reading video with time-based slicing."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        archive_path = temp_dir / "video_archive.bin"
        with open(video_path, 'rb') as src, open(archive_path, 'wb') as dst:
            video_bytes = src.read()
            dst.write(video_bytes)

        start_time = 0.5
        end_time = 1.5

        result = video_read_local(
            archive_path, 0, len(video_bytes),
            start_time=start_time,
            end_time=end_time
        )

        # Calculate expected frames
        expected_frames = int((end_time - start_time) * fps)
        assert abs(result.video_array.shape[0] - expected_frames) <= 2

    def test_read_video_with_offset(self, temp_dir, sample_video_mp4):
        """Test reading video from archive with offset."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        # Create archive with padding
        archive_path = temp_dir / "offset_video_archive.bin"
        padding = b'\x00' * 1024
        with open(video_path, 'rb') as src:
            video_bytes = src.read()

        with open(archive_path, 'wb') as dst:
            dst.write(padding)
            dst.write(video_bytes)

        result = video_read_local(
            archive_path,
            start_offset=len(padding),
            file_size=len(video_bytes)
        )

        assert result.video_array is not None
        assert result.width == width
        assert result.height == height


class TestDecodeVideo:
    """Test video decoding function."""

    def test_decode_video_full(self, sample_video_mp4):
        """Test decoding full video."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        with open(video_path, 'rb') as f:
            blob = f.read()

        result = _decode_video(blob)

        assert isinstance(result, VideoRead)
        assert result.fps == fps
        assert result.width == width
        assert result.height == height

    def test_decode_video_frame_priority(self, sample_video_mp4):
        """Test that frame indices take priority over time."""
        video_path, fps, _, _, _ = sample_video_mp4

        with open(video_path, 'rb') as f:
            blob = f.read()

        # Provide both frame and time parameters
        start_frame = 15
        end_frame = 45
        result = _decode_video(
            blob,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=0.0,  # Should be ignored
            end_time=10.0    # Should be ignored
        )

        expected_frames = end_frame - start_frame
        # Frame parameters should take priority
        assert abs(result.video_array.shape[0] - expected_frames) <= 2

    def test_decode_video_audio_sync(self, sample_video_mp4):
        """Test that audio and video are properly synced."""
        video_path, fps, _, _, _ = sample_video_mp4

        with open(video_path, 'rb') as f:
            blob = f.read()

        start_time = 0.5
        end_time = 1.5
        result = _decode_video(blob, start_time=start_time, end_time=end_time)

        # Calculate expected durations
        video_duration = result.video_array.shape[0] / result.fps
        audio_duration = result.audio_array.shape[0] / result.sample_rate

        # Audio and video durations should be approximately equal
        assert abs(video_duration - audio_duration) < 0.1

"""Tests for data definitions."""

import numpy as np
import pytest

from omniio.definitions import (
    ArchiveRead,
    AudioRead,
    VideoRead,
    TextRead,
)


class TestArchiveRead:
    """Test ArchiveRead base class."""

    def test_create_archive_read(self):
        """Test creating ArchiveRead instance."""
        archive = ArchiveRead(file_type="test", modality="test_modality")

        assert archive.file_type == "test"
        assert archive.modality == "test_modality"

    def test_archive_read_defaults(self):
        """Test ArchiveRead with default values."""
        archive = ArchiveRead()

        assert archive.file_type is None
        assert archive.modality is None


class TestAudioRead:
    """Test AudioRead dataclass."""

    def test_create_audio_read(self):
        """Test creating AudioRead instance."""
        array = np.random.rand(1000, 2).astype(np.float32)

        audio = AudioRead(
            file_type="wav",
            modality="audio",
            sample_rate=16000,
            array=array
        )

        assert audio.file_type == "wav"
        assert audio.modality == "audio"
        assert audio.sample_rate == 16000
        assert np.array_equal(audio.array, array)

    def test_audio_read_inheritance(self):
        """Test AudioRead inherits from ArchiveRead."""
        audio = AudioRead()

        assert isinstance(audio, ArchiveRead)
        assert hasattr(audio, 'file_type')
        assert hasattr(audio, 'modality')
        assert hasattr(audio, 'sample_rate')
        assert hasattr(audio, 'array')


class TestVideoRead:
    """Test VideoRead dataclass."""

    def test_create_video_read(self):
        """Test creating VideoRead instance."""
        video_array = np.random.randint(0, 255, (100, 240, 320, 3), dtype=np.uint8)
        audio_array = np.random.rand(44100, 2).astype(np.float32)

        video = VideoRead(
            file_type="mp4",
            modality="video",
            sample_rate=44100,
            fps=30.0,
            height=240,
            width=320,
            audio_array=audio_array,
            video_array=video_array
        )

        assert video.file_type == "mp4"
        assert video.modality == "video"
        assert video.sample_rate == 44100
        assert video.fps == 30.0
        assert video.height == 240
        assert video.width == 320
        assert np.array_equal(video.audio_array, audio_array)
        assert np.array_equal(video.video_array, video_array)

    def test_video_read_inheritance(self):
        """Test VideoRead inherits from ArchiveRead."""
        video = VideoRead()

        assert isinstance(video, ArchiveRead)
        assert hasattr(video, 'file_type')
        assert hasattr(video, 'modality')
        assert hasattr(video, 'sample_rate')
        assert hasattr(video, 'fps')
        assert hasattr(video, 'height')
        assert hasattr(video, 'width')
        assert hasattr(video, 'audio_array')
        assert hasattr(video, 'video_array')


class TestTextRead:
    """Test TextRead dataclass."""

    def test_create_text_read(self):
        """Test creating TextRead instance."""
        text_content = "This is a test text."

        text = TextRead(
            file_type="text",
            modality="text",
            text=text_content
        )

        assert text.file_type == "text"
        assert text.modality == "text"
        assert text.text == text_content

    def test_text_read_inheritance(self):
        """Test TextRead inherits from ArchiveRead."""
        text = TextRead()

        assert isinstance(text, ArchiveRead)
        assert hasattr(text, 'file_type')
        assert hasattr(text, 'modality')
        assert hasattr(text, 'text')

    def test_text_read_unicode(self):
        """Test TextRead with unicode content."""
        unicode_text = "Hello 世界! Привет мир! 🌍"

        text = TextRead(
            file_type="text",
            modality="text",
            text=unicode_text
        )

        assert text.text == unicode_text

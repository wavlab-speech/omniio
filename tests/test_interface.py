"""Tests for main interface functions."""

import pytest
from unittest.mock import patch, MagicMock

from omniio.interface import audio_read
from omniio.definitions import AudioRead


class TestAudioRead:
    """Test audio_read interface function."""

    @patch('omniio.interface.os.path.exists')
    @patch('omniio.interface.audio_read_local')
    def test_audio_read_local_path(self, mock_local, mock_exists):
        """Test audio_read routes to local when path exists."""
        mock_exists.return_value = True
        mock_local.return_value = AudioRead(
            file_type="wav",
            modality="audio",
            sample_rate=16000,
            array=MagicMock()
        )

        result = audio_read(
            archive_path="/path/to/local.bin",
            start_offset=0,
            file_size=1000
        )

        mock_exists.assert_called_once_with("/path/to/local.bin")
        mock_local.assert_called_once_with(
            "/path/to/local.bin", 0, 1000, None, None
        )
        assert result.file_type == "wav"

    @patch('omniio.interface.os.path.exists')
    @patch('omniio.interface.audio_read_remote')
    def test_audio_read_remote_url(self, mock_remote, mock_exists):
        """Test audio_read routes to remote when path doesn't exist."""
        mock_exists.return_value = False
        mock_remote.return_value = AudioRead(
            file_type="flac",
            modality="audio",
            sample_rate=44100,
            array=MagicMock()
        )

        result = audio_read(
            archive_path="https://example.com/remote.bin",
            start_offset=100,
            file_size=2000,
            start_time=5.0,
            end_time=10.0
        )

        mock_exists.assert_called_once_with("https://example.com/remote.bin")
        mock_remote.assert_called_once_with(
            "https://example.com/remote.bin", 100, 2000, 5.0, 10.0
        )
        assert result.file_type == "flac"

    def test_audio_read_real_local(self, sample_binary_archive):
        """Test audio_read with real local file."""
        archive_path, start_offset, file_size = sample_binary_archive

        result = audio_read(archive_path, start_offset, file_size)

        assert isinstance(result, AudioRead)
        assert result.sample_rate == 16000
        assert result.array is not None


class TestTextRead:
    """Test text_read interface function."""

    @patch('omniio.interface.os.path.exists')
    @patch('omniio.interface.text_read_local')
    def test_text_read_local_path(self, mock_local, mock_exists):
        """Test text_read routes to local when path exists."""
        from omniio.definitions import TextRead

        mock_exists.return_value = True
        mock_local.return_value = TextRead(
            file_type="text",
            modality="text",
            text="Sample text content"
        )

        # Import here to avoid issues with patching
        from omniio.interface import text_read

        result = text_read(
            archive_path="/path/to/local.bin",
            start_offset=0,
            file_size=100
        )

        mock_exists.assert_called_once()
        mock_local.assert_called_once()
        assert result.text == "Sample text content"

    @patch('omniio.interface.os.path.exists')
    @patch('omniio.interface.text_read_remote')
    def test_text_read_remote_url(self, mock_remote, mock_exists):
        """Test text_read routes to remote when path doesn't exist."""
        from omniio.definitions import TextRead

        mock_exists.return_value = False
        mock_remote.return_value = TextRead(
            file_type="text",
            modality="text",
            text="Remote text content"
        )

        # Import here
        from omniio.interface import text_read

        result = text_read(
            archive_path="https://example.com/remote.bin",
            start_offset=50,
            file_size=200
        )

        mock_exists.assert_called_once()
        mock_remote.assert_called_once()
        assert result.text == "Remote text content"

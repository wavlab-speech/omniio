"""Tests for audio reading functionality."""

import io
import pytest
import numpy as np

from omniio.audio.read import (
    _detect_format,
    _read_pcm,
    _read_webm,
    audio_read_local,
)
from omniio.definitions import AudioRead


class TestFormatDetection:
    """Test audio format detection from magic bytes."""

    def test_detect_flac(self):
        """Test FLAC format detection."""
        header = b"fLaC" + b"\x00" * 12
        assert _detect_format(header) == "flac"

    def test_detect_wav(self):
        """Test WAV format detection."""
        header = b"RIFF" + b"\x00" * 12
        assert _detect_format(header) == "wav"

    def test_detect_webm(self):
        """Test WebM format detection."""
        header = b"\x1aE\xdf\xa3" + b"\x00" * 12
        assert _detect_format(header) == "webm"

    def test_detect_ogg(self):
        """Test OGG format detection."""
        header = b"OggS" + b"\x00" * 12
        assert _detect_format(header) == "ogg"

    def test_unknown_format(self):
        """Test unknown format raises ValueError."""
        header = b"XXXX" + b"\x00" * 12
        with pytest.raises(ValueError, match="Unknown audio format"):
            _detect_format(header)


class TestAudioReadLocal:
    """Test local audio reading."""

    def test_read_wav_full(self, sample_binary_archive):
        """Test reading full WAV file from archive."""
        archive_path, start_offset, file_size = sample_binary_archive

        result = audio_read_local(archive_path, start_offset, file_size)

        assert isinstance(result, AudioRead)
        assert result.file_type == "wav"
        assert result.modality == "audio"
        assert result.sample_rate == 16000
        assert result.array.shape[1] == 2  # Stereo
        assert result.array.dtype == np.float32
        assert result.array.shape[0] == 32000  # 2 seconds at 16kHz

    def test_read_wav_with_time_slice(self, sample_binary_archive):
        """Test reading WAV with time-based slicing."""
        archive_path, start_offset, file_size = sample_binary_archive

        # Read 0.5 seconds starting from 0.5 seconds
        result = audio_read_local(
            archive_path, start_offset, file_size,
            start_time=0.5, end_time=1.0
        )

        assert isinstance(result, AudioRead)
        assert result.sample_rate == 16000
        expected_frames = int(0.5 * 16000)
        assert abs(result.array.shape[0] - expected_frames) <= 1

    def test_read_flac(self, temp_dir, sample_audio_flac):
        """Test reading FLAC file."""
        audio_path, sample_rate, num_samples = sample_audio_flac

        # Create archive with FLAC data
        archive_path = temp_dir / "archive_flac.bin"
        with open(audio_path, 'rb') as src, open(archive_path, 'wb') as dst:
            audio_bytes = src.read()
            dst.write(audio_bytes)

        result = audio_read_local(archive_path, 0, len(audio_bytes))

        assert result.file_type == "flac"
        assert result.sample_rate == sample_rate
        assert result.array.shape[1] == 1  # Mono

    def test_read_webm(self, temp_dir, sample_audio_webm):
        """Test reading WebM/Opus file."""
        audio_path, sample_rate, num_samples = sample_audio_webm

        # Create archive with WebM data
        archive_path = temp_dir / "archive_webm.bin"
        with open(audio_path, 'rb') as src, open(archive_path, 'wb') as dst:
            audio_bytes = src.read()
            dst.write(audio_bytes)

        result = audio_read_local(archive_path, 0, len(audio_bytes))

        assert result.file_type == "webm"
        assert result.sample_rate == sample_rate
        assert result.array.shape[1] == 2  # Stereo

    def test_read_with_offset(self, temp_dir, sample_audio_wav, sample_audio_flac):
        """Test reading from archive with multiple files."""
        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac

        # Create archive with two files
        archive_path = temp_dir / "multi_archive.bin"
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        with open(flac_path, 'rb') as f:
            flac_bytes = f.read()

        with open(archive_path, 'wb') as f:
            f.write(wav_bytes)
            f.write(flac_bytes)

        # Read second file (FLAC)
        result = audio_read_local(
            archive_path,
            start_offset=len(wav_bytes),
            file_size=len(flac_bytes)
        )

        assert result.file_type == "flac"
        assert result.sample_rate == 44100


class TestAudioReadPCM:
    """Test PCM audio reading."""

    def test_read_pcm_full(self, sample_audio_wav):
        """Test reading full PCM audio."""
        audio_path, sample_rate, num_samples = sample_audio_wav

        with open(audio_path, 'rb') as f:
            blob = f.read()

        result = _read_pcm(blob, "wav", None, None)

        assert result.sample_rate == sample_rate
        assert result.array.shape[0] == num_samples
        assert result.file_type == "wav"

    def test_read_pcm_with_time_slice(self, sample_audio_wav):
        """Test reading PCM with time slicing."""
        audio_path, sample_rate, _ = sample_audio_wav

        with open(audio_path, 'rb') as f:
            blob = f.read()

        start_time = 0.5
        end_time = 1.5
        result = _read_pcm(blob, "wav", start_time, end_time)

        expected_frames = int((end_time - start_time) * sample_rate)
        assert abs(result.array.shape[0] - expected_frames) <= 1


class TestAudioReadWebM:
    """Test WebM/Opus audio reading."""

    def test_read_webm_full(self, sample_audio_webm):
        """Test reading full WebM audio."""
        audio_path, sample_rate, num_samples = sample_audio_webm

        with open(audio_path, 'rb') as f:
            blob = f.read()

        result = _read_webm(blob, None, None)

        assert result.sample_rate == sample_rate
        assert result.file_type == "webm"
        assert result.array.shape[1] == 2  # Stereo

    def test_read_webm_with_time_slice(self, sample_audio_webm):
        """Test reading WebM with time slicing."""
        audio_path, sample_rate, _ = sample_audio_webm

        with open(audio_path, 'rb') as f:
            blob = f.read()

        start_time = 0.2
        end_time = 0.8
        result = _read_webm(blob, start_time, end_time)

        expected_frames = int((end_time - start_time) * sample_rate)
        # Allow some tolerance due to frame boundaries
        assert abs(result.array.shape[0] - expected_frames) < sample_rate * 0.1

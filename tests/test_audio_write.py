"""Tests for audio writing functionality."""

import pytest
import numpy as np
import soundfile as sf

from omniio.audio.write import (
    audio_write,
    _read_webm,
    _write_webm,
    _get_webm_info,
)


class TestAudioWrite:
    """Test audio writing and conversion."""

    def test_write_wav_no_conversion(self, sample_audio_wav):
        """Test writing WAV without conversion (fast path)."""
        audio_path, sample_rate, num_samples = sample_audio_wav

        raw_bytes, metadata = audio_write(
            audio_path, "test_001", target_format=None
        )

        assert metadata['format'] == 'wav'
        assert metadata['sample_rate'] == sample_rate
        assert metadata['channels'] == 2
        assert metadata['samples'] == num_samples
        assert len(raw_bytes) > 0

    def test_write_flac_no_conversion(self, sample_audio_flac):
        """Test writing FLAC without conversion."""
        audio_path, sample_rate, num_samples = sample_audio_flac

        raw_bytes, metadata = audio_write(
            audio_path, "test_002", target_format=None
        )

        assert metadata['format'] == 'flac'
        assert metadata['sample_rate'] == sample_rate
        assert metadata['channels'] == 1

    def test_write_wav_to_flac(self, sample_audio_wav):
        """Test converting WAV to FLAC."""
        audio_path, sample_rate, num_samples = sample_audio_wav

        raw_bytes, metadata = audio_write(
            audio_path, "test_003",
            target_format="flac",
            target_bit_depth=16
        )

        assert metadata['format'] == 'flac'
        assert metadata['sample_rate'] == sample_rate
        assert metadata['bit_depth'] == 16
        assert metadata['channels'] == 2

    def test_write_wav_to_webm(self, sample_audio_wav):
        """Test converting WAV to WebM/Opus."""
        audio_path, _, _ = sample_audio_wav

        raw_bytes, metadata = audio_write(
            audio_path, "test_004",
            target_format="webm"
        )

        assert metadata['format'] == 'webm'
        assert metadata['sample_rate'] == 48000  # Opus always 48kHz
        assert metadata['bit_depth'] is None
        assert metadata['channels'] == 2

    def test_write_flac_to_wav(self, sample_audio_flac):
        """Test converting FLAC to WAV."""
        audio_path, sample_rate, _ = sample_audio_flac

        raw_bytes, metadata = audio_write(
            audio_path, "test_005",
            target_format="wav",
            target_bit_depth=24
        )

        assert metadata['format'] == 'wav'
        assert metadata['sample_rate'] == sample_rate
        assert metadata['bit_depth'] == 24
        assert metadata['channels'] == 1

    def test_write_change_bit_depth(self, sample_audio_wav):
        """Test that lower source bit depth is preserved (no upsampling waste)."""
        audio_path, sample_rate, _ = sample_audio_wav

        raw_bytes, metadata = audio_write(
            str(audio_path), "test_006",
            target_format="wav",
            target_bit_depth=32
        )

        # Source is 16-bit, requesting 32-bit should keep 16-bit to save storage
        assert metadata['bit_depth'] == 16

    def test_write_unsupported_bit_depth(self, sample_audio_wav):
        """Test unsupported bit depth raises ValueError."""
        audio_path, _, _ = sample_audio_wav

        with pytest.raises(ValueError, match="Unsupported target bit depth"):
            audio_write(
                audio_path, "test_007",
                target_format="wav",
                target_bit_depth=12
            )


class TestWebMOperations:
    """Test WebM-specific operations."""

    def test_write_webm_encoding(self):
        """Test WebM encoding from numpy array."""
        sample_rate = 48000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Generate test audio
        t = np.linspace(0, duration, num_samples)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        audio = np.stack([audio, audio], axis=1)  # Stereo

        raw_bytes = _write_webm(audio, sample_rate)

        assert len(raw_bytes) > 0
        assert raw_bytes[:4] == b'\x1aE\xdf\xa3'  # EBML header

    def test_read_webm_roundtrip(self, sample_audio_webm):
        """Test reading back WebM data."""
        audio_path, sample_rate, _ = sample_audio_webm

        data, sr, channels, frames = _read_webm(str(audio_path))

        assert sr == sample_rate
        assert channels == 2
        assert frames > 0
        assert data.dtype == np.float64

    def test_get_webm_info(self, sample_audio_webm):
        """Test getting WebM file info."""
        audio_path, sample_rate, _ = sample_audio_webm

        info = _get_webm_info(str(audio_path))

        assert info['format'] == 'webm'
        assert info['sample_rate'] == sample_rate
        assert info['channels'] == 2
        assert info['bit_depth'] is None  # Opus is not PCM

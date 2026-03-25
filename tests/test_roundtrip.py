"""Tests for round-trip data integrity - verifying that data is preserved through write-read cycles."""

import io
import pytest
import numpy as np
import soundfile as sf

from omniio.audio.write import audio_write
from omniio.audio.read import audio_read_local
from omniio.text.write import text_write
from omniio.text.read import text_read_local
from omniio.video.write import video_write
from omniio.video.read import video_read_local


class TestAudioRoundtrip:
    """Test audio data preservation through write-read cycles."""

    def test_wav_lossless_roundtrip(self, temp_dir):
        """Test that WAV format preserves audio data exactly."""
        # Generate test signal
        sample_rate = 16000
        duration = 0.5
        num_samples = int(sample_rate * duration)

        # Create a multi-frequency test signal (stereo)
        t = np.linspace(0, duration, num_samples)
        freq1, freq2 = 440.0, 880.0
        left = 0.5 * (np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t))
        right = 0.5 * (np.sin(2 * np.pi * freq2 * t) + 0.3 * np.sin(2 * np.pi * freq1 * t))
        original_audio = np.stack([left, right], axis=1).astype(np.float32)

        # Write original audio to file
        input_path = temp_dir / "original.wav"
        sf.write(input_path, original_audio, sample_rate, subtype='PCM_16')

        # Write to archive format
        raw_bytes, metadata = audio_write(
            str(input_path),
            "test_001",
            target_format="wav",
            target_bit_depth=16
        )

        # Create archive
        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = audio_read_local(archive_path, 0, len(raw_bytes))

        # Verify metadata
        assert result.sample_rate == sample_rate
        assert result.array.shape[1] == 2  # Stereo

        # Verify audio data (allow small error due to 16-bit quantization)
        # 16-bit PCM has quantization error of about 1/32768 ≈ 3e-5
        assert np.allclose(original_audio, result.array, atol=5e-5)

        # Verify RMS difference is very small
        rms_diff = np.sqrt(np.mean((original_audio - result.array) ** 2))
        assert rms_diff < 1e-4

    def test_flac_lossless_roundtrip(self, temp_dir):
        """Test that FLAC format preserves audio data exactly."""
        # Generate test signal (mono)
        sample_rate = 44100
        duration = 0.3
        num_samples = int(sample_rate * duration)

        t = np.linspace(0, duration, num_samples)
        original_audio = 0.7 * np.sin(2 * np.pi * 440 * t).astype(np.float32).reshape(-1, 1)

        # Write original audio to file
        input_path = temp_dir / "original.flac"
        sf.write(input_path, original_audio, sample_rate, format='FLAC')

        # Write to archive format
        raw_bytes, metadata = audio_write(
            str(input_path),
            "test_002",
            target_format="flac"
        )

        # Create archive
        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = audio_read_local(archive_path, 0, len(raw_bytes))

        # FLAC is lossless for integer samples, but float32->16bit conversion has quantization error
        # Quantization step for 16-bit: 1/32767 ≈ 3.05e-5
        assert result.sample_rate == sample_rate
        assert np.allclose(original_audio, result.array, atol=5e-5)  # Allow for 16-bit quantization

        # Verify RMS difference is within quantization noise
        rms_diff = np.sqrt(np.mean((original_audio - result.array) ** 2))
        assert rms_diff < 5e-5

    def test_wav_different_bit_depths(self, temp_dir):
        """Test that different bit depths preserve appropriate precision."""
        sample_rate = 16000
        duration = 0.2
        num_samples = int(sample_rate * duration)

        # Create test signal
        t = np.linspace(0, duration, num_samples)
        original_audio = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32).reshape(-1, 1)

        for bit_depth, atol in [(16, 5e-5), (24, 1e-6), (32, 1e-7)]:
            input_path = temp_dir / f"original_{bit_depth}.wav"
            sf.write(input_path, original_audio, sample_rate,
                    subtype=f'PCM_{bit_depth}')

            # Write and read
            raw_bytes, _ = audio_write(
                str(input_path),
                f"test_{bit_depth}",
                target_format="wav",
                target_bit_depth=bit_depth
            )

            archive_path = temp_dir / f"archive_{bit_depth}.bin"
            with open(archive_path, 'wb') as f:
                f.write(raw_bytes)

            result = audio_read_local(archive_path, 0, len(raw_bytes))

            # Higher bit depth = better precision
            assert np.allclose(original_audio, result.array, atol=atol)

    def test_webm_lossy_acceptable_quality(self, temp_dir):
        """Test that WebM/Opus lossy compression maintains acceptable quality."""
        # Generate test signal
        sample_rate = 48000
        duration = 0.5
        num_samples = int(sample_rate * duration)

        t = np.linspace(0, duration, num_samples)
        freq = 440.0
        original_audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        original_audio = np.stack([original_audio, original_audio], axis=1)  # Stereo

        # Write original as WAV first
        input_path = temp_dir / "original.wav"
        sf.write(input_path, original_audio, sample_rate)

        # Convert to WebM
        raw_bytes, metadata = audio_write(
            str(input_path),
            "test_webm",
            target_format="webm"
        )

        # Create archive
        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = audio_read_local(archive_path, 0, len(raw_bytes))

        # Opus is lossy, so we expect some difference
        # But for simple sine waves, it should be quite good
        # Allow higher tolerance for lossy compression
        assert result.sample_rate == 48000  # Opus always 48kHz

        # Calculate correlation (should be very high)
        correlation = np.corrcoef(original_audio.flatten(), result.array.flatten())[0, 1]
        assert correlation > 0.99  # Very high correlation

        # RMS error should be reasonable
        rms_diff = np.sqrt(np.mean((original_audio - result.array) ** 2))
        assert rms_diff < 0.05  # Allow up to 5% RMS difference for lossy

    def test_time_slicing_preserves_data(self, temp_dir):
        """Test that time-based slicing extracts correct portion."""
        # Generate test signal with distinctive segments
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)

        t = np.linspace(0, duration, num_samples)
        # Different frequencies in different time segments
        audio = np.zeros(num_samples)
        audio[:num_samples//2] = np.sin(2 * np.pi * 440 * t[:num_samples//2])  # First half: 440Hz
        audio[num_samples//2:] = np.sin(2 * np.pi * 880 * t[num_samples//2:])  # Second half: 880Hz
        audio = audio.astype(np.float32).reshape(-1, 1)

        # Write to file
        input_path = temp_dir / "segmented.wav"
        sf.write(input_path, audio, sample_rate)

        # Write to archive
        raw_bytes, _ = audio_write(str(input_path), "test_slice", target_format="wav")

        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read first half only (440Hz segment)
        result_first = audio_read_local(
            archive_path, 0, len(raw_bytes),
            start_time=0.0, end_time=1.0
        )

        # Read second half only (880Hz segment)
        result_second = audio_read_local(
            archive_path, 0, len(raw_bytes),
            start_time=1.0, end_time=2.0
        )

        # Verify we got the right segments
        expected_first = audio[:num_samples//2]
        expected_second = audio[num_samples//2:]

        assert np.allclose(expected_first, result_first.array, atol=1e-4)
        assert np.allclose(expected_second, result_second.array, atol=1e-4)


class TestTextRoundtrip:
    """Test text data preservation through write-read cycles."""

    def test_text_exact_preservation(self, temp_dir):
        """Test that text is preserved exactly through compression."""
        original_text = "This is a test of text preservation.\n" * 20

        # Write compressed
        raw_bytes, metadata = text_write(
            original_text,
            "test_001",
            is_path=False,
            compression_level=3
        )

        # Create archive
        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = text_read_local(archive_path, 0, len(raw_bytes))

        # Must be exactly equal
        assert result.text == original_text

    def test_unicode_text_preservation(self, temp_dir):
        """Test that unicode text is preserved perfectly."""
        original_text = (
            "English text\n"
            "中文文本 (Chinese)\n"
            "日本語テキスト (Japanese)\n"
            "한국어 텍스트 (Korean)\n"
            "Русский текст (Russian)\n"
            "النص العربي (Arabic)\n"
            "Emoji: 🌍🎵✨💾🔬\n"
            "Special chars: \"quotes\", 'apostrophes', <brackets>, {braces}\n"
        ) * 5

        # Write compressed
        raw_bytes, metadata = text_write(
            original_text,
            "test_unicode",
            is_path=False,
            compression_level=5
        )

        # Create archive
        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = text_read_local(archive_path, 0, len(raw_bytes))

        # Must be exactly equal, byte-for-byte
        assert result.text == original_text
        assert len(result.text) == len(original_text)

    def test_empty_text_preservation(self, temp_dir):
        """Test that empty text is handled correctly."""
        original_text = ""

        raw_bytes, _ = text_write(
            original_text,
            "test_empty",
            is_path=False,
            compression_level=3
        )

        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        result = text_read_local(archive_path, 0, len(raw_bytes))

        assert result.text == ""

    def test_large_text_preservation(self, temp_dir):
        """Test that large text files are preserved correctly."""
        # Generate large text (1MB+)
        chunk = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100
        original_text = (chunk + "\n") * 200

        assert len(original_text) > 1_000_000  # Verify it's large

        # Write compressed
        raw_bytes, metadata = text_write(
            original_text,
            "test_large",
            is_path=False,
            compression_level=3
        )

        # Verify compression worked
        assert len(raw_bytes) < len(original_text)

        # Create archive
        archive_path = temp_dir / "archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = text_read_local(archive_path, 0, len(raw_bytes))

        # Must be exactly equal
        assert result.text == original_text

    def test_text_compression_levels(self, temp_dir):
        """Test that different compression levels all preserve data."""
        original_text = "Test text for compression level testing.\n" * 100

        for level in [1, 3, 9, 19]:
            raw_bytes, metadata = text_write(
                original_text,
                f"test_level_{level}",
                is_path=False,
                compression_level=level
            )

            archive_path = temp_dir / f"archive_{level}.bin"
            with open(archive_path, 'wb') as f:
                f.write(raw_bytes)

            result = text_read_local(archive_path, 0, len(raw_bytes))

            # All compression levels must preserve data exactly
            assert result.text == original_text


class TestVideoRoundtrip:
    """Test video data preservation through write-read cycles."""

    def test_video_frames_acceptable_quality(self, temp_dir, sample_video_mp4):
        """Test that video frames maintain acceptable quality through H.264 encoding."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        # Read original video
        with open(video_path, 'rb') as f:
            original_bytes = f.read()

        from omniio.video.read import _decode_video
        original_video = _decode_video(original_bytes)

        # Write to archive format (will re-encode)
        raw_bytes, metadata = video_write(
            str(video_path),
            "test_video",
            target_fps=fps,
            target_height=height,
            target_width=width,
            crf=18  # High quality
        )

        # Create archive
        archive_path = temp_dir / "video_archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = video_read_local(archive_path, 0, len(raw_bytes))

        # Verify metadata
        assert result.fps == fps
        assert result.width == width
        assert result.height == height

        # H.264 is lossy, but high-quality encoding should be very close
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        # Good quality: PSNR > 30 dB, Excellent: PSNR > 40 dB
        mse = np.mean((original_video.video_array.astype(float) -
                       result.video_array.astype(float)) ** 2)

        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            assert psnr > 30.0  # Good quality threshold
        else:
            # Perfect match (unlikely with lossy codec)
            pass

    def test_video_audio_sync_preserved(self, temp_dir, sample_video_mp4):
        """Test that audio/video synchronization is maintained."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        # Read original
        with open(video_path, 'rb') as f:
            original_bytes = f.read()

        from omniio.video.read import _decode_video
        original_video = _decode_video(original_bytes)

        # Write and read back
        raw_bytes, _ = video_write(
            str(video_path),
            "test_sync",
            crf=23
        )

        archive_path = temp_dir / "video_archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        result = video_read_local(archive_path, 0, len(raw_bytes))

        # Calculate durations
        video_duration = result.video_array.shape[0] / result.fps
        audio_duration = result.audio_array.shape[0] / result.sample_rate

        # Audio and video should be in sync (within 100ms tolerance)
        assert abs(video_duration - audio_duration) < 0.1

    def test_video_no_reencoding_when_compatible(self, temp_dir, sample_video_mp4):
        """Test that compatible MP4/H.264 files are not re-encoded."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        # Write without any transformation (should use fast path)
        raw_bytes, metadata = video_write(
            str(video_path),
            "test_noreencode",
            target_fps=None,  # Keep original
            target_height=None,
            target_width=None
        )

        # Read original file
        with open(video_path, 'rb') as f:
            original_bytes = f.read()

        # Should be identical (fast path, no re-encoding)
        assert len(raw_bytes) == len(original_bytes)
        assert metadata['video_codec'] == 'h264'

    def test_video_frame_slicing_preserves_content(self, temp_dir, sample_video_mp4):
        """Test that frame-based slicing extracts correct frames."""
        video_path, fps, width, height, num_frames = sample_video_mp4

        # Write to archive
        raw_bytes, _ = video_write(str(video_path), "test_slice")

        archive_path = temp_dir / "video_archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read full video
        full_video = video_read_local(archive_path, 0, len(raw_bytes))

        # Read first half
        mid_frame = num_frames // 2
        first_half = video_read_local(
            archive_path, 0, len(raw_bytes),
            start_frame=0, end_frame=mid_frame
        )

        # Read second half
        second_half = video_read_local(
            archive_path, 0, len(raw_bytes),
            start_frame=mid_frame, end_frame=num_frames
        )

        # Verify frame counts
        assert abs(first_half.video_array.shape[0] - mid_frame) <= 2
        assert abs(second_half.video_array.shape[0] - (num_frames - mid_frame)) <= 2

        # Together should roughly equal full video
        total_frames = first_half.video_array.shape[0] + second_half.video_array.shape[0]
        assert abs(total_frames - full_video.video_array.shape[0]) <= 3

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


class TestRangedSliceRead:
    """A time slice reads only the bytes it needs, and returns exactly what a
    full read followed by slicing would."""

    @pytest.fixture
    def long_archive(self, temp_dir):
        """A 60 s mono WAV and a 60 s mono FLAC, behind some leading junk."""
        import soundfile as sf

        sr = 16000
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal((60 * sr, 1)) * 0.1).astype(np.float32)
        entries = {}
        archive = temp_dir / "blob_0.bin"
        with open(archive, "wb") as f:
            f.write(b"\0" * 1000)
            for fmt in ("WAV", "FLAC"):
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format=fmt, subtype="PCM_16")
                entries[fmt.lower()] = (f.tell(), len(buf.getvalue()))
                f.write(buf.getvalue())
        return archive, entries

    @pytest.fixture
    def counting(self, monkeypatch):
        """Record ``bytes_read`` of every range reader that is closed."""
        from omniio.modalities.audio import read as read_mod

        seen = []

        class Counting(read_mod._RangeReader):
            def close(self):
                if not self.closed:
                    seen.append(self.bytes_read)
                super().close()

        monkeypatch.setattr(read_mod, "_RangeReader", Counting)
        return seen

    @pytest.mark.parametrize("fmt", ["wav", "flac"])
    @pytest.mark.parametrize(
        "start,end", [(30.25, 31.5), (0.0, 0.1), (59.5, None), (None, 0.5), (59.0, 70.0)]
    )
    def test_matches_full_read(self, long_archive, counting, fmt, start, end):
        archive, entries = long_archive
        offset, size = entries[fmt]
        full = audio_read_local(str(archive), offset, size).array
        got = audio_read_local(str(archive), offset, size, start_time=start, end_time=end)
        lo = 0 if start is None else int(start * 16000)
        hi = len(full) if end is None else int(end * 16000)
        np.testing.assert_array_equal(got.array, full[lo:hi])
        assert got.file_type == fmt
        assert counting and counting[-1] < size

    def test_reads_a_fraction_of_the_entry(self, long_archive, counting):
        archive, entries = long_archive
        offset, size = entries["wav"]  # ~1.9 MB
        audio_read_local(str(archive), offset, size, start_time=30.0, end_time=31.0)
        assert counting[-1] < size // 5

    @pytest.mark.parametrize("fmt", ["wav", "flac"])
    def test_remote(self, long_archive, counting, fmt):
        import socket
        import threading
        import time

        from omniio.audio.read import audio_read_remote
        from omniio.remote import serve

        archive, entries = long_archive
        offset, size = entries[fmt]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        threading.Thread(
            target=serve, args=(port, str(archive.parent), "127.0.0.1"), daemon=True
        ).start()
        time.sleep(0.5)

        url = f"http://127.0.0.1:{port}/{archive.name}"
        full = audio_read_remote(url, offset, size).array
        got = audio_read_remote(url, offset, size, start_time=10.0, end_time=11.0).array
        np.testing.assert_array_equal(got, full[160000:176000])
        assert counting[-1] < size


def test_slice_bounds_round_rather_than_truncate(sample_binary_archive, sample_audio_wav):
    """258.52 * 16000 == 4136319.9999999995: int() would start a sample early."""
    import soundfile as sf

    archive_path, start_offset, file_size = sample_binary_archive
    full, sr = sf.read(sample_audio_wav[0], dtype="float32", always_2d=True)
    start, end = 1.001, 1.5  # 1.001 * 16000 == 16015.999999999998
    assert int(start * sr) != round(start * sr)
    got = audio_read_local(archive_path, start_offset, file_size, start_time=start, end_time=end)
    np.testing.assert_array_equal(got.array, full[round(start * sr) : round(end * sr)])

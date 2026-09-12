"""Tests for MP3 reading and writing."""

import io
import struct

import numpy as np
import pytest
import soundfile as sf

import omniio.audio.read as audio_read_mod
from omniio.audio.read import _detect_format, audio_read_local
from omniio.audio.write import audio_write
from omniio.blob.blob import Blob
from omniio.interface import audio_read


def _signal(sr, seconds, channels=1, seed=0):
    """Broadband content (tones + noise) so the encoder actually uses its bit reservoir,
    which is what makes a naive mid-stream seek decode wrong."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    cols = [0.2 * np.sin(2 * np.pi * (220 + 110 * c) * t * (1 + 0.3 * t))
            + 0.1 * rng.standard_normal(n) for c in range(channels)]
    return np.stack(cols, axis=1)


def _id3_tag() -> bytes:
    """Minimal ID3v2.3 tag holding one TIT2 (title) frame."""
    body = b"\x00" + b"omniio"                       # text encoding + title
    frame = b"TIT2" + struct.pack(">I", len(body)) + b"\x00\x00" + body
    size = len(frame)
    syncsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    return b"ID3\x03\x00\x00" + syncsafe + frame


@pytest.fixture
def sample_audio_mp3(temp_dir):
    """Tagless MP3 as libsndfile writes it (starts on an MPEG frame sync)."""
    sr, x = 24000, _signal(24000, 3.0)
    path = temp_dir / "sample.mp3"
    sf.write(path, x, sr, format="MP3", subtype="MPEG_LAYER_III")
    return path, sr, x.shape[0]


@pytest.fixture
def sample_audio_mp3_id3(temp_dir, sample_audio_mp3):
    """Same stream with an ID3v2 tag in front, as most real-world mp3s (e.g. Emilia)."""
    src, sr, n = sample_audio_mp3
    path = temp_dir / "tagged.mp3"
    path.write_bytes(_id3_tag() + src.read_bytes())
    return path, sr, n


@pytest.fixture
def sample_audio_mp3_lowrate(temp_dir):
    """22.05 kHz stereo at the encoder's lowest bitrate: small frames, so the bit
    reservoir reaches back the most frames — the case a fixed pre-roll gets wrong."""
    sr, x = 22050, _signal(22050, 4.0, channels=2, seed=3)
    path = temp_dir / "lowrate.mp3"
    sf.write(path, x, sr, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=0.99, bitrate_mode="VARIABLE")
    return path, sr, x.shape[0]


def _archive(temp_dir, raw: bytes, pad: int = 137):
    """Write `raw` into a bin at a non-zero offset, like a real blob entry."""
    path = temp_dir / "archive.bin"
    path.write_bytes(b"\xAB" * pad + raw + b"\xCD" * 64)
    return str(path), pad, len(raw)


class TestMP3FormatDetection:

    @pytest.mark.parametrize("header", [b"ID3\x03\x00", b"ID3\x04\x00"])
    def test_detect_id3(self, header):
        assert _detect_format(header + b"\x00" * 11) == "mp3"

    @pytest.mark.parametrize("sync", [b"\xff\xfb", b"\xff\xf3", b"\xff\xe3", b"\xff\xfd"])
    def test_detect_frame_sync(self, sync):
        """MPEG-1 / MPEG-2 / MPEG-2.5 layer III, and MPEG-1 layer II."""
        assert _detect_format(sync + b"\x90\x00" + b"\x00" * 12) == "mp3"

    @pytest.mark.parametrize("sync", [b"\xff\xf1", b"\xff\xf9"])
    def test_adts_aac_is_not_mp3(self, sync):
        """AAC ADTS shares the sync word but has layer bits 00."""
        with pytest.raises(ValueError, match="Unknown audio format"):
            _detect_format(sync + b"\x00" * 14)

    def test_detect_libsndfile_output(self, sample_audio_mp3):
        path, _, _ = sample_audio_mp3
        assert _detect_format(path.read_bytes()[:16]) == "mp3"


class TestMP3Write:

    def test_mp3_passthrough_is_byte_copy(self, sample_audio_mp3_id3):
        path, sr, n = sample_audio_mp3_id3
        raw, meta = audio_write(path, "a")
        assert raw == path.read_bytes()
        assert meta["format"] == "mp3"
        assert meta["bit_depth"] is None
        assert meta["sample_rate"] == sr
        assert meta["samples"] == n

    def test_explicit_mp3_target_ignores_bit_depth(self, sample_audio_mp3):
        path, _, _ = sample_audio_mp3
        raw, meta = audio_write(path, "a", target_format="mp3", target_bit_depth=24)
        assert raw == path.read_bytes()
        assert meta["bit_depth"] is None

    def test_mp3_to_flac_defaults_to_16_bit(self, sample_audio_mp3):
        path, sr, n = sample_audio_mp3
        raw, meta = audio_write(path, "a", target_format="flac")
        assert meta["format"] == "flac" and meta["bit_depth"] == 16
        info = sf.info(io.BytesIO(raw))
        assert info.subtype == "PCM_16" and info.frames == n and info.samplerate == sr

    @pytest.mark.parametrize("sr", [8000, 16000, 22050, 24000, 44100, 48000])
    @pytest.mark.parametrize("channels", [1, 2])
    def test_array_to_mp3_is_gapless(self, sr, channels):
        x = _signal(sr, 1.37, channels)
        raw, meta = audio_write((x, sr), "a", target_format="mp3")
        assert meta["format"] == "mp3" and meta["bit_depth"] is None
        assert meta["samples"] == x.shape[0] and meta["channels"] == channels
        y, sr_out = sf.read(io.BytesIO(raw), always_2d=True)
        assert sr_out == sr and y.shape == x.shape

    def test_wav_to_mp3(self, sample_audio_wav):
        path, sr, n = sample_audio_wav
        raw, meta = audio_write(path, "a", target_format="mp3")
        assert _detect_format(raw[:16]) == "mp3"
        assert meta["samples"] == n and meta["sample_rate"] == sr

    def test_unsupported_sample_rate_raises(self):
        with pytest.raises(ValueError, match="MP3 does not support sample rate 96000"):
            audio_write((_signal(96000, 0.2), 96000), "a", target_format="mp3")

    def test_compression_level_controls_size(self):
        x = _signal(24000, 2.0)
        big, _ = audio_write((x, 24000), "a", target_format="mp3",
                             compression_level=0.0, bitrate_mode="CONSTANT")
        small, _ = audio_write((x, 24000), "a", target_format="mp3",
                               compression_level=0.9, bitrate_mode="CONSTANT")
        assert len(small) < len(big)

    @pytest.mark.parametrize("level", [1.0, -0.1])
    def test_compression_level_out_of_range_raises(self, level):
        with pytest.raises(ValueError, match="compression_level must be in"):
            audio_write((_signal(24000, 0.2), 24000), "a", target_format="mp3",
                        compression_level=level)


class TestMP3Read:

    @pytest.mark.parametrize("fixture", ["sample_audio_mp3", "sample_audio_mp3_id3"])
    def test_full_read_from_archive(self, temp_dir, fixture, request):
        path, sr, n = request.getfixturevalue(fixture)
        archive, off, size = _archive(temp_dir, path.read_bytes())
        res = audio_read_local(archive, off, size)
        ref, _ = sf.read(path, dtype="float32", always_2d=True)
        assert res.file_type == "mp3" and res.sample_rate == sr
        assert res.array.shape == (n, 1) and res.array.dtype == np.float32
        np.testing.assert_array_equal(res.array, ref)

    def _bad_windows(self, archive, off, size, full, sr, n_windows=60):
        """Windows that are NOT bit-identical to the same slice of a full decode."""
        rng = np.random.default_rng(1)
        bad = 0
        for _ in range(n_windows):
            s = int(rng.integers(0, full.shape[0] - sr // 2))
            e = s + sr // 2
            win = audio_read_local(archive, off, size, start_time=s / sr, end_time=e / sr).array
            if not np.array_equal(win, full[int(s / sr * sr):int(e / sr * sr)]):
                bad += 1
        return bad

    @pytest.mark.parametrize("fixture", ["sample_audio_mp3_id3", "sample_audio_mp3_lowrate"])
    def test_windowed_read_is_exact(self, temp_dir, fixture, request):
        """Every window is bit-identical to a full decode's slice."""
        path, sr, _ = request.getfixturevalue(fixture)
        archive, off, size = _archive(temp_dir, path.read_bytes())
        full, _ = sf.read(path, dtype="float32", always_2d=True)
        assert self._bad_windows(archive, off, size, full, sr) == 0

    def test_fixture_defeats_libsndfile_seek(self, sample_audio_mp3_lowrate):
        """Gives the exactness test teeth: on this stream a bare libsndfile seek (what
        reintroducing seeking would do) returns wrong samples."""
        path, sr, _ = sample_audio_mp3_lowrate
        full, _ = sf.read(path, dtype="float32", always_2d=True)
        rng = np.random.default_rng(1)
        wrong = 0
        for _ in range(60):
            s = int(rng.integers(0, full.shape[0] - sr // 2))
            w, _ = sf.read(path, start=s, stop=s + sr // 2, dtype="float32", always_2d=True)
            wrong += not np.array_equal(w, full[s:s + sr // 2])
        if wrong == 0:
            pytest.skip("libsndfile's mp3 seek is exact on this stream now; "
                        "decode-from-start could be revisited")

    def test_window_at_start_and_end(self, temp_dir, sample_audio_mp3):
        path, sr, n = sample_audio_mp3
        archive, off, size = _archive(temp_dir, path.read_bytes())
        full, _ = sf.read(path, dtype="float32", always_2d=True)
        head = audio_read_local(archive, off, size, start_time=0.0, end_time=0.25).array
        np.testing.assert_array_equal(head, full[:sr // 4])
        tail = audio_read_local(archive, off, size, start_time=2.5).array
        np.testing.assert_array_equal(tail, full[int(2.5 * sr):])

    def test_lazy_range_path(self, temp_dir, sample_audio_mp3_id3, monkeypatch):
        """Large entries read windows through _RangeFile instead of one pread."""
        monkeypatch.setattr(audio_read_mod, "_LAZY_MIN_BYTES", 0)
        path, sr, _ = sample_audio_mp3_id3
        archive, off, size = _archive(temp_dir, path.read_bytes())
        full, _ = sf.read(path, dtype="float32", always_2d=True)
        assert self._bad_windows(archive, off, size, full, sr, n_windows=20) == 0


class TestMP3Blob:

    def test_blob_roundtrip(self, temp_dir, sample_audio_mp3, sample_audio_mp3_id3):
        items = [sample_audio_mp3[0], sample_audio_mp3_id3[0]]
        blob = Blob(str(temp_dir / "arch"), modality="audio")
        blob.append(items=[str(p) for p in items], ids=["plain", "tagged"], progress=False)
        meta = {r["id"]: r for r in blob.get_metadata().to_pylist()}
        for item_id, src in zip(["plain", "tagged"], items):
            r = meta[item_id]
            assert r["format"] == "mp3" and r["bit_depth"] is None
            res = audio_read(r["path"], r["start_byte"], r["end_byte"] - r["start_byte"])
            ref, _ = sf.read(src, dtype="float32", always_2d=True)
            assert res.array.shape[0] == r["samples"]
            np.testing.assert_array_equal(res.array, ref)

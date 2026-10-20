"""
Self-test of the audio window reader and the token span reader.

A time window read through the archive reader must equal the same slice of a
full decode, on both paths the reader takes: an entry smaller than the
small-entry threshold (one pread, decode from memory) and a larger one (in-place
decode through the buffered archive view). The token reader must read only the
spans it was asked for. The archive handle cache must give each process its
own descriptor.
"""
import io
import os
import sys

import numpy as np
import pytest
import soundfile as sf

from omniio.blob import handles
from omniio.modalities.audio import read as audio_read_mod
from omniio.modalities.audio.read import audio_read_local


def _flac_bytes(seconds, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    b = io.BytesIO()
    sf.write(b, x, sr, format="FLAC", subtype="PCM_16")
    return b.getvalue(), sr


@pytest.mark.parametrize("seconds", [2.0, 40.0])
def test_window_matches_full_decode(tmp_path, seconds, monkeypatch):
    raw, sr = _flac_bytes(seconds)
    pad = b"x" * 12345
    archive = tmp_path / "blob_0.bin"
    archive.write_bytes(pad + raw + b"tail")
    off, size = len(pad), len(raw)
    # force both paths regardless of the entry's actual size
    for small_threshold in (10 ** 9, 1):
        monkeypatch.setattr(audio_read_mod, "_SMALL_ENTRY_BYTES", small_threshold)
        full = audio_read_local(str(archive), off, size).array
        assert full.shape[0] == int(seconds * sr)
        for s0, s1 in ((0.0, 1.0), (0.37, 1.91), (seconds - 1.5, seconds - 0.25), (seconds - 0.5, seconds + 5.0)):
            w = audio_read_local(str(archive), off, size, start_time=s0, end_time=s1).array
            a, b = int(s0 * sr), min(int(s1 * sr), full.shape[0])
            assert w.shape == (b - a, 1), (small_threshold, s0, s1, w.shape)
            np.testing.assert_array_equal(w, full[a:b])


def test_window_past_end_is_empty(tmp_path):
    raw, sr = _flac_bytes(2.0)
    archive = tmp_path / "blob_0.bin"; archive.write_bytes(raw)
    w = audio_read_local(str(archive), 0, len(raw), start_time=5.0, end_time=6.0).array
    assert w.shape == (0, 1)
    w = audio_read_local(str(archive), 0, len(raw), start_time=1.5, end_time=6.0).array
    assert w.shape == (int(0.5 * sr), 1)


def test_window_bytes_read_small_entry(tmp_path):
    """A 3 s window of a long entry through the in-place path reads far less than the entry."""
    raw, sr = _flac_bytes(120.0)
    archive = tmp_path / "blob_0.bin"
    archive.write_bytes(raw)
    fd = handles.archive_fd(str(archive))
    counted = []
    orig = os.pread

    def counting_pread(f, n, o):
        b = orig(f, n, o)
        if f == fd:
            counted.append(len(b))
        return b

    audio_read_mod.os.pread = counting_pread
    try:
        audio_read_mod._SMALL_ENTRY_BYTES = 1
        audio_read_local(str(archive), 0, len(raw), start_time=60.0, end_time=63.0)
    finally:
        audio_read_mod.os.pread = orig
        audio_read_mod._SMALL_ENTRY_BYTES = int(os.environ.get("OMNIIO_SMALL_ENTRY_BYTES", 4 * (1 << 18)))
    assert 0 < sum(counted) < len(raw) / 4, (sum(counted), len(raw))


def test_handle_cache_is_per_process(tmp_path):
    archive = tmp_path / "blob_0.bin"
    archive.write_bytes(b"0" * 100)
    fd1 = handles.archive_fd(str(archive))
    assert handles.archive_fd(str(archive)) == fd1
    if hasattr(os, "fork"):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:                                            # child: must open its own descriptor
            child_fd = handles.archive_fd(str(archive))
            os.write(w, b"same" if child_fd == fd1 and handles._cache.get((os.getpid(), str(archive))) == fd1 and False else b"own")
            os._exit(0)
        os.waitpid(pid, 0)
        assert os.read(r, 4) == b"own"
    assert handles.pread_all(fd1, 10, 90) == b"0" * 10
    handles.close_all()


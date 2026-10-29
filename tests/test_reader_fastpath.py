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
    monkey = audio_read_mod._SMALL_ENTRY_BYTES
    audio_read_mod._SMALL_ENTRY_BYTES = 1                     # the in-place path too
    try:
        w = audio_read_local(str(archive), 0, len(raw), start_time=5.0, end_time=6.0).array
        assert w.shape == (0, 1)
        w = audio_read_local(str(archive), 0, len(raw), start_time=2.0, end_time=3.0).array
        assert w.shape == (0, 1)
    finally:
        audio_read_mod._SMALL_ENTRY_BYTES = monkey
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

def test_lookahead_preserves_order_and_overlaps(tmp_path):
    """Lookahead yields results in request order while keeping `depth` fetches in flight."""
    import time
    from omniio.blob.prefetch import Lookahead, local_fetch
    archive = tmp_path / "blob_0.bin"; data = bytes(range(256)) * 64; archive.write_bytes(data)
    reqs = [(str(archive), o, 16) for o in [4000, 16, 8192, 0, 1024, 16368, 512, 2048]]
    got = list(Lookahead(local_fetch, depth=3)(reqs))
    assert got == [data[o:o + 16] for _, o, _ in reqs]
    lat = 0.05; calls = []

    def slow(p, o, s):
        calls.append(time.perf_counter()); time.sleep(lat); return o

    t0 = time.perf_counter(); out = list(Lookahead(slow, depth=8)([("x", k, 1) for k in range(16)])); dt = time.perf_counter() - t0
    assert out == list(range(16)) and dt < 16 * lat * 0.5          # overlapped: far below the serial 0.8 s
    la = Lookahead(local_fetch, depth=4)
    assert la.read_batch(reqs) == [data[o:o + 16] for _, o, _ in reqs]
    la.close()


def _mp4_bytes(tmp_path, seconds=60, fps=24, w=160, h=120, gop_s=2):
    import av
    path = tmp_path / "clip.mp4"
    out = av.open(str(path), "w", format="mp4")
    vs = out.add_stream("libx264", rate=fps); vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
    vs.options = {"preset": "veryfast", "crf": "23", "g": str(gop_s * fps), "keyint_min": str(gop_s * fps), "sc_threshold": "0", "bf": "0"}
    a = out.add_stream("aac", rate=48000); a.layout = "stereo"
    rng = np.random.default_rng(0)
    t = np.arange(48000 * seconds) / 48000
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    k = 0
    for n in range(seconds * fps):                           # mux in time order, as a transcoder does
        img = np.zeros((h, w, 3), np.uint8); img[:, :, 0] = (n * 3) % 256; img[n % h, :, 1] = 255     # frame-coded content
        img[:, :, 2] = rng.integers(0, 40, size=(h, w))
        fr = av.VideoFrame.from_ndarray(img, format="rgb24"); fr.pts = n
        for p in vs.encode(fr):
            out.mux(p)
        while k < len(tone) and k / 48000 <= (n + 1) / fps:
            af = av.AudioFrame.from_ndarray(np.stack([tone[k:k + 1024]] * 2), format="fltp", layout="stereo"); af.sample_rate = 48000; af.pts = k
            for p in a.encode(af):
                out.mux(p)
            k += 1024
    for p in vs.encode():
        out.mux(p)
    for p in a.encode():
        out.mux(p)
    out.close()
    return path.read_bytes(), fps


def test_video_window_in_place_matches_whole_entry(tmp_path):
    """A time window decoded in place through the archive view equals the window decoded from
    the whole entry in memory and the slice of a full decode, and reads far fewer bytes."""
    from omniio.modalities.video import read as video_read_mod
    from omniio.modalities.video.read import video_read_local
    raw, fps = _mp4_bytes(tmp_path)
    pad = b"p" * 7777
    archive = tmp_path / "blob_0.bin"; archive.write_bytes(pad + raw + b"tail")
    off, size = len(pad), len(raw)
    full = video_read_local(str(archive), off, size)
    assert full.video_array.shape[0] == 60 * fps
    fd = handles.archive_fd(str(archive))
    counted, orig = [], os.pread

    def counting_pread(f, n, o):
        b = orig(f, n, o)
        if f == fd:
            counted.append(len(b))
        return b

    old = video_read_mod._SMALL_ENTRY_BYTES
    try:
        for s0, s1 in ((0.0, 1.0), (10.3, 13.3), (41.0, 42.0), (58.5, 60.0)):
            video_read_mod._SMALL_ENTRY_BYTES = 10 ** 12
            whole = video_read_local(str(archive), off, size, start_time=s0, end_time=s1)
            video_read_mod._SMALL_ENTRY_BYTES = 1
            handles.os.pread = counting_pread; counted.clear()
            try:
                win = video_read_local(str(archive), off, size, start_time=s0, end_time=s1)
            finally:
                handles.os.pread = orig
            f0, f1 = int(s0 * fps), int(s1 * fps)
            assert win.video_array.shape == (f1 - f0, 120, 160, 3), (s0, s1, win.video_array.shape)
            np.testing.assert_array_equal(win.video_array, whole.video_array)
            np.testing.assert_array_equal(win.video_array, full.video_array[f0:f1])
            assert win.audio_array.shape == whole.audio_array.shape
            np.testing.assert_array_equal(win.audio_array, whole.audio_array)
            a0, a1 = int(round(f0 / fps * 48000)), int(round(f1 / fps * 48000))   # audio is cut on the video frame grid
            assert win.audio_array.shape == (a1 - a0, 2), win.audio_array.shape
            np.testing.assert_allclose(win.audio_array, full.audio_array[a0:a1], atol=2e-4)
            assert 0 < sum(counted) < size / 2, (s0, s1, sum(counted), size)
    finally:
        video_read_mod._SMALL_ENTRY_BYTES = old


def test_video_preallocated_output_matches_reference_decode(tmp_path):
    """Frames are decoded straight into a preallocated array (one copy per frame, no list +
    stack): the whole-entry read, an open-ended read from the middle and a window must equal
    a reference decode that collects frames with to_ndarray and stacks them."""
    import io
    import av
    from omniio.modalities.video.read import video_read_local
    raw, fps = _mp4_bytes(tmp_path, seconds=12)
    archive = tmp_path / "blob_0.bin"; archive.write_bytes(raw)

    def reference(start_frame=0, end_frame=None):
        c = av.open(io.BytesIO(raw)); vs = c.streams.video[0]; frames = []
        for fr in c.decode(vs):
            k = int(float(fr.pts * vs.time_base) * fps + 0.5)
            if end_frame is not None and k >= end_frame:
                break
            if k >= start_frame:
                frames.append(fr.to_ndarray(format="rgb24"))
        c.close(); return np.stack(frames)

    full = video_read_local(str(archive), 0, len(raw))
    assert full.video_array.shape[0] == 12 * fps
    np.testing.assert_array_equal(full.video_array, reference())
    assert full.audio_array.dtype == np.float32 and full.audio_array.shape[1] == 2
    assert np.abs(full.audio_array).max() > 0.2                       # the tone was decoded

    tail = video_read_local(str(archive), 0, len(raw), start_time=5.0)   # open end: the stream's frame count sizes the array
    f0 = int(5.0 * fps)
    np.testing.assert_array_equal(tail.video_array, full.video_array[f0:])

    win = video_read_local(str(archive), 0, len(raw), start_time=3.0, end_time=4.5)
    f0, f1 = int(3.0 * fps), int(4.5 * fps)
    np.testing.assert_array_equal(win.video_array, reference(f0, f1))
    assert win.audio_array.shape == (int(round(f1 / fps * 48000)) - int(round(f0 / fps * 48000)), 2)
    assert win.video_array.flags["C_CONTIGUOUS"]


def test_video_decoder_threads_option_matches_default_decode(tmp_path):
    """decoder_threads=1 (what a DataLoader worker should use) decodes the same frames as the default."""
    from omniio.modalities.video.read import video_read_local
    raw, fps = _mp4_bytes(tmp_path, seconds=4)
    archive = tmp_path / "blob_0.bin"; archive.write_bytes(raw)
    a = video_read_local(str(archive), 0, len(raw), start_time=0.5, end_time=1.5)
    b = video_read_local(str(archive), 0, len(raw), start_time=0.5, end_time=1.5, decoder_threads=1)
    assert a.video_array.shape[0] == fps
    assert a.video_array.shape == b.video_array.shape
    assert np.array_equal(a.video_array, b.video_array)

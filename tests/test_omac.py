"""OMAC entries through omniio: Blob writing, length_mismatch, indexed reads.

Skips unless the ``omac`` extension (rust/omac) is installed.
"""
import http.server
import threading

import numpy as np
import pytest
import soundfile as sf

omac = pytest.importorskip("omac")

from omniio.audio.omac import omac_write  # noqa: E402
from omniio.audio.read import audio_read_remote  # noqa: E402
from omniio.audio.write import audio_write  # noqa: E402
from omniio.blob.blob import Blob  # noqa: E402
from omniio.interface import audio_read  # noqa: E402

SR = 16000


def _speechlike(n, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n)) * 40
    x = x - np.convolve(x, np.ones(201) / 201, "same")
    return np.clip(x, -32768, 32767).astype(np.int16)


@pytest.fixture
def mono_wavs(tmp_path):
    """Four 16-bit mono WAVs of 3 s and one of 2.5 s, like an array plus a short mic."""
    paths = []
    for c in range(4):
        p = tmp_path / f"U01.CH{c + 1}.wav"
        sf.write(p, _speechlike(3 * SR, c), SR, subtype="PCM_16")
        paths.append(str(p))
    short = tmp_path / "U02.CH1.wav"
    sf.write(short, _speechlike(int(2.5 * SR), 9), SR, subtype="PCM_16")
    return paths, str(short)


def _stack(paths):
    return np.stack([sf.read(p, dtype="int16")[0] for p in paths], 1)


def _row(blob, i=0):
    return blob.get_metadata().to_pylist()[i]


def _read(row, **kw):
    return audio_read(row["path"], row["start_byte"], row["end_byte"] - row["start_byte"], **kw)


def test_blob_roundtrip_window_and_channels(tmp_path, mono_wavs):
    paths, _ = mono_wavs
    blob = Blob(archive_dir=str(tmp_path / "arch"), modality="audio")
    blob.append(items=[paths], ids=["U01"], num_workers=0, target_format="omac", progress=False)
    row = _row(blob)
    assert row["format"] == "omac" and row["channels"] == 4 and row["samples"] == 3 * SR
    assert row["source_frames"] == [3 * SR] * 4
    ref = _stack(paths).astype(np.float32) / 32768

    full = _read(row)
    assert full.file_type == "omac" and full.sample_rate == SR
    assert np.array_equal(full.array, ref)

    part = _read(row, start_time=0.7, end_time=2.25, channels=[3, 1])
    assert np.array_equal(part.array, ref[round(0.7 * SR):round(2.25 * SR)][:, [3, 1]])
    assert part.valid_frames == [3 * SR, 3 * SR]


def test_length_mismatch(mono_wavs):
    paths, short = mono_wavs
    srcs = paths + [short]
    with pytest.raises(ValueError) as e:
        omac_write(srcs, "x")
    msg = str(e.value)
    assert "length_mismatch" in msg and "U02.CH1.wav: 40,000 frames (2.50 s)" in msg
    assert "U01.CH1.wav: 48,000 frames (3.00 s)" in msg

    raw, meta = omac_write(srcs, "x", length_mismatch="truncate")
    assert meta["samples"] == 40000 and meta["source_frames"] == [48000] * 4 + [40000]
    y, h = omac.decode(raw, dtype="int16")
    assert np.array_equal(y, np.concatenate([_stack(paths)[:40000], _stack([short])], 1))
    assert not h["padded"]

    raw, meta = omac_write(srcs, "x", length_mismatch="pad")
    y, h = omac.decode(raw, dtype="int16")
    assert y.shape == (48000, 5) and h["padded"]
    assert h["valid_frames"] == [48000] * 4 + [40000]
    assert np.array_equal(y[:40000, 4], _stack([short])[:, 0]) and not y[40000:, 4].any()


def test_mixed_bit_depths_and_arrays(tmp_path):
    x16 = _speechlike(5000, 1)
    x24 = (_speechlike(5000, 2).astype(np.int32) << 8) + 77
    p16, p24 = tmp_path / "a.flac", tmp_path / "b.wav"
    sf.write(p16, x16, SR, subtype="PCM_16")
    sf.write(p24, x24.astype(np.int32) << 8, SR, subtype="PCM_24")  # soundfile left-justifies
    raw, meta = omac_write([str(p16), str(p24)], "x")
    assert meta["bit_depth"] == 24
    y, _ = omac.decode(raw, dtype="int32")
    assert np.array_equal(y[:, 0], x16.astype(np.int32) << 8)
    assert np.array_equal(y[:, 1], x24)

    f = np.sin(np.arange(4000) / 30.0)[:, None] * [0.5, -0.25]
    raw, meta = audio_write((f, 8000), "arr", target_format="omac")
    y, _ = omac.decode(raw)
    assert meta["channels"] == 2 and np.abs(y - f).max() <= 0.5 / 32768

    with pytest.raises(ValueError, match="sample rates"):
        omac_write([(f, 8000), (f, 16000)], "x")
    with pytest.raises(ValueError, match="needs target_format='omac'"):
        audio_write([str(p16)], "x", target_format="flac")


def test_more_than_eight_channels(tmp_path):
    x = np.stack([_speechlike(20000, s) for s in range(24)], 1)
    raw, meta = audio_write((x, SR), "kinects", target_format="omac")
    assert meta["channels"] == 24
    y, _ = omac.decode(raw, dtype="int16")
    assert np.array_equal(y, x)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves byte ranges and counts requests."""

    requests = []

    def do_GET(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        lo, hi = 0, len(data) - 1
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            lo, hi = int(a), min(int(b), len(data) - 1)
        type(self).requests.append((lo, hi))
        body = data[lo:hi + 1]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    handler = lambda *a, **k: _RangeHandler(*a, directory=str(root), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _RangeHandler.requests = []
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_remote_reads_only_the_window(server, mono_wavs):
    root, url = server
    paths, _ = mono_wavs
    x = np.concatenate([_stack(paths)] * 20)  # 60 s, 4 channels
    raw, _ = audio_write((x, SR), "long", target_format="omac")
    pad = b"\0" * 1234  # the entry sits inside a larger bin
    (root / "blob_0.bin").write_bytes(pad + raw + pad)
    entry = (f"{url}/blob_0.bin", len(pad), len(raw))

    r = audio_read_remote(*entry, start_time=31.0, end_time=33.0, channels=[2])
    assert np.array_equal(r.array[:, 0], x[31 * SR:33 * SR, 2].astype(np.float32) / 32768)
    cold = list(_RangeHandler.requests)
    assert len(cold) == 2  # header + index, then the window's blocks
    # Cold: the first HEAD_FETCH bytes (header and index), then ~2 s of blocks.
    from omniio.audio.omac import HEAD_FETCH
    assert sum(hi - lo + 1 for lo, hi in cold) < HEAD_FETCH + len(raw) / 20

    _RangeHandler.requests = []
    r = audio_read_remote(*entry, start_time=50.0, end_time=50.5)
    assert np.array_equal(r.array, x[50 * SR:round(50.5 * SR)].astype(np.float32) / 32768)
    assert len(_RangeHandler.requests) == 1  # header and index cached
    lo, hi = _RangeHandler.requests[0]
    assert hi - lo + 1 < len(raw) / 20  # just the window's blocks


def test_remote_long_index(server):
    """An index larger than the first read costs one more request when cold."""
    root, url = server
    x = _speechlike(70_000 * 16, 3)  # 70k blocks of 16 frames: a 560 KB index
    raw, meta = audio_write((x, SR), "long", target_format="omac", omac_block_size=16)
    (root / "blob_0.bin").write_bytes(raw)
    entry = (f"{url}/blob_0.bin", 0, len(raw))
    r = audio_read_remote(*entry, start_time=60.0, end_time=61.0)
    assert np.array_equal(r.array[:, 0], x[60 * SR:61 * SR].astype(np.float32) / 32768)
    assert len(_RangeHandler.requests) == 3  # head, rest of the index, data
    _RangeHandler.requests = []
    audio_read_remote(*entry, start_time=10.0, end_time=10.1)
    assert len(_RangeHandler.requests) == 1


def test_local_reads_are_cached_and_small(tmp_path, monkeypatch):
    x = np.stack([_speechlike(SR * 30, s) for s in range(3)], 1)
    raw, _ = audio_write((x, SR), "a", target_format="omac")
    p = tmp_path / "blob_0.bin"
    p.write_bytes(raw)
    seen = []
    real_open = open

    class Counting:
        def __init__(self, f):
            self.f = f
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.f.close()
        def read(self, n=-1):
            seen.append(n)
            return self.f.read(n)
        def __getattr__(self, name):
            return getattr(self.f, name)

    import omniio.modalities.audio.read as rd
    monkeypatch.setattr(rd, "open", lambda *a, **k: Counting(real_open(*a, **k)), raising=False)
    audio_read(str(p), 0, len(raw), start_time=1.0, end_time=2.0)
    seen.clear()
    r = audio_read(str(p), 0, len(raw), start_time=20.0, end_time=22.0, channels=[1])
    assert np.array_equal(r.array[:, 0], x[20 * SR:22 * SR, 1].astype(np.float32) / 32768)
    assert sum(seen) < len(raw) / 10  # magic sniff + the window's blocks only


def test_server_without_range_support(server, monkeypatch):
    """A 200 with the whole file must not be taken for the requested range."""
    root, url = server
    x = np.stack([_speechlike(SR * 5, s) for s in range(2)], 1)
    raw, _ = audio_write((x, SR), "a", target_format="omac")
    pad = b"\0" * 777
    (root / "blob_0.bin").write_bytes(pad + raw)
    monkeypatch.setattr(_RangeHandler, "do_GET", _no_range_get)
    r = audio_read_remote(f"{url}/blob_0.bin", len(pad), len(raw), start_time=1.0, end_time=2.0)
    assert np.array_equal(r.array, x[SR:2 * SR].astype(np.float32) / 32768)


def _no_range_get(self):
    data = open(self.translate_path(self.path), "rb").read()
    self.send_response(200)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def test_remote_non_omac_unchanged(server, tmp_path):
    root, url = server
    x = (np.random.default_rng(0).standard_normal((SR * 40, 2)) * 3000).astype(np.int16)
    src = tmp_path / "big.flac"
    sf.write(src, x, SR, subtype="PCM_16")
    raw, _ = audio_write(str(src), "big")
    small, _ = audio_write((x[:1000].astype(np.float32) / 32768, SR), "small")
    (root / "blob_0.bin").write_bytes(raw + small)
    assert len(raw) > (1 << 20)

    r = audio_read_remote(f"{url}/blob_0.bin", 0, len(raw), channels=[1])
    assert r.file_type == "flac" and np.array_equal(r.array[:, 0], x[:, 1].astype(np.float32) / 32768)
    assert len(_RangeHandler.requests) == 2  # large entry: first chunk, then the rest

    _RangeHandler.requests = []
    r = audio_read_remote(f"{url}/blob_0.bin", len(raw), len(small))
    assert r.array.shape == (1000, 2) and len(_RangeHandler.requests) == 1  # small: one request


def test_audio_read_bytes(tmp_path):
    """Bytes the caller read decode like an archive read, window and channels included."""
    from omniio.audio.read import audio_read_bytes

    x = np.stack([_speechlike(SR * 4, s) for s in range(3)], 1)
    raw, _ = audio_write((x, SR), "a", target_format="omac")
    r = audio_read_bytes(raw, 1.0, 2.5, channels=[2, 0])
    assert r.file_type == "omac"
    assert np.array_equal(r.array, x[SR:round(2.5 * SR)][:, [2, 0]].astype(np.float32) / 32768)

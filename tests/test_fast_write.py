"""fast_mode writes and the modality-agnostic base modality (omniio.modalities.base)."""
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from omniio.blob.blob import Blob
from omniio.blob.write import modality_writer, writer_for
from omniio.definitions import BytesRead
from omniio.interface import audio_read, bytes_read
from omniio.modalities.base import INDEX_COLUMNS, MODALITIES, Modality, get_modality, item_bytes


def _entries(blob):
    t = pq.read_table(blob.metadata_file)
    rows = t.to_pylist()
    return {r["id"]: (str(blob._get_bin_file_path(r["bin_index"])), r["start_byte"], r["end_byte"] - r["start_byte"], r) for r in rows}


def _wav(path, seconds=0.5, sr=16000, seed=0):
    x = np.random.default_rng(seed).uniform(-0.5, 0.5, int(seconds * sr)).astype(np.float32)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path.read_bytes()


def test_registry_and_generic_modality():
    assert set(MODALITIES) == set(modality_writer) == {"audio", "text", "video", "image", "midi", "discrete", "seq", "track", "cell"}
    for name, m in MODALITIES.items():
        assert m.name == name and type(m) is not Modality
    g = get_modality("npz")
    assert type(g) is Modality and g.name == "npz"
    with pytest.raises(ValueError):
        get_modality("")
    # the base writer takes no options; the raw copy accepts bytes, paths and binary file objects
    assert item_bytes(b"ab") == b"ab" and item_bytes(memoryview(b"ab")) == b"ab" and item_bytes(io.BytesIO(b"ab")) == b"ab"
    with pytest.raises(TypeError):
        g.write(b"x", "x", target_format="flac")
    with pytest.raises(TypeError):
        item_bytes(io.StringIO("text"))
    with pytest.raises(TypeError):
        item_bytes(3.5)


def test_writer_for_resolves_fast_and_normal(tmp_path):
    raw = _wav(tmp_path / "a.wav")
    assert writer_for("audio", fast_mode=True)(str(tmp_path / "a.wav"), "a") == (raw, {})
    b, meta = writer_for("audio")(str(tmp_path / "a.wav"), "a", target_format="flac")
    assert b[:4] == b"fLaC" and meta["format"] == "flac"
    assert writer_for("anything", fast_mode=False)(b"xyz", "x") == (b"xyz", {})


@pytest.mark.parametrize("num_workers", [0, 2])
def test_fast_mode_stores_bytes_verbatim_with_index_only(tmp_path, num_workers):
    srcs = {f"w{i}": _wav(tmp_path / f"w{i}.wav", seconds=0.2 + 0.1 * i, seed=i) for i in range(4)}
    blob = Blob(archive_dir=str(tmp_path / "arch"), modality="audio")
    items = [str(tmp_path / f"{k}.wav") for k in srcs]
    items[1] = srcs["w1"]                                            # raw bytes
    if num_workers == 0:
        items[2] = open(tmp_path / "w2.wav", "rb")                   # binary file object (not picklable: single process only)
    blob.append(items=items, ids=list(srcs), num_workers=num_workers, fast_mode=True, progress=False)
    if num_workers == 0:
        items[2].close()
    ent = _entries(blob)
    assert set(ent) == set(srcs)
    for k, raw in srcs.items():
        path, off, size, row = ent[k]
        assert set(row) == set(INDEX_COLUMNS), sorted(row)           # no modality metadata
        r = bytes_read(path, off, size)
        assert isinstance(r, BytesRead) and r.data == raw
        assert audio_read(path, off, size).array.shape[0] == len(raw[44:]) // 2   # still a WAV: the audio reader decodes it


def test_fast_mode_rejects_write_options(tmp_path):
    blob = Blob(archive_dir=str(tmp_path / "arch"), modality="audio")
    with pytest.raises(ValueError):
        blob.append(items=[b"x"], ids=["x"], fast_mode=True, target_format="flac", progress=False)


def test_mixed_fast_and_transcoding_appends_share_one_archive(tmp_path):
    raw = _wav(tmp_path / "a.wav")
    blob = Blob(archive_dir=str(tmp_path / "arch"), modality="audio")
    blob.append(items=[str(tmp_path / "a.wav")], ids=["flac"], target_format="flac", progress=False)
    blob.append(items=[str(tmp_path / "a.wav")], ids=["raw"], fast_mode=True, progress=False)
    ent = _entries(blob)
    assert ent["flac"][3]["format"] == "flac" and ent["raw"][3]["format"] is None   # the schema is the union, nulls where fast mode wrote nothing
    assert bytes_read(*ent["raw"][:3]).data == raw
    a, b = audio_read(*ent["flac"][:3]).array, audio_read(*ent["raw"][:3]).array
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_generic_modality_archive_roundtrip_local_and_remote(tmp_path):
    payloads = {"p0": b"%PDF-1.4 fake", "p1": bytes(range(256)) * 3, "p2": b""}
    (tmp_path / "p1.bin").write_bytes(payloads["p1"])
    blob = Blob(archive_dir=str(tmp_path / "arch"), modality="pdf")
    blob.append(items=[payloads["p0"], str(tmp_path / "p1.bin"), payloads["p2"]], ids=list(payloads), progress=False)
    blob.append(items=[b"second append"], ids=["p3"], num_workers=2, fast_mode=True, progress=False)   # fast_mode is a no-op here: same writer
    payloads["p3"] = b"second append"
    ent = _entries(blob)
    for k, raw in payloads.items():
        path, off, size, row = ent[k]
        assert set(row) == set(INDEX_COLUMNS)
        assert get_modality("pdf").read(path, off, size).data == raw
        assert bytes_read(path, off, size).data == raw

    class RangeHandler(BaseHTTPRequestHandler):            # the stock SimpleHTTPRequestHandler ignores Range
        def do_GET(self):
            data = (tmp_path / "arch" / self.path.lstrip("/")).read_bytes()
            lo, hi = map(int, self.headers["Range"].split("=")[1].split("-"))
            body = data[lo:hi + 1]
            self.send_response(206); self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}"); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for k, raw in payloads.items():
            path, off, size, _ = ent[k]
            if size == 0:
                continue                                              # a zero-length range is not a valid Range header
            got = bytes_read(f"{base}/{path.rsplit('/', 1)[1]}", off, size).data
            assert got == raw, k
    finally:
        server.shutdown(); server.server_close()


def test_image_write_copies_matching_format(tmp_path):
    """A JPEG stored as JPEG (or PNG as PNG) keeps its bytes: no re-encode."""
    from PIL import Image
    from omniio.modalities.image.write import image_write
    arr = (np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8)
    for fmt, ext in (("jpeg", "jpg"), ("png", "png")):
        p = tmp_path / f"x.{ext}"; Image.fromarray(arr).save(p, format=fmt.upper())
        raw, meta = image_write(str(p), "x", target_format=fmt)
        assert raw == p.read_bytes() and meta["width"] == 64 and meta["height"] == 48 and meta["channels"] == 3
    raw, meta = image_write(str(tmp_path / "x.png"), "x", target_format="jpeg")   # conversion still re-encodes
    assert raw[:2] == b"\xff\xd8" and meta["format"] == "jpeg"

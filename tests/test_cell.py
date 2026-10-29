"""The ``cell`` modality: a codec for per-cell sparse rows, written through ``Blob.append``
(one row per item, or many rows per `CellRows` item) and read back one byte range per cell."""
import os
import socket
import threading

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

from omniio.blob.blob import Blob
from omniio.blob.entries import Entries
from omniio.cell.common import FLAG_DENSE, HEADER_SIZE, auto_dtype_code, decode_row, encode_row, parse_header, varint_decode, varint_encode
from omniio.cell.read import RangeNotSupportedError, coalesce
from omniio.cell.write import CellRows, cell_write
from omniio.interface import cell_read, cell_read_batch
from tests.cell_fixtures import Rows, counts_matrix, entries, expected_row


def _rows(X):
    return {f"cell_{i:05d}": expected_row(X, i) for i in range(X.shape[0])}


# ------------------------------------------------------------------ codec
def test_varint_delta_coding_is_exact():
    rng = np.random.default_rng(0)
    v = np.concatenate([[0, 1, 127, 128, 16383, 16384, (1 << 21) - 1, 1 << 21, 1 << 28, (1 << 35) - 1], rng.integers(0, 1 << 20, 5000)])
    assert np.array_equal(varint_decode(varint_encode(v), len(v)), v.astype(np.uint64))
    assert varint_encode(np.zeros(0)) == b"" and varint_decode(b"", 0).size == 0


@pytest.mark.parametrize("kw", [{}, {"compress": True}, {"raw_indices": True}, {"layout": "dense"}, {"value_dtype": "uint32"}, {"value_dtype": "float32"}])
def test_encode_decode_row_round_trip_drops_zeros_and_sums_duplicates(kw):
    rng = np.random.default_rng(1)
    for n_cols, nnz in [(0, 0), (1, 0), (1, 1), (33000, 0), (33000, 2000), (33000, 33000), (500, 499)]:
        idx = rng.choice(n_cols, nnz, replace=False) if nnz else np.zeros(0, dtype=int)
        val = rng.integers(0, 50, nnz).astype(np.float32)             # some explicit zeros
        raw, head = encode_row(idx, val, n_cols, **kw)
        h2, i2, v2 = decode_row(raw)
        o = np.argsort(idx)
        keep = val[o] != 0
        assert np.array_equal(i2, idx[o][keep]) and np.array_equal(v2, val[o][keep]) and h2.n_cols == n_cols
        assert parse_header(raw).nnz == (n_cols if h2.dense else int(keep.sum()))
    raw, _ = encode_row([3, 3, 1], [2, 5, 1], 5)
    _, i, v = decode_row(raw)
    assert i.tolist() == [1, 3] and v.tolist() == [1, 7]
    assert parse_header(encode_row([0, 1, 2], [1, 1, 1], 4)[0]).flags & FLAG_DENSE   # auto: 3 of 4 columns
    assert HEADER_SIZE == 15


def test_value_casts_are_checked():
    with pytest.raises(ValueError, match="overflows"):
        encode_row([1], [70000], 5)
    with pytest.raises(ValueError, match="non-integral"):
        encode_row([1], [1.5], 5)
    with pytest.raises(ValueError, match="negative"):
        encode_row([1], [-1], 5)
    with pytest.raises(ValueError, match="out of range"):
        encode_row([5], [1], 5)
    raw, _ = encode_row([1], [70000], 5, value_dtype="uint32")
    assert decode_row(raw)[2].tolist() == [70000]
    raw, _ = encode_row([1, 2], [1.5, -2.0], 5, value_dtype="float32")
    assert decode_row(raw)[2].tolist() == [1.5, -2.0]
    assert [auto_dtype_code(np.array(v)) for v in ([1, 2], [300], [70000], [5e9], [1.5], [-1], np.zeros(0))] == [1, 2, 3, 5, 4, 4, 2]


# ------------------------------------------------------------------ writer forms
def test_cell_write_single_row_forms_and_metadata():
    row = sp.csr_matrix(np.array([[0, 3, 0, 0, 2, 0]], dtype=np.float32))
    for src in (row, row.toarray()[0], (np.array([1, 4]), np.array([3, 2]), 6), {"indices": [1, 4], "values": [3, 2], "n_cols": 6, "cell_type": "T"}):
        raw, meta = cell_write(src, "c0")
        assert meta["nnz"] == 2 and meta["total_counts"] == 5.0 and meta["n_genes"] == 6 and meta["format"] == "cell"
        assert meta["stored_size"] == len(raw) and meta["value_dtype"] == "uint16"
        _, i, v = decode_row(raw)
        assert i.tolist() == [1, 4] and v.tolist() == [3, 2]
    assert cell_write({"indices": [1, 4], "values": [3, 2], "n_cols": 6, "cell_type": "T"}, "c")[1]["cell_type"] == "T"
    assert cell_write(row, "c", compress=True)[1]["format"] == "cell.zst"


def test_cell_write_many_rows_returns_entries():
    X = counts_matrix(0, 20, 50)
    e = cell_write(Rows(0, 20, 50, 0, 3, prefix="p:"), "ignored")
    assert isinstance(e, Entries)
    got = list(e)
    assert [g[0] for g in got] == ["p:cell_00000", "p:cell_00001", "p:cell_00002"]
    assert got[1][2]["source_row"] == 1 and got[1][2]["group"] == "odd" and got[1][2]["n_genes"] == 50
    assert np.array_equal(decode_row(got[2][1])[1], expected_row(X, 2)[0])
    gen = ((f"g{i}", X[i]) for i in range(2))                        # a bare generator works too (single process)
    assert [g[0] for g in cell_write(gen, "x")] == ["g0", "g1"]


# ------------------------------------------------------------------ through Blob.append
@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """300 cells x 500 genes written as 5 `Rows` items by 2 workers, compressed or not."""
    out = {}
    for compress in (False, True):
        d = tmp_path_factory.mktemp(f"arc_{compress}")
        b = Blob(str(d), "cell")
        items = [Rows(0, 300, 500, s, s + 60) for s in range(0, 300, 60)]
        b.append(items, num_workers=2, progress=False, compress=compress)
        out[compress] = (b, entries(b), counts_matrix(0, 300, 500))
    return out


@pytest.mark.parametrize("compress", [False, True])
def test_every_cell_round_trips_exactly(archive, compress):
    b, ent, X = archive[compress]
    assert len(b) == 300 and set(ent) == set(_rows(X))
    for cid, (idx, val) in _rows(X).items():
        path, off, size, row = ent[cid]
        r = cell_read(path, off, size)
        assert np.array_equal(r.indices, idx) and np.array_equal(r.values, val.astype(np.uint16)), cid
        assert r.values.dtype == np.uint16 and r.nnz == idx.size and r.n_genes == 500 and r.entry_size == size
        assert r.file_type == ("cell.zst" if compress else "cell") and r.modality == "cell"
        assert row["nnz"] == idx.size and row["n_genes"] == 500 and row["format"] == r.file_type
        assert row["total_counts"] == float(val.sum()) and row["group"] == ("even" if row["source_row"] % 2 == 0 else "odd")
    i = 11
    r = cell_read(*ent[f"cell_{i:05d}"][:3])
    assert np.array_equal(r.dense(), X[i].toarray()[0].astype(np.uint16)) and r.to_csr().shape == (1, 500)
    assert cell_read(*ent["cell_00007"][:3]).nnz == 0                                     # an empty cell


def test_batch_read_equals_stacked_single_reads(archive):
    b, ent, X = archive[False]
    ids = list(ent)
    rng = np.random.default_rng(6)
    pick = [ids[k] for k in rng.choice(len(ids), 256, replace=True)]
    by_path = {}
    for cid in pick:
        by_path.setdefault(ent[cid][0], []).append(cid)
    for path, cids in by_path.items():
        B = cell_read_batch(path, [(ent[c][1], ent[c][2]) for c in cids])
        assert B.shape == (len(cids), 500) and B.dtype == np.uint16
        singles = sp.vstack([cell_read(*ent[c][:3]).to_csr() for c in cids])
        assert (B != singles).nnz == 0
        assert cell_read_batch(path, [(ent[c][1], ent[c][2]) for c in cids], max_gap=0).shape == B.shape
    assert coalesce([(100, 10), (0, 10), (10, 5), (500, 1)], max_gap=0) == [[0, 15, [1, 2]], [100, 10, [0]], [500, 1, [3]]]
    assert cell_read_batch(next(iter(by_path)), []).shape[0] == 0


def test_batch_refuses_mixed_panel_widths(tmp_path):
    b = Blob(str(tmp_path / "mix"), "cell")
    b.append([(np.array([1]), np.array([2]), 10), (np.array([1]), np.array([2]), 11)], ids=["a", "b"], progress=False)
    ent = entries(b)
    with pytest.raises(ValueError, match="different panel widths"):
        cell_read_batch(ent["a"][0], [(ent["a"][1], ent["a"][2]), (ent["b"][1], ent["b"][2])])


def _payloads(blob) -> dict:
    out = {}
    for cid, (path, off, size, _) in entries(blob).items():
        with open(path, "rb") as f:
            f.seek(off)
            out[cid] = f.read(size)
    return out


@pytest.mark.parametrize("compress", [False, True])
def test_worker_invariance(tmp_path, compress):
    ref = None
    for nw in (0, 1, 4):
        b = Blob(str(tmp_path / f"w{nw}_{compress}"), "cell")
        b.append([Rows(0, 300, 500, s, s + 20, prefix="a:") for s in range(0, 300, 20)]
                 + [Rows(1, 300, 500, s, s + 20, prefix="b:") for s in range(0, 300, 20)],
                 num_workers=nw, progress=False, compress=compress)
        m = _payloads(b)
        assert len(m) == 600
        if ref is None:
            ref = m
        assert m == ref, nw


def test_blob_append_takes_single_rows_and_generators(tmp_path):
    X = counts_matrix(3, 20, 40)
    b = Blob(str(tmp_path / "direct"), "cell")
    b.append([X[0], X[1].toarray()[0], {"indices": [2], "values": [9], "n_cols": 40, "note": "d"}], ids=["r0", "r1", "r2"], progress=False)
    b.append([((f"g{i}", X[i]) for i in range(2, 5))], ids=["gen"], progress=False)      # generator item: single process
    ent = entries(b)
    assert set(ent) == {"r0", "r1", "r2", "g2", "g3", "g4"} and ent["r2"][3]["note"] == "d"
    for cid, i in (("r0", 0), ("r1", 1), ("g4", 4)):
        assert np.array_equal(cell_read(*ent[cid][:3]).dense(), X[i].toarray()[0].astype(np.uint16))


# ------------------------------------------------------------------ remote
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(archive_dir):
    import socketserver
    from omniio import remote
    port = _free_port()
    ready = threading.Event()

    def run():
        orig = socketserver.TCPServer.serve_forever

        def hooked(self, *a, **k):
            ready.set()
            return orig(self, *a, **k)
        socketserver.TCPServer.serve_forever = hooked
        try:
            remote.serve(port, str(archive_dir), host="127.0.0.1")
        finally:
            socketserver.TCPServer.serve_forever = orig
    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(10)
    return f"http://127.0.0.1:{port}"


def test_remote_reads_equal_local_and_a_range_blind_server_is_detected(archive, tmp_path):
    pytest.importorskip("requests")
    b, ent, X = archive[False]
    url = _serve(b.archive_path.parent)
    remote_of = lambda path: url + "/" + os.path.basename(path)
    for cid in list(ent)[::7]:
        path, off, size, _ = ent[cid]
        a, r = cell_read(path, off, size), cell_read(remote_of(path), off, size)
        assert np.array_equal(a.indices, r.indices) and np.array_equal(a.values, r.values)
    path = ent["cell_00000"][0]
    sel = [c for c in ent if ent[c][0] == path][:40]
    ranges = [(ent[c][1], ent[c][2]) for c in sel]
    assert (cell_read_batch(path, ranges) != cell_read_batch(remote_of(path), ranges)).nnz == 0

    import functools
    import http.server
    import socketserver
    port = _free_port()
    httpd = socketserver.TCPServer(("127.0.0.1", port), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(b.archive_path.parent)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RangeNotSupportedError):
            cell_read(f"http://127.0.0.1:{port}/" + os.path.basename(path), *ent["cell_00000"][1:3])
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------ failure recovery
def test_failed_worker_then_rerun_completes_the_archive(tmp_path, monkeypatch):
    from omniio.cell import write as cw
    items = [Rows(0, 300, 500, s, s + 50, prefix="a:") for s in range(0, 300, 50)] + [Rows(1, 300, 500, s, s + 50, prefix="b:") for s in range(0, 300, 50)]
    full = Blob(str(tmp_path / "full"), "cell")
    full.append(items, num_workers=2, progress=False)
    want = _payloads(full)

    real = cw.cell_write

    def flaky(source, item_id, **kw):
        if isinstance(source, Rows) and source.prefix == "b:" and source.start == 250:
            # worker 1's last item: wait until worker 0 has flushed its shard, then die.
            # The pool breaks, every future raises, and only what is on disk survives.
            import time
            for _ in range(600):
                if (part / "logs" / "shard_0_0.parquet").exists():
                    break
                time.sleep(0.05)
            os._exit(3)
        return real(source, item_id, **kw)
    monkeypatch.setattr(cw, "cell_write", flaky)
    part = tmp_path / "part"
    with pytest.raises(RuntimeError):
        Blob(str(part), "cell").append(items, num_workers=2, progress=False)
    survivors = _payloads(Blob(str(part), "cell"))
    assert len(survivors) == 300 and all(want[k] == v for k, v in survivors.items())     # worker 0's 6 items

    monkeypatch.setattr(cw, "cell_write", real)
    blob = Blob(str(part), "cell")
    blob.append(items, num_workers=2, progress=False)       # ids already there are skipped (the workers warn; warnings stay in the workers)
    got = _payloads(blob)
    assert got == want and len(blob) == len(want)


# ------------------------------------------------------------------ dtypes, duplicates, registry
def test_value_dtype_auto_and_overflow(tmp_path):
    X = counts_matrix(0, 300, 500, big=True)
    with pytest.raises(RuntimeError, match="overflows uint16"):
        Blob(str(tmp_path / "o16"), "cell").append([Rows(0, 300, 500, big=True)], progress=False)
    b = Blob(str(tmp_path / "auto"), "cell")
    b.append([Rows(0, 300, 500, big=True)], progress=False, value_dtype="auto")
    ent = entries(b)
    assert ent["cell_00012"][3]["value_dtype"] == "uint32" and ent["cell_00011"][3]["value_dtype"] == "uint8"
    assert sum(e[3]["value_dtype"] == "uint8" for e in ent.values()) > 100
    for cid, i in (("cell_00012", 12), ("cell_00011", 11), ("cell_00000", 0)):
        assert np.array_equal(cell_read(*ent[cid][:3]).dense().astype(np.float64), X[i].toarray()[0])
    B = cell_read_batch(ent["cell_00012"][0], [ent["cell_00012"][1:3], ent["cell_00011"][1:3]])     # mixed widths upcast
    assert B.dtype == np.uint32 and B[0, 3] == 70000
    b32 = Blob(str(tmp_path / "o32"), "cell")
    b32.append([Rows(0, 300, 500, big=True)], progress=False, value_dtype="uint32")
    r = cell_read(*entries(b32)["cell_00012"][:3])
    assert r.values.dtype == np.uint32 and r.dense()[3] == 70000


def test_duplicate_ids_are_skipped_with_a_warning(tmp_path):
    b = Blob(str(tmp_path / "dup"), "cell")
    b.append([Rows(0, 20, 50, dup=True)], progress=False)                # the repeated id inside one item is written twice: Blob checks against the archive
    assert len(b) == 21
    with pytest.warns(UserWarning, match="duplicate"):
        b.append([Rows(0, 20, 50)], progress=False)                       # a second append of the same ids: skipped
    assert len(b) == 21
    with pytest.raises(ValueError, match="Duplicate id"):
        b.append([X_ for X_ in [(np.array([1]), np.array([1]), 50)] * 2], ids=["x", "x"], progress=False)


def test_public_alias_and_registry():
    import omniio.cell
    from omniio.blob.write import modality_writer
    from omniio.modalities.base import MODALITIES
    from omniio.modalities.cell.write import cell_write as real
    assert omniio.cell.write.cell_write is real and modality_writer["cell"] is real and MODALITIES["cell"].name == "cell"
    assert isinstance(Rows(), CellRows)

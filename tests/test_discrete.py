"""The `discrete` modality: integer streams (VQ / RVQ codes, token ids), codec agnostic."""
import numpy as np
import pytest

from omniio.interface import discrete_read
from omniio.modalities.discrete.common import bits_for, decode, encode, pack_bits, unpack_bits
from omniio.modalities.discrete.write import discrete_write


def _archive(tmp_path, entries):
    """Write several (raw_bytes) into one .bin with junk in between -> [(offset, size)]."""
    path = tmp_path / "archive.bin"
    spans = []
    with open(path, "wb") as f:
        f.write(b"junk-header")
        for raw in entries:
            spans.append((f.tell(), len(raw)))
            f.write(raw)
            f.write(b"\x00" * 3)
    return str(path), spans


@pytest.mark.parametrize("bits", [1, 2, 3, 7, 8, 9, 10, 11, 12, 15, 16, 17, 24, 31, 32, 33, 56, 64])
def test_bit_packing_round_trip_is_exact_and_minimal(bits):
    rng = np.random.default_rng(bits)
    for n in (0, 1, 5, 8, 9, 1000):
        a = rng.integers(0, 1 << min(bits, 62), size=n, dtype=np.int64)
        buf = pack_bits(a, bits)
        assert len(buf) == (n * bits + 7) // 8
        back = unpack_bits(buf, bits, n)
        assert np.array_equal(back.astype(np.int64), a)
        assert back.dtype.itemsize == (1 if bits <= 8 else 2 if bits <= 16 else 4 if bits <= 32 else 8)
    with pytest.raises(ValueError):
        pack_bits(np.array([1 << bits]) if bits < 62 else np.array([1]), bits if bits < 62 else 60)


def test_bits_for_alphabets():
    assert [bits_for(v) for v in (1, 2, 3, 256, 257, 1024, 1025, 2048, 65536)] == [1, 1, 2, 8, 9, 10, 11, 11, 16]
    assert bits_for(2 ** 60) == 64          # wider than the packer's window: stored as u64


def test_rvq_entry_is_bit_packed_codebook_major_and_reads_back(tmp_path):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1024, size=(8, 750))            # 8 codebooks x 10 s @ 75 Hz
    raw, meta = discrete_write(codes, "utt0", vocab_size=1024, rate=75.0, codec="encodec_24khz_6kbps")
    assert meta["n_streams"] == 8 and meta["lengths"] == [750] * 8 and meta["bits"] == [10] * 8
    assert meta["rates"] == [75.0] * 8 and meta["duration"] == pytest.approx(10.0) and meta["n_tokens"] == 6000
    assert meta["codec"] == "encodec_24khz_6kbps" and meta["format"] == "discrete" and not meta["ragged"]
    header = 8 + 8 * 13
    assert meta["stored_size"] == header + 8 * ((750 * 10 + 7) // 8)   # 1.25 B/token + header
    path, [(off, size)] = _archive(tmp_path, [raw])
    r = discrete_read(path, off, size)
    assert r.modality == "discrete" and r.n_streams == 8 and r.array.shape == (8, 750)
    assert np.array_equal(r.array, codes) and r.array.dtype == np.uint16
    assert r.vocab_sizes == [1024] * 8 and r.rates == [75.0] * 8 and r.stream_indices == list(range(8))


def test_subset_of_streams_time_window_and_index_window(tmp_path):
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 1024, size=(4, 750))
    raw, _ = discrete_write(codes, "u", vocab_size=1024, rate=75.0)
    path, [(off, size)] = _archive(tmp_path, [raw])
    r = discrete_read(path, off, size, streams=[0, 1])                   # coarse codebooks only
    assert r.stream_indices == [0, 1] and np.array_equal(r.array, codes[:2]) and r.n_streams == 4
    r = discrete_read(path, off, size, start_time=1.0, end_time=2.5)     # 75..188 (floor / ceil)
    assert r.array.shape == (4, 113) and np.array_equal(r.array, codes[:, 75:188])
    r = discrete_read(path, off, size, start_frame=10, end_frame=20, streams=[3])
    assert np.array_equal(r.array[0], codes[3, 10:20])
    r = discrete_read(path, off, size, end_time=100.0)                   # clipped to the entry
    assert r.array.shape == (4, 750)
    r = discrete_read(path, off, size, start_frame=5, end_frame=8, start_time=1.0, end_time=2.0)   # frames win
    assert np.array_equal(r.array, codes[:, 5:8])


def test_ragged_streams_with_their_own_rates_and_alphabets(tmp_path):
    rng = np.random.default_rng(2)
    semantic = rng.integers(0, 2048, size=125)      # 12.5 Hz
    acoustic = [rng.integers(0, 1024, size=750) for _ in range(3)]   # 75 Hz
    raw, meta = discrete_write([semantic, *acoustic], "u", vocab_size=[2048, 1024, 1024, 1024], rate=[12.5, 75, 75, 75])
    assert meta["ragged"] and meta["lengths"] == [125, 750, 750, 750] and meta["bits"] == [11, 10, 10, 10]
    assert meta["duration"] == pytest.approx(10.0)
    path, [(off, size)] = _archive(tmp_path, [raw])
    r = discrete_read(path, off, size)
    assert r.ragged and r.lengths == [125, 750, 750, 750]
    with pytest.raises(ValueError):
        _ = r.array
    padded = r.to_array(pad_value=-1)
    assert padded.shape == (4, 750) and padded[0, 125] == -1 and np.array_equal(padded[0, :125], semantic)
    # a time window maps through each stream's own rate
    r = discrete_read(path, off, size, start_time=2.0, end_time=4.0)
    assert r.lengths == [25, 150, 150, 150]
    assert np.array_equal(r.streams[0], semantic[25:50]) and np.array_equal(r.streams[1], acoustic[0][150:300])


def test_rate_less_streams_reject_time_windows_but_slice_by_index(tmp_path):
    ids = np.arange(0, 500) % 300
    raw, meta = discrete_write(ids, "tokens", vocab_size=300)
    assert meta["rates"] is None and meta["duration"] is None and meta["bits"] == [9]
    path, [(off, size)] = _archive(tmp_path, [raw])
    with pytest.raises(ValueError, match="rate"):
        discrete_read(path, off, size, start_time=0.0, end_time=1.0)
    r = discrete_read(path, off, size, start_frame=100, end_frame=105)
    assert np.array_equal(r.streams[0], ids[100:105]) and r.rates is None


def test_compress_flag_zstd_wraps_the_payload_and_reads_back(tmp_path):
    codes = np.zeros((8, 750), dtype=np.int64)                       # silence-like: compressible
    codes[:, ::7] = 5
    raw, meta = discrete_write(codes, "u", vocab_size=1024, rate=75.0, compress=True)
    assert meta["format"] == "discrete.zst" and meta["stored_size"] < meta["original_size"] // 4
    path, [(off, size)] = _archive(tmp_path, [raw])
    assert np.array_equal(discrete_read(path, off, size).array, codes)


def test_source_forms_npy_npz_dict_and_validation(tmp_path):
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 1024, size=(2, 100))
    np.save(tmp_path / "c.npy", codes)
    np.savez(tmp_path / "c.npz", stream_0=codes[0], stream_1=codes[1], vocab_sizes=np.array([1024, 1024]), rates=np.array([50.0, 50.0]))
    r1, m1 = discrete_write(str(tmp_path / "c.npy"), "a", vocab_size=1024)
    r2, m2 = discrete_write(str(tmp_path / "c.npz"), "b")
    r3, m3 = discrete_write({"streams": codes, "vocab_size": 1024, "rate": 50.0, "codec": "x"}, "c")
    assert m1["bits"] == m2["bits"] == m3["bits"] == [10, 10] and m2["rates"] == [50.0, 50.0] and m3["codec"] == "x"
    infos, arrs = decode(r2)
    assert np.array_equal(np.stack(arrs), codes)
    with pytest.raises(ValueError):                                   # value outside the declared alphabet
        discrete_write(np.array([0, 1024]), "bad", vocab_size=1024)
    with pytest.raises(ValueError):                                   # negative
        discrete_write(np.array([0, -1]), "bad")
    with pytest.raises(ValueError):                                   # per-stream list of the wrong length
        discrete_write(codes, "bad", vocab_size=[1024])
    raw, meta = discrete_write(np.array([3, 1, 2]), "auto")           # default alphabet = max + 1
    assert meta["vocab_sizes"] == [4] and meta["bits"] == [2]


def test_blob_append_and_public_import_path(tmp_path):
    import omniio.discrete                                            # alias -> omniio.modalities.discrete
    from omniio.modalities.discrete.write import discrete_write as real
    assert omniio.discrete.write.discrete_write is real
    from omniio.blob.write import modality_writer
    assert modality_writer["discrete"] is real


# ---- true partial reads: only the bytes of the requested streams / frames are fetched

def _counting_reader(raw):
    calls = []

    def read(off, size):
        size = min(size, len(raw) - off)
        calls.append((off, size))
        return raw[off: off + size]
    return read, calls


@pytest.mark.parametrize("bits", [3, 10, 13, 16, 17])
def test_frame_window_unpacks_at_a_bit_offset_and_fetches_only_its_bytes(bits):
    from omniio.modalities.discrete.common import HEADER_GUESS_STREAMS, decode_partial, header_size
    rng = np.random.default_rng(bits)
    codes = rng.integers(0, 1 << bits, size=(3, 1000))
    raw, _ = discrete_write(codes, "x", vocab_size=1 << bits)
    for lo, hi in [(0, 1000), (1, 2), (7, 8), (3, 999), (500, 500), (123, 456), (999, 1000)]:
        read, calls = _counting_reader(raw)
        infos, idx, arrays, windows, n_read = decode_partial(read, start_frame=lo, end_frame=hi)
        assert idx == [0, 1, 2] and windows == [(lo, hi)] * 3
        for k in range(3):
            assert np.array_equal(arrays[k].astype(np.int64), codes[k, lo:hi])
        # one header read + one read per stream, each just the window's byte span (a span
        # that lies inside the header over-read is served from it, no extra I/O)
        span = (hi * bits + 7) // 8 - (lo * bits) // 8 if hi > lo else 0
        assert calls[0] == (0, header_size(HEADER_GUESS_STREAMS))
        assert all(s == span for _, s in calls[1:]) and len(calls) <= 4
        assert n_read <= header_size(HEADER_GUESS_STREAMS) + 3 * span
        if span > header_size(HEADER_GUESS_STREAMS):          # nothing served from the over-read
            assert n_read == header_size(HEADER_GUESS_STREAMS) + 3 * span


def test_level_window_is_one_contiguous_read_and_matches_streams_list(tmp_path):
    from omniio.modalities.discrete.common import HEADER_GUESS_STREAMS, decode_partial, header_size
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 1024, size=(8, 750))
    raw, _ = discrete_write(codes, "x", vocab_size=1024, rate=75.0)
    per_stream = (750 * 10 + 7) // 8
    read, calls = _counting_reader(raw)
    infos, idx, arrays, windows, n_read = decode_partial(read, start_level=2, end_level=5)
    assert idx == [2, 3, 4] and np.array_equal(np.stack(arrays), codes[2:5])
    assert calls[1:] == [(header_size(8) + 2 * per_stream, 3 * per_stream)]     # one read
    assert n_read == header_size(HEADER_GUESS_STREAMS) + 3 * per_stream
    # the public reader: start_level/end_level == streams=range(...); bytes_read reported
    path, [(off, size)] = _archive(tmp_path, [raw])
    a = discrete_read(path, off, size, start_level=2, end_level=5)
    b = discrete_read(path, off, size, streams=[2, 3, 4])
    assert np.array_equal(a.array, b.array) and a.stream_indices == b.stream_indices == [2, 3, 4]
    assert a.bytes_read == b.bytes_read == n_read and a.entry_size == size
    assert a.frame_windows == [(0, 750)] * 3
    # first k codebooks (the usual coarse-to-fine use) and open-ended windows
    assert discrete_read(path, off, size, end_level=1).array.shape == (1, 750)
    assert discrete_read(path, off, size, start_level=6).stream_indices == [6, 7]
    assert discrete_read(path, off, size, start_level=6, end_level=100).stream_indices == [6, 7]
    # combined with a time window: 3 small reads, decoded == full-then-slice
    c = discrete_read(path, off, size, start_level=2, end_level=5, start_time=1.0, end_time=2.5)
    assert c.frame_windows == [(75, 188)] * 3 and np.array_equal(c.array, codes[2:5, 75:188])
    assert c.bytes_read == header_size(HEADER_GUESS_STREAMS) + 3 * ((188 * 10 + 7) // 8 - (75 * 10) // 8)
    with pytest.raises(ValueError):
        discrete_read(path, off, size, streams=[0], start_level=1)
    with pytest.raises(IndexError):
        discrete_read(path, off, size, streams=[8])


def test_zstd_entries_read_the_payload_once_and_still_window(tmp_path):
    rng = np.random.default_rng(6)
    codes = rng.integers(0, 512, size=(4, 400))
    raw, meta = discrete_write(codes, "x", vocab_size=512, rate=50.0, compress=True)
    path, [(off, size)] = _archive(tmp_path, [raw])
    r = discrete_read(path, off, size, start_level=1, end_level=3, start_frame=10, end_frame=33)
    assert np.array_equal(r.array, codes[1:3, 10:33]) and r.bytes_read == size


def test_remote_partial_reads_issue_one_range_request_per_fetched_span(tmp_path, monkeypatch):
    from omniio.modalities.discrete import read as rmod
    from omniio.modalities.discrete.common import HEADER_GUESS_STREAMS, header_size
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 1024, size=(4, 750))
    raw, _ = discrete_write(codes, "x", vocab_size=1024)
    blob = b"pad" * 5 + raw + b"tail"
    ranges = []

    class _Resp:
        def __init__(self, content): self.content = content
        def raise_for_status(self): pass

    def fake_get(url, headers):
        a, b = map(int, headers["Range"][len("bytes="):].split("-"))
        ranges.append((a, b))
        return _Resp(blob[a: b + 1])
    monkeypatch.setattr(rmod.requests, "get", fake_get)
    r = rmod.discrete_read_remote("http://x/a.bin", 15, len(raw), start_level=1, end_level=3, start_frame=100, end_frame=200)
    assert np.array_equal(r.array, codes[1:3, 100:200])
    assert len(ranges) == 3 and ranges[0] == (15, 15 + header_size(HEADER_GUESS_STREAMS) - 1)   # header, then 2 stream spans
    span = (200 * 10 + 7) // 8 - (100 * 10) // 8
    assert all(b - a + 1 == span for a, b in ranges[1:])

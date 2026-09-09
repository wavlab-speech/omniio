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

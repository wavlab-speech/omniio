"""The `seq` (2-bit FASTA) and `track` (binned bigWig) modalities, plan 3.1 / 3.4.1; real
hg38 / ENCODE checks are gated on OMNIIO_CELL_DATA."""
import gzip
import os

import numpy as np
import pytest

from omniio.blob.blob import Blob
from omniio.interface import seq_read, track_read
from omniio.seq.common import decode_window, encode as seq_encode, parse_header as seq_header
from omniio.seq.write import iter_fasta, seq_write
from omniio.track.common import bin_values, encode as track_encode
from omniio.track.write import track_write
from tests.genome_fixtures import GenomeArchive

DATA = os.environ.get("OMNIIO_CELL_DATA")
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rc(s):
    return s.translate(_COMP)[::-1]


def _random_seq(rng, n):
    s = rng.choice(list("ACGT"), n)
    for _ in range(n // 500 + 1):                       # N runs and soft-masked runs
        a = int(rng.integers(0, n)); s[a: a + int(rng.integers(1, 40))] = "N"
    s = "".join(s)
    out = []
    i = 0
    while i < len(s):
        k = int(rng.integers(5, 300))
        out.append(s[i: i + k].lower() if rng.random() < 0.3 else s[i: i + k]); i += k
    return "".join(out)


# ------------------------------------------------------------------ seq
@pytest.mark.parametrize("n", [0, 1, 3, 4, 5, 8, 9, 1000, 12345])
def test_seq_codec_round_trip_windows_and_reverse_complement(n):
    rng = np.random.default_rng(n)
    s = _random_seq(rng, n) if n else ""
    raw, head, stats = seq_encode(s)
    assert head.length == n and stats["n_count"] == s.upper().count("N") and stats["lower_count"] == sum(c.islower() for c in s)
    assert seq_header(raw).entry_size == len(raw)
    from omniio.modalities.seq.common import HEADER_SIZE
    bases = raw[HEADER_SIZE: head.n_offset]
    nm = raw[head.n_offset: head.lower_offset] if head.has_n else None
    lo = raw[head.lower_offset:] if head.has_lower else None
    assert decode_window(head, bases, 0, n, nm, lo) == s
    for _ in range(20 if n else 0):
        a, b = sorted(rng.integers(0, n + 1, 2))
        bb = bases[a // 4: (b + 3) // 4]
        nn = nm[a // 8: (b + 7) // 8] if nm else None
        ll = lo[a // 8: (b + 7) // 8] if lo else None
        assert decode_window(head, bb, a, b, nn, ll) == s[a:b]
        assert decode_window(head, bb, a, b, nn, ll, reverse_complement=True) == _rc(s[a:b])
        assert decode_window(head, bb, a, b, nn, ll, soft_mask=False) == s[a:b].upper()


def test_fasta_ingest_and_windows(tmp_path):
    rng = np.random.default_rng(1)
    recs = {"chrA": _random_seq(rng, 20000), "chrB": _random_seq(rng, 777), "chrM": "ACGT" * 10}
    fa = tmp_path / "toy.fa.gz"
    with gzip.open(fa, "wt") as f:
        for k, v in recs.items():
            f.write(f">{k} some description\n")
            for i in range(0, len(v), 60):
                f.write(v[i: i + 60] + "\n")
    assert dict(iter_fasta(str(fa))) == {k: v.encode() for k, v in recs.items()}
    b = Blob(str(tmp_path / "seq"), "seq")
    b.append([str(fa)], ids=["toy"], progress=False)
    t = b.get_metadata().to_pandas().set_index("id")
    assert set(t.index) == set(recs) and t.loc["chrA", "length"] == 20000 and t.loc["chrA", "n_count"] == recs["chrA"].upper().count("N")
    G = GenomeArchive(tmp_path / "seq")
    assert G.modality == "seq" and G.lengths["chrB"] == 777
    for chrom, s in recs.items():
        assert G.read(chrom).sequence == s
        for _ in range(10):
            a, bnd = sorted(rng.integers(0, len(s) + 1, 2))
            r = G.read(chrom, a, bnd)
            assert r.sequence == s[a:bnd] and (r.start, r.end, r.length) == (a, bnd, len(s))
            assert G.read(chrom, a, bnd, reverse_complement=True).sequence == _rc(s[a:bnd])
            assert r.bytes_read <= 16 + (bnd - a) // 4 + 2 * ((bnd - a) // 8) + 6      # three small spans
    assert G.read("chrA", 19990, 30000).sequence == recs["chrA"][19990:]                # clipped to the record
    # single-record forms and the registry
    raw, meta = seq_write("acgtNNAC", "x")
    assert meta["length"] == 8 and meta["n_count"] == 2 and meta["lower_count"] == 4
    raw2, _ = seq_write(("chr1", "ACGT"), "x", keep_case=False)
    assert seq_header(raw2).has_lower is False


# ------------------------------------------------------------------ track
def test_bin_values_is_exact_nanmean():
    v = np.array([1, 2, np.nan, np.nan, 3, np.nan, 4, 5, 6], dtype=float)
    out = bin_values(v, 2)
    assert np.allclose(out[[0, 2, 3]], [1.5, 3, 4.5]) and np.isnan(out[1]) and out[4] == 6
    assert bin_values(np.zeros(0), 4).size == 0


def test_bigwig_ingest_matches_pybigwig_at_bin_resolution(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    rng = np.random.default_rng(2)
    chroms = {"chr1": 300_000, "chr2": 12_345}
    bw_path = tmp_path / "toy.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader(list(chroms.items()))
    truth = {}
    for c, L in chroms.items():
        vals = np.full(L, np.nan)
        pos = 0
        starts, ends, vs = [], [], []
        while pos < L:                                                     # runs of coverage with gaps
            run = int(rng.integers(50, 3000)); gap = int(rng.integers(0, 500))
            e = min(pos + run, L); val = float(rng.gamma(2.0)) * 10
            starts.append(pos); ends.append(e); vs.append(val); vals[pos:e] = val
            pos = e + gap
        bw.addEntries([c] * len(starts), starts, ends=ends, values=vs)
        truth[c] = vals
    bw.close()
    b = Blob(str(tmp_path / "trk"), "track")
    b.append([str(bw_path)], ids=["toy"], progress=False, bin=128)
    G = GenomeArchive(tmp_path / "trk")
    assert G.modality == "track" and G.bin == 128 and set(G.chroms) == set(chroms)
    bw = pyBigWig.open(str(bw_path))
    for c, L in chroms.items():
        want = bin_values(bw.values(c, 0, L, numpy=True), 128)
        r = G.read(c)
        assert r.array.size == want.size == (L + 127) // 128 and r.n_bins == want.size
        ok = np.isfinite(want)
        assert np.array_equal(np.isnan(r.array), ~ok)
        assert np.allclose(r.array[ok], want[ok], rtol=1e-3, atol=1e-3)               # float16
        for _ in range(10):
            a, e = sorted(rng.integers(0, L + 1, 2))
            r = G.read(c, a, e)
            k0, k1 = a // 128, min((e + 127) // 128, want.size)
            assert r.start == k0 * 128 and r.end == k1 * 128 and r.array.size == k1 - k0
            w = want[k0:k1]; m = np.isfinite(w)
            assert np.allclose(r.array[m], w[m], rtol=1e-3, atol=1e-3) and np.array_equal(np.isnan(r.array), ~m)
            assert r.bytes_read == 24 + 2 * (k1 - k0)
    t = b.get_metadata().to_pandas().set_index("id")
    assert 0 < t.loc["chr1", "coverage_fraction"] < 1 and t.loc["chr1", "length"] == 300_000
    # one chromosome and a prebinned array
    raw, meta = track_write((str(bw_path), "chr2"), "x", bin=64)
    assert meta["n_bins"] == (12_345 + 63) // 64 and meta["name"] == "chr2"
    raw, meta = track_write({"values": [1.0, np.nan, 2.0], "length": 300, "name": "p"}, "x", bin=100, dtype="float32")
    from omniio.track.common import parse_header
    assert parse_header(raw).dtype_code == 2 and meta["coverage_fraction"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------ real data (gated)
GENOME = os.path.join(DATA or "", "genome")
ENCODE = os.path.join(DATA or "", "encode")


@pytest.mark.skipif(not os.path.exists(os.path.join(GENOME, "chr21.fa.gz")), reason="needs OMNIIO_CELL_DATA/genome/chr21.fa.gz")
def test_hg38_chr21_chr22_against_pyfaidx(tmp_path):
    pyfaidx = pytest.importorskip("pyfaidx")
    import shutil
    fa = tmp_path / "chr21_22.fa"
    with open(fa, "wb") as out:
        for c in ("chr21", "chr22"):
            with gzip.open(os.path.join(GENOME, f"{c}.fa.gz"), "rb") as f:
                shutil.copyfileobj(f, out)
    b = Blob(str(tmp_path / "seq"), "seq")
    b.append([str(fa)], ids=["hg38"], progress=False)
    G = GenomeArchive(tmp_path / "seq")
    ref = pyfaidx.Fasta(str(fa))
    rng = np.random.default_rng(0)
    for c in ("chr21", "chr22"):
        L = len(ref[c]); assert G.lengths[c] == L
        for _ in range(100):
            a = int(rng.integers(0, L - 2000)); e = a + int(rng.integers(1, 2000))
            assert G.read(c, a, e).sequence == str(ref[c][a:e])
            assert G.read(c, a, e, reverse_complement=True).sequence == str(ref[c][a:e].reverse.complement)
    assert G.read("chr21", 0, 10_000).sequence == str(ref["chr21"][0:10_000])          # the N-run telomere


@pytest.mark.skipif(not os.path.exists(os.path.join(ENCODE, "K562_DNase_ENCFF029JXV.bigWig")), reason="needs OMNIIO_CELL_DATA/encode/*.bigWig")
def test_encode_dnase_track_against_pybigwig(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    path = os.path.join(ENCODE, "K562_DNase_ENCFF029JXV.bigWig")
    b = Blob(str(tmp_path / "trk"), "track")
    b.append([path], ids=["dnase"], progress=False, bin=128, chroms=["chr21", "chr22"])
    G = GenomeArchive(tmp_path / "trk")
    bw = pyBigWig.open(path)
    rng = np.random.default_rng(0)
    for c in ("chr21", "chr22"):
        L = bw.chroms(c)
        for _ in range(50):
            a = int(rng.integers(0, L - 100_000)); e = a + int(rng.integers(128, 100_000))
            r = G.read(c, a, e)
            want = bin_values(bw.values(c, r.start, min(r.end, L), numpy=True), 128)
            m = np.isfinite(want)
            assert r.array.size == want.size and np.array_equal(np.isnan(r.array), ~m)
            assert np.allclose(r.array[m], want[m], rtol=2e-3, atol=1e-3)

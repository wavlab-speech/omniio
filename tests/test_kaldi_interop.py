"""Byte-level interoperability checks against ``kaldiio``.

``kaldiio`` is *not* a dependency of omniio -- its license does not permit
redistribution, which is precisely why :mod:`omniio.kaldi` exists.  These tests
skip unless someone has installed it locally, in which case it is used purely
as a reference implementation to confirm that archives written by omniio are
byte-identical to the ones Kaldi tooling produces, and vice versa.
"""

import numpy as np
import pytest

from omniio import kaldi

kaldiio = pytest.importorskip("kaldiio", reason="kaldiio not installed")


@pytest.fixture
def feats():
    rng = np.random.RandomState(7)
    return (rng.randn(30, 13) * 4).astype(np.float32)


@pytest.fixture
def wav():
    rng = np.random.RandomState(8)
    return (rng.uniform(-1, 1, 4000) * 32767).astype(np.int16)


def _both(tmp_path, name, write):
    """Run ``write`` once through omniio and once through kaldiio."""
    ours = str(tmp_path / (name + ".ours"))
    theirs = str(tmp_path / (name + ".theirs"))
    write(kaldi, ours)
    write(kaldiio, theirs)
    return ours, theirs


def test_matrix_ark_bytes_match(tmp_path, feats):
    values = {"u1": feats, "u2": feats[0], "u3": feats[:3].astype(np.float64)}
    ours, theirs = _both(tmp_path, "m", lambda mod, p: mod.save_ark(p, values))
    assert open(ours, "rb").read() == open(theirs, "rb").read()
    for key, want in values.items():
        assert np.array_equal(dict(kaldiio.load_ark(ours))[key], want)
        assert np.array_equal(dict(kaldi.load_ark(theirs))[key], want)


def test_int32_vector_bytes_match(tmp_path):
    ali = np.arange(10, dtype=np.int32)
    ours, theirs = _both(tmp_path, "i", lambda mod, p: mod.save_ark(p, {"u1": ali}))
    assert open(ours, "rb").read() == open(theirs, "rb").read()


def test_save_mat_bytes_match(tmp_path, feats):
    ours, theirs = _both(tmp_path, "s", lambda mod, p: mod.save_mat(p, feats))
    assert open(ours, "rb").read() == open(theirs, "rb").read()
    assert np.array_equal(kaldiio.load_mat(ours), feats)
    assert np.array_equal(kaldi.load_mat(theirs), feats)


@pytest.mark.parametrize("method", [1, 2, 3, 4, 5])
def test_compressed_bytes_match(tmp_path, feats, method):
    def write(mod, p):
        with mod.WriteHelper("ark:" + p, compression_method=method) as w:
            w["u1"] = feats

    ours, theirs = _both(tmp_path, "c{}".format(method), write)
    assert open(ours, "rb").read() == open(theirs, "rb").read()
    assert np.array_equal(dict(kaldi.load_ark(theirs))["u1"], dict(kaldiio.load_ark(theirs))["u1"])


def test_wave_holder_ark_bytes_match(tmp_path, wav):
    ours, theirs = _both(tmp_path, "w", lambda mod, p: mod.save_ark(p, {"r1": (16000, wav)}))
    assert open(ours, "rb").read() == open(theirs, "rb").read()
    rate, array = dict(kaldi.load_ark(theirs))["r1"]
    assert (rate, array.dtype) == (16000, np.dtype(np.int16))
    assert np.array_equal(array, wav)


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_extended_audio_ark_bytes_match(tmp_path, wav, fmt):
    def write(mod, p):
        with open(p, "wb") as ark:
            mod.save_ark(
                ark,
                {"r1": (wav, 16000)},
                append=True,
                write_function="soundfile",
                write_kwargs={"format": fmt, "subtype": None},
            )

    ours, theirs = _both(tmp_path, "e" + fmt, write)
    assert open(ours, "rb").read() == open(theirs, "rb").read()
    mine = dict(kaldi.load_ark(ours))["r1"]
    reference = dict(kaldiio.load_ark(ours))["r1"]
    assert mine[0] == reference[0]
    assert np.array_equal(mine[1], reference[1])


def test_text_ark_bytes_match(tmp_path, feats):
    def write(mod, p):
        with mod.WriteHelper("ark,t:" + p) as w:
            w["u1"] = feats
            w["u2"] = feats[0]

    ours, theirs = _both(tmp_path, "t", write)
    assert open(ours).read() == open(theirs).read()


def test_scp_offsets_match(tmp_path, feats):
    def write(mod, p):
        mod.save_ark(p, {"u1": feats, "u2": feats[:3]}, scp=p + ".scp")

    ours, theirs = _both(tmp_path, "o", write)
    strip = lambda path: [  # noqa: E731
        line.replace(path, "ARK") for line in open(path + ".scp").read().splitlines()
    ]
    assert strip(ours) == strip(theirs)


def test_segments_match(tmp_path, wav):
    ark = str(tmp_path / "s.ark")
    scp = str(tmp_path / "s.scp")
    segments = tmp_path / "segments"
    kaldi.save_ark(ark, {"rec1": (16000, wav)}, scp=scp)
    segments.write_text("utt_a rec1 0.0 0.1\nutt_b rec1 0.1 0.2\n")

    mine = list(kaldi.ReadHelper("scp:" + scp, segments=str(segments)))
    reference = list(kaldiio.ReadHelper("scp:" + scp, segments=str(segments)))
    assert [k for k, _ in mine] == [k for k, _ in reference]
    for (_, (r1, a1)), (_, (r2, a2)) in zip(mine, reference):
        assert r1 == r2
        assert np.array_equal(a1, a2)

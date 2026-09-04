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


@pytest.fixture
def recordings(tmp_path, wav):
    scp = str(tmp_path / "w.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(
        str(tmp_path / "w.ark"),
        {"rec1": (16000, wav), "rec2": (16000, wav[::-1].copy())},
        scp=scp,
    )
    # -1 is Kaldi's "to the end" sentinel; 0 is not, and gives an empty slice.
    seg.write_text("utt_a rec1 0.05 0.1\nutt_b rec2 0.1 -1\nutt_c rec1 0.05 0\n")
    return scp, str(seg)


def _same(a, b):
    if isinstance(a, tuple) != isinstance(b, tuple):
        return False
    if isinstance(a, tuple):
        return a[0] == b[0] and np.array_equal(a[1], b[1])
    return np.array_equal(a, b)


def test_load_scp_with_segments_matches(recordings):
    scp, seg = recordings
    mine = kaldi.load_scp(scp, segments=seg)
    reference = kaldiio.load_scp(scp, segments=seg)
    assert list(mine) == list(reference)
    assert all(_same(mine[k], reference[k]) for k in reference)


def test_load_scp_sequential_with_segments_matches(recordings):
    scp, seg = recordings
    mine = list(kaldi.load_scp_sequential(scp, segments=seg))
    reference = list(kaldiio.load_scp_sequential(scp, segments=seg))
    assert [k for k, _ in mine] == [k for k, _ in reference]
    assert all(_same(a, b) for (_, a), (_, b) in zip(mine, reference))


def test_load_wav_scp_matches(recordings):
    scp, seg = recordings
    assert _same(kaldi.load_wav_scp(scp)["rec1"], kaldiio.load_wav_scp(scp)["rec1"])
    assert _same(
        kaldi.load_wav_scp(scp, segments=seg)["utt_a"],
        kaldiio.load_wav_scp(scp, segments=seg)["utt_a"],
    )


@pytest.mark.parametrize(
    "specifier",
    [
        "ark:a.ark",
        "scp:a.scp",
        "ark,scp:a.ark,b.scp",
        "ark,t:a.txt",
        "ark,scp,t:a.ark,b.scp",
        "ark,f:a.ark",
    ],
)
def test_parse_specifier_matches(specifier):
    assert kaldi.parse_specifier(specifier) == kaldiio.parse_specifier(specifier)


def test_separator_matches(tmp_path, feats):
    scp = str(tmp_path / "a.scp")
    kaldi.save_ark(str(tmp_path / "a.ark"), {"u1": feats}, scp=scp)
    tabbed = tmp_path / "tab.scp"
    tabbed.write_text(open(scp).read().replace(" ", "\t", 1))
    assert _same(
        kaldi.load_scp(str(tabbed), separator="\t")["u1"],
        kaldiio.load_scp(str(tabbed), separator="\t")["u1"],
    )


def test_load_mat_fd_dict_matches(tmp_path, feats):
    scp = str(tmp_path / "a.scp")
    kaldi.save_ark(str(tmp_path / "a.ark"), {"u1": feats}, scp=scp)
    name = open(scp).read().split()[1]
    mine, reference = {}, {}
    assert _same(kaldi.load_mat(name, fd_dict=mine), kaldiio.load_mat(name, fd_dict=reference))
    assert sorted(mine) == sorted(reference) and len(mine) == 1

    # A second read must reuse that handle, not open another one under the same
    # key -- which would leak the first while leaving the dict the same size.
    opened = next(iter(mine.values()))
    kaldi.load_mat(name, fd_dict=mine)
    assert next(iter(mine.values())) is opened

    for fd in mine.values():
        fd.close()
    for fd in reference.values():
        fd.close()
    assert opened.closed


@pytest.mark.parametrize(
    "array",
    [
        np.array([5, 12, 3, 0], dtype=np.int32),
        np.array([5, 12, 3, 0], dtype=np.int64),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        np.array([0, 255], dtype=np.uint8),
        np.array([1.5, 2.0], dtype=np.float32),
    ],
)
def test_text_ark_bytes_match_for_every_dtype(tmp_path, array):
    """Integers must not pick up a decimal point on the way out."""

    def write(mod, p):
        with mod.WriteHelper("ark,t:" + p) as w:
            w["u"] = array

    ours, theirs = _both(tmp_path, "t{}".format(array.dtype), write)
    assert open(ours).read() == open(theirs).read()


@pytest.mark.parametrize(
    "text",
    [
        "u  [ 5 12 3 0 ]\n",
        "u  [ -5 12 ]\n",
        "u  [\n  1 2 \n  3 4 ]\n",
        "u  [\n  1 2.5 \n  3 4 ]\n",
    ],
)
def test_text_ark_read_dtype_matches(tmp_path, text):
    p = tmp_path / "t.txt"
    p.write_text(text)
    mine = dict(kaldi.load_ark(str(p)))["u"]
    reference = dict(kaldiio.load_ark(str(p)))["u"]
    assert mine.dtype == reference.dtype
    assert np.array_equal(mine, reference)


def test_signed_integer_vector_matches(tmp_path):
    """kaldiio reads -5 correctly; an unsigned read gives 4294967291."""
    alignment = np.array([-5, 0, 7, -(2**31), 2**31 - 1], dtype=np.int32)
    ours, theirs = _both(tmp_path, "ali", lambda mod, p: mod.save_ark(p, {"u": alignment}))
    assert open(ours, "rb").read() == open(theirs, "rb").read()
    assert np.array_equal(dict(kaldi.load_ark(theirs))["u"], alignment)
    assert np.array_equal(dict(kaldiio.load_ark(ours))["u"], alignment)

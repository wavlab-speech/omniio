"""Tests for the Kaldi ark/scp compatibility layer.

The golden byte strings below were checked against Kaldi's own on-disk layout;
``test_interop.py`` re-checks them against ``kaldiio`` when it is installed.
"""

import gzip
import io
import os
import shutil
import sys

import numpy as np
import pytest

from omniio import kaldi


@pytest.fixture
def feats():
    rng = np.random.RandomState(0)
    return (rng.randn(30, 13) * 4).astype(np.float32)


@pytest.fixture
def wav():
    rng = np.random.RandomState(1)
    return (rng.uniform(-1, 1, 8000) * 32767).astype(np.int16)


# --------------------------------------------------------------------------
# on-disk layout
# --------------------------------------------------------------------------


def test_float_matrix_layout(tmp_path):
    p = str(tmp_path / "a.ark")
    kaldi.save_ark(p, {"a": np.arange(12, dtype=np.float32).reshape(3, 4)})
    raw = open(p, "rb").read()
    assert raw[:2] == b"a "
    assert raw[2:4] == b"\x00B"
    assert raw[4:7] == b"FM "
    assert raw[7:12] == b"\x04\x03\x00\x00\x00"  # \4 + int32 rows
    assert raw[12:17] == b"\x04\x04\x00\x00\x00"  # \4 + int32 cols
    assert len(raw) == 17 + 12 * 4


def test_tokens_follow_dtype_and_rank(tmp_path):
    cases = {
        "fm": (np.zeros((2, 3), np.float32), b"FM "),
        "fv": (np.zeros(3, np.float32), b"FV "),
        "dm": (np.zeros((2, 3), np.float64), b"DM "),
        "dv": (np.zeros(3, np.float64), b"DV "),
    }
    for name, (array, token) in cases.items():
        p = str(tmp_path / (name + ".ark"))
        kaldi.save_ark(p, {"a": array})
        assert open(p, "rb").read()[4:7] == token, name


def test_int32_vector_has_no_token(tmp_path):
    p = str(tmp_path / "ali.ark")
    kaldi.save_ark(p, {"a": np.arange(3, dtype=np.int32)})
    raw = open(p, "rb").read()
    assert raw == (
        b"a \x00B"
        b"\x04\x03\x00\x00\x00"
        b"\x04\x00\x00\x00\x00"
        b"\x04\x01\x00\x00\x00"
        b"\x04\x02\x00\x00\x00"
    )
    assert np.array_equal(dict(kaldi.load_ark(p))["a"], np.arange(3))


def test_scp_offset_points_past_the_key(tmp_path):
    p = str(tmp_path / "a.ark")
    s = str(tmp_path / "a.scp")
    kaldi.save_ark(p, {"a": np.zeros((2, 2), np.float32), "bb": np.zeros(2, np.float32)})
    kaldi.save_ark(p, {"a": np.zeros((2, 2), np.float32), "bb": np.zeros(2, np.float32)}, scp=s)
    lines = open(s).read().splitlines()
    assert lines[0] == "a {}:2".format(p)
    # 2 (key) + 2 (marker) + 3 (token) + 10 (shape) + 16 (data) = 33, then "bb ".
    assert lines[1] == "bb {}:36".format(p)


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "array",
    [
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(5, dtype=np.float32),
        np.arange(6, dtype=np.float64).reshape(2, 3),
        np.arange(4, dtype=np.float64),
        np.arange(7, dtype=np.int32),
    ],
)
def test_ark_roundtrip_preserves_dtype_and_shape(tmp_path, array):
    p = str(tmp_path / "a.ark")
    kaldi.save_ark(p, {"a": array})
    got = dict(kaldi.load_ark(p))["a"]
    assert got.dtype == array.dtype
    assert np.array_equal(got, array)


def test_scp_roundtrip(tmp_path, feats):
    p = str(tmp_path / "a.ark")
    s = str(tmp_path / "a.scp")
    kaldi.save_ark(p, {"u1": feats, "u2": feats[:5]}, scp=s)
    loader = kaldi.load_scp(s)
    assert list(loader.keys()) == ["u1", "u2"]
    assert len(loader) == 2
    assert "u1" in loader and "nope" not in loader
    assert np.array_equal(loader["u1"], feats)
    assert np.array_equal(loader["u2"], feats[:5])


def test_load_scp_max_cache_fd_reuses_descriptors(tmp_path, feats):
    p = str(tmp_path / "a.ark")
    s = str(tmp_path / "a.scp")
    kaldi.save_ark(p, {"u{}".format(i): feats[: i + 1] for i in range(5)}, scp=s)
    loader = kaldi.load_scp(s, max_cache_fd=2)
    for _ in range(3):
        for i in range(5):
            assert np.array_equal(loader["u{}".format(i)], feats[: i + 1])
    loader.close()


def test_load_mat_with_and_without_offset(tmp_path, feats):
    p = str(tmp_path / "a.ark")
    s = str(tmp_path / "a.scp")
    kaldi.save_ark(p, {"u1": feats}, scp=s)
    name = open(s).read().split()[1]
    assert np.array_equal(kaldi.load_mat(name), feats)

    single = str(tmp_path / "cmvn.mat")
    kaldi.save_mat(single, feats)
    assert np.array_equal(kaldi.load_mat(single), feats)


def test_append_continues_offsets(tmp_path, feats):
    p = str(tmp_path / "a.ark")
    s = str(tmp_path / "a.scp")
    with open(p, "wb") as ark, open(s, "w") as scp:
        for i in range(3):
            kaldi.save_ark(ark, {"u{}".format(i): feats[: i + 1]}, scp=scp, append=True)
    loader = kaldi.load_scp(s)
    for i in range(3):
        assert np.array_equal(loader["u{}".format(i)], feats[: i + 1])


# --------------------------------------------------------------------------
# compression
# --------------------------------------------------------------------------


# Worst-case absolute error each method may introduce on the ``feats`` fixture.
# 1/2 quantise to one byte per column against percentile buckets, 3 to two
# bytes over the global range, 4 to a signed integer, 5 to one byte over the
# global range.
@pytest.mark.parametrize(
    "method,tolerance", [(1, 0.07), (2, 0.07), (3, 1e-3), (4, 0.51), (5, 0.05)]
)
def test_compressed_roundtrip(tmp_path, feats, method, tolerance):
    p = str(tmp_path / "c.ark")
    with kaldi.WriteHelper("ark:" + p, compression_method=method) as w:
        w["u1"] = feats
    got = dict(kaldi.load_ark(p))["u1"]
    assert got.shape == feats.shape
    assert got.dtype == np.float32
    assert np.abs(got - feats).max() <= tolerance


def test_per_column_headers_beat_a_global_byte_range(tmp_path, feats):
    """The point of format 1: one byte per value, but scaled per column.

    Compared on RMSE, not peak error: the percentile buckets deliberately
    spend their resolution on the bulk of each column and let the tails be
    coarse.
    """
    want = feats * np.arange(1, feats.shape[1] + 1, dtype=np.float32)
    errors = {}
    for method in (2, 5):
        p = str(tmp_path / "c{}.ark".format(method))
        with kaldi.WriteHelper("ark:" + p, compression_method=method) as w:
            w["u1"] = want
        got = dict(kaldi.load_ark(p))["u1"]
        errors[method] = float(np.sqrt(((got - want) ** 2).mean()))
    assert errors[2] < errors[5]


def test_compression_method_selects_token(tmp_path, feats):
    tokens = {}
    for method, expected in [(1, b"CM "), (2, b"CM "), (3, b"CM2"), (5, b"CM3")]:
        p = str(tmp_path / "c{}.ark".format(method))
        with kaldi.WriteHelper("ark:" + p, compression_method=method) as w:
            w["u1"] = feats
        tokens[method] = open(p, "rb").read()[5:8]  # "u1 " + \0B
        assert tokens[method] == expected, method


def test_automatic_method_falls_back_below_nine_rows(tmp_path):
    rng = np.random.RandomState(2)
    short = rng.randn(4, 6).astype(np.float32)
    p = str(tmp_path / "s.ark")
    with kaldi.WriteHelper("ark:" + p, compression_method=1) as w:
        w["u1"] = short
    assert open(p, "rb").read()[5:9] == b"CM2 "


def test_fixed_range_methods_clip_out_of_range_values(tmp_path):
    array = np.array([[-5.0, 0.5, 300.0]], dtype=np.float32)
    p = str(tmp_path / "z.ark")
    with kaldi.WriteHelper("ark:" + p, compression_method=7) as w:
        w["u1"] = array
    got = dict(kaldi.load_ark(p))["u1"]
    assert got[0, 0] == pytest.approx(0.0)
    assert got[0, 2] == pytest.approx(1.0)


def test_compression_rejects_bad_method(feats):
    from omniio.kaldi import compression

    with pytest.raises(ValueError, match="1..7"):
        compression.compress(feats, 99)


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


def test_wave_holder_ark_is_bare_riff(tmp_path, wav):
    p = str(tmp_path / "w.ark")
    s = str(tmp_path / "w.scp")
    kaldi.save_ark(p, {"r1": (16000, wav)}, scp=s)
    raw = open(p, "rb").read()
    assert raw[:2] == b"r1"[:2]
    assert raw[3:7] == b"RIFF"

    rate, array = kaldi.load_scp(s)["r1"]
    assert rate == 16000
    assert array.dtype == np.int16
    assert np.array_equal(array, wav)


def test_extended_audio_ark(tmp_path, wav):
    p = str(tmp_path / "e.ark")
    s = str(tmp_path / "e.scp")
    with open(p, "wb") as ark, open(s, "w") as scp:
        kaldi.save_ark(
            ark,
            {"r1": (wav, 16000)},
            scp=scp,
            append=True,
            write_function="soundfile",
            write_kwargs={"format": "flac", "subtype": None},
        )
    raw = open(p, "rb").read()
    assert raw[3:8] == b"AUDIO"
    width = raw[8]
    length = int.from_bytes(raw[9 : 9 + width], "little")
    assert len(raw) == 3 + 5 + 1 + width + length
    assert raw[9 + width : 13 + width] == b"fLaC"

    rate, array = kaldi.load_scp(s)["r1"]
    assert rate == 16000
    assert np.allclose(array, wav / 32768.0, atol=1e-4)


def test_extended_audio_length_prefix_grows(tmp_path):
    rng = np.random.RandomState(3)
    big = (rng.uniform(-1, 1, 200000) * 32767).astype(np.int16)
    p = str(tmp_path / "big.ark")
    with open(p, "wb") as ark:
        kaldi.save_ark(
            ark,
            {"r1": (big, 16000)},
            append=True,
            write_function="soundfile",
            write_kwargs={"format": "wav"},
        )
    raw = open(p, "rb").read()
    assert raw[8] == 3  # a 400 kB payload needs a three-byte length


def test_rate_and_array_order_is_detected(tmp_path, wav):
    a = str(tmp_path / "a.ark")
    b = str(tmp_path / "b.ark")
    kaldi.save_ark(a, {"r1": (16000, wav)})
    kaldi.save_ark(b, {"r1": (wav, 16000)})
    assert open(a, "rb").read() == open(b, "rb").read()


def test_sound_entry_rejects_two_arrays(tmp_path, wav):
    with pytest.raises(ValueError, match="rate, array"):
        kaldi.save_ark(str(tmp_path / "x.ark"), {"r1": (wav, wav)})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_write_helper_ark_scp(tmp_path, feats):
    a = str(tmp_path / "f.ark")
    s = str(tmp_path / "f.scp")
    with kaldi.WriteHelper("ark,scp:{},{}".format(a, s)) as w:
        w["u1"] = feats
        w["u2"] = feats[:3]
    with kaldi.ReadHelper("scp:" + s) as r:
        got = dict(r)
    assert np.array_equal(got["u1"], feats)
    assert np.array_equal(got["u2"], feats[:3])


def test_read_helper_over_ark(tmp_path, feats):
    a = str(tmp_path / "f.ark")
    kaldi.save_ark(a, {"u1": feats, "u2": feats[:3]})
    with kaldi.ReadHelper("ark:" + a) as r:
        keys = [k for k, _ in r]
    assert keys == ["u1", "u2"]


def test_write_helper_rejects_use_after_close(tmp_path, feats):
    w = kaldi.WriteHelper("ark:" + str(tmp_path / "f.ark"))
    w["u1"] = feats
    w.close()
    with pytest.raises(ValueError, match="closed"):
        w["u2"] = feats


def test_text_ark_roundtrip(tmp_path, feats):
    p = str(tmp_path / "t.ark")
    with kaldi.WriteHelper("ark,t:" + p) as w:
        w["u1"] = feats
        w["u2"] = feats[0]
    text = open(p).read()
    assert text.startswith("u1  [\n")
    got = dict(kaldi.load_ark(p))
    assert got["u1"].shape == feats.shape
    assert got["u2"].shape == feats[0].shape
    assert np.allclose(got["u1"], feats)


def test_text_ark_single_row_stays_a_matrix(tmp_path):
    p = str(tmp_path / "t.ark")
    with kaldi.WriteHelper("ark,t:" + p) as w:
        w["u1"] = np.arange(3, dtype=np.float32).reshape(1, 3)
    assert dict(kaldi.load_ark(p))["u1"].shape == (1, 3)


def test_segments(tmp_path, wav):
    p = str(tmp_path / "s.ark")
    s = str(tmp_path / "s.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(p, {"rec1": (16000, wav)}, scp=s)
    seg.write_text("utt_a rec1 0.0 0.25\nutt_b rec1 0.25 0.5\n")

    with kaldi.ReadHelper("scp:" + s, segments=str(seg)) as r:
        got = list(r)
    assert [k for k, _ in got] == ["utt_a", "utt_b"]
    assert np.array_equal(got[0][1][1], wav[:4000])
    assert np.array_equal(got[1][1][1], wav[4000:8000])


def test_segments_requires_scp(tmp_path):
    with pytest.raises(ValueError, match="scp rspecifier"):
        kaldi.ReadHelper("ark:x.ark", segments="segments")


def test_segments_reports_missing_recording(tmp_path, wav):
    p = str(tmp_path / "s.ark")
    s = str(tmp_path / "s.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(p, {"rec1": (16000, wav)}, scp=s)
    seg.write_text("utt_a other_rec 0.0 0.25\n")
    with pytest.raises(kaldi.ReadError, match="other_rec"):
        list(kaldi.ReadHelper("scp:" + s, segments=str(seg)))


# --------------------------------------------------------------------------
# specifiers and streams
# --------------------------------------------------------------------------


def test_parse_wspecifier():
    assert kaldi.parse_wspecifier("ark:a.ark") == {"ark": "a.ark"}
    assert kaldi.parse_wspecifier("ark,scp:a.ark,b.scp") == {
        "ark": "a.ark",
        "scp": "b.scp",
    }
    assert kaldi.parse_wspecifier("ark,t:a.txt") == {"ark": "a.txt", "t": None}
    assert kaldi.parse_wspecifier("ark:| gzip -c > a.gz") == {"ark": "| gzip -c > a.gz"}


def test_parse_wspecifier_errors():
    with pytest.raises(ValueError, match="Invalid wspecifier"):
        kaldi.parse_wspecifier("a.ark")
    with pytest.raises(ValueError, match="must contain 'ark'"):
        kaldi.parse_wspecifier("scp:a.scp")
    with pytest.raises(ValueError, match="Unsupported"):
        kaldi.parse_wspecifier("ark,zzz:a.ark")
    with pytest.raises(ValueError, match="file"):
        kaldi.parse_wspecifier("ark,scp:a.ark")


def test_parse_rspecifier():
    assert kaldi.parse_rspecifier("scp:a.scp") == ("scp", "a.scp")
    assert kaldi.parse_rspecifier("ark,s,cs:a.ark") == ("ark", "a.ark")
    assert kaldi.parse_rspecifier("ark:gunzip -c a.gz |") == (
        "ark",
        "gunzip -c a.gz |",
    )
    with pytest.raises(ValueError, match="exactly one"):
        kaldi.parse_rspecifier("t:a.ark")


def test_parse_extended_filename():
    assert kaldi.parse_extended_filename("a.ark:42") == ("a.ark", 42)
    assert kaldi.parse_extended_filename("a.ark") == ("a.ark", None)
    assert kaldi.parse_extended_filename("cat a.wav |") == ("cat a.wav |", None)


def _py(snippet, *args):
    """A shell command running ``snippet`` in this interpreter.

    The pipe tests need commands that work under both ``sh`` and ``cmd.exe``,
    so they go through ``sys.executable`` rather than through coreutils, which
    Windows does not have. Keep snippets free of double quotes and of shell
    metacharacters -- only the outer quoting is portable.
    """
    quoted = "".join(' "{}"'.format(a) for a in args)
    return '"{}" -c "{}"{}'.format(sys.executable, snippet, quoted)


def test_open_like_kaldi_read_pipe():
    emit = _py("import sys; sys.stdout.write('hello')")
    with kaldi.open_like_kaldi(emit + " |", "r") as f:
        assert f.read() == "hello"


def test_open_like_kaldi_write_pipe(tmp_path):
    out = tmp_path / "out.txt"
    sink = _py("import sys; open(sys.argv[1],'wb').write(sys.stdin.buffer.read())", out)
    with kaldi.open_like_kaldi("| " + sink, "w") as f:
        f.write("world")
    assert out.read_text() == "world"


def test_open_like_kaldi_propagates_failure():
    fail = _py("import sys; sys.exit(3)")
    with pytest.raises(IOError, match="status 3"):
        with kaldi.open_like_kaldi(fail + " |", "rb") as f:
            f.read()


def test_open_like_kaldi_gzip(tmp_path):
    p = str(tmp_path / "a.gz")
    with kaldi.open_like_kaldi(p, "wb") as f:
        f.write(b"payload")
    with kaldi.open_like_kaldi(p, "rb") as f:
        assert f.read() == b"payload"


def test_open_like_kaldi_does_not_close_caller_streams(tmp_path):
    buf = io.BytesIO()
    with kaldi.open_like_kaldi(buf, "wb") as f:
        f.write(b"x")
    assert not buf.closed


def test_ark_through_a_pipe(tmp_path, feats):
    p = str(tmp_path / "a.ark")
    kaldi.save_ark(p, {"u1": feats})
    with open(p, "rb") as plain, gzip.open(p + ".gz", "wb") as packed:
        shutil.copyfileobj(plain, packed)

    unpack = _py(
        "import gzip,shutil,sys; "
        "shutil.copyfileobj(gzip.open(sys.argv[1],'rb'), sys.stdout.buffer)",
        p + ".gz",
    )
    with kaldi.ReadHelper("ark:" + unpack + " |") as r:
        got = dict(r)
    assert np.array_equal(got["u1"], feats)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_truncated_archive_is_reported(tmp_path, feats):
    p = tmp_path / "a.ark"
    kaldi.save_ark(str(p), {"u1": feats})
    p.write_bytes(p.read_bytes()[:-40])
    with pytest.raises(kaldi.ReadError, match="Unexpected end"):
        list(kaldi.load_ark(str(p)))


def test_unknown_token_is_reported(tmp_path):
    p = tmp_path / "a.ark"
    p.write_bytes(b"a \x00BZZ \x04\x01\x00\x00\x00")
    with pytest.raises(kaldi.ReadError, match="Unsupported"):
        list(kaldi.load_ark(str(p)))


def test_malformed_scp_line_is_reported(tmp_path):
    s = tmp_path / "a.scp"
    s.write_text("only_a_key\n")
    with pytest.raises(kaldi.ReadError, match="expected"):
        kaldi.load_scp(str(s))


def test_keys_with_whitespace_are_rejected(tmp_path, feats):
    with pytest.raises(ValueError, match="whitespace"):
        kaldi.save_ark(str(tmp_path / "a.ark"), {"bad key": feats})


def test_three_dim_array_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ndim"):
        kaldi.save_ark(str(tmp_path / "a.ark"), {"u1": np.zeros((2, 2, 2), np.float32)})


# --------------------------------------------------------------------------
# import paths
# --------------------------------------------------------------------------


def test_public_import_paths_all_resolve():
    """``omniio.kaldi`` is the stable path; the code lives in ``omniio.tools``."""
    import importlib

    from omniio import kaldi as attr_import
    from omniio.kaldi import ReadHelper as from_import

    assert importlib.import_module("omniio.kaldi") is attr_import
    assert from_import is attr_import.ReadHelper

    real = importlib.import_module("omniio.tools.kaldi")
    assert real is attr_import, "both names must refer to one module object"

    assert importlib.import_module("omniio.kaldi.compression") is importlib.import_module(
        "omniio.tools.kaldi.compression"
    )


def test_importing_omniio_does_not_pull_in_the_subpackages():
    """The aliases are lazy, so ``import omniio`` stays cheap."""
    import subprocess
    import sys

    code = "import sys; import omniio; " "print('omniio.tools.kaldi' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_unknown_attribute_still_raises():
    import omniio

    with pytest.raises(AttributeError, match="no attribute"):
        omniio.definitely_not_a_module


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------


def test_segments_end_of_recording(tmp_path, wav):
    """Kaldi writes -1 for "to the end of the recording".

    Exactly 0 is not that sentinel: extract-segments rejects such a line as an
    invalid segment, and ordinary indexing gives the empty slice.
    """
    ark = str(tmp_path / "s.ark")
    scp = str(tmp_path / "s.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(ark, {"rec1": (16000, wav)}, scp=scp)
    seg.write_text("utt_a rec1 0.1 -1\nutt_b rec1 0.2 0\nutt_c rec1 0.1 0.2\n")

    with kaldi.ReadHelper("scp:" + scp, segments=str(seg)) as r:
        got = dict(r)
    assert np.array_equal(got["utt_a"][1], wav[1600:])
    assert got["utt_b"][1].size == 0
    assert np.array_equal(got["utt_c"][1], wav[1600:3200])


def test_two_dim_integer_array_is_refused(tmp_path):
    """Silently writing it as float32 would lose exactness above 2**24."""
    array = np.array([[16777217, 16777219]], dtype=np.int32)
    with pytest.raises(ValueError, match="no integer matrix type"):
        kaldi.save_ark(str(tmp_path / "a.ark"), {"u": array})

    # 1-dim integer arrays are still fine: those Kaldi does have a type for.
    kaldi.save_ark(str(tmp_path / "b.ark"), {"u": array[0]})
    assert np.array_equal(dict(kaldi.load_ark(str(tmp_path / "b.ark")))["u"], array[0])


@pytest.mark.parametrize("endian", ["<", ">"])
def test_endian_roundtrip(tmp_path, feats, endian):
    p = str(tmp_path / "e.ark")
    kaldi.save_ark(p, {"u": feats, "v": np.arange(4, dtype=np.int32)}, endian=endian)
    got = dict(kaldi.load_ark(p, endian=endian))
    assert np.array_equal(got["u"], feats)
    assert np.array_equal(got["v"], np.arange(4))


def test_stale_scp_offset_raises_read_error(tmp_path, feats):
    """A wrong offset must not surface as UnicodeDecodeError or MemoryError."""
    p = str(tmp_path / "a.ark")
    kaldi.save_ark(p, {"u1": feats, "u2": feats})
    size = os.path.getsize(p)
    for offset in (7, 40, size // 2):
        with pytest.raises(kaldi.ReadError):
            kaldi.load_mat("{}:{}".format(p, offset))


def test_absurd_dimensions_raise_read_error(tmp_path):
    p = tmp_path / "huge.ark"
    dim = (2**30).to_bytes(4, "little")
    p.write_bytes(b"u \x00BFM \x04" + dim + b"\x04" + dim)
    with pytest.raises(kaldi.ReadError, match="Unexpected end"):
        kaldi.load_mat("{}:2".format(p))

    q = tmp_path / "huge_vec.ark"
    q.write_bytes(b"u \x00B\x04" + (2**31 - 1).to_bytes(4, "little"))
    with pytest.raises(kaldi.ReadError):
        kaldi.load_mat("{}:2".format(q))


def test_scp_for_an_unnamed_stream_is_refused(feats):
    """Every line would point at a path that cannot be opened."""
    with pytest.raises(ValueError, match="no file name"):
        kaldi.save_ark(io.BytesIO(), {"u1": feats}, scp=io.StringIO())

    # Without an scp there is nothing to point anywhere, so this stays allowed.
    buf = io.BytesIO()
    kaldi.save_ark(buf, {"u1": feats})
    assert buf.getvalue().startswith(b"u1 \x00BFM ")


def test_non_finite_values_are_refused_by_compression(tmp_path, feats):
    for bad in (np.nan, np.inf, -np.inf):
        broken = feats.copy()
        broken[3, 2] = bad
        with pytest.raises(ValueError, match="NaN or inf"):
            with kaldi.WriteHelper("ark:" + str(tmp_path / "c.ark"), compression_method=2) as w:
                w["u"] = broken


def test_alias_does_not_shadow_an_existing_package(monkeypatch):
    """An _ALIASES entry added before its package moves must not break it."""
    import importlib

    import omniio

    monkeypatch.setitem(omniio._ALIASES, "omniio.text", "omniio.modalities.text")
    for name in [n for n in list(sys.modules) if n.startswith("omniio.text")]:
        monkeypatch.delitem(sys.modules, name)

    # The target does not exist, so the real omniio/text/ must still be found.
    assert importlib.import_module("omniio.text.read") is not None


@pytest.mark.parametrize("method", [1, 2, 3, 4, 5, 6, 7])
def test_non_finite_refused_for_every_method(tmp_path, feats, method):
    """The fixed-range methods never look at the data's min/max, so the check
    has to sit in compress() rather than in the global-header branch."""
    for bad in (np.nan, np.inf, -np.inf):
        broken = feats.copy()
        broken[3, 2] = bad
        with pytest.raises(ValueError, match="NaN or inf"):
            with kaldi.WriteHelper(
                "ark:" + str(tmp_path / "c.ark"), compression_method=method
            ) as w:
                w["u"] = broken


def test_two_dim_integer_array_is_refused_under_compression(tmp_path):
    """compress() casts to float32, so the guard has to run before it."""
    array = np.array([[16777217, 16777219]], dtype=np.int32)
    with pytest.raises(ValueError, match="no integer matrix type"):
        with kaldi.WriteHelper("ark:" + str(tmp_path / "c.ark"), compression_method=2) as w:
            w["u"] = array


@pytest.mark.parametrize("target", ["-", "| cat > /dev/null", "gzip -c > x.gz |"])
def test_scp_for_a_non_seekable_target_is_refused(tmp_path, feats, target):
    """An scp entry is read back by reopening the path and seeking."""
    scp = str(tmp_path / "f.scp")
    with pytest.raises(ValueError, match="non-seekable"):
        kaldi.save_ark(target, {"u1": feats}, scp=scp)

    # Without an scp there is nothing to point anywhere, so a pipe stays fine.
    out = tmp_path / "piped.ark"
    sink = _py("import sys; open(sys.argv[1],'wb').write(sys.stdin.buffer.read())", out)
    kaldi.save_ark("| " + sink, {"u1": feats})
    assert np.array_equal(dict(kaldi.load_ark(str(out)))["u1"], feats)


def test_reads_do_not_seek(tmp_path, feats):
    """Bounding a read by seeking to the end would defeat buffering on a plain
    file and decompress the remainder on a gzipped one."""
    kaldi.save_ark(str(tmp_path / "a.ark"), {"u{}".format(i): feats for i in range(5)})

    seeks = []

    class NoSeek(io.BufferedReader):
        def seek(self, *args):
            seeks.append(args)
            return super().seek(*args)

    with NoSeek(open(str(tmp_path / "a.ark"), "rb")) as fd:
        got = dict(kaldi.load_ark(fd))
    assert len(got) == 5
    assert seeks == [], "sequential reading must not seek"


def test_absurd_length_does_not_allocate(tmp_path):
    """A stale offset can ask for petabytes; the read is chunked so the first
    short read rejects it."""
    p = tmp_path / "huge.ark"
    dim = (2**40).to_bytes(8, "little")
    p.write_bytes(b"u \x00BFV \x08" + dim + b"payload")
    with pytest.raises(kaldi.ReadError, match="Unexpected end"):
        kaldi.load_mat("{}:2".format(p))


# --------------------------------------------------------------------------
# x-vectors, as ESPnet's TTS recipe produces and consumes them
#
# egs2/TEMPLATE/tts1/tts.sh passes "<tag>.scp,spembs,kaldi_ark", which reaches
# load_scp. The archives come from either Kaldi itself (spk_embed_tool=kaldi ->
# nnet3-xvector-compute) or from pyscripts/utils/extract_spk_embed.py, so both
# writers are covered here.
# --------------------------------------------------------------------------


def _kaldi_native_vector_ark(path, vectors):
    """Write vectors exactly as Kaldi's BaseFloatVectorWriter does.

    Built from the format rather than from this module's writer, so the test
    still fails if both sides drift together.
    """
    scp = path + ".scp"
    with open(path, "wb") as ark, open(scp, "w") as index:
        for key, vec in vectors.items():
            offset = ark.tell() + len(key) + 1
            ark.write(
                key.encode()
                + b" \x00BFV \x04"
                + np.int32(vec.size).tobytes()
                + vec.astype("<f4").tobytes()
            )
            index.write("{} {}:{}\n".format(key, path, offset))
    return scp


@pytest.fixture
def xvectors():
    rng = np.random.RandomState(11)
    return {"utt{}".format(i): rng.randn(512).astype(np.float32) for i in range(3)}


def test_xvector_from_kaldi_is_read_verbatim(tmp_path, xvectors):
    scp = _kaldi_native_vector_ark(str(tmp_path / "xvector.ark"), xvectors)
    loader = kaldi.load_scp(scp)

    assert list(loader.keys()) == ["utt0", "utt1", "utt2"]
    assert len(loader) == 3 and "utt1" in loader
    for key, want in xvectors.items():
        got = loader[key]
        assert got.dtype == np.float32 and got.shape == (512,)
        assert np.array_equal(got, want)


def test_xvector_from_extract_spk_embed(tmp_path, xvectors):
    """Per-utterance embeddings are 1-dim; the per-speaker mean keeps the
    extractor's leading axis and lands as a (1, D) matrix."""
    utt_ark = str(tmp_path / "xvector.ark")
    spk_ark = str(tmp_path / "spk_xvector.ark")
    with (
        kaldi.WriteHelper("ark,scp:{0},{0}.scp".format(utt_ark)) as utt,
        kaldi.WriteHelper("ark,scp:{0},{0}.scp".format(spk_ark)) as spk,
    ):
        for key, vec in xvectors.items():
            utt[key] = np.squeeze(vec)
        spk["spk1"] = np.mean(np.stack([v[None, :] for v in xvectors.values()], 0), 0)

    # "utt0 " is five bytes, then the two-byte binary marker.
    assert open(utt_ark, "rb").read()[5:10] == b"\x00BFV "
    assert open(spk_ark, "rb").read()[5:10] == b"\x00BFM "

    per_utt = kaldi.load_scp(utt_ark + ".scp")
    for key, want in xvectors.items():
        assert np.array_equal(per_utt[key], want)

    per_spk = kaldi.load_scp(spk_ark + ".scp")["spk1"]
    assert per_spk.shape == (1, 512) and per_spk.dtype == np.float32
    assert np.allclose(per_spk[0], np.mean(list(xvectors.values()), 0))


def test_xvector_in_kaldi_text_mode(tmp_path, xvectors):
    """copy-vector --binary=false, which some recipes leave enabled."""
    ark = str(tmp_path / "t.ark")
    scp = str(tmp_path / "t.scp")
    with open(ark, "wb") as f, open(scp, "w") as index:
        for key, vec in xvectors.items():
            offset = f.tell() + len(key) + 1
            body = " [ " + " ".join(repr(float(x)) for x in vec) + " ]\n"
            f.write(key.encode() + b" " + body.encode())
            index.write("{} {}:{}\n".format(key, ark, offset))

    loader = kaldi.load_scp(scp)
    for key, want in xvectors.items():
        assert loader[key].shape == (512,)
        assert np.allclose(loader[key], want, atol=1e-6)


def test_xvector_whole_ark_iteration(tmp_path, xvectors):
    """espnet2/sds/tts/espnet_tts.py reads the ark rather than the scp."""
    ark = str(tmp_path / "x.ark")
    kaldi.save_ark(ark, xvectors)
    got = {k: v for k, v in kaldi.load_ark(ark)}
    assert sorted(got) == sorted(xvectors)
    assert all(np.array_equal(got[k], v) for k, v in xvectors.items())


@pytest.mark.parametrize("dim", [512, 192, 256])
def test_xvector_embedding_sizes(tmp_path, dim):
    """512 for kaldi/espnet, 192 for speechbrain ECAPA, 256 for rawnet."""
    vec = np.random.RandomState(dim).randn(dim).astype(np.float32)
    ark = str(tmp_path / "e.ark")
    kaldi.save_ark(ark, {"u": vec}, scp=ark + ".scp")
    assert np.array_equal(kaldi.load_scp(ark + ".scp")["u"], vec)


def test_xvector_float64_extractor_output(tmp_path):
    """An extractor that skips the float32 cast writes DV, not FV."""
    vec = np.random.RandomState(1).randn(512)
    ark = str(tmp_path / "d.ark")
    kaldi.save_ark(ark, {"u": vec}, scp=ark + ".scp")
    assert open(ark, "rb").read()[2:7] == b"\x00BDV "
    got = kaldi.load_scp(ark + ".scp")["u"]
    assert got.dtype == np.float64 and np.array_equal(got, vec)


def test_xvector_scp_random_access_with_fd_cache(tmp_path, xvectors):
    """The recipe reads spembs out of order, one utterance at a time."""
    scp = _kaldi_native_vector_ark(str(tmp_path / "xvector.ark"), xvectors)
    loader = kaldi.load_scp(scp, max_cache_fd=8)
    for _ in range(3):
        for key in reversed(list(xvectors)):
            assert np.array_equal(loader[key], xvectors[key])
    loader.close()


# --------------------------------------------------------------------------
# scp options: separator, segments, fd caches
# --------------------------------------------------------------------------


def test_scp_separator(tmp_path, feats):
    """An scp may use its own delimiter rather than whitespace."""
    ark = str(tmp_path / "a.ark")
    scp = str(tmp_path / "a.scp")
    kaldi.save_ark(ark, {"u1": feats}, scp=scp)
    key, value = open(scp).read().strip().split(None, 1)

    tabbed = tmp_path / "tab.scp"
    tabbed.write_text("{}\t{}\n".format(key, value))
    assert np.array_equal(kaldi.load_scp(str(tabbed), separator="\t")["u1"], feats)
    assert np.array_equal(list(kaldi.load_scp_sequential(str(tabbed), separator="\t"))[0][1], feats)


def test_scp_default_split_keeps_spaces_in_the_path(tmp_path, feats):
    """The default splits once, so a path may contain spaces."""
    directory = tmp_path / "with space"
    directory.mkdir()
    ark = str(directory / "a.ark")
    scp = str(tmp_path / "a.scp")
    kaldi.save_ark(ark, {"u1": feats}, scp=scp)
    assert " " in open(scp).read().split(None, 1)[1]
    assert np.array_equal(kaldi.load_scp(scp)["u1"], feats)


@pytest.fixture
def segmented(tmp_path, wav):
    ark = str(tmp_path / "w.ark")
    scp = str(tmp_path / "w.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(ark, {"rec1": (16000, wav), "rec2": (16000, wav[::-1].copy())}, scp=scp)
    seg.write_text("utt_a rec1 0.1 0.2\nutt_b rec2 0.2 -1\nutt_c rec1 0.1 0\n")
    return scp, str(seg), wav


def test_load_scp_with_segments(segmented):
    scp, seg, wav = segmented
    loader = kaldi.load_scp(scp, segments=seg)

    assert list(loader.keys()) == ["utt_a", "utt_b", "utt_c"]
    assert len(loader) == 3 and "utt_b" in loader and "rec1" not in loader

    assert np.array_equal(loader["utt_a"][1], wav[1600:3200])
    assert np.array_equal(loader["utt_b"][1], wav[::-1][3200:])
    assert loader["utt_c"][1].size == 0
    assert all(rate == 16000 for rate, _ in (loader[k] for k in loader))


def test_load_scp_sequential_with_segments(segmented):
    scp, seg, wav = segmented
    got = list(kaldi.load_scp_sequential(scp, segments=seg))
    assert [k for k, _ in got] == ["utt_a", "utt_b", "utt_c"]
    assert np.array_equal(got[0][1][1], wav[1600:3200])


def test_load_wav_scp(segmented):
    scp, seg, wav = segmented
    rate, array = kaldi.load_wav_scp(scp)["rec1"]
    assert rate == 16000 and np.array_equal(array, wav)
    assert np.array_equal(kaldi.load_wav_scp(scp, segments=seg)["utt_a"][1], wav[1600:3200])


def test_segments_reports_a_recording_missing_from_the_scp(tmp_path, wav):
    scp = str(tmp_path / "w.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(str(tmp_path / "w.ark"), {"rec1": (16000, wav)}, scp=scp)
    seg.write_text("utt_a nope 0.0 0.1\n")
    with pytest.raises(kaldi.ReadError, match="nope"):
        kaldi.load_scp(scp, segments=seg.as_posix())["utt_a"]


def test_segments_on_a_plain_array_is_rejected(tmp_path, feats):
    scp = str(tmp_path / "f.scp")
    seg = tmp_path / "segments"
    kaldi.save_ark(str(tmp_path / "f.ark"), {"rec1": feats}, scp=scp)
    seg.write_text("utt_a rec1 0.0 0.1\n")
    with pytest.raises(kaldi.ReadError, match="audio entries"):
        kaldi.load_scp(scp, segments=seg.as_posix())["utt_a"]


def test_load_mat_fd_dict_reuses_one_handle(tmp_path, feats):
    ark = str(tmp_path / "a.ark")
    scp = str(tmp_path / "a.scp")
    kaldi.save_ark(ark, {"u{}".format(i): feats[: i + 1] for i in range(4)}, scp=scp)
    names = dict(line.split(None, 1) for line in open(scp).read().splitlines())

    fds = {}
    for _ in range(3):
        for key, name in names.items():
            got = kaldi.load_mat(name, fd_dict=fds)
            assert np.array_equal(got, feats[: int(key[1]) + 1])
    assert list(fds) == [ark], "one archive should mean one handle"
    for fd in fds.values():
        fd.close()


def test_parse_specifier_reports_every_flag():
    assert kaldi.parse_specifier("ark:a.ark") == {
        "ark": "a.ark",
        "scp": None,
        "t": False,
        "o": False,
        "p": False,
        "f": False,
        "s": False,
        "cs": False,
    }
    both = kaldi.parse_specifier("ark,scp:a.ark,b.scp")
    assert (both["ark"], both["scp"]) == ("a.ark", "b.scp")
    assert kaldi.parse_specifier("scp:a.scp")["ark"] is None
    assert kaldi.parse_specifier("ark,t:a.txt")["t"] is True
    assert kaldi.parse_specifier("ark,f:a.ark")["f"] is True


def test_parse_specifier_errors():
    with pytest.raises(ValueError, match="Invalid specifier"):
        kaldi.parse_specifier("a.ark")
    with pytest.raises(ValueError, match="must contain"):
        kaldi.parse_specifier("t:a.ark")
    with pytest.raises(ValueError, match="file"):
        kaldi.parse_specifier("ark,scp:a.ark")


@pytest.mark.parametrize("specifier", ["ark,x:a.ark", "ark,ns:a.ark", "ark,bg:a.ark"])
def test_parse_specifier_rejects_unknown_options(specifier):
    with pytest.raises(ValueError, match="Unsupported"):
        kaldi.parse_specifier(specifier)


@pytest.mark.parametrize(
    "specifier",
    ["ark,ark:a.ark,b.ark", "scp,scp:a.scp,b.scp", "ark,scp,scp:a.ark,b.scp,c.scp"],
)
def test_parse_specifier_rejects_a_repeated_file_option(specifier):
    """Keeping only the last would drop the other file silently."""
    with pytest.raises(ValueError, match="more than once"):
        kaldi.parse_specifier(specifier)


def test_parse_specifier_tolerates_a_repeated_flag():
    assert kaldi.parse_specifier("ark,t,t:a.txt")["t"] is True


@pytest.mark.parametrize("wspecifier", ["ark,ark:a.ark,b.ark", "ark,scp,scp:a.ark,b.scp,c.scp"])
def test_parse_wspecifier_rejects_a_repeated_file_option(wspecifier):
    with pytest.raises(ValueError, match="more than once"):
        kaldi.parse_wspecifier(wspecifier)


# --------------------------------------------------------------------------
# text arks keep integers integral
#
# ESPnet writes k-means pseudo-labels with `ark,t:` and later parses the tokens
# back with int(). Writing 5 as "5.0" makes that fail with
# `ValueError: invalid literal for int() with base 10: '5.0'`.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
def test_text_ark_writes_integers_without_a_decimal_point(tmp_path, dtype):
    labels = np.array([5, 12, 3, 0], dtype=dtype)
    p = str(tmp_path / "labels.txt")
    with kaldi.WriteHelper("ark,t:" + p) as w:
        w["utt1"] = labels

    assert open(p).read() == "utt1  [ 5 12 3 0 ]\n"

    # What the recipe actually does with the file it just wrote.
    key, rest = open(p).read().split(None, 1)
    tokens = rest.strip().lstrip("[").rstrip("]").split()
    assert [int(t) for t in tokens] == [5, 12, 3, 0]


def test_text_ark_integer_matrix(tmp_path):
    p = str(tmp_path / "m.txt")
    with kaldi.WriteHelper("ark,t:" + p) as w:
        w["u"] = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert open(p).read() == "u  [\n  1 2 \n  3 4 ]\n"


def test_text_ark_floats_are_unaffected(tmp_path):
    p = str(tmp_path / "f.txt")
    with kaldi.WriteHelper("ark,t:" + p) as w:
        w["u"] = np.array([1.5, 2.0], dtype=np.float32)
    assert open(p).read() == "u  [ 1.5 2.0 ]\n"


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_text_ark_integer_vectors_round_trip(tmp_path, dtype):
    """A vector of whole numbers is a label sequence, so it comes back integral."""
    labels = np.array([5, 12, 3, 0], dtype=dtype)
    p = str(tmp_path / "labels.txt")
    with kaldi.WriteHelper("ark,t:" + p) as w:
        w["u"] = labels
    got = dict(kaldi.load_ark(p))["u"]
    assert got.dtype == np.int32
    assert np.array_equal(got, labels)


def test_text_ark_reader_dtype_rules(tmp_path):
    """Vectors follow their contents; matrices are always float, as in Kaldi."""
    cases = {
        "ints.txt": ("u  [ 5 12 3 ]\n", np.int32),
        "signed.txt": ("u  [ -5 12 ]\n", np.int32),
        "floats.txt": ("u  [ 5 12.5 ]\n", np.float32),
        "matrix.txt": ("u  [\n  1 2 \n  3 4 ]\n", np.float32),
    }
    for name, (text, dtype) in cases.items():
        p = tmp_path / name
        p.write_text(text)
        assert dict(kaldi.load_ark(str(p)))["u"].dtype == dtype, name

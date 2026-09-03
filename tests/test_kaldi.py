"""Tests for the Kaldi ark/scp compatibility layer.

The golden byte strings below were checked against Kaldi's own on-disk layout;
``test_interop.py`` re-checks them against ``kaldiio`` when it is installed.
"""

import io
import os

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


def test_open_like_kaldi_read_pipe():
    with kaldi.open_like_kaldi("printf hello |", "r") as f:
        assert f.read() == "hello"


def test_open_like_kaldi_write_pipe(tmp_path):
    out = tmp_path / "out.txt"
    with kaldi.open_like_kaldi("| cat > {}".format(out), "w") as f:
        f.write("world")
    assert out.read_text() == "world"


def test_open_like_kaldi_propagates_failure():
    with pytest.raises(IOError, match="status 3"):
        with kaldi.open_like_kaldi("exit 3 |", "rb") as f:
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
    os.system("gzip -c {} > {}.gz".format(p, p))
    with kaldi.ReadHelper("ark:gunzip -c {}.gz |".format(p)) as r:
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

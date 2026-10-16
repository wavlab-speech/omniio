"""lhotse corpora -> omniio archives (omniio.lhotse)."""

import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

lhotse = pytest.importorskip("lhotse")

from lhotse import (  # noqa: E402
    AudioSource,
    CutSet,
    MonoCut,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)
from lhotse.supervision import AlignmentItem  # noqa: E402

from omniio import lhotse as olh  # noqa: E402
from omniio.tools.lhotse import read as olh_read  # noqa: E402

SR = 16000


def _wav(path, seconds, channels, seed):
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((int(seconds * SR), channels)) * 0.1).astype(np.float32)
    sf.write(path, data, SR, subtype="PCM_16")
    return str(path)


def _sup(sid, rec, start, dur, channel, words=None, **kw):
    alignment = None
    if words:
        alignment = {"word": [AlignmentItem(symbol=w, start=s, duration=d) for w, s, d in words]}
    return SupervisionSegment(
        id=sid,
        recording_id=rec,
        start=start,
        duration=dur,
        channel=channel,
        text=" ".join(w for w, _, _ in words) if words else kw.pop("text", None),
        alignment=alignment,
        **kw,
    )


@pytest.fixture
def corpus(temp_dir):
    """MEET: two mono headset files (channels 0 and 1); ROOM: one stereo file."""
    wav = temp_dir / "wav"
    wav.mkdir()
    h0 = _wav(wav / "MEET.Headset-0.wav", 6.0, 1, 0)
    h1 = _wav(wav / "MEET.Headset-1.wav", 6.0, 1, 1)
    room = _wav(wav / "ROOM.Array.wav", 4.0, 2, 2)
    meet = Recording(
        id="MEET",
        sources=[
            AudioSource(type="file", channels=[0], source=h0),
            AudioSource(type="file", channels=[1], source=h1),
        ],
        sampling_rate=SR,
        num_samples=6 * SR,
        duration=6.0,
    )
    recs = RecordingSet.from_recordings([meet, Recording.from_file(room, recording_id="ROOM")])
    sups = SupervisionSet.from_segments(
        [
            _sup(
                "MEET-b",
                "MEET",
                3.2,
                1.0,
                1,
                [("HI", 3.2, 0.4), ("THERE", 3.7, 0.5)],
                speaker="S2",
                language="English",
                gender="F",
            ),
            _sup(
                "MEET-a",
                "MEET",
                0.5,
                2.0,
                0,
                [("HELLO", 0.5, 0.6)],
                speaker="S1",
                custom={"emotion": "happy"},
            ),
            _sup("MEET-c", "MEET", 4.0, 1.5, [0, 1], text="BOTH", speaker="S1"),
            _sup("ROOM-a", "ROOM", 1.0, 2.0, None, text="ROOM TALK", speaker="S3"),
        ]
    )
    return recs, sups, temp_dir


def _ref_audio(rec, sup):
    ch = sup.channel if sup.channel is not None else rec.channel_ids
    return np.atleast_2d(rec.load_audio(channels=ch, offset=sup.start, duration=sup.duration)).T


class TestConvert:
    def test_layout_and_index(self, corpus):
        recs, sups, tmp = corpus
        audio, sup_blob = olh.from_lhotse(tmp / "out", recs, sups, name="v1", progress=False)
        assert (tmp / "out" / "audio" / "metadata.parquet").exists()
        assert (tmp / "out" / "supervisions_v1" / "metadata.parquet").exists()

        rows = {r["id"]: r for r in audio.get_metadata().to_pylist()}
        assert set(rows) == {"MEET.Headset-0", "MEET.Headset-1", "ROOM.Array"}
        assert rows["MEET.Headset-1"]["recording_id"] == "MEET"
        assert rows["ROOM.Array"]["channels"] == 2
        assert all(r["duration"] is not None for r in rows.values())

        idx = sup_blob.get_metadata().to_pylist()
        # written in (recording_id, start) order, so a window's segments are adjacent
        assert [r["id"] for r in idx] == ["MEET-a", "MEET-b", "MEET-c", "ROOM-a"]
        by_id = {r["id"]: r for r in idx}
        assert by_id["MEET-a"]["audio_ids"] == ["MEET.Headset-0"]
        assert by_id["MEET-b"]["audio_ids"] == ["MEET.Headset-1"]
        assert by_id["MEET-c"]["audio_ids"] == ["MEET.Headset-0", "MEET.Headset-1"]
        assert by_id["ROOM-a"]["audio_ids"] == ["ROOM.Array"]
        assert by_id["MEET-b"]["speaker"] == "S2" and by_id["MEET-b"]["gender"] == "F"
        assert by_id["MEET-b"]["channel"] == [1] and by_id["ROOM-a"]["channel"] is None
        assert "text" not in by_id["MEET-a"]  # payload, not index

    def test_file_sources_are_byte_copied(self, corpus):
        recs, sups, tmp = corpus
        audio, _ = olh.from_lhotse(tmp / "out", recs, sups, progress=False)
        for row in audio.get_metadata().to_pylist():
            with open(row["path"], "rb") as f:
                f.seek(row["start_byte"])
                stored = f.read(row["end_byte"] - row["start_byte"])
            with open(row["source"], "rb") as f:
                assert stored == f.read()

    def test_target_format_reencodes(self, corpus):
        recs, sups, tmp = corpus
        audio, _ = olh.from_lhotse(tmp / "out", recs, sups, target_format="flac", progress=False)
        assert set(audio.get_metadata().column("format").to_pylist()) == {"flac"}
        reader = olh.SegmentReader(
            str(tmp / "out" / "audio"), str(tmp / "out" / "supervisions_default")
        )
        seg = reader.segment("MEET-a")
        np.testing.assert_allclose(
            seg["audio"].array, _ref_audio(recs["MEET"], sups["MEET-a"]), atol=1e-4
        )

    def test_shared_audio_across_variants(self, corpus):
        recs, sups, tmp = corpus
        olh.from_lhotse(tmp / "out", recs, sups, name="all", progress=False)
        # a second "variant" that only uses headset 1 of MEET, as AMI SDM is a subset of MDM
        h1 = recs["MEET"].sources[1]
        sub = Recording(
            id="MEET",
            sources=[AudioSource("file", [0], h1.source)],
            sampling_rate=SR,
            num_samples=6 * SR,
            duration=6.0,
        )
        sub_sups = [_sup("MEET-x", "MEET", 1.0, 1.0, 0, text="SUB")]
        audio, sup_blob = olh.from_lhotse(
            tmp / "out",
            RecordingSet.from_recordings([sub]),
            SupervisionSet.from_segments(sub_sups),
            name="sub",
            progress=False,
        )
        assert len(audio) == 3  # nothing new stored
        # channel 0 of this variant is Headset-1, and the supervision says so
        assert sup_blob.get_metadata().column("audio_ids").to_pylist() == [["MEET.Headset-1"]]

    def test_same_stem_different_files_rejected(self, corpus):
        recs, sups, tmp = corpus
        olh.from_lhotse(tmp / "out", recs, sups, progress=False)
        other = tmp / "elsewhere"
        other.mkdir()
        clash = Recording.from_file(_wav(other / "ROOM.Array.wav", 1.0, 1, 9), recording_id="R2")
        with pytest.raises(ValueError, match="ROOM.Array"):
            olh.from_lhotse(tmp / "out", RecordingSet.from_recordings([clash]), progress=False)

    def test_reconversion_adds_nothing(self, corpus):
        recs, sups, tmp = corpus
        olh.from_lhotse(tmp / "out", recs, sups, progress=False)
        audio, sup_blob = olh.from_lhotse(tmp / "out", recs, sups, progress=False)
        assert len(audio) == 3 and len(sup_blob) == 4

    def test_transformed_recording_is_decoded(self, corpus):
        recs, sups, tmp = corpus
        slow = recs["ROOM"].resample(8000)
        audio, _ = olh.from_lhotse(
            tmp / "out", RecordingSet.from_recordings([slow]), progress=False
        )
        (row,) = audio.get_metadata().to_pylist()
        assert row["id"] == "ROOM.Array" and row["sample_rate"] == 8000 and row["format"] == "flac"
        reader = olh.SegmentReader(str(tmp / "out" / "audio"))
        got = reader.audio("ROOM", 1.0, 2.0).array
        # the whole recording is resampled once; lhotse resampling just a slice
        # differs at the slice edges, so compare against the full load
        ref = slow.load_audio().T[8000:16000]
        np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_needs_audio(self, corpus):
        _, sups, tmp = corpus
        with pytest.raises(ValueError, match="no audio"):
            olh.from_lhotse(tmp / "out", supervisions=sups)

    def test_manifest_paths(self, corpus):
        recs, sups, tmp = corpus
        recs.to_file(tmp / "recs.jsonl.gz")
        sups.to_file(tmp / "sups.jsonl.gz")
        audio, sup_blob = olh.from_lhotse(
            tmp / "out", str(tmp / "recs.jsonl.gz"), str(tmp / "sups.jsonl.gz"), progress=False
        )
        assert len(audio) == 3 and len(sup_blob) == 4


class TestCuts:
    def _index(self, sup_blob):
        return {r["id"]: r for r in sup_blob.get_metadata().to_pylist()}

    def test_windows_reduce_to_the_supervisions(self, corpus):
        recs, sups, tmp = corpus
        olh.from_lhotse(tmp / "ref", recs, sups, progress=False)
        windows = CutSet.from_manifests(recordings=recs, supervisions=sups).cut_into_windows(2.0)
        _, sup_blob = olh.from_lhotse(tmp / "out", cuts=windows, progress=False)
        ref = olh.SegmentReader(
            str(tmp / "ref" / "audio"), str(tmp / "ref" / "supervisions_default")
        )
        got = olh.SegmentReader(
            str(tmp / "out" / "audio"), str(tmp / "out" / "supervisions_default")
        )
        assert set(self._index(sup_blob)) == {s.id for s in sups}
        for s in sups:
            a = ref.supervisions().to_pylist()
            r = next(x for x in a if x["id"] == s.id)
            g = next(x for x in got.supervisions().to_pylist() if x["id"] == s.id)
            assert g["start"] == pytest.approx(r["start"]) and g["duration"] == pytest.approx(
                r["duration"]
            )
            assert got.payloads([s.id]) == ref.payloads([s.id])

    def test_absolute_alignment_in_a_cut_is_not_shifted(self, corpus):
        """AMI's 30 s cuts: cut-relative supervision start, absolute word times."""
        recs, _, tmp = corpus
        sup = _sup("S", "MEET", 0.2, 0.5, 0, [("WORD", 3.2, 0.5)])  # cut starts at 3.0
        cut = MonoCut(
            id="c", start=3.0, duration=2.0, channel=0, recording=recs["MEET"], supervisions=[sup]
        )
        _, sup_blob = olh.from_lhotse(tmp / "out", cuts=CutSet.from_cuts([cut]), progress=False)
        reader = olh.SegmentReader(
            str(tmp / "out" / "audio"), str(tmp / "out" / "supervisions_default")
        )
        assert self._index(sup_blob)["S"]["start"] == pytest.approx(3.2)
        assert reader.payloads(["S"])[0]["alignment"]["word"][0][1] == pytest.approx(3.2)

    def test_relative_alignment_in_a_cut_is_shifted(self, corpus):
        recs, _, tmp = corpus
        sup = _sup("S", "MEET", 0.2, 0.5, 0, [("WORD", 0.2, 0.5)])
        cut = MonoCut(
            id="c", start=3.0, duration=2.0, channel=0, recording=recs["MEET"], supervisions=[sup]
        )
        olh.from_lhotse(tmp / "out", cuts=CutSet.from_cuts([cut]), progress=False)
        reader = olh.SegmentReader(
            str(tmp / "out" / "audio"), str(tmp / "out" / "supervisions_default")
        )
        assert reader.payloads(["S"])[0]["alignment"]["word"][0][1] == pytest.approx(3.2)

    def test_mixed_cuts_are_skipped_with_a_warning(self, corpus):
        recs, sups, tmp = corpus
        cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
        mixed = cuts["MEET"] if "MEET" in cuts else next(iter(cuts))
        mixed = mixed.pad(duration=mixed.duration + 1.0)  # a MixedCut
        with pytest.warns(UserWarning, match="MixedCut"):
            olh.from_lhotse(
                tmp / "out", recordings=recs, cuts=CutSet.from_cuts([mixed]), progress=False
            )


class TestReader:
    @pytest.fixture
    def reader(self, corpus):
        recs, sups, tmp = corpus
        olh.from_lhotse(tmp / "out", recs, sups, progress=False)
        return olh.SegmentReader(
            str(tmp / "out" / "audio"), str(tmp / "out" / "supervisions_default")
        )

    def test_segment_matches_lhotse(self, corpus, reader):
        recs, sups, _ = corpus
        for s in sups:
            seg = reader.segment(s.id)
            np.testing.assert_array_equal(seg["audio"].array, _ref_audio(recs[s.recording_id], s))
            assert seg["text"] == s.text and seg["speaker"] == s.speaker
            assert seg["custom"] == s.custom
            if s.alignment:
                assert seg["alignment"]["word"] == [
                    [i.symbol, i.start, i.duration] for i in s.alignment["word"]
                ]
            else:
                assert seg["alignment"] is None

    def test_filtering_reads_no_payloads(self, reader, monkeypatch):
        monkeypatch.setattr(olh_read, "_read_range", lambda *a: pytest.fail("read bytes"))
        assert reader.supervisions("MEET", 3.0, 4.1).column("id").to_pylist() == [
            "MEET-b",
            "MEET-c",
        ]
        assert reader.supervisions(speaker="S1").column("id").to_pylist() == ["MEET-a", "MEET-c"]
        assert reader.supervisions("MEET", 2.5, 3.2).num_rows == 0  # MEET-a ends at 2.5

    def test_window_payloads_in_one_read(self, reader, monkeypatch):
        calls = []
        real = olh_read._read_range
        monkeypatch.setattr(olh_read, "_read_range", lambda *a: calls.append(a) or real(*a))
        rows = reader.supervisions("MEET")
        texts = [p["text"] for p in reader.payloads(rows)]
        assert texts == ["HELLO", "HI THERE", "BOTH"] and len(calls) == 1

    def test_payload_order_follows_the_request(self, reader):
        assert [p["text"] for p in reader.payloads(["ROOM-a", "MEET-a"])] == ["ROOM TALK", "HELLO"]

    def test_multichannel_audio(self, corpus, reader):
        recs, _, _ = corpus
        got = reader.audio("MEET", 1.0, 1.5)
        ref = recs["MEET"].load_audio(offset=1.0, duration=0.5).T
        assert got.array.shape == (8000, 2)
        np.testing.assert_array_equal(got.array, ref)
        one = reader.audio("MEET", 1.0, 1.5, audio_ids=["MEET.Headset-1"]).array
        np.testing.assert_array_equal(one, ref[:, 1:])

    def test_recording_duration(self, reader):
        assert reader.recording_duration("MEET") == pytest.approx(6.0)
        assert sorted(reader.recording_ids) == ["MEET", "ROOM"]


def test_windows():
    assert olh.windows(65.0, 30.0) == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]
    assert olh.windows(20.0, 10.0, hop=5.0)[:3] == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0)]


def test_reader_imports_without_lhotse():
    code = """
import sys
sys.modules["lhotse"] = None
from omniio.lhotse import SegmentReader, windows, from_lhotse
try:
    from_lhotse("out", recordings="recs.jsonl.gz")
except ImportError as e:
    assert "omniio[lhotse]" in str(e), e
else:
    raise SystemExit("from_lhotse should need lhotse")
"""
    subprocess.run([sys.executable, "-c", code], check=True)

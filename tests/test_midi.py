"""Tests for MIDI reading, writing, slicing and synthesis."""

import io

import numpy as np
import pytest
import zstandard as zstd

pretty_midi = pytest.importorskip("pretty_midi")

from omniio.blob.blob import Blob
from omniio.definitions import MidiRead
from omniio.interface import midi_read
from omniio.midi.common import (
    NOTE_DTYPE,
    as_note_array,
    midi_from_bytes,
    midi_from_notes,
    midi_to_bytes,
    notes_from_midi,
    slice_midi,
)
from omniio.midi.read import _detect_format, midi_read_local
from omniio.midi.synth import fluidsynth_available, synthesize
from omniio.midi.write import ZSTD_MAGIC, midi_write


# (onset, offset, pitch, velocity, program, is_drum, instrument)
EXPECTED_NOTES = [
    (0.0, 1.0, 60, 100, 0, False, 0),
    (0.5, 2.0, 64, 90, 0, False, 0),
    (1.5, 3.0, 67, 80, 0, False, 0),
    (0.0, 4.0, 40, 70, 33, False, 1),
    (2.0, 2.1, 36, 120, 0, True, 2),
]


def _build_midi():
    pm = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="piano")
    bass = pretty_midi.Instrument(program=33, name="bass")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    insts = [piano, bass, drums]
    for onset, offset, pitch, vel, _prog, _drum, k in EXPECTED_NOTES:
        insts[k].notes.append(pretty_midi.Note(vel, pitch, onset, offset))
    piano.control_changes.append(pretty_midi.ControlChange(64, 127, 0.2))  # sustain on
    piano.control_changes.append(pretty_midi.ControlChange(64, 0, 2.5))    # sustain off
    piano.pitch_bends.append(pretty_midi.PitchBend(1000, 1.0))
    pm.instruments.extend(insts)
    return pm


@pytest.fixture
def sample_midi(temp_dir):
    pm = _build_midi()
    path = temp_dir / "sample.mid"
    pm.write(str(path))
    return path, pm


def _notes_close(notes, expected, tol=2e-3):
    assert notes.dtype == NOTE_DTYPE
    assert notes.shape[0] == len(expected), (notes, expected)
    exp = np.array(expected, dtype=NOTE_DTYPE)
    exp = exp[np.lexsort((exp["offset"], exp["pitch"], exp["program"], exp["is_drum"],
                          exp["onset"]))]
    assert np.allclose(notes["onset"], exp["onset"], atol=tol)
    assert np.allclose(notes["offset"], exp["offset"], atol=tol)
    for col in ("pitch", "velocity", "program", "is_drum"):
        assert np.array_equal(notes[col], exp[col]), col


class TestNotes:
    def test_notes_from_midi_canonical_order(self):
        pm = _build_midi()
        notes = notes_from_midi(pm)
        _notes_close(notes, EXPECTED_NOTES)
        # Sorted by onset, then melodic before drums, then program, then pitch.
        assert notes["onset"].tolist() == sorted(notes["onset"].tolist())
        first_two = notes[:2]
        assert first_two["pitch"].tolist() == [60, 40] or first_two["program"].tolist() == [0, 33]

    def test_chord_order_is_by_pitch(self):
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for pitch in (67, 60, 64):
            inst.notes.append(pretty_midi.Note(100, pitch, 1.0, 2.0))
        pm.instruments.append(inst)
        assert notes_from_midi(pm)["pitch"].tolist() == [60, 64, 67]

    def test_as_note_array_from_rows_and_dict(self):
        rows = [(0.0, 1.0, 60), (1.0, 2.0, 62, 50, 5, True)]
        arr = as_note_array(rows)
        assert arr.shape[0] == 2
        assert arr[0]["velocity"] == 100 and arr[0]["program"] == 0 and not arr[0]["is_drum"]
        assert arr[1]["velocity"] == 50 and arr[1]["program"] == 5 and arr[1]["is_drum"]
        d = as_note_array({"onset": [0.0], "offset": [0.5], "pitch": [70]})
        assert d[0]["pitch"] == 70 and d[0]["velocity"] == 100

    def test_midi_from_notes_roundtrip(self):
        pm = _build_midi()
        notes = notes_from_midi(pm)
        pm2 = midi_from_notes(notes)
        assert len(pm2.instruments) == 3
        _notes_close(notes_from_midi(pm2), EXPECTED_NOTES)
        # And it serializes to a real MIDI file.
        raw = midi_to_bytes(pm2)
        assert raw.startswith(b"MThd")
        _notes_close(notes_from_midi(midi_from_bytes(raw)), EXPECTED_NOTES)

    def test_empty_midi(self):
        pm = pretty_midi.PrettyMIDI()
        notes = notes_from_midi(pm)
        assert notes.shape == (0,) and notes.dtype == NOTE_DTYPE


class TestSlice:
    def test_window_clips_and_shifts(self):
        pm = slice_midi(_build_midi(), 1.0, 2.5)
        notes = notes_from_midi(pm)
        # piano 60 [0,1] ends at the window start -> dropped
        _notes_close(notes, [
            (0.0, 1.0, 64, 90, 0, False, 0),   # [0.5,2.0] -> clipped to [1.0,2.0]
            (0.5, 1.5, 67, 80, 0, False, 0),   # [1.5,3.0] -> [1.5,2.5]
            (0.0, 1.5, 40, 70, 33, False, 1),  # [0,4] -> [1.0,2.5]
            (1.0, 1.1, 36, 120, 0, True, 2),
        ])
        assert pm.get_end_time() <= 1.5 + 1e-6

    def test_include_partial_false(self):
        pm = slice_midi(_build_midi(), 1.0, 2.5, include_partial=False)
        notes = notes_from_midi(pm)
        _notes_close(notes, [
            (0.5, 1.5, 67, 80, 0, False, 0),
            (1.0, 1.1, 36, 120, 0, True, 2),
        ])

    def test_controller_and_bend_state_carried_in(self):
        pm = slice_midi(_build_midi(), 1.5, 3.0)
        piano = pm.instruments[0]
        ccs = [(c.number, c.value, c.time) for c in piano.control_changes]
        # sustain was on (127 @0.2) before the window -> re-asserted at t=0;
        # the release (0 @2.5) lands at 1.0
        assert ccs[0] == (64, 127, 0.0)
        assert ccs[-1][0] == 64 and ccs[-1][1] == 0 and abs(ccs[-1][2] - 1.0) < 1e-9
        assert piano.pitch_bends[0].pitch == 1000 and piano.pitch_bends[0].time == 0.0

    def test_open_ended_window(self):
        pm = slice_midi(_build_midi(), 2.0, None)
        notes = notes_from_midi(pm)
        assert notes.shape[0] == 3  # 67 [1.5,3], bass [0,4], drums [2,2.1]
        assert abs(pm.get_end_time() - 2.0) < 1e-6

    def test_bad_window(self):
        with pytest.raises(ValueError):
            slice_midi(_build_midi(), 2.0, 1.0)


class TestMidiWrite:
    def test_write_from_path(self, sample_midi):
        path, _ = sample_midi
        raw, meta = midi_write(str(path), "m1")
        assert raw.startswith(b"MThd")
        assert raw == path.read_bytes()
        assert meta["format"] == "midi"
        assert abs(meta["duration"] - 4.0) < 1e-3
        assert meta["n_notes"] == 5
        assert meta["n_instruments"] == 3
        assert meta["programs"] == [0, 33]
        assert meta["has_drums"] is True
        assert meta["resolution"] == 480
        assert meta["min_pitch"] == 36 and meta["max_pitch"] == 67
        assert meta["original_size"] == meta["stored_size"] == len(raw)

    def test_write_from_bytes_and_object(self, sample_midi):
        path, pm = sample_midi
        raw_b, meta_b = midi_write(path.read_bytes(), "m2")
        assert raw_b == path.read_bytes() and meta_b["n_notes"] == 5
        raw_o, meta_o = midi_write(pm, "m3")
        assert raw_o.startswith(b"MThd") and meta_o["n_notes"] == 5
        raw_f, _ = midi_write(io.BytesIO(path.read_bytes()), "m4")
        assert raw_f == raw_b

    def test_write_compressed(self, sample_midi):
        path, _ = sample_midi
        raw, meta = midi_write(str(path), "m5", compress=True, compression_level=5)
        assert raw.startswith(ZSTD_MAGIC)
        assert meta["format"] == "midi.zst"
        assert meta["stored_size"] == len(raw)
        assert zstd.ZstdDecompressor().decompress(raw) == path.read_bytes()

    def test_write_rejects_non_midi(self):
        with pytest.raises(ValueError, match="Not a Standard MIDI File"):
            midi_write(b"RIFF" + b"\x00" * 32, "bad")


class TestMidiRead:
    def _archive(self, temp_dir, entries):
        archive = temp_dir / "midi_archive.bin"
        offsets = []
        with open(archive, "wb") as f:
            for raw in entries:
                offsets.append((f.tell(), len(raw)))
                f.write(raw)
        return archive, offsets

    def test_detect_format(self):
        assert _detect_format(b"MThd" + b"\x00" * 4) == "midi"
        assert _detect_format(ZSTD_MAGIC + b"\x00" * 4) == "midi.zst"
        with pytest.raises(ValueError, match="Unknown MIDI format"):
            _detect_format(b"XXXX" + b"\x00" * 4)

    def test_read_full(self, temp_dir, sample_midi):
        path, _ = sample_midi
        raw, _ = midi_write(str(path), "m1")
        archive, [(off, size)] = self._archive(temp_dir, [raw])

        result = midi_read_local(archive, off, size)
        assert isinstance(result, MidiRead)
        assert result.file_type == "midi" and result.modality == "midi"
        assert isinstance(result.midi, pretty_midi.PrettyMIDI)
        assert result.start_time == 0.0 and abs(result.end_time - 4.0) < 1e-3
        assert abs(result.duration - 4.0) < 1e-3
        assert result.array is None and result.sample_rate is None
        _notes_close(result.notes, EXPECTED_NOTES)

    def test_read_with_offset_and_compression(self, temp_dir, sample_midi):
        path, _ = sample_midi
        raw1, _ = midi_write(str(path), "a")
        raw2, _ = midi_write(str(path), "b", compress=True)
        archive, offsets = self._archive(temp_dir, [b"junk" * 10, raw1, raw2])

        r1 = midi_read_local(archive, *offsets[1])
        r2 = midi_read_local(archive, *offsets[2])
        assert r1.file_type == "midi" and r2.file_type == "midi.zst"
        _notes_close(r1.notes, EXPECTED_NOTES)
        _notes_close(r2.notes, EXPECTED_NOTES)

    def test_read_time_slice(self, temp_dir, sample_midi):
        path, _ = sample_midi
        raw, _ = midi_write(str(path), "m1")
        archive, [(off, size)] = self._archive(temp_dir, [raw])

        result = midi_read_local(archive, off, size, start_time=1.0, end_time=2.5)
        assert result.start_time == 1.0 and result.end_time == 2.5
        assert abs(result.duration - 1.5) < 1e-9
        assert result.notes.shape[0] == 4
        assert result.notes["onset"].min() >= 0.0
        assert result.notes["offset"].max() <= 1.5 + 1e-6

        partial = midi_read_local(archive, off, size, start_time=1.0, end_time=2.5,
                                  include_partial=False)
        assert partial.notes.shape[0] == 2

    def test_sliced_read_outputs_midi(self, temp_dir, sample_midi):
        path, _ = sample_midi
        raw, _ = midi_write(str(path), "m1")
        archive, [(off, size)] = self._archive(temp_dir, [raw])

        result = midi_read_local(archive, off, size, start_time=1.0, end_time=2.5)
        out = result.to_bytes()
        assert out.startswith(b"MThd")
        reread = notes_from_midi(midi_from_bytes(out))
        _notes_close(reread, [tuple(r) for r in result.notes.tolist()])

        out_path = temp_dir / "slice.mid"
        result.write(str(out_path))
        assert out_path.read_bytes() == out

    def test_interface_routes_local(self, temp_dir, sample_midi):
        path, _ = sample_midi
        raw, _ = midi_write(str(path), "m1")
        archive, [(off, size)] = self._archive(temp_dir, [raw])
        result = midi_read(str(archive), off, size, start_time=0.0, end_time=1.0)
        assert isinstance(result, MidiRead)
        assert result.notes.shape[0] == 3  # 60, 64 (clipped), bass (clipped)

    def test_read_with_synthesis_sine(self, temp_dir, sample_midi):
        path, _ = sample_midi
        raw, _ = midi_write(str(path), "m1")
        archive, [(off, size)] = self._archive(temp_dir, [raw])

        result = midi_read_local(archive, off, size, start_time=1.0, end_time=2.5,
                                 synthesize=True, sample_rate=8000, backend="sine")
        assert result.sample_rate == 8000
        assert result.array.shape == (12000, 1)
        assert result.array.dtype == np.float32
        assert np.abs(result.array).max() > 0


class TestSynthesize:
    def test_sine_backend_shape_and_content(self):
        pm = _build_midi()
        audio = synthesize(pm, sample_rate=8000, backend="sine")
        assert audio.shape == (32000, 1) and audio.dtype == np.float32
        assert np.abs(audio[:8000]).max() > 0        # piano+bass sounding in the first second
        assert np.abs(audio).max() <= 1.0
        stereo = synthesize(pm, sample_rate=8000, channels=2, backend="sine",
                            duration=1.0, tail_sec=0.5)
        assert stereo.shape == (12000, 2)

    def test_empty_midi_is_silence(self):
        pm = pretty_midi.PrettyMIDI()
        audio = synthesize(pm, sample_rate=8000, backend="sine", duration=0.5)
        assert audio.shape == (4000, 1) and not audio.any()

    def test_bad_args(self):
        pm = _build_midi()
        with pytest.raises(ValueError):
            synthesize(pm, channels=3, backend="sine")
        with pytest.raises(ValueError):
            synthesize(pm, backend="nope")

    @pytest.mark.skipif(not fluidsynth_available(), reason="libfluidsynth not available")
    def test_fluidsynth_backend(self):
        pm = _build_midi()
        audio = synthesize(pm, sample_rate=22050, backend="fluidsynth", duration=4.0)
        assert audio.shape == (88200, 1) and audio.dtype == np.float32
        assert np.abs(audio[:22050]).max() > 1e-3
        assert np.abs(audio).max() <= 1.0
        # Drum hit at 2.0 s must be audible (sine backend keeps drums silent).
        drums_only = pretty_midi.PrettyMIDI()
        drums_only.instruments.append(pm.instruments[2])
        d = synthesize(drums_only, sample_rate=22050, backend="fluidsynth", duration=3.0)
        assert np.abs(d[int(2.0 * 22050):int(2.5 * 22050)]).max() > 1e-3
        # Before the hit: nothing but fluidsynth's 1-LSB dither, and in particular no
        # tail leaking from the previous render through the cached synthesizer.
        assert np.abs(d[: int(1.9 * 22050)]).max() <= 2.0 / 32768
        # Second render reuses the cached synthesizer and matches to within dither.
        again = synthesize(pm, sample_rate=22050, backend="fluidsynth", duration=4.0)
        assert np.abs(audio - again).max() <= 4.0 / 32768
        stereo = synthesize(pm, sample_rate=22050, channels=2, backend="fluidsynth",
                            duration=1.0, normalize=True)
        assert stereo.shape == (22050, 2) and abs(np.abs(stereo).max() - 1.0) < 1e-6

    @pytest.mark.skipif(not fluidsynth_available(), reason="libfluidsynth not available")
    def test_fluidsynth_missing_soundfont(self):
        with pytest.raises(FileNotFoundError):
            synthesize(_build_midi(), backend="fluidsynth", soundfont="/nonexistent.sf2")


class TestBlobMidi:
    def test_append_paths_and_bytes(self, temp_dir, sample_midi):
        path, pm = sample_midi
        blob = Blob(archive_dir=str(temp_dir / "midi_blob"), modality="midi")
        blob.append(items=[str(path), path.read_bytes(), pm], ids=["p", "b", "o"],
                    num_workers=0, compress=True)
        assert len(blob) == 3

        meta = blob.get_metadata().to_pylist()
        for row in meta:
            assert row["format"] == "midi.zst"
            assert row["n_notes"] == 5
            assert row["programs"] == [0, 33]
            result = midi_read(row["path"], row["start_byte"],
                               row["end_byte"] - row["start_byte"])
            _notes_close(result.notes, EXPECTED_NOTES)

    def test_append_parallel(self, temp_dir, sample_midi):
        path, _ = sample_midi
        blob = Blob(archive_dir=str(temp_dir / "midi_blob_par"), modality="midi")
        blob.append(items=[str(path)] * 4, ids=[f"m{i}" for i in range(4)], num_workers=2)
        assert len(blob) == 4
        row = blob.get_metadata().to_pylist()[0]
        result = midi_read(row["path"], row["start_byte"],
                           row["end_byte"] - row["start_byte"], start_time=0.0, end_time=2.0)
        assert result.notes.shape[0] == 4

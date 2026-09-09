"""Shared MIDI helpers: parsing, note tables, time slicing, and building MIDI from notes.

Everything here is seconds-based. A MIDI file's tick grid and tempo map are resolved by
``pretty_midi`` on load, so notes carry absolute onset/offset times that line up with the
audio the file was recorded against — which is what a transcription pipeline needs.

``pretty_midi`` is imported lazily so that ``omniio`` keeps working (for audio/text/...)
when it is not installed.
"""
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np

MIDI_MAGIC = b"MThd"

# One row per note, sorted canonically (see `notes_from_midi`). `instrument` is the index
# of the pretty_midi Instrument (track) the note came from.
NOTE_DTYPE = np.dtype([
    ("onset", "f8"),
    ("offset", "f8"),
    ("pitch", "i2"),
    ("velocity", "i2"),
    ("program", "i2"),
    ("is_drum", "?"),
    ("instrument", "i2"),
])


def load_pretty_midi():
    try:
        import pretty_midi
    except ImportError as e:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "MIDI support requires pretty_midi (pip install pretty_midi)"
        ) from e
    return pretty_midi


def midi_from_bytes(raw: bytes):
    """Parse Standard MIDI File bytes into a ``pretty_midi.PrettyMIDI``."""
    if not raw.startswith(MIDI_MAGIC):
        raise ValueError(
            f"Not a Standard MIDI File (header bytes: {raw[:8].hex()})"
        )
    pretty_midi = load_pretty_midi()
    return pretty_midi.PrettyMIDI(io.BytesIO(raw))


def midi_to_bytes(pm) -> bytes:
    """Serialize a ``pretty_midi.PrettyMIDI`` to Standard MIDI File bytes."""
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def load_midi(source: Any) -> Tuple[Any, Optional[bytes]]:
    """Accept a path, raw SMF bytes, a file-like object, or a PrettyMIDI.

    Returns ``(pretty_midi.PrettyMIDI, raw_bytes)``. ``raw_bytes`` is the original file
    content when the source was a path/bytes/file (so it can be stored verbatim) and
    ``None`` for an in-memory PrettyMIDI (serialize it with `midi_to_bytes`).
    """
    pretty_midi = load_pretty_midi()
    if isinstance(source, pretty_midi.PrettyMIDI):
        return source, None
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        with open(str(source), "rb") as f:
            raw = f.read()
    return midi_from_bytes(raw), raw


def notes_from_midi(pm) -> np.ndarray:
    """Flatten every instrument's notes into one `NOTE_DTYPE` array.

    Rows are sorted by ``(onset, is_drum, program, pitch, offset)``. The order is
    deterministic on purpose: when a sequence model is trained to emit these notes, the
    order inside a chord is part of the prediction target, so it must not depend on track
    order or on how the file happened to be written.
    """
    rows = []
    for k, inst in enumerate(pm.instruments):
        prog = int(inst.program)
        drum = bool(inst.is_drum)
        for n in inst.notes:
            rows.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity),
                         prog, drum, k))
    arr = np.array(rows, dtype=NOTE_DTYPE) if rows else np.empty(0, dtype=NOTE_DTYPE)
    return sort_notes(arr)


def sort_notes(notes: np.ndarray) -> np.ndarray:
    """Canonical order: onset, then drums after melodic, program, pitch, offset."""
    if notes.shape[0] == 0:
        return notes
    order = np.lexsort((notes["offset"], notes["pitch"], notes["program"],
                        notes["is_drum"], notes["onset"]))
    return notes[order]


def as_note_array(notes: Any) -> np.ndarray:
    """Coerce a notes table into a `NOTE_DTYPE` array.

    Accepts an existing structured array, a dict of columns, or a sequence of rows
    ``(onset, offset, pitch[, velocity[, program[, is_drum[, instrument]]]])``. Missing
    columns default to velocity 100, program 0, not drums, instrument 0.
    """
    if isinstance(notes, np.ndarray) and notes.dtype == NOTE_DTYPE:
        return notes
    defaults = {"velocity": 100, "program": 0, "is_drum": False, "instrument": 0}
    if isinstance(notes, dict):
        n = len(notes["onset"])
        out = np.empty(n, dtype=NOTE_DTYPE)
        for name in NOTE_DTYPE.names:
            if name in notes:
                out[name] = np.asarray(notes[name])
            elif name in defaults:
                out[name] = defaults[name]
            else:
                raise KeyError(f"notes dict is missing required column {name!r}")
        return out
    rows = [tuple(r) for r in notes]
    out = np.empty(len(rows), dtype=NOTE_DTYPE)
    names = NOTE_DTYPE.names
    for i, r in enumerate(rows):
        if len(r) < 3:
            raise ValueError("each note needs at least (onset, offset, pitch)")
        vals = list(r) + [defaults[n] for n in names[len(r):]]
        out[i] = tuple(vals)
    return out


def midi_from_notes(
    notes: Any,
    resolution: int = 480,
    initial_tempo: float = 120.0,
    instrument_names: Optional[Dict[int, str]] = None,
):
    """Build a ``pretty_midi.PrettyMIDI`` from a notes table.

    Notes are grouped into instruments by ``(instrument, program, is_drum)``, so a table
    produced by `notes_from_midi` round-trips to the same track structure, and a model's
    ``(onset, offset, pitch, program)`` output becomes a playable multi-track file.
    """
    pretty_midi = load_pretty_midi()
    arr = as_note_array(notes)
    pm = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
    groups: Dict[Tuple[int, int, bool], Any] = {}
    for row in arr:
        key = (int(row["instrument"]), int(row["program"]), bool(row["is_drum"]))
        inst = groups.get(key)
        if inst is None:
            name = (instrument_names or {}).get(key[0], "")
            inst = pretty_midi.Instrument(program=key[1], is_drum=key[2], name=name)
            groups[key] = inst
        inst.notes.append(pretty_midi.Note(
            velocity=int(row["velocity"]), pitch=int(row["pitch"]),
            start=float(row["onset"]), end=float(row["offset"]),
        ))
    for key in sorted(groups):
        pm.instruments.append(groups[key])
    return pm


def tempo_at(pm, t: float) -> float:
    """Tempo (BPM) in effect at second ``t``."""
    times, tempi = pm.get_tempo_changes()
    tempo = 120.0
    for tt, bpm in zip(times, tempi):
        if tt <= t:
            tempo = float(bpm)
        else:
            break
    return tempo


def slice_midi(pm, start_time: Optional[float], end_time: Optional[float],
               include_partial: bool = True):
    """Restrict a PrettyMIDI to the window ``[start_time, end_time)`` and re-zero time.

    Semantics match slicing the paired audio: a note is kept if it *sounds* inside the
    window (``offset > start`` and ``onset < end``), its onset/offset are clipped to the
    window, and all times are shifted by ``-start_time`` so ``0.0`` is the window start.
    Set ``include_partial=False`` to drop notes that began before the window (only notes
    whose onset lies inside it survive; offsets are still clipped).

    Controller and pitch-bend *state* at ``start_time`` (e.g. a held sustain pedal) is
    carried in as an event at ``t=0`` so the slice synthesizes the way the full file
    would at that moment.
    """
    pretty_midi = load_pretty_midi()
    start = 0.0 if start_time is None else float(start_time)
    end = float(pm.get_end_time()) if end_time is None else float(end_time)
    if start < 0:
        raise ValueError(f"start_time must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"end_time ({end}) must be >= start_time ({start})")

    out = pretty_midi.PrettyMIDI(resolution=pm.resolution, initial_tempo=tempo_at(pm, start))

    for inst in pm.instruments:
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum,
                                     name=inst.name)
        for n in inst.notes:
            if n.end <= start or n.start >= end:
                continue
            if not include_partial and n.start < start:
                continue
            new.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=max(n.start, start) - start, end=min(n.end, end) - start,
            ))

        last_cc: Dict[int, int] = {}
        kept_cc = []
        for cc in sorted(inst.control_changes, key=lambda c: c.time):
            if cc.time < start:
                last_cc[cc.number] = cc.value
            elif cc.time < end:
                kept_cc.append(pretty_midi.ControlChange(cc.number, cc.value, cc.time - start))
        new.control_changes = [pretty_midi.ControlChange(num, val, 0.0)
                               for num, val in sorted(last_cc.items())] + kept_cc

        last_bend = None
        kept_bends = []
        for b in sorted(inst.pitch_bends, key=lambda b: b.time):
            if b.time < start:
                last_bend = b.pitch
            elif b.time < end:
                kept_bends.append(pretty_midi.PitchBend(b.pitch, b.time - start))
        if last_bend is not None and last_bend != 0:
            kept_bends.insert(0, pretty_midi.PitchBend(last_bend, 0.0))
        new.pitch_bends = kept_bends

        out.instruments.append(new)

    for ts in pm.time_signature_changes:
        if start <= ts.time < end:
            out.time_signature_changes.append(
                pretty_midi.TimeSignature(ts.numerator, ts.denominator, ts.time - start))
    for ks in pm.key_signature_changes:
        if start <= ks.time < end:
            out.key_signature_changes.append(
                pretty_midi.KeySignature(ks.key_number, ks.time - start))
    return out


def midi_metadata(pm) -> dict:
    """Parquet-friendly summary of a PrettyMIDI (used by the writer)."""
    notes = notes_from_midi(pm)
    programs = sorted({int(i.program) for i in pm.instruments if not i.is_drum})
    return {
        "duration": float(pm.get_end_time()),
        "n_notes": int(notes.shape[0]),
        "n_instruments": int(len(pm.instruments)),
        "programs": programs,
        "has_drums": bool(any(i.is_drum for i in pm.instruments)),
        "resolution": int(pm.resolution),
        "min_pitch": int(notes["pitch"].min()) if notes.shape[0] else None,
        "max_pitch": int(notes["pitch"].max()) if notes.shape[0] else None,
    }

"""Read one MIDI entry out of an archive, optionally time-sliced and/or synthesized."""
from typing import Optional

import requests
import zstandard as zstd

from omniio.definitions import MidiRead
from omniio.midi.common import MIDI_MAGIC, midi_from_bytes, notes_from_midi, slice_midi
from omniio.midi.synth import DEFAULT_SAMPLE_RATE, synthesize
from omniio.midi.write import ZSTD_MAGIC


def _detect_format(header: bytes) -> str:
    if header[:4] == MIDI_MAGIC:
        return "midi"
    if header[:4] == ZSTD_MAGIC:
        return "midi.zst"
    raise ValueError(f"Unknown MIDI format (header bytes: {header[:8].hex()})")


def _decode_midi(
    blob: bytes,
    start_time: Optional[float],
    end_time: Optional[float],
    include_partial: bool,
    synthesize_audio: bool,
    sample_rate: int,
    channels: int,
    soundfont: Optional[str],
    backend: str,
) -> MidiRead:
    fmt = _detect_format(blob[:8])
    if fmt == "midi.zst":
        blob = zstd.ZstdDecompressor().decompress(blob)

    pm = midi_from_bytes(blob)
    full_end = float(pm.get_end_time())

    if start_time is None and end_time is None:
        start, end = 0.0, full_end
    else:
        start = 0.0 if start_time is None else float(start_time)
        end = full_end if end_time is None else float(end_time)
        pm = slice_midi(pm, start, end, include_partial=include_partial)
    duration = max(end - start, 0.0)

    result = MidiRead(
        file_type=fmt,
        modality="midi",
        midi=pm,
        notes=notes_from_midi(pm),
        duration=duration,
        start_time=start,
        end_time=end,
    )
    if synthesize_audio:
        result.sample_rate = int(sample_rate)
        result.array = synthesize(pm, sample_rate=sample_rate, channels=channels,
                                  duration=duration, soundfont=soundfont, backend=backend)
    return result


def midi_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    include_partial: bool = True,
    synthesize: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    soundfont: Optional[str] = None,
    backend: str = "auto",
) -> MidiRead:
    """
    Read a single MIDI entry from a binary archive blob.

    Args:
        archive_path:    Path to the .bin file.
        start_offset:    Byte offset where this entry begins.
        file_size:       Number of bytes for this entry.
        start_time:      Window start in seconds (None = beginning).
        end_time:        Window end in seconds (None = end of file).
        include_partial: Keep notes that began before ``start_time`` (clipped to the
                         window). ``False`` keeps only notes whose onset is inside it.
        synthesize:      Also render the (sliced) MIDI to a waveform.
        sample_rate:     Synthesis sample rate.
        channels:        Synthesis channels, 1 or 2.
        soundfont:       ``.sf2`` for the fluidsynth backend.
        backend:         ``"auto"`` | ``"fluidsynth"`` | ``"sine"``.

    Returns:
        MidiRead with ``midi`` (PrettyMIDI, re-zeroed to the window), a ``notes`` table,
        ``duration``, and — when ``synthesize`` — ``sample_rate`` and a float32 ``array``
        of shape (frames, channels) covering exactly ``duration`` seconds.
    """
    with open(archive_path, "rb") as f:
        f.seek(start_offset)
        blob = f.read(file_size)
    return _decode_midi(blob, start_time, end_time, include_partial, synthesize,
                        sample_rate, channels, soundfont, backend)


def midi_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    include_partial: bool = True,
    synthesize: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    soundfont: Optional[str] = None,
    backend: str = "auto",
) -> MidiRead:
    """Same as `midi_read_local`, over an HTTP range request."""
    end_byte = start_offset + file_size - 1
    headers = {"Range": f"bytes={start_offset}-{end_byte}"}

    resp = requests.get(archive_url, headers=headers)
    resp.raise_for_status()
    return _decode_midi(resp.content, start_time, end_time, include_partial, synthesize,
                        sample_rate, channels, soundfont, backend)

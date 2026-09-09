"""Store MIDI in an omniio archive.

Bytes are kept as a Standard MIDI File — the native format, exactly as audio is kept as
FLAC/WAV — so an entry can be handed straight to any MIDI tool. ``compress=True`` wraps
the file in zstandard (MIDI compresses 2–4×); the reader detects either form from the
first bytes.
"""
from typing import Any, Tuple

import zstandard as zstd

from omniio.midi.common import load_midi, midi_metadata, midi_to_bytes

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def midi_write(
    midi_source: Any,
    item_id: str,
    compress: bool = False,
    compression_level: int = 3,
) -> Tuple[bytes, dict]:
    """Read a MIDI file (or in-memory MIDI) and return raw bytes + metadata dict.

    Args:
        midi_source:       Path to a ``.mid``/``.midi`` file (str or Path), raw Standard
                           MIDI File ``bytes`` (e.g. a parquet ``binary`` column), a
                           file-like object, or a ``pretty_midi.PrettyMIDI``.
        item_id:           Unique identifier for this sample.
        compress:          zstandard-compress the stored bytes.
        compression_level: zstandard level (1–22) when ``compress`` is set.

    Returns:
        (raw_bytes, metadata_dict) with ``duration`` (s), ``n_notes``, ``n_instruments``,
        ``programs`` (GM programs of the melodic tracks), ``has_drums``, ``resolution``
        (ticks per beat), pitch range, ``format`` ('midi' or 'midi.zst') and sizes.
    """
    pm, raw = load_midi(midi_source)
    if raw is None:
        raw = midi_to_bytes(pm)

    metadata = midi_metadata(pm)
    metadata["original_size"] = len(raw)

    if compress:
        raw = zstd.ZstdCompressor(level=compression_level).compress(raw)
        metadata["format"] = "midi.zst"
    else:
        metadata["format"] = "midi"
    metadata["stored_size"] = len(raw)
    return raw, metadata

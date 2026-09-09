from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class ArchiveRead:
    file_type: str = None
    modality: str = None

@dataclass
class AudioRead(ArchiveRead):
    sample_rate: int = None
    array: np.ndarray = None

@dataclass
class TextRead(ArchiveRead):
    text: str = None

@dataclass
class VideoRead(ArchiveRead):
    sample_rate: int = None
    fps: float = None
    height: int = None
    width: int = None
    audio_array: np.ndarray = None
    video_array: np.ndarray = None

@dataclass
class ImageRead(ArchiveRead):
    height: int = None
    width: int = None
    channels: int = None
    array: np.ndarray = None

@dataclass
class MidiRead(ArchiveRead):
    """A MIDI entry, optionally restricted to a time window and/or synthesized.

    `midi` is a pretty_midi.PrettyMIDI whose times are relative to `start_time`;
    `notes` is a structured array (see omniio.modalities.midi.common.NOTE_DTYPE) with onset/offset
    in seconds on the same clock; `duration` is the seconds the window covers.
    `sample_rate` / `array` (frames, channels) float32 are set only when synthesized.
    """
    midi: Any = None
    notes: np.ndarray = None
    duration: float = None
    start_time: float = None
    end_time: float = None
    sample_rate: int = None
    array: np.ndarray = None

    def to_bytes(self) -> bytes:
        """Serialize `midi` (i.e. the window that was read) to Standard MIDI File bytes."""
        from omniio.modalities.midi.common import midi_to_bytes
        return midi_to_bytes(self.midi)

    def write(self, path: str) -> None:
        """Write `midi` to a .mid file."""
        with open(path, "wb") as f:
            f.write(self.to_bytes())


@dataclass
class DiscreteRead(ArchiveRead):
    """Integer streams (VQ / RVQ codes, token ids) from one entry, optionally a subset of
    streams and a time / index window.

    `streams` holds one 1-D unsigned array per requested stream (narrowest dtype for its
    alphabet); `stream_indices` says which entry streams they are; `lengths`,
    `vocab_sizes` and `rates` (units per second, or None) line up with them. `array`
    stacks the streams into `(n, T)` when their lengths agree (RVQ) — use `to_array()`
    to pad ragged streams.
    """
    streams: list = None
    stream_indices: list = None
    lengths: list = None
    vocab_sizes: list = None
    rates: list = None
    n_streams: int = None
    frame_windows: list = None        # [lo, hi) decoded frames per returned stream
    start_time: float = None
    end_time: float = None
    start_frame: int = None
    end_frame: int = None
    bytes_read: int = None            # I/O actually done (header + the windows' bytes)
    entry_size: int = None

    @property
    def ragged(self) -> bool:
        return len(set(self.lengths or [])) > 1

    @property
    def array(self) -> np.ndarray:
        """`(n, T)` stack of the streams; raises when the lengths differ."""
        if self.ragged:
            raise ValueError("streams have different lengths; use to_array(pad_value=...)")
        return np.stack(self.streams) if self.streams else np.zeros((0, 0), dtype=np.uint8)

    def to_array(self, pad_value: int = -1) -> np.ndarray:
        """`(n, max_T)` int64 with `pad_value` past each stream's length."""
        T = max(self.lengths) if self.lengths else 0
        out = np.full((len(self.streams), T), pad_value, dtype=np.int64)
        for i, s in enumerate(self.streams):
            out[i, : s.size] = s
        return out

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
    `notes` is a structured array (see omniio.midi.common.NOTE_DTYPE) with onset/offset
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
        from omniio.midi.common import midi_to_bytes
        return midi_to_bytes(self.midi)

    def write(self, path: str) -> None:
        """Write `midi` to a .mid file."""
        with open(path, "wb") as f:
            f.write(self.to_bytes())

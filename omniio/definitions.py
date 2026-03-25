from dataclasses import dataclass
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
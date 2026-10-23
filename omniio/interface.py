import os
from typing import Optional
from omniio.definitions import ArchiveRead, AudioRead, BytesRead, TextRead, VideoRead, ImageRead, MidiRead
from omniio.modalities.audio.read import audio_read_local, audio_read_remote
from omniio.modalities.text.read import text_read_local, text_read_remote
from omniio.modalities.video.read import video_read_local, video_read_remote
from omniio.modalities.image.read import image_read_local, image_read_remote
from omniio.modalities.midi.read import midi_read_local, midi_read_remote

def audio_read(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> AudioRead:

    if os.path.exists(archive_path):
        return audio_read_local(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )

    else:
        return audio_read_remote(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )

def text_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> TextRead:

    if os.path.exists(archive_path):
        return text_read_local(
            archive_path,
            start_offset,
            file_size,
        )

    else:
        return text_read_remote(
            archive_path,
            start_offset,
            file_size,
        )

def video_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> VideoRead:
    """
    Read video from archive, automatically routing to local or remote.

    Args:
        archive_path: Path or URL to the archive bin file
        start_offset: Byte offset where video entry begins
        file_size: Number of bytes for this entry
        start_frame: Optional start frame for slicing
        end_frame: Optional end frame for slicing
        start_time: Optional start time in seconds for slicing
        end_time: Optional end time in seconds for slicing

    Returns:
        VideoRead object with video and audio data
    """
    if os.path.exists(archive_path):
        return video_read_local(
            archive_path,
            start_offset,
            file_size,
            start_frame,
            end_frame,
            start_time,
            end_time
        )
    else:
        return video_read_remote(
            archive_path,
            start_offset,
            file_size,
            start_frame,
            end_frame,
            start_time,
            end_time
        )

def image_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> ImageRead:
    if os.path.exists(archive_path):
        return image_read_local(archive_path, start_offset, file_size)
    else:
        return image_read_remote(archive_path, start_offset, file_size)


def midi_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    include_partial: bool = True,
    synthesize: bool = False,
    sample_rate: int = 44100,
    channels: int = 1,
    soundfont: Optional[str] = None,
    backend: str = "auto",
) -> MidiRead:
    """
    Read MIDI from archive, automatically routing to local or remote.

    Args:
        archive_path:    Path or URL to the archive bin file
        start_offset:    Byte offset where the MIDI entry begins
        file_size:       Number of bytes for this entry
        start_time:      Optional window start in seconds
        end_time:        Optional window end in seconds
        include_partial: Keep (clipped) notes that started before start_time
        synthesize:      Also render the MIDI to a waveform (`array`, `sample_rate`)
        sample_rate:     Synthesis sample rate
        channels:        Synthesis channels (1 or 2)
        soundfont:       .sf2 path for the fluidsynth backend
        backend:         "auto" | "fluidsynth" | "sine"

    Returns:
        MidiRead with the (sliced) PrettyMIDI, a notes table, and optional waveform
    """
    kwargs = dict(start_time=start_time, end_time=end_time, include_partial=include_partial,
                  synthesize=synthesize, sample_rate=sample_rate, channels=channels,
                  soundfont=soundfont, backend=backend)
    if os.path.exists(archive_path):
        return midi_read_local(archive_path, start_offset, file_size, **kwargs)
    else:
        return midi_read_remote(archive_path, start_offset, file_size, **kwargs)


def bytes_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> BytesRead:
    """
    Read one entry back as the bytes that were stored, local or remote. The
    modality-agnostic read: what ``fast_mode`` wrote, an archive of a modality omniio has
    no decoder for, or any entry whose decoding the caller wants to do itself.

    Args:
        archive_path: Path or URL to the archive bin file
        start_offset: Byte offset where the entry begins
        file_size:    Number of bytes for the entry

    Returns:
        BytesRead with ``data`` (bytes)
    """
    if os.path.exists(archive_path):
        with open(archive_path, "rb") as f:
            f.seek(start_offset)
            data = f.read(file_size)
    else:
        import requests
        resp = requests.get(archive_path, headers={"Range": f"bytes={start_offset}-{start_offset + file_size - 1}"})
        resp.raise_for_status()
        if resp.status_code != 206:                          # a server that ignores Range sends the whole file
            raise IOError(f"{archive_path}: range request answered with HTTP {resp.status_code}, not 206 Partial Content")
        data = resp.content
    return BytesRead(file_type=None, modality="bytes", data=data)

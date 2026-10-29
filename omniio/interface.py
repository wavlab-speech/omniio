import os
from typing import Optional
from omniio.definitions import ArchiveRead, AudioRead, BytesRead, TextRead, VideoRead, ImageRead, MidiRead, DiscreteRead, CellRead, SeqRead, TrackRead
from omniio.modalities.audio.read import audio_read_local, audio_read_remote
from omniio.modalities.text.read import text_read_local, text_read_remote
from omniio.modalities.video.read import video_read_local, video_read_remote
from omniio.modalities.image.read import image_read_local, image_read_remote
from omniio.modalities.midi.read import midi_read_local, midi_read_remote
from omniio.modalities.discrete.read import discrete_read_local, discrete_read_remote
from omniio.modalities.cell.read import cell_read_local, cell_read_remote, cell_read_batch_local, cell_read_batch_remote
from omniio.modalities.seq.read import seq_read_local, seq_read_remote
from omniio.modalities.track.read import track_read_local, track_read_remote


def _is_url(path) -> bool:
    """Local unless it is an http(s) URL: a stat per read was 2 of the 12 syscalls of a small read."""
    return isinstance(path, str) and path.startswith(("http://", "https://"))

def audio_read(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> AudioRead:

    if not _is_url(archive_path):
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

    if not _is_url(archive_path):
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
    if not _is_url(archive_path):
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
    if not _is_url(archive_path):
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
    frequency_shift: float = 0.0,
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
        frequency_shift: Transpose by this many semitones (fractional allowed); the MIDI,
                         the notes table and the synthesized waveform are all shifted.
                         Drums stay put; notes pushed outside 0..127 are dropped.

    Returns:
        MidiRead with the (sliced) PrettyMIDI, a notes table, and optional waveform
    """
    kwargs = dict(start_time=start_time, end_time=end_time, include_partial=include_partial,
                  synthesize=synthesize, sample_rate=sample_rate, channels=channels,
                  soundfont=soundfont, backend=backend, frequency_shift=frequency_shift)
    if not _is_url(archive_path):
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
    if not _is_url(archive_path):
        from omniio.blob.handles import archive_fd, pread_all
        data = pread_all(archive_fd(archive_path), file_size, start_offset)
    else:
        import requests
        resp = requests.get(archive_path, headers={"Range": f"bytes={start_offset}-{start_offset + file_size - 1}"})
        resp.raise_for_status()
        if resp.status_code != 206:                          # a server that ignores Range sends the whole file
            raise IOError(f"{archive_path}: range request answered with HTTP {resp.status_code}, not 206 Partial Content")
        data = resp.content
    return BytesRead(file_type=None, modality="bytes", data=data)


def discrete_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
    streams: Optional[list] = None,
    start_level: Optional[int] = None,
    end_level: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> DiscreteRead:
    """
    Read integer streams (VQ / RVQ codes, token ids) from an archive, local or remote.
    Partial reads fetch only the bytes they need: a level window is one contiguous read,
    a frame window is one small read per stream (zstd-wrapped entries read the payload).

    Args:
        archive_path: Path or URL to the archive bin file
        start_offset: Byte offset where the entry begins
        file_size:    Number of bytes for this entry
        streams:      Explicit stream indices to decode (default all); or
        start_level / end_level: window of stream (RVQ level) indices, [start, end)
        start_frame / end_frame: window in elements (frames); wins over the time window
        start_time / end_time:   window in seconds through each stream's own rate

    Returns:
        DiscreteRead with `streams` (list of 1-D arrays), `array` ((n, T) when regular),
        `lengths`, `vocab_sizes`, `rates`, `frame_windows`, `bytes_read`
    """
    kwargs = dict(streams=streams, start_level=start_level, end_level=end_level,
                  start_frame=start_frame, end_frame=end_frame, start_time=start_time, end_time=end_time)
    if not _is_url(archive_path):
        return discrete_read_local(archive_path, start_offset, file_size, **kwargs)
    return discrete_read_remote(archive_path, start_offset, file_size, **kwargs)


def cell_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> CellRead:
    """
    Read one cell's sparse row from an archive, local or remote: one pread / one range
    request of the entry, then a vectorised decode.

    Args:
        archive_path: Path or URL to the archive bin file
        start_offset: Byte offset where the entry begins
        file_size:    Number of bytes for this entry

    Returns:
        CellRead with `indices`, `values`, `n_genes`, `nnz`, `dense()`, `to_csr()`
    """
    if not _is_url(archive_path):
        return cell_read_local(archive_path, start_offset, file_size)
    return cell_read_remote(archive_path, start_offset, file_size)


def cell_read_batch(
    archive_path: str,
    ranges,
    max_gap: int = 64 * 1024,
):
    """
    Read several cells of one bin into a `scipy.sparse.csr_matrix` (rows in the order of
    `ranges`). Byte ranges closer than `max_gap` are fetched with one pread / request.

    Args:
        archive_path: Path or URL to the archive bin file
        ranges:       [(start_byte, size), ...]
        max_gap:      Coalescing distance in bytes
    """
    if not _is_url(archive_path):
        return cell_read_batch_local(archive_path, ranges, max_gap=max_gap)
    return cell_read_batch_remote(archive_path, ranges, max_gap=max_gap)


def seq_read(archive_path: str, start_offset: int, file_size: int, start: Optional[int] = None, end: Optional[int] = None,
             reverse_complement: bool = False, soft_mask: bool = True) -> SeqRead:
    """
    Read bases `[start, end)` (0-based, half-open) of a sequence entry, local or remote:
    three small byte-range reads (2-bit bases, N bitmap, soft-mask bitmap).

    Args:
        archive_path: Path or URL to the archive bin file
        start_offset: Byte offset where the entry begins
        file_size:    Number of bytes for this entry
        start / end:  Window in bases (default the whole record)
        reverse_complement: Return the other strand
        soft_mask:    Keep lowercase for repeat-masked bases

    Returns:
        SeqRead with `sequence`, `start`, `end`, `length`
    """
    kw = dict(start=start, end=end, reverse_complement=reverse_complement, soft_mask=soft_mask)
    if not _is_url(archive_path):
        return seq_read_local(archive_path, start_offset, file_size, **kw)
    return seq_read_remote(archive_path, start_offset, file_size, **kw)


def track_read(archive_path: str, start_offset: int, file_size: int, start: Optional[int] = None,
               end: Optional[int] = None) -> TrackRead:
    """
    Read the bins covering bp `[start, end)` of a binned signal track entry (one chromosome),
    local or remote: the header, then one contiguous read.

    Returns:
        TrackRead with `array` (float32, NaN = no coverage), `bin`, `start`, `end`
    """
    if not _is_url(archive_path):
        return track_read_local(archive_path, start_offset, file_size, start=start, end=end)
    return track_read_remote(archive_path, start_offset, file_size, start=start, end=end)

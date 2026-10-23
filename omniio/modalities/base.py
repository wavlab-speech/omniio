"""
The modality-agnostic archive entry type, and the registry of the modalities built on it.

An archive entry is a byte string at an offset of a bin file. Everything a modality adds
sits on top of that: a writer that may transcode the item and describe it (metadata
columns), and a reader that decodes the bytes. :class:`Modality` is the base: it stores
an item's bytes as they are and reads them back as bytes, so an archive can hold any
file type, including ones omniio has no decoder for. Each supported modality subclasses
it and overrides :meth:`Modality.write` with its transcoding, metadata-producing writer.

``Blob.append(fast_mode=True)`` uses the base writer for every modality: a straight byte
copy, no transcode, no per-item metadata beyond the index columns the archive needs.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

INDEX_COLUMNS = ("id", "start_byte", "end_byte", "bin_index", "path")   # what the archive index needs; a fast-mode row has nothing else


def item_bytes(item: Any) -> bytes:
    """The bytes of an item as handed to ``append``: a path, raw bytes, or a readable file object."""
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, (str, Path, os.PathLike)):
        with open(item, "rb") as f:
            return f.read()
    if hasattr(item, "read"):
        data = item.read()
        if isinstance(data, str):
            raise TypeError("fast mode stores bytes: open the file in binary mode")
        return bytes(data)
    raise TypeError(f"cannot store a {type(item).__name__} as raw bytes: pass a path, bytes or a binary file object")


class Modality:
    """An archive entry type.

    The base class is modality-agnostic: :meth:`write` stores the item's bytes unchanged
    with no metadata, and :meth:`read` returns the stored bytes. A subclass overrides
    :meth:`write` with a writer that may transcode the item and returns the metadata
    columns describing it (``audio_write`` and its siblings), and :meth:`read` with its
    decoder. :meth:`write_raw` is the byte copy itself and is the same for every modality;
    it is what ``fast_mode`` uses.
    """

    name: str = "bytes"

    def __init__(self, name: Optional[str] = None):
        if name is not None:
            self.name = name

    # --- write --------------------------------------------------------------------
    def write_raw(self, item: Any, item_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """Store the item's bytes as they are. No transcode, no metadata."""
        return item_bytes(item), {}

    def write(self, item: Any, item_id: str, **kwargs) -> Tuple[bytes, Dict[str, Any]]:
        """The modality's writer: ``(stored bytes, metadata columns)``.

        The base class has no transcoder and nothing to describe, so this is the raw copy;
        it takes no keyword arguments.
        """
        if kwargs:
            raise TypeError(f"modality {self.name!r} stores raw bytes and takes no write options: {sorted(kwargs)}")
        return self.write_raw(item, item_id)

    # --- read ---------------------------------------------------------------------
    def read(self, archive_path: str, start_offset: int, file_size: int, **kwargs):
        """Read one entry back. The base class returns the bytes (:class:`BytesRead`)."""
        from omniio.interface import bytes_read
        if kwargs:
            raise TypeError(f"modality {self.name!r} reads raw bytes and takes no read options: {sorted(kwargs)}")
        r = bytes_read(archive_path, start_offset, file_size)
        r.modality = self.name
        return r

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class AudioModality(Modality):
    name = "audio"

    def write(self, item, item_id, **kwargs):
        from omniio.modalities.audio.write import audio_write
        return audio_write(item, item_id, **kwargs)

    def read(self, archive_path, start_offset, file_size, **kwargs):
        from omniio.interface import audio_read
        return audio_read(archive_path, start_offset, file_size, **kwargs)


class TextModality(Modality):
    name = "text"

    def write(self, item, item_id, **kwargs):
        from omniio.modalities.text.write import text_write
        return text_write(item, item_id, **kwargs)

    def read(self, archive_path, start_offset, file_size, **kwargs):
        from omniio.interface import text_read
        return text_read(archive_path, start_offset, file_size, **kwargs)


class VideoModality(Modality):
    name = "video"

    def write(self, item, item_id, **kwargs):
        from omniio.modalities.video.write import video_write
        return video_write(item, item_id, **kwargs)

    def read(self, archive_path, start_offset, file_size, **kwargs):
        from omniio.interface import video_read
        return video_read(archive_path, start_offset, file_size, **kwargs)


class ImageModality(Modality):
    name = "image"

    def write(self, item, item_id, **kwargs):
        from omniio.modalities.image.write import image_write
        return image_write(item, item_id, **kwargs)

    def read(self, archive_path, start_offset, file_size, **kwargs):
        from omniio.interface import image_read
        return image_read(archive_path, start_offset, file_size, **kwargs)


class MidiModality(Modality):
    name = "midi"

    def write(self, item, item_id, **kwargs):
        from omniio.modalities.midi.write import midi_write
        return midi_write(item, item_id, **kwargs)

    def read(self, archive_path, start_offset, file_size, **kwargs):
        from omniio.interface import midi_read
        return midi_read(archive_path, start_offset, file_size, **kwargs)


# The modalities with a transcoding writer and a decoder. Any other name is served by the
# base class: raw bytes in, raw bytes out.
MODALITIES: Dict[str, Modality] = {m.name: m for m in (AudioModality(), TextModality(), VideoModality(), ImageModality(), MidiModality())}


def get_modality(name: str) -> Modality:
    """The registered modality of that name, or a bytes-only :class:`Modality` named after it."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"modality must be a non-empty string, got {name!r}")
    return MODALITIES.get(name) or Modality(name)

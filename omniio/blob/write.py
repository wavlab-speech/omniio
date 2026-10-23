"""Writers by modality name.

``modality_writer`` is the registry of transcoding, metadata-producing writers (kept as a
plain mapping for the code that imports it). :func:`writer_for` resolves what
``Blob.append`` calls: the modality's writer, the raw byte copy for ``fast_mode``, and the
raw byte copy for a modality omniio has no writer for (see :mod:`omniio.modalities.base`).
"""
from omniio.modalities.audio.write import audio_write
from omniio.modalities.base import MODALITIES, get_modality
from omniio.modalities.image.write import image_write
from omniio.modalities.midi.write import midi_write
from omniio.modalities.text.write import text_write
from omniio.modalities.video.write import video_write

modality_writer = {
    'audio': audio_write,
    'text': text_write,
    'video': video_write,
    'image': image_write,
    'midi': midi_write,
}


def writer_for(modality: str, fast_mode: bool = False):
    """The ``(item, item_id, **kwargs) -> (bytes, metadata)`` callable ``append`` uses."""
    m = get_modality(modality)
    if fast_mode:
        return lambda item, item_id, **kw: m.write_raw(item, item_id)
    return m.write

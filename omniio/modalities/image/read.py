import io

import numpy as np
import requests
from PIL import Image

from omniio.definitions import ImageRead


# Magic bytes for format detection
_MAGIC = {
    b"\x89PNG":    "png",
    b"\xff\xd8\xff": "jpeg",
    b"RIFF":       "webp",  # RIFF....WEBP — checked further below
    b"GIF8":       "gif",
}

_WEBP_MARKER = b"WEBP"


def _detect_format(header: bytes) -> str:
    for magic, fmt in _MAGIC.items():
        if header[: len(magic)] == magic:
            if fmt == "webp" and header[8:12] != _WEBP_MARKER:
                continue
            return fmt
    raise ValueError(f"Unknown image format (header bytes: {header[:16].hex()})")


def _decode_image(blob: bytes, fmt: str) -> ImageRead:
    img = Image.open(io.BytesIO(blob))
    img.load()
    array = np.array(img)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    height, width, channels = array.shape
    return ImageRead(
        file_type=fmt,
        modality="image",
        height=height,
        width=width,
        channels=channels,
        array=array,
    )


def image_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> ImageRead:
    """
    Read a single image entry from a binary archive blob.

    Args:
        archive_path: Path to the .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.

    Returns:
        ImageRead with uint8 array (height, width, channels).
    """
    with open(archive_path, "rb") as f:
        f.seek(start_offset)
        blob = f.read(file_size)

    fmt = _detect_format(blob[:16])
    return _decode_image(blob, fmt)


def image_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
) -> ImageRead:
    """
    Read a single image entry from a remote binary archive via HTTP range request.

    Args:
        archive_url:  URL to the remote .bin file.
        start_offset: Byte offset where this entry begins.
        file_size:    Number of bytes for this entry.

    Returns:
        ImageRead with uint8 array (height, width, channels).
    """
    end_byte = start_offset + file_size - 1
    headers = {"Range": f"bytes={start_offset}-{end_byte}"}

    resp = requests.get(archive_url, headers=headers)
    resp.raise_for_status()

    blob = resp.content
    fmt = _detect_format(blob[:16])
    return _decode_image(blob, fmt)

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


SUPPORTED_FORMATS = {"png", "jpeg"}


def image_write(
    image_path: Union[str, Path, np.ndarray],
    item_id: str,
    target_format: Optional[str] = "png",
) -> Tuple[bytes, dict]:
    """
    Read an image file or numpy array, encode to target format, and return
    raw bytes + metadata dict.

    Args:
        image_path:    Path to source image file, or a numpy array (H, W) / (H, W, C) uint8.
        item_id:       Unique identifier for this sample.
        target_format: Output format, 'png' (default) or 'jpeg'.

    Returns:
        (raw_bytes, metadata_dict)
    """
    if target_format is None:
        target_format = "png"
    target_format = target_format.lower()
    if target_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported target format: {target_format!r}. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    if isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path)
        src_format = None
    else:
        img = Image.open(str(image_path))
        src_format = img.format.lower() if img.format else None

    # JPEG does not support alpha — drop alpha channel if present
    if target_format == "jpeg" and img.mode in ("RGBA", "LA", "PA"):
        img = img.convert("RGB")
    elif target_format == "jpeg" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    channels = len(img.getbands())

    buf = io.BytesIO()
    pil_format = "JPEG" if target_format == "jpeg" else "PNG"
    img.save(buf, format=pil_format)
    raw_bytes = buf.getvalue()

    metadata = {
        "format": target_format,
        "height": height,
        "width": width,
        "channels": channels,
    }
    return raw_bytes, metadata

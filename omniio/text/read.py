from dataclasses import dataclass
from typing import Optional

import requests
import zstandard as zstd

from omniio.definitions import TextRead

def _decompress_blob(blob: bytes) -> str:
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(blob).decode("utf-8")


def text_read_local(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> TextRead:
    with open(archive_path, "rb") as f:
        f.seek(start_offset)
        blob = f.read(file_size)

    return TextRead(
        file_type="text",
        modality="text",
        text=_decompress_blob(blob),
    )


def text_read_remote(
    archive_url: str,
    start_offset: int,
    file_size: int,
) -> TextRead:
    end_byte = start_offset + file_size - 1
    headers = {"Range": f"bytes={start_offset}-{end_byte}"}

    resp = requests.get(archive_url, headers=headers)
    resp.raise_for_status()

    return TextRead(
        file_type="text",
        modality="text",
        text=_decompress_blob(resp.content),
    )
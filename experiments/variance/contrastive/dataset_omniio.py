"""
PyTorch Dataset backed by Omni-IO Blob archives.

The modality items and the captions live in separate Blobs; the split index
maps each pair to its item. __getitem__ returns raw decoded data; the caller
runs the frozen encoder on the GPU once per batch (see run_common.py).

Per item:
  image modality : (PIL.Image,                 caption, label)
  audio modality : ((np.ndarray 1-D, sr),      caption, label)
"""
import os
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from omniio.interface import audio_read, image_read, text_read


def _identity_collate(batch):
    """List of (raw, caption, label) -> (list_raw, list_caption, LongTensor)."""
    raws = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return raws, texts, labels


class OmniIODataset(Dataset):
    """
    Args:
        omniio_dir: directory holding {split}_{modality}/, {split}_text/ and
                    {split}_index.parquet as written by ingest_common.write_omniio.
        split:      "train" or "val"
        modality:   "image" or "audio"
    """

    def __init__(self, omniio_dir: str, split: str, modality: str = "image", remap=None):
        """remap=(old_prefix, new_prefix) rewrites the archive paths stored in the
        metadata (absolute at ingest time) so a copy of the archive directory can
        be read in place."""
        assert modality in ("image", "audio"), f"Unsupported modality: {modality}"
        self.modality = modality

        index = pq.read_table(os.path.join(omniio_dir, f"{split}_index.parquet")).to_pandas()
        mod_meta = pq.read_table(
            os.path.join(omniio_dir, f"{split}_{modality}", "metadata.parquet"),
            columns=["id", "path", "start_byte", "end_byte"],
        ).to_pandas().rename(columns={"id": "mod_id"})
        txt_meta = pq.read_table(
            os.path.join(omniio_dir, f"{split}_text", "metadata.parquet"),
            columns=["id", "path", "start_byte", "end_byte"],
        ).to_pandas()

        df = index.merge(mod_meta, on="mod_id", how="left").merge(
            txt_meta, on="id", how="left", suffixes=("_mod", "_txt")
        )
        assert not df["path_mod"].isna().any(), "pair with no modality entry"
        assert not df["path_txt"].isna().any(), "pair with no text entry"

        if remap is not None:
            old, new = remap
            for col in ("path_mod", "path_txt"):
                df[col] = df[col].map(lambda p: new + p[len(old):] if p.startswith(old) else p)
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.mod_paths = df["path_mod"].tolist()
        self.mod_starts = df["start_byte_mod"].to_numpy(dtype=np.int64)
        self.mod_sizes = (df["end_byte_mod"] - df["start_byte_mod"]).to_numpy(dtype=np.int64)
        self.txt_paths = df["path_txt"].tolist()
        self.txt_starts = df["start_byte_txt"].to_numpy(dtype=np.int64)
        self.txt_sizes = (df["end_byte_txt"] - df["start_byte_txt"]).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        if self.modality == "image":
            r = image_read(self.mod_paths[idx], int(self.mod_starts[idx]), int(self.mod_sizes[idx]))
            arr = r.array
            raw = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr).convert("RGB")
        else:
            r = audio_read(self.mod_paths[idx], int(self.mod_starts[idx]), int(self.mod_sizes[idx]))
            raw = (r.array[:, 0], r.sample_rate)
        txt = text_read(self.txt_paths[idx], int(self.txt_starts[idx]), int(self.txt_sizes[idx]))
        return raw, txt.text, int(self.labels[idx])

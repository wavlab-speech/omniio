"""
Shared writers for the contrastive experiments.

A corpus is a set of modality items (images or audio clips) and (item, caption,
label) pairs; an item can have several captions.

Omni-IO layout (per split):
  omniio/{split}_{modality}/      Blob with one entry per item (stored once)
  omniio/{split}_text/            Blob with one zstd entry per pair
  omniio/{split}_index.parquet    id (pair), mod_id (item), label, in storage order

WebDataset layout:
  webdataset/{split}_{NNNN}.tar   one sample per pair: {key}.{ext} + {key}.txt + {key}.json
  Every TAR sample must be self-contained, so an item with five captions is
  stored five times; that is a property of the format, not of this script.

Pairs are written in the order given, which the WebDataset baseline streams
through; callers sort them so that pairs sharing an item or a class are
contiguous, the way collected corpora are laid out.
"""
import io
import json
import os
import tarfile
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from omniio.blob.blob import Blob

Pair = Tuple[str, str, str, int]  # (pair_id, mod_id, caption, label)


def write_omniio(mod_paths: Dict[str, str], pairs: List[Pair], out_dir: str, name: str,
                 modality: str, num_workers: int = 4, **writer_kwargs) -> None:
    """mod_paths: {mod_id: path to the raw image/audio file}."""
    mod_dir = os.path.join(out_dir, f"{name}_{modality}")
    txt_dir = os.path.join(out_dir, f"{name}_text")

    mod_ids = list(mod_paths)
    Blob(mod_dir, modality=modality).append(
        items=[mod_paths[k] for k in mod_ids], ids=mod_ids, num_workers=num_workers, **writer_kwargs
    )
    Blob(txt_dir, modality="text").append(
        items=[p[2] for p in pairs], ids=[p[0] for p in pairs], num_workers=0,
        is_path=False, compression_level=3,
    )
    index = pa.Table.from_pydict({
        "id": [p[0] for p in pairs],
        "mod_id": [p[1] for p in pairs],
        "label": [p[3] for p in pairs],
    })
    pq.write_table(index, os.path.join(out_dir, f"{name}_index.parquet"))
    print(f"  Omni-IO: {len(mod_ids)} items, {len(pairs)} pairs -> {mod_dir}, {txt_dir}")


def write_webdataset(mod_paths: Dict[str, str], pairs: List[Pair], out_dir: str, name: str,
                     ext: str, shard_size: int = 200) -> None:
    cache: Dict[str, bytes] = {}

    def item_bytes(mod_id: str) -> bytes:
        if mod_id not in cache:
            with open(mod_paths[mod_id], "rb") as f:
                cache[mod_id] = f.read()
            if len(cache) > 64:  # bounded: pairs sharing an item are contiguous
                cache.pop(next(iter(cache)))
        return cache[mod_id]

    n_shards = 0
    for shard_idx, start in enumerate(range(0, len(pairs), shard_size)):
        tar_path = os.path.join(out_dir, f"{name}_{shard_idx:04d}.tar")
        with tarfile.open(tar_path, "w") as tar:
            for j, (pair_id, mod_id, caption, label) in enumerate(pairs[start : start + shard_size]):
                key = f"{shard_idx:04d}_{j:04d}"
                for suffix, data in [
                    (ext, item_bytes(mod_id)),
                    ("txt", caption.encode()),
                    ("json", json.dumps({"label": label, "mod_id": mod_id, "pair_id": pair_id}).encode()),
                ]:
                    info = tarfile.TarInfo(name=f"{key}.{suffix}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        n_shards += 1
    print(f"  WebDataset: {n_shards} shards -> {out_dir}/{name}_*.tar")

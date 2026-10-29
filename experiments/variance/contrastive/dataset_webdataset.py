"""
WebDataset-backed loader for the contrastive experiments.

TAR samples hold {key}.{jpg|png|flac} + {key}.txt + {key}.json (see
ingest_common.write_webdataset). The pipeline is the library's own:
shard-level shuffle, a `buffer_size` shuffle buffer over the raw samples,
then decoding, so the buffer holds encoded bytes rather than decoded images.
"""
import io
import json
from typing import List, Union

from PIL import Image
from torch.utils.data import DataLoader

from dataset_omniio import _identity_collate


class _WDSMapper:
    """Picklable: decodes a raw sample into (raw_mod, caption, label)."""

    def __init__(self, modality: str):
        self.modality = modality

    def __call__(self, sample: dict):
        meta = sample["json"]
        if isinstance(meta, bytes):
            meta = json.loads(meta)
        label = int(meta["label"])
        text = sample["txt"]
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        if self.modality == "image":
            data = sample.get("jpg", sample.get("png", sample.get("jpeg")))
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.load()
            return img, text, label

        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(sample["flac"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
        return (arr, sr), text, label


class _PreMap:
    """Picklable: apply a per-item preprocessor to the raw modality of a decoded sample."""

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        raw, text, label = sample
        return self.preprocess(raw), text, label


def make_wds_loader(shard_paths: Union[str, List[str]], buffer_size: int, batch_size: int,
                    modality: str = "image", num_workers: int = 2, preprocess=None,
                    collate_fn=_identity_collate, worker_init_fn=None, persistent: bool = True) -> DataLoader:
    """Each batch is collate_fn applied to a list of (raw_or_features, caption, label).
    With `preprocess`, the frozen encoder's feature extractor runs in the worker
    right after decoding, as it does for the Omni-IO loader."""
    import webdataset as wds

    pipeline = (
        wds.WebDataset(shard_paths, shardshuffle=100, nodesplitter=None)
        .shuffle(buffer_size)
        .map(_WDSMapper(modality))
    )
    if preprocess is not None:
        pipeline = pipeline.map(_PreMap(preprocess))
    dataset = pipeline.batched(batch_size, partial=False, collation_fn=collate_fn)
    kwargs = {"num_workers": 0}
    if num_workers > 0:  # fresh worker interpreters, never forked copies of a big parent
        kwargs = {"num_workers": num_workers, "multiprocessing_context": "forkserver",
                  "persistent_workers": persistent, "worker_init_fn": worker_init_fn}
    return DataLoader(dataset, batch_size=None, **kwargs)

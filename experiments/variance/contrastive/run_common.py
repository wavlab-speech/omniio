"""
Shared pieces of the CLIP (Flickr8k) and CLAP (ESC-50) experiments: the per-item feature
extraction that runs inside the loader workers, the collate, and the DataLoader worker
settings; `features_tensor` unwraps a transformers feature output.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class Preprocessor:
    """Picklable: runs the frozen encoder's feature extractor on one raw item
    inside a loader worker (CLIP image transforms, CLAP mel features), so the
    CPU-bound preprocessing is parallel for every condition alike. Returns a
    dict of per-item tensors; `feature_collate` stacks them into a batch."""

    def __init__(self, modality: str, extractor):
        self.modality = modality
        self.extractor = extractor

    def __call__(self, raw) -> dict:
        if self.modality == "image":
            out = self.extractor(images=raw, return_tensors="pt")
        else:
            arr, sr = raw
            arr = np.asarray(arr.numpy() if torch.is_tensor(arr) else arr, dtype=np.float32).reshape(-1)
            out = self.extractor(arr, sampling_rate=int(sr), return_tensors="pt")  # raw_speech
        return {k: v[0] for k, v in out.items() if torch.is_tensor(v)}


class PreprocessedDataset(Dataset):
    def __init__(self, base: Dataset, pre: Preprocessor):
        self.base = base
        self.pre = pre

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        raw, text, label = self.base[idx]
        return self.pre(raw), text, label


def feature_collate(batch):
    """List of (feature_dict, caption, label) -> (stacked dict, list, LongTensor)."""
    feats = default_collate([b[0] for b in batch])
    texts = [b[1] for b in batch]
    labels = torch.as_tensor([int(b[2]) for b in batch], dtype=torch.long)
    return feats, texts, labels


def _worker_init(_):
    torch.set_num_threads(2)  # fresh interpreters default to every core: 8 workers x 30 threads


def worker_kwargs(num_workers: int) -> dict:
    """DataLoader worker settings. Workers are started with forkserver, never fork:
    a forked worker gradually copies the parent's heap (refcount writes break
    copy-on-write), and with a multi-GB parent and 8 workers per epoch that
    multiplied the job's memory until the OOM killer stopped it."""
    if num_workers <= 0:
        return {"num_workers": 0}
    return {"num_workers": num_workers, "multiprocessing_context": "forkserver",
            "persistent_workers": True, "worker_init_fn": _worker_init}


def features_tensor(out, dim: int) -> torch.Tensor:
    """Unwrap a get_*_features() result: a tensor (transformers 4) or a model
    output whose pooler_output holds the projected features (transformers 5)."""
    if torch.is_tensor(out):
        feat = out
    elif getattr(out, "pooler_output", None) is not None:
        feat = out.pooler_output
    else:
        feat = out[0]
    if feat.ndim != 2 or feat.shape[-1] != dim:
        raise RuntimeError(f"expected projected features of shape (B, {dim}), got {tuple(feat.shape)}")
    return feat / feat.norm(dim=-1, keepdim=True)

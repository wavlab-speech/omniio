"""
Metrics for the contrastive data-loading experiment.

  recall_at_k             label-aware retrieval recall in both directions
  batch_effective_entropy exp(entropy) of the label histogram of a batch
"""
from typing import Dict, Sequence

import numpy as np


def recall_at_k(a_embs: np.ndarray, b_embs: np.ndarray, labels: np.ndarray,
                ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """
    a_embs, b_embs: (N, D) L2-normalised embeddings of the two sides of N pairs.
    labels: (N,) label of each pair. A query is a hit at k if any of its top-k
    retrieved items shares its label: for Flickr8k (label = image) that is the
    standard "any of the five captions" recall; for ESC-50 (label = class) it
    is class retrieval, which is the only well-defined target when the
    captions of a class are identical strings.
    Returns R@k for a->b ("i2t") and b->a ("t2i") and their mean ("R@k").
    """
    labels = np.asarray(labels)
    kmax = max(ks)
    sim = a_embs @ b_embs.T
    out: Dict[str, float] = {}
    for direction, mat in (("i2t", sim), ("t2i", sim.T)):
        top = np.argpartition(-mat, kth=kmax - 1, axis=1)[:, :kmax]
        order = np.argsort(-np.take_along_axis(mat, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        hits = labels[top] == labels[:, None]
        for k in ks:
            out[f"R@{k}_{direction}"] = float(hits[:, :k].any(axis=1).mean())
    for k in ks:
        out[f"R@{k}"] = 0.5 * (out[f"R@{k}_i2t"] + out[f"R@{k}_t2i"])
    return out


def batch_effective_entropy(labels: Sequence[int], n_classes: int) -> float:
    """exp(H) of the label histogram: the effective number of distinct labels."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

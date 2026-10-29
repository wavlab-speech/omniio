"""
Figure 5 (center): batch composition on the real 31-dataset mixture, from the index alone: the corpus in storage
order (datasets one after another, each a run of shard files) fed through a shuffle buffer of
k items, the same stream with the shard files shuffled first (the WebDataset / MDS recipe:
shardshuffle + a k-item sample buffer), vs true random access. Per batch of B: distinct datasets, n_eff over datasets (1 / sum
p^2), n_eff over categories (speech / sound / music), and the fraction of batches drawn from a
single dataset. The buffered stream is simulated exactly on the label stream (reservoir of k
slots, replace-on-draw), sampling 3 000 batches at random positions over the epoch; random
access draws B indices from the whole corpus.

    python bigscale/composition.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data, results  # noqa: E402

IDX = data("bagpiper", "index")


def load_labels():
    """dataset id and category per row in storage order (alphabetical datasets, shards in order)"""
    ds = sorted(f[:-4] for f in os.listdir(IDX) if f.endswith(".npz"))
    labs, cats, sizes, shards = [], [], [], []; base = 0
    for d_id, d in enumerate(ds):
        z = np.load(os.path.join(IDX, d + ".npz")); sh = z["shard"]
        n = len(z["offset"]); order = np.argsort(sh, kind="stable")                # storage order within the dataset
        labs.append(np.full(n, d_id, np.uint8)); cats.append(z["cat"][order]); sizes.append(n)
        shards.append(sh[order].astype(np.int32) + base); base += int(sh.max()) + 1   # global shard-file id per row
    return ds, np.concatenate(labs), np.concatenate(cats), sizes, np.concatenate(shards)


def shard_shuffled_order(shards, rng):
    """row positions of the stream a shard-shuffling reader sees: shard files in random order, rows inside a shard in order"""
    N = len(shards); cuts = np.flatnonzero(np.diff(shards)) + 1; starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [N]])
    perm = rng.permutation(len(starts))
    return np.concatenate([np.arange(starts[i], ends[i]) for i in perm])


def n_eff(x):
    _, c = np.unique(x, return_counts=True); p = c / c.sum(); return 1.0 / np.sum(p * p)


def buffered_batches(labels, k, B, n_batches, rng, n_runs_per_pos=3):
    """exact reservoir-buffer simulation started at random positions: the buffer is filled with
    the k items at that position, then batches are drawn with replace-on-draw from the stream"""
    N = len(labels); out = []
    for _ in range(n_batches // n_runs_per_pos):
        p = int(rng.integers(0, N - k - B * n_runs_per_pos - 1))
        buf = labels[p:p + k].copy(); head = p + k
        for _ in range(n_runs_per_pos):
            slots = rng.integers(0, k, B); batch = buf[slots].copy()
            buf[slots] = labels[head:head + B]; head += B                        # refill from the stream
            out.append(batch)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=256); ap.add_argument("--n_batches", type=int, default=3000)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    a.out = a.out or results("bigscale", "composition.json")
    t0 = time.time(); ds, labels, cats, sizes, shards = load_labels(); N = len(labels)
    print(f"{N:,} rows over {len(ds)} datasets loaded in {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(0); res = {"N": N, "B": a.B, "datasets": ds, "sizes": sizes, "shard_files": int(shards.max()) + 1, "settings": {}}
    t0 = time.time(); ss = shard_shuffled_order(shards, rng); labels_ss = labels[ss]; cats_ss = cats[ss]; del ss
    print(f"shard-shuffled stream over {res['shard_files']:,} shard files built in {time.time() - t0:.0f} s", flush=True)
    corpus_neff_ds = n_eff(labels[rng.integers(0, N, 2_000_000)]); corpus_neff_cat = n_eff(cats[rng.integers(0, N, 2_000_000)])
    res["corpus_n_eff_datasets"] = corpus_neff_ds; res["corpus_n_eff_categories"] = corpus_neff_cat
    ks = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
    settings = [("random_access", None)] + [(f"buffer_{k}", k) for k in ks] + [(f"shardshuffle_buffer_{k}", k) for k in ks]
    for name, k in settings:
        t0 = time.time()
        if k is None:
            batches = [rng.integers(0, N, a.B) for _ in range(a.n_batches)]; lab_b = [labels[b] for b in batches]; cat_b = [cats[b] for b in batches]
        else:
            L, Cc = (labels_ss, cats_ss) if name.startswith("shardshuffle") else (labels, cats)
            idx_b = buffered_batches(np.arange(N, dtype=np.int64), k, a.B, a.n_batches, rng)      # positions, then labels
            lab_b = [L[b] for b in idx_b]; cat_b = [Cc[b] for b in idx_b]
        nd = np.array([len(np.unique(b)) for b in lab_b]); ne = np.array([n_eff(b) for b in lab_b]); nc = np.array([n_eff(b) for b in cat_b])
        row = {"k": k, "shard_shuffle": name.startswith("shardshuffle"), "distinct_datasets_mean": float(nd.mean()), "n_eff_datasets_mean": float(ne.mean()), "n_eff_categories_mean": float(nc.mean()),
               "single_dataset_batches_frac": float((nd == 1).mean()), "batches": len(lab_b), "seconds": time.time() - t0}
        res["settings"][name] = row
        print(f"{name:16s} distinct datasets/batch {row['distinct_datasets_mean']:6.2f}  n_eff(datasets) {row['n_eff_datasets_mean']:6.2f}  n_eff(categories) {row['n_eff_categories_mean']:5.2f}  "
              f"single-dataset batches {row['single_dataset_batches_frac'] * 100:5.1f} %   ({row['seconds']:.0f} s)", flush=True)
    print(f"corpus n_eff: datasets {corpus_neff_ds:.2f} (of {len(ds)}), categories {corpus_neff_cat:.2f}")
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()

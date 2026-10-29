"""Figures 1, 2, 8, 9 (Tabula Sapiens column): shuffle-buffer composition, gradient variance and
training on the Tabula Sapiens `cell` archive (variance/cell/prep_tabula_sapiens.py), source =
(tissue, donor), class = cell ontology class.

    tabula_sapiens.py composition   n_eff, n_distinct, c per loader order over the corpus's storage
                                    order (tissue-major, cells in file order), B = 256, 2 000 batches
    tabula_sapiens.py gradvar       the MLP (2 x 512, GELU) on log1p counts of the 2 000 most variable
                                    genes, at init: r by one-way ANOVA on per-cell gradients (500
                                    cells, 100 sources), R_hat over M = 200 independent batches per
                                    loader vs random access (same theta), R_pred = 1 + r (c - 1)
    tabula_sapiens.py train         held-out accuracy per epoch under each loader order, 3 seeds; cells
                                    are read through the archive for every order (the order is the
                                    variable, the reader is not)

Loader names: omniio_random, wds_buf<k> (WebDataset buffer of k cells over shards of
--shard-size rows of storage order), wds_preshuffled_buf<k>, block1000_buf<k> (block sampling:
1 000-cell blocks, k // 1000 whole blocks per buffer). Rows go to results/variance/cell/tabula_sapiens.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from archive import CellArchive  # noqa: E402
from common.orders import independent_batches, loader_batches, parse_block, parse_k  # noqa: E402
from common.paths import data, results  # noqa: E402
from common.stats import StreamingANOVA, R_hat, composition, concentration, corpus_simpson, predict_R_c, run_lengths  # noqa: E402

JSON_OUT = None
LOADERS = ["omniio_random", "wds_buf1k", "wds_buf10k", "wds_buf100k", "wds_buf1m", "block1000_buf16k", "wds_preshuffled_buf10k"]
TRAIN_LOADERS = ["omniio_random", "wds_buf10k", "block1000_buf16k", "wds_preshuffled_buf10k"]


def log(rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(rec), flush=True)
    with open(JSON_OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ corpus in storage order
class Corpus:
    """The archive's cells in storage order with integer source labels."""

    def __init__(self, archive_dir, label_cols, class_col=None):
        self.arc = CellArchive(archive_dir)
        lab = [self.arc.column(c) for c in label_cols]
        keys = ["|".join(str(x[i]) for x in lab) for i in range(len(self.arc))]
        src = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.labels = np.fromiter((src[k] for k in keys), dtype=np.int64, count=len(keys))   # storage order
        self.n = len(self.arc)
        self.n_sources = len(src)
        self.entry_bytes = float(np.mean(self.arc.sizes))
        self.classes = None
        if class_col:
            cl = self.arc.column(class_col)
            cmap = {c: i for i, c in enumerate(sorted(set(map(str, cl))))}
            self.classes = np.fromiter((cmap[str(c)] for c in cl), dtype=np.int64, count=len(cl))
            self.n_classes = len(cmap)


def cmd_composition(a):
    C = Corpus(a.archive, a.label)
    rl = run_lengths(C.labels, a.shard_size)
    log({"bench": "composition_corpus", "archive": a.archive, "n_cells": C.n, "n_sources": C.n_sources, "label": a.label,
         "corpus_simpson": corpus_simpson(C.labels), "run_lengths": rl, "entry_bytes_mean": C.entry_bytes})
    B = a.batch
    for name in a.loaders:
        rng = np.random.default_rng(a.seed)
        t = time.perf_counter()
        comps = [composition(C.labels[b]) for b in loader_batches(name, C.n, B, a.n_batches, a.shard_size, rng)]
        ne = np.array([c["n_eff"] for c in comps]); nd = np.array([c["n_distinct"] for c in comps]); simp = np.array([c["simpson"] for c in comps])
        k = _buffer_cells(name)
        log({"bench": "composition", "archive": a.archive, "loader": name, "B": B, "n_batches": a.n_batches,
             "n_eff": float(ne.mean()), "n_eff_sd": float(ne.std()), "n_distinct": float(nd.mean()), "c": float(B / ne.mean()),
             "simpson": float(simp.mean()), "buffer_cells": k, "buffer_bytes_per_worker": (k * C.entry_bytes) if k else 0.0,
             "seconds": time.perf_counter() - t})


def _buffer_cells(name):
    if name == "omniio_random":
        return 0
    if name.startswith("block"):
        return parse_block(name)[0]
    return parse_k(name.split("buf")[1].split("_")[0])


# ------------------------------------------------------------------ model + data
def _torch():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    return torch


def make_model(torch, n_in, n_out, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, n_out))


class Feats:
    """log1p(counts) on the top-`hvg` genes by variance (of log1p counts on a 20k-cell sample)."""

    def __init__(self, C: Corpus, hvg: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = C.arc.read_batch(rng.choice(C.n, min(20_000, C.n), replace=False)).astype(np.float32)
        X.data = np.log1p(X.data)
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        self.cols = np.sort(np.argsort(sq - mean ** 2)[::-1][:hvg])
        self.C = C

    def batch(self, torch, idx):
        X = self.C.arc.read_batch(idx, cols=self.cols).astype(np.float32)
        X.data = np.log1p(X.data)
        return torch.from_numpy(X.toarray())


def _flat_grad(torch, model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy()


def cmd_gradvar(a):
    torch = _torch()
    C = Corpus(a.archive, a.label, a.classes)
    F = Feats(C, a.hvg, a.seed)
    model = make_model(torch, a.hvg, C.n_classes, a.seed)
    loss_fn = torch.nn.CrossEntropyLoss()
    d = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(a.seed)
    # r: 500 cells from 100 sources (5 each), sources with >= 5 cells
    srcs, counts = np.unique(C.labels, return_counts=True)
    ok = srcs[counts >= 5]
    pick = rng.choice(ok, min(100, ok.size), replace=False)
    cells, labels = [], []
    for s in pick:
        pos = np.flatnonzero(C.labels == s)
        cells.extend(rng.choice(pos, 5, replace=False).tolist()); labels.extend([int(s)] * 5)
    anova = StreamingANOVA(int(pick.max()) + 1, d)
    X = F.batch(torch, cells); y = torch.from_numpy(C.classes[cells])
    for i in range(len(cells)):
        model.zero_grad()
        loss_fn(model(X[i: i + 1]), y[i: i + 1]).backward()
        anova.add(_flat_grad(torch, model), labels[i])
    r = anova.result()
    log({"bench": "r_estimate", "archive": a.archive, "model_params": d, "hvg": a.hvg, **r})
    # R_hat: M independent batches per loader at the same theta
    coord = rng.choice(d, min(a.coords, d), replace=False)        # a fixed random coordinate subset for the bootstrap
    gbar, comps = {}, {}
    for name in ["omniio_random"] + [l for l in a.loaders if l != "omniio_random"]:
        lrng = np.random.default_rng(a.seed + 1)
        G = np.empty((a.M, coord.size), dtype=np.float32)
        ne = []
        t = time.perf_counter()
        for m, b in enumerate(independent_batches(name, C.n, a.batch, a.M, a.shard_size, lrng)):
            X = F.batch(torch, b); y = torch.from_numpy(C.classes[b])
            model.zero_grad(); loss_fn(model(X), y).backward()
            G[m] = _flat_grad(torch, model)[coord]
            ne.append(composition(C.labels[b]))
        gbar[name] = G
        comps[name] = {"n_eff": float(np.mean([c["n_eff"] for c in ne])), "simpson": float(np.mean([c["simpson"] for c in ne]))}
        print(f"{name}: {a.M} batches in {time.perf_counter() - t:.0f}s", flush=True)
    S_corpus = corpus_simpson(C.labels)
    for name, G in gbar.items():
        rh = R_hat(G, gbar["omniio_random"], n_boot=300, seed=a.seed)
        c = concentration(comps[name]["simpson"], S_corpus, a.batch)
        log({"bench": "gradvar", "archive": a.archive, "loader": name, "B": a.batch, "M": a.M, "c": c, "R_bound": c,
             "simpson_batch": comps[name]["simpson"], "simpson_corpus": S_corpus, "n_eff": comps[name]["n_eff"],
             "R_pred": predict_R_c(r["r_trace"], c), "r_trace": r["r_trace"], "r_coord_mean": r["r"], **rh, "coords": int(coord.size)})


class _BatchDS:
    """One item = one index batch -> (X, y) tensors; the reads run in DataLoader workers while
    the order itself (the variable under test) is generated in the main process."""

    def __init__(self, archive, label, classes, hvg, seed):
        self.args = (archive, label, classes, hvg, seed)
        self.F = None

    def __getitem__(self, idx):
        if self.F is None:
            import torch
            torch.set_num_threads(1)
            C = Corpus(*self.args[:3])
            self.F = Feats(C, self.args[3], self.args[4])
            self.torch = torch
        X = self.F.batch(self.torch, idx)
        y = self.torch.from_numpy(self.F.C.classes[idx])
        return X, y

    def __len__(self):
        return 1 << 30


def _ident(x):
    return x


def cmd_train(a):
    torch = _torch()
    from torch.utils.data import DataLoader
    C = Corpus(a.archive, a.label, a.classes)
    F = Feats(C, a.hvg, 0)
    rng = np.random.default_rng(0)
    held = np.zeros(C.n, dtype=bool); held[rng.choice(C.n, C.n // 10, replace=False)] = True     # 10 % of cells held out
    train_pos = np.flatnonzero(~held); test_pos = np.flatnonzero(held)
    Xte = F.batch(torch, test_pos); yte = torch.from_numpy(C.classes[test_pos])
    ds = _BatchDS(a.archive, a.label, a.classes, a.hvg, 0)
    for name in a.loaders:
        for seed in a.seeds:
            model = make_model(torch, a.hvg, C.n_classes, seed)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            loss_fn = torch.nn.CrossEntropyLoss()
            orng = np.random.default_rng(seed)
            nb = len(train_pos) // a.batch
            for ep in range(a.epochs):
                t = time.perf_counter(); tot = 0.0
                order = [train_pos[b] for b in loader_batches(name, len(train_pos), a.batch, nb, a.shard_size, orng)]   # orders over the training cells' storage order
                dl = DataLoader(ds, sampler=order, batch_size=None, num_workers=a.workers, collate_fn=_ident,
                                multiprocessing_context="forkserver" if a.workers else None, prefetch_factor=4 if a.workers else None)
                model.train()
                for X, y in dl:
                    opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); opt.step(); tot += float(loss.detach())
                model.eval()
                with torch.no_grad():
                    acc = float((model(Xte).argmax(1) == yte).float().mean())
                log({"bench": "train", "archive": a.archive, "loader": name, "seed": seed, "epoch": ep + 1, "train_loss": tot / nb,
                     "held_out_acc": acc, "seconds": time.perf_counter() - t, "B": a.batch, "n_train": int(len(train_pos)), "n_test": int(len(test_pos))})


def main():
    global JSON_OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default=None, help="jsonl of results (default results/variance/cell/tabula_sapiens.jsonl)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("composition", "gradvar", "train"):
        s = sub.add_parser(name)
        s.add_argument("--archive", default=data("cell", "tabula_sapiens")); s.add_argument("--label", nargs="+", default=["organ_tissue", "donor"])
        s.add_argument("--shard-size", type=int, default=10_000); s.add_argument("--seed", type=int, default=0); s.add_argument("--batch", type=int, default=256)
        s.add_argument("--loaders", nargs="+", default=TRAIN_LOADERS if name == "train" else LOADERS)
        if name == "composition":
            s.add_argument("--n-batches", type=int, default=2000)
        else:
            s.add_argument("--classes", default="cell_ontology_class"); s.add_argument("--hvg", type=int, default=2000)
        if name == "gradvar":
            s.add_argument("--M", type=int, default=200); s.add_argument("--coords", type=int, default=200_000)
        if name == "train":
            s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); s.add_argument("--epochs", type=int, default=8); s.add_argument("--lr", type=float, default=1e-3)
            s.add_argument("--workers", type=int, default=8, help="DataLoader workers reading the batches (the order is unchanged)")
    a = p.parse_args()
    JSON_OUT = a.json or results("variance", "cell", "tabula_sapiens.jsonl")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

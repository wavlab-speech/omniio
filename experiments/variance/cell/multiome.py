"""Figures 1, 2, 8, 9 (NeurIPS Multiome column): shuffle-buffer composition, gradient variance and
training on the Multiome cohort of NeurIPS 2021 (variance/cell/prep_neurips2021.py: an RNA and an
ATAC `cell` archive in the same cell order), source = batch (site x donor), class = cell type.
Storage order = release order (the file's cell order).

    multiome.py composition   n_eff, n_distinct, c per loader order, B = 256, 2 000 batches
    multiome.py gradvar       two towers (RNA: log-normalized counts of the 2 000 most variable genes;
                              ATAC: the 20 000 most variable peaks, binarized) -> 512 each, concatenated
                              -> 256 -> cell type; at init: r by one-way ANOVA on per-cell gradients
                              (500 cells over the sources), R_hat over M = 200 independent batches per
                              loader vs random access (R_hat_all over all parameters; per tower too),
                              R_pred = 1 + r (c - 1); 3 seeds
    multiome.py train         held-out accuracy per epoch (raw and EMA weights) under each loader
                              order, 3 seeds, 8 epochs

Loader names as in tabula_sapiens.py. Rows go to results/variance/cell/multiome.jsonl.
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
from common.stats import R_hat, StreamingANOVA, composition, concentration, corpus_simpson, predict_R_c, run_lengths  # noqa: E402

JSON_OUT = None
LOADERS = ["omniio_random", "wds_buf1k", "wds_buf10k", "wds_buf100k", "block1000_buf16k", "wds_preshuffled_buf10k"]
TRAIN_LOADERS = ["omniio_random", "wds_buf10k", "block1000_buf16k", "wds_preshuffled_buf10k"]
SHARD = 1000


def log(rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(rec), flush=True)
    with open(JSON_OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ corpus
class Multi:
    """The two archives in storage order, with labels, classes and feature readers."""

    def __init__(self, corpus_dir, label_col="batch", class_col="cell_type"):
        self.rna = CellArchive(os.path.join(corpus_dir, "rna"))
        self.atac = CellArchive(os.path.join(corpus_dir, "atac"))
        assert len(self.rna) == len(self.atac) and np.array_equal(self.rna.ids[self.rna.order], self.atac.ids[self.atac.order]), "the two archives must hold the same cells in the same order"
        lab = [str(x) for x in self.rna.column(label_col)]
        lmap = {k: i for i, k in enumerate(dict.fromkeys(lab))}
        self.label_names = list(lmap)
        self.labels = np.fromiter((lmap[k] for k in lab), dtype=np.int64, count=len(lab))
        cl = [str(x) for x in self.rna.column(class_col)]
        cmap = {k: i for i, k in enumerate(sorted(set(cl)))}
        self.classes = np.fromiter((cmap[k] for k in cl), dtype=np.int64, count=len(cl))
        self.n_classes = len(cmap)
        self.n = len(self.rna)
        self.feats = None

    def select_features(self, hvg=2000, peaks=20000, seed=0):
        rng = np.random.default_rng(seed)
        samp = rng.choice(self.n, min(20_000, self.n), replace=False)
        X = _lognorm(self.rna.read_batch(samp))
        mean = np.asarray(X.mean(axis=0)).ravel(); sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        rna_cols = np.sort(np.argsort(sq - mean ** 2)[::-1][:hvg])
        A = self.atac.read_batch(samp); A.data[:] = 1                       # binarised
        A = A.astype(np.float32); m = np.asarray(A.mean(axis=0)).ravel()
        atac_cols = np.sort(np.argsort(m * (1 - m))[::-1][:peaks])         # Bernoulli variance
        self.feats = {"rna": rna_cols, "atac": atac_cols, "n_rna": int(rna_cols.size), "n_atac": int(atac_cols.size)}
        return self.feats

    def batch(self, torch, pos):
        """positions in storage order -> dict of dense tensors + labels"""
        pos = np.asarray(pos)
        X = _lognorm(self.rna.read_batch(pos, cols=self.feats["rna"]))
        A = self.atac.read_batch(pos, cols=self.feats["atac"]); A.data[:] = 1
        return {"rna": torch.from_numpy(X.toarray()), "atac": torch.from_numpy(A.toarray().astype(np.float32)), "y": torch.from_numpy(self.classes[pos])}


def _lognorm(X, target=1e4):
    """normalize_total to `target` counts per cell, then log1p (scanpy's default recipe)"""
    import scipy.sparse as sp
    X = sp.csr_matrix(X, dtype=np.float32)
    tot = np.asarray(X.sum(axis=1)).ravel(); tot[tot == 0] = 1
    X = sp.diags((target / tot).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


# ------------------------------------------------------------------ model
def _torch():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    return torch


def make_model(torch, n_rna, n_atac, n_classes, seed):
    """two towers -> 512 -> GELU each, concatenated -> 256 -> head"""
    torch.manual_seed(seed)
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.towers = nn.ModuleDict({"rna": nn.Sequential(nn.Linear(n_rna, 512), nn.GELU()), "atac": nn.Sequential(nn.Linear(n_atac, 512), nn.GELU())})
            self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, n_classes))

        def forward(self, b):
            return self.head(torch.cat([self.towers[t](b[t]) for t in self.towers], dim=1))

    return Net()


def param_slices(model):
    """coordinate ranges of the flattened gradient per module group ('rna', 'atac', 'head')"""
    out, pos = {}, 0
    for name, p in model.named_parameters():
        g = name.split(".")[1] if name.startswith("towers.") else "head"
        n = p.numel()
        out.setdefault(g, [pos, pos])
        out[g][1] = pos + n
        pos += n
    return {k: slice(a, b) for k, (a, b) in out.items()}, pos


def flat_grad(torch, model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).numpy()


def batches_for(name, n, B, n_batches, rng, independent=False):
    return independent_batches(name, n, B, n_batches, SHARD, rng) if independent else loader_batches(name, n, B, n_batches, SHARD, rng)


def buffer_cells(name):
    if name == "omniio_random":
        return 0
    return parse_block(name)[0] if name.startswith("block") else parse_k(name.split("buf")[1].split("_")[0])


# ------------------------------------------------------------------ composition
def cmd_composition(a):
    C = Multi(a.corpus)
    rl = run_lengths(C.labels, SHARD)
    entry = float(np.mean(C.rna.sizes)) + float(np.mean(C.atac.sizes))
    S_corpus = corpus_simpson(C.labels)
    log({"bench": "composition_corpus", "corpus": a.corpus, "n_cells": C.n, "n_sources": len(C.label_names), "label": "batch",
         "corpus_simpson": S_corpus, "run_lengths": rl, "entry_bytes_mean": entry})
    for name in a.loaders:
        rng = np.random.default_rng(a.seed)
        comps = [composition(C.labels[b]) for b in batches_for(name, C.n, a.batch, a.n_batches, rng)]
        ne = np.array([c["n_eff"] for c in comps]); nd = np.array([c["n_distinct"] for c in comps]); simp = np.array([c["simpson"] for c in comps])
        k = buffer_cells(name)
        log({"bench": "composition", "corpus": a.corpus, "loader": name, "B": a.batch, "n_batches": a.n_batches, "n_eff": float(ne.mean()),
             "n_eff_sd": float(ne.std()), "n_distinct": float(nd.mean()), "simpson": float(simp.mean()), "simpson_corpus": S_corpus,
             "c": concentration(float(simp.mean()), S_corpus, a.batch), "buffer_cells": k, "buffer_bytes_per_worker": k * entry})


# ------------------------------------------------------------------ gradient variance
def _r_cells(C, rng, n=500):
    """5 cells per source, all sources, topped up round-robin to n"""
    by = {l: rng.permutation(np.flatnonzero(C.labels == l)).tolist() for l in np.unique(C.labels)}
    cells, labels, k = [], [], 0
    while len(cells) < n:
        progressed = False
        for l, lst in by.items():
            if k < len(lst):
                cells.append(lst[k]); labels.append(int(l)); progressed = True
                if len(cells) >= n:
                    break
        if not progressed:
            break
        k += 1
    return np.asarray(cells), np.asarray(labels)


def cmd_gradvar(a):
    torch = _torch()
    C = Multi(a.corpus); C.select_features(a.hvg, a.peaks)
    S_corpus = corpus_simpson(C.labels)
    loss_fn = torch.nn.CrossEntropyLoss()
    for seed in a.seeds:
        model = make_model(torch, C.feats["n_rna"], C.feats["n_atac"], C.n_classes, seed)
        slices, d = param_slices(model)
        rng = np.random.default_rng(seed)
        # r: trace ratio over all parameters and per tower
        cells, labs = _r_cells(C, rng)
        an = StreamingANOVA(len(C.label_names), d)
        X = C.batch(torch, cells)
        for i in range(len(cells)):
            model.zero_grad()
            loss_fn(model({k: v[i: i + 1] for k, v in X.items()}), X["y"][i: i + 1]).backward()
            an.add(flat_grad(torch, model), labs[i])
        res = {"all": an.result(), **{g: an.result(sl) for g, sl in slices.items()}}
        log({"bench": "r_estimate", "corpus": a.corpus, "seed": seed, "model_params": d, "n_cells": int(len(cells)),
             **{f"{g}_{k}": v for g, r in res.items() for k, v in r.items()}})
        r_trace = res["all"]["r_trace"]
        # R_hat: M independent batches per loader at fixed theta, on fixed coordinate subsets (all parameters, each tower)
        sub = {"all": np.sort(rng.choice(d, min(a.coords, d), replace=False))}
        for g, sl in slices.items():
            idx = np.arange(sl.start, sl.stop); sub[g] = np.sort(rng.choice(idx, min(a.coords, idx.size), replace=False))
        gbar, comps = {}, {}
        for name in ["omniio_random"] + [l for l in a.loaders if l != "omniio_random"]:
            lrng = np.random.default_rng(seed + 1)
            G = {k: np.empty((a.M, v.size), dtype=np.float32) for k, v in sub.items()}
            simp, t = [], time.perf_counter()
            for m, b in enumerate(batches_for(name, C.n, a.batch, a.M, lrng, independent=True)):
                X = C.batch(torch, b)
                model.zero_grad(); loss_fn(model(X), X["y"]).backward()
                g = flat_grad(torch, model)
                for k, v in sub.items():
                    G[k][m] = g[v]
                simp.append(composition(C.labels[b])["simpson"])
            gbar[name] = G; comps[name] = float(np.mean(simp))
            print(f"  seed {seed} {name}: {a.M} batches {time.perf_counter() - t:.0f}s", flush=True)
        for name in gbar:
            c = concentration(comps[name], S_corpus, a.batch)
            rec = {"bench": "gradvar", "corpus": a.corpus, "seed": seed, "loader": name, "B": a.batch, "M": a.M, "c": c, "R_bound": c,
                   "simpson_batch": comps[name], "simpson_corpus": S_corpus, "r_trace": r_trace, "R_pred": predict_R_c(r_trace, c)}
            for k in sub:
                rh = R_hat(gbar[name][k], gbar["omniio_random"][k], n_boot=300, seed=seed)
                rec[f"R_hat_{k}"] = rh["R_hat"]; rec[f"R_hat_{k}_ci"] = rh["R_hat_ci_iid"]; rec[f"bias2_{k}"] = rh["bias2"]; rec[f"coords_{k}"] = int(sub[k].size)
            log(rec)


# ------------------------------------------------------------------ training
class _BatchDS:
    def __init__(self, corpus, hvg, peaks):
        self.corpus, self.hvg, self.peaks, self.C = corpus, hvg, peaks, None

    def __len__(self):
        return 1 << 30

    def __getitem__(self, pos):
        if self.C is None:
            import torch
            torch.set_num_threads(1)
            self.C = Multi(self.corpus); self.C.select_features(self.hvg, self.peaks)
            self.torch = torch
        return self.C.batch(self.torch, pos)


def _ident(x):
    return x


def _stratified_split(labels, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    test = np.zeros(labels.size, bool)
    for l in np.unique(labels):
        idx = np.flatnonzero(labels == l)
        test[rng.choice(idx, max(1, int(round(frac * idx.size))), replace=False)] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


def cmd_train(a):
    torch = _torch()
    from torch.utils.data import DataLoader
    C = Multi(a.corpus); C.select_features(a.hvg, a.peaks)
    tr, te = _stratified_split(C.labels, 0.1, 0)
    Xte = C.batch(torch, te)
    ds = _BatchDS(a.corpus, a.hvg, a.peaks)
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            acc = float((model(Xte).argmax(1) == Xte["y"]).float().mean())
        model.train()
        return acc

    for name in a.loaders:
        for seed in a.seeds:
            model = make_model(torch, C.feats["n_rna"], C.feats["n_atac"], C.n_classes, seed)
            ema = [p.detach().clone() for p in model.parameters()]
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            orng = np.random.default_rng(seed)
            nb = len(tr) // a.batch
            for ep in range(a.epochs):
                t = time.perf_counter(); tot = 0.0
                order = [tr[b] for b in batches_for(name, len(tr), a.batch, nb, orng)]
                dl = DataLoader(ds, sampler=order, batch_size=None, num_workers=a.workers, collate_fn=_ident,
                                multiprocessing_context="forkserver" if a.workers else None, prefetch_factor=4 if a.workers else None)
                for X in dl:
                    opt.zero_grad(); loss = loss_fn(model(X), X["y"]); loss.backward(); opt.step(); tot += float(loss.detach())
                    with torch.no_grad():
                        for e, p in zip(ema, model.parameters()):
                            e.mul_(0.999).add_(p.detach(), alpha=0.001)
                raw = evaluate(model)
                backup = [p.detach().clone() for p in model.parameters()]
                with torch.no_grad():
                    for p, e in zip(model.parameters(), ema):
                        p.copy_(e)
                em = evaluate(model)
                with torch.no_grad():
                    for p, b in zip(model.parameters(), backup):
                        p.copy_(b)
                log({"bench": "train", "corpus": a.corpus, "loader": name, "seed": seed, "epoch": ep + 1, "train_loss": tot / max(1, len(order)),
                     "held_out_acc": raw, "held_out_acc_ema": em, "seconds": time.perf_counter() - t, "B": a.batch, "n_train": int(len(tr)), "n_test": int(len(te))})


def main():
    global JSON_OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default=None, help="jsonl of results (default results/variance/cell/multiome.jsonl)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("composition", "gradvar", "train"):
        s = sub.add_parser(name)
        s.add_argument("--corpus", default=data("cell", "multiome")); s.add_argument("--batch", type=int, default=256); s.add_argument("--seed", type=int, default=0)
        s.add_argument("--loaders", nargs="+", default=TRAIN_LOADERS if name == "train" else LOADERS)
        if name == "composition":
            s.add_argument("--n-batches", type=int, default=2000)
        else:
            s.add_argument("--hvg", type=int, default=2000); s.add_argument("--peaks", type=int, default=20_000); s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
        if name == "gradvar":
            s.add_argument("--M", type=int, default=200); s.add_argument("--coords", type=int, default=200_000)
        if name == "train":
            s.add_argument("--epochs", type=int, default=8); s.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    JSON_OUT = a.json or results("variance", "cell", "multiome.jsonl")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

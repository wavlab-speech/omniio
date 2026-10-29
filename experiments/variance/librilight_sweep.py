"""
Figure 1 (Libri-Light column) and the r of Figure 2's Libri-Light panel: batch composition and
gradient-variance inflation of each loader order on Libri-Light medium cut into 15 s segments
(data/librilight), with a real encoder at init.

Model: the 5.0 M-parameter log-Mel Transformer (audio_encoder.py, tiny) at random init,
seed-matched across loaders. Per-sample loss: contrastive loss of two 1 s crops of the same
utterance against a fixed background of 255 other utterances' crop embeddings (computed once
per seed, no gradient).

Per seed:
  r         per-sample gradients of 500 utterances (100 sources x 5) for the reader and,
            separately, the recording label; one-way ANOVA per coordinate over all 5 M
            coordinates (StreamingANOVA); the trace ratio enters R_pred.
  per loader and B in {32, 256}:
    composition over the first 2 000 batches of an epoch from a freshly filled buffer
            (n_eff, n_distinct, S_batch, c; reader and recording labels);
    R_hat   across-batch variance of the batch-mean gradient over M = 200 independent
            batches (fresh stream, buffer filled, random epoch position, one batch),
            averaged over all coordinates, divided by the same under uniform random
            access; iid bootstrap 95% CI on a fixed 1 M-coordinate subset;
    R_pred = 1 + r (c - 1); R_bound = c; B_eff = B / R_hat.

    python variance/librilight_sweep.py --seed 0 --workers 24
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from audio_encoder import CONFIGS, AudioEncoder  # noqa: E402
from common.harness import write_result  # noqa: E402
from common.orders import independent_batches, loader_batches  # noqa: E402
from common.paths import data, results  # noqa: E402
from common.stats import StreamingANOVA, composition, corpus_simpson, R_hat as R_hat_boot  # noqa: E402

MANIFEST = data("librilight", "seg15", "manifest.parquet")                 # data/librilight/segment_librilight.py
ARCHIVE = data("librilight", "archives", "omniio", "metadata.parquet")   # data/librilight/build_archives.py
RESULTS = results("variance", "librilight")
SR, CROP, SHARD = 16000, 1.0, 1000
LOADERS = ["omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_buf100k", "wds_preshuffled_buf1k"]


class Crops(Dataset):
    """Two random 1 s crops of an utterance, read as time windows of the archive entry."""

    def __init__(self, paths, offs, sizes, durs, seed):
        self.paths, self.offs, self.sizes, self.durs, self.seed = paths, offs, sizes, durs, seed

    def __len__(self):
        return len(self.offs)

    def __getitem__(self, i):
        from omniio.interface import audio_read
        rng = np.random.default_rng((self.seed, int(i)))
        d = float(self.durs[i])
        out = []
        for _ in range(2):
            s = float(rng.uniform(0, max(d - CROP, 0)))
            x = audio_read(self.paths[i], int(self.offs[i]), int(self.sizes[i]), start_time=s, end_time=s + CROP).array[:, 0]
            n = int(CROP * SR)
            if len(x) < n:
                x = np.pad(x, (0, n - len(x)))
            out.append(torch.from_numpy(np.ascontiguousarray(x[:n], dtype=np.float32)))
        return out[0], out[1], int(i)


class ListSampler(torch.utils.data.Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def load_corpus():
    man = pq.read_table(MANIFEST, columns=["id", "speaker", "source", "duration"])
    arc = pq.read_table(ARCHIVE, columns=["id", "path", "start_byte", "end_byte"])
    assert man["id"].to_pylist()[:1000] == arc["id"].to_pylist()[:1000] and man.num_rows == arc.num_rows, "archive order != manifest order"
    spk = man["speaker"].to_numpy(zero_copy_only=False)
    rec = man["source"].to_numpy(zero_copy_only=False)
    _, spk_lab = np.unique(spk, return_inverse=True)
    _, rec_lab = np.unique(rec, return_inverse=True)
    return {"paths": arc["path"].to_pylist(), "offs": arc["start_byte"].to_numpy(), "sizes": (arc["end_byte"].to_numpy() - arc["start_byte"].to_numpy()),
            "durs": man["duration"].to_numpy(), "reader": spk_lab.astype(np.int64), "recording": rec_lab.astype(np.int64), "n": man.num_rows}


def embed_pairs(enc, loader, device):
    out = []
    with torch.no_grad():
        for a, b, _ in loader:
            out.append(enc(b.to(device), torch.full((len(b),), int(CROP * SR), device=device)))
    return torch.cat(out)


def per_sample_loss(enc, a, b, zb_bg, temp, device):
    """Mean over the batch of each utterance's loss against the fixed background (so the
    batch gradient is the mean of per-sample gradients)."""
    L = torch.full((len(a),), int(CROP * SR), device=device)
    za, zb = enc(a.to(device), L), enc(b.to(device), L)
    pos = (za * zb).sum(-1, keepdim=True) * temp                 # (B, 1)
    neg = (za @ zb_bg.T) * temp                                   # (B, 255)
    logits = torch.cat([pos, neg], 1)
    return F.cross_entropy(logits, torch.zeros(len(a), dtype=torch.long, device=device))


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loaders", nargs="+", default=LOADERS)
    ap.add_argument("--B", nargs="+", type=int, default=[32, 256])
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--n_comp", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"
    C = load_corpus()
    n = C["n"]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    enc = AudioEncoder(CONFIGS["tiny"]).to(device)
    params = [p for p in enc.parameters()]
    d = sum(p.numel() for p in params)
    temp = 1 / 0.07
    rng = np.random.default_rng(args.seed)
    wk = {"num_workers": args.workers, "multiprocessing_context": "forkserver", "persistent_workers": False, "prefetch_factor": 4}
    ds = Crops(C["paths"], C["offs"], C["sizes"], C["durs"], seed=args.seed)
    print(f"seed {args.seed}: {n} utterances, {d / 1e6:.2f}M params", flush=True)

    # fixed background: 255 utterances' crop-B embeddings
    bg = rng.choice(n, size=255, replace=False)
    zb_bg = embed_pairs(enc, DataLoader(torch.utils.data.Subset(ds, [int(i) for i in bg]), batch_size=64, **wk), device)

    # ---- r for both labels
    r_path = os.path.join(RESULTS, "r", f"{args.seed}.json")
    if os.path.exists(r_path):
        r_all = {k: v for k, v in json.load(open(r_path)).items() if k in ("reader", "recording")}
    else:
        r_all = {}
        for label in ("reader", "recording"):
            lab = C[label]
            srcs, counts = np.unique(lab, return_counts=True)
            pick = rng.choice(srcs[counts >= 5], size=100, replace=False)
            idx = np.concatenate([rng.choice(np.flatnonzero(lab == s), size=5, replace=False) for s in pick])
            label_of = {int(s): k for k, s in enumerate(pick)}
            anova = StreamingANOVA(100, d)
            t0 = time.time()
            dl = DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=1, **wk)
            for k, (a, b, i) in enumerate(dl):
                loss = per_sample_loss(enc, a, b, zb_bg, temp, device)
                enc.zero_grad(set_to_none=True); loss.backward()
                anova.add(flat_grad(params).cpu().numpy(), label_of[int(lab[int(i)])])
            r_all[label] = {**anova.result(save_components=os.path.join(RESULTS, "r", f"{args.seed}_{label}_components.npz")), "n_params": d, "seconds": time.time() - t0}
            print(f"  r[{label}] = {r_all[label]['r']:.4f} [{r_all[label]['r_p10']:.3f}, {r_all[label]['r_p90']:.3f}] ({time.time() - t0:.0f} s)", flush=True)
        write_result(r_path, {"seed": args.seed, **r_all, "background": 255, "crop_s": CROP})

    S_corpus = {lab: corpus_simpson(C[lab]) for lab in ("reader", "recording")}
    coord_idx = torch.from_numpy(np.sort(np.random.default_rng(12345).choice(d, size=min(1_000_000, d), replace=False))).to(device)
    rand_stats = {}
    for name in ["omniio_random"] + [l for l in args.loaders if l != "omniio_random"]:
        for B in args.B:
            out_path = os.path.join(RESULTS, f"{name}_B{B}", f"{args.seed}.json")
            ref_path = os.path.join(RESULTS, f"omniio_random_B{B}", f"{args.seed}_ref.npz")
            if os.path.exists(out_path) and name != "omniio_random":
                print(f"skip {name} B={B} (done)", flush=True); continue
            if name == "omniio_random" and os.path.exists(out_path) and os.path.exists(ref_path):
                z = np.load(ref_path)
                rand_stats[B] = {"var_all": float(z["var_all"]), "sub": z["sub"], "simpson": json.load(open(out_path))["composition"]}
                print(f"skip omniio_random B={B} (reference cached)", flush=True); continue
            lock = out_path + ".lock"
            try:
                os.makedirs(os.path.dirname(lock), exist_ok=True); os.mkdir(lock)
            except FileExistsError:
                if time.time() - os.path.getmtime(lock) < 3600:   # a lock older than an hour belongs to a killed job
                    if name == "omniio_random":          # the reference is being computed elsewhere: wait for it
                        t_w = time.time()
                        while not (os.path.exists(ref_path) and os.path.exists(out_path)) and time.time() - t_w < 3600:
                            time.sleep(30)
                        if os.path.exists(ref_path):
                            z = np.load(ref_path)
                            rand_stats[B] = {"var_all": float(z["var_all"]), "sub": z["sub"], "simpson": json.load(open(out_path))["composition"]}
                            print(f"omniio_random B={B}: reference from the other job", flush=True); continue
                        raise RuntimeError("reference never appeared")
                    print(f"skip {name} B={B} (claimed by another job)", flush=True); continue
                os.utime(lock)
            t0 = time.time()
            # composition over the first n_comp batches of a fresh epoch
            comp_rng = np.random.default_rng((args.seed, B, 7))
            comps = {lab: [] for lab in ("reader", "recording")}
            for b in loader_batches(name, n, B, args.n_comp, SHARD, comp_rng):
                for lab in comps:
                    comps[lab].append(composition(C[lab][b]))
            comp = {lab: {k: float(np.mean([c[k] for c in comps[lab]])) for k in ("n_eff", "n_distinct", "n_95", "simpson")} for lab in comps}
            row = {"experiment": "librilight_sweep", "loader": name, "B": B, "seed": args.seed, "M": args.M, "n_comp": args.n_comp, "shard_size": SHARD,
                   "composition": comp, "S_corpus": S_corpus, "r": {lab: r_all[lab]["r"] for lab in r_all}}
            if name == "wds_preshuffled_buf1k":
                comps2 = [composition(C["reader"][b]) for b in loader_batches(name, n, B, args.n_comp, SHARD, np.random.default_rng((args.seed, B, 8)))]
                row["composition_epoch2_reader"] = {k: float(np.mean([c[k] for c in comps2])) for k in ("n_eff", "n_distinct", "simpson")}
            # R_hat batches: M independent batches, batch-mean gradient each
            batches = [np.asarray(b, dtype=np.int64) for b in independent_batches(name, n, B, args.M, SHARD, np.random.default_rng((args.seed, B, 9)))]
            dl = DataLoader(ds, batch_sampler=ListSampler([list(map(int, b)) for b in batches]), **wk)
            s1 = torch.zeros(d, dtype=torch.float64, device=device); s2 = torch.zeros(d, dtype=torch.float64, device=device)
            sub = np.empty((args.M, coord_idx.numel()), dtype=np.float32)
            enc.train()
            for k, (a, b, _) in enumerate(dl):
                loss = per_sample_loss(enc, a, b, zb_bg, temp, device)
                enc.zero_grad(set_to_none=True); loss.backward()
                g = flat_grad(params)
                s1 += g.double(); s2 += g.double() ** 2
                sub[k] = g[coord_idx].cpu().numpy()
            enc.zero_grad(set_to_none=True)
            var_all = float(((s2 - s1 ** 2 / args.M) / (args.M - 1)).mean())
            row["batch_var_all_coords"] = var_all
            row["batches_file"] = os.path.join(RESULTS, f"{name}_B{B}", f"{args.seed}_batches.npy")
            os.makedirs(os.path.dirname(row["batches_file"]), exist_ok=True)
            np.save(row["batches_file"], np.stack(batches))
            if name == "omniio_random":
                rand_stats[B] = {"var_all": var_all, "sub": sub, "simpson": comp}
                np.savez(ref_path, var_all=np.float64(var_all), sub=sub)
                row.update({"R_hat": 1.0, "R_hat_ci": [1.0, 1.0], "R_pred": 1.0, "R_bound": 1.0, "c": {lab: 1.0 for lab in comp}, "B_eff": float(B)})
            else:
                rs = rand_stats[B]
                row["R_hat"] = var_all / rs["var_all"]
                boot = R_hat_boot(sub, rs["sub"], seed=args.seed)
                row["R_hat_subset"], row["R_hat_ci"] = boot["R_hat"], boot["R_hat_ci_iid"]
                row["c"], row["R_pred"], row["R_bound"] = {}, {}, {}
                for lab in comp:
                    kappa = comp[lab]["simpson"] - S_corpus[lab]
                    kappa_rand = rs["simpson"][lab]["simpson"] - S_corpus[lab]
                    c = kappa / kappa_rand
                    row["c"][lab] = c; row["R_bound"][lab] = c
                    row["R_pred"][lab] = 1 + r_all[lab]["r_trace"] * (c - 1)                 # trace-ratio r
                    row.setdefault("R_pred_coordmean", {})[lab] = 1 + r_all[lab]["r"] * (c - 1)
                row["kappa_rand_formula"] = {lab: (1 - S_corpus[lab]) / B for lab in comp}
                row["B_eff"] = B / row["R_hat"]
            row["seconds"] = time.time() - t0
            write_result(out_path, row)
            try:
                os.rmdir(lock)
            except OSError:
                pass
            rp = row["R_pred"]["reader"] if isinstance(row["R_pred"], dict) else row["R_pred"]
            print(f"  {name:24s} B={B:3d}  n_eff {comp['reader']['n_eff']:6.1f} distinct {comp['reader']['n_distinct']:6.1f}  "
                  f"R_hat {row['R_hat']:7.2f} {row.get('R_hat_ci')}  R_pred {rp:6.2f}  ({row['seconds']:.0f} s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

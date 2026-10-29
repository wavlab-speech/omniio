"""
Figures 1, 2 and 8 (ESC-50 and Flickr8k panels): contrastive training with frozen encoders
under each loader, with the init-time quantities of Section 3.

Loaders: omniio_random (DataLoader shuffle=True over Omni-IO), wds_buf1k and wds_buf100
(the WebDataset pipeline over the storage-order shards), wds_buf10k (Flickr8k), and
wds_preshuffled_buf1k (the pairs globally permuted once with seed 12345, written as new
shards, then the same pipeline with a 1 000-item buffer).

At initialization, for each loader and seed, with the same head parameters:
  M batches -> batch-mean head gradients gbar, their composition (n_eff, n_distinct),
  R_hat against omniio_random with a bootstrap CI; r by one-way ANOVA on per-sample head
  gradients, each sample scored against one fixed background of negatives (InfoNCE has no
  per-sample gradient otherwise); R_pred. Then 20 epochs of training under that loader,
  label-aware R@1 on the validation pairs per epoch.

    python variance/contrastive/contrastive.py --dataset clap --seeds 0 1 2
    python variance/contrastive/contrastive.py --dataset clip --seeds 0 1 2
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.harness import write_result  # noqa: E402
from common.orders import PRESHUFFLE_SEED  # noqa: E402
from common.paths import data, results  # noqa: E402
from common.stats import R_hat, bound_R, bound_R_fp, composition, corpus_simpson, estimate_r, predict_R, predict_R_fp  # noqa: E402
from dataset_omniio import OmniIODataset  # noqa: E402
from dataset_webdataset import make_wds_loader  # noqa: E402
from ingest_common import write_webdataset  # noqa: E402
from loss import infonce_loss  # noqa: E402
from model import ContrastiveModel  # noqa: E402
from omniio.interface import text_read  # noqa: E402
from run_common import PreprocessedDataset, _worker_init, feature_collate, worker_kwargs  # noqa: E402
from train import measure_grad_variance, train_one_run  # noqa: E402

DATASETS = {   # out_dir: written by ingest_esc50.py / ingest_flickr8k.py
    "clap": {"out_dir": data("esc50"), "modality": "audio", "ext": "flac", "batch": 128, "module": "encoder_clap",
             "loaders": ["omniio_random", "wds_buf1k", "wds_buf100", "wds_preshuffled_buf1k"]},
    "clip": {"out_dir": data("flickr8k"), "modality": "image", "ext": "jpg", "batch": 256, "module": "encoder_clip",
             "loaders": ["omniio_random", "wds_buf1k", "wds_buf100", "wds_buf10k", "wds_preshuffled_buf1k"]},
}


def preshuffled_shards(cfg, shard_size=200):
    """Write train shards of the pairs in a global random order (seed 12345) once."""
    out = os.path.join(cfg["out_dir"], "webdataset_preshuffled")
    done = sorted(glob.glob(os.path.join(out, "train_*.tar")))
    if done:
        return done, None
    os.makedirs(out, exist_ok=True)
    om = os.path.join(cfg["out_dir"], "omniio")
    idx = pq.read_table(os.path.join(om, "train_index.parquet")).to_pylist()
    txt = pq.read_table(os.path.join(om, "train_text", "metadata.parquet"), columns=["id", "path", "start_byte", "end_byte"]).to_pylist()
    cap = {r["id"]: text_read(r["path"], r["start_byte"], r["end_byte"] - r["start_byte"]).text for r in txt}
    raw = os.path.join(cfg["out_dir"], "raw")
    pairs = [(r["id"], r["mod_id"], cap[r["id"]], int(r["label"])) for r in idx]
    mod_paths = {}
    for r in idx:
        for ext in (cfg["ext"], "jpeg", "png", "flac", "wav"):
            p = os.path.join(raw, f"{r['mod_id']}.{ext}")
            if os.path.exists(p):
                mod_paths[r["mod_id"]] = p
                break
    assert len(mod_paths) == len({r["mod_id"] for r in idx}), "raw file missing for some items"
    perm = np.random.default_rng(PRESHUFFLE_SEED).permutation(len(pairs))
    t0 = time.time()
    write_webdataset(mod_paths, [pairs[i] for i in perm], out, "train", cfg["ext"], shard_size)
    cost = {"seed": PRESHUFFLE_SEED, "pairs": len(pairs), "wall_s": time.time() - t0,
            "bytes": sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "train_*.tar")))}
    json.dump(cost, open(os.path.join(out, "preshuffle_cost.json"), "w"), indent=1)
    return sorted(glob.glob(os.path.join(out, "train_*.tar"))), cost


def make_loader(name, cfg, pre, modality, seed, num_workers, device):
    om = os.path.join(cfg["out_dir"], "omniio")
    B = cfg["batch"]
    if name == "omniio_random":
        return DataLoader(PreprocessedDataset(OmniIODataset(om, "train", modality=modality), pre), batch_size=B, shuffle=True,
                          drop_last=True, pin_memory=(device == "cuda"), collate_fn=feature_collate,
                          generator=torch.Generator().manual_seed(seed), **worker_kwargs(num_workers))
    if name.startswith("wds_preshuffled_buf"):
        shards, _ = preshuffled_shards(cfg)
        k = int(float(name.split("buf")[1].replace("k", "")) * (1000 if name.endswith("k") else 1))
    else:
        shards = sorted(glob.glob(os.path.join(cfg["out_dir"], "webdataset", "train_*.tar")))
        k = int(float(name.split("buf")[1].replace("k", "")) * (1000 if name.endswith("k") else 1))
    return make_wds_loader(shards, buffer_size=k, batch_size=B, modality=modality, num_workers=num_workers,
                           preprocess=pre, collate_fn=feature_collate, worker_init_fn=_worker_init)


@torch.no_grad()
def embed(ds, pre, encoder, indices, device, modality):
    """Frozen-encoder embeddings of the given pairs: (emb_a, emb_b, labels)."""
    sub = torch.utils.data.Subset(PreprocessedDataset(ds, pre), [int(i) for i in indices])
    dl = DataLoader(sub, batch_size=128, shuffle=False, collate_fn=feature_collate, **worker_kwargs(4))
    A, Bm, L = [], [], []
    for batch in dl:
        a, b, lab = encoder(batch, device)
        A.append(a); Bm.append(b); L.append(lab)
    return torch.cat(A), torch.cat(Bm), torch.cat(L)


def per_sample_r(model, ds, pre, encoder, device, modality, seed, n_sources=200, per_source=5, n_background=255):
    """r on the head gradients: per-sample InfoNCE loss of sample i against a
    fixed background set of negatives (symmetric), gradient w.r.t. the head."""
    rng = np.random.default_rng(seed)
    labels = ds.labels
    srcs, counts = np.unique(labels, return_counts=True)
    ok = srcs[counts >= per_source]
    pick = rng.choice(ok, size=min(n_sources, len(ok)), replace=False)
    per_source = max(per_source, int(np.ceil(500 / len(pick))))  # at least 500 samples over the picked sources
    pick = pick[counts[np.searchsorted(srcs, pick)] >= per_source] if per_source > 5 else pick
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == s), size=per_source, replace=False) for s in pick])
    bg = rng.choice(np.setdiff1d(np.arange(len(labels)), idx), size=n_background, replace=False)
    a_s, b_s, lab_s = embed(ds, pre, encoder, idx, device, modality)
    a_bg, b_bg, _ = embed(ds, pre, encoder, bg, device, modality)
    params = [p for p in model.parameters()]
    grads = []
    model.train()
    for i in range(len(idx)):
        pa, pb = model(torch.cat([a_s[i:i + 1], a_bg]), torch.cat([b_s[i:i + 1], b_bg]))
        sim = (pa @ pb.T) * model.temp
        loss = 0.5 * (torch.nn.functional.cross_entropy(sim[0:1], torch.zeros(1, dtype=torch.long, device=device))
                      + torch.nn.functional.cross_entropy(sim[:, 0:1].T, torch.zeros(1, dtype=torch.long, device=device)))
        g = torch.autograd.grad(loss, params)
        grads.append(torch.cat([x.reshape(-1) for x in g]).cpu().numpy())
    model.zero_grad(set_to_none=True)
    return estimate_r(np.stack(grads), lab_s.cpu().numpy()), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--loaders", nargs="+", default=None, help="default: the dataset's loaders")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--results_root", default=results("contrastive"))
    ap.add_argument("--no_train", action="store_true", help="init measurements only")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]; args.loaders = args.loaders or cfg["loaders"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = __import__(cfg["module"])
    encoder, emb_dim, pre = mod.load_encoder(device)
    modality = cfg["modality"]
    om = os.path.join(cfg["out_dir"], "omniio")
    n_classes = json.load(open(os.path.join(cfg["out_dir"], "meta.json")))["n_classes"]
    train_ds = OmniIODataset(om, "train", modality=modality)
    S_corpus = corpus_simpson(train_ds.labels)
    val_loader = DataLoader(PreprocessedDataset(OmniIODataset(om, "val", modality=modality), pre), batch_size=256, shuffle=False,
                            pin_memory=(device == "cuda"), collate_fn=feature_collate, **worker_kwargs(args.num_workers))
    _, cost = preshuffled_shards(cfg) if any("preshuffled" in l for l in args.loaders) else (None, None)
    exp = args.dataset

    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model0 = ContrastiveModel(in_dim=emb_dim, out_dim=128).to(device)
        init_state = {k: v.clone() for k, v in model0.state_dict().items()}
        r_est, n_r = per_sample_r(model0, train_ds, pre, encoder, device, modality, seed)
        print(f"seed {seed}: r = {r_est['r']:.4f} [{r_est['r_p10']:.3f}, {r_est['r_p90']:.3f}] trace {r_est['r_trace']:.4f} from {n_r} per-sample gradients", flush=True)
        gbar_rand, rand_row = None, None
        for name in (["omniio_random"] + [l for l in args.loaders if l != "omniio_random"]):
            out_path = os.path.join(args.results_root, exp, name, f"{seed}.json")
            if os.path.exists(out_path) and name != "omniio_random":
                print(f"skip {name} seed {seed} (done)"); continue
            if name == "omniio_random" and os.path.exists(out_path) and gbar_rand is None:
                pass  # the random reference is always recomputed (cheap) so gbar_rand is available
            print(f"\n=== {args.dataset} {name} seed {seed}", flush=True)
            model = ContrastiveModel(in_dim=emb_dim, out_dim=128).to(device)
            model.load_state_dict(init_state)
            loader = make_loader(name, cfg, pre, modality, seed, args.num_workers, device)
            gv = measure_grad_variance(model, loader, torch.device(device), encoder, args.M, return_grads=True)
            gbar = gv.pop("gbar")
            row = {"experiment": exp, "dataset": args.dataset, "loader": name, "seed": seed, "B": cfg["batch"], "M": gv["n_batches"],
                   "init": {k: v for k, v in gv.items()}, "r": r_est, "simpson_corpus": S_corpus, "n_eff_corpus": 1 / S_corpus}
            if name == "omniio_random":
                gbar_rand, rand_row = gbar, row
                row["R_hat"], row["R_pred"], row["R_bound"], row["R_pred_fp"], row["R_bound_fp"], row["bias2"], row["MSE_ratio"] = 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0
            else:
                rh = R_hat(gbar, gbar_rand, seed=seed)
                row.update(rh)
                ne, ner = gv["n_eff"], rand_row["init"]["n_eff"]
                row["R_pred"] = predict_R(r_est["r"], ne, ner, cfg["batch"]); row["R_bound"] = bound_R(ne, ner)
                row["R_pred_fp"] = predict_R_fp(r_est["r"], gv["simpson"], rand_row["init"]["simpson"], S_corpus, cfg["batch"])
                row["R_bound_fp"] = bound_R_fp(gv["simpson"], rand_row["init"]["simpson"], S_corpus)
                print(f"  init: n_eff {ne:.1f} (rand {ner:.1f}) n_distinct {gv['n_distinct']:.1f}  R_hat {rh['R_hat']:.2f} {rh['R_hat_ci']}  "
                      f"R_pred {row['R_pred']:.2f} fp {row['R_pred_fp']:.2f}  R_bound {row['R_bound']:.2f} fp {row['R_bound_fp']:.2f}  "
                      f"bias2 {rh['bias2']:.3f} MSE_ratio {rh['MSE_ratio']:.2f}", flush=True)
            if name == "omniio_random" and os.path.exists(out_path):
                print(f"  omniio_random seed {seed}: reference gradients re-measured; training result kept from {out_path}", flush=True)
                continue
            if not args.no_train:
                model.load_state_dict(init_state)
                loader = make_loader(name, cfg, pre, modality, seed, args.num_workers, device)
                run = train_one_run(loader, val_loader, n_classes, encoder, in_dim=emb_dim, n_epochs=args.epochs, lr=args.lr,
                                    seed=seed, device_str=device, n_var_batches=2)
                row["train"] = {k: v for k, v in run.items() if k not in ("grad_var_init", "grad_var_batches")}
            if cost:
                row["preshuffle_cost"] = cost
            write_result(out_path, row)
    print("done", flush=True)


if __name__ == "__main__":
    main()

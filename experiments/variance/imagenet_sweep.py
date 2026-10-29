"""
Figure 1 (ImageNet column): batch composition and gradient-variance inflation of each loader
order on ImageNet-1k train (the 256 px copy, 1 281 167 JPEGs; source = class; items in
(class, file name) order, i.e. folder order; model torchvision resnet18 at init, cross-entropy).

Stages (idempotent):
  ingest  an Omni-IO archive of the JPEG bytes in storage order, with the class as `source`
  sweep   composition for omniio_random / wds_buf100 / wds_buf1k / wds_buf10k /
          wds_preshuffled_buf1k at B = 256 and B = 32 (index only); r from 500 items
          (100 sources x 5) by one-way ANOVA on per-sample gradients; R_hat at wds_buf1k,
          M = 200 independent batches against M = 200 random batches, same parameters;
          R_pred = 1 + r (c - 1)
  rhat    R_hat for the other buffer sizes (wds_buf100, wds_buf10k, wds_buf100k)

    python variance/imagenet_sweep.py --stage ingest --workers 40
    python variance/imagenet_sweep.py --stage sweep --seed 0
    python variance/imagenet_sweep.py --stage rhat --seed 0
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.harness import write_result  # noqa: E402
from common.orders import independent_batches, loader_batches  # noqa: E402
from common.paths import data, results  # noqa: E402
from common.stats import StreamingANOVA, composition, corpus_simpson, R_hat as R_hat_boot  # noqa: E402

SRC = data("sources", "imagenet256", "data")           # data/download_formats.py
ARCH = data("imagenet", "archive")
RESULTS = results("variance", "imagenet")
CFG = {"B": 256, "shard": 1000}
LOADERS = ["omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_preshuffled_buf1k"]
BIN = 320 << 20


# ---------------------------------------------------------------- ingest
def _write_bins(out_dir, entries, wid):
    """entries: iterable of (id, source, bytes, extra dict). Writes w<wid>_<k>.bin; returns metadata rows."""
    meta, k, cur, path, written = [], 0, None, None, 0

    def new_bin():
        nonlocal cur, path, k, written
        if cur:
            cur.close()
        path = os.path.join(out_dir, f"w{wid:03d}_{k:05d}.bin"); cur, k, written = open(path, "wb"), k + 1, 0

    new_bin()
    for id_, src, data, extra in entries:
        if written >= BIN:
            new_bin()
        cur.write(data)
        meta.append({"id": id_, "source": src, "start_byte": written, "end_byte": written + len(data), "bin_path": path, **extra})
        written += len(data)
    cur.close()
    return meta


def _finish(out_dir, metas):
    all_meta, bin_index = [], 0
    for meta in metas:
        seen = []
        for m in meta:
            if m["bin_path"] not in seen:
                seen.append(m["bin_path"])
        for p in seen:
            final = os.path.join(out_dir, f"blob_{bin_index}.bin"); os.rename(p, final)
            for m in meta:
                if m["bin_path"] == p:
                    m["path"], m["bin_index"] = final, bin_index
            bin_index += 1
        all_meta.extend(meta)
    for m in all_meta:
        m.pop("bin_path")
    pq.write_table(pa.Table.from_pylist(all_meta), os.path.join(out_dir, "metadata.parquet"))
    return len(all_meta), bin_index


def _imagenet_worker(job):
    files, out_dir, wid = job
    def entries():
        for f in files:
            t = pq.read_table(f, columns=["image", "label"])
            for r in t.to_pylist():
                yield (f"{int(r['label'])}/{r['image']['path'] or ''}", int(r["label"]), r["image"]["bytes"], {"format": "jpeg"})
    return _write_bins(out_dir, entries(), wid)


def ingest_imagenet(out_dir, workers):
    """The parquet copy is already in (class, file name) order, i.e. folder order, and the
    files are sorted; each worker writes its contiguous slice of files and the worker bins
    are concatenated in order."""
    files = sorted(glob.glob(os.path.join(SRC, "train-*.parquet")))
    chunk = (len(files) + workers - 1) // workers
    jobs = [(files[i:i + chunk], out_dir, k) for k, i in enumerate(range(0, len(files), chunk))]
    with ProcessPoolExecutor(len(jobs)) as ex:
        metas = list(ex.map(_imagenet_worker, jobs))
    return _finish(out_dir, metas)


def cmd_ingest(args):
    out = ARCH
    if os.path.exists(os.path.join(out, "metadata.parquet")):
        print("already ingested", out); return
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    n, bins = ingest_imagenet(out, args.workers)
    res = {"corpus": "imagenet", "entries": n, "bins": bins, "seconds": time.time() - t0, "bytes": sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "blob_*.bin")))}
    write_result(os.path.join(RESULTS, "ingest.json"), res); print(res, flush=True)


# ---------------------------------------------------------------- items and models
class Items(Dataset):
    """an ImageNet entry: the 256 px JPEG from the archive, center-cropped to 224 and normalized"""

    def __init__(self, meta):
        self.paths, self.offs, self.sizes = meta["path"].to_pylist(), meta["start_byte"].to_numpy(), meta["end_byte"].to_numpy() - meta["start_byte"].to_numpy()
        self.labels = meta["source"].to_numpy()

    def __len__(self):
        return len(self.offs)

    def __getitem__(self, i):
        from PIL import Image
        from omniio.interface import image_read
        p, o, s = self.paths[i], int(self.offs[i]), int(self.sizes[i])
        arr = image_read(p, o, s).array
        im = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr).convert("RGB")
        w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
        x = torch.from_numpy(np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, int(self.labels[i]), int(i)


def make_model(device):
    import torchvision
    return torchvision.models.resnet18(weights=None).to(device)


def backward_batch(model, x, y, device):
    """the batch-mean gradient of the cross-entropy"""
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x.to(device, non_blocking=True)), y.to(device)); loss.backward(); return float(loss)


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


class ListSampler(torch.utils.data.Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# ---------------------------------------------------------------- sweep
def _rss_monitor(tag="rss"):
    """Print the process tree's resident memory every 20 s (the sweeps were OOM-killed without a trace)."""
    import threading
    import psutil

    def run():
        me = psutil.Process()
        while True:
            try:
                tot = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                print(f"  [{tag}] {tot / 1e9:.1f} GB", flush=True)
            except Exception:
                pass
            time.sleep(20)
    threading.Thread(target=run, daemon=True).start()


def cmd_sweep(args):
    _rss_monitor()
    cfg = CFG
    meta = pq.read_table(os.path.join(ARCH, "metadata.parquet"))
    labels = meta["source"].to_numpy().astype(np.int64)
    n = len(labels)
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = make_model(device)
    # eval mode: BatchNorm with batch statistics makes a single-sample forward input-independent
    # (per-sample gradients collapse to label-only) and couples the batch's per-sample gradients
    # through the batch statistics. With running statistics (mean 0, var 1 at init) the per-sample
    # gradients are additive, which is what the model check assumes. ResNet-18 has no dropout.
    model.eval()
    params = [p for p in model.parameters()]; d = sum(p.numel() for p in params)
    ds = Items(meta)
    wk = {"num_workers": args.workers, "multiprocessing_context": "forkserver", "persistent_workers": False, "prefetch_factor": 4}
    S_corpus = corpus_simpson(labels)
    runs = np.diff(np.flatnonzero(np.diff(labels) != 0), prepend=-1)
    lam = float(np.mean(np.minimum(runs, cfg["shard"]))) if len(runs) else float(n)
    out = {"corpus": "imagenet", "seed": args.seed, "n": n, "n_sources": int(labels.max() + 1), "S_corpus": S_corpus, "lambda_capped": lam,
           "shard": cfg["shard"], "n_params": d, "loaders": {}}
    print(f"imagenet: {n} items, {out['n_sources']} sources, lambda {lam:.1f}, {d / 1e6:.1f}M params", flush=True)
    # composition, index only
    for name in LOADERS:
        for B in (cfg["B"], 32):
            comps = [composition(labels[b]) for b in loader_batches(name, n, B, args.n_comp, cfg["shard"], np.random.default_rng((args.seed, B, 7)))]
            out["loaders"][f"{name}_B{B}"] = {k: float(np.mean([c[k] for c in comps])) for k in ("n_eff", "n_distinct", "n_95", "simpson")}
    for B in (cfg["B"], 32):
        kr = out["loaders"][f"omniio_random_B{B}"]["simpson"] - S_corpus
        for name in LOADERS:
            key = f"{name}_B{B}"; k = out["loaders"][key]["simpson"] - S_corpus
            out["loaders"][key]["c"] = k / kr; out["loaders"][key]["R_bound"] = k / kr; out["loaders"][key]["kappa_rand_formula"] = (1 - S_corpus) / B
    # r: 500 items, 100 sources x 5
    rng = np.random.default_rng(args.seed)
    srcs, counts = np.unique(labels, return_counts=True)
    pick = rng.choice(srcs[counts >= 5], size=100, replace=False)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == s), size=5, replace=False) for s in pick])
    label_of = {int(s): k for k, s in enumerate(pick)}
    # r over a fixed random subset of coordinates for the large models: the ANOVA holds
    # 100 sources x d float64 sums (40 GB for d = 50 M); a 5 M-coordinate subset gives an
    # unbiased estimate of the coordinate mean and of the percentiles
    r_idx = torch.from_numpy(np.sort(np.random.default_rng(777).choice(d, size=min(5_000_000, d), replace=False))).to(device)
    anova = StreamingANOVA(100, int(r_idx.numel())); t0 = time.time()
    for x, y, i in DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=1, **wk):
        backward_batch(model, x, y, device)
        anova.add(flat_grad(params)[r_idx].cpu().numpy(), label_of[int(labels[int(i)])])
    out["r"] = anova.result(save_components=os.path.join(RESULTS, f"{args.seed}_r_components.npz")); out["r"]["seconds"] = time.time() - t0; out["r"]["n_coords"] = int(r_idx.numel())
    print(f"  r = {out['r']['r']:.4f} [{out['r']['r_p10']:.3f}, {out['r']['r_p90']:.3f}] ({time.time() - t0:.0f} s)", flush=True)
    # R_hat at wds_buf1k, B = cfg B
    B = cfg["B"]
    coord_idx = torch.from_numpy(np.sort(np.random.default_rng(12345).choice(d, size=min(1_000_000, d), replace=False))).to(device)
    stats = {}
    for name in ("omniio_random", "wds_buf1k"):
        batches = [np.asarray(b, dtype=np.int64) for b in independent_batches(name, n, B, args.M, cfg["shard"], np.random.default_rng((args.seed, B, 9)))]
        dl = DataLoader(ds, batch_sampler=ListSampler([list(map(int, b)) for b in batches]), **wk)
        s1 = torch.zeros(d, dtype=torch.float64, device=device); s2 = torch.zeros(d, dtype=torch.float64, device=device)
        sub = np.empty((args.M, coord_idx.numel()), dtype=np.float32); t0 = time.time()
        for k, (x, y, _) in enumerate(dl):
            backward_batch(model, x, y, device)
            g = flat_grad(params); s1 += g.double(); s2 += g.double() ** 2; sub[k] = g[coord_idx].cpu().numpy()
        stats[name] = (float(((s2 - s1 ** 2 / args.M) / (args.M - 1)).mean()), sub, [composition(labels[b])["simpson"] for b in batches])
        print(f"  {name}: {args.M} batches of {B} in {time.time() - t0:.0f} s", flush=True)
    boot = R_hat_boot(stats["wds_buf1k"][1], stats["omniio_random"][1], seed=args.seed)
    c = out["loaders"][f"wds_buf1k_B{B}"]["c"]
    out["R_hat_wds_buf1k"] = {"B": B, "M": args.M, "R_hat": stats["wds_buf1k"][0] / stats["omniio_random"][0], "R_hat_ci": boot["R_hat_ci_iid"],
                              "R_pred": 1 + out["r"]["r_trace"] * (c - 1), "R_pred_coordmean": 1 + out["r"]["r"] * (c - 1), "R_pred_weighted": 1 + out["r"]["r_weighted"] * (c - 1), "R_bound": c,
                              "c_from_R_hat_batches": (np.mean(stats["wds_buf1k"][2]) - S_corpus) / (np.mean(stats["omniio_random"][2]) - S_corpus)}
    print(f"  wds_buf1k B={B}: R_hat {out['R_hat_wds_buf1k']['R_hat']:.2f} {boot['R_hat_ci_iid']}  R_pred {out['R_hat_wds_buf1k']['R_pred']:.2f}  c {c:.2f}", flush=True)
    write_result(os.path.join(RESULTS, f"{args.seed}.json"), out)


def cmd_rhat(args):
    """R_hat (and the composition c if missing) for extra loaders at the corpus B, against a fresh
    random-access reference with the same M; appended to the seed's result file as R_hat_<loader>."""
    out_path = os.path.join(RESULTS, f"{args.seed}.json")
    out = json.load(open(out_path)); cfg = CFG
    meta = pq.read_table(os.path.join(ARCH, "metadata.parquet"))
    labels = meta["source"].to_numpy().astype(np.int64); n = len(labels); B = cfg["B"]
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = make_model(device); model.eval()
    params = [p for p in model.parameters()]; d = sum(p.numel() for p in params)
    ds = Items(meta)
    wk = {"num_workers": args.workers, "multiprocessing_context": "forkserver", "persistent_workers": False, "prefetch_factor": 4}
    S_corpus = out["S_corpus"]; kr = out["loaders"][f"omniio_random_B{B}"]["simpson"] - S_corpus
    for name in args.loaders:                                          # composition of a loader the sweep did not cover
        key = f"{name}_B{B}"
        if key not in out["loaders"]:
            comps = [composition(labels[b]) for b in loader_batches(name, n, B, args.n_comp, cfg["shard"], np.random.default_rng((args.seed, B, 7)))]
            out["loaders"][key] = {k: float(np.mean([c[k] for c in comps])) for k in ("n_eff", "n_distinct", "n_95", "simpson")}
            k = out["loaders"][key]["simpson"] - S_corpus
            out["loaders"][key]["c"] = k / kr; out["loaders"][key]["R_bound"] = k / kr; out["loaders"][key]["kappa_rand_formula"] = (1 - S_corpus) / B
            print(f"  composition {key}: n_eff {out['loaders'][key]['n_eff']:.1f} c {k / kr:.2f}", flush=True)
    coord_idx = torch.from_numpy(np.sort(np.random.default_rng(12345).choice(d, size=min(1_000_000, d), replace=False))).to(device)
    stats = {}
    for name in ["omniio_random"] + list(args.loaders):
        batches = [np.asarray(b, dtype=np.int64) for b in independent_batches(name, n, B, args.M, cfg["shard"], np.random.default_rng((args.seed, B, 9)))]
        dl = DataLoader(ds, batch_sampler=ListSampler([list(map(int, b)) for b in batches]), **wk)
        s1 = torch.zeros(d, dtype=torch.float64, device=device); s2 = torch.zeros(d, dtype=torch.float64, device=device)
        sub = np.empty((args.M, coord_idx.numel()), dtype=np.float32); t0 = time.time()
        for k, (x, y, _) in enumerate(dl):
            backward_batch(model, x, y, device)
            g = flat_grad(params); s1 += g.double(); s2 += g.double() ** 2; sub[k] = g[coord_idx].cpu().numpy()
        stats[name] = (float(((s2 - s1 ** 2 / args.M) / (args.M - 1)).mean()), sub, [composition(labels[b])["simpson"] for b in batches])
        print(f"  {name}: {args.M} batches of {B} in {time.time() - t0:.0f} s", flush=True)
        if name != "omniio_random":
            boot = R_hat_boot(stats[name][1], stats["omniio_random"][1], seed=args.seed)
            c = out["loaders"][f"{name}_B{B}"]["c"]
            out[f"R_hat_{name}"] = {"B": B, "M": args.M, "R_hat": stats[name][0] / stats["omniio_random"][0], "R_hat_ci": boot["R_hat_ci_iid"],
                                    "R_pred": 1 + out["r"]["r_trace"] * (c - 1), "R_pred_coordmean": 1 + out["r"]["r"] * (c - 1), "R_bound": c,
                                    "c_from_R_hat_batches": (np.mean(stats[name][2]) - S_corpus) / (np.mean(stats["omniio_random"][2]) - S_corpus)}
            write_result(out_path, out)
            print(f"  {name} B={B}: R_hat {out[f'R_hat_{name}']['R_hat']:.2f} {boot['R_hat_ci_iid']}  R_pred {out[f'R_hat_{name}']['R_pred']:.2f}  c {c:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ingest", "sweep", "rhat"])
    ap.add_argument("--loaders", nargs="*", default=["wds_buf100", "wds_buf10k", "wds_buf100k"], help="rhat: loaders to measure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--n_comp", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    {"ingest": cmd_ingest, "sweep": cmd_sweep, "rhat": cmd_rhat}[args.stage](args)


if __name__ == "__main__":
    main()

"""
Figures 2, 8 and 9 (ImageNet and Libri-Light panels): the same model, data and schedule under
each loader's sample order, so the only difference between runs is the order the loader under
test emits (omniio_random, wds_buf<k>, wds_preshuffled_buf<k>; orders from common/orders.py,
the ones the composition and R_hat measurements used). Data is pre-decoded by data/prep_cls.py
in archive storage order.

  imagenet_cls     ResNet-18 from scratch, 112 px crops of the 128 px copy, cross-entropy
                   over the 1 000 classes; val top-1 / top-5 on the 50 000 val images.
  librilight_spk   the tiny log-mel Transformer of librilight_sweep.py + linear head, reader
                   identification over the first 200 000 utterances (storage order =
                   reader); 10% of each reader's utterances held out; 1 s random crops.

    python variance/train_cls.py --task imagenet_cls --loader wds_buf1k --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from common.harness import write_result  # noqa: E402
from common.orders import loader_batches  # noqa: E402
from common.paths import data, results  # noqa: E402

RESULTS = results("training")
DATA = data("cls")                                        # data/prep_cls.py: imagenet/, librilight/
TASKS = {
    #              B    epochs  shard  lr      loaders
    "imagenet_cls":    dict(B=256, epochs=16, shard=1000, lr=0.25, loaders=["omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_buf100k", "wds_preshuffled_buf1k"]),
    "librilight_spk":  dict(B=256, epochs=8, shard=1000, lr=1e-3, loaders=["omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_preshuffled_buf1k"]),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ───────────────────────────── GPU image augmentation ─────────────────────────────
def gpu_views(x_u8, out=112, train=True, scale=(0.35, 1.0), gen=None):
    """x_u8: (B, H, W, 3) uint8 on the device -> (B, 3, out, out) float, normalised.
    Train: random-resized crop (area fraction in `scale`, aspect 3/4..4/3) + horizontal flip
    through one affine grid_sample per sample. Eval: center crop out/H of the image."""
    x = x_u8.permute(0, 3, 1, 2).float().div_(255.0)
    B, _, H, W = x.shape
    if not train:
        c = (H - out) // 2
        x = x[:, :, c:c + out, c:c + out]
    else:
        dev = x.device
        area = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=gen)
        logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3), generator=gen)
        w = torch.sqrt(area * torch.exp(logr)).clamp(max=1.0); h = torch.sqrt(area / torch.exp(logr)).clamp(max=1.0)
        cx = (torch.rand(B, device=dev, generator=gen) * (1 - w) + w / 2) * 2 - 1
        cy = (torch.rand(B, device=dev, generator=gen) * (1 - h) + h / 2) * 2 - 1
        flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = w * flip; theta[:, 0, 2] = cx; theta[:, 1, 1] = h; theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ───────────────────────────── tasks: data, model, loss, eval ─────────────────────────────
class ImageNetData:
    def __init__(self, seed):
        root = os.path.join(DATA, "imagenet")
        self.train = np.load(os.path.join(root, "train_128.npy"), mmap_mode="r")
        self.train_y = torch.from_numpy(np.load(os.path.join(root, "train_labels.npy")).astype(np.int64))
        self.val = np.load(os.path.join(root, "val_128.npy"), mmap_mode="r")
        self.val_y = torch.from_numpy(np.load(os.path.join(root, "val_labels.npy")).astype(np.int64))
        self.n = len(self.train_y)

    def batch(self, idx):
        idx = np.sort(np.asarray(idx))                                    # memmap gather in ascending order
        return torch.from_numpy(np.ascontiguousarray(self.train[idx])), self.train_y[idx]


class _Prefetch:
    """Gathers the next batch from the memmap in a background thread while the GPU works."""

    def __init__(self, data, batches, device):
        import queue
        import threading
        self.q = queue.Queue(maxsize=6); self.n = len(batches)

        def run():
            for b in batches:
                x, y = data.batch(b)
                self.q.put((x.pin_memory().to(device, non_blocking=True), y.to(device, non_blocking=True)))
            self.q.put(None)
        threading.Thread(target=run, daemon=True).start()

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item


def resnet18(num_classes=1000):
    import torchvision
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return m.to(memory_format=torch.channels_last)


@torch.no_grad()
def eval_imagenet_cls(model, data, device, bs=500):
    model.eval(); top1 = top5 = 0
    for a in range(0, len(data.val_y), bs):
        x = torch.from_numpy(np.ascontiguousarray(data.val[a:a + bs])).to(device); y = data.val_y[a:a + bs].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(gpu_views(x, train=False).contiguous(memory_format=torch.channels_last))
        p5 = out.topk(5, dim=1).indices
        top1 += (p5[:, 0] == y).sum().item(); top5 += (p5 == y[:, None]).any(1).sum().item()
    model.train(); n = len(data.val_y)
    return {"top1": top1 / n, "top5": top5 / n}


class SpkModel(nn.Module):
    def __init__(self, n_readers):
        super().__init__()
        from audio_encoder import CONFIGS, AudioEncoder
        self.enc = AudioEncoder(CONFIGS["tiny"]); self.head = nn.Linear(CONFIGS["tiny"].out_dim, n_readers)

    def forward(self, wave):
        return self.head(self.enc(wave, torch.full((wave.shape[0],), wave.shape[1], device=wave.device)) * 10.0)   # unit vectors: scale for the softmax


def load_librilight():
    """The pre-decoded 3 s utterances: returns dict(train_x, train_idx, train_y, eval_idx, eval_y, n_classes, sr, secs)."""
    if True:
        root = os.path.join(DATA, "librilight"); meta = json.load(open(os.path.join(root, "meta.json")))
        wave = np.load(os.path.join(root, f"wave_{meta['secs']}s.npy"), mmap_mode="r")
        lab = np.load(os.path.join(root, "labels.npy")).astype(np.int64)
        counts = np.bincount(lab); keep = counts[lab] >= 8                # readers with fewer than 8 utterances dropped
        rng = np.random.default_rng(777)                                  # the same held-out set for every run
        held = np.zeros(len(lab), dtype=bool)
        for r in np.flatnonzero(counts >= 8):
            pos = np.flatnonzero(lab == r); held[rng.choice(pos, size=max(1, len(pos) // 10), replace=False)] = True
        tr_i, ev_i = np.flatnonzero(keep & ~held), np.flatnonzero(keep & held)
        _, remap = np.unique(lab[keep], return_inverse=True); new = np.full(len(lab), -1); new[keep] = remap
        return dict(train_x=wave, train_idx=tr_i, train_y=torch.from_numpy(new[tr_i]), eval_idx=ev_i, eval_y=torch.from_numpy(new[ev_i]),
                    n_classes=int(new.max() + 1), sr=meta["sr"], secs=meta["secs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--loader", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int); ap.add_argument("--B", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = TASKS[args.task]; B = args.B or cfg["B"]; epochs = args.epochs or cfg["epochs"]; lr = args.lr or cfg["lr"]
    out_path = args.out or os.path.join(RESULTS, args.task, args.loader, f"{args.seed}.json")
    if os.path.exists(out_path):
        print("done already:", out_path); return
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed); gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    t0 = time.time()

    # ---- data, model, step function per task
    if args.task == "imagenet_cls":
        data = ImageNetData(args.seed); n = data.n; n_classes = 1000
        model = resnet18(1000).to(device)
        def step(x, y):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return F.cross_entropy(model(gpu_views(x, train=True, gen=gen).contiguous(memory_format=torch.channels_last)), y, label_smoothing=0.1)
        evaluate = lambda: eval_imagenet_cls(model, data, device)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-5)
        make_batches = lambda bs: _Prefetch(data, bs, device)
    else:
        d = load_librilight(); n_classes = d["n_classes"]
        if True:
            n = len(d["train_idx"]); model = SpkModel(n_classes).to(device)
            wave, secs, sr = d["train_x"], d["secs"], d["sr"]
            def fetch(idx):
                rows = np.sort(d["train_idx"][idx])
                x = torch.from_numpy(np.ascontiguousarray(wave[rows])).to(device)
                off = torch.randint(0, (secs - 1) * sr, (len(rows),), device=device, generator=gen)
                ar = off[:, None] + torch.arange(sr, device=device)[None, :]
                y = d["train_y"][torch.from_numpy(np.searchsorted(d["train_idx"], rows))]      # train_idx is sorted
                return x.gather(1, ar).float() / 32767.0, y.to(device)
            def step(x, y):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return F.cross_entropy(model(x), y)
            @torch.no_grad()
            def evaluate():
                model.eval(); top1 = top5 = 0; ev = d["eval_idx"]; ey = d["eval_y"]
                for a in range(0, len(ev), 512):
                    rows = ev[a:a + 512]; x = torch.from_numpy(np.ascontiguousarray(wave[rows])).to(device)
                    c = (secs - 1) * sr // 2; x = x[:, c:c + sr].float() / 32767.0
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p5 = model(x).topk(5, dim=1).indices
                    y = ey[a:a + 512].to(device); top1 += (p5[:, 0] == y).sum().item(); top5 += (p5 == y[:, None]).any(1).sum().item()
                model.train(); return {"top1": top1 / len(ev), "top5": top5 / len(ev)}
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.05)
        make_batches = lambda bs: ((lambda b: fetch(b))(b) for b in bs)

    steps_per_epoch = n // B; total = steps_per_epoch * epochs; warm = max(1, steps_per_epoch if args.task == "imagenet_cls" else total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    order = loader_batches(args.loader, n, B, total, cfg["shard"], rng)
    print(f"{args.task} {args.loader} seed {args.seed}: n_train {n}, B {B}, {steps_per_epoch} steps/epoch x {epochs}, lr {lr}", flush=True)
    log, step_i = [], 0
    model.train()
    for ep in range(1, epochs + 1):
        batches = [next(order) for _ in range(steps_per_epoch)]
        te = time.time(); losses = []
        for x, y in make_batches(batches):
            loss = step(x, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.task == "librilight_spk":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step_i += 1
            losses.append(loss.item() if step_i % 10 == 0 else None)
        losses = [v for v in losses if v is not None]
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "epoch_seconds": time.time() - te, "lr": sched.get_last_lr()[0]}
        if ep % args.eval_every == 0 or ep == epochs:
            row.update(evaluate())
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()) + f"  ({time.time() - t0:.0f} s)", flush=True)
    metric = "top1"
    res = {"task": args.task, "loader": args.loader, "seed": args.seed, "B": B, "epochs": epochs, "n_train": int(n), "lr": lr, "shard_size": cfg["shard"],
           "n_classes": int(n_classes),
           "log": log, "final": {k: v for k, v in log[-1].items() if k not in ("epoch",)},
           "best": max((r.get(metric, -1) for r in log)), "metric": metric, "seconds": time.time() - t0, "node": os.uname().nodename}
    write_result(out_path, res)
    print(f"done: final {metric} {log[-1][metric]:.4f} best {res['best']:.4f} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

"""
Training loop for the Omni-IO vs. WebDataset contrastive experiment.

train_one_run() takes any DataLoader yielding (raw, captions, labels) batches
plus a frozen encoder that maps a batch to (emb_a, emb_b, labels) on the
device, and returns per-epoch loss, label-aware retrieval recall, batch
diversity, and the gradient variance of the loader measured at the initial
parameters (the quantity of Section 4.3 of the paper: same theta for every
loader, no optimizer steps).
"""
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from loss import infonce_loss
from metrics import batch_effective_entropy, recall_at_k
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.stats import composition  # noqa: E402
from model import ContrastiveModel

Encoder = Callable[[Any, torch.device], tuple]


def _flat_grad(model: torch.nn.Module) -> np.ndarray:
    return np.concatenate([p.grad.detach().cpu().numpy().ravel()
                           for p in model.parameters() if p.grad is not None])


def measure_grad_variance(model: ContrastiveModel, loader: DataLoader, device: torch.device,
                          encoder: Encoder, n_batches: int, return_grads: bool = False) -> Dict[str, Any]:
    """Across-batch variance of the mini-batch gradient at fixed parameters,
    averaged over coordinates, over the first n_batches of a fresh pass, with
    the batch composition (n_eff, n_distinct) of those batches."""
    model.train()
    grads, comps = [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        emb_a, emb_b, labels = encoder(batch, device)
        pa, pb = model(emb_a, emb_b)
        loss = infonce_loss(pa, pb, model.temp)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(_flat_grad(model))
        comps.append(composition(labels.cpu().numpy()))
    model.zero_grad(set_to_none=True)
    if len(grads) < 2:
        return {"grad_var": float("nan"), "n_batches": len(grads), "entropy": float("nan")}
    mat = np.stack(grads)
    out = {"grad_var": float(mat.var(axis=0, ddof=1).mean()), "n_batches": len(grads),
           "n_eff": float(np.mean([c["n_eff"] for c in comps])), "n_distinct": float(np.mean([c["n_distinct"] for c in comps])),
           "n_95": float(np.mean([c["n_95"] for c in comps])), "simpson": float(np.mean([c["simpson"] for c in comps])),
           "entropy": float(np.mean([c["H_eff"] for c in comps]))}
    if return_grads:
        out["gbar"] = mat
    return out


@torch.no_grad()
def _val_embeddings(val_loader: DataLoader, model: ContrastiveModel, device: torch.device,
                    encoder: Encoder):
    model.eval()
    a_list, b_list, lab_list = [], [], []
    for batch in val_loader:
        emb_a, emb_b, labels = encoder(batch, device)
        pa, pb = model(emb_a, emb_b)
        a_list.append(pa.cpu().numpy())
        b_list.append(pb.cpu().numpy())
        lab_list.append(labels.cpu().numpy())
    model.train()
    return np.concatenate(a_list), np.concatenate(b_list), np.concatenate(lab_list)


def train_one_run(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    encoder: Encoder,
    in_dim: int = 512,
    out_dim: int = 128,
    n_epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
    device_str: str = "cpu",
    n_var_batches: int = 50,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    model = ContrastiveModel(in_dim=in_dim, out_dim=out_dim).to(device)

    # Gradient variance of this loader at the initial parameters, which are
    # identical across loaders for a given seed.
    gv = measure_grad_variance(model, train_loader, device, encoder, n_var_batches)
    print(f"    grad var at init: {gv['grad_var']:.4e} over {gv['n_batches']} batches "
          f"(n_eff {gv.get('n_eff', float('nan')):.1f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    hist: Dict[str, List[float]] = {k: [] for k in
                                    ["loss", "r1", "r5", "r1_i2t", "r1_t2i", "batch_entropy", "n_eff", "n_distinct"]}
    for epoch in range(n_epochs):
        model.train()
        losses, entropies, neffs, ndist = [], [], [], []
        for batch in train_loader:
            emb_a, emb_b, labels = encoder(batch, device)
            c = composition(labels.cpu().numpy())
            entropies.append(c["H_eff"]); neffs.append(c["n_eff"]); ndist.append(c["n_distinct"])
            pa, pb = model(emb_a, emb_b)
            loss = infonce_loss(pa, pb, model.temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        va, vb, vlab = _val_embeddings(val_loader, model, device, encoder)
        rk = recall_at_k(va, vb, vlab, ks=(1, 5, 10))
        hist["loss"].append(float(np.mean(losses)))
        hist["r1"].append(rk["R@1"])
        hist["r5"].append(rk["R@5"])
        hist["r1_i2t"].append(rk["R@1_i2t"])
        hist["r1_t2i"].append(rk["R@1_t2i"])
        hist["batch_entropy"].append(float(np.mean(entropies)))
        hist["n_eff"].append(float(np.mean(neffs))); hist["n_distinct"].append(float(np.mean(ndist)))
        print(f"  Epoch {epoch + 1:>3}/{n_epochs}  loss={hist['loss'][-1]:.4f}  "
              f"R@1={rk['R@1']:.3f} (i2t {rk['R@1_i2t']:.3f}, t2i {rk['R@1_t2i']:.3f})  "
              f"R@5={rk['R@5']:.3f}  n_eff={hist['n_eff'][-1]:.1f} n_distinct={hist['n_distinct'][-1]:.1f}")

    return {
        **{f"epoch_{k}": v for k, v in hist.items()},
        "final_loss": hist["loss"][-1],
        "final_r1": hist["r1"][-1],
        "final_r5": hist["r5"][-1],
        "mean_batch_entropy": float(np.mean(hist["batch_entropy"])),
        "mean_n_eff": float(np.mean(hist["n_eff"])), "mean_n_distinct": float(np.mean(hist["n_distinct"])),
        "grad_var_init": gv["grad_var"],
        "grad_var_batches": gv["n_batches"],
    }


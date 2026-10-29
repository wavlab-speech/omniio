"""
Figures 2, 8 and 9 (fig_training_grid.pdf, fig_training_preshuffled.pdf, fig_training_block.pdf):
held-out accuracy per epoch under each loader's sample order, six corpora in a 2 x 3 grid, each
line labelled with the loader's measured gradient-variance ratio R_hat and the model's
prediction R_pred = 1 + r (c - 1).
  ImageNet Classification      results/training/imagenet_cls          (val top-1)
  Libri-Light Classification   results/training/librilight_spk        (held-out reader top-1)
  Tabula Sapiens               results/variance/cell/tabula_sapiens.jsonl (cell type, held-out accuracy)
  Flickr8k Contrastive         results/contrastive/clip               (R@1, 5 captions per image)
  ESC-50 Contrastive           results/contrastive/clap               (R@1 against the class captions)
  NeurIPS Multiome             results/variance/cell/multiome.jsonl   (two towers, EMA weights)
    python variance/plot_training_grid.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common.paths import figures, results  # noqa: E402

R = results()
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # colour, marker, short name
    "omniio_random": ("#2a78d6", "o", "random access"),
    "wds_preshuffled_buf1k": ("#eda100", "^", "pre-shuffled + buf 1k"),
    "wds_preshuffled_buf10k": ("#eda100", "^", "pre-shuffled + buf 10k"),
    "wds_buf100": ("#8b1a1a", "v", "buf 100"),
    "wds_buf1k": ("#eb6834", "s", "buf 1k"),
    "wds_buf10k": ("#1baf7a", "D", "buf 10k"),
    "wds_buf100k": ("#7b4fb8", "P", "buf 100k"),
    "block1000_buf16k": ("#0f6f8f", "X", "block sampling (1k blocks, buf 16k)"),
}
ORDER = list(STYLE)


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ───────────── accuracy curves: {loader: {epoch: [values over seeds]}} ─────────────
def curves_train_cls(task, key="top1"):
    by = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(R, "training", task, "*", "[0-9]*.json")):
        d = json.load(open(f))
        for row in d["log"]:
            if key in row:
                by[d["loader"]][row["epoch"]].append(row[key])
    return by


def curves_contrastive(ds):
    by = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(R, "contrastive", ds, "*", "[0-9]*.json")):
        d = json.load(open(f))
        if "train" not in d:
            continue
        for e, v in enumerate(d["train"]["epoch_r1"], 1):
            by[d["loader"]][e].append(v)
    return by


def curves_jsonl(path, bench, key):
    by = defaultdict(lambda: defaultdict(list))
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("bench") == bench and key in r:
            by[r["loader"]][r["epoch"]].append(r[key])
    return by


# ───────────── R_hat (measured) and R_pred (trace-ratio r) per loader ─────────────
def r_imagenet():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, "variance", "imagenet", "[0-9]*.json"))]
    r = _mean([x["r"]["r_trace"] for x in rows]); out = {}
    for l in ("omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_buf100k", "wds_preshuffled_buf1k"):
        c = _mean([x["loaders"][f"{l}_B256"]["R_bound"] for x in rows if f"{l}_B256" in x["loaders"]])
        hat = _mean([x.get(f"R_hat_{l}", {}).get("R_hat") for x in rows]) if l != "omniio_random" else 1.0      # per-loader R_hat (imagenet_sweep --stage rhat)
        out[l] = {"pred": 1 + r * (c - 1), "hat": hat}
    return out


def r_librilight():
    out = {}
    for l in ("omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_preshuffled_buf1k"):
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, "variance", "librilight", f"{l}_B256", "[0-9]*.json"))]
        if rows:
            out[l] = {"hat": _mean([x["R_hat"] for x in rows]), "pred": _mean([x["R_pred"]["reader"] if isinstance(x["R_pred"], dict) else 1.0 for x in rows])}
    return out


def r_contrastive(ds):
    """R_pred = 1 + r_trace (c - 1) with c = (S_batch - S_corpus) / (S_rand - S_corpus)."""
    rand = [json.load(open(f)) for f in glob.glob(os.path.join(R, "contrastive", ds, "omniio_random", "[0-9]*.json"))]
    S_rand = _mean([x["init"]["simpson"] for x in rand]); out = {}
    for l in ("omniio_random", "wds_buf100", "wds_buf1k", "wds_buf10k", "wds_preshuffled_buf1k"):
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, "contrastive", ds, l, "[0-9]*.json"))]
        if not rows:
            continue
        S_c = rows[0]["simpson_corpus"]; r = _mean([x["r"]["r_trace"] for x in rows]); S_b = _mean([x["init"]["simpson"] for x in rows])
        c = (S_b - S_c) / (S_rand - S_c)
        out[l] = {"hat": _mean([x["R_hat"] for x in rows]), "pred": 1 + r * (c - 1)}
    return out


def r_cell(path, hat_key="R_hat"):
    rows = defaultdict(list)
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if r.get("bench") == "gradvar":
                rows[r["loader"]].append(r)
    return {l: {"hat": _mean([x[hat_key] for x in rs]), "pred": _mean([x["R_pred"] for x in rs])} for l, rs in rows.items()}


SHORT = {"omniio_random": "random", "wds_preshuffled_buf1k": "pre-shuffled", "wds_preshuffled_buf10k": "pre-shuffled", "wds_buf100": "buf 100",
         "wds_buf1k": "buf 1k", "wds_buf10k": "buf 10k", "wds_buf100k": "buf 100k", "block1000_buf16k": "block"}
LEGEND = [("omniio_random", "random access"), ("wds_buf100", "buffer 100"),
          ("wds_buf1k", "buffer 1k"), ("wds_buf10k", "buffer 10k"), ("wds_buf100k", "buffer 100k")]


MAIN = [l for l in ORDER if not l.startswith("wds_preshuffled") and not l.startswith("block")]   # the main-text figure


def panel(ax, by, rv, title, ylabel, note, table_at=(0.98, 0.04, "right", "bottom"), table_cols=1, ylim=None, loaders=None):
    loaders = MAIN if loaders is None else loaders
    for l in ORDER:
        if l not in by or l not in loaders:
            continue
        col, mk, name = STYLE[l]
        ep = sorted(by[l]); m = [np.mean(by[l][e]) for e in ep]; lo = [min(by[l][e]) for e in ep]; hi = [max(by[l][e]) for e in ep]
        ax.plot(ep, m, color=col, linewidth=1.8, marker=mk, markersize=4, markeredgecolor="white", markeredgewidth=0.8)
    # the R table: one line per loader in its colour, measured R_hat and predicted R
    x, y, ha, va = table_at
    lines = [(INK2, "$\\hat R$ measured / predicted")]
    for l in ORDER:
        if l in by and l in rv and l in loaders:
            hat, pred = rv[l].get("hat"), rv[l].get("pred")
            lines.append((STYLE[l][0], f"{SHORT[l]}: " + (f"{hat:.1f} / {pred:.1f}" if hat is not None else f"— / {pred:.1f}")))
    step = 0.072
    if table_cols == 1:
        for k, (col, txt) in enumerate(lines if va == "top" else lines[::-1]):
            ax.text(x, y + (-k if va == "top" else k) * step, txt, transform=ax.transAxes, fontsize=10.5, color=col, ha=ha, va=va,
                    fontweight="bold" if k == (0 if va == "top" else len(lines) - 1) else "normal")
    else:                                                  # header on its own line, entries in columns below it
        ax.text(x, y, lines[0][1], transform=ax.transAxes, fontsize=10.5, color=lines[0][0], ha=ha, va="top", fontweight="bold")
        per = int(np.ceil((len(lines) - 1) / table_cols)); w = 0.42
        for k, (col, txt) in enumerate(lines[1:]):
            cx = x + (k // per) * w if ha == "left" else x - (table_cols - 1 - k // per) * w
            ax.text(cx, y - (1 + k % per) * step, txt, transform=ax.transAxes, fontsize=10.5, color=col, ha=ha, va="top")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, loc="left", color=INK, fontweight="bold", pad=18)
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    import matplotlib.ticker as mt
    ax.xaxis.set_major_locator(mt.MaxNLocator(integer=True, nbins=8))
    if note:
        ax.text(0.0, 1.015, note, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")


def panels():
    """(position, curves, R values, title, ylabel, note, kwargs) for the six corpora, main-text layout"""
    TS = os.path.join(R, "variance", "cell", "tabula_sapiens.jsonl"); NP = os.path.join(R, "variance", "cell", "multiome.jsonl")
    return [
        ((0, 0), curves_train_cls("imagenet_cls"), r_imagenet(), "ImageNet Classification", "val top-1", "Image",
         dict(table_at=(0.03, 0.97, "left", "top"), table_cols=2, ylim=(-0.02, 0.86))),
        ((0, 1), curves_train_cls("librilight_spk"), r_librilight(), "Libri-Light Classification", "held-out reader top-1", "Audio", dict(ylim=(-0.3, 1.04))),
        ((0, 2), curves_jsonl(TS, "train", "held_out_acc"), r_cell(TS), "Tabula Sapiens Classification", "held-out accuracy", "Single-cell RNA", {}),
        ((1, 0), curves_contrastive("clip"), r_contrastive("clip"), "Flickr8k Contrastive Learning", "R@1 (image ↔ caption)", "Image", {}),
        ((1, 1), curves_contrastive("clap"), r_contrastive("clap"), "ESC-50 Contrastive Learning", "R@1 (audio → class caption)", "Audio", {}),
        ((1, 2), curves_jsonl(NP, "train", "held_out_acc_ema"), r_cell(NP, hat_key="R_hat_all"), "NeurIPS Multiome Classification", "held-out accuracy (EMA)",
         "Multi-omics (RNA + ATAC)", {}),
    ]


def _save(fig, name):
    d = figures()
    fig.savefig(os.path.join(d, name + ".pdf")); fig.savefig(os.path.join(d, name + ".png"), dpi=130)
    print("wrote", name + ".pdf / .png")


def _legend(fig, entries, y=0.005):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=STYLE[l][0], marker=STYLE[l][1], markersize=6, markeredgecolor="white", markeredgewidth=0.8, linewidth=2.0, label=name) for l, name in entries]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10, bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=1.5)


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = panels()
    # ── main text: random access against shuffle buffers (no block sampling, no pre-shuffled copy) ──
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.6), dpi=150)
    for (i, j), by, rv, title, yl, note, kw in P:
        panel(ax[i, j], by, rv, title, yl, note, **kw)
    ax[0, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _legend(fig, LEGEND)
    fig.tight_layout(w_pad=2.0, h_pad=2.8, rect=(0, 0.05, 1, 1)); _save(fig, "fig_training_grid"); plt.close(fig)

    # ── appendix 1: the pre-shuffled copy (random permutation written once, then streamed through
    #    the same buffer) against random access and the plain buffer of the same size ──
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.6), dpi=150)
    for (i, j), by, rv, title, yl, note, kw in P:
        pre = [l for l in by if l.startswith("wds_preshuffled")][0]; k = pre.split("_buf")[1]
        kw = dict(kw); kw.pop("table_cols", None)
        panel(ax[i, j], by, rv, title, yl, f"{note} · buffer {k}", loaders=["omniio_random", f"wds_buf{k}", pre], **kw)
    ax[0, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _legend(fig, [("omniio_random", "random access"), ("wds_buf1k", "shuffle buffer (1k or 10k, see panel)"), ("wds_preshuffled_buf1k", "pre-shuffled copy + the same buffer")])
    fig.tight_layout(w_pad=2.0, h_pad=2.8, rect=(0, 0.05, 1, 1)); _save(fig, "fig_training_preshuffled"); plt.close(fig)

    # ── appendix 2: block sampling (1 000-cell blocks through a 16k buffer) on the two single-cell corpora ──
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6), dpi=150)
    for a, (_, by, rv, title, yl, note, kw) in zip(ax, [p for p in P if p[0] in ((0, 2), (1, 2))]):
        blk = [l for l in by if l.startswith("block")][0]
        panel(a, by, rv, title, yl, note, loaders=["omniio_random", "wds_buf10k", blk], **kw)
    _legend(fig, [("omniio_random", "random access"), ("wds_buf10k", "buffer 10k"), ("block1000_buf16k", "block sampling (1k-cell blocks, buffer 16k)")], y=0.01)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.09, 1, 1)); _save(fig, "fig_training_block"); plt.close(fig)


if __name__ == "__main__":
    main()

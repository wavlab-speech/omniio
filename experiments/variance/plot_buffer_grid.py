"""
Figure 1 (fig_buffer_grid.pdf): shuffle-buffer size against (left axis) the effective number of
sources per batch n_eff and (right axis) the gradient-variance ratio R_hat against random access,
with the model's prediction R_pred = 1 + r (c - 1), for the six corpora: ImageNet, Libri-Light,
Tabula Sapiens, Flickr8k, ESC-50, NeurIPS Multiome.
    python variance/plot_buffer_grid.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common.orders import parse_k  # noqa: E402
from common.paths import figures, results  # noqa: E402

R = results()
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_BUF, C_PRED, C_RAND, C_PRE, C_BLOCK = "#eb6834", "#eb6834", "#2a78d6", "#eda100", "#0f6f8f"


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _k(name):
    """buffer size of a loader name (wds_buf<k>)."""
    return parse_k(name.split("buf")[-1].lower())


# every corpus -> {"buf": {k: (n_eff, R_hat, R_pred, ci_lo, ci_hi)}, "random": n_eff}
def data_imagenet():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, "variance", "imagenet", "[0-9]*.json"))]
    r = _mean([x["r"]["r_trace"] for x in rows]); B = 256
    d = {"buf": {}}
    for key in rows[0]["loaders"]:
        name, b = key.rsplit("_B", 1)
        if int(b) != B:
            continue
        ne = _mean([x["loaders"][key]["n_eff"] for x in rows if key in x["loaders"]]); c = _mean([x["loaders"][key]["c"] for x in rows if key in x["loaders"]])
        pred = 1 + r * (c - 1)
        hat = _mean([x.get(f"R_hat_{name}", {}).get("R_hat") for x in rows]) if name != "omniio_random" else 1.0
        cis = [x[f"R_hat_{name}"]["R_hat_ci"] for x in rows if f"R_hat_{name}" in x]
        lo, hi = (np.mean(cis, axis=0).tolist() if cis else (None, None))
        if name == "omniio_random":
            d["random"] = ne
        elif name.startswith("wds_buf"):
            d["buf"][_k(name)] = (ne, hat, pred, lo, hi)
    return d


def data_librilight(B=256):
    """Libri-Light at B=256: tiny log-mel Transformer, contrastive crop loss at init; reader label."""
    d = {"buf": {}}
    for ldir in sorted(glob.glob(os.path.join(R, "variance", "librilight", f"*_B{B}"))):
        name = os.path.basename(ldir)[: -len(f"_B{B}")]
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(ldir, "[0-9]*.json"))]
        if not rows:
            continue
        ne = _mean([x["composition"]["reader"]["n_eff"] for x in rows]); hat = _mean([x["R_hat"] for x in rows])
        pred = _mean([x["R_pred"]["reader"] if isinstance(x["R_pred"], dict) else 1.0 for x in rows])
        lo, hi = np.mean([x["R_hat_ci"] for x in rows], axis=0).tolist()
        if name == "omniio_random":
            d["random"] = ne
        elif name.startswith("wds_buf"):
            d["buf"][_k(name)] = (ne, hat, pred, lo, hi)
    return d


def data_contrastive(ds):
    rand = [json.load(open(f)) for f in glob.glob(os.path.join(R, "contrastive", ds, "omniio_random", "[0-9]*.json"))]
    S_rand = _mean([x["init"]["simpson"] for x in rand]); d = {"buf": {}, "random": _mean([x["init"]["n_eff"] for x in rand])}
    for ldir in glob.glob(os.path.join(R, "contrastive", ds, "*")):
        name = os.path.basename(ldir); rows = [json.load(open(f)) for f in glob.glob(os.path.join(ldir, "[0-9]*.json"))]
        if name == "omniio_random" or not rows:
            continue
        S_c = rows[0]["simpson_corpus"]; r = _mean([x["r"]["r_trace"] for x in rows]); S_b = _mean([x["init"]["simpson"] for x in rows])
        c = (S_b - S_c) / (S_rand - S_c); pred = 1 + r * (c - 1)
        ne, hat = _mean([x["init"]["n_eff"] for x in rows]), _mean([x["R_hat"] for x in rows])
        lo, hi = np.mean([x.get("R_hat_ci_iid", x["R_hat_ci"]) for x in rows], axis=0).tolist()
        if name.startswith("wds_buf"):
            d["buf"][_k(name)] = (ne, hat, pred, lo, hi)
    return d


def data_cell(path, hat_key="R_hat", B=256):
    """the jsonl of variance/cell/tabula_sapiens.py or multiome.py: 'composition' and 'gradvar' rows"""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    comp = defaultdict(list); grad = defaultdict(list)
    for x in recs:
        if x.get("bench") == "composition" and x.get("B") == B:
            comp[x["loader"]].append(x)
        if x.get("bench") == "gradvar" and x.get("B") == B:
            grad[x["loader"]].append(x)
    d = {"buf": {}}
    for name in comp:
        ne = _mean([x.get("n_eff") for x in comp[name]])
        g = grad.get(name, [])
        hat = _mean([x[hat_key] for x in g]) if g else None; pred = _mean([x["R_pred"] for x in g]) if g else None
        lo, hi = (np.mean([x[hat_key + "_ci"] for x in g], axis=0).tolist() if g else (None, None))
        if name == "omniio_random":
            d["random"] = ne
        elif name.startswith("wds_buf"):
            d["buf"][_k(name)] = (ne, hat, pred, lo, hi)
    return d


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


C_NEFF, C_R, C_PRED = "#2a78d6", "#eb6834", "#1a1a1a"


def column(ax, d, title, sub, B, first, last):
    """One panel: n_eff on the left axis (blue), R_hat on the right axis (orange, log) with the
    prediction (black dashed, hollow markers drawn on top). Marker shape = loader family."""
    ks = sorted(d["buf"]); axr = ax.twinx()
    # left axis: effective sources per batch
    ax.plot(ks, [d["buf"][k][0] for k in ks], color=C_NEFF, marker="s", markersize=6, linewidth=2.2, markeredgecolor="white")
    ax.axhline(d["random"], color=C_NEFF, linestyle=(0, (4, 3)), linewidth=1.4, alpha=0.8)
    ax.text(float(np.sqrt(ks[0] * ks[-1])), d["random"], "random access $n_{eff}$", color=C_NEFF, fontsize=10.5, va="bottom", ha="center")
    ax.set_xscale("log"); ax.set_ylim(0, max(d["random"], max(v[0] for v in d["buf"].values())) * 1.12)
    ax.set_ylabel("effective sources per batch" if first else "", fontsize=12, color=C_NEFF)
    ax.tick_params(axis="y", colors=C_NEFF, labelsize=11); ax.tick_params(axis="x", colors=INK2, labelsize=11)
    # right axis: gradient-variance ratio, measured and predicted
    hat = [(k, d["buf"][k][1]) for k in ks if d["buf"][k][1] is not None]; pred = [(k, d["buf"][k][2]) for k in ks if d["buf"][k][2] is not None]
    axr.plot([k for k, _ in hat], [v for _, v in hat], color=C_R, linewidth=2.2, marker="s", markersize=6, markeredgecolor="white", zorder=3)
    axr.plot([k for k, _ in pred], [v for _, v in pred], color=C_PRED, linestyle=(0, (5, 3)), linewidth=1.6, marker="s", markersize=9,
             markerfacecolor="none", markeredgewidth=1.4, zorder=4)
    axr.axhline(1.0, color=C_R, linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.7)
    axr.text(float(np.sqrt(ks[0] * ks[-1])), 1.0, "random access $R$ = 1", color=C_R, fontsize=10.5, va="bottom", ha="center")
    axr.set_yscale("log")
    import matplotlib.ticker as mt
    top = max([v for _, v in hat] + [v for _, v in pred] + [1.0]) * 1.5
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= top]
    axr.yaxis.set_major_locator(mt.FixedLocator(ticks)); axr.yaxis.set_major_formatter(mt.FixedFormatter([str(t) for t in ticks]))
    axr.yaxis.set_minor_formatter(mt.NullFormatter()); axr.set_ylim(0.85, top)
    axr.set_ylabel("gradient-variance ratio vs random access" if last else "", fontsize=12, color=C_R)
    axr.tick_params(axis="y", colors=C_R, labelsize=11)
    ax.set_title(title, fontsize=15, loc="left", color=INK, fontweight="bold", pad=20)
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=12, color=INK2, va="bottom")
    ax.set_xlabel("shuffle-buffer size (items)", fontsize=12)
    for a in (ax, axr):
        for sp in ("top",):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom", "right"):
            a.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); axr.grid(False)


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cols = [("ImageNet", "Image", 256, data_imagenet()),
            ("Libri-Light", "Audio", 256, data_librilight()),
            ("Tabula Sapiens", "Single-cell RNA", 256, data_cell(os.path.join(R, "variance", "cell", "tabula_sapiens.jsonl"))),
            ("Flickr8k", "Image", 256, data_contrastive("clip")),
            ("ESC-50", "Audio", 128, data_contrastive("clap")),
            ("NeurIPS Multiome", "Multi-omics (RNA + ATAC)", 256, data_cell(os.path.join(R, "variance", "cell", "multiome.jsonl"), hat_key="R_hat_all"))]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150)
    for j, (title, sub, B, d) in enumerate(cols):
        column(axes[j // 3, j % 3], d, title, sub, B, first=(j % 3 == 0), last=(j % 3 == 2))
    handles = [Line2D([], [], color=C_NEFF, marker="s", markersize=6, linewidth=2.2, markeredgecolor="white", label="effective sources per batch (left axis)"),
               Line2D([], [], color=C_R, marker="s", markersize=6, linewidth=2.2, markeredgecolor="white", label="measured $\\hat R$ (right axis)"),
               Line2D([], [], color=C_PRED, linestyle=(0, (5, 3)), marker="s", markersize=9, markerfacecolor="none", markeredgewidth=1.4, linewidth=1.6, label="predicted $R$ = 1 + r (c − 1)"),
               Line2D([], [], color=C_NEFF, linestyle=(0, (4, 3)), linewidth=1.4, label="random access $n_{eff}$ (left axis)"),
               Line2D([], [], color=C_R, linestyle=(0, (4, 3)), linewidth=1.4, label="random access $R$ = 1 (right axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005), handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(w_pad=2.6, h_pad=2.8, rect=(0, 0.07, 1, 1))
    dd = figures(); fig.savefig(os.path.join(dd, "fig_buffer_grid.pdf")); fig.savefig(os.path.join(dd, "fig_buffer_grid.png"), dpi=130)
    for title, _, _, d in cols:
        print(title, "buf:", {k: tuple(round(v, 2) if v is not None else None for v in d["buf"][k]) for k in sorted(d["buf"])}, "random", round(d["random"], 1))
    print("wrote fig_buffer_grid.pdf / .png")


if __name__ == "__main__":
    main()

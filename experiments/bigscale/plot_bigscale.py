"""
Figure 5 (fig_bigscale.pdf): the whole-corpus experiments, three panels in the style of plot_scale.py

  (a) cold random-read latency (audio + caption) against the number of indexed rows (1 M .. all), mean [min-max] over the fresh-node repeats
  (b) distinct datasets per batch of 256 over the whole corpus: random access against a shuffle buffer of 1 k .. 10 M
  (c) aggregate DataLoader items/s on 1 .. 8 nodes (32 workers each, 8 reads in flight per worker), random order bytes / decode

    python bigscale/plot_bigscale.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "formats"))
import plot_scale as P  # noqa: E402  (palette, style(), save())
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from common.paths import results  # noqa: E402

R = results("bigscale")
C_RAND, C_DEC, C_WIN, C_SEQ, C_BUF = P.C["omniio"], P.C["lance"], P.C["deeplake"], P.INK2, P.C["wds"]


def _fmt_rows(n):
    return f"{n / 1e6:.0f} M" if n < 1e9 else f"{n / 1e9:.1f} B"


def panel_access(ax):
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "access_rep*_*.json")))]
    rows = [l["rows"] for l in reps[0]["levels"]]
    gb = [l["index_rss_mb"] / 1e3 for l in reps[0]["levels"]]
    series = [("mm_cold_median_ms", C_RAND, "audio + text", 2)]
    top = 0
    for key, col, name, zo in series:
        use = [r for r in reps if all(key in l for l in r["levels"])]
        if not use:
            continue
        med = np.array([[l[key] for l in r["levels"]] for r in use]); mean, lo, hi = med.mean(0), med.min(0), med.max(0); top = max(top, hi.max())
        for x, a_, b_ in zip(rows, lo, hi):
            ax.plot([x, x], [a_, b_], color=col, linewidth=2.2, alpha=0.45, solid_capstyle="butt", zorder=zo)
        ax.plot(rows, mean, "-o", color=col, linewidth=2, markersize=9, markeredgecolor="white", markeredgewidth=1.2, zorder=zo + 2, label=f"{name} ({len(use)} nodes)")
        for x, m, h in zip(rows, mean, hi):
            ax.annotate(f"{m:.1f} ms", (x, m), textcoords="offset points", xytext=(10, -3), ha="left", fontsize=P.fs(9.5), color=P.INK2)
    P.style(ax, "Scalability of random access by data size", None, logx=True, logy=True)
    ax.set_xticks(rows); ax.set_xticklabels([f"{_fmt_rows(n)}\n{g:.2f} GB" if g < 1 else f"{_fmt_rows(n)}\n{g:.1f} GB" for n, g in zip(rows, gb)], fontsize=P.fs(10)); ax.minorticks_off()
    ax.set_xlabel("indexed rows / index resident", color=P.INK2, fontsize=P.fs(11))
    ax.set_ylabel("ms per sample", color=P.INK2, fontsize=P.fs(11))
    ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"]); ax.set_ylim(0.5, max(top * 2.5, 6)); ax.set_xlim(rows[0] * 0.7, rows[-1] * 2.5)
    ax.yaxis.grid(True, color=P.GRID, linewidth=0.6, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=P.fs(10))


def panel_composition(ax):
    c = json.load(open(os.path.join(R, "composition.json"))); S = c["settings"]
    rnd = S["random_access"]["distinct_datasets_mean"]
    fam = {"ss": sorted((v["k"], v["distinct_datasets_mean"]) for n, v in S.items() if n.startswith("shardshuffle_buffer_")),
           "so": sorted((v["k"], v["distinct_datasets_mean"]) for n, v in S.items() if n.startswith("buffer_"))}
    ks = [k for k, _ in (fam["ss"] or fam["so"])]
    ax.axhline(rnd, color=C_RAND, linewidth=2.2, zorder=3)
    ax.annotate(f"random access (Omni-IO): {rnd:.1f} datasets in every batch", (ks[0], rnd), textcoords="offset points", xytext=(0, 7), ha="left", fontsize=P.fs(10), color=C_RAND)
    if fam["ss"]:
        x, y = zip(*fam["ss"])
        ax.plot(x, y, "-o", color=C_BUF, linewidth=2, markersize=9, markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.annotate("stream, shards shuffled + k-item buffer", (x[-2], y[-2]), textcoords="offset points", xytext=(-10, 6), ha="right", fontsize=P.fs(10), color=C_BUF)
    if fam["so"]:
        x, y = zip(*fam["so"])
        ax.plot(x, y, "--o", color=C_SEQ, linewidth=1.8, markersize=8, markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.annotate("stream in storage order + k-item buffer", (x[2], y[2]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=P.fs(10), color=C_SEQ)
    P.style(ax, "Shuffle buffer vs random access at scale", None, logx=True)
    ax.set_xticks(ks); ax.set_xticklabels([f"{k // 1000} k" if k < 1e6 else f"{k // 1_000_000} M" for k in ks]); ax.minorticks_off()
    ax.set_xlabel("shuffle buffer k (items); a shard file holds ~49 k items", color=P.INK2, fontsize=P.fs(11)); ax.set_ylabel("distinct datasets per batch", color=P.INK2, fontsize=P.fs(11))
    ax.set_ylim(0, rnd * 1.2); ax.yaxis.grid(True, color=P.GRID, linewidth=0.6)


def _loader_totals():
    """{world: {mode: items/s summed over ranks}} from loader_nodes<world>_rank<r>.json"""
    per = {}
    for f in sorted(glob.glob(os.path.join(R, "loader_nodes*_rank*.json"))):
        d = json.load(open(f))
        for m, row in d["modes"].items():
            per.setdefault(d["world"], {}).setdefault(m, {})[d["rank"]] = row["items_per_s"]
    return {w: {m: sum(v.values()) for m, v in ms.items() if len(v) == w} for w, ms in per.items()}


MODE_STYLE = {"random_bytes": (C_RAND, "-", "random order, bytes only"), "random_decode": (C_DEC, "-", "random order + Opus decode")}


def panel_nodes(ax):
    T = _loader_totals(); ws = sorted(T)
    for m, (col, ls, name) in MODE_STYLE.items():
        xs = [w for w in ws if m in T[w]]; ys = [T[w][m] for w in xs]
        if not xs:
            continue
        if len(xs) > 1:
            ax.plot([xs[0], ws[-1]], [ys[0], ys[0] * ws[-1] / xs[0]], color=col, linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
        ax.plot(xs, ys, ls, color=col, linewidth=2, marker="o", markersize=9, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        lab = f"{ys[-1]:,.0f}"
        ax.annotate(lab, (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, -3), ha="left", fontsize=P.fs(9.5), color=P.INK2)
    P.style(ax, "Scalability of random access by parallel processes", None, logx=True, logy=True)
    ax.set_xticks(ws); ax.set_xticklabels([str(w) for w in ws]); ax.minorticks_off(); ax.set_xlim(0.85, ws[-1] * 2.6)
    ax.set_xlabel("nodes (32 workers each)", color=P.INK2, fontsize=P.fs(11)); ax.set_ylabel("items / s (sum over nodes)", color=P.INK2, fontsize=P.fs(11))
    ax.yaxis.grid(True, color=P.GRID, linewidth=0.6, which="major")
    handles = [Line2D([], [], color=c, linestyle=ls, marker="o", markersize=7, linewidth=2, markeredgecolor="white", label=n) for c, ls, n in MODE_STYLE.values()]
    ax.set_ylim(600, 3e5); ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=P.fs(9.5), handlelength=2.2)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.2))
    panel_access(axes[0]); panel_composition(axes[1]); panel_nodes(axes[2])
    fig.tight_layout(w_pad=3.0)
    P.save(fig, "fig_bigscale")


if __name__ == "__main__":
    main()

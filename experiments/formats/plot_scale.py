"""
Figures 4 and 10 from results/formats/<modality>/*.json:
  fig_scale_ingest.pdf     Figure 4: ingestion items/s against writer processes (top row) and DataLoader
                           items/s with decode against worker count (bottom row), one column per corpus
  fig_scale_formats.pdf    Figure 10: one dot per format and corpus, cold random-access latency (top) and
                           sequential read throughput (bottom)
    python formats/plot_scale.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import figures, results  # noqa: E402

R = results("formats")
OUTS = [figures()]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# one hue per format, fixed order everywhere (Omni-IO variants share the Omni-IO hue and differ by line style)
C = {"omniio": "#2a78d6", "wds": "#eb6834", "lance": "#1baf7a", "hdf5": "#eda100", "raw": "#e87ba4", "deeplake": "#6250d6", "mds": "#8a5a1c"}
NAME = {"omniio": "Omni-IO", "wds": "WebDataset", "lance": "Lance", "hdf5": "HDF5", "raw": "raw files", "deeplake": "Deep Lake", "mds": "MosaicML MDS"}
ORDER = ["omniio", "wds", "lance", "hdf5", "deeplake", "mds", "raw"]
MODS = ["image", "text", "video"]
MTITLE = {"image": "ImageNet-1k, 1.28 M JPEGs", "text": "PG-19, 2.50 M text chunks", "video": "UCF101, 13 k MP4 clips"}
MODALITY_NAME = {"image": "Image", "text": "Text", "video": "Video"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2, "pdf.fonttype": 42})
FS = [1.0]                                                            # font-size multiplier (the stacked formats figure uses 1.35)


def fs(x):
    return x * FS[0]


def newest(pattern, pred=lambda r: True):
    fs = sorted(glob.glob(os.path.join(R, pattern)), key=os.path.getmtime)
    rows = [r for r in (json.load(open(f)) for f in fs) if pred(r)]
    return rows[-1] if rows else None


def _mean_tree(rows):
    """element-wise mean of nested dicts of numbers over the union of keys (a repeat may hold only some readers)"""
    out = {}
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    for k in keys:
        v = next(r[k] for r in rows if k in r)
        vals = [r[k] for r in rows if k in r]
        if isinstance(v, dict):
            out[k] = _mean_tree([x for x in vals if isinstance(x, dict)])
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(np.mean(vals)); out[k + "_runs"] = vals
        else:
            out[k] = v
    out["n_runs"] = len(rows)
    return out


def reps(pattern, pred=lambda r: True):
    fs = sorted(glob.glob(os.path.join(R, pattern)), key=os.path.getmtime)
    return [r for f, r in ((f, json.load(open(f))) for f in fs) if pred(r) and "_rep" in os.path.basename(f)]


def access(m):
    rows = reps(f"{m}/access_*.json", lambda r: isinstance(r.get("passes"), dict))
    return _mean_tree(rows) if rows else dict(newest(f"{m}/access_*.json", lambda r: isinstance(r.get("passes"), dict)), n_runs=1)


def _is_stream_run(r):
    return any(k.endswith("_sequential") or k == "wds_stream" for k in r["runs"])


def stream(m):
    rows = reps(f"{m}/stream_*.json", _is_stream_run)
    return _mean_tree(rows) if rows else dict(newest(f"{m}/stream_*.json", _is_stream_run), n_runs=1)


def loader(m):
    """every loader file for the modality, oldest first, so the newest measurement of a (loader, workers) pair wins"""
    fs = sorted(glob.glob(os.path.join(R, m, "loader_*.json")), key=os.path.getmtime)
    rep_fs = [f for f in fs if "_rep" in os.path.basename(f)]
    acc = {}
    for f in (rep_fs or fs):                                            # repeats: mean per (loader, workers); else newest value wins
        for run in json.load(open(f))["runs"]:
            acc.setdefault(run["loader"], {}).setdefault(run["workers"], []).append(run["items_per_s"])
    return {k: {w: (float(np.mean(v)) if rep_fs else v[-1]) for w, v in d.items()} for k, d in acc.items()}


def ingest(m):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(R, m, "ingest_*.json"))]
    best, full = {}, {}
    for r in rows:
        if r["tag"] == "full":
            full[r["fmt"]] = r; continue
        if not (r["tag"].endswith("_v2") or "_rep" in r["tag"]):
            continue                                                  # the first sweep paid cold source reads; warm-source runs only
        key = (r["fmt"], r["workers"])
        if key not in best:                                           # mean over the warm-source runs (user rule: repeat and report the mean)
            best[key] = dict(r, n_runs=0, _sum=0.0, _runs=[])
        b = best[key]; b["n_runs"] += 1; b["_sum"] += r["items_per_s"]; b["_runs"].append(r["items_per_s"])
        b["items_per_s"] = b["_sum"] / b["n_runs"]; b["items_min"], b["items_max"] = min(b["_runs"]), max(b["_runs"])
    return best, full                                                 # mean of the warm-source runs; n_runs says how many


def style(ax, title, sub=None, logx=False, logy=False):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=fs(11)); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=fs(14), loc="left", color=INK, fontweight="bold", pad=fs(18) if sub else 8)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=fs(11), color=INK2, va="bottom")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def save(fig, name):
    for d in OUTS:
        os.makedirs(d, exist_ok=True); fig.savefig(os.path.join(d, name + ".pdf")); fig.savefig(os.path.join(d, name + ".png"), dpi=130)
    plt.close(fig); print("wrote", name)


def legend(fig, keys, extra=(), y=0.01, ncol=None):
    handles = [Line2D([], [], color=C[k], marker="o", markersize=8, linewidth=2, markeredgecolor="white", label=NAME[k]) for k in keys] + list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles), frameon=False, fontsize=fs(12), bbox_to_anchor=(0.5, y),
               handlelength=2.4 / FS[0], columnspacing=1.8 / FS[0], handletextpad=0.6 / FS[0])


# ---------------------------------------------------------------- fig 1: one dot per format and modality
ACCESS_KEY = {"omniio": "omniio", "wds": "wds_member", "lance": "lance", "hdf5": "hdf5", "raw": "raw", "deeplake": "deeplake", "mds": "mds"}
STREAM_KEY = {"omniio": "omniio_sequential", "wds": "wds_stream", "lance": "lance_scan", "hdf5": "hdf5_sequential", "raw": "raw_sequential", "deeplake": "deeplake_sequential", "mds": "mds_sequential"}
LOADER_KEY = {"omniio": "omniio_random", "wds": "wds_stream_buf1k", "lance": "lance_random_batched", "hdf5": "hdf5_random", "raw": "raw_random", "deeplake": "deeplake_random_batched", "mds": "mds_shuffled"}


def dots(ax, values, unit_fmt, spread=None):
    """values: {modality: {fmt: x}}; one row per modality, a dot per format, direct labels on the two ends;
    spread: {modality: {fmt: (lo, hi)}} draws the min-max over repeats as a whisker behind the dot"""
    mods = [m for m in MODS if m in values]; ys = {m: i for i, m in enumerate(reversed(mods))}
    for m, v in values.items():
        xs = [v[f] for f in ORDER if f in v]
        ax.plot([min(xs), max(xs)], [ys[m]] * 2, color=GRID, linewidth=1.2, zorder=1)
        for f in reversed(ORDER):                                      # Omni-IO drawn last, on top of any dot it overlaps
            if f in v:
                if spread and f in spread.get(m, {}):
                    lo, hi = spread[m][f]
                    ax.plot([lo, hi], [ys[m]] * 2, color=C[f], linewidth=2.2, alpha=0.45, solid_capstyle="butt", zorder=2)
                ax.plot(v[f], ys[m], "o", color=C[f], markersize=12 if f == "omniio" else 11, markeredgecolor="white", markeredgewidth=1.2, zorder=4 if f == "omniio" else 3)
        lo, hi = min(v, key=v.get), max(v, key=v.get)
        ax.annotate(unit_fmt(v[lo]), (v[lo], ys[m]), textcoords="offset points", xytext=(0, fs(11)), ha="center", fontsize=fs(10), color=INK2)
        ax.annotate(unit_fmt(v[hi]), (v[hi], ys[m]), textcoords="offset points", xytext=(0, fs(11)), ha="center", fontsize=fs(10), color=INK2)
    ax.set_yticks(list(ys.values())); ax.set_yticklabels(list(ys), fontsize=fs(12), color=INK)
    ax.set_ylim(-0.6, len(mods) - 0.3); ax.xaxis.grid(True, color=GRID, linewidth=0.6)


def fig_formats():
    nvme = list(MODS)
    A = {m: access(m) for m in nvme}; S = {m: stream(m) for m in nvme}
    acc = {m: {f: A[m]["passes"]["cold"][k]["median_ms"] for f, k in ACCESS_KEY.items() if k in A[m]["passes"]["cold"]} for m in nvme}
    seq = {m: {f: S[m]["runs"][k]["MB_per_s"] for f, k in STREAM_KEY.items() if k in S[m]["runs"]} for m in nvme}
    rng_ = lambda d, key: (min(d[key + "_runs"]), max(d[key + "_runs"])) if key + "_runs" in d else None
    acc_sp = {m: {f: rng_(A[m]["passes"]["cold"][k], "median_ms") for f, k in ACCESS_KEY.items() if k in A[m]["passes"]["cold"] and rng_(A[m]["passes"]["cold"][k], "median_ms")} for m in nvme}
    seq_sp = {m: {f: rng_(S[m]["runs"][k], "MB_per_s") for f, k in STREAM_KEY.items() if k in S[m]["runs"] and rng_(S[m]["runs"][k], "MB_per_s")} for m in nvme}
    runs_txt = "mean of 3 fresh nodes per format, whisker = min–max"
    FS[0] = 1.35
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9), dpi=150)
    dots(axes[0], acc, lambda x: f"{x:.2f}" if x < 10 else f"{x:.0f}", acc_sp); style(axes[0], "Random access", f"median ms per item (lower is better); {runs_txt}", logx=True)
    axes[0].set_xlabel("ms per read (log)", fontsize=fs(12))
    dots(axes[1], seq, lambda x: f"{x:.0f}", seq_sp); style(axes[1], "Sequential read", f"MB/s (higher is better); {runs_txt}", logx=True)
    axes[1].set_xlabel("MB/s (log)", fontsize=fs(12))
    legend(fig, ORDER, y=0.0, ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=3); save(fig, "fig_scale_formats")
    FS[0] = 1.0
    return acc, seq


# ---------------------------------------------------------------- Figure 4: ingestion (top) and DataLoader throughput (bottom)
LOADERS = [  # key, hue, linestyle, marker, filled, legend name  (solid = random access, dashed = sequential access)
    ("omniio_random", "omniio", "-", "o", True, "Omni-IO"),
    ("omniio_random_prefetch", "omniio", "-", "o", False, "Omni-IO (batch prefetch)"),
    ("omniio_random_lookahead8", "omniio", "-", "s", False, "Omni-IO (random access lookahead)"),
    ("wds_stream_buf1k", "wds", (0, (5, 3)), "o", True, "WebDataset (sequential lookahead)"),
    ("lance_random_batched", "lance", "-", "o", True, "Lance (batch prefetch)"),
    ("hdf5_random", "hdf5", "-", "o", True, "HDF5"),
    ("deeplake_random_batched", "deeplake", "-", "o", True, "Deep Lake (batch prefetch)"),
    ("mds_shuffled", "mds", (0, (5, 3)), "o", True, "MosaicML MDS (shard shuffle, sequential)"),
    ("mds_random", "mds", "-", "o", False, "MosaicML MDS (random access)"),
    ("raw_random", "raw", "-", "o", True, "raw files"),
]


def _line(hue, ls, marker, filled, lw=2.2, ms=7):
    return dict(color=C[hue], linestyle=ls, marker=marker, markersize=fs(ms) if filled else fs(ms + 1), linewidth=lw,
                markeredgecolor="white" if filled else C[hue], markerfacecolor=C[hue] if filled else "none", markeredgewidth=1.8)


def draw_loader(axes, mods):
    for ax, m in zip(axes, mods):
        by = loader(m); ws = sorted(by["omniio_random"])
        for key, hue, ls, marker, filled, _ in LOADERS:
            if key in by:
                ax.plot(ws, [by[key].get(w, np.nan) for w in ws], **_line(hue, ls, marker, filled))
        style(ax, f"{MTITLE[m].split(',')[0]} - {MODALITY_NAME[m]}", None, logx=True)
        ax.set_xticks(ws); ax.set_xticklabels([str(w) for w in ws]); ax.minorticks_off(); ax.set_xlabel("DataLoader workers", fontsize=fs(12))
        ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e3:g}k" if x >= 1e3 else f"{x:g}"))
    axes[0].set_ylabel("items/s (with decode)", fontsize=fs(12))
    axes[0].text(0.02, 0.97, "solid: random access   dashed: sequential access", transform=axes[0].transAxes, fontsize=fs(10), color=INK2, va="top")
    return [Line2D([], [], label=name, **_line(hue, ls, marker, filled)) for _, hue, ls, marker, filled, name in LOADERS]




# ---------------------------------------------------------------- fig 4: ingestion
WRITERS = [("omniio_bytes", "omniio", "-", "Omni-IO (fast mode)"), ("omniio", "omniio", (0, (4, 2.5)), "Omni-IO (with metadata)"),
           ("wds", "wds", "-", "WebDataset"), ("lance", "lance", "-", "Lance"), ("hdf5", "hdf5", "-", "HDF5"), ("deeplake", "deeplake", "-", "Deep Lake"), ("mds", "mds", "-", "MosaicML MDS")]


def draw_ingest(axes, mods):
    for ax, m in zip(axes, mods):
        best, full = ingest(m); ws = sorted({w for _, w in best if w <= 16})
        for fmt, hue, ls, _ in WRITERS:
            xs = [w for w in ws if (fmt, w) in best]
            ax.plot(xs, [best[(fmt, w)]["items_per_s"] for w in xs], **_line(hue, ls, "o", ls == "-"))
        style(ax, f"{MTITLE[m].split(',')[0]} - {MODALITY_NAME[m]}", None, logx=True)
        ax.set_xticks(ws); ax.set_xticklabels([str(w) for w in ws]); ax.minorticks_off(); ax.set_xlabel("writer processes", fontsize=fs(12))
        ax.set_ylim(0, None); ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e3:g}k" if x >= 1e3 else f"{x:g}"))
    axes[0].set_ylabel("items/s (ingestion)", fontsize=fs(12))
    return [Line2D([], [], label=lab, **_line(h, ls, "o", ls == "-")) for _, h, ls, lab in WRITERS]


def fig_ingest():
    """2 x 3: ingestion (top row) and DataLoader throughput (bottom row), one column per corpus"""
    mods = ["image", "text", "video"]
    FS[0] = 1.35
    fig, axes = plt.subplots(2, 3, figsize=(19, 14), dpi=150)
    h_top = draw_ingest(axes[0], mods)
    h_bot = draw_loader(axes[1], mods)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.955, bottom=0.115, hspace=0.6, wspace=0.2)
    fig.legend(handles=h_top, loc="center", ncol=5, frameon=False, fontsize=fs(12), bbox_to_anchor=(0.5, 0.525), handlelength=2.2, columnspacing=1.4)
    ncol = -(-len(h_bot) // 2); row1, row2 = h_bot[:ncol], h_bot[ncol:]  # two rows, row-major reading order (matplotlib fills columns top-down)
    ordered = [h for k in range(ncol) for h in (row1[k:k + 1] + row2[k:k + 1])]
    fig.legend(handles=ordered, loc="center", ncol=ncol, frameon=False, fontsize=fs(11.5), bbox_to_anchor=(0.5, 0.035), handlelength=2.0, columnspacing=1.1)
    save(fig, "fig_scale_ingest"); FS[0] = 1.0



if __name__ == "__main__":
    fig_ingest(); fig_formats()

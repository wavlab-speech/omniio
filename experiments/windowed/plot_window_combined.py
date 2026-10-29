"""Figure 6: partial video reads, speed (top) and peak memory (bottom) against window length, for
Omni-IO, raw MP4 and the WebDataset member, with 'decode the whole recording, then take the
window' as a flat reference in both panels (measured time from speed_full_decode.py; memory =
the decoded recording's size, which no node can hold). Speed: warm pass, mean over the
fresh-node repeats of speed_windowed.py; memory: mem_windowed.py.

    python windowed/plot_window_combined.py
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

R = results("windowed")
OUTS = [figures()]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEEP = [("tar_member", "#eb6834", "WebDataset (whole member, then seek)"), ("raw_seek", "#e87ba4", "raw MP4 (PyAV seek)"), ("omniio_window", "#2a78d6", "Omni-IO (window read)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
FS = 1.3

# ---- speed: warm ms per window, every repeat ----
fs = sorted(glob.glob(os.path.join(R, "speed_windowed_rep*.json")))
speed = {}
for f in fs:
    for x in json.load(open(f))["runs"]:
        if x["modality"] == "video" and not np.isnan(x["warm_ms_median"]):
            speed.setdefault(x["reader"], {}).setdefault(x["window_s"], []).append(x["warm_ms_median"])
full, full_frames = [], []
for f in glob.glob(os.path.join(R, "speed_full_decode_*.json")):
    for x in json.load(open(f))["runs"]:
        if x["method"] == "full_decode":
            full.append(x["seconds"] * 1e3); full_frames.append(x["frames_decoded"])
# ---- memory: peak extra MB ----
mem = {}
for f in sorted(glob.glob(os.path.join(R, "mem_windowed*.json")), key=os.path.getmtime):
    for x in json.load(open(f))["runs"]:
        if x["modality"] != "video" or "peak_extra_mb" not in x:
            continue
        mem.setdefault(x["reader"], {}).setdefault(x["window_s"], []).append(x["peak_extra_mb"])
ws = sorted({w for d in speed.values() for w in d})
FRAMES_TOTAL = np.mean(full_frames) if full_frames else 59.0 * 60 * 30   # frames the full decode actually produced
full_mem_mb = FRAMES_TOTAL * 1280 * 720 * 3 / 1e6

fig, axes = plt.subplots(2, 1, figsize=(10, 13.5), dpi=150)


def panel(ax, data, ylabel, title, fmt, ref_val, ref_label):
    for key, col, _ in KEEP:
        if key not in data:
            continue
        wr = [w for w in ws if w in data[key]]
        ys = [np.mean(data[key][w]) for w in wr]; lo = [min(data[key][w]) for w in wr]; hi = [max(data[key][w]) for w in wr]
        ax.fill_between(wr, lo, hi, color=col, alpha=0.18, linewidth=0)
        ax.plot(wr, ys, color=col, marker="o", markersize=9, linewidth=2.6, markeredgecolor="white", zorder=4 if key == "omniio_window" else 3)
        for w, y in zip(wr, ys):
            above = key != "omniio_window"
            ax.annotate(fmt(y), (w, y), textcoords="offset points", xytext=(0, 14) if above else (0, -23), ha="center", fontsize=11 * FS, color=col)
    if ref_val is not None:
        ax.axhline(ref_val, color=INK2, linestyle=(0, (4, 3)), linewidth=1.8)
        ax.text(ws[0], ref_val, "  " + ref_label, color=INK2, fontsize=10.5 * FS, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ws); ax.set_xticklabels([f"{w} s" for w in ws]); ax.minorticks_off()
    ax.set_xlabel("window length", fontsize=13 * FS); ax.set_ylabel(ylabel, fontsize=12 * FS)
    ax.tick_params(labelsize=11 * FS, colors=INK2); ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=15 * FS, loc="left", fontweight="bold", color=INK, pad=12)


panel(axes[0], speed, "read time (ms)", "Speed of a partial video read", lambda y: f"{y / 1e3:.2f} s" if y >= 1e3 else f"{y:.0f} ms",
      np.mean(full) if full else None, f"full raw decode + subsample: {np.mean(full) / 1e3:.0f} s" if full else "")
panel(axes[1], mem, "peak memory (MB)", "Memory of a partial video read", lambda y: f"{y / 1e3:.1f} GB" if y >= 1e3 else f"{y:.0f} MB",
      full_mem_mb, f"full raw decode + subsample: {full_mem_mb / 1e3:.0f} GB (decoded recording)")
if full:
    axes[0].set_ylim(top=np.mean(full) * 4)
axes[1].set_ylim(top=full_mem_mb * 4)
handles = [Line2D([], [], color=c, marker="o", markersize=9, linewidth=2.6, markeredgecolor="white", label=n) for _, c, n in KEEP]
handles.append(Line2D([], [], color=INK2, linestyle=(0, (4, 3)), linewidth=1.8, label="full raw decode + subsample"))
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12 * FS, bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=1.8)
fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=3)
for d in OUTS:
    os.makedirs(d, exist_ok=True); fig.savefig(os.path.join(d, "fig_window_combined.pdf")); fig.savefig(os.path.join(d, "fig_window_combined.png"), dpi=130)
print("wrote fig_window_combined; speed files", [os.path.basename(f) for f in fs], "| full-decode samples", len(full))

#!/usr/bin/env bash
# Figure 6: speed and memory of partial video reads (Omni-IO window read, raw MP4 seek, WebDataset member).
# Three speed repeats, each meant for a fresh node (cold client cache on the first pass); memory and the
# full-decode reference once. Then the figure.
set -euo pipefail
cd "$(dirname "$0")"
python data/make_video_corpus.py make --n 50 --workers "${WORKERS:-16}"
python data/make_video_corpus.py ingest
for rep in 1 2 3; do python windowed/speed_windowed.py --rep $rep; done
python windowed/speed_full_decode.py
python windowed/mem_windowed.py --n 20 --n_member 4 --passes 3
python windowed/plot_window_combined.py

#!/usr/bin/env bash
# Figure 1: batch diversity and gradient-variance inflation against the shuffle-buffer size on six corpora.
# Each block prepares its corpus (once) and measures composition, r and R_hat at initialization, three seeds.
# GPU runs: ImageNet, Libri-Light, ESC-50, Flickr8k. CPU runs: Tabula Sapiens, NeurIPS Multiome.
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-16}
# --- ImageNet (the 256 px copy from data/download_formats.py)
python data/download_formats.py
python variance/imagenet_sweep.py --stage ingest --workers "$W"
for s in 0 1 2; do python variance/imagenet_sweep.py --stage sweep --seed $s --workers "$W"; python variance/imagenet_sweep.py --stage rhat --seed $s --workers "$W"; done
# --- Libri-Light medium, cut into 15 s segments and byte-copied into an Omni-IO archive
python data/librilight/segment_librilight.py --workers "$W"
python data/librilight/build_archives.py --workers "$W"
for s in 0 1 2; do python variance/librilight_sweep.py --seed $s --workers "$W"; done
# --- ESC-50 and Flickr8k with frozen CLAP / CLIP encoders (init-time measurements; run_fig2.sh adds the training)
python variance/contrastive/ingest_esc50.py
python variance/contrastive/ingest_flickr8k.py
python variance/contrastive/contrastive.py --dataset clap --seeds 0 1 2 --num_workers 8 --no_train
python variance/contrastive/contrastive.py --dataset clip --seeds 0 1 2 --num_workers 8 --no_train
# --- Tabula Sapiens and the NeurIPS Multiome cohort (CPU)
python variance/cell/prep_tabula_sapiens.py --workers "$W"
python variance/cell/tabula_sapiens.py composition
python variance/cell/tabula_sapiens.py gradvar --M 200
python variance/cell/prep_neurips2021.py --workers "$W"
python variance/cell/multiome.py composition
python variance/cell/multiome.py gradvar --M 200 --seeds 0 1 2
python variance/plot_buffer_grid.py

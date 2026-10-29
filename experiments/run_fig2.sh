#!/usr/bin/env bash
# Figure 2 (and Figures 8 and 9, which the same runs and plot script produce): held-out accuracy per epoch
# under each loader's sample order, six corpora, three seeds. Assumes run_fig1.sh has prepared the corpora
# and measured r / R_hat (the R values printed in the panels come from those results).
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-16}
# --- ImageNet and Libri-Light: decoded once into memmaps, then one GPU run per (loader, seed)
python data/prep_cls.py imagenet --workers "$W"
python data/prep_cls.py librilight --n 200000 --workers "$W"
for s in 0 1 2; do
  for l in omniio_random wds_buf100 wds_buf1k wds_buf10k wds_buf100k wds_preshuffled_buf1k; do python variance/train_cls.py --task imagenet_cls --loader $l --seed $s; done
  for l in omniio_random wds_buf100 wds_buf1k wds_buf10k wds_preshuffled_buf1k; do python variance/train_cls.py --task librilight_spk --loader $l --seed $s; done
done
# --- ESC-50 and Flickr8k: 20 epochs of the projection heads under each loader (the init measurements are redone cheaply)
python variance/contrastive/contrastive.py --dataset clap --seeds 0 1 2 --num_workers 8 --epochs 20
python variance/contrastive/contrastive.py --dataset clip --seeds 0 1 2 --num_workers 8 --epochs 20
# --- Tabula Sapiens and NeurIPS Multiome (CPU): 8 epochs
python variance/cell/tabula_sapiens.py train --epochs 8 --seeds 0 1 2 --workers 8
python variance/cell/multiome.py train --epochs 8 --seeds 0 1 2 --workers 8
python variance/plot_training_grid.py      # fig_training_grid (Figure 2), fig_training_preshuffled (8), fig_training_block (9)

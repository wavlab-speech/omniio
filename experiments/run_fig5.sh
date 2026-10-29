#!/usr/bin/env bash
# Figure 5: Omni-IO on the Bagpiper corpus (downloaded from the Hugging Face Hub by data/download_bagpiper.py, 14 TB).
# Download and index once, build the caption archives; then composition (index only), random access on three fresh
# nodes, and the DataLoader on 1, 2, 4 and 8 nodes (one rank per node through Slurm's srun); then the figure.
set -euo pipefail
cd "$(dirname "$0")"
python data/download_bagpiper.py --workers "${WORKERS:-8}"
python bigscale/build_index.py --workers "${WORKERS:-16}"
python bigscale/build_text.py --workers "${WORKERS:-32}"
python bigscale/composition.py
for rep in 1 2 3; do python bigscale/access_scale.py --rep $rep; done          # one repeat per fresh node
for n in 1 2 4 8; do
  srun -N $n --ntasks-per-node=1 --cpus-per-task=32 python bigscale/loader_nodes.py --seconds 600 --workers 32
done
python bigscale/plot_bigscale.py

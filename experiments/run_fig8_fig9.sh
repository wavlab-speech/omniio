#!/usr/bin/env bash
# Figures 8 and 9 use the training runs of run_fig2.sh (the pre-shuffled copy and block-sampling orders are part of
# them); this only regenerates the two appendix figures from those results.
set -euo pipefail
cd "$(dirname "$0")"
python variance/plot_training_grid.py

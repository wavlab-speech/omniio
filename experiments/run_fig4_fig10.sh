#!/usr/bin/env bash
# Figures 4 and 10: ingestion, DataLoader, random-access and sequential-read benchmarks of Omni-IO against
# WebDataset, Lance, HDF5, Deep Lake, MosaicML MDS and raw files on ImageNet-1k, PG-19 and UCF101.
# Ingestion sweeps write a 100 k-item store per (format, writer count), three repeats, and one full store per
# format for the read benchmarks. Each read repeat (REP 1..3) is meant for a fresh node: its first pass is cold.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
python data/download_formats.py
for m in image text video; do python formats/scale_bench.py materialize --modality $m --workers "${WORKERS:-32}"; done
for m in image text video; do
  python formats/scale_bench.py ingest --modality $m --fmt omniio_bytes --workers 8 --n 100000 --tag warm0     # source files into the client cache, untimed
  for r in 1 2 3; do for w in 1 4 8 16; do for f in omniio_bytes omniio wds lance hdf5 deeplake mds; do
    python formats/scale_bench.py ingest --modality $m --fmt $f --workers $w --n 100000 --tag w${w}_rep$r
    rm -rf "${OMNIIO_EXPERIMENTS:-.}/data/formats/$m/$f/w${w}_rep$r"
  done; done; done
  for f in omniio wds lance hdf5 deeplake mds; do python formats/scale_bench.py ingest --modality $m --fmt $f --workers 32 --tag full; done
done
n_items() { case $1 in video) echo 13320;; *) echo 20000;; esac; }
for r in 1 2 3; do for m in image text video; do                      # one (repeat, corpus) per fresh node
  python formats/scale_bench.py access --modality $m --n_reads 2000 --rep $r --tag rep${r}_$(hostname -s)
  python formats/scale_bench.py stream --modality $m --n "$(n_items $m)" --rep $r --tag rep${r}_$(hostname -s)
  python formats/scale_bench.py loader --modality $m --workers 1 4 8 16 --batch 64 --steps 150 --rep $r --tag rep${r}_$(hostname -s)
done; done
python formats/plot_scale.py

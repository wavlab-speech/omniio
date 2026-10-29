"""Sources of the three corpora of Figures 4 and 10, from the Hugging Face Hub:

  evanarlian/imagenet_1k_resized_256   ImageNet-1k, the 256 px copy (train parquet shards)
  emozilla/pg19                        PG-19 (train parquet shards)
  quchenyuan/UCF101-ZIP                UCF101 (UCF-101.zip)

    python data/download_formats.py [--workers 8]
Then formats/scale_bench.py materialize writes the raw files from them.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data  # noqa: E402

SOURCES = (("evanarlian/imagenet_1k_resized_256", "imagenet256"), ("emozilla/pg19", "pg19"), ("quchenyuan/UCF101-ZIP", "ucf101"))


def main():
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    for rid, sub in SOURCES:
        t0 = time.time()
        p = snapshot_download(rid, repo_type="dataset", local_dir=data("sources", sub), max_workers=a.workers)
        print(rid, "->", p, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

"""
Ingest Flickr8k (HuggingFace `jxie/flickr8k`) into Omni-IO and WebDataset.

Each image is stored once in Omni-IO; its five captions are five pairs that
point at it. Pairs are written grouped by image, in the dataset's order, so
the WebDataset stream sees the five captions of an image back to back. The
label of a pair is the index of its image, which the retrieval metric uses to
count any caption of the same image as a hit.

Usage:
    python ingest_flickr8k.py --out_dir /path/to/flickr8k_bench [--skip_if_exists]
"""
import argparse
import json
import os
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.paths import data  # noqa: E402
from ingest_common import write_omniio, write_webdataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=data("flickr8k"))
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--shard_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    omniio_dir = os.path.join(args.out_dir, "omniio")
    wds_dir = os.path.join(args.out_dir, "webdataset")
    raw_dir = os.path.join(args.out_dir, "raw")
    for d in (omniio_dir, wds_dir, raw_dir):
        os.makedirs(d, exist_ok=True)

    if args.skip_if_exists and os.path.exists(os.path.join(omniio_dir, "train_index.parquet")):
        print("Ingestion already done, skipping.")
        return

    from datasets import load_dataset

    print("Loading jxie/flickr8k ...")
    ds = load_dataset("jxie/flickr8k")
    val_key = "test" if "test" in ds else "validation"

    meta = {}
    for split_name, split_key in [("train", "train"), ("val", val_key)]:
        split = ds[split_key]
        caption_cols = sorted(c for c in split.column_names if c.startswith("caption"))
        print(f"\n{split_name}: {len(split)} images, caption columns {caption_cols}")

        mod_paths, pairs = {}, []
        for row_idx, row in enumerate(tqdm(split, desc="  writing raw images")):
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            fmt = (img.format or "JPEG").upper()
            ext = "jpg" if fmt == "JPEG" else fmt.lower()
            img_id = f"{split_name}_{row_idx:05d}"
            path = os.path.join(raw_dir, f"{img_id}.{ext}")
            if not os.path.exists(path):
                img.convert("RGB").save(path, format=fmt)
            mod_paths[img_id] = path
            captions = []
            for c in caption_cols:
                v = row[c]
                captions.extend(v if isinstance(v, list) else [v])
            for k, cap in enumerate(captions):
                pairs.append((f"{img_id}_c{k}", img_id, str(cap), row_idx))

        # One image format per split keeps the writer kwargs simple.
        exts = {os.path.splitext(p)[1][1:] for p in mod_paths.values()}
        assert len(exts) == 1, f"mixed image formats: {exts}"
        ext = exts.pop()
        target = "jpeg" if ext == "jpg" else ext

        write_omniio(mod_paths, pairs, omniio_dir, split_name, "image",
                     num_workers=args.num_workers, target_format=target)
        write_webdataset(mod_paths, pairs, wds_dir, split_name, ext, args.shard_size)
        meta[split_name] = {"n_items": len(mod_paths), "n_pairs": len(pairs)}
        if split_name == "train":
            meta["n_classes"] = len(mod_paths)

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nIngestion complete: {args.out_dir}")


if __name__ == "__main__":
    main()

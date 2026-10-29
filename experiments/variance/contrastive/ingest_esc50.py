"""
Ingest ESC-50 (HuggingFace `ashraq/esc50`) into Omni-IO and WebDataset.

ESC-50: 2000 five-second clips, 50 classes, 40 clips per class, 5 folds.
Clips are resampled to 48 kHz, which is what the CLAP feature extractor
requires, and stored once as 16-bit FLAC. Each clip gets five caption
templates, so a split of N clips has 5N pairs; the label is the class.

Pairs are written sorted by class, then by clip, so the WebDataset stream
sees whole classes back to back: the layout of a corpus collected per class.

Usage:
    python ingest_esc50.py --out_dir /path/to/esc50_bench [--val_folds 5]
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.paths import data  # noqa: E402
from ingest_common import write_omniio, write_webdataset  # noqa: E402

TARGET_SR = 48_000  # laion/clap-htsat-unfused feature extractor sampling rate

CAPTION_TEMPLATES = [
    "a sound of {cls}",
    "this is a recording of {cls}",
    "the audio contains {cls}",
    "an example of {cls}",
    "{cls} sounds",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=data("esc50"))
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--val_folds", nargs="+", type=int, default=[5])
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

    import librosa
    from datasets import load_dataset

    from datasets import Audio

    print("Loading ashraq/esc50 ...")
    ds = load_dataset("ashraq/esc50", split="train")
    # Take the stored WAV bytes and decode them with soundfile: no torchcodec needed.
    ds = ds.cast_column("audio", Audio(decode=False))
    class_names = sorted(set(ds["category"]))
    class_to_label = {c: i for i, c in enumerate(class_names)}

    splits = {"train": ([], {}), "val": ([], {})}  # name -> (pairs, mod_paths)
    for row_idx, row in enumerate(tqdm(ds, desc="Resampling to 48 kHz")):
        a = row["audio"]
        src = io.BytesIO(a["bytes"]) if a.get("bytes") is not None else a["path"]
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = int(sr)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        cls = row["category"]
        label = class_to_label[cls]
        split_name = "val" if int(row["fold"]) in args.val_folds else "train"
        clip_id = f"{split_name}_{row_idx:04d}"
        path = os.path.join(raw_dir, f"{clip_id}.flac")
        if not os.path.exists(path):
            sf.write(path, audio, TARGET_SR, format="FLAC", subtype="PCM_16")
        pairs, mod_paths = splits[split_name]
        mod_paths[clip_id] = path
        for k, tmpl in enumerate(CAPTION_TEMPLATES):
            pairs.append((f"{clip_id}_c{k}", clip_id, tmpl.format(cls=cls), label))

    meta = {"n_classes": len(class_names), "class_names": class_names, "sample_rate": TARGET_SR}
    for split_name, (pairs, mod_paths) in splits.items():
        pairs.sort(key=lambda p: (p[3], p[1]))  # by class, then clip: contiguous classes
        ordered_paths = {p[1]: mod_paths[p[1]] for p in pairs}  # storage order follows pairs
        print(f"\n{split_name}: {len(ordered_paths)} clips, {len(pairs)} pairs")
        write_omniio(ordered_paths, pairs, omniio_dir, split_name, "audio",
                     num_workers=args.num_workers, target_format="flac", target_bit_depth=16)
        write_webdataset(ordered_paths, pairs, wds_dir, split_name, "flac", args.shard_size)
        meta[split_name] = {"n_items": len(ordered_paths), "n_pairs": len(pairs)}

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nIngestion complete: {args.out_dir}")


if __name__ == "__main__":
    main()

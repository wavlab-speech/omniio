"""
Data preparation for the training experiments of variance/train_cls.py (Figures 2, 8, 9): decode
each corpus ONCE into arrays in archive storage order, so the training runs differ only in the
order the loader under test emits (the quantity the experiments are about) and not in decode
throughput.

  imagenet   train: the 1 281 167 entries of the Omni-IO archive (storage order = class order,
             variance/imagenet_sweep.py --stage ingest), shorter side resized to 128 and
             center-cropped to 128x128 uint8, into a memmap; labels = class. val: the 50 000
             images of the val split, same transform.  ~63 GB + 2.5 GB.
  librilight the first --n entries of the 15 s archive (storage order = reader order,
             data/librilight/): the middle 3 s at 16 kHz int16 into a memmap, label = reader.

    python data/prep_cls.py imagenet --workers 28
    python data/prep_cls.py librilight --n 200000 --workers 28
"""
import argparse
import glob
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data  # noqa: E402

IN_ARCH = data("imagenet", "archive", "metadata.parquet")
IN_VAL = data("sources", "imagenet256", "data", "val-*.parquet")
LL_MANIFEST = data("librilight", "seg15", "manifest.parquet")
LL_ARCHIVE = data("librilight", "archives", "omniio", "metadata.parquet")
S = 128


def _resize_center(im, s=S):
    from PIL import Image
    w, h = im.size; r = s / min(w, h)
    im = im.convert("RGB").resize((max(s, round(w * r)), max(s, round(h * r))), Image.BILINEAR)
    w, h = im.size; l, t = (w - s) // 2, (h - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.uint8)


def _in_chunk(args):
    """Decode entries [a, b) of the archive into the memmap."""
    rows, a, out_path, n = args
    from PIL import Image
    from omniio.blob.handles import archive_fd, pread_all
    mm = np.lib.format.open_memmap(out_path, mode="r+")
    for k, r in enumerate(rows):
        fd = archive_fd(r["path"])
        raw = pread_all(fd, r["end_byte"] - r["start_byte"], r["start_byte"])
        mm[a + k] = _resize_center(Image.open(io.BytesIO(raw)))
    mm.flush(); del mm
    return len(rows)


def _val_chunk(args):
    path, out_path, a = args
    from PIL import Image
    t = pq.read_table(path)
    mm = np.lib.format.open_memmap(out_path, mode="r+")
    labels = []
    for k, (im, lab) in enumerate(zip(t["image"].to_pylist(), t["label"].to_pylist())):
        mm[a + k] = _resize_center(Image.open(io.BytesIO(im["bytes"]))); labels.append(int(lab))
    mm.flush(); del mm
    return a, labels


def cmd_imagenet(args):
    root = os.path.join(args.root, "imagenet"); os.makedirs(root, exist_ok=True)
    done = os.path.join(root, "DONE")
    if os.path.exists(done):
        print("imagenet already prepared at", root); return
    m = pq.read_table(IN_ARCH, columns=["path", "start_byte", "end_byte", "source"]).to_pylist()
    n = len(m); t0 = time.time()
    np.save(os.path.join(root, "train_labels.npy"), np.array([r["source"] for r in m], dtype=np.int16))
    tp = os.path.join(root, "train_128.npy")
    mm = np.lib.format.open_memmap(tp, mode="w+", dtype=np.uint8, shape=(n, S, S, 3)); del mm
    chunk = 2000
    jobs = [(m[a:a + chunk], a, tp, n) for a in range(0, n, chunk)]
    with ProcessPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(_in_chunk, jobs, chunksize=1)):
            if k % 50 == 0:
                print(f"  train {min((k + 1) * chunk, n)}/{n} ({time.time() - t0:.0f} s)", flush=True)
    files = sorted(glob.glob(IN_VAL)); nv = sum(pq.read_metadata(f).num_rows for f in files)
    vp = os.path.join(root, "val_128.npy")
    mm = np.lib.format.open_memmap(vp, mode="w+", dtype=np.uint8, shape=(nv, S, S, 3)); del mm
    offs, a = [], 0
    for f in files:
        offs.append((f, vp, a)); a += pq.read_metadata(f).num_rows
    labels = np.zeros(nv, dtype=np.int16)
    with ProcessPoolExecutor(min(args.workers, len(files))) as ex:
        for a0, lab in ex.map(_val_chunk, offs):
            labels[a0:a0 + len(lab)] = lab
    np.save(os.path.join(root, "val_labels.npy"), labels)
    json.dump({"n_train": n, "n_val": nv, "size": S, "seconds": time.time() - t0, "source": IN_ARCH}, open(os.path.join(root, "meta.json"), "w"), indent=1)
    open(done, "w").write("ok\n"); print(f"imagenet prepared: {n} train, {nv} val in {time.time() - t0:.0f} s")


def _ll_bin(args):
    rows, a, out_path, sr, secs = args
    import soundfile as sf
    import torch
    import torchaudio
    from omniio.blob.handles import archive_fd, pread_all
    mm = np.lib.format.open_memmap(out_path, mode="r+")
    n_out = sr * secs
    res = {}
    for k, r in enumerate(rows):
        fd = archive_fd(r["path"])
        raw = pread_all(fd, r["end_byte"] - r["start_byte"], r["start_byte"])
        x, s0 = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True); x = x[:, 0]
        if s0 != sr:
            if s0 not in res:
                res[s0] = torchaudio.transforms.Resample(s0, sr)
            x = res[s0](torch.from_numpy(x)).numpy()
        c = max(0, (len(x) - n_out) // 2); x = x[c:c + n_out]
        x = np.pad(x, (0, n_out - len(x)))
        mm[a + k] = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    mm.flush(); del mm
    return len(rows)


def cmd_librilight(args):
    root = os.path.join(args.root, "librilight"); os.makedirs(root, exist_ok=True)
    done = os.path.join(root, "DONE")
    if os.path.exists(done):
        print("librilight already prepared at", root); return
    t0 = time.time()
    man = pq.read_table(LL_MANIFEST, columns=["id", "speaker", "source", "duration"]).slice(0, args.n)
    arc = pq.read_table(LL_ARCHIVE, columns=["id", "path", "start_byte", "end_byte", "bin_index"]).slice(0, args.n)
    assert man["id"].to_pylist()[:1000] == arc["id"].to_pylist()[:1000]
    spk = man["speaker"].to_numpy(zero_copy_only=False); _, lab = np.unique(spk, return_inverse=True)
    np.save(os.path.join(root, "labels.npy"), lab.astype(np.int32))
    np.save(os.path.join(root, "recording.npy"), np.unique(man["source"].to_numpy(zero_copy_only=False), return_inverse=True)[1].astype(np.int32))
    rows = arc.to_pylist(); n = len(rows)
    wp = os.path.join(root, f"wave_{args.secs}s.npy")
    mm = np.lib.format.open_memmap(wp, mode="w+", dtype=np.int16, shape=(n, 16000 * args.secs)); del mm
    # one job per bin (sequential reads inside a bin)
    jobs, a = [], 0
    bins = arc["bin_index"].to_numpy()
    for b in np.unique(bins):
        sel = np.flatnonzero(bins == b); jobs.append((rows[sel[0]:sel[-1] + 1], int(sel[0]), wp, 16000, args.secs))
    with ProcessPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(_ll_bin, jobs, chunksize=1)):
            if k % 20 == 0:
                print(f"  bins {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    json.dump({"n": n, "n_readers": int(lab.max() + 1), "secs": args.secs, "sr": 16000, "seconds": time.time() - t0,
               "reader_runs": int(np.sum(np.diff(lab) != 0) + 1)}, open(os.path.join(root, "meta.json"), "w"), indent=1)
    open(done, "w").write("ok\n"); print(f"librilight prepared: {n} utterances, {lab.max() + 1} readers in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["imagenet", "librilight"])
    ap.add_argument("--root", default=data("cls"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=200_000, help="librilight: entries (storage order)")
    ap.add_argument("--secs", type=int, default=3, help="librilight: seconds kept per utterance")
    args = ap.parse_args()
    {"imagenet": cmd_imagenet, "librilight": cmd_librilight}[args.corpus](args)


if __name__ == "__main__":
    main()

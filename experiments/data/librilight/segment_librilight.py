"""
Cut Libri-Light recordings into fixed-length FLAC utterances, one file each,
the way ASR corpora are distributed (LibriSpeech: 280k files of ~12 s).

Output tree mirrors the input: out_dir/<speaker>/<book>/<file>_<k>.flac, so the
storage order (speaker, book, file, segment) is the order of the manifest.
Each input file is one job; a job is skipped when its manifest shard exists,
so the run is resumable.

    python data/librilight/segment_librilight.py --src .../librilight/medium --workers 30
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.paths import data  # noqa: E402
import soundfile as sf


def _one(job):
    src, rel, out_dir, seg_s, min_s = job
    spk, book, fname = rel.split(os.sep)
    stem = fname[:-5]
    dst_dir = os.path.join(out_dir, spk, book)
    os.makedirs(dst_dir, exist_ok=True)
    rows = []
    with sf.SoundFile(src) as f:
        sr, n_seg = f.samplerate, int(seg_s * f.samplerate)
        k = 0
        while True:
            x = f.read(n_seg, dtype="int16", always_2d=True)
            if len(x) < int(min_s * sr):
                break
            buf = io.BytesIO()
            sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
            data = buf.getvalue()
            path = os.path.join(dst_dir, f"{stem}_{k:04d}.flac")
            with open(path, "wb") as o:
                o.write(data)
            rows.append({"id": f"{spk}_{book}_{stem}_{k:04d}", "path": path, "speaker": spk, "book": book,
                         "source": rel, "segment": k, "sample_rate": sr, "channels": x.shape[1],
                         "samples": len(x), "duration": len(x) / sr, "size_bytes": len(data)})
            k += 1
            if len(x) < n_seg:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=data("sources", "librilight", "medium"), help="the Libri-Light medium subset (speaker/book/file.flac)")
    ap.add_argument("--out_dir", default=data("librilight", "seg15"))
    ap.add_argument("--segment_s", type=float, default=15.0)
    ap.add_argument("--min_s", type=float, default=5.0, help="drop a final piece shorter than this")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()

    shards_dir = os.path.join(args.out_dir, "_manifest_shards")
    os.makedirs(shards_dir, exist_ok=True)
    files = []
    for spk in sorted(os.listdir(args.src)):
        sd = os.path.join(args.src, spk)
        if not os.path.isdir(sd):
            continue
        for book in sorted(os.listdir(sd)):
            bd = os.path.join(sd, book)
            for fn in sorted(os.listdir(bd)):
                if fn.endswith(".flac"):
                    files.append(os.path.join(spk, book, fn))
    if args.max_files:
        files = files[: args.max_files]
    print(f"{len(files)} source files", flush=True)

    todo = [rel for rel in files if not os.path.exists(os.path.join(shards_dir, rel.replace(os.sep, "__") + ".json"))]
    print(f"{len(todo)} to segment ({len(files) - len(todo)} done)", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_one, (os.path.join(args.src, rel), rel, args.out_dir, args.segment_s, args.min_s)): rel
                for rel in todo}
        for fut in as_completed(futs):
            rel = futs[fut]
            rows = fut.result()
            with open(os.path.join(shards_dir, rel.replace(os.sep, "__") + ".json"), "w") as f:
                json.dump(rows, f)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} files, {time.time() - t0:.0f} s", flush=True)

    # manifest in storage order; the shards are read with many threads because a
    # file open on the parallel filesystem costs tens of milliseconds each
    from concurrent.futures import ThreadPoolExecutor

    def load(rel):
        with open(os.path.join(shards_dir, rel.replace(os.sep, "__") + ".json")) as f:
            return json.load(f)

    rows = []
    with ThreadPoolExecutor(64) as ex:
        for part in ex.map(load, files):
            rows.extend(part)
    print(f"  manifest rows collected, {time.time() - t0:.0f} s", flush=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, os.path.join(args.out_dir, "manifest.parquet"))
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"manifest: {len(rows)} segments, {hours:.0f} h, {sum(r['size_bytes'] for r in rows) / 1e9:.0f} GB, "
          f"{len(set(r['speaker'] for r in rows))} speakers, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

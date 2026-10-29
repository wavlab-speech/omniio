"""
Build the Omni-IO archive of the segmented Libri-Light corpus by byte copy (the segments are
already FLAC, so no transcoding): blob_<k>.bin files of --bin_mb each plus metadata.parquet with
the audio index columns; written by N workers, each owning a contiguous slice of the manifest
and writing its own bins, concatenated by renaming, never copying. Storage order = the
manifest's (speaker, book, file, segment).

    python data/librilight/build_archives.py --workers 16
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.paths import data  # noqa: E402


def _omniio_worker(args):
    rows, out_dir, wid, bin_bytes = args
    meta, k, cur, cur_path, written = [], 0, None, None, 0

    def new_bin():
        nonlocal cur, cur_path, k, written
        if cur is not None:
            cur.close()
        cur_path = os.path.join(out_dir, f"w{wid:03d}_{k:05d}.bin")
        cur, k, written = open(cur_path, "wb"), k + 1, 0

    new_bin()
    for r in rows:
        if written >= bin_bytes:
            new_bin()
        with open(r["path"], "rb") as f:
            data = f.read()
        cur.write(data)
        meta.append({"sample_rate": r["sample_rate"], "channels": r["channels"], "samples": r["samples"],
                     "format": "flac", "bit_depth": 16, "id": r["id"], "start_byte": written,
                     "end_byte": written + len(data), "bin_path": cur_path})
        written += len(data)
    cur.close()
    return meta


def build_omniio(rows, out_dir, workers, bin_mb):
    os.makedirs(out_dir, exist_ok=True)
    chunk = (len(rows) + workers - 1) // workers
    jobs = [(rows[i * chunk:(i + 1) * chunk], out_dir, i, bin_mb << 20) for i in range(workers) if rows[i * chunk:(i + 1) * chunk]]
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        metas = list(ex.map(_omniio_worker, jobs))
    t_write = time.time() - t0
    # rename worker bins into place in order; offsets are per bin so nothing shifts
    all_meta, bin_index = [], 0
    for meta in metas:
        for p in sorted({m["bin_path"] for m in meta}, key=lambda p: [m["bin_path"] for m in meta].index(p)):
            final = os.path.join(out_dir, f"blob_{bin_index}.bin")
            os.rename(p, final)
            for m in meta:
                if m["bin_path"] == p:
                    m["path"], m["bin_index"] = final, bin_index
            bin_index += 1
        all_meta.extend(meta)
    for m in all_meta:
        m.pop("bin_path")
    pq.write_table(pa.Table.from_pylist(all_meta), os.path.join(out_dir, "metadata.parquet"))
    return {"write_s": t_write, "total_s": time.time() - t0, "n_bins": bin_index,
            "bytes": sum(m["end_byte"] - m["start_byte"] for m in all_meta)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=data("librilight", "seg15", "manifest.parquet"))
    ap.add_argument("--out_dir", default=data("librilight", "archives", "omniio"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bin_mb", type=int, default=1024)
    args = ap.parse_args()
    rows = pq.read_table(args.manifest).to_pylist()
    print(f"{len(rows)} files, {sum(r['size_bytes'] for r in rows) / 1e9:.1f} GB", flush=True)
    if os.path.exists(os.path.join(args.out_dir, "metadata.parquet")):
        print("exists:", args.out_dir); return
    shutil.rmtree(args.out_dir, ignore_errors=True)
    r = build_omniio(rows, args.out_dir, args.workers, args.bin_mb)
    print(f"{len(rows)} files in {r['total_s']:.0f} s ({len(rows) / r['total_s']:.0f} files/s), {r['n_bins']} bins", flush=True)


if __name__ == "__main__":
    main()

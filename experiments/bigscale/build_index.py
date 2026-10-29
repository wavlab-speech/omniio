"""
Figure 5: compact read index for the Bagpiper corpus (one Omni-IO audio archive per dataset
under data/bagpiper/audio, from data/download_bagpiper.py): per dataset one .npz with
  shard   uint16  index into the dataset's shard-file list
  offset  uint32  byte offset inside the shard (shards are < 4 GB)
  size    uint32  entry size in bytes
  cat     uint8   category (speech / sound / music / none)
plus the shard path list and the dataset name. 9 bytes per row: 4.2 GB for the whole corpus,
against the 300 GB of merged metadata.parquet (which carry the captions) and ~60 GB that the
three loader columns take as Arrow strings. Built from the merged metadata with a dictionary
read of the path column, one dataset per worker; also records how long each dataset's index
took to read and how many bytes of Parquet that touched.

    python bigscale/build_index.py --workers 16
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data  # noqa: E402

SRC = os.environ.get("OMNIIO_BAGPIPER", data("bagpiper", "audio"))    # <dataset>/metadata.parquet + shard bins (data/download_bagpiper.py)
OUT = data("bagpiper", "index")
CATS = {"speech": 1, "sound": 2, "music": 3}


def one(ds):
    t0 = time.time(); m = os.path.join(SRC, ds, "metadata.parquet")
    pf = pq.ParquetFile(m); cols = pf.schema_arrow.names
    want = ["path", "start_byte_offset", "file_size_bytes"] + (["category"] if "category" in cols else []) + [c for c in ("start_time", "end_time") if c in cols]
    t = pq.read_table(m, columns=want, read_dictionary=["path"] + (["category"] if "category" in cols else []))
    path = t["path"].combine_chunks()
    dictionary = path.dictionary.to_pylist() if hasattr(path, "dictionary") else None
    if dictionary is None:                                              # not dictionary-encoded on disk: build one
        uniq, inv = np.unique(np.asarray(t["path"].to_pylist()), return_inverse=True); dictionary = uniq.tolist(); shard = inv.astype(np.uint16)
    else:
        shard = path.indices.to_numpy().astype(np.uint16)
    off = t["start_byte_offset"].to_numpy(); size = t["file_size_bytes"].to_numpy()
    assert off.max() < 2 ** 32 and size.max() < 2 ** 32, (ds, off.max(), size.max())
    if "category" in cols:
        c = t["category"].combine_chunks()
        if hasattr(c, "dictionary"):                                     # null categories (e.g. 13,684 rows in one dataset) -> code 0
            lut = np.asarray([CATS.get(x, 0) for x in c.dictionary.to_pylist()] + [0], dtype=np.uint8)   # extra slot for all-null columns
            ix = c.indices.fill_null(len(lut) - 1).to_numpy(zero_copy_only=False)
            cat = lut[ix]
        else:
            cat = np.asarray([CATS.get(x, 0) for x in c.to_pylist()], dtype=np.uint8)
    else:
        cat = np.zeros(len(off), dtype=np.uint8)
    # the segment window: rows of the music sets are 10 s windows into a stored track (the collection's segmentation); NaN = the whole entry
    w0 = t["start_time"].to_numpy(zero_copy_only=False).astype(np.float32) if "start_time" in cols else np.full(len(off), np.nan, np.float32)
    w1 = t["end_time"].to_numpy(zero_copy_only=False).astype(np.float32) if "end_time" in cols else np.full(len(off), np.nan, np.float32)
    np.savez(os.path.join(OUT, f"{ds}.npz"), shard=shard, offset=off.astype(np.uint32), size=size.astype(np.uint32), cat=cat, paths=np.asarray(dictionary), t0=w0, t1=w1)
    rg_bytes = sum(pf.metadata.row_group(i).column(j).total_compressed_size for i in range(pf.metadata.num_row_groups) for j in range(pf.metadata.num_columns)
                   if pf.metadata.row_group(i).column(j).path_in_schema in want)
    win = np.isfinite(w0) & np.isfinite(w1) & (w1 > w0)
    return {"dataset": ds, "rows": len(off), "shards": len(dictionary), "bytes": int(size.sum()), "index_read_s": time.time() - t0, "windowed_rows": int(win.sum()),
            "stored_bytes": int(size[np.unique(shard.astype(np.int64) * (1 << 32) + off.astype(np.int64), return_index=True)[1]].sum()),
            "parquet_bytes_for_columns": int(rg_bytes), "parquet_file_bytes": os.path.getsize(m)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ds = sorted(d for d in os.listdir(SRC) if os.path.exists(os.path.join(SRC, d, "metadata.parquet")))
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, ds):
            rows.append(r); print(f"[{time.strftime('%H:%M:%S')}] {r['dataset']:34s} {r['rows']:>12,} rows {r['shards']:>5} shards {r['stored_bytes'] / 1e12:6.2f} TB stored, {r['windowed_rows'] / max(r['rows'], 1) * 100:5.1f}% windowed rows  index read {r['index_read_s']:6.1f} s "
                                  f"({r['parquet_bytes_for_columns'] / 1e6:,.0f} MB of {r['parquet_file_bytes'] / 1e9:.1f} GB parquet)", flush=True)
    tot = {"rows": sum(r["rows"] for r in rows), "bytes": sum(r["bytes"] for r in rows), "stored_bytes": sum(r["stored_bytes"] for r in rows), "windowed_rows": sum(r["windowed_rows"] for r in rows),
           "shards": sum(r["shards"] for r in rows), "wall_s": time.time() - t0, "datasets": rows,
           "index_bytes_on_disk": sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".npz"))}
    json.dump(tot, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"TOTAL {tot['rows']:,} rows ({tot['windowed_rows']:,} windowed), {tot['stored_bytes'] / 1e12:.2f} TB stored, {tot['shards']:,} shards, index {tot['index_bytes_on_disk'] / 1e9:.2f} GB on disk, built in {tot['wall_s']:.0f} s")


if __name__ == "__main__":
    main()

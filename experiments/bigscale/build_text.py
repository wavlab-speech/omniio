"""
Figure 5: ingest the captions of the Bagpiper corpus into Omni-IO text archives (zstd, the text writer's encoding), one
archive per dataset, in the AUDIO row order (the caption is the `rich_caption` column of the
audio metadata, so text row i pairs with audio row i of the same dataset and no join is needed).
This is the parallel ingestion path measured in the scale benchmark: every worker takes a block
of row groups, compresses each caption, appends to its own blob file and returns the index rows;
the rows are concatenated once into metadata.parquet (id, path, start_byte, end_byte, bin_index,
original_size, compressed_size). Reports items/s per dataset and overall.

    python bigscale/build_text.py --workers 32
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import sys

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data  # noqa: E402

SRC = os.environ.get("OMNIIO_BAGPIPER", data("bagpiper", "audio"))
TEXT = data("bagpiper", "text")
MAX_BIN = 1 << 30


def worker(job):
    """one worker: its row groups, in order; every audio row gets a text row (an empty entry when
    the caption is missing) so that text row i is audio row i of the dataset"""
    src, out, wid, groups = job
    pf = pq.ParquetFile(src); cctx = zstandard.ZstdCompressor(level=3)
    k = 0; fh = None; pos = 0; rows = []; path = None
    for g in groups:
        t = pf.read_row_group(g, columns=["utt_id", "rich_caption"])
        for i, (uid, cap) in enumerate(zip(t["utt_id"].to_pylist(), t["rich_caption"].to_pylist())):
            raw = (cap or "").encode("utf-8"); comp = cctx.compress(raw)
            if fh is None or pos >= MAX_BIN:
                if fh:
                    fh.close()
                path = os.path.join(out, f"blob_w{wid:02d}_{k}.bin"); fh = open(path, "wb"); pos = 0; k += 1
            fh.write(comp); rows.append((g, i, uid, path, pos, pos + len(comp), len(raw), len(comp))); pos += len(comp)
    if fh:
        fh.close()
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--out", default=TEXT)
    ap.add_argument("--datasets", nargs="*", default=None); a = ap.parse_args()
    ds = a.datasets or sorted(d for d in os.listdir(SRC) if os.path.exists(os.path.join(SRC, d, "metadata.parquet")))
    summary = []; T0 = time.time(); os.makedirs(a.out, exist_ok=True)
    for d in ds:
        out = os.path.join(a.out, d)
        if os.path.exists(os.path.join(out, "metadata.parquet")):
            print(f"{d}: done already", flush=True); continue
        os.makedirs(out, exist_ok=True); src = os.path.join(SRC, d, "metadata.parquet"); pf = pq.ParquetFile(src)
        ng = pf.metadata.num_row_groups; W = min(a.workers, ng)
        jobs = [(src, out, w, list(range(ng))[w::W]) for w in range(W)]        # row groups dealt round-robin: equal work, order restored below
        t0 = time.time()
        with ProcessPoolExecutor(W) as ex:
            parts = list(ex.map(worker, jobs))
        ordered = sorted((r for part in parts for r in part), key=lambda r: (r[0], r[1]))      # audio row order
        bin_of = {}
        for r in ordered:
            bin_of.setdefault(r[3], len(bin_of))
        tbl = pa.Table.from_arrays([pa.array([r[2] for r in ordered]), pa.array([r[3] for r in ordered]).dictionary_encode(), pa.array([r[4] for r in ordered], pa.int64()),
                                    pa.array([r[5] for r in ordered], pa.int64()), pa.array([bin_of[r[3]] for r in ordered], pa.int32()),
                                    pa.array([r[6] for r in ordered], pa.int64()), pa.array([r[7] for r in ordered], pa.int64())],
                                   names=["id", "path", "start_byte", "end_byte", "bin_index", "original_size", "compressed_size"])
        pq.write_table(tbl, os.path.join(out, "metadata.parquet"))
        sec = time.time() - t0; n = tbl.num_rows; nb = sum(r[7] for r in ordered); raw_b = sum(r[6] for r in ordered)
        row = {"dataset": d, "items": n, "seconds": sec, "items_per_s": n / sec, "workers": W, "compressed_bytes": nb, "raw_bytes": raw_b}
        summary.append(row); json.dump(summary, open(os.path.join(a.out, "ingest_summary.json"), "w"), indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] {d:34s} {n:>12,} captions in {sec:7.0f} s = {n / sec:8,.0f} items/s ({W} workers), {raw_b / 1e9:.1f} GB raw -> {nb / 1e9:.1f} GB zstd", flush=True)
    tot_n = sum(r["items"] for r in summary); tot_s = time.time() - T0
    print(f"TOTAL {tot_n:,} captions in {tot_s / 3600:.2f} h = {tot_n / tot_s:,.0f} items/s overall")


if __name__ == "__main__":
    main()

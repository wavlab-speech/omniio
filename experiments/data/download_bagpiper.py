"""Figure 5's corpus from the Hugging Face Hub: the Opus audio archives of the Bagpiper
pre-training collection (one directory per dataset with shard-NNNNNN.bin blobs and a merged
metadata.parquet whose rows carry path, start_byte_offset, file_size_bytes, start_time,
end_time, category and the caption; 14 TB). The index stores absolute paths from the machine that wrote it, so after the
download the `path` column of every metadata.parquet is pointed at the local shards (the same
rewrite as the dataset's tools/prepare_local.py).

    python data/download_bagpiper.py [--workers 8] [--datasets clotho_test ...]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data  # noqa: E402

REPO = "espnet/Bagpiper_PreTrain_Data_Collection"


def rewrite_paths(ds_dir: Path) -> int:
    """Repoint the `path` column of metadata.parquet at this machine's copy of the shards (streamed:
    the tables reach tens of millions of rows and carry the captions)."""
    meta = ds_dir / "metadata.parquet"
    pf = pq.ParquetFile(meta)
    if pf.metadata.num_rows == 0:
        return 0
    head = pf.read_row_group(0, columns=["path"]).column("path")[0].as_py()
    if str(Path(head).parent) == str(ds_dir):
        return 0
    schema = pf.schema_arrow; i = schema.names.index("path"); ptype = schema.field("path").type
    tmp = meta.with_suffix(".parquet.tmp"); n = 0
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    try:
        for b in pf.iter_batches(batch_size=200_000):
            t = pa.Table.from_batches([b], schema=schema)
            t = t.set_column(i, "path", pa.array([str(ds_dir / Path(x).name) for x in t.column("path").to_pylist()], ptype))
            writer.write_table(t); n += t.num_rows
    finally:
        writer.close()
    os.replace(tmp, meta)
    return n


def main():
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=None, help="only these datasets (default: all of audio/)")
    a = ap.parse_args()
    root = Path(data("bagpiper"))
    patterns = [f"audio/{d}/*" for d in a.datasets] if a.datasets else ["audio/*"]
    t0 = time.time()
    snapshot_download(REPO, repo_type="dataset", local_dir=str(root), allow_patterns=patterns, max_workers=a.workers)
    print(f"downloaded in {time.time() - t0:.0f} s", flush=True)
    for d in sorted(p for p in (root / "audio").iterdir() if (p / "metadata.parquet").exists()):
        n = rewrite_paths(d)
        print(f"{d.name}: {'paths rewritten for ' + f'{n:,} rows' if n else 'paths already local'}", flush=True)


if __name__ == "__main__":
    main()

"""
Figure 5 (left): random access that does not know the corpus size. The compact index of the
Bagpiper corpus is loaded for growing row counts (1 M, 10 M, 100 M, all), the load time
and resident memory recorded, and 2 000 cold random samples read: the row's audio entry and its
caption, two preads from two archives, then both decodes for 200 of them. The index subsets are
nested so the larger ones include the same shards plus more. One repeat per fresh node.

    python bigscale/access_scale.py --rep 1
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data, node, results  # noqa: E402

IDX = data("bagpiper", "index")
TEXT = data("bagpiper", "text")


def rss_mb() -> float:
    """Resident set of this process in MB (smaps_rollup, then VmHWM, then getrusage)."""
    for path, key in (("/proc/self/smaps_rollup", "Rss:"), ("/proc/self/status", "VmHWM:")):
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith(key):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_text_index(datasets):
    """caption (path, offset, size) per row, concatenated over datasets in the audio index's order"""
    import pyarrow.parquet as pq
    t_sh, t_off, t_sz, t_paths = [], [], [], []
    for d in datasets:
        n = len(np.load(os.path.join(IDX, d + ".npz"))["offset"])
        t = pq.read_table(os.path.join(TEXT, d, "metadata.parquet"), columns=["path", "start_byte", "end_byte"])
        assert t.num_rows == n, (d, t.num_rows, n)
        p = t["path"].combine_chunks()
        if hasattr(p, "dictionary"):
            tp = p.dictionary.to_pylist(); tix = p.indices.to_numpy().astype(np.int32)
        else:
            tp, inv = np.unique(np.asarray(p.to_pylist()), return_inverse=True); tp = tp.tolist(); tix = inv.astype(np.int32)
        t_sh.append(tix + len(t_paths)); t_paths += tp; so = t["start_byte"].to_numpy(); t_off.append(so); t_sz.append(t["end_byte"].to_numpy() - so)
    return {"sh": np.concatenate(t_sh), "off": np.concatenate(t_off), "sz": np.concatenate(t_sz), "paths": t_paths}


def load_index(limit=None):
    """compact arrays over all datasets (storage order), optionally the first `limit` rows"""
    ds = sorted(f[:-4] for f in os.listdir(IDX) if f.endswith(".npz")); parts = []; paths = []; n = 0
    for d in ds:
        z = np.load(os.path.join(IDX, d + ".npz")); m = len(z["offset"])
        take = m if limit is None else max(0, min(m, limit - n))
        if take == 0:
            break
        base = len(paths); paths += z["paths"].tolist()
        parts.append((z["shard"][:take].astype(np.int32) + base, z["offset"][:take], z["size"][:take], z["t0"][:take], z["t1"][:take])); n += take
        if limit is not None and n >= limit:
            break
    shard = np.concatenate([p[0] for p in parts]); off = np.concatenate([p[1] for p in parts]); size = np.concatenate([p[2] for p in parts])
    t0 = np.concatenate([p[3] for p in parts]); t1 = np.concatenate([p[4] for p in parts])
    return {"shard": shard, "offset": off, "size": size, "paths": paths, "t0": t0, "t1": t1}


def window(ix, i):
    """(start_time, end_time) of row i, or (None, None) for a whole-entry row"""
    a, b = float(ix["t0"][i]), float(ix["t1"][i])
    return (a, b) if np.isfinite(a) and np.isfinite(b) and b > a else (None, None)


def rchar():
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("rchar"):
                return int(line.split()[1])
    return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_reads", type=int, default=2000); ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", default=None); ap.add_argument("--levels", nargs="+", type=int, default=[1_000_000, 10_000_000, 100_000_000, 0], help="row counts, 0 = all")
    a = ap.parse_args()
    out = a.out or results("bigscale", f"access_rep{a.rep}_{node()}.json")
    from omniio.blob.handles import archive_fd, pread_all
    from omniio.interface import audio_read
    import zstandard
    res = {"node": os.uname().nodename, "rep": a.rep, "rss_after_imports_mb": rss_mb(), "levels": []}
    t0 = time.perf_counter(); ds = sorted(f[:-4] for f in os.listdir(IDX) if f.endswith(".npz")); text = load_text_index(ds)
    print(f"caption index for {len(text['off']):,} rows in {time.perf_counter() - t0:.0f} s", flush=True)
    dz = zstandard.ZstdDecompressor()
    for limit in [l or None for l in a.levels]:
        r0 = rss_mb(); t0 = time.perf_counter(); ix = load_index(limit); tl = time.perf_counter() - t0; r1 = rss_mb(); n = len(ix["offset"])
        rng = random.Random(7 + 100 * a.rep + (limit or 0) % 97); picks = [rng.randrange(n) for _ in range(a.n_reads)]
        lat = []; latt = []; byt = 0
        for i in picks:                                           # cold: a fresh node, and a sample disjoint from the other levels' samples in expectation
            t1 = time.perf_counter(); ba = pread_all(archive_fd(ix["paths"][ix["shard"][i]]), int(ix["size"][i]), int(ix["offset"][i])); ta = time.perf_counter() - t1
            t1 = time.perf_counter(); bt = pread_all(archive_fd(text["paths"][text["sh"][i]]), int(text["sz"][i]), int(text["off"][i])); tt_ = time.perf_counter() - t1
            lat.append(ta + tt_); latt.append(tt_); byt += len(ba) + len(bt)
        dec = []; samples = 0; c0 = rchar()
        for i in picks[:200]:                                     # read + decode of both (the audio segment where the row is a window of a stored track)
            w0, w1 = window(ix, i)
            t1 = time.perf_counter(); samples += audio_read(ix["paths"][ix["shard"][i]], int(ix["offset"][i]), int(ix["size"][i]), start_time=w0, end_time=w1).array.shape[0]
            bt = pread_all(archive_fd(text["paths"][text["sh"][i]]), int(text["sz"][i]), int(text["off"][i])); dz.decompress(bt) if bt else b""; dec.append(time.perf_counter() - t1)
        row = {"rows": n, "index_load_s": tl, "index_rss_mb": r1 - r0, "index_bytes_per_row": (r1 - r0) * 1e6 / n, "shards": len(ix["paths"]),
               "mm_cold_median_ms": float(np.median(lat) * 1e3), "mm_cold_p90_ms": float(np.percentile(lat, 90) * 1e3), "mm_text_cold_median_ms": float(np.median(latt) * 1e3),
               "kb_per_sample": byt / len(picks) / 1e3, "mm_decode_median_ms": float(np.median(dec) * 1e3), "mm_decode_mean_ms": float(np.mean(dec) * 1e3),
               "decode_audio_s_per_item": samples / 16000 / len(dec), "decode_kb_touched_per_item": (rchar() - c0) / len(dec) / 1e3}
        res["levels"].append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {n:>12,} rows: index {tl:5.1f} s, {r1 - r0:7.0f} MB resident ({row['index_bytes_per_row']:.1f} B/row), {row['shards']:,} shards | "
              f"audio + caption cold read median {row['mm_cold_median_ms']:.3f} ms (caption {row['mm_text_cold_median_ms']:.3f}) p90 {row['mm_cold_p90_ms']:.2f} ({row['kb_per_sample']:.0f} KB) | read + decode {row['mm_decode_median_ms']:.1f} ms", flush=True)
        json.dump(res, open(out, "w"), indent=1); del ix
    print("wrote", out)


if __name__ == "__main__":
    main()

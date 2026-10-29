"""
Figure 5 (right): random-order DataLoader throughput over the whole Bagpiper corpus from 1 to 8
nodes. Every rank loads the compact index, takes its share of
one global permutation (8 reads in flight per worker through omniio.blob.prefetch.Lookahead),
runs a DataLoader with W workers for a fixed wall time, and records items/s and MB/s, bytes only
and with the Opus decode. Aggregate = the sum over ranks.

    srun -N 4 --ntasks-per-node=1 --cpus-per-task=32 python bigscale/loader_nodes.py --seconds 600 --workers 32
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigscale.access_scale import load_index, window  # noqa: E402
from common.paths import results  # noqa: E402


def _iter_random(self):
    import torch
    from omniio.blob.prefetch import Lookahead, local_fetch
    info = torch.utils.data.get_worker_info(); wid, nw = (info.id, info.num_workers) if info else (0, 1)
    ix = self.ix; mine = self.perm[wid::nw]
    reqs = ((ix["paths"][ix["shard"][i]], int(ix["offset"][i]), int(ix["size"][i])) for i in mine)
    la = Lookahead(local_fetch, depth=self.depth)
    try:
        for i, raw in zip(mine, la(reqs)):
            yield self.decode(raw, i)
    finally:
        la.close()


def make_dataset(ix, perm, depth, decode):
    import torch
    cls = type("_BigStream", (torch.utils.data.IterableDataset,), {"__iter__": _iter_random})
    obj = cls(); obj.ix, obj.perm, obj.depth = ix, perm, depth
    if decode:
        import io
        from omniio.modalities.audio.read import _read_pcm
        def dec(raw, i):                                                # the entry's bytes are in memory; decode the row's segment (Ogg/Opus via libsndfile, the decoder audio_read uses)
            w0, w1 = window(ix, i)
            return _read_pcm(io.BytesIO(raw), "ogg", w0, w1).array.shape[0], len(raw)
        obj.decode = dec
    else:
        obj.decode = lambda raw, i: (len(raw), len(raw))
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modes", nargs="+", default=["random_bytes", "random_decode"], choices=["random_bytes", "random_decode"])
    ap.add_argument("--seconds", type=int, default=600); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="rows of the index to use (smoke tests)")
    a = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0)); world = int(os.environ.get("SLURM_NTASKS", 1)); node = os.uname().nodename.split(".")[0]
    RES = results("bigscale")
    import torch
    from torch.utils.data import DataLoader
    t0 = time.time(); ix = load_index(a.limit); N = len(ix["offset"]); t_index = time.time() - t0
    t0 = time.time(); perm = np.random.default_rng(a.seed).permutation(N)[rank::world]; t_perm = time.time() - t0
    print(f"[rank {rank}/{world} {node}] index {N:,} rows in {t_index:.0f} s, permutation share {len(perm):,} in {t_perm:.0f} s", flush=True)
    out = {"rank": rank, "world": world, "node": node, "rows": N, "workers": a.workers, "depth": a.depth, "index_load_s": t_index, "modes": {}}
    for mode in a.modes:
        ds = make_dataset(ix, perm, a.depth, decode=mode.endswith("decode"))
        dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=list, multiprocessing_context="fork", prefetch_factor=4)
        n = 0; byt = 0; val = 0; t0 = time.perf_counter(); first = None
        for b in dl:
            if first is None:
                first = time.perf_counter() - t0; t1 = time.perf_counter(); continue
            n += len(b); val += sum(v for v, _ in b); byt += sum(t for _, t in b)   # val: bytes (bytes modes) or decoded samples; byt: bytes read
            if time.perf_counter() - t0 > a.seconds:
                break
        sec = time.perf_counter() - t1
        dec = mode.endswith("decode")
        row = {"items": n, "seconds": sec, "items_per_s": n / sec, "MB_per_s": byt / 1e6 / sec, "kb_per_item": byt / n / 1e3, "first_batch_s": first,
               "audio_s_per_s": val / 16000 / sec if dec else None, "audio_s_per_item": val / 16000 / n if dec else None}
        out["modes"][mode] = row
        print(f"[rank {rank}/{world} {node}] {mode:18s} {n:>10,} items in {sec:5.0f} s = {n / sec:8,.0f} items/s" + f", {byt / 1e6 / sec:7.0f} MB/s ({byt / n / 1e3:.0f} KB/item)" + (f", {val / 16000 / sec:8,.0f} audio-s/s ({val / 16000 / n:.1f} s/item)" if dec else "") + f" (first batch {first:.1f} s)", flush=True)
        json.dump(out, open(os.path.join(RES, f"loader_nodes{world}{('_' + a.tag) if a.tag else ''}_rank{rank}.json"), "w"), indent=1)
        del dl, ds
    print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()

"""Figure 6 (top), the flat reference: decode the whole recording, then take the window (what a
format with no window support pays). Three recordings, times only, frames counted not kept.

    python windowed/speed_full_decode.py
"""
import json
import os
import random
import sys
import time

import av
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data, node, results  # noqa: E402

VIDEO = data("video")


def decode_all(path):
    c = av.open(path); vs = c.streams.video[0]; n = 0
    for _ in c.decode(vs):
        n += 1
    c.close(); return n


def main():
    raw = sorted(pq.read_table(os.path.join(VIDEO, "raw", "manifest.parquet")).to_pylist(), key=lambda r: r["path"])[:10]
    rng = random.Random(7); res = {"node": os.uname().nodename, "runs": []}
    out = results("windowed", f"speed_full_decode_{node()}.json")
    for _ in range(3):
        r = rng.choice(raw)
        t0 = time.perf_counter(); n = decode_all(r["path"]); dt = time.perf_counter() - t0
        row = {"method": "full_decode", "recording": os.path.basename(r["path"]), "frames_decoded": n, "seconds": dt}
        res["runs"].append(row); print(f"[{time.strftime('%H:%M:%S')}] full decode of {row['recording']}: {n} frames in {dt:.1f} s", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

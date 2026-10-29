"""
Figure 6 (bottom): peak memory of a partial (time-window) video read, per format.

Each (reader, window length) runs in a fresh process: open the reader, note the resident
high-water mark (VmHWM), perform the window reads, note VmHWM again. The difference is the peak
extra memory the windowed reads needed, C allocations included (PyAV); tracemalloc adds the
Python-side peak per read. Readers as in speed_windowed.py.

    python windowed/mem_windowed.py --n 20 --n_member 4 --passes 3
"""
import argparse
import gc
import io
import json
import os
import random
import subprocess
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data, node, results  # noqa: E402

VIDEO = data("video")
WINDOWS = [1, 3, 10, 30]
READERS = ["omniio_window", "raw_seek", "tar_member"]


def status(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0           # kB -> MB
    return float("nan")


def _decode_video(src, s, e, fps, seek):
    import av
    c = av.open(src); vs = c.streams.video[0]
    f0, f1 = int(round(s * fps)), int(round(e * fps))
    if seek:
        c.seek(int(s / float(vs.time_base)), stream=vs, backward=True, any_frame=False)
    frames = []
    for fr in c.decode(vs):
        idx = int(round(float(fr.pts * vs.time_base) * fps))
        if idx >= f1:
            break
        if idx >= f0:
            frames.append(fr.to_ndarray(format="rgb24"))
    c.close()
    return frames


def child(reader, w, n, seed):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    om = pq.read_table(os.path.join(VIDEO, "omniio", "metadata.parquet")).to_pylist()
    raw = {os.path.basename(r["path"])[:-4]: r for r in pq.read_table(os.path.join(VIDEO, "raw", "manifest.parquet")).to_pylist()}
    idx = json.load(open(os.path.join(VIDEO, "webdataset", "member_index.json")))
    if reader == "omniio_window":
        from omniio.modalities.video.read import video_read_local

        def read(k, s, e):
            m = om[k]; r = video_read_local(m["path"], m["start_byte"], m["end_byte"] - m["start_byte"], start_time=s, end_time=e); return len(r.video_array)
    elif reader == "raw_seek":
        def read(k, s, e):
            return len(_decode_video(raw[om[k]["id"]]["path"], s, e, om[k]["fps"], seek=True))
    else:
        from omniio.blob.handles import archive_fd, pread_all

        def read(k, s, e):
            shard, off, size = idx[om[k]["id"]]
            blob = pread_all(archive_fd(shard), size, off)                # the whole member: the format has no inner offsets
            return len(_decode_video(io.BytesIO(blob), s, e, om[k]["fps"], seek=True))
    recs = list(range(len(om))); dur = lambda k: om[k]["duration"]
    gc.collect()
    base = {"rss_after_open_mb": status("VmRSS"), "hwm_after_open_mb": status("VmHWM")}
    picks = []
    for _ in range(n):
        k = rng.choice(recs); d = dur(k); s = rng.uniform(0, max(0.0, d - w - 1)); picks.append((k, s, s + w))
    peaks, items, t0 = [], 0, time.perf_counter()
    for k, s, e in picks:
        gc.collect(); tracemalloc.start()
        items += read(k, s, e)
        _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); peaks.append(pk / 1e6)
    gc.collect()
    out = dict(base, hwm_after_reads_mb=status("VmHWM"), rss_after_reads_mb=status("VmRSS"), tracemalloc_peak_mb_median=float(np.median(peaks)),
               tracemalloc_peak_mb_max=float(np.max(peaks)), reads=n, seconds=time.perf_counter() - t0, items=items)
    out["peak_extra_mb"] = out["hwm_after_reads_mb"] - out["hwm_after_open_mb"]
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=4, metavar=("READER", "W", "N", "SEED"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--n_member", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    if a.child:
        r, w, n, seed = a.child; child(r, float(w), int(n), int(seed)); return
    out = a.out or results("windowed", f"mem_windowed_{node()}.json")
    res = {"node": os.uname().nodename, "time": time.ctime(), "runs": []}
    for reader in READERS:
        for w in WINDOWS:
            for p in range(a.passes):
                n = a.n_member if reader == "tar_member" else a.n
                t0 = time.time()
                cp = subprocess.run([sys.executable, __file__, "--child", reader, str(w), str(n), str(p)], capture_output=True, text=True)
                line = [l for l in cp.stdout.strip().splitlines() if l.startswith("{")]
                row = {"modality": "video", "reader": reader, "window_s": w, "pass": p, "wall_s": time.time() - t0}
                row.update(json.loads(line[-1]) if line else {"error": (cp.stderr or cp.stdout)[-400:]})
                res["runs"].append(row)
                print(f"[{time.strftime('%H:%M:%S')}] {reader} w={w}s pass {p}: " + (f"peak extra {row['peak_extra_mb']:.0f} MB (open {row['rss_after_open_mb']:.0f} MB resident), "
                      f"tracemalloc peak {row['tracemalloc_peak_mb_median']:.1f} MB, {row['reads']} reads in {row['seconds']:.0f} s" if "peak_extra_mb" in row else f"ERROR {row.get('error', '')[:200]}"), flush=True)
                json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

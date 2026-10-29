"""
Figure 6 (top): speed of a partial (time-window) video read, per format, on one node with the
readers interleaved.

For each window length: 20 windows at random offsets (fixed per repeat); pass 1 reads every
window once with the readers in round-robin order, so no reader is systematically first on a
cold client cache; pass 2 repeats the same windows (warm, the pass the figure shows). Every
reader returns the window's RGB frames and its decoded audio, as Omni-IO's reader does, so the
work compared is the same. Readers: Omni-IO window read, raw MP4 with a PyAV seek, and the
WebDataset member (the whole member read from the tar, then the seek inside it).

    python windowed/speed_windowed.py --rep 1
"""
import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data, node, results  # noqa: E402

VIDEO = data("video")
WINDOWS = [1, 3, 10, 30]


def decode_av(src, s, e, fps, seek=True, audio=True):
    """frames (count) and audio samples of [s, e) from an MP4 source via PyAV"""
    import av
    c = av.open(src); vs = c.streams.video[0]
    f0, f1 = int(round(s * fps)), int(round(e * fps))
    if seek:
        c.seek(int(s / float(vs.time_base)), stream=vs, backward=True, any_frame=False)
    frames = []
    for fr in c.decode(vs):
        k = int(round(float(fr.pts * vs.time_base) * fps))
        if k >= f1:
            break
        if k >= f0:
            frames.append(fr.to_ndarray(format="rgb24"))
    n_audio = 0
    if audio and c.streams.audio:
        a = c.streams.audio[0]; tb = float(a.time_base); sr = a.rate
        c.seek(int(max(0.0, s - 0.1) / tb), stream=a, backward=True, any_frame=False)
        a0, a1 = int(s * sr), int(e * sr)
        for fr in c.decode(a):
            pos = int(round(fr.pts * tb * sr)) if fr.pts is not None else 0
            arr = fr.to_ndarray(); n = arr.shape[-1]
            if pos + n <= a0:
                continue
            if pos >= a1:
                break
            n_audio += min(pos + n, a1) - max(pos, a0)
    c.close()
    return len(frames), n_audio


def video_readers():
    import pyarrow.parquet as pq
    from omniio.blob.handles import archive_fd, pread_all
    from omniio.modalities.video.read import video_read_local
    om = pq.read_table(os.path.join(VIDEO, "omniio", "metadata.parquet")).to_pylist()
    raw = {os.path.basename(r["path"])[:-4]: r for r in pq.read_table(os.path.join(VIDEO, "raw", "manifest.parquet")).to_pylist()}
    idx = json.load(open(os.path.join(VIDEO, "webdataset", "member_index.json")))
    fps = lambda k: om[k]["fps"]; dur = lambda k: om[k]["duration"]
    R = {}

    def omniio(k, s, e):
        m = om[k]; r = video_read_local(m["path"], m["start_byte"], m["end_byte"] - m["start_byte"], start_time=s, end_time=e); return len(r.video_array)
    R["omniio_window"] = omniio
    R["raw_seek"] = lambda k, s, e: decode_av(raw[om[k]["id"]]["path"], s, e, fps(k))[0]

    def tar(k, s, e):
        shard, off, size = idx[om[k]["id"]]; blob = pread_all(archive_fd(shard), size, off); return decode_av(io.BytesIO(blob), s, e, fps(k))[0]
    R["tar_member"] = tar
    return R, list(range(len(om))), dur


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rep", type=int, default=1); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    out = a.out or results("windowed", f"speed_windowed_rep{a.rep}_{node()}.json")
    res = {"node": os.uname().nodename, "rep": a.rep, "time": time.ctime(), "runs": []}
    R, recs, dur = video_readers(); names = list(R)
    for w in WINDOWS:
        rng = random.Random(1000 * a.rep + w)
        picks = []
        for _ in range(a.n):
            k = rng.choice(recs); d = dur(k); s = rng.uniform(0, max(0.0, d - w - 1)); picks.append((k, s, s + w))
        times = {nm: [[], []] for nm in names}
        for p in (0, 1):                                          # pass 0: first touch (round-robin over readers); pass 1: the same windows warm
            for j, (k, s, e) in enumerate(picks):
                order = names[j % len(names):] + names[:j % len(names)]     # rotate which reader goes first
                for nm in order:
                    if nm == "tar_member" and j >= 4:
                        continue                                  # the whole-member read is up to 3.75 GB a time: 4 windows
                    t0 = time.perf_counter()
                    try:
                        R[nm](k, s, e); times[nm][p].append(time.perf_counter() - t0)
                    except Exception as ex:
                        times[nm][p].append(float("nan")); print(f"{nm} failed: {repr(ex)[:160]}", file=sys.stderr)
        for nm in names:
            c, h = np.array(times[nm][0]) * 1e3, np.array(times[nm][1]) * 1e3
            row = {"modality": "video", "reader": nm, "window_s": w, "n": len(c), "first_pass_ms_median": float(np.nanmedian(c)), "first_pass_ms_mean": float(np.nanmean(c)),
                   "warm_ms_median": float(np.nanmedian(h)), "warm_ms_mean": float(np.nanmean(h))}
            res["runs"].append(row)
            print(f"[{time.strftime('%H:%M:%S')}] w={w:>2}s {nm:16s} first {row['first_pass_ms_median']:8.1f} ms  warm {row['warm_ms_median']:8.1f} ms  (n={len(c)})", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""
The long-recording video corpus of Figure 6: 50 synthetic MP4 recordings of 30 to 60 minutes
(1280x720, 30 fps, H.264 via libx264 with a 2 s GOP, AAC 48 kHz stereo), made with PyAV from a
procedural source (a moving pattern with a burned-in timecode and frame index), then stored
three ways: an Omni-IO video archive (one entry per recording), WebDataset tar shards with a
member offset index (5 recordings per shard), and the raw MP4 files.

    python data/make_video_corpus.py make --n 50 --workers 16
    python data/make_video_corpus.py ingest
"""
import argparse
import os
import random
import sys
import tarfile
import time
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import data  # noqa: E402

W, H, FPS, SR = 1280, 720, 30, 48000
BLOCK = 32


def frame_image(i, phase):
    """Moving pattern + frame-index code blocks + text."""
    from PIL import Image, ImageDraw
    y, x = np.mgrid[0:H, 0:W]
    shift = (i * 3) % W
    r = ((x + shift) // 40 % 2) * 120 + 40
    g = ((y + shift // 2) // 40 % 2) * 120 + 40
    b = (((x + y + i * 5) // 60) % 2) * 120 + 40
    img = np.stack([r, g, b], -1).astype(np.uint8)
    img[:BLOCK, :24 * BLOCK] = 0
    for k in range(24):
        if (i >> (23 - k)) & 1:
            img[:BLOCK, k * BLOCK:(k + 1) * BLOCK] = 255
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    t = i / FPS
    d.rectangle([0, H - 60, 420, H], fill=(0, 0, 0))
    d.text((10, H - 50), f"frame {i:07d}  t={int(t // 60):02d}:{t % 60:06.3f}  {phase}", fill=(255, 255, 255))
    return np.asarray(im)



def _make_one(job):
    import av
    path, seconds, gop, seed = job
    if os.path.exists(path):
        return path, os.path.getsize(path), 0.0
    t0 = time.time()
    n_frames = int(seconds * FPS)
    out = av.open(path + ".tmp.mp4", "w", format="mp4")
    vs = out.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"preset": "ultrafast", "crf": "23", "g": str(gop * FPS), "keyint_min": str(gop * FPS), "sc_threshold": "0", "bf": "0"}
    as_ = out.add_stream("aac", rate=SR)
    as_.layout = "stereo"
    rng = np.random.default_rng(seed)
    t_audio = np.arange(1024) / SR
    a_pts = 0
    phase = f"v{seed}"
    for i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(frame_image(i, phase), format="rgb24")
        frame.pts = i
        for pkt in vs.encode(frame):
            out.mux(pkt)
        # audio: a tone whose frequency changes every second, 1024-sample AAC frames to keep up with the video clock
        while a_pts < (i + 1) * SR / FPS:
            f0 = 220 + 20 * ((a_pts // SR) % 20)
            sig = (0.3 * np.sin(2 * np.pi * f0 * (t_audio + a_pts / SR))).astype(np.float32)
            af = av.AudioFrame.from_ndarray(np.stack([sig, sig])[None].reshape(1, -1) if False else np.stack([sig, sig]), format="fltp", layout="stereo")
            af.sample_rate = SR; af.pts = a_pts
            for pkt in as_.encode(af):
                out.mux(pkt)
            a_pts += 1024
    for pkt in vs.encode():
        out.mux(pkt)
    for pkt in as_.encode():
        out.mux(pkt)
    out.close()
    os.rename(path + ".tmp.mp4", path)
    return path, os.path.getsize(path), time.time() - t0


def cmd_make(args):
    d = os.path.join(args.root, "raw")
    os.makedirs(d, exist_ok=True)
    rng = random.Random(1234)
    jobs = [(os.path.join(d, f"rec_{k:03d}.mp4"), rng.uniform(30 * 60, 60 * 60), args.gop, k) for k in range(args.n)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_make_one, jobs))
    meta = [{"path": p, "bytes": b, "seconds": s, "duration": j[1], "gop_s": args.gop, "fps": FPS, "width": W, "height": H,
             "codec": "libx264 ultrafast crf23 yuv420p, bf=0, sc_threshold=0", "audio": "aac 48 kHz stereo"} for (p, b, s), j in zip(res, jobs)]
    pq.write_table(pa.Table.from_pylist(meta), os.path.join(d, "manifest.parquet"))
    tot = sum(m["bytes"] for m in meta)
    print(f"{len(meta)} videos, {tot / 1e9:.1f} GB, {sum(m['duration'] for m in meta) / 3600:.1f} h, wall {time.time() - t0:.0f} s, "
          f"mean bitrate {tot * 8 / sum(m['duration'] for m in meta) / 1e6:.2f} Mb/s", flush=True)


def cmd_ingest(args):
    if True:
        d = os.path.join(args.root, "raw")
        rows = pq.read_table(os.path.join(d, "manifest.parquet")).to_pylist()
        if os.path.exists(os.path.join(args.root, "webdataset", "member_index.json")):
            print("already ingested", flush=True); return
        # Omni-IO: byte copy into bins (one bin per recording keeps bins near the library's default size)
        od = os.path.join(args.root, "omniio"); os.makedirs(od, exist_ok=True)
        meta = []
        for k, r in enumerate(rows):
            bp = os.path.join(od, f"blob_{k}.bin")
            with open(r["path"], "rb") as f, open(bp, "wb") as g:
                n = 0
                while True:
                    chunk = f.read(16 << 20)
                    if not chunk:
                        break
                    g.write(chunk); n += len(chunk)
            meta.append({"id": os.path.basename(r["path"])[:-4], "path": bp, "start_byte": 0, "end_byte": n, "bin_index": k, "duration": r["duration"], "fps": FPS})
        pq.write_table(pa.Table.from_pylist(meta), os.path.join(od, "metadata.parquet"))
        # WebDataset: 5 recordings per shard, member index with data offsets
        wd = os.path.join(args.root, "webdataset"); os.makedirs(wd, exist_ok=True)
        index = {}
        for s in range(0, len(rows), 5):
            sp = os.path.join(wd, f"shard_{s // 5:04d}.tar")
            with tarfile.open(sp, "w") as tar:
                for r in rows[s:s + 5]:
                    tar.add(r["path"], arcname=os.path.basename(r["path"]))
            with tarfile.open(sp, "r:") as tar:
                for m in tar:
                    index[m.name[:-4]] = (sp, m.offset_data, m.size)
        json.dump(index, open(os.path.join(wd, "member_index.json"), "w"))
        print(len(rows), "recordings ingested", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "ingest"])
    ap.add_argument("--root", default=data("video"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gop", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    {"make": cmd_make, "ingest": cmd_ingest}[args.cmd](args)


if __name__ == "__main__":
    main()

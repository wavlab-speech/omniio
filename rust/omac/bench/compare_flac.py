"""Round trip and size/speed comparison of OMAC against FLAC on real arrays.

    CHIME6_AUDIO=/path/to/chime6/audio AMI_WAV_DB=/path/to/AMI/wav_db python compare_flac.py

Uses CHiME-6 dev session S02 and AMI meeting ES2003a. Everything is in memory:
this measures the codecs, not disk or network.
"""
import io
import os
import time

import numpy as np
import soundfile as sf

import omac

CHIME6 = os.environ.get("CHIME6_AUDIO", "chime6/audio") + "/dev/S02_U0{u}.CH{c}.wav"
AMI = os.environ.get("AMI_WAV_DB", "AMI/wav_db") + "/ES2003a/audio/ES2003a.Array{a}-0{c}.wav"


def load(paths, max_s=None):
    chans = []
    for p in paths:
        with sf.SoundFile(p) as f:
            sr = f.samplerate
            n = f.frames if max_s is None else min(f.frames, int(max_s * sr))
            chans.append(f.read(n, dtype="int16"))
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], 1), sr


def best(fn, n=5):
    t = 1e9
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t * 1e3


def flac_bytes(x, sr):
    b = io.BytesIO()
    sf.write(b, x, sr, format="FLAC", subtype="PCM_16")
    return b.getvalue()


def window(enc, lo, hi, channels=None, threads=1):
    """Decode a window the way a range reader would: header, then only its blocks."""
    info = omac.read_info(enc[:40])
    b = info["block_size"]
    k0, k1 = lo // b, (hi - 1) // b
    idx = 40 + 8 * info["channels"]
    offs = np.frombuffer(enc, "<u8", k1 - k0 + 2, idx + 8 * k0).tolist()
    data = enc[offs[0]:offs[-1]]
    return omac.decode_range(enc[:40], data, offs, k0, lo, hi, channels, "float32", True, threads)


def run(name, x, sr):
    frames, C = x.shape
    raw = x.nbytes
    print(f"\n== {name}: {C} ch, {frames / sr / 60:.1f} min, {sr} Hz")

    t = time.perf_counter()
    enc5 = omac.encode(x.astype(np.int32), sr, 16, level=5, threads=16)
    t5 = time.perf_counter() - t
    t = time.perf_counter()
    enc8 = omac.encode(x.astype(np.int32), sr, 16, level=8, threads=16)
    t8 = time.perf_counter() - t
    y, _ = omac.decode(enc5, dtype="int32", threads=16)
    assert np.array_equal(y, x), "level 5 round trip failed"
    y, _ = omac.decode(enc8, dtype="int32", threads=16)
    assert np.array_equal(y, x), "level 8 round trip failed"
    print("  round trip: bit-exact (levels 5 and 8)")

    groups = [list(range(i, min(i + 8, C))) for i in range(0, C, 8)]
    flac_multi = [flac_bytes(x[:, g], sr) for g in groups]  # FLAC caps at 8 channels
    flac_mono = [flac_bytes(x[:, c], sr) for c in range(C)]
    fm = sum(map(len, flac_multi))
    print(f"  raw PCM16           {raw / 1e6:8.1f} MB")
    print(f"  FLAC ({'1x' if C <= 8 else f'{len(groups)}x'}<=8ch)     {fm / 1e6:8.1f} MB")
    print(f"  FLAC (mono files)   {sum(map(len, flac_mono)) / 1e6:8.1f} MB")
    print(f"  OMAC level 5        {len(enc5) / 1e6:8.1f} MB  ({100 * (len(enc5) / fm - 1):+.2f}% vs FLAC)  encode {t5:.1f}s, 16 threads")
    print(f"  OMAC level 8        {len(enc8) / 1e6:8.1f} MB  ({100 * (len(enc8) / fm - 1):+.2f}% vs FLAC)  encode {t8:.1f}s, 16 threads")

    def sf_window(blobs, lo, n):
        out = [sf.read(io.BytesIO(b), start=lo, frames=n, dtype="float32", always_2d=True)[0] for b in blobs]
        return np.concatenate(out, 1)

    one_min = (min(2 * 60 * sr, frames // 2), 60 * sr)
    two_s = (frames // 2, 2 * sr)
    print("  decode time, ms (best of 5)        FLAC <=8ch   FLAC mono   OMAC 1 thr   OMAC 8 thr")
    for label, (lo, n) in [("whole entry", (0, frames)), ("1 min window", one_min), ("2 s window", two_s)]:
        n = min(n, frames - lo)
        a = best(lambda: sf_window(flac_multi, lo, n), 3 if label == "whole entry" else 5)
        b = best(lambda: sf_window(flac_mono, lo, n), 3 if label == "whole entry" else 5)
        c = best(lambda: window(enc5, lo, lo + n, None, 1), 3 if label == "whole entry" else 5)
        d = best(lambda: window(enc5, lo, lo + n, None, 8), 3 if label == "whole entry" else 5)
        print(f"    {label:14s} all channels   {a:10.1f}  {b:10.1f}  {c:10.1f}  {d:10.1f}")
    lo, n = two_s
    a = best(lambda: sf.read(io.BytesIO(flac_multi[0]), start=lo, frames=n, dtype="float32")[0][:, 0], 10)
    b = best(lambda: sf.read(io.BytesIO(flac_mono[0]), start=lo, frames=n, dtype="float32"), 10)
    c = best(lambda: window(enc5, lo, lo + n, [0], 1), 10)
    print(f"    {'2 s window':14s} 1 channel      {a:10.2f}  {b:10.2f}  {c:10.2f}")
    ref = sf_window(flac_mono, one_min[0], one_min[1])
    assert np.array_equal(window(enc5, one_min[0], sum(one_min), None, 8), ref), "window mismatch"


if __name__ == "__main__":
    x, sr = load([CHIME6.format(u=1, c=c) for c in range(1, 5)])
    run("CHiME-6 dev S02 U01 (one Kinect)", x, sr)
    x, sr = load([CHIME6.format(u=u, c=c) for u in range(1, 7) for c in range(1, 5)], max_s=20 * 60)
    run("CHiME-6 dev S02 U01-U06 (6 Kinects, first 20 min)", x, sr)
    x, sr = load([AMI.format(a=1, c=c) for c in range(1, 9)])
    run("AMI ES2003a Array1", x, sr)
    x, sr = load([AMI.format(a=a, c=c) for a in (1, 2) for c in range(1, 9)])
    run("AMI ES2003a Array1+Array2 (MDM)", x, sr)

"""
Figures 4 and 10: ingestion, random access, DataLoader throughput and sequential streaming for
the image / text / video corpora, Omni-IO against WebDataset, Lance, HDF5, Deep Lake, MosaicML
MDS and raw files, on a parallel filesystem.

Corpora (storage order = source order; sources downloaded by data/download_formats.py):
  image  ImageNet-1k train, 1 281 167 JPEGs (the 256 px copy)
  text   PG-19 train, books split into ~4 KB paragraph-aligned chunks
  video  UCF101, 13 320 clips, transcoded once from AVI to MP4 H.264 (the common input)

  materialize  write the raw files from the source (this is the "raw files" ingestion;
               video: transcode avi -> mp4 once, timed separately)
  ingest       build one format's store from the raw files, timed; --n limits to the first n
               items (worker sweeps), --tag names the output; omniio_bytes is Omni-IO's fast
               mode (Blob.append(fast_mode=True): byte copies, no metadata)
  access       random reads, single process: cold pass (first touch on this node) then warm,
               one entry per read for every format that is present
  loader       DataLoader items/s with decode: Omni-IO random (one pread per item), Omni-IO
               with a threaded prefetch of the batch's reads, Omni-IO random order with 8 reads
               in flight per worker (Lookahead), WebDataset streaming with its 1 000-item
               buffer, Lance / Deep Lake batched takes, HDF5, MDS shuffled and random, raw files
  stream       sequential pass: every format in storage order; bytes only and with decode

    python formats/scale_bench.py materialize --modality image --workers 32
    python scale_bench.py ingest --modality image --fmt omniio --workers 8 --n 100000 --tag w8
    python scale_bench.py access --modality image
    python scale_bench.py loader --modality image --workers 1 4 8 16
    python scale_bench.py stream --modality image
"""
import argparse
import glob
import io
import json
import os
import random
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.harness import write_result  # noqa: E402
from common.paths import data, results  # noqa: E402

ROOT = data("formats")                                                 # <modality>/raw, <modality>/<fmt>/<tag>
RESULTS = results("formats")
SRC = {"image": data("sources", "imagenet256", "data"), "text": data("sources", "pg19", "data"), "video": data("sources", "ucf101", "UCF-101.zip")}
EXT = {"image": "jpg", "text": "txt", "video": "mp4"}
SHARD = {"image": 1000, "text": 1000, "video": 500}      # items per WebDataset shard (video: 27 shards for 16 workers)


def key_of(i):
    """archive id / WebDataset key: no '/' and no '.', so the tar key splits cleanly"""
    return i.replace("/", "_").replace(".", "_")


def P(modality, *parts):
    return os.path.join(ROOT, modality, *parts)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────────── source enumeration ─────────────────────────────
def manifest_path(modality):
    return P(modality, "manifest.parquet")


def load_manifest(modality):
    """(ids, raw paths, labels) in storage order."""
    import pyarrow.parquet as pq
    t = pq.read_table(manifest_path(modality))
    return {"ids": t["id"].to_pylist(), "raw": t["raw_path"].to_pylist(), "label": t["label"].to_pylist()}


# ───────────────────────────── materialize raw files ─────────────────────────────
def _image_chunk(args):
    """Write the JPEGs of one HF parquet file as raw files; returns rows (id, raw_path, label)."""
    f, out, limit = args
    import pyarrow.parquet as pq
    rows = []
    t = pq.read_table(f, columns=["image", "label"])
    if limit:
        t = t.slice(0, limit)
    for k, (im, lab) in enumerate(zip(t["image"].to_pylist(), t["label"].to_pylist())):
        name = os.path.basename(im.get("path") or f"{os.path.basename(f)}_{k}.JPEG")
        d = os.path.join(out, f"{lab:04d}"); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name if name.lower().endswith((".jpg", ".jpeg")) else name + ".jpg")
        with open(p, "wb") as fh:
            fh.write(im["bytes"])
        rows.append((f"{lab:04d}/{os.path.basename(p)}", p, int(lab)))
    return rows


def _text_chunks(text, target=4096):
    """Paragraph-aligned chunks of about `target` characters."""
    out, cur = [], []
    n = 0
    for para in text.split("\n\n"):
        cur.append(para); n += len(para) + 2
        if n >= target:
            out.append("\n\n".join(cur)); cur, n = [], 0
    if cur:
        out.append("\n\n".join(cur))
    return [c for c in out if len(c) >= 200]


def _text_file(args):
    f, out, first_book, limit = args
    import pyarrow.parquet as pq
    rows = []; book = first_book
    t = pq.read_table(f, columns=["short_book_title", "text"])
    if limit:
        t = t.slice(0, limit)
    for text in t["text"].to_pylist():
        d = os.path.join(out, f"{book:05d}"); os.makedirs(d, exist_ok=True)
        for c, chunk in enumerate(_text_chunks(text)):
            p = os.path.join(d, f"{c:04d}.txt")
            with open(p, "w") as fh:
                fh.write(chunk)
            rows.append((f"{book:05d}/{c:04d}", p, book))
        book += 1
    return rows


def _video_clip(args):
    names, out = args
    import zipfile
    from omniio.modalities.video.write import video_write
    z = zipfile.ZipFile(SRC["video"]); rows = []
    for n in names:
        cls = n.split("/")[-2]; base = os.path.splitext(n.split("/")[-1])[0]
        d = os.path.join(out, cls); os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, base + ".avi"); p = os.path.join(d, base + ".mp4")
        with open(tmp, "wb") as fh:
            fh.write(z.read(n))
        raw, meta = video_write(tmp, base)                      # avi -> mp4 H.264 + aac (one transcode)
        with open(p, "wb") as fh:
            fh.write(raw)
        os.remove(tmp); rows.append((f"{cls}/{base}", p, cls))
    return rows


def cmd_materialize(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    m = args.modality; out = P(m, "raw"); os.makedirs(out, exist_ok=True)
    if os.path.exists(manifest_path(m)):
        log(f"{m}: raw files already materialized"); return
    t0 = time.time()
    lim = args.n                                                 # smoke tests: first file(s) only, n rows
    if m == "image":
        files = sorted(glob.glob(os.path.join(SRC["image"], "train-*.parquet")))
        if lim:
            files = files[:1]
        jobs = [(f, out, lim) for f in files]; fn = _image_chunk
    elif m == "text":
        files = sorted(glob.glob(os.path.join(SRC["text"], "train-*.parquet")))
        if lim:
            files = files[:1]
        counts = [pq.read_metadata(f).num_rows for f in files]; starts = np.cumsum([0] + counts[:-1])
        jobs = [(f, out, int(s), lim) for f, s in zip(files, starts)]; fn = _text_file
    else:
        import zipfile
        names = sorted(n for n in zipfile.ZipFile(SRC["video"]).namelist() if n.lower().endswith(".avi"))
        if lim:
            names = names[:lim]
        step = max(1, len(names) // (args.workers * 8)); jobs = [(names[i:i + step], out) for i in range(0, len(names), step)]; fn = _video_clip
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(fn, jobs)):
            rows += r
            if k % 10 == 0:
                log(f"  {m}: {len(rows)} files ({time.time() - t0:.0f} s)")
    rows.sort(key=lambda r: r[1])                               # storage order = directory order (class / book major)
    if m == "video":
        labels = sorted({r[2] for r in rows}); lab = {c: i for i, c in enumerate(labels)}; rows = [(a, b, lab[c]) for a, b, c in rows]
    pq.write_table(pa.table({"id": [r[0] for r in rows], "raw_path": [r[1] for r in rows], "label": [int(r[2]) for r in rows]}), manifest_path(m))
    sec = time.time() - t0; nbytes = sum(os.path.getsize(r[1]) for r in rows[:: max(1, len(rows) // 20000)]) * (len(rows) / len(rows[:: max(1, len(rows) // 20000)]))
    res = {"modality": m, "n": len(rows), "workers": args.workers, "seconds": sec, "files_per_s": len(rows) / sec, "approx_bytes": nbytes,
           "note": "video: includes the avi -> mp4 transcode" if m == "video" else "raw files written from the source parquet"}
    write_result(os.path.join(RESULTS, m, "materialize.json"), res); log(f"{m}: materialized {len(rows)} files in {sec:.0f} s ({res['files_per_s']:.0f}/s)")


# ───────────────────────────── ingestion ─────────────────────────────
def _wds_shard(args):
    paths, ids, out_tar, ext = args
    import webdataset as wds
    with wds.TarWriter(out_tar) as sink:
        for p, i in zip(paths, ids):
            with open(p, "rb") as fh:
                sink.write({"__key__": key_of(i), ext: fh.read()})
    # member index written by the writer (the file is in this node's cache now), so the reading
    # node never has to scan the tar headers, which would warm the shards before its cold pass
    idx = {}
    with tarfile.open(out_tar) as tf:
        for mem in tf:
            if mem.isfile():
                idx[mem.name] = (out_tar, mem.offset_data, mem.size)
    json.dump(idx, open(out_tar + ".idx.json", "w"))
    return len(paths)



def cmd_ingest(args):
    m = args.modality; man = load_manifest(m)
    n = min(args.n or len(man["ids"]), len(man["ids"])); ids, paths = man["ids"][:n], man["raw"][:n]
    tag = args.tag or f"w{args.workers}_n{n}"
    out = P(m, args.fmt, tag)
    if os.path.exists(os.path.join(out, "DONE")):
        log(f"{m} {args.fmt} {tag}: done already"); return
    if os.path.exists(out):
        import shutil; shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    if not args.no_warm:
        # read every source file once, untimed: the first format measured otherwise pays the
        # cold reads of the raw files and the later ones get them from the client cache
        def _touch(p):
            with open(p, "rb") as fh:
                return len(fh.read())
        tw = time.time()
        with ThreadPoolExecutor(32) as ex:
            warm_bytes = sum(ex.map(_touch, paths, chunksize=64))
        log(f"{m}: warmed {n} source files, {warm_bytes / 1e9:.2f} GB in {time.time() - tw:.0f} s")
    t0 = time.time()
    if args.fmt in ("omniio", "omniio_bytes"):
        from omniio.blob.blob import Blob
        kw = {"image": {"target_format": "jpeg"}, "text": {"is_path": True, "compression_level": 3}, "video": {}}[m]
        if args.fmt == "omniio_bytes":                                  # fast mode: the bytes as they are, index columns only
            kw = {"fast_mode": True}
        blob = Blob(archive_dir=out, modality=m, max_bin_size=320 * 1024 * 1024)
        blob.append(paths, ids=[key_of(i) for i in ids], num_workers=args.workers, progress=False, **kw)
        nbytes = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "blob_*.bin")))
    elif args.fmt == "lance":
        import lance
        import pyarrow as pa
        # single writer (Lance has no parallel writer): `workers` threads read the files
        pool = ThreadPoolExecutor(max(1, args.workers))

        def rd(p):
            with open(p, "rb") as fh:
                return fh.read()
        schema = pa.schema([("id", pa.string()), ("label", pa.int64()), ("data", pa.large_binary())])
        labels = man["label"][:n]

        def batches():
            for a in range(0, n, 4096):
                yield pa.record_batch([pa.array([key_of(i) for i in ids[a:a + 4096]]), pa.array(labels[a:a + 4096], pa.int64()),
                                       pa.array(list(pool.map(rd, paths[a:a + 4096])), pa.large_binary())], schema=schema)
        lance.write_dataset(pa.RecordBatchReader.from_batches(schema, batches()), out, mode="create", max_rows_per_file=1 << 20)
        pool.shutdown(); nbytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    elif args.fmt == "hdf5":
        import h5py
        pool = ThreadPoolExecutor(max(1, args.workers))

        def rd(p):
            with open(p, "rb") as fh:
                return np.frombuffer(fh.read(), dtype=np.uint8)
        with h5py.File(os.path.join(out, "data.h5"), "w") as h:
            d = h.create_dataset("data", (n,), dtype=h5py.vlen_dtype(np.uint8))
            h.create_dataset("id", data=np.array([key_of(i) for i in ids], dtype=h5py.string_dtype()))
            h.create_dataset("label", data=np.asarray(man["label"][:n], dtype=np.int64))
            for a in range(0, n, 4096):
                d[a:a + 4096] = list(pool.map(rd, paths[a:a + 4096]))
        pool.shutdown(); nbytes = os.path.getsize(os.path.join(out, "data.h5"))
    elif args.fmt == "deeplake":
        import deeplake
        pool = ThreadPoolExecutor(max(1, args.workers))

        def rd(p):
            with open(p, "rb") as fh:
                return fh.read()
        ds = deeplake.create(out)                                   # Deep Lake 4: one appending writer; the workers read the files
        ds.add_column("id", deeplake.types.Text()); ds.add_column("label", deeplake.types.Int64()); ds.add_column("data", deeplake.types.Bytes())
        labels = [int(x) for x in man["label"][:n]]
        for a in range(0, n, 4096):
            ds.append({"id": [key_of(i) for i in ids[a:a + 4096]], "label": labels[a:a + 4096], "data": list(pool.map(rd, paths[a:a + 4096]))})
        ds.commit(); pool.shutdown()
        nbytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    elif args.fmt == "mds":
        # MosaicML Streaming: an MDSWriter is sequential, so each worker writes its own shard set into a
        # sub-directory and merge_index joins them into one index.json (the documented parallel recipe)
        W = max(1, args.workers); per = -(-n // W)
        jobs = [(paths[k * per:(k + 1) * per], ids[k * per:(k + 1) * per], list(man["label"][k * per:(k + 1) * per]), os.path.join(out, f"part{k:03d}")) for k in range(W) if paths[k * per:(k + 1) * per]]
        if W <= 1:
            for j in jobs:
                _mds_part(j)
        else:
            with ProcessPoolExecutor(W) as ex:
                list(ex.map(_mds_part, jobs, chunksize=1))
        from streaming.base.util import merge_index
        merge_index(out, keep_local=True)
        nbytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    else:
        ext = EXT[m]; jobs = []
        sh = SHARD[m]
        for s in range(0, n, sh):
            jobs.append((paths[s:s + sh], ids[s:s + sh], os.path.join(out, f"shard-{s // sh:06d}.tar"), ext))
        if args.workers <= 1:
            for j in jobs:
                _wds_shard(j)
        else:
            with ProcessPoolExecutor(args.workers) as ex:
                list(ex.map(_wds_shard, jobs, chunksize=1))            # one shard per task: chunksize 4 capped the parallelism at 7 for 27 shards
        nbytes = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "*.tar")))
    sec = time.time() - t0
    open(os.path.join(out, "DONE"), "w").write("ok\n")
    res = {"modality": m, "fmt": args.fmt, "tag": tag, "n": n, "workers": args.workers, "seconds": sec, "items_per_s": n / sec, "MB_per_s": nbytes / 1e6 / sec, "bytes": nbytes, "node": os.uname().nodename}
    write_result(os.path.join(RESULTS, m, f"ingest_{args.fmt}_{tag}.json"), res); log(f"{m} {args.fmt} {tag}: {n} items in {sec:.0f} s = {n / sec:.0f} items/s, {nbytes / 1e6 / sec:.0f} MB/s")


def _mds_part(job):
    """one worker's MDS shard set: native bytes per sample, 64 MB shards, no hashing / compression"""
    paths, ids, labels, out = job
    from streaming import MDSWriter
    with MDSWriter(out=out, columns={"id": "str", "label": "int64", "data": "bytes"}, compression=None, hashes=[], size_limit=1 << 26) as w:
        for p, i, lab in zip(paths, ids, labels):
            with open(p, "rb") as fh:
                w.write({"id": key_of(i), "label": int(lab), "data": fh.read()})
    return len(paths)


class DeeplakeStore:
    """rows of the Deep Lake dataset by position: bytes column (the native payload)"""

    def __init__(self, m):
        import deeplake
        self.ds = deeplake.open(P(m, "deeplake", "full")); self.col = self.ds["data"]

    def read(self, i):
        return self.col[int(i)]

    def read_many(self, idxs):
        return self.col[[int(i) for i in idxs]]


class MdsStore:
    """samples of the MDS shard set by position (StreamingDataset is also a map-style dataset)"""

    def __init__(self, m, **kw):
        from streaming import StreamingDataset
        self.ds = StreamingDataset(local=P(m, "mds", "full"), remote=None, shuffle=False, batch_size=1, **kw)

    def read(self, i):
        return self.ds[int(i)]["data"]

    def __len__(self):
        return len(self.ds)


# ───────────────────────────── readers ─────────────────────────────
class _Paths:
    """1.3 M path strings as (unique paths, int32 index): a few MB to pickle instead of ~100 MB."""

    def __init__(self, paths):
        self.uniq, inv = np.unique(np.asarray(paths), return_inverse=True); self.inv = inv.astype(np.int32); self.uniq = self.uniq.tolist()

    def __getitem__(self, i):
        return self.uniq[self.inv[i]]

    def __len__(self):
        return len(self.inv)


def omniio_table(m):
    import pyarrow.parquet as pq
    d = P(m, "omniio", "full")
    t = pq.read_table(os.path.join(d, "metadata.parquet"), columns=["id", "path", "start_byte", "end_byte"])
    return {"path": _Paths(t["path"].to_pylist()), "off": t["start_byte"].to_numpy(), "size": (t["end_byte"].to_numpy() - t["start_byte"].to_numpy()), "id": t["id"].to_pylist()}


def wds_index(m):
    """{key: (tar, offset, size)} over the shard set, built once from the tar headers (cached)."""
    d = P(m, "wds", "full")
    cache = os.path.join(d, "member_index.json")
    if os.path.exists(cache):
        return json.load(open(cache)), d
    idx = {}
    parts = sorted(glob.glob(os.path.join(d, "*.tar.idx.json")))
    if parts:                                                     # per-shard indices left by the writer
        for f in parts:
            idx.update(json.load(open(f)))
        json.dump(idx, open(cache, "w")); return idx, d
    for tar in sorted(glob.glob(os.path.join(d, "*.tar"))):
        with tarfile.open(tar) as tf:
            for mem in tf:
                if mem.isfile():
                    idx[mem.name] = (tar, mem.offset_data, mem.size)
    json.dump(idx, open(cache, "w")); return idx, d


class LanceStore:
    """Rows of the Lance dataset by position (take), one call per batch where possible."""

    def __init__(self, m):
        import lance
        self.ds = lance.dataset(P(m, "lance", "full"))

    def read(self, i):
        return self.ds.take([int(i)], columns=["data"]).column("data")[0].as_py()

    def read_many(self, idxs):
        t = self.ds.take([int(i) for i in idxs], columns=["data"]).column("data")
        return [t[k].as_py() for k in range(len(idxs))]


class H5Store:
    def __init__(self, m):
        import h5py
        self.f = h5py.File(os.path.join(P(m, "hdf5", "full"), "data.h5"), "r"); self.d = self.f["data"]

    def read(self, i):
        return self.d[int(i)].tobytes()


def decode(m, raw):
    if m == "image":
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).shape[0]
    if m == "text":
        return len(raw.decode("utf-8", "replace"))
    import av
    with av.open(io.BytesIO(raw)) as c:
        n = 0
        for fr in c.decode(video=0):
            n += 1
            if n >= 16:
                break
    return n


def decode_omniio(m, path, off, size):
    """The modality reader of Omni-IO (text is zstd inside the archive)."""
    if m == "text":
        from omniio.modalities.text.read import text_read_local
        return len(text_read_local(path, int(off), int(size)).text)
    if m == "image":
        from omniio.interface import image_read
        return image_read(path, int(off), int(size)).array.shape[0]
    from omniio.modalities.video.read import video_read_local
    return video_read_local(path, int(off), int(size), start_frame=0, end_frame=16).video_array.shape[0]


def _rchar():
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("rchar"):
                return int(line.split()[1])
    return 0


def read_whole(path):
    from omniio.blob.handles import pread_all
    fd = os.open(path, os.O_RDONLY)
    try:
        return pread_all(fd, 1 << 30, 0)
    finally:
        os.close(fd)


def _want(args, name):
    return not getattr(args, "readers", None) or name in args.readers


def cmd_access(args):
    m = args.modality; man = load_manifest(m); tab = omniio_table(m); idx, wdir = wds_index(m)
    from omniio.blob.handles import archive_fd, pread_all
    n = len(tab["id"]); rng = random.Random(args.seed + args.rep); sample = [rng.randrange(n) for _ in range(args.n_reads)]
    raw_of = {key_of(i): p for i, p in zip(man["ids"], man["raw"])}
    key_ext = EXT[m]

    def read_omniio(i):
        return pread_all(archive_fd(tab["path"][i]), int(tab["size"][i]), int(tab["off"][i]))

    def read_raw(i):
        fd = os.open(raw_of[tab["id"][i]], os.O_RDONLY)                 # open + read the whole file, no buffering
        try:
            return pread_all(fd, 1 << 30, 0)
        finally:
            os.close(fd)

    def read_wds(i):
        # same reader engineering as Omni-IO (cached descriptor, one pread): a naive open/seek/read
        # pulls st_blksize = 4 MB per member on Lustre (measured 3.7 MB per 18 KB JPEG)
        tar, off, size = idx[f"{tab['id'][i]}.{key_ext}"]
        return pread_all(archive_fd(tar), size, off)

    readers = [("omniio", read_omniio), ("raw", read_raw), ("wds_member", read_wds)]
    if os.path.exists(P(m, "lance", "full")):
        ls = LanceStore(m); readers.append(("lance", ls.read))
    if os.path.exists(os.path.join(P(m, "hdf5", "full"), "data.h5")):
        hs = H5Store(m); readers.append(("hdf5", hs.read))
    if os.path.exists(os.path.join(P(m, "deeplake", "full"), "dataset_meta.json")) or glob.glob(os.path.join(P(m, "deeplake", "full"), "*")):
        if os.path.exists(P(m, "deeplake", "full") + "/DONE"):
            dl = DeeplakeStore(m); readers.append(("deeplake", dl.read))
    if os.path.exists(os.path.join(P(m, "mds", "full"), "index.json")):
        ms = MdsStore(m); readers.append(("mds", ms.read))
    if args.readers:
        readers = [(nm, fn) for nm, fn in readers if nm in args.readers]
    res = {"modality": m, "n_reads": args.n_reads, "node": os.uname().nodename, "rep": args.rep, "passes": {}}
    for pas in ("cold", "warm"):
        res["passes"][pas] = {}
        for name, fn in readers:
            lat, byt = [], 0
            for i in sample:
                b0 = _rchar(); t0 = time.perf_counter(); raw = fn(i); lat.append(time.perf_counter() - t0); byt += _rchar() - b0
            lat = np.array(lat) * 1e3
            res["passes"][pas][name] = {"median_ms": float(np.median(lat)), "p90_ms": float(np.percentile(lat, 90)), "mean_ms": float(lat.mean()),
                                        "reads_per_s_1thread": float(1e3 / lat.mean()), "KB_per_read": byt / 1e3 / len(sample)}
            log(f"{m} {pas} {name}: median {np.median(lat):.3f} ms p90 {np.percentile(lat, 90):.2f} ms")
    # decode cost (warm) for reference
    t0 = time.perf_counter()
    for i in sample[:200]:
        decode(m, read_omniio(i))
    res["decode_ms_warm"] = (time.perf_counter() - t0) / 200 * 1e3
    write_result(os.path.join(RESULTS, m, f"access_{args.tag or os.uname().nodename}.json"), res)


# ───────────────────────────── DataLoader throughput ─────────────────────────────
class _OmniIODS:
    """One pread of the entry, then the same decode function as the other formats (text entries
    are zstd in the archive, so that path decompresses first)."""

    def __init__(self, m, tab):
        self.m, self.tab = m, tab

    def __len__(self):
        return len(self.tab["id"])

    def __getitem__(self, i):
        from omniio.blob.prefetch import local_fetch
        raw = local_fetch(self.tab["path"][i], int(self.tab["off"][i]), int(self.tab["size"][i]))
        if self.m == "text":
            import zstandard
            raw = zstandard.ZstdDecompressor().decompress(raw) if raw[:4] == b"\x28\xb5\x2f\xfd" else raw
        return decode(self.m, raw)


class _RawDS:
    def __init__(self, m, paths):
        self.m, self.paths = m, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as fh:
            return decode(self.m, fh.read())


class _PrefetchDS(_OmniIODS):
    """Batch-level reads: the batch's byte ranges are sorted by (file, offset) and read with a
    pool of threads, then decoded; this is the 'coalesced / concurrent random access' variant."""

    def __init__(self, m, tab, threads=16):
        super().__init__(m, tab); self.pool = None; self.threads = threads

    def __getitems__(self, idxs):
        from omniio.blob.handles import archive_fd, pread_all
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.threads)
        order = sorted(idxs, key=lambda i: (self.tab["path"][i], int(self.tab["off"][i])))
        raws = list(self.pool.map(lambda i: pread_all(archive_fd(self.tab["path"][i]), int(self.tab["size"][i]), int(self.tab["off"][i])), order))
        pos = {i: k for k, i in enumerate(order)}
        if self.m == "text":
            import zstandard
            dz = zstandard.ZstdDecompressor(); raws = [dz.decompress(r) if r[:4] == b"\x28\xb5\x2f\xfd" else r for r in raws]
        return [decode(self.m, raws[pos[i]]) for i in idxs]


def _LookaheadIterDS(m, tab, depth=8, seed=0):
    """Random order with `depth` reads in flight per worker: each worker takes its share of a
    random permutation and streams it through omniio.blob.prefetch.Lookahead (the sampler's
    order is known ahead, so the reads can be issued ahead of the decode). Built as an
    IterableDataset subclass at call time so the module does not import torch."""
    import torch
    return _LookaheadIter(m, tab, depth, seed, torch.utils.data.IterableDataset)


def _lookahead_iter(self):
    return _LookaheadBody.run(self)


class _LookaheadBody:
    """the __iter__ body, shared by the dynamically built IterableDataset"""

    @staticmethod
    def run(self):
        import torch
        from omniio.blob.prefetch import Lookahead, local_fetch
        info = torch.utils.data.get_worker_info(); wid, nw = (info.id, info.num_workers) if info else (0, 1)
        perm = np.random.default_rng(self.seed + (info.seed % 1000 if info else 0)).permutation(len(self.tab["id"]))[wid::nw]
        reqs = ((self.tab["path"][i], int(self.tab["off"][i]), int(self.tab["size"][i])) for i in perm)
        la = Lookahead(local_fetch, depth=self.depth)
        dz = None
        if self.m == "text":
            import zstandard
            dz = zstandard.ZstdDecompressor()
        try:
            for raw in la(reqs):
                if dz is not None and raw[:4] == b"\x28\xb5\x2f\xfd":
                    raw = dz.decompress(raw)
                yield decode(self.m, raw)
        finally:
            la.close()


def _LookaheadIter(m, tab, depth, seed, base):
    cls = type("_LookaheadIterDS", (base,), {"__iter__": _lookahead_iter})
    obj = cls(); obj.m, obj.tab, obj.depth, obj.seed = m, tab, depth, seed
    return obj


class _DeeplakeDS:
    """Deep Lake's own PyTorch path: map-style rows of ds.pytorch(), opened in the worker"""

    def __init__(self, m, n):
        self.m, self.n, self.store = m, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.store is None:
            self.store = DeeplakeStore(self.m)
        return decode(self.m, self.store.read(i))

    def __getitems__(self, idxs):                                  # one column take per batch (Deep Lake fetches the rows together)
        if self.store is None:
            self.store = DeeplakeStore(self.m)
        return [decode(self.m, r) for r in self.store.read_many(idxs)]


class _MdsRandomDS:
    """StreamingDataset as a map-style dataset (it supports __getitem__): true random access. The
    StreamingDataset is built once in the parent (its shared-memory registry allows one instance
    per local directory) and inherited by the forked workers."""

    def __init__(self, m, store):
        self.m, self.store = m, store

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        return decode(self.m, self.store.read(i))


def _mds_shuffled_iter(self):
    """StreamingDataset's native path: shard-aware shuffle (py1e), the dataset built once in the
    parent and iterated inside each worker (it partitions the samples over workers itself)"""
    for s in self.ds:
        yield decode(self.m, s["data"])


def _MdsShuffledIter(m, batch):
    """an IterableDataset subclass built at call time (torch imported lazily); StreamingDataset is
    an IterableDataset itself, but the decode has to happen inside the worker, hence the wrapper"""
    import torch
    from streaming import StreamingDataset
    cls = type("_MdsShuffledIter", (torch.utils.data.IterableDataset,), {"__iter__": _mds_shuffled_iter})
    obj = cls(); obj.m = m
    obj.ds = StreamingDataset(local=P(m, "mds", "full"), remote=None, shuffle=True, shuffle_algo="py1e", shuffle_seed=0, batch_size=batch)
    return obj


class _LanceDS:
    def __init__(self, m, n):
        self.m, self.n, self.store = m, n, None

    def __len__(self):
        return self.n

    def _s(self):
        if self.store is None:
            self.store = LanceStore(self.m)
        return self.store

    def __getitem__(self, i):
        return decode(self.m, self._s().read(i))

    def __getitems__(self, idxs):                                 # one take per batch
        return [decode(self.m, r) for r in self._s().read_many(idxs)]


class _H5DS:
    def __init__(self, m, n):
        self.m, self.n, self.store = m, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.store is None:
            self.store = H5Store(self.m)                          # opened in the worker
        return decode(self.m, self.store.read(i))


class WdsDecode:
    def __init__(self, m):
        self.m, self.ext = m, EXT[m]

    def __call__(self, sample):
        return decode(self.m, sample[self.ext])


def cmd_loader(args):
    import torch
    from torch.utils.data import DataLoader
    import webdataset as wds
    m = args.modality; man = load_manifest(m); tab = omniio_table(m); _, wdir = wds_index(m)
    raw_of = dict(zip([key_of(i) for i in man["ids"]], man["raw"])); raw_paths = [raw_of[k] for k in tab["id"]]; del raw_of
    shards = sorted(glob.glob(os.path.join(wdir, "*.tar")))
    res = {"modality": m, "batch": args.batch, "steps": args.steps, "node": os.uname().nodename, "runs": []}
    mp = {"multiprocessing_context": "fork", "persistent_workers": False, "prefetch_factor": 4}     # datasets inherited, not pickled per worker

    def run(name, dl):
        if args.loaders and name not in args.loaders:
            return
        t0 = time.perf_counter(); seen = 0; first = None; t1 = None
        try:
            for k, b in enumerate(dl):
                if first is None:
                    first = time.perf_counter() - t0; t1 = time.perf_counter(); continue      # first batch: worker start-up / buffer fill, reported apart
                seen += len(b) if hasattr(b, "__len__") else args.batch
                if k + 1 >= args.steps:
                    break
        except Exception as e:                                           # a baseline that cannot run in this setting is recorded, not fatal
            log(f"{m} {name} w={w}: ERROR {type(e).__name__}: {str(e)[:200]}")
            res["runs"].append({"loader": name, "workers": w, "error": f"{type(e).__name__}: {str(e)[:300]}"}); return
        sec = time.perf_counter() - t1
        row = {"loader": name, "workers": w, "items_per_s": seen / sec, "first_batch_s": first, "seconds_after_first": sec, "items": seen}
        res["runs"].append(row); log(f"{m} {name} w={w}: {seen / sec:.0f} items/s (first batch {first:.1f} s)")
        write_result(os.path.join(RESULTS, m, f"loader_{args.tag or os.uname().nodename}.json"), res)

    g = torch.Generator().manual_seed(args.rep); res["rep"] = args.rep
    mds_store = mds_iter = None
    for w in args.workers:
        wk = {"num_workers": w, **mp} if w > 0 else {"num_workers": 0}
        run("omniio_random", DataLoader(_OmniIODS(m, tab), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
        run("omniio_random_prefetch", DataLoader(_PrefetchDS(m, tab), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
        run("omniio_random_lookahead8", DataLoader(_LookaheadIterDS(m, tab, depth=8, seed=args.rep), batch_size=args.batch, collate_fn=list, **wk))
        run("raw_random", DataLoader(_RawDS(m, raw_paths), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
        if os.path.exists(P(m, "lance", "full")):
            run("lance_random_batched", DataLoader(_LanceDS(m, len(tab["id"])), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
        if os.path.exists(os.path.join(P(m, "hdf5", "full"), "data.h5")):
            run("hdf5_random", DataLoader(_H5DS(m, len(tab["id"])), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
        if os.path.exists(P(m, "deeplake", "full") + "/DONE"):
            run("deeplake_random_batched", DataLoader(_DeeplakeDS(m, len(tab["id"])), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
        if os.path.exists(os.path.join(P(m, "mds", "full"), "index.json")):
            if mds_store is None:
                mds_store = MdsStore(m)                              # one StreamingDataset per local directory per process
            run("mds_random", DataLoader(_MdsRandomDS(m, mds_store), batch_size=args.batch, shuffle=True, generator=g, collate_fn=list, **wk))
            if mds_iter is None:
                mds_iter = _MdsShuffledIter(m, args.batch)
            run("mds_shuffled", DataLoader(mds_iter, batch_size=args.batch, collate_fn=list, **wk))
        pipe = wds.WebDataset(shards, shardshuffle=100, nodesplitter=None, empty_check=False).shuffle(1000).map(WdsDecode(m)).batched(args.batch, collation_fn=list)
        run("wds_stream_buf1k", DataLoader(pipe, batch_size=None, **wk))


# ───────────────────────────── sequential streaming ─────────────────────────────
def cmd_stream(args):
    m = args.modality; tab = omniio_table(m); _, wdir = wds_index(m); man = load_manifest(m)
    from omniio.blob.handles import archive_fd, pread_all
    args.n = min(args.n, len(tab["id"]))
    res = {"modality": m, "n_items": args.n, "node": os.uname().nodename, "runs": {}}
    import webdataset as wds
    decode(m, pread_all(archive_fd(tab["path"][0]), int(tab["size"][0]), int(tab["off"][0])))      # decoder warm-up (imports, codec init)
    nrep = max(1, len(tab["id"]) // args.n); off = (args.rep % nrep) * args.n         # repeat k reads the k-th disjoint slice (cold on a fresh node)
    res["rep"], res["offset"] = args.rep, off
    for dec in (False, True):
        if True:
            # Omni-IO: storage order, one archive after another
            order = np.lexsort((tab["off"], np.array(tab["path"])))[off: off + args.n]
            if _want(args, "omniio"):
                t0 = time.perf_counter(); byt = 0
                for i in order:
                    raw = pread_all(archive_fd(tab["path"][i]), int(tab["size"][i]), int(tab["off"][i])); byt += len(raw)
                    if dec:
                        decode(m, raw)
                sec = time.perf_counter() - t0; res["runs"][f"omniio_sequential{'_decode' if dec else ''}"] = {"items_per_s": len(order) / sec, "MB_per_s": byt / 1e6 / sec}
            # WebDataset: tar streaming
            if _want(args, "wds"):
                shards = sorted(glob.glob(os.path.join(wdir, "*.tar")))[off // SHARD[m]:]; t0 = time.perf_counter(); n = 0; byt = 0
                for s in wds.WebDataset(shards, shardshuffle=False, nodesplitter=None, empty_check=False):
                    raw = s[EXT[m]]; byt += len(raw); n += 1
                    if dec:
                        decode(m, raw)
                    if n >= args.n:
                        break
                sec = time.perf_counter() - t0; res["runs"][f"wds_stream{'_decode' if dec else ''}"] = {"items_per_s": n / sec, "MB_per_s": byt / 1e6 / sec}
            # raw files in directory order
            if _want(args, "raw"):
                paths = man["raw"][off: off + args.n]; t0 = time.perf_counter(); byt = 0
                for p in paths:
                    with open(p, "rb") as fh:
                        raw = fh.read()
                    byt += len(raw)
                    if dec:
                        decode(m, raw)
                sec = time.perf_counter() - t0; res["runs"][f"raw_sequential{'_decode' if dec else ''}"] = {"items_per_s": len(paths) / sec, "MB_per_s": byt / 1e6 / sec}
            if os.path.exists(P(m, "lance", "full")) and _want(args, "lance"):
                import lance
                ds = lance.dataset(P(m, "lance", "full")); t0 = time.perf_counter(); n = 0; byt = 0
                for b in ds.scanner(columns=["data"], batch_size=1024, offset=off, limit=args.n).to_batches():
                    for raw in b.column("data").to_pylist():
                        byt += len(raw); n += 1
                        if dec:
                            decode(m, raw)
                    if n >= args.n:
                        break
                sec = time.perf_counter() - t0; res["runs"][f"lance_scan{'_decode' if dec else ''}"] = {"items_per_s": n / sec, "MB_per_s": byt / 1e6 / sec}
            if os.path.exists(os.path.join(P(m, "hdf5", "full"), "data.h5")) and _want(args, "hdf5"):
                hs = H5Store(m); t0 = time.perf_counter(); byt = 0; nh = min(args.n, len(hs.d) - off)
                for i in range(off, off + nh):
                    raw = hs.read(i); byt += len(raw)
                    if dec:
                        decode(m, raw)
                sec = time.perf_counter() - t0; res["runs"][f"hdf5_sequential{'_decode' if dec else ''}"] = {"items_per_s": nh / sec, "MB_per_s": byt / 1e6 / sec}
            if os.path.exists(P(m, "deeplake", "full") + "/DONE") and _want(args, "deeplake"):
                dl = DeeplakeStore(m); t0 = time.perf_counter(); byt = 0; nd = 0
                for a in range(off, off + args.n, 1024):                  # column slices: Deep Lake's sequential path
                    for raw in dl.col[a:min(a + 1024, off + args.n)]:
                        byt += len(raw); nd += 1
                        if dec:
                            decode(m, raw)
                sec = time.perf_counter() - t0; res["runs"][f"deeplake_sequential{'_decode' if dec else ''}"] = {"items_per_s": nd / sec, "MB_per_s": byt / 1e6 / sec}
            if os.path.exists(os.path.join(P(m, "mds", "full"), "index.json")) and _want(args, "mds"):
                # StreamingDataset's own sequential path: iterate the dataset in sample order (its shard
                # reader streams a shard at a time); iteration cannot start at an offset, so every repeat
                # reads the first n samples, cold on its fresh node
                from streaming import StreamingDataset
                sds = StreamingDataset(local=P(m, "mds", "full"), remote=None, shuffle=False, batch_size=1)
                t0 = time.perf_counter(); byt = 0; nd = 0
                for smp in sds:
                    raw = smp["data"]; byt += len(raw); nd += 1
                    if dec:
                        decode(m, raw)
                    if nd >= args.n:
                        break
                sec = time.perf_counter() - t0; res["runs"][f"mds_sequential{'_decode' if dec else ''}"] = {"items_per_s": nd / sec, "MB_per_s": byt / 1e6 / sec}
                del sds
        log(f"{m} stream decode={dec}: " + ", ".join(f"{k} {v['items_per_s']:.0f}/s {v['MB_per_s']:.0f} MB/s" for k, v in res["runs"].items() if k.endswith("_decode") == dec))
    write_result(os.path.join(RESULTS, m, f"stream_{args.tag or os.uname().nodename}.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["materialize", "ingest", "access", "loader", "stream"])
    ap.add_argument("--modality", required=True, choices=["image", "text", "video"])
    ap.add_argument("--fmt", choices=["omniio", "omniio_bytes", "wds", "lance", "hdf5", "deeplake", "mds"], default="omniio")
    ap.add_argument("--workers", type=int, nargs="*", default=[8])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_reads", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_warm", action="store_true", help="ingest: do not pre-read the source files")
    ap.add_argument("--loaders", nargs="*", default=None, help="loader: run only these loader names")
    ap.add_argument("--readers", nargs="*", default=None, help="access / stream: run only these readers (omniio, raw, wds, lance, hdf5, deeplake, mds)")
    ap.add_argument("--rep", type=int, default=1, help="repeat index: access/loader use another sample or seed, stream reads the rep-th disjoint slice")
    args = ap.parse_args()
    os.makedirs(os.path.join(RESULTS, args.modality), exist_ok=True)
    if args.cmd in ("materialize", "ingest"):
        args.workers = args.workers[0]
    if args.cmd == "stream" and args.n is None:
        args.n = 20000
    {"materialize": cmd_materialize, "ingest": cmd_ingest, "access": cmd_access, "loader": cmd_loader, "stream": cmd_stream}[args.cmd](args)


if __name__ == "__main__":
    main()

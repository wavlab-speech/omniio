"""Read a corpus packed by :func:`omniio.lhotse.from_lhotse`.

Needs neither lhotse nor the original files: everything comes from the two
archives, locally or over HTTP.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import zstandard as zstd

from omniio.definitions import AudioRead
from omniio.interface import audio_read
from omniio.remote import load as remote_load

# Neighbouring payloads closer than this are fetched with one read.
_COALESCE_GAP = 64 * 1024


def _is_url(path: str) -> bool:
    return str(path).startswith(("http://", "https://"))


def _load_index(archive: str) -> pa.Table:
    """``metadata.parquet`` of an archive dir or URL, with ``path`` pointing at its bins."""
    archive = str(archive).rstrip("/")
    if _is_url(archive):
        return remote_load(f"{archive}/metadata.parquet", archive)
    return pq.read_table(os.path.join(archive, "metadata.parquet"))


def _read_range(path: str, start: int, end: int) -> bytes:
    """Bytes ``[start, end)`` of a local file or URL."""
    if _is_url(path):
        resp = requests.get(path, headers={"Range": f"bytes={start}-{end - 1}"})
        resp.raise_for_status()
        return resp.content[: end - start]
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def windows(
    duration: float, length: float, hop: Optional[float] = None
) -> List[Tuple[float, float]]:
    """Fixed ``(start, end)`` windows over ``[0, duration)``, the last one shortened.

    Matches lhotse's ``cut_into_windows``: ``hop`` defaults to ``length``.
    """
    hop = length if hop is None else hop
    out, start = [], 0.0
    while start < duration:
        out.append((start, min(start + length, duration)))
        start = round(start + hop, 8)
    return out


class SegmentReader:
    """Audio and supervisions of a packed corpus.

    Args:
        audio:        The ``audio`` archive: a directory, or its URL.
        supervisions: A ``supervisions_<name>`` archive: a directory, or its URL.
                      Optional; without it only :meth:`audio` is available.
        num_threads:  Remote sources read in parallel by :meth:`audio`.  Local
                      sources are read one after the other, which measured
                      faster than any thread pool (16-channel AMI MDM segments:
                      6 ms sequential, 10 ms pooled).
    """

    def __init__(
        self,
        audio: str,
        supervisions: Optional[str] = None,
        num_threads: int = 8,
    ):
        self.audio_index = _load_index(audio)
        self.sup_index = _load_index(supervisions) if supervisions is not None else None
        self._num_threads = num_threads
        self._pool: Optional[ThreadPoolExecutor] = None

        self._audio_rows: Dict[str, dict] = {}
        self._rec_sources: Dict[str, List[str]] = {}
        cols = ["id", "path", "start_byte", "end_byte", "sample_rate", "recording_id"]
        for row in self.audio_index.select(cols).to_pylist():
            self._audio_rows[row["id"]] = row
            self._rec_sources.setdefault(row["recording_id"], []).append(row["id"])

        self._sup_pos: Dict[str, int] = {}
        if self.sup_index is not None:
            ids = self.sup_index.column("id").to_pylist()
            self._sup_pos = {sid: i for i, sid in enumerate(ids)}

    # ------------------------------------------------------------------ #
    # Index
    # ------------------------------------------------------------------ #

    @property
    def recording_ids(self) -> List[str]:
        return list(self._rec_sources)

    def recording_duration(self, recording_id: str) -> float:
        """Shortest source of a recording, in seconds: the span :meth:`audio` returns
        for every channel."""
        mask = pc.equal(self.audio_index.column("recording_id"), recording_id)
        return float(pc.min(self.audio_index.filter(mask).column("duration")).as_py())

    def supervisions(
        self,
        recording_id: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        speaker: Optional[str] = None,
    ) -> pa.Table:
        """Index rows of the supervisions that match, by ``(recording_id, start)``.

        ``start``/``end`` keep the segments that overlap ``[start, end)`` by
        any amount: a segment ending 20 ms into the window is kept (lhotse's
        cut windows may drop it), one ending exactly at ``start`` is not.
        Segments are returned whole, not clipped to the window.  Reads no
        archive bytes.
        """
        table = self._require_sups()
        mask = None

        def _and(m):
            return m if mask is None else pc.and_(mask, m)

        if recording_id is not None:
            mask = _and(pc.equal(table.column("recording_id"), recording_id))
        if speaker is not None:
            mask = _and(pc.equal(table.column("speaker"), speaker))
        if end is not None:
            mask = _and(pc.less(table.column("start"), end))
        if start is not None:
            seg_end = pc.add(table.column("start"), table.column("duration"))
            mask = _and(pc.greater(seg_end, start))
        return table if mask is None else table.filter(mask)

    # ------------------------------------------------------------------ #
    # Payloads
    # ------------------------------------------------------------------ #

    def payloads(self, rows: Union[pa.Table, Sequence[str]]) -> List[dict]:
        """Text, alignment and ``custom`` of supervisions, in the order given.

        ``rows`` is a table from :meth:`supervisions` or a list of ids.  Payloads
        that lie close together in a bin -- the segments of one time window, as
        written by the converter -- are fetched with a single read.
        """
        table = self._require_sups()
        if not isinstance(rows, pa.Table):
            rows = table.take([self._sup_pos[i] for i in rows])
        spans = rows.select(["path", "start_byte", "end_byte"]).to_pylist()

        order = sorted(range(len(spans)), key=lambda i: (spans[i]["path"], spans[i]["start_byte"]))
        out: List[Optional[dict]] = [None] * len(spans)
        dctx = zstd.ZstdDecompressor()
        run: List[int] = []

        def flush():
            if not run:
                return
            path = spans[run[0]]["path"]
            lo = spans[run[0]]["start_byte"]
            hi = max(spans[i]["end_byte"] for i in run)
            data = _read_range(path, lo, hi)
            for i in run:
                s = spans[i]
                raw = data[s["start_byte"] - lo : s["end_byte"] - lo]
                out[i] = json.loads(dctx.decompress(raw).decode("utf-8"))
            run.clear()

        for i in order:
            if run:
                prev = spans[run[-1]]
                if (
                    spans[i]["path"] != prev["path"]
                    or spans[i]["start_byte"] - prev["end_byte"] > _COALESCE_GAP
                ):
                    flush()
            run.append(i)
        flush()
        return out

    # ------------------------------------------------------------------ #
    # Audio
    # ------------------------------------------------------------------ #

    def audio(
        self,
        recording_id: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        audio_ids: Optional[Sequence[str]] = None,
    ) -> AudioRead:
        """``[start, end)`` of a recording as ``(frames, channels)``.

        ``audio_ids`` picks and orders the sources (default: all of the
        recording's, in archive order).  Each source is one ranged read (in
        parallel when remote); channels are concatenated and cut to the
        shortest source.
        """
        ids = list(audio_ids) if audio_ids is not None else self._rec_sources[recording_id]
        rows = [self._audio_rows[i] for i in ids]
        rates = {r["sample_rate"] for r in rows}
        if len(rates) > 1:
            raise ValueError(f"Sources of {recording_id} differ in sample rate: {sorted(rates)}")

        def _one(r):
            return audio_read(
                r["path"], r["start_byte"], r["end_byte"] - r["start_byte"], start, end
            )

        if len(rows) > 1 and _is_url(rows[0]["path"]) and self._num_threads > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self._num_threads)
            reads = list(self._pool.map(_one, rows))
        else:
            reads = [_one(r) for r in rows]
        n = min(r.array.shape[0] for r in reads)
        return AudioRead(
            file_type=reads[0].file_type,
            modality="audio",
            sample_rate=reads[0].sample_rate,
            array=np.concatenate([r.array[:n] for r in reads], axis=1),
        )

    def segment(self, sup_id: str) -> dict:
        """One supervision: its index row, payload, and audio over its ``audio_ids``."""
        table = self._require_sups()
        row = table.slice(self._sup_pos[sup_id], 1)
        info = row.to_pylist()[0]
        info.update(self.payloads(row)[0])
        info["audio"] = self.audio(
            info["recording_id"],
            info["start"],
            info["start"] + info["duration"],
            audio_ids=info["audio_ids"],
        )
        return info

    def _require_sups(self) -> pa.Table:
        if self.sup_index is None:
            raise ValueError("This reader was opened without a supervisions archive.")
        return self.sup_index

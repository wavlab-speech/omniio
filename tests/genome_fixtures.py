"""Test-side helper for seq / track archives: chromosome name -> entry -> interval read.

The library keeps one `metadata.parquet` per archive and the plain `seq_read` / `track_read`
entry points; this is the few lines of index lookup the tests would otherwise repeat."""
from __future__ import annotations

import os
from typing import Optional, Union

import pyarrow.parquet as pq


class GenomeArchive:
    def __init__(self, archive_dir: Union[str, os.PathLike], modality: Optional[str] = None, url: Optional[str] = None):
        self.dir = os.fspath(archive_dir)
        meta = os.path.join(self.dir, "metadata.parquet")
        if url is not None:
            from omniio.remote import load
            self.table = load(meta, url)
        else:
            self.table = pq.read_table(meta)
        fmt = self.table.column("format")[0].as_py() if "format" in self.table.column_names and self.table.num_rows else None
        self.modality = modality or fmt
        assert self.modality in ("seq", "track"), fmt
        names = self.table.column("name").to_pylist()
        self.ids = self.table.column("id").to_pylist()
        self.paths = self.table.column("path").to_pylist()
        self.starts = self.table.column("start_byte").to_numpy()
        self.sizes = self.table.column("end_byte").to_numpy() - self.starts
        self.lengths = dict(zip(names, self.table.column("length").to_pylist()))
        self.rows = {n: i for i, n in enumerate(names)}
        self.bin = self.table.column("bin")[0].as_py() if self.modality == "track" and self.table.num_rows else None
        import omniio.interface as I
        self._read = I.seq_read if self.modality == "seq" else I.track_read

    @property
    def chroms(self) -> list:
        return list(self.rows)

    def read(self, chrom: str, start: Optional[int] = None, end: Optional[int] = None, **kw):
        r = self.rows[chrom]
        return self._read(self.paths[r], int(self.starts[r]), int(self.sizes[r]), start=start, end=end, **kw)

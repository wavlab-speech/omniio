"""A `cell` archive for the single-cell experiments: the index (metadata.parquet), the rows in
storage order, and batched reads through `cell_read_batch` grouped by bin. The feature panel is
the one the preparation script wrote the archive against; a column subset is applied after the
read (the experiments select highly variable genes / peaks)."""
import os

import numpy as np
import pyarrow.parquet as pq

from omniio.interface import cell_read_batch


class CellArchive:
    def __init__(self, archive_dir):
        self.dir = os.fspath(archive_dir)
        self.table = pq.read_table(os.path.join(self.dir, "metadata.parquet"))
        self.ids = np.asarray(self.table.column("id").to_pylist(), dtype=object)
        self.paths = np.asarray(self.table.column("path").to_pylist(), dtype=object)
        self.starts = self.table.column("start_byte").to_numpy()
        self.sizes = self.table.column("end_byte").to_numpy() - self.starts
        self.n_genes = int(self.table.column("n_genes")[0].as_py())
        # storage order: the source file, then the row inside it (the writer's workers interleave them in the index)
        files = np.asarray(self.table.column("source_file").to_pylist(), dtype=object)
        uniq = {f: i for i, f in enumerate(sorted(set(files.tolist())))}
        fi = np.fromiter((uniq[f] for f in files), dtype=np.int64, count=len(files))
        self.order = np.lexsort((self.table.column("source_row").to_numpy(), fi))     # archive row of the k-th cell in storage order
        self.files = list(uniq)

    def __len__(self):
        return self.table.num_rows

    def column(self, name):
        """an index column in storage order"""
        return np.asarray(self.table.column(name).to_pylist(), dtype=object)[self.order]

    def read_batch(self, positions, cols=None, max_gap=64 * 1024):
        """storage positions -> CSR (len(positions), n_genes) in that order, one coalesced read pass
        per bin touched; `cols` keeps those columns (a gene / peak subset)."""
        import scipy.sparse as sp
        rows = self.order[np.asarray(positions, dtype=np.int64)]
        if rows.size == 0:
            return sp.csr_matrix((0, self.n_genes if cols is None else len(cols)))
        paths = self.paths[rows]; parts, where = [], []
        for p in np.unique(paths):
            ks = np.flatnonzero(paths == p); sel = rows[ks]
            parts.append(cell_read_batch(str(p), list(zip(self.starts[sel].tolist(), self.sizes[sel].tolist())), max_gap=max_gap)); where.append(ks)
        X = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
        inv = np.empty(rows.size, dtype=np.int64); inv[np.concatenate(where)] = np.arange(rows.size)
        X = X[inv]
        return X[:, cols] if cols is not None else X

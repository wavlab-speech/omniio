"""Single-cell expression: .h5ad files into an omniio `cell` archive, and back with gene names.

The `cell` modality stores one sparse row per cell over a fixed panel of columns and does
not know what the columns are, the way `discrete` does not know its vocabulary. This
example owns that knowledge:

  * `gene_panel`   -- the column order of the archive: the union of the files' gene ids
                      (first-appearance order), or any list you decide on;
  * `H5adRows`     -- a `CellRows` item: rows [start, stop) of one .h5ad file, remapped by
                      gene id onto the panel, obs columns as metadata; `Blob.append` sends
                      these to its workers and unrolls one entry per cell;
  * `gene_symbols` -- gene id -> symbol from a file's var, for naming columns on read;
  * `select_genes` -- a cell's row restricted to a list of gene ids, in that order.

    python examples/cell_h5ad.py ./atlas lung.h5ad liver.h5ad --obs cell_type donor --workers 8

Needs h5py and anndata (for its element reader of obs / var) on top of `omniio[cell]`.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from omniio.blob.blob import Blob
from omniio.cell.write import CellRows
from omniio.interface import cell_read


# ------------------------------------------------------------------ the file
def _open(path: str):
    import h5py
    return h5py.File(path, "r")


def _frame(f, key: str):
    """obs or var as a DataFrame (anndata's element reader: handles current and legacy layouts)"""
    from anndata.io import read_elem
    return read_elem(f[key])


def file_gene_ids(path: str, gene_id_column: Optional[str] = None) -> np.ndarray:
    """The gene ids of one file's var: a column, or the var index."""
    with _open(path) as f:
        var = _frame(f, "var")
    return (var[gene_id_column].to_numpy() if gene_id_column else var.index.to_numpy()).astype(object)


def n_obs(path: str) -> int:
    with _open(path) as f:
        return len(_frame(f, "obs"))


def gene_panel(paths: Sequence[str], gene_id_column: Optional[str] = None) -> List[str]:
    """Union of the files' gene ids in first-appearance order: the archive's column order."""
    seen: Dict[str, None] = {}
    for p in paths:
        for g in file_gene_ids(p, gene_id_column):
            seen.setdefault(str(g), None)
    return list(seen)


def gene_symbols(path: str, gene_id_column: Optional[str] = None, symbol_column: Optional[str] = None) -> Dict[str, str]:
    """gene id -> symbol (the var index unless `symbol_column` is given)."""
    with _open(path) as f:
        var = _frame(f, "var")
    ids = var[gene_id_column].to_numpy() if gene_id_column else var.index.to_numpy()
    syms = var[symbol_column].to_numpy() if symbol_column else var.index.to_numpy()
    return {str(i): str(s) for i, s in zip(ids, syms)}


def _rows_csr(g, s: int, e: int):
    """rows [s, e) of a matrix stored in the h5ad file as scipy CSR (h5py slices of indptr,
    indices and data: only those rows are read), or of a dense dataset"""
    import scipy.sparse as sp
    enc = g.attrs.get("encoding-type", "")
    if isinstance(enc, bytes):
        enc = enc.decode()
    if enc == "csr_matrix" or (hasattr(g, "keys") and "indptr" in g and enc != "csc_matrix"):
        indptr = g["indptr"][s:e + 1]; lo, hi = int(indptr[0]), int(indptr[-1])
        n_cols = int(g.attrs["shape"][1])
        return sp.csr_matrix((g["data"][lo:hi], g["indices"][lo:hi], indptr - lo), shape=(e - s, n_cols))
    if enc == "csc_matrix":                       # column-major on disk: the whole matrix has to be read; convert once
        from anndata.io import read_elem
        return sp.csr_matrix(read_elem(g))[s:e]
    return sp.csr_matrix(np.asarray(g[s:e]))     # dense dataset


# ------------------------------------------------------------------ writing
class H5adRows(CellRows):
    """Rows [start, stop) of one .h5ad file as `(cell_id, (indices, values, n_cols), metadata)`.

    Columns are remapped by gene id onto `panel` (genes absent from the panel are dropped);
    `obs_columns` are copied into each cell's metadata; ids are `prefix + obs_name`, the
    file stem by default so two files cannot collide. The instance holds only paths and
    numbers, so `Blob.append` can pickle it to a worker, where `__iter__` opens the file and
    reads its rows `chunk` at a time (h5py slices of the CSR arrays: nothing else is loaded).
    """

    def __init__(self, path: str, start: int, stop: int, panel: Sequence[str], layer: str = "X",
                 gene_id_column: Optional[str] = None, obs_columns: Sequence[str] = (), prefix: Optional[str] = None,
                 chunk: int = 2000):
        self.path, self.start, self.stop = os.fspath(path), int(start), int(stop)
        self.panel, self.layer, self.gene_id_column = list(panel), layer, gene_id_column
        self.obs_columns, self.chunk = list(obs_columns), int(chunk)
        self.prefix = prefix if prefix is not None else os.path.splitext(os.path.basename(path))[0] + ":"

    def __iter__(self) -> Iterator:
        pos = {g: i for i, g in enumerate(self.panel)}
        col_map = np.array([pos.get(str(g), -1) for g in file_gene_ids(self.path, self.gene_id_column)], dtype=np.int64)
        n_cols = len(self.panel)
        with _open(self.path) as f:
            obs = _frame(f, "obs").iloc[self.start:self.stop]
            names = obs.index.to_numpy(); cols = {c: obs[c].to_numpy() for c in self.obs_columns}
            g = f["X"] if self.layer == "X" else f["layers"][self.layer]
            for s in range(self.start, self.stop, self.chunk):
                e = min(s + self.chunk, self.stop)
                X = _rows_csr(g, s, e)
                for j in range(e - s):
                    lo, hi = X.indptr[j], X.indptr[j + 1]
                    idx, val = col_map[X.indices[lo:hi]], X.data[lo:hi]
                    keep = idx >= 0
                    meta = {c: _scalar(cols[c][s - self.start + j]) for c in self.obs_columns}
                    meta["source_file"], meta["source_row"] = self.path, s + j
                    yield self.prefix + str(names[s - self.start + j]), (idx[keep], val[keep], n_cols), meta


def _scalar(v):
    return v.item() if hasattr(v, "item") else v


def h5ad_items(paths: Sequence[str], panel: Sequence[str], rows: int = 10_000, **kw) -> List[H5adRows]:
    """Cut each file into `rows`-row items so `Blob.append`'s workers share a file."""
    items = []
    for p in paths:
        n = n_obs(p)
        items += [H5adRows(p, s, min(s + rows, n), panel, **kw) for s in range(0, n, rows)]
    return items


def ingest(archive_dir: str, paths: Sequence[str], panel: Sequence[str], obs_columns: Sequence[str] = (),
           gene_id_column: Optional[str] = None, num_workers: int = 0, rows: int = 10_000, **write_kw) -> Blob:
    blob = Blob(archive_dir, modality="cell")
    items = h5ad_items(paths, panel, rows=rows, gene_id_column=gene_id_column, obs_columns=obs_columns)
    blob.append(items, num_workers=num_workers, **write_kw)
    return blob


# ------------------------------------------------------------------ reading with names
def select_genes(r, panel: Sequence[str], wanted: Sequence[str], dtype=None) -> np.ndarray:
    """A `CellRead` row as a dense vector over `wanted` gene ids (zeros where absent)."""
    pos = {g: i for i, g in enumerate(panel)}
    col_map = np.full(len(panel), -1, dtype=np.int64)
    for k, g in enumerate(wanted):
        if g in pos:
            col_map[pos[g]] = k
    out = np.zeros(len(wanted), dtype=dtype or r.values.dtype)
    m = col_map[r.indices]
    keep = m >= 0
    out[m[keep]] = r.values[keep]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive_dir"); ap.add_argument("h5ad", nargs="+"); ap.add_argument("--obs", nargs="*", default=[])
    ap.add_argument("--gene-id-column", default=None); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--rows", type=int, default=10_000); ap.add_argument("--value-dtype", default="uint16")
    a = ap.parse_args()
    panel = gene_panel(a.h5ad, a.gene_id_column)
    with open(os.path.join(a.archive_dir, "panel.txt"), "w") as f:        # the panel is yours to keep; this is one place
        f.write("\n".join(panel) + "\n")
    blob = ingest(a.archive_dir, a.h5ad, panel, obs_columns=a.obs, gene_id_column=a.gene_id_column,
                  num_workers=a.workers, rows=a.rows, value_dtype=a.value_dtype)
    blob.summary()
    t = blob.get_metadata().to_pylist()[0]
    r = cell_read(t["path"], t["start_byte"], t["end_byte"] - t["start_byte"])
    names = gene_symbols(a.h5ad[0], a.gene_id_column)
    top = r.indices[np.argsort(r.values)[::-1][:5]]
    print(f"{t['id']}: {r.nnz} genes expressed of {r.n_genes}; top: " + ", ".join(names.get(panel[i], panel[i]) for i in top))


if __name__ == "__main__":
    main()

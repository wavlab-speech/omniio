"""Fixtures for the ``cell`` modality tests.

`Rows` is a picklable `CellRows` item over a seeded synthetic count matrix (the library
tests need no file format at all); `entries` indexes an archive by id. The .h5ad writers
below serve the example test (``tests/test_cell_h5ad_example.py``): CSR / CSC / dense
``X``, a named layer, empty cells, a cell with every gene, explicit zeros, counts > 65535,
non-unique ``obs_names``, non-ASCII categories, a different gene order / a subset of genes."""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from omniio.cell.write import CellRows

try:
    import pandas as pd
except ImportError:                      # only the .h5ad fixtures need it
    pd = None


def counts_matrix(seed: int, n_cells: int, n_genes: int, big: bool = False):
    """Deterministic CSR count matrix with explicit zeros, empty rows and a full row."""
    _lazy()
    return make_counts(n_cells, n_genes, np.random.default_rng(seed), big=big)


class Rows(CellRows):
    """Cells [start, stop) of `counts_matrix(seed, n_cells, n_genes)`: ids `prefix + cell_<i>`,
    metadata `source_row` and `group`; `dup` repeats the first id at the end."""

    def __init__(self, seed=0, n_cells=300, n_genes=500, start=0, stop=None, prefix="", big=False, dup=False):
        self.seed, self.n_cells, self.n_genes, self.big, self.dup = seed, n_cells, n_genes, big, dup
        self.start, self.stop, self.prefix = start, n_cells if stop is None else stop, prefix

    def __iter__(self):
        X = counts_matrix(self.seed, self.n_cells, self.n_genes, self.big)
        for i in range(self.start, self.stop):
            yield f"{self.prefix}cell_{i:05d}", X[i], {"source_row": i, "group": "even" if i % 2 == 0 else "odd"}
        if self.dup:
            yield f"{self.prefix}cell_{self.start:05d}", X[self.start], {"source_row": self.start, "group": "dup"}


def entries(blob) -> dict:
    """id -> (path, start_byte, size, row dict) from the archive index."""
    rows = pq.read_table(blob.metadata_file).to_pylist()
    return {r["id"]: (str(blob._get_bin_file_path(r["bin_index"])), r["start_byte"], r["end_byte"] - r["start_byte"], r) for r in rows}

anndata = None
sp = None


def _lazy():
    """scipy for the matrices (the library tests); anndata only where an .h5ad is written."""
    global anndata, sp
    if sp is None:
        import scipy.sparse as _sp
        sp = _sp
    if anndata is None:
        try:
            import anndata as _ad
            anndata = _ad
        except ImportError:
            pass


N_CELLS, N_GENES = 300, 500
CATS = ["T細胞", "β-cell", "NK", "Monocyte/Macrophage", "naïve B"]


def make_obs(n: int, rng, dup_names: bool = False) -> pd.DataFrame:
    names = [f"cell_{i:05d}" for i in range(n)]
    if dup_names:
        names[5] = names[3]
        names[n - 1] = names[0]
    obs = pd.DataFrame(index=pd.Index(names, name="obs_names"))
    obs["cell_type"] = pd.Categorical(rng.choice(CATS, n))
    obs["donor"] = pd.Categorical(rng.choice(["TSP1", "TSP2", "TSP14"], n))
    obs["tissue"] = rng.choice(["lung", "liver", "blood"], n).astype(object)      # plain strings
    obs["plate"] = rng.integers(0, 7, n).astype(np.int64)
    obs["qc_score"] = rng.random(n)
    obs.loc[obs.index[2], "qc_score"] = np.nan
    obs["is_doublet"] = rng.random(n) < 0.1
    obs["n_umi_nullable"] = pd.array(np.where(rng.random(n) < 0.2, None, rng.integers(100, 5000, n)), dtype="Int64")
    return obs


def make_var(n: int, offset: int = 0) -> pd.DataFrame:
    var = pd.DataFrame(index=pd.Index([f"GENE{offset + i}" for i in range(n)], name="var_names"))
    var["gene_ids"] = [f"ENSG{offset + i:011d}" for i in range(n)]
    var["feature_types"] = pd.Categorical(["Gene Expression"] * (n - 20) + ["Antibody Capture"] * 20)   # last 20: an ADT panel
    var["highly_variable"] = np.arange(n) % 3 == 0
    return var


def make_counts(n_cells: int, n_genes: int, rng, *, big: bool = False) -> "sp.csr_matrix":
    """Count-like CSR with explicit zeros, empty rows and a full row."""
    _lazy()
    X = sp.random(n_cells, n_genes, density=0.08, format="csr", random_state=int(rng.integers(1 << 30)), dtype=np.float32)
    X.data = np.floor(X.data * 40).astype(np.float32)                   # 0..39, some explicit zeros
    X.data[: max(1, X.data.size // 50)] = 0.0                            # more explicit zeros
    X = X.tolil()
    X[7, :] = 0                                                           # empty cells
    X[8, :] = 0
    X[11, :] = np.arange(1, n_genes + 1) % 60 + 1                        # a cell with nnz = n_genes
    if big:
        X[12, 3] = 70000.0                                               # > uint16
    X = X.tocsr()
    X.sort_indices()
    return X


def write_fixture(path, kind: str = "csr", *, seed: int = 0, n_cells: int = N_CELLS, n_genes: int = N_GENES) -> str:
    """kind: 'csr' | 'csc' | 'dense' | 'layer' | 'overflow' | 'dup_names' | 'reordered' | 'subset'."""
    _lazy()
    rng = np.random.default_rng(seed)
    X = make_counts(n_cells, n_genes, rng, big=(kind == "overflow"))
    obs = make_obs(n_cells, rng, dup_names=(kind == "dup_names"))
    var = make_var(n_genes)
    if kind == "csc":
        X = X.tocsc()
    elif kind == "dense":
        X = X.toarray()
    if kind == "reordered":
        perm = rng.permutation(n_genes)
        X = X[:, perm]
        var = var.iloc[perm]
    if kind == "subset":
        keep = np.sort(rng.choice(n_genes, n_genes // 2, replace=False))
        perm = rng.permutation(keep)
        X = X[:, perm]
        var = var.iloc[perm]
    a = anndata.AnnData(X=X, obs=obs, var=var)
    if kind == "layer":
        L = X.copy()
        L.data = L.data * 2 + 1
        a.layers["counts2"] = L
        # raw with a wider gene space than X
        raw_var = make_var(n_genes + 50)
        RX = sp.hstack([X, sp.random(n_cells, 50, density=0.3, format="csr", random_state=1, dtype=np.float32)], format="csr")
        RX.data = np.floor(RX.data)
        a.raw = anndata.AnnData(X=RX, obs=obs[[]], var=raw_var)
    a.write_h5ad(path)
    return str(path)


def write_legacy_fixture(path, *, seed: int = 0, n_cells: int = N_CELLS, n_genes: int = N_GENES) -> str:
    """The anndata < 0.8 layout (Tabula Sapiens v1 files): categorical obs columns are int
    code datasets with a ``categories`` attribute referencing ``obs/__categories/<col>``,
    no per-column ``encoding-type``; ``obs.attrs['_index']`` names the id column."""
    _lazy()
    import h5py
    rng = np.random.default_rng(seed)
    X = make_counts(n_cells, n_genes, rng)
    obs = make_obs(n_cells, rng)
    var = make_var(n_genes)
    with h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"; g.attrs["encoding-version"] = "0.1.0"; g.attrs["shape"] = np.array(X.shape)
        g.create_dataset("data", data=X.data); g.create_dataset("indices", data=X.indices); g.create_dataset("indptr", data=X.indptr)
        o = f.create_group("obs")
        o.attrs["_index"] = "cell_id"; o.attrs["encoding-type"] = "dataframe"; o.attrs["encoding-version"] = "0.1.0"
        o.attrs["column-order"] = np.array(["cell_type", "donor", "tissue", "plate", "qc_score", "is_doublet"], dtype=object)
        st = h5py.string_dtype()
        o.create_dataset("cell_id", data=np.array(obs.index, dtype=object), dtype=st)
        cats = o.create_group("__categories")
        for c in ("cell_type", "donor"):
            cd = cats.create_dataset(c, data=np.array(obs[c].cat.categories, dtype=object), dtype=st)
            d = o.create_dataset(c, data=obs[c].cat.codes.to_numpy().astype(np.int8))
            d.attrs["categories"] = cd.ref
        o.create_dataset("tissue", data=np.array(obs["tissue"], dtype=object), dtype=st)
        o.create_dataset("plate", data=obs["plate"].to_numpy()); o.create_dataset("qc_score", data=obs["qc_score"].to_numpy()); o.create_dataset("is_doublet", data=obs["is_doublet"].to_numpy())
        v = f.create_group("var")
        v.attrs["_index"] = "_index"; v.attrs["encoding-type"] = "dataframe"; v.attrs["encoding-version"] = "0.1.0"
        v.attrs["column-order"] = np.array(["gene_ids", "highly_variable"], dtype=object)
        v.create_dataset("_index", data=np.array(var.index, dtype=object), dtype=st)
        v.create_dataset("gene_ids", data=np.array(var["gene_ids"], dtype=object), dtype=st)
        v.create_dataset("highly_variable", data=var["highly_variable"].to_numpy())
    return str(path)


def expected_row(X, i: int):
    """(indices, values) of row i with explicit zeros dropped, indices sorted."""
    _lazy()
    r = sp.csr_matrix(X[i]) if not sp.issparse(X) else X[i].tocsr()
    r.sum_duplicates()
    r.sort_indices()
    keep = r.data != 0
    return r.indices[keep].astype(np.uint32), r.data[keep]

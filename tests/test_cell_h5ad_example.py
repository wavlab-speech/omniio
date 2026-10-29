"""examples/cell_h5ad.py: .h5ad files onto one gene panel through Blob.append, names on read."""
import importlib.util
import os
import sys

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
sp = pytest.importorskip("scipy.sparse")

from omniio.blob.blob import Blob
from omniio.interface import cell_read
from tests.cell_fixtures import CATS, entries, expected_row, write_fixture

_spec = importlib.util.spec_from_file_location("cell_h5ad", os.path.join(os.path.dirname(__file__), "..", "examples", "cell_h5ad.py"))
ex = importlib.util.module_from_spec(_spec); sys.modules["cell_h5ad"] = ex; _spec.loader.exec_module(ex)   # registered so items pickle to workers

KINDS = ["csr", "csc", "dense", "reordered", "subset"]


@pytest.fixture(scope="module")
def h5ad(tmp_path_factory):
    d = tmp_path_factory.mktemp("h5ad")
    return {k: write_fixture(d / f"{k}.h5ad", k) for k in KINDS}


def test_panel_items_and_round_trip_on_one_panel(tmp_path, h5ad):
    panel = ex.gene_panel([h5ad[k] for k in KINDS], gene_id_column="gene_ids")
    assert len(panel) == 500 and panel[:3] == ["ENSG00000000000", "ENSG00000000001", "ENSG00000000002"]
    items = ex.h5ad_items([h5ad[k] for k in KINDS], panel, rows=64, gene_id_column="gene_ids", obs_columns=["cell_type", "plate"])
    assert len(items) == 5 * 5 and all(isinstance(i, ex.H5adRows) for i in items)
    b = Blob(str(tmp_path / "atlas"), "cell")
    b.append(items, num_workers=2, progress=False)
    ent = entries(b)
    assert len(ent) == 5 * 300
    pos = {g: i for i, g in enumerate(panel)}
    for kind in KINDS:
        a = anndata.read_h5ad(h5ad[kind])
        col = np.array([pos[g] for g in a.var["gene_ids"]])              # file column -> panel column
        for i, name in enumerate(a.obs_names):
            path, off, size, row = ent[f"{kind}:{name}"]
            r = cell_read(path, off, size)
            idx, val = expected_row(a.X, i)
            o = np.argsort(col[idx])
            assert r.n_genes == 500 and np.array_equal(r.indices, col[idx][o]) and np.array_equal(r.values, val[o].astype(np.uint16)), (kind, name)
            assert row["cell_type"] in CATS and row["plate"] == int(a.obs["plate"].iloc[i]) and row["source_row"] == i


def test_gene_symbols_and_select_genes(tmp_path, h5ad):
    panel = ex.gene_panel([h5ad["csr"]], gene_id_column="gene_ids")
    b = Blob(str(tmp_path / "one"), "cell")
    b.append(ex.h5ad_items([h5ad["csr"]], panel, rows=1000, gene_id_column="gene_ids"), progress=False)
    names = ex.gene_symbols(h5ad["csr"], gene_id_column="gene_ids")
    assert names["ENSG00000000007"] == "GENE7"
    a = anndata.read_h5ad(h5ad["csr"])
    ent = entries(b)
    wanted = ["ENSG00000000011", "ENSG00000000003", "NOT_A_GENE", "ENSG00000000499"]
    r = cell_read(*ent["csr:cell_00011"][:3])
    v = ex.select_genes(r, panel, wanted)
    dense = a.X[11].toarray()[0]
    assert v.tolist() == [dense[11], dense[3], 0, dense[499]]

"""Figures 1, 2, 8, 9 (Tabula Sapiens): the 24 tissue files of Tabula Sapiens v1 (figshare
article 14267219, TS_<Tissue>.h5ad) into one `cell` archive through Blob.append, tissue file
by tissue file (storage order = tissue-major), raw counts (layer `raw_counts`) at the
narrowest width per cell, columns = the union of the files' Ensembl ids. The index carries
the obs columns the experiments use as source (organ_tissue, donor) and class
(cell_ontology_class).

    python variance/cell/prep_tabula_sapiens.py --workers 16
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.paths import data  # noqa: E402

OBS = ["cell_ontology_class", "donor", "organ_tissue"]


def example():
    """examples/cell_h5ad.py of the repository: the panel and the H5adRows item"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    spec = importlib.util.spec_from_file_location("cell_h5ad", os.path.join(root, "examples", "cell_h5ad.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules["cell_h5ad"] = mod; spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=data("sources", "tabula_sapiens"), help="directory with the TS_*.h5ad files")
    ap.add_argument("--out", default=data("cell", "tabula_sapiens"))
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--rows", type=int, default=10_000)
    a = ap.parse_args()
    from omniio.blob.blob import Blob
    ex = example()
    files = sorted(glob.glob(os.path.join(a.src, "TS_*.h5ad")))
    if os.path.exists(os.path.join(a.out, "metadata.parquet")):
        print("exists:", a.out); return
    t0 = time.time()
    panel = ex.gene_panel(files, gene_id_column="ensemblid")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "panel.txt"), "w") as f:
        f.write("\n".join(panel) + "\n")
    items = ex.h5ad_items(files, panel, rows=a.rows, layer="raw_counts", gene_id_column="ensemblid", obs_columns=OBS)
    blob = Blob(a.out, modality="cell")
    blob.append(items, num_workers=a.workers, value_dtype="auto", progress=False)
    print(f"{len(blob)} cells over {len(panel)} genes from {len(files)} files in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

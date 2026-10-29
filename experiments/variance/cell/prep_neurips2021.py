"""Figures 1, 2, 8, 9 (NeurIPS Multiome): the Multiome cohort of the NeurIPS 2021 multimodal
single-cell integration data (GEO GSE194122, GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad)
into two `cell` archives through Blob.append, the RNA counts (feature type GEX) and the ATAC peak
counts (feature type ATAC), both in the file's cell order so row i of one is row i of the other.
The index carries cell_type (the class), batch (= site x donor, the source), Site and DonorNumber.

    python variance/cell/prep_neurips2021.py --workers 16
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.paths import data  # noqa: E402
from prep_tabula_sapiens import example  # noqa: E402

OBS = ["cell_type", "batch", "Site", "DonorNumber"]
FILE = "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=data("sources", "neurips2021"), help="directory with the GSE194122 h5ad")
    ap.add_argument("--out", default=data("cell", "multiome"))
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--rows", type=int, default=10_000)
    a = ap.parse_args()
    from omniio.blob.blob import Blob
    ex = example()
    path = os.path.join(a.src, FILE)
    with ex._open(path) as f:
        var = ex._frame(f, "var")
    for name, ft in (("rna", "GEX"), ("atac", "ATAC")):
        out = os.path.join(a.out, name)
        if os.path.exists(os.path.join(out, "metadata.parquet")):
            print("exists:", out); continue
        t0 = time.time()
        panel = [str(v) for v in var.index[var["feature_types"] == ft]]       # the panel: this feature type's var names
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "panel.txt"), "w") as f:
            f.write("\n".join(panel) + "\n")
        items = ex.h5ad_items([path], panel, rows=a.rows, layer="counts", obs_columns=OBS)
        blob = Blob(out, modality="cell")
        blob.append(items, num_workers=a.workers, value_dtype="auto", progress=False)
        print(f"{name}: {len(blob)} cells over {len(panel)} features in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

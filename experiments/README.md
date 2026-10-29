# Reproducing the paper's figures

Every figure of the paper is produced by one script here, `run_fig<N>.sh`, which runs the
measurements and then the plot. Figures 8 and 9 come out of the runs of Figure 2.

| script | figure | what runs |
|---|---|---|
| `run_fig1.sh` | 1 | composition, `r` and `R_hat` at initialization on ImageNet, Libri-Light, Tabula Sapiens, Flickr8k, ESC-50, NeurIPS Multiome |
| `run_fig2.sh` | 2, 8, 9 | training under each loader order on the same six corpora |
| `run_fig4_fig10.sh` | 4, 10 | ingestion, DataLoader, random-access and sequential-read benchmarks of the formats on ImageNet-1k, PG-19, UCF101 |
| `run_fig5.sh` | 5 | the Bagpiper corpus: random access against corpus size, batch composition, DataLoader on 1 to 8 nodes |
| `run_fig6.sh` | 6 | speed and memory of partial video reads |
| `run_fig8_fig9.sh` | 8, 9 | only the plot, from the runs of `run_fig2.sh` |

## Install

```bash
pip install -e ..                       # omniio, from the repository root
pip install -r experiments/requirements.txt
```

The GPU runs (Figures 1 and 2: ImageNet, Libri-Light, ESC-50, Flickr8k) used one NVIDIA GH200;
everything else ran on CPU nodes. Figure 5's DataLoader panel runs one rank per node through
Slurm's `srun`.

## Data

Set `OMNIIO_EXPERIMENTS` to a directory with room for the corpora (default: this directory). Data
goes under `$OMNIIO_EXPERIMENTS/data`, results under `results`, figures under `figures`.

Downloaded by the scripts: ImageNet-1k (the 256 px copy), PG-19 and UCF101 from the Hugging
Face Hub (`data/download_formats.py`), ESC-50 and Flickr8k (`variance/contrastive/ingest_*.py`).
The video corpus of Figure 6 is generated (`data/make_video_corpus.py`).

To place by hand under `data/sources/`:

- `librilight/medium/`: the Libri-Light *medium* subset (speaker/book/file.flac), from
  https://github.com/facebookresearch/libri-light
- `tabula_sapiens/TS_*.h5ad`: the 24 tissue files of Tabula Sapiens v1 (figshare article 14267219)
- `neurips2021/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad`: the Multiome
  cohort of GEO GSE194122
Figure 5's corpus is the Bagpiper pre-training collection, downloaded by `run_fig5.sh` from
`espnet/Bagpiper_PreTrain_Data_Collection` on the Hub (`data/download_bagpiper.py`).

## Layout

- `common/`: the definitions every experiment shares. `stats.py` (n_eff, the intraclass
  correlation `r` by one-way ANOVA, `R_hat`, `R_pred`), `orders.py` (the index order each
  loader emits: random access, a WebDataset shuffle buffer over shards of the storage order,
  the pre-shuffled copy, block sampling), `paths.py`, `harness.py` (provenance in every result).
- `data/`: corpus preparation.
- `variance/`: Figures 1 and 2. `imagenet_sweep.py`, `librilight_sweep.py` (with the
  `audio_encoder.py` Transformer), `contrastive/` (ESC-50 with CLAP, Flickr8k with CLIP),
  `cell/` (Tabula Sapiens, NeurIPS Multiome), `train_cls.py`, and the two plot scripts.
- `formats/`: Figures 4 and 10, `scale_bench.py` and `plot_scale.py`.
- `windowed/`: Figure 6.
- `bigscale/`: Figure 5.

Every measurement script is idempotent: a result that exists is not recomputed, so a run script
can be restarted. Repeats that the paper reports as means over fresh nodes (random access,
sequential reads, the windowed-read speed) are numbered `--rep 1 2 3`; run each on its own node
so that the first pass is cold.

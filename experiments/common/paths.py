"""Where the experiments read and write.

    OMNIIO_EXPERIMENTS   root for data, results and figures (default: this directory)
        $OMNIIO_EXPERIMENTS/data/<corpus>/...     the materialized corpora and archives
        $OMNIIO_EXPERIMENTS/results/<figure>/...  one JSON per run
        $OMNIIO_EXPERIMENTS/figures/              the paper's figures
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("OMNIIO_EXPERIMENTS", HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def data(*parts):
    return os.path.join(DATA, *parts)


def results(*parts):
    p = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def figures():
    os.makedirs(FIGURES, exist_ok=True)
    return FIGURES


def node():
    return os.uname().nodename.split(".")[0]

"""
Batch-composition and gradient-variance quantities of the revised Section 4
(random-effects / cluster-sampling model). Every experiment imports these so
that one definition is used everywhere.

For a batch of B items with n_s items from source s and p_s = n_s / B:

  n_eff       1 / sum_s p_s^2  (inverse Simpson index)
  n_distinct  number of sources with n_s > 0
  r           intraclass correlation tau^2 / (tau^2 + sigma_e^2), estimated by
              one-way ANOVA on per-sample gradients, per coordinate, averaged
  R_pred      [1 + r (B/n_eff - 1)] / [1 + r (B/n_eff_rand - 1)]
  R_bound     n_eff_rand / n_eff  (the r = 1 limit; labels only)
  R_hat       Var_across_batches(gbar_loader) / Var_across_batches(gbar_rand),
              variances averaged over coordinates, same theta, M batches each
  bias2       ||mean_loader - mean_rand||^2 / trace Var_rand (in-batch losses)
  buffer_bound  ceil((3k + B) / lambda) + 1  on n_distinct (Eq. 11)
"""
import math
from typing import Dict, Sequence

import os
import numpy as np


# ── composition (labels only) ──────────────────────────────────────────────────

def composition(labels: Sequence[int]) -> Dict[str, float]:
    """n_eff, n_distinct and (for one comparison run) H_eff of one batch."""
    counts = np.unique(np.asarray(labels), return_counts=True)[1].astype(np.float64)
    p = counts / counts.sum()
    srt = np.sort(p)[::-1]
    n95 = int(np.searchsorted(np.cumsum(srt), 0.95 - 1e-12) + 1)  # sources that supply 95% of the batch
    return {"n_eff": float(1.0 / np.sum(p * p)), "simpson": float(np.sum(p * p)), "n_distinct": int(len(counts)),
            "n_95": n95, "H_eff": float(np.exp(-np.sum(p * np.log(p))))}


def n_eff(labels: Sequence[int]) -> float:
    return composition(labels)["n_eff"]


def n_distinct(labels: Sequence[int]) -> int:
    return composition(labels)["n_distinct"]


def buffer_bound(k: int, B: int, lam: float) -> int:
    """Eq. 11: sources that supply 95% of a batch, for buffer k, batch B, run length lambda."""
    return int(math.ceil((3 * k + B) / lam) + 1)


def run_lengths(labels: np.ndarray, shard_size: int = None) -> Dict[str, float]:
    """lambda statistics of a storage order: lengths of the maximal runs of
    equal labels, optionally capped at the shard size (a run cannot be used
    beyond the shard that holds it)."""
    labels = np.asarray(labels)
    change = np.flatnonzero(labels[1:] != labels[:-1])
    runs = np.diff(np.concatenate([[-1], change, [len(labels) - 1]])).astype(np.float64)
    out = {"mean": float(runs.mean()), "median": float(np.median(runs)), "n_runs": int(len(runs))}
    if shard_size:
        capped = np.minimum(runs, shard_size)
        out.update({"mean_capped": float(capped.mean()), "median_capped": float(np.median(capped)), "shard_size": shard_size})
    return out


# ── intraclass correlation from per-sample gradients ───────────────────────────

def estimate_r(grads: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    """One-way random-effects ANOVA per coordinate.

    grads: (N, d) per-sample gradients at fixed theta; labels: (N,) source of
    each sample. Per coordinate, with groups of sizes n_s, grand mean, group
    means m_s:  MSB = sum_s n_s (m_s - m)^2 / (S - 1),  MSW = within SS / (N - S),
    n0 = (N - sum n_s^2 / N) / (S - 1),  tau^2 = max(0, (MSB - MSW) / n0),
    sigma_e^2 = MSW,  r = tau^2 / (tau^2 + sigma_e^2).
    Returns the mean over coordinates and the 10th/90th percentiles."""
    grads = np.asarray(grads, dtype=np.float64)
    labels = np.asarray(labels)
    N, d = grads.shape
    src, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    S = len(src)
    if S < 2 or N <= S:
        raise ValueError("need at least 2 sources and more samples than sources")
    sums = np.zeros((S, d))
    np.add.at(sums, inv, grads)
    means = sums / counts[:, None]
    grand = grads.mean(axis=0)
    ssb = np.sum(counts[:, None] * (means - grand) ** 2, axis=0)
    ssw = np.sum((grads - means[inv]) ** 2, axis=0)
    msb, msw = ssb / (S - 1), ssw / (N - S)
    n0 = (N - np.sum(counts ** 2) / N) / (S - 1)
    tau2_raw = (msb - msw) / n0
    return anova_summary(tau2_raw, msw, int(N), int(S), int(counts.min()))


def trace_ratio(tau2_raw: np.ndarray, sigma_e2: np.ndarray) -> float:
    """r as a trace ratio, sum_j tau2_j / sum_j (tau2_j + sigma_e2_j) over all coordinates,
    with the per-coordinate tau2 left unclipped (negative estimates cancel in the sum) and
    only the final sum clipped at 0. This is the r that enters R_pred: R_hat is a ratio of
    sums of variances over coordinates, so coordinates with large variance dominate it,
    while the coordinate-mean r weights every coordinate equally and under-predicts."""
    num = float(np.sum(tau2_raw)); den = float(np.sum(tau2_raw + sigma_e2))
    return max(0.0, num) / den if den > 0 else 0.0


def anova_summary(tau2_raw: np.ndarray, msw: np.ndarray, N: int, S: int, min_per_source: int) -> Dict[str, float]:
    """The r record: r_trace (what R_pred uses), the coordinate-mean r with p10/p90 of the
    per-coordinate (clipped) ICC as diagnostics, r_weighted (the clipped-sum variant kept
    for comparison with earlier records), and the unclipped / clipped tau2 means."""
    tau2 = np.maximum(0.0, tau2_raw)
    r = tau2 / (tau2 + msw + 1e-30)
    return {"r": float(r.mean()), "r_p10": float(np.percentile(r, 10)), "r_p90": float(np.percentile(r, 90)),
            "r_trace": trace_ratio(tau2_raw, msw),
            "r_weighted": float(tau2.sum() / (tau2.sum() + msw.sum() + 1e-30)),
            "tau2_mean": float(tau2.mean()), "tau2_raw_mean": float(tau2_raw.mean()), "sigma_e2_mean": float(msw.mean()),
            "frac_tau2_negative": float(np.mean(tau2_raw < 0)),
            "n_samples": int(N), "n_sources": int(S), "min_per_source": int(min_per_source)}


class StreamingANOVA:
    """One-pass version of estimate_r: accumulates per-source sums and the per-
    coordinate sum of squares so that N gradient vectors never have to be held at
    once (N x d float32 for a 5 M-parameter model and 500 samples is 10 GB before
    the float64 copies). Identical formulas; same return dict."""

    def __init__(self, n_sources: int, d: int):
        self.sums = np.zeros((n_sources, d))
        self.sumsq = np.zeros(d)
        self.counts = np.zeros(n_sources, dtype=np.int64)

    def add(self, grad: np.ndarray, label: int) -> None:
        g = np.asarray(grad, dtype=np.float64)
        self.sums[label] += g
        self.sumsq += g * g
        self.counts[label] += 1

    def components(self, chunk: int = 1 << 20):
        """Per-coordinate (tau2_raw, sigma_e2) in float64 plus N, S, min count; chunked."""
        used = self.counts > 0
        counts = self.counts[used]
        S, N = int(used.sum()), int(counts.sum())
        if S < 2 or N <= S:
            raise ValueError("need at least 2 sources and more samples than sources")
        d = self.sums.shape[1]
        tau2_raw = np.empty(d); msw = np.empty(d)
        n0 = (N - np.sum(counts ** 2) / N) / (S - 1)
        for a in range(0, d, chunk):
            b = min(a + chunk, d)
            sums = self.sums[used, a:b]
            means = sums / counts[:, None]
            grand = sums.sum(axis=0) / N
            ssb = np.sum(counts[:, None] * (means - grand) ** 2, axis=0)
            ssw = np.maximum(0.0, self.sumsq[a:b] - np.sum(counts[:, None] * means ** 2, axis=0))
            msb = ssb / (S - 1); msw[a:b] = ssw / (N - S)
            tau2_raw[a:b] = (msb - msw[a:b]) / n0
        return tau2_raw, msw, N, S, int(counts.min())

    def result(self, coords=None, chunk: int = 1 << 20, save_components: str = None) -> Dict[str, float]:
        """Computed in coordinate chunks (the one-shot temporaries of a 100 x 50 M problem got
        the process OOM-killed). `coords`: a slice of the coordinates (one tower's parameters)
        for r over that part alone. `save_components`: path of an .npz that receives the
        per-coordinate tau2_raw and sigma_e2 (float32), so r can be re-weighted offline."""
        tau2_raw, msw, N, S, mn = self.components(chunk)
        if coords is not None:
            tau2_raw, msw = tau2_raw[coords], msw[coords]
        if save_components:
            os.makedirs(os.path.dirname(save_components) or ".", exist_ok=True)
            np.savez(save_components, tau2_raw=tau2_raw.astype(np.float32), sigma_e2=msw.astype(np.float32), n_samples=N, n_sources=S)
        return anova_summary(tau2_raw, msw, N, S, mn)


def predict_R(r: float, n_eff_loader: float, n_eff_rand: float, B: int) -> float:
    return (1 + r * (B / n_eff_loader - 1)) / (1 + r * (B / n_eff_rand - 1))


def bound_R(n_eff_loader: float, n_eff_rand: float) -> float:
    return n_eff_rand / n_eff_loader


def concentration(simpson_batch: float, simpson_corpus: float, B: int) -> float:
    """c = kappa / kappa_rand with kappa = S_batch - S_corpus and kappa_rand = (1 - S_corpus) / B
    (the expected excess Simpson index of a uniformly random batch). R_pred = 1 + r (c - 1),
    R_bound = c (the r = 1 limit)."""
    return (simpson_batch - simpson_corpus) / ((1.0 - simpson_corpus) / B)


def predict_R_c(r: float, c: float) -> float:
    return 1.0 + r * (c - 1.0)


def corpus_simpson(labels: Sequence[int]) -> float:
    """sum_s pi_s^2 over the corpus shares pi_s (1 / the corpus-level effective number of sources)."""
    counts = np.unique(np.asarray(labels), return_counts=True)[1].astype(np.float64)
    pi = counts / counts.sum()
    return float(np.sum(pi * pi))


def predict_R_fp(r: float, simpson_loader: float, simpson_rand: float, simpson_corpus: float, B: int) -> float:
    """Finite-population form. With a fixed set of sources the corpus mean
    sum_s pi_s v_s is common to every batch, so across batches
    Var(gbar) = tau^2 (E[sum_s p_s^2] - sum_s pi_s^2) + sigma_e^2 / B;
    dividing by tau^2 + sigma_e^2 gives r (S_loader - S_corpus) + (1 - r) / B,
    with S the mean Simpson index sum_s p_s^2 of the loader's batches."""
    return (r * (simpson_loader - simpson_corpus) + (1 - r) / B) / (r * (simpson_rand - simpson_corpus) + (1 - r) / B)


def bound_R_fp(simpson_loader: float, simpson_rand: float, simpson_corpus: float) -> float:
    """r = 1 limit of predict_R_fp (labels only)."""
    return (simpson_loader - simpson_corpus) / (simpson_rand - simpson_corpus)


# ── gradient variance across batches at fixed theta ────────────────────────────

def batch_mean_stats(gbar: np.ndarray) -> Dict[str, np.ndarray]:
    """gbar: (M, d) batch-mean gradients. Per-coordinate mean and unbiased variance."""
    gbar = np.asarray(gbar, dtype=np.float64)
    return {"mean": gbar.mean(axis=0), "var": gbar.var(axis=0, ddof=1)}


def _block_indices(m: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap sample of m indices: blocks of `block` consecutive
    batches drawn with replacement, which keeps the serial correlation that
    consecutive batches from one shuffle buffer have."""
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, max(1, m - block + 1), size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:m]


def R_hat(gbar_loader: np.ndarray, gbar_rand: np.ndarray, n_boot: int = 1000, seed: int = 0, block: int = 10) -> Dict[str, float]:
    """R_hat = mean_coord Var(gbar_loader) / mean_coord Var(gbar_rand) over the M
    batch means of each loader (in stream order), with a moving-block bootstrap
    CI (block of `block` consecutive batches; the iid bootstrap is also given),
    and bias2 = ||mean_loader - mean_rand||^2 / trace Var_rand."""
    a, b = np.asarray(gbar_loader, dtype=np.float64), np.asarray(gbar_rand, dtype=np.float64)
    d = a.shape[1]
    # The coordinate-mean variance of a resample is a function of the Gram matrix only:
    #   mean_j Var_j(a[idx]) = (sum_i w_i |a_i|^2 - |sum_i w_i a_i|^2 / M) / (d (M - 1)),
    # with w_i the multiplicity of row i in the resample, so each bootstrap draw is O(M^2)
    # instead of O(M d) (identical to a[idx].var(axis=0, ddof=1).mean() up to rounding).
    Ga, Gb = a @ a.T, b @ b.T
    def _mvar(G, idx):
        w = np.bincount(idx, minlength=len(G)).astype(np.float64); M = len(idx)
        return (w @ np.diag(G) - w @ G @ w / M) / (d * (M - 1))
    va, vb = _mvar(Ga, np.arange(len(a))), _mvar(Gb, np.arange(len(b)))
    R = va / vb
    bias2 = float(np.sum((a.mean(axis=0) - b.mean(axis=0)) ** 2) / (vb * d))
    rng = np.random.default_rng(seed)
    boots, boots_iid = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        ia, ib = _block_indices(len(a), block, rng), _block_indices(len(b), block, rng)
        boots[i] = _mvar(Ga, ia) / _mvar(Gb, ib)
        ia, ib = rng.integers(len(a), size=len(a)), rng.integers(len(b), size=len(b))
        boots_iid[i] = _mvar(Ga, ia) / _mvar(Gb, ib)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lo2, hi2 = np.percentile(boots_iid, [2.5, 97.5])
    return {"R_hat": float(R), "R_hat_ci": [float(lo), float(hi)], "R_hat_ci_iid": [float(lo2), float(hi2)], "bootstrap_block": block,
            "bias2": bias2, "MSE_ratio": float(R + bias2), "B_eff_over_B": float(1.0 / R), "M": int(len(a))}


# ── pseudo-gradient model (synthetic experiments) ──────────────────────────────

class SourceGradients:
    """Per-sample pseudo-gradient g = v_s + e, with v_s ~ N(0, signal_std^2 I) fixed
    per source and e ~ N(0, noise_std^2 I) fresh per sample, d coordinates.
    Under the random-effects model the intraclass correlation is exactly
    r_true = signal_std^2 / (signal_std^2 + noise_std^2) per coordinate."""

    def __init__(self, n_sources: int, d: int = 256, signal_std: float = 1.0, noise_std: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.v = (rng.standard_normal((n_sources, d)) * signal_std).astype(np.float64)
        self.noise_std, self.d = noise_std, d
        self.rng = np.random.default_rng(seed + 1)
        self.r_true = signal_std ** 2 / (signal_std ** 2 + noise_std ** 2)

    def per_sample(self, labels: Sequence[int]) -> np.ndarray:
        labels = np.asarray(labels)
        return self.v[labels] + self.rng.standard_normal((len(labels), self.d)) * self.noise_std

    def batch_mean(self, labels: Sequence[int]) -> np.ndarray:
        return self.per_sample(labels).mean(axis=0)

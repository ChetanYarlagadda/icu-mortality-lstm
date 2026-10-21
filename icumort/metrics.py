"""Discrimination, calibration and challenge metrics for binary risk predictions, plus bootstrap and Holm helpers.

The two challenge scores follow the organisers' ``score.c`` (version 1.4) exactly:

- Event 1 = min(Se, +P) of the binary predictions, Se = TP / (TP + FN), +P = TP / (TP + FP).
- Event 2 = H / D: stays are sorted by risk (ties by RecordID) and cut into 10 groups of equal size;
  H = sum_g (O_g - E_g)^2 / (N_g (pi_g (1 - pi_g) + 0.001)), with O_g observed deaths, E_g = N_g pi_g expected deaths
  and pi_g the mean risk of group g, and D = pi_10 - pi_1. Lower is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata

EPS = 1e-7


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(score: np.ndarray, y: np.ndarray) -> float:
    """Average precision: sum over distinct thresholds of (recall_k - recall_{k-1}) * precision_k."""
    y = np.asarray(y).astype(float)
    order = np.argsort(-np.asarray(score), kind="stable")
    s, yy = np.asarray(score)[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # last index of each distinct score
    tp = np.cumsum(yy)[last]
    fp = (last + 1) - tp
    if tp[-1] == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of the risk."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed death rate against mean predicted risk in ``n_bins`` equal-size groups of increasing risk."""
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, n_bins)
    return pd.DataFrame({"bin": np.arange(1, n_bins + 1), "n": [len(g) for g in groups],
                         "mean_risk": [float(p[g].mean()) for g in groups],
                         "observed_rate": [float(np.mean(y[g])) for g in groups]})


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-size risk groups: sum_g (n_g / n) |observed_g - predicted_g|."""
    r = reliability(p, y, n_bins)
    return float(np.sum(r["n"] / r["n"].sum() * np.abs(r["observed_rate"] - r["mean_risk"])))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _logistic_fit(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    """Newton-Raphson maximum likelihood for y ~ sigmoid(x @ beta + offset)."""
    beta = np.zeros(x.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(x @ beta + off)))
        w = mu * (1 - mu) + 1e-12
        step = np.linalg.solve(x.T @ (x * w[:, None]) + 1e-9 * np.eye(x.shape[1]), x.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with the slope fixed at 1) and calibration slope on logit(p).

    A perfectly calibrated model has intercept 0 and slope 1; a slope below 1 means the risks are too extreme.
    """
    lp = _logit(p)
    intercept = _logistic_fit(np.ones((len(y), 1)), y.astype(float), offset=lp)[0]
    slope = _logistic_fit(np.c_[np.ones(len(y)), lp], y.astype(float))[1]
    return float(intercept), float(slope)


def platt_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: (a, b) of sigmoid(a + b logit(p)) fitted by maximum likelihood (on validation data)."""
    a, b = _logistic_fit(np.c_[np.ones(len(y)), _logit(p)], y.astype(float))
    return float(a), float(b)


def platt_apply(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    """Apply fitted Platt parameters to risks ``p``."""
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * _logit(p))))


def hosmer_lemeshow(p: np.ndarray, y: np.ndarray, groups: int = 10) -> tuple[float, float]:
    """Classical Hosmer-Lemeshow C statistic over risk deciles and its chi-square p-value (groups - 2 df)."""
    r = reliability(p, y, groups)
    o, e, n = r["observed_rate"] * r["n"], r["mean_risk"] * r["n"], r["n"]
    stat = float(np.sum((o - e) ** 2 / (e * (1 - e / n) + 1e-12)))
    return stat, float(chi2.sf(stat, groups - 2))


def event1(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Challenge Event 1: (min(Se, +P), Se, +P) of binary predictions."""
    pred, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    tp = float(np.sum(pred & y))
    se = tp / max(float(y.sum()), 1.0)
    ppv = tp / float(pred.sum()) if pred.sum() else 0.0
    return min(se, ppv), se, ppv


def event2(risk: np.ndarray, y: np.ndarray, record_id: np.ndarray, n_groups: int = 10) -> tuple[float, float, float]:
    """Challenge Event 2 as in score.c: (H / D, H, D). Requires len(risk) divisible by ``n_groups``."""
    n = len(risk)
    if n % n_groups:
        raise ValueError("score.c needs equal-size risk groups")
    order = np.lexsort((np.asarray(record_id), np.asarray(risk)))
    r, obs = np.asarray(risk, dtype=float)[order], np.asarray(y, dtype=float)[order]
    m = n // n_groups
    pi = r.reshape(n_groups, m).mean(axis=1)
    o = obs.reshape(n_groups, m).sum(axis=1)
    h = float(np.sum((o - m * pi) ** 2 / (m * (pi * (1 - pi) + 0.001))))
    d = float(pi[-1] - pi[0])
    return h / d, h, d


def event1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Risk threshold (predict death if risk >= threshold) maximising Event 1 on the given (validation) data."""
    order = np.argsort(-np.asarray(p), kind="stable")
    ys, ps = np.asarray(y)[order].astype(float), np.asarray(p)[order]
    tp, k = np.cumsum(ys), np.arange(1, len(ys) + 1)
    score = np.minimum(tp / ys.sum(), tp / k)
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]  # only cut between distinct risks
    return float(ps[last[np.argmax(score[last])]])


def summarize(p: np.ndarray, y: np.ndarray, record_id: np.ndarray, threshold: float) -> dict[str, float]:
    """Every headline metric of one set of risks; ``threshold`` (chosen on validation data) gives Event 1."""
    e1, se, ppv = event1(p >= threshold, y)
    e2, h, d = event2(p, y, record_id) if len(p) % 10 == 0 else (np.nan, np.nan, np.nan)
    intercept, slope = calibration_slope_intercept(p, y)
    hl, hl_p = hosmer_lemeshow(p, y)
    return {"auroc": auroc(p, y), "auprc": auprc(p, y), "brier": brier(p, y), "log_loss": log_loss(p, y),
            "ece": ece(p, y), "cal_intercept": intercept, "cal_slope": slope, "hl_stat": hl, "hl_p": hl_p,
            "event1": e1, "sensitivity": se, "ppv": ppv, "threshold": threshold, "event2": e2, "event2_h": h,
            "event2_d": d, "mean_risk": float(np.mean(p)), "observed_rate": float(np.mean(y))}


# Event 2 is left out: H grows with resampling noise, so its bootstrap distribution sits above the point estimate
# and percentile intervals would be misleading; seed-to-seed spread is reported for it instead.
BOOT_METRICS = ("auroc", "auprc", "brier", "event1")


def bootstrap_indices(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n) stay indices resampled with replacement; the same draws are reused for paired comparisons."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_metrics(p: np.ndarray, y: np.ndarray, threshold: float, draws: np.ndarray) -> dict[str, np.ndarray]:
    """``BOOT_METRICS`` on every bootstrap resample (rows of ``draws``)."""
    out = {m: np.empty(len(draws)) for m in BOOT_METRICS}
    pred = p >= threshold
    for b, idx in enumerate(draws):
        pb, yb = p[idx], y[idx]
        out["auroc"][b] = auroc(pb, yb)
        out["auprc"][b] = auprc(pb, yb)
        out["brier"][b] = brier(pb, yb)
        out["event1"][b] = event1(pred[idx], yb)[0]
    return out


def paired_p(diff: np.ndarray) -> float:
    """Two-sided p-value that a paired difference is zero, from its bootstrap mean and standard error."""
    se = float(np.std(diff, ddof=1))
    if se == 0:
        return 0.0 if float(np.mean(diff)) != 0 else 1.0
    return float(2 * norm.sf(abs(float(np.mean(diff))) / se))


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr)
    m = len(p_arr)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_arr[i])
        adj[i] = min(1.0, running)
    return adj.tolist()

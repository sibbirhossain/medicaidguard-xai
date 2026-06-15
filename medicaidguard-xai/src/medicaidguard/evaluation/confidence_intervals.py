"""Bootstrap confidence intervals and paired model comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n_boot: int

    def as_dict(self):
        return {"point": self.point, "ci_lo": self.lo, "ci_hi": self.hi, "n_boot": self.n_boot}

    def __str__(self):
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def bootstrap_metric(y_true, y_score, metric_fn, n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> Interval:
    """Percentile bootstrap over claims.

    Resampling is at the claim level, which understates dependence between
    claims from the same caregiver. That limitation is stated in the report
    rather than hidden; a cluster bootstrap over caregivers is available via
    `cluster_bootstrap_metric`.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    if not vals:
        return Interval(point, np.nan, np.nan, 0)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), len(vals))


def cluster_bootstrap_metric(y_true, y_score, clusters, metric_fn,
                             n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> Interval:
    """Bootstrap over entities (e.g. caregivers) rather than individual claims."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    point = float(metric_fn(y_true, y_score))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in pick])
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    if not vals:
        return Interval(point, np.nan, np.nan, 0)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), len(vals))


def paired_bootstrap_difference(y_true, score_a, score_b, metric_fn,
                                n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap of metric(A) - metric(B) on the same resampled rows.

    Pairing is what makes the comparison informative: both models are scored on
    exactly the same bootstrap sample, so shared sampling noise cancels.
    Returns (interval, two_sided_p_like) where the second value is the bootstrap
    proportion of resamples on the opposite side of zero, doubled. It is a
    descriptive quantity, not an exact p-value.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a, b = np.asarray(score_a), np.asarray(score_b)
    n = len(y_true)
    point = float(metric_fn(y_true, a) - metric_fn(y_true, b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(metric_fn(y_true[idx], a[idx]) - metric_fn(y_true[idx], b[idx]))
    if not diffs:
        return Interval(point, np.nan, np.nan, 0), np.nan
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return Interval(point, float(lo), float(hi), len(diffs)), float(min(p, 1.0))


def delong_roc_test(y_true, score_a, score_b):
    """DeLong test for two correlated ROC-AUCs. Returns (auc_a, auc_b, z, p)."""
    from scipy import stats

    y_true = np.asarray(y_true)
    pos = score_a[y_true == 1], score_b[y_true == 1]
    neg = score_a[y_true == 0], score_b[y_true == 0]
    m, n = len(pos[0]), len(neg[0])
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan

    def _structural(p, q):
        # Midrank-free O(mn) would be costly; m,n here are modest.
        comp = (p[:, None] > q[None, :]).astype(float) + 0.5 * (p[:, None] == q[None, :])
        return comp

    v = [_structural(pos[k], neg[k]) for k in (0, 1)]
    auc = np.array([c.mean() for c in v])
    v10 = np.array([c.mean(axis=1) for c in v])   # 2 x m
    v01 = np.array([c.mean(axis=0) for c in v])   # 2 x n
    s10 = np.cov(v10)
    s01 = np.cov(v01)
    s = s10 / m + s01 / n
    contrast = np.array([1.0, -1.0])
    var = contrast @ s @ contrast
    if var <= 0:
        return float(auc[0]), float(auc[1]), np.nan, np.nan
    z = (auc[0] - auc[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc[0]), float(auc[1]), float(z), float(p)


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test on the discordant pairs of two binary classifiers."""
    from scipy import stats

    y_true = np.asarray(y_true)
    a_ok = (np.asarray(pred_a) == y_true)
    b_ok = (np.asarray(pred_b) == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return n01, n10, np.nan
    # Exact binomial version; valid for small discordant counts too.
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return n01, n10, float(p)

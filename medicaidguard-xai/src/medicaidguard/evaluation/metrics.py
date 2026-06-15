"""Evaluation metrics for imbalanced, capacity-constrained detection.

The metric set is deliberately wider than accuracy/ROC-AUC. Two reasons:

* Under 3-4% prevalence, accuracy is dominated by the negative class. A
  constant-negative predictor scores ~0.96 here while detecting nothing, so
  accuracy cannot rank models. It is reported for completeness and explicitly
  annotated with the trivial-baseline value.
* Investigators have finite capacity. What matters operationally is how many
  true cases appear in the top K% of the ranked queue, which is what
  Precision@K and Recall@K measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true, y_score, k_pct: float) -> float:
    """Precision among the top k_pct fraction of the ranked queue."""
    y_true = np.asarray(y_true)
    n = max(1, int(round(len(y_true) * k_pct)))
    order = np.argsort(-np.asarray(y_score), kind="stable")[:n]
    return float(y_true[order].mean())


def recall_at_k(y_true, y_score, k_pct: float) -> float:
    y_true = np.asarray(y_true)
    pos = y_true.sum()
    if pos == 0:
        return float("nan")
    n = max(1, int(round(len(y_true) * k_pct)))
    order = np.argsort(-np.asarray(y_score), kind="stable")[:n]
    return float(y_true[order].sum() / pos)


def lift_at_k(y_true, y_score, k_pct: float) -> float:
    """Precision@K divided by base prevalence: how much better than random."""
    base = float(np.asarray(y_true).mean())
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, y_score, k_pct) / base


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def classification_at_threshold(y_true, y_prob, threshold: float) -> dict:
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "alerts_raised": int(tp + fp),
        "alert_rate": float((tp + fp) / len(y_true)),
    }


def evaluate(y_true, y_prob, k_pcts=(0.01, 0.05, 0.10), threshold: float = 0.5,
             is_probability: bool = True) -> dict:
    """Full metric block for one model on one evaluation set."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    prevalence = float(y_true.mean())

    res: dict[str, float | int | str] = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    for k in k_pcts:
        tag = f"{int(round(k * 100))}pct"
        res[f"precision_at_{tag}"] = precision_at_k(y_true, y_prob, k)
        res[f"recall_at_{tag}"] = recall_at_k(y_true, y_prob, k)
        res[f"lift_at_{tag}"] = lift_at_k(y_true, y_prob, k)
        res[f"reviews_at_{tag}"] = int(round(len(y_true) * k))

    res.update(classification_at_threshold(y_true, y_prob, threshold))
    pred = (y_prob >= threshold).astype(int)
    res["accuracy"] = float((pred == y_true).mean())
    # The number any accuracy claim must be compared against.
    res["accuracy_trivial_baseline"] = float(max(prevalence, 1 - prevalence))

    if is_probability:
        p = np.clip(y_prob, 0.0, 1.0)
        res["brier"] = float(brier_score_loss(y_true, p))
        res["ece"] = expected_calibration_error(y_true, p)
    else:
        res["brier"] = float("nan")
        res["ece"] = float("nan")
    return res


def best_f1_threshold(y_true, y_prob) -> float:
    """Threshold maximising F1 on the given set (use validation, never test)."""
    y_true = np.asarray(y_true)
    cand = np.unique(np.quantile(np.asarray(y_prob), np.linspace(0.50, 0.999, 120)))
    best, best_t = -1.0, 0.5
    for t in cand:
        s = f1_score(y_true, (np.asarray(y_prob) >= t).astype(int), zero_division=0)
        if s > best:
            best, best_t = s, float(t)
    return best_t


def metrics_frame(results: dict[str, dict]) -> pd.DataFrame:
    """Model-name -> metric dict, as a tidy comparison table."""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)

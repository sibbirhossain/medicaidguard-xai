"""Leakage guards, run as assertions inside training and as pytest cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import LEAKAGE_COLUMNS


class LeakageError(AssertionError):
    pass


def assert_no_leakage_columns(features: pd.DataFrame) -> None:
    bad = set(features.columns) & set(LEAKAGE_COLUMNS) - {"claim_id"}
    if bad:
        raise LeakageError(f"label/provenance columns present in features: {sorted(bad)}")


def assert_no_perfect_predictor(features: pd.DataFrame, y: pd.Series,
                                auc_threshold: float = 0.999) -> list[str]:
    """Flag single columns that alone separate the classes almost perfectly.

    A feature at AUC ~1.0 is nearly always a leak, not a discovery. Returns the
    offending column names so the caller can decide (some rule-derived flags are
    legitimately strong but not perfect).
    """
    from sklearn.metrics import roc_auc_score

    suspects = []
    yv = np.asarray(y)
    if len(np.unique(yv)) < 2:
        return suspects
    for c in features.select_dtypes(include=[np.number]).columns:
        v = features[c].to_numpy()
        if not np.isfinite(v).all() or np.nanstd(v) == 0:
            continue
        try:
            a = roc_auc_score(yv, v)
        except ValueError:
            continue
        if max(a, 1 - a) >= auc_threshold:
            suspects.append(c)
    return suspects


def assert_temporal_order(claims: pd.DataFrame, split: pd.Series) -> None:
    """Every training claim must precede every test claim."""
    d = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    tr = d.loc[split.to_numpy() == "train", "service_date"]
    te = d.loc[split.to_numpy() == "test", "service_date"]
    if len(tr) and len(te) and tr.max() > te.min():
        raise LeakageError("training window overlaps the test window")

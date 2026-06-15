"""Leakage is the failure mode that most often invalidates fraud-detection
results, so it gets its own test module rather than a single assertion."""

import numpy as np
import pandas as pd
import pytest

from medicaidguard.data.schema import LEAKAGE_COLUMNS
from medicaidguard.evaluation.leakage import (
    LeakageError,
    assert_no_leakage_columns,
    assert_no_perfect_predictor,
    assert_temporal_order,
)
from medicaidguard.evaluation.splits import time_split
from medicaidguard.features.pipeline import FeaturePipeline


def test_label_columns_never_reach_features(tables, settings):
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    p = FeaturePipeline()
    feats = p.fit_transform(claims, tables["evv_events"], tables["patients"],
                            pd.Series((split == "train").to_numpy(), index=claims.index))
    X = feats.drop(columns=["claim_id"])
    assert_no_leakage_columns(X)
    assert not (set(X.columns) & (set(LEAKAGE_COLUMNS) - {"claim_id"}))


def test_guard_rejects_a_planted_label():
    X = pd.DataFrame({"a": [1.0, 2.0], "is_suspicious": [0, 1]})
    with pytest.raises(LeakageError):
        assert_no_leakage_columns(X)


def test_no_single_feature_is_a_perfect_predictor(tables, settings):
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    p = FeaturePipeline()
    feats = p.fit_transform(claims, tables["evv_events"], tables["patients"],
                            pd.Series((split == "train").to_numpy(), index=claims.index))
    y = (tables["scenario_labels"].set_index("claim_id")["is_suspicious"]
         .reindex(claims["claim_id"]).fillna(0).astype(int).to_numpy())
    tr = (split == "train").to_numpy()
    suspects = assert_no_perfect_predictor(
        feats.drop(columns=["claim_id"]).loc[tr], y[tr])
    assert suspects == [], f"suspiciously perfect features: {suspects}"


def test_temporal_order_holds(tables, settings):
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    assert_temporal_order(claims, time_split(claims, settings.split))


def test_split_is_strictly_chronological(tables, settings):
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    tr_max = claims.loc[(split == "train").to_numpy(), "service_date"].max()
    va_min = claims.loc[(split == "valid").to_numpy(), "service_date"].min()
    te_min = claims.loc[(split == "test").to_numpy(), "service_date"].min()
    assert tr_max < va_min <= te_min


def test_peer_statistics_are_fitted_on_train_only(tables, settings):
    """Refitting on a truncated dataset must not change training-row features."""
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    mask = pd.Series((split == "train").to_numpy(), index=claims.index)

    p = FeaturePipeline()
    p.fit(claims, tables["evv_events"], tables["patients"], mask)
    p.transform(claims, tables["evv_events"], tables["patients"])

    # Peer medians learned from train only must be identical whether or not the
    # future rows were present at fit time.
    p2 = FeaturePipeline()
    tr_claims = claims.loc[mask.to_numpy()]
    p2.fit(tr_claims, tables["evv_events"], tables["patients"],
           pd.Series(True, index=range(len(tr_claims))))
    for col in p.peer_.stats_:
        a = p.peer_.stats_[col]["median"]
        b = p2.peer_.stats_[col]["median"]
        common = a.index.intersection(b.index)
        assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(),
                           equal_nan=True), f"peer median for {col} used future data"

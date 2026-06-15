import numpy as np
import pandas as pd

from medicaidguard.evaluation.experiments import run_pipeline
from medicaidguard.evaluation.metrics import (
    evaluate,
    expected_calibration_error,
    precision_at_k,
    recall_at_k,
)
from medicaidguard.models.calibration import Calibrator
from medicaidguard.models.fusion import HybridRiskModel, select_final_model
from medicaidguard.models.rules import apply_rules, rule_score


def test_precision_at_k_is_correct():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert precision_at_k(y, s, 0.2) == 0.5     # top 2: one positive
    assert recall_at_k(y, s, 0.3) == 1.0        # top 3 contains both positives


def test_accuracy_baseline_is_reported():
    y = np.zeros(100, dtype=int)
    y[:3] = 1
    s = np.full(100, 0.01)
    r = evaluate(y, s, threshold=0.5)
    assert r["accuracy"] == 0.97
    assert r["accuracy_trivial_baseline"] == 0.97
    # The point of the pairing: a useless model matches the trivial baseline.


def test_calibration_improves_ece():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 4000)
    raw = np.clip(y * 0.5 + rng.random(4000) * 0.5 + 0.25, 0, 1)  # badly scaled
    cal = Calibrator("isotonic").fit(raw[:2000], y[:2000])
    before = expected_calibration_error(y[2000:], raw[2000:])
    after = expected_calibration_error(y[2000:], cal.transform(raw[2000:]))
    assert after < before


def test_rule_score_bounded_and_monotone(tables):
    from medicaidguard.features.pipeline import FeaturePipeline
    p = FeaturePipeline()
    mask = pd.Series(True, index=range(len(tables["claims"])))
    X = p.fit_transform(tables["claims"], tables["evv_events"],
                        tables["patients"], mask).drop(columns=["claim_id"])
    s = rule_score(X)
    assert s.min() >= 0 and s.max() <= 100
    sev = apply_rules(X)
    assert ((sev >= 0) & (sev <= 1)).all().all()


def test_priority_bands_cover_full_range():
    bands = HybridRiskModel.priority_band(np.array([0.0, 39.9, 40.0, 64.9, 85.0, 100.0]))
    assert list(bands) == ["low", "low", "medium", "medium", "critical", "critical"]
    assert not pd.isna(bands).any()


def test_selection_ignores_accuracy_and_respects_calibration():
    from medicaidguard.config import SelectionConfig
    table = pd.DataFrame({
        "pr_auc": [0.80, 0.50, 0.90],
        "precision_at_5pct": [0.60, 0.30, 0.70],
        "brier": [0.02, 0.02, 0.50],      # third model is badly uncalibrated
        "accuracy": [0.97, 0.99, 0.98],   # second is the trivial classifier
        "accuracy_trivial_baseline": [0.99, 0.99, 0.99],
    }, index=["good", "trivial", "uncalibrated"])
    winner, out = select_final_model(table, SelectionConfig())
    assert winner == "good"
    assert not out.loc["uncalibrated", "calibration_ok"]


def test_pipeline_beats_trivial_and_random(tables, settings):
    art = run_pipeline(tables, settings, verbose=False,
                       models_subset=("lightgbm", "logistic_regression"),
                       skip_unsupervised_lof=True)
    best = art.results.loc[art.winner]
    assert best["pr_auc"] > best["prevalence"] * 3, "model barely beats random ranking"
    assert best["recall_at_5pct"] > best["prevalence"]


def test_hybrid_fusion_strategies_both_produce_scores(tables, settings):
    art = run_pipeline(tables, settings, verbose=False,
                       models_subset=("lightgbm",), skip_unsupervised_lof=True)
    for name in ("hybrid_weighted", "hybrid_stacked"):
        s = art.scores[name]["test"]
        assert np.isfinite(s).all() and s.min() >= 0

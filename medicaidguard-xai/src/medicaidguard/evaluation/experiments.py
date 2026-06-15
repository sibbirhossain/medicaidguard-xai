"""End-to-end training and the experiment suite.

`run_pipeline` is the single path used by scripts/train.py, the notebooks and
the tests, so the numbers in the report, the dashboard and the API can never
drift apart.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HELDOUT_SCENARIOS, RESULTS_DIR, Settings, ensure_dirs
from ..data.schema import LABEL_COLUMN
from ..features.pipeline import BLOCKS, FeaturePipeline
from ..models.anomaly import IsolationForestScorer, LOFScorer
from ..models.calibration import Calibrator
from ..models.fusion import HybridRiskModel, evv_inconsistency_score, select_final_model
from ..models.rules import rule_score
from ..models.supervised import model_zoo
from .confidence_intervals import (
    bootstrap_metric,
    cluster_bootstrap_metric,
    delong_roc_test,
    mcnemar_test,
    paired_bootstrap_difference,
)
from .leakage import (
    assert_no_leakage_columns,
    assert_no_perfect_predictor,
    assert_temporal_order,
)
from .metrics import best_f1_threshold, evaluate, metrics_frame, precision_at_k
from .splits import time_split


@dataclass
class RunArtifacts:
    features: pd.DataFrame
    labels: pd.Series
    split: pd.Series
    claims: pd.DataFrame
    models: dict
    calibrators: dict
    scores: dict
    results: pd.DataFrame
    winner: str
    fusion: HybridRiskModel
    timings: dict
    leakage_report: dict


def _prepare(tables: dict, settings: Settings, blocks=BLOCKS):
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    assert_temporal_order(claims, split)

    labels = (tables["scenario_labels"].set_index("claim_id")[LABEL_COLUMN]
              .reindex(claims["claim_id"]).fillna(0).astype(int).reset_index(drop=True))

    pipe = FeaturePipeline(blocks=blocks)
    train_mask = pd.Series((split == "train").to_numpy(), index=claims.index)
    pipe.fit(claims, tables["evv_events"], tables["patients"], train_mask)
    feats = pipe.transform(claims, tables["evv_events"], tables["patients"])
    return claims, split, labels, feats, pipe


def run_pipeline(tables: dict, settings: Settings | None = None,
                 blocks=BLOCKS, seed: int | None = None,
                 verbose: bool = True,
                 models_subset: tuple[str, ...] | None = None,
                 skip_unsupervised_lof: bool = False) -> RunArtifacts:
    """Train every model and return the full comparison.

    `models_subset` restricts the supervised zoo. Experiments 2 and 6 re-run the
    whole pipeline many times, and repeating the slow learners there would add
    hours without changing the conclusion those experiments are testing (which
    *features* and which *prevalence*, not which *learner*). The restriction is
    recorded in the results so the report can state it.
    """
    settings = settings or Settings()
    seed = settings.seed if seed is None else seed
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    claims, split, labels, feats, pipe = _prepare(tables, settings, blocks)
    timings["feature_engineering_s"] = time.perf_counter() - t0

    X = feats.drop(columns=["claim_id"])
    assert_no_leakage_columns(X)

    tr = (split == "train").to_numpy()
    va = (split == "valid").to_numpy()
    te = (split == "test").to_numpy()
    y = labels.to_numpy()

    leakage_report = {
        "perfect_predictors_train": assert_no_perfect_predictor(X.loc[tr], y[tr]),
        "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
        "train_prevalence": float(y[tr].mean()),
        "test_prevalence": float(y[te].mean()),
        "train_end": settings.split.train_end, "valid_end": settings.split.valid_end,
    }

    k = settings.selection.review_capacity_pcts
    scores: dict[str, dict[str, np.ndarray]] = {}
    models: dict = {}
    calibrators: dict = {}

    # --- Model A: rules --------------------------------------------------
    t0 = time.perf_counter()
    rs = rule_score(X) / 100.0
    # The noisy-OR rule score is a severity ranking, not a probability. It is
    # calibrated on validation exactly like the learned models so that the
    # Brier/ECE comparison is fair and the baseline is not disqualified for a
    # reason that has nothing to do with its ranking quality.
    cal_rules = Calibrator("isotonic").fit(rs.to_numpy()[va], y[va])
    scores["rules_only"] = {"valid": cal_rules.transform(rs.to_numpy()[va]),
                            "test": cal_rules.transform(rs.to_numpy()[te])}
    calibrators["rules_only"] = cal_rules
    timings["rules_s"] = time.perf_counter() - t0

    # --- Model C: unsupervised (fit on training rows only, unlabelled) ---
    t0 = time.perf_counter()
    iso = IsolationForestScorer(seed=seed).fit(X.loc[tr])
    iso_all = iso.score(X)
    scores["isolation_forest"] = {"valid": iso_all[va], "test": iso_all[te]}
    models["isolation_forest"] = iso
    timings["isolation_forest_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        if skip_unsupervised_lof:
            raise RuntimeError("LOF disabled for this run")
        tr_idx = np.flatnonzero(tr)
        if len(tr_idx) > 12_000:
            tr_idx = np.random.default_rng(seed).choice(tr_idx, 12_000, replace=False)
        lof = LOFScorer().fit(X.iloc[tr_idx])
        lof_all = lof.score(X)
        scores["local_outlier_factor"] = {"valid": lof_all[va], "test": lof_all[te]}
        models["local_outlier_factor"] = lof
    except Exception as exc:  # LOF is memory-hungry; degrade rather than fail
        if verbose:
            print(f"  LOF skipped: {exc}")
        lof_all = None
    timings["lof_s"] = time.perf_counter() - t0

    # --- Model B: supervised --------------------------------------------
    zoo = model_zoo(seed=seed)
    if models_subset:
        zoo = {k: v for k, v in zoo.items() if k in models_subset}
    for name, spec in zoo.items():
        t0 = time.perf_counter()
        clf = spec.build()
        clf.fit(X.loc[tr], y[tr])
        timings[f"{name}_train_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        p_va = clf.predict_proba(X.loc[va])[:, 1]
        p_te = clf.predict_proba(X.loc[te])[:, 1]
        timings[f"{name}_inference_s"] = time.perf_counter() - t0

        # Calibrate on validation, apply to test. Never fit on test.
        cal = Calibrator("isotonic").fit(p_va, y[va])
        scores[name] = {"valid": cal.transform(p_va), "test": cal.transform(p_te),
                        "valid_raw": p_va, "test_raw": p_te}
        models[name] = clf
        calibrators[name] = cal
        if verbose:
            print(f"  trained {name} in {timings[f'{name}_train_s']:.1f}s")

    # --- Model D: hybrid fusion -----------------------------------------
    best_supervised = max(
        (n for n in scores if n in zoo),
        key=lambda n: precision_at_k(y[va], scores[n]["valid"], 0.05),
    )
    evv_all = evv_inconsistency_score(X)
    comp = {
        "valid": {
            "supervised": scores[best_supervised]["valid"],
            "rules": rs.to_numpy()[va],
            "anomaly": iso_all[va],
            "evv_inconsistency": evv_all[va],
        },
        "test": {
            "supervised": scores[best_supervised]["test"],
            "rules": rs.to_numpy()[te],
            "anomaly": iso_all[te],
            "evv_inconsistency": evv_all[te],
        },
    }

    fusion_w = HybridRiskModel(strategy="weighted").fit(comp["valid"])
    fusion_s = HybridRiskModel(strategy="stacked").fit(comp["valid"], y[va])
    scores["hybrid_weighted"] = {"valid": fusion_w.raw_score(comp["valid"]),
                                 "test": fusion_w.raw_score(comp["test"])}
    # The stacked meta-learner is fitted with class_weight="balanced", which
    # deliberately distorts its output scale. Calibrate before reporting.
    st_va = fusion_s.raw_score(comp["valid"])
    st_te = fusion_s.raw_score(comp["test"])
    cal_st = Calibrator("isotonic").fit(st_va, y[va])
    scores["hybrid_stacked"] = {"valid": cal_st.transform(st_va),
                                "test": cal_st.transform(st_te)}
    calibrators["hybrid_stacked"] = cal_st

    # The weighted score is a ranking, not a probability, so calibrate it too
    # before it is allowed anywhere near the priority bands.
    cal_w = Calibrator("isotonic").fit(scores["hybrid_weighted"]["valid"], y[va])
    scores["hybrid_weighted"]["test"] = cal_w.transform(scores["hybrid_weighted"]["test"])
    scores["hybrid_weighted"]["valid"] = cal_w.transform(scores["hybrid_weighted"]["valid"])
    calibrators["hybrid_weighted"] = cal_w

    # --- evaluation -------------------------------------------------------
    results = {}
    for name, s in scores.items():
        thr = best_f1_threshold(y[va], s["valid"])
        # Unsupervised scores are min-max ranks, not probabilities, so
        # calibration metrics are reported as NaN rather than faked.
        is_prob = name not in {"isolation_forest", "local_outlier_factor"}
        results[name] = evaluate(y[te], s["test"], k_pcts=k, threshold=thr,
                                 is_probability=is_prob)
        results[name]["model"] = name
    table = metrics_frame(results)

    winner, table = select_final_model(table, settings.selection)
    fusion = fusion_s if winner == "hybrid_stacked" else fusion_w

    return RunArtifacts(
        features=feats, labels=labels, split=split, claims=claims,
        models=models, calibrators=calibrators, scores=scores,
        results=table, winner=winner, fusion=fusion, timings=timings,
        leakage_report=leakage_report,
    )


# --- experiment suite -----------------------------------------------------

def experiment_ablation(tables, settings, seed=None) -> pd.DataFrame:
    """Experiment 2: which feature families actually earn their place."""
    configs = {
        "billing_only": ("billing",),
        "evv_only": ("evv",),
        "billing_evv": ("billing", "evv"),
        "billing_evv_temporal": ("billing", "evv", "temporal"),
        "billing_evv_temporal_stat": ("billing", "evv", "temporal", "statistical"),
        "full": BLOCKS,
    }
    rows = []
    for name, blocks in configs.items():
        art = run_pipeline(tables, settings, blocks=blocks, seed=seed, verbose=False,
                           models_subset=("lightgbm", "logistic_regression"),
                           skip_unsupervised_lof=True)
        r = art.results.loc[art.winner].to_dict()
        r["feature_set"] = name
        r["n_features"] = art.features.shape[1] - 1
        r["selected_model"] = art.winner
        rows.append(r)
    return pd.DataFrame(rows).set_index("feature_set")


def experiment_scenario_generalization(tables, settings, seed=None) -> pd.DataFrame:
    """Experiment 5: train with some scenarios masked, test on all.

    Held-out scenario rows are relabelled 0 in training only, simulating a
    scheme that has never been investigated and therefore has no label.
    """
    masked = {k: v.copy() for k, v in tables.items()}
    lab = masked["scenario_labels"]
    claims = masked["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    train_ids = set(claims.loc[(split == "train").to_numpy(), "claim_id"])
    hide = lab["scenario"].isin(HELDOUT_SCENARIOS) & lab["claim_id"].isin(train_ids)
    lab.loc[hide, "is_suspicious"] = 0

    art = run_pipeline(masked, settings, seed=seed, verbose=False)
    # Score the held-out scenarios specifically, using true labels.
    true_lab = tables["scenario_labels"].set_index("claim_id")
    te = (art.split == "test").to_numpy()
    test_ids = art.claims.loc[te, "claim_id"]
    scen = true_lab.reindex(test_ids)["scenario"].to_numpy()
    y_true = true_lab.reindex(test_ids)["is_suspicious"].fillna(0).astype(int).to_numpy()
    s = art.scores[art.winner]["test"]

    rows = []
    for name in set(scen) - {"none"}:
        m = (scen == name) | (y_true == 0)
        if (y_true[m] == 1).sum() < 5:
            continue
        rows.append({
            "scenario": name,
            "held_out_in_training": name in HELDOUT_SCENARIOS,
            "n_positive": int((scen == name).sum()),
            "recall_at_5pct": float(
                np.mean(s[(scen == name)] >= np.quantile(s, 0.95))),
            "median_score_rank_pct": float(
                100 * (1 - np.mean(s[:, None] >= s[(scen == name)][None, :]))),
        })
    return pd.DataFrame(rows).sort_values("recall_at_5pct", ascending=False)


def experiment_prevalence(tables_fn, settings, prevalences=(0.01, 0.02, 0.035, 0.06),
                          seed=None) -> pd.DataFrame:
    """Experiment 6: how performance moves with the suspicious-activity rate."""
    rows = []
    for p in prevalences:
        tb = tables_fn(p)
        art = run_pipeline(tb, settings, seed=seed, verbose=False,
                           models_subset=("lightgbm", "logistic_regression"),
                           skip_unsupervised_lof=True)
        r = art.results.loc[art.winner].to_dict()
        r["prevalence_setting"] = p
        r["selected_model"] = art.winner
        rows.append(r)
    return pd.DataFrame(rows).set_index("prevalence_setting")


def statistical_comparison(art: RunArtifacts, baseline: str = "rules_only") -> pd.DataFrame:
    """Experiment 1 significance layer: winner vs each competitor."""
    te = (art.split == "test").to_numpy()
    y = art.labels.to_numpy()[te]
    groups = art.claims.loc[te, "caregiver_id"].to_numpy()
    win = art.scores[art.winner]["test"]

    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for name, s in art.scores.items():
        if name == art.winner:
            continue
        other = s["test"]
        iv, p_boot = paired_bootstrap_difference(
            y, win, other, average_precision_score, n_boot=400, seed=7)
        auc_a, auc_b, z, p_delong = delong_roc_test(y, win, other)
        n01, n10, p_mc = mcnemar_test(
            y, (win >= np.quantile(win, 0.95)).astype(int),
            (other >= np.quantile(other, 0.95)).astype(int))
        rows.append({
            "comparison": f"{art.winner} vs {name}",
            "pr_auc_diff": iv.point, "pr_auc_ci_lo": iv.lo, "pr_auc_ci_hi": iv.hi,
            "paired_bootstrap_p": p_boot,
            "roc_auc_winner": auc_a, "roc_auc_other": auc_b, "delong_p": p_delong,
            "mcnemar_discordant": f"{n01}/{n10}", "mcnemar_p": p_mc,
        })
    df = pd.DataFrame(rows)

    ci = cluster_bootstrap_metric(y, win, groups, average_precision_score,
                                  n_boot=300, seed=11)
    df.attrs["winner_pr_auc_cluster_ci"] = ci.as_dict()
    df.attrs["winner_roc_auc_ci"] = bootstrap_metric(
        y, win, roc_auc_score, n_boot=500, seed=11).as_dict()
    return df


def save_results(art: RunArtifacts, extra: dict | None = None, out_dir: Path | None = None):
    ensure_dirs()
    out_dir = out_dir or RESULTS_DIR
    art.results.to_csv(out_dir / "model_comparison.csv")
    payload = {
        "winner": art.winner,
        "timings": art.timings,
        "leakage_report": art.leakage_report,
        "results": json.loads(art.results.to_json(orient="index")),
    }
    if extra:
        payload.update(extra)
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    return out_dir

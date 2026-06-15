#!/usr/bin/env python3
"""Train every model, select the final one, and write the results tables."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from medicaidguard.config import (  # noqa: E402
    ARTIFACTS_DIR,
    RESULTS_DIR,
    SYNTHETIC_DIR,
    ensure_dirs,
    load_settings,
)
from medicaidguard.evaluation.experiments import (  # noqa: E402
    run_pipeline,
    save_results,
    statistical_comparison,
)


def load_tables(path=SYNTHETIC_DIR):
    names = ["regions", "providers", "caregivers", "patients",
             "authorizations", "claims", "evv_events", "scenario_labels"]
    missing = [n for n in names if not (Path(path) / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing tables {missing}; run scripts/generate_data.py first")
    return {n: pd.read_parquet(Path(path) / f"{n}.parquet") for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-stats", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    s = load_settings(args.config)
    tables = load_tables()

    art = run_pipeline(tables, s)
    print("\n=== model comparison (held-out test window) ===")
    cols = ["pr_auc", "roc_auc", "precision_at_5pct", "recall_at_5pct",
            "accuracy", "accuracy_trivial_baseline", "brier", "ece", "selected"]
    print(art.results[cols].round(4).to_string())
    print(f"\nselected model: {art.winner}")

    extra = {"settings": s.to_dict()}
    if not args.skip_stats:
        stats = statistical_comparison(art)
        stats.to_csv(RESULTS_DIR / "statistical_comparison.csv", index=False)
        extra["winner_pr_auc_cluster_ci"] = stats.attrs["winner_pr_auc_cluster_ci"]
        extra["winner_roc_auc_ci"] = stats.attrs["winner_roc_auc_ci"]
        print("\n=== winner vs competitors ===")
        print(stats[["comparison", "pr_auc_diff", "pr_auc_ci_lo",
                     "pr_auc_ci_hi", "delong_p"]].round(4).to_string(index=False))

    save_results(art, extra)

    import joblib
    joblib.dump({"winner": art.winner,
                 "model": art.models.get(art.winner),
                 "calibrator": art.calibrators.get(art.winner),
                 "columns": [c for c in art.features.columns if c != "claim_id"]},
                ARTIFACTS_DIR / "final_model.joblib")
    print(f"\nartifacts -> {ARTIFACTS_DIR}\nresults   -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()

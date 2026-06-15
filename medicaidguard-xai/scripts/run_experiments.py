#!/usr/bin/env python3
"""Run the full experiment suite and export every table used by the report."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from train import load_tables  # noqa: E402

from medicaidguard.config import TABLES_DIR, ensure_dirs, load_settings  # noqa: E402
from medicaidguard.data.generator import SyntheticMedicaidEVVGenerator  # noqa: E402
from medicaidguard.evaluation.experiments import (  # noqa: E402
    experiment_ablation,
    experiment_prevalence,
    experiment_scenario_generalization,
    run_pipeline,
    statistical_comparison,
)
from medicaidguard.impact.simulator import SimulationInputs, capacity_curve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    ensure_dirs()
    s = load_settings(args.config)
    tables = load_tables()
    done = {}

    t0 = time.perf_counter()
    art = run_pipeline(tables, s)
    done["exp1_baseline_comparison"] = art.results
    art.results.to_csv(TABLES_DIR / "exp1_baseline_comparison.csv")

    stats = statistical_comparison(art)
    stats.to_csv(TABLES_DIR / "exp1_statistical_comparison.csv", index=False)

    if "ablation" not in args.skip:
        ab = experiment_ablation(tables, s)
        ab.to_csv(TABLES_DIR / "exp2_feature_ablation.csv")
        done["exp2_feature_ablation"] = ab

    if "scenario" not in args.skip:
        sc = experiment_scenario_generalization(tables, s)
        sc.to_csv(TABLES_DIR / "exp5_scenario_generalization.csv", index=False)
        done["exp5_scenario_generalization"] = sc

    if "prevalence" not in args.skip:
        def make(p):
            cfg = s.generator
            cfg.suspicious_prevalence = p
            return SyntheticMedicaidEVVGenerator(cfg).generate()
        pv = experiment_prevalence(make, s)
        pv.to_csv(TABLES_DIR / "exp6_prevalence.csv")
        done["exp6_prevalence"] = pv

    # Experiment 7: operational thresholds + workload.
    te = (art.split == "test").to_numpy()
    y = art.labels.to_numpy()[te]
    sc_win = art.scores[art.winner]["test"]
    cap = capacity_curve(y, sc_win, inp=SimulationInputs())
    cap.to_csv(TABLES_DIR / "exp7_operational_thresholds.csv", index=False)
    done["exp7_operational_thresholds"] = cap

    # Experiment 8: fairness diagnostics across synthetic cohorts.
    fair = fairness_table(art, tables)
    fair.to_csv(TABLES_DIR / "exp8_fairness.csv", index=False)
    done["exp8_fairness"] = fair

    # Experiment 10: computational efficiency.
    eff = pd.DataFrame([art.timings]).T.rename(columns={0: "seconds"})
    eff.to_csv(TABLES_DIR / "exp10_efficiency.csv")
    done["exp10_efficiency"] = eff

    print(f"\nsuite finished in {time.perf_counter() - t0:.0f}s; "
          f"{len(done)} tables -> {TABLES_DIR}")
    for k, v in done.items():
        print(f"  {k}: {getattr(v, 'shape', None)}")


def fairness_table(art, tables) -> pd.DataFrame:
    from medicaidguard.evaluation.metrics import evaluate

    te = (art.split == "test").to_numpy()
    claims = art.claims.loc[te]
    y = art.labels.to_numpy()[te]
    s = art.scores[art.winner]["test"]
    pts = tables["patients"].set_index("patient_id")

    rows = []
    for col in ("demographic_group", "service_region"):
        g = pts.reindex(claims["patient_id"])[col].to_numpy()
        for grp in pd.unique(g):
            m = g == grp
            if m.sum() < 200 or len(np.unique(y[m])) < 2:
                continue
            r = evaluate(y[m], s[m], threshold=float(np.quantile(s, 0.95)))
            rows.append({"attribute": col, "group": str(grp), "n": int(m.sum()),
                         "prevalence": r["prevalence"], "pr_auc": r["pr_auc"],
                         "precision": r["precision"], "recall": r["recall"],
                         "fpr": r["false_positive_rate"],
                         "fnr": r["false_negative_rate"], "ece": r["ece"]})
    df = pd.DataFrame(rows)
    df.attrs["note"] = ("Synthetic cohort labels. These diagnostics validate the "
                        "measurement code; they say nothing about real populations.")
    return df


if __name__ == "__main__":
    main()

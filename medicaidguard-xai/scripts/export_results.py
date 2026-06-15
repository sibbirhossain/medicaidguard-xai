#!/usr/bin/env python3
"""Render the figures used in the README and the research report."""

from __future__ import annotations  # noqa: I001

import sys
import warnings
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from train import load_tables  # noqa: E402

from medicaidguard.config import FIGURES_DIR, TABLES_DIR, ensure_dirs, load_settings  # noqa: E402
from medicaidguard.evaluation.experiments import run_pipeline  # noqa: E402


def main():
    ensure_dirs()
    s = load_settings()
    art = run_pipeline(load_tables(), s, verbose=False)
    te = (art.split == "test").to_numpy()
    y = art.labels.to_numpy()[te]
    sc = art.scores[art.winner]["test"]

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    p, rc, _ = precision_recall_curve(y, sc)
    ax[0].plot(rc, p, lw=2)
    ax[0].axhline(y.mean(), ls="--", c="grey", label=f"random = {y.mean():.3f}")
    ax[0].set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    ax[0].legend()
    fpr, tpr, _ = roc_curve(y, sc)
    ax[1].plot(fpr, tpr, lw=2)
    ax[1].plot([0, 1], [0, 1], ls="--", c="grey")
    ax[1].set(xlabel="FPR", ylabel="TPR", title="ROC")
    frac, mp = calibration_curve(y, sc, n_bins=10, strategy="quantile")
    ax[2].plot(mp, frac, "o-")
    ax[2].plot([0, 1], [0, 1], ls="--", c="grey")
    ax[2].set(xlabel="predicted", ylabel="observed", title="Calibration")
    fig.suptitle(f"{art.winner} — held-out test window (synthetic data, simulated labels)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_curves.png", dpi=140)

    # Accuracy vs its trivial baseline: the figure that makes the argument.
    r = art.results
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(r))
    ax.bar(idx - 0.2, r["accuracy"], 0.4, label="accuracy")
    ax.bar(idx + 0.2, r["pr_auc"], 0.4, label="PR-AUC")
    ax.axhline(r["accuracy_trivial_baseline"].iloc[0], ls="--", c="red",
               label="trivial all-negative accuracy")
    ax.set_xticks(idx)
    ax.set_xticklabels(r.index, rotation=40, ha="right")
    ax.set_title("Accuracy is uninformative at this prevalence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_vs_prauc.png", dpi=140)

    f = TABLES_DIR / "exp2_feature_ablation.csv"
    if f.exists():
        ab = pd.read_csv(f, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(len(ab)), ab["pr_auc"], "o-", lw=2)
        ax.set_xticks(range(len(ab)))
        ax.set_xticklabels(ab.index, rotation=30, ha="right")
        ax.set(ylabel="PR-AUC", title="Feature ablation")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_ablation.png", dpi=140)

    print(f"figures -> {FIGURES_DIR}")
    for p_ in sorted(FIGURES_DIR.glob("*.png")):
        print(" ", p_.name)


if __name__ == "__main__":
    main()

"""Model B: supervised learners, with calibration and class-imbalance handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LGBM = False


@dataclass
class ModelSpec:
    name: str
    build: Any
    needs_scaling: bool = False
    notes: str = ""
    param_grid: dict = field(default_factory=dict)


def model_zoo(seed: int = 0, n_jobs: int = 1) -> dict[str, ModelSpec]:
    """The candidate set compared in Experiment 1.

    `class_weight="balanced"` is preferred over resampling: it changes the loss
    without duplicating rows, so it cannot leak synthetic near-copies of a
    positive claim across a CV fold boundary.
    """
    zoo = {
        "logistic_regression": ModelSpec(
            "logistic_regression",
            lambda: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                           C=0.5, random_state=seed)),
            ]),
            needs_scaling=True,
            notes="Transparent linear baseline; coefficients are directly readable.",
        ),
        "random_forest": ModelSpec(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=400, min_samples_leaf=4, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed),
            notes="Strong tabular baseline, robust to monotone feature transforms.",
        ),
        "gradient_boosting": ModelSpec(
            "gradient_boosting",
            lambda: GradientBoostingClassifier(
                n_estimators=250, learning_rate=0.08, max_depth=3,
                subsample=0.85, random_state=seed),
            notes="Sequential boosting; no native class weighting, relies on calibration.",
        ),
    }
    if HAS_LGBM:
        zoo["lightgbm"] = ModelSpec(
            "lightgbm",
            lambda: LGBMClassifier(
                n_estimators=600, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.85, subsample_freq=1,
                colsample_bytree=0.8, class_weight="balanced",
                n_jobs=n_jobs, random_state=seed, verbose=-1),
            notes="MIT-licensed gradient boosting; fastest of the tree models here.",
        )
    return zoo

"""Model D: hybrid risk fusion, and the final model selection rule.

Two fusion strategies are implemented and compared rather than assumed:

* `weighted`  - a fixed, auditable convex combination of the normalised
                component scores. Fully transparent; every weight is visible in
                the configuration and the contribution of each component to a
                given alert can be printed.
* `stacked`   - a logistic meta-learner fitted on the validation window, taking
                the component scores as inputs. Usually more accurate, less
                transparent, and it needs a calibration set of its own.

The selection rule is in `select_final_model`. It is deliberately not accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PRIORITY_BANDS = ((0, 40, "low"), (40, 65, "medium"), (65, 85, "high"), (85, 101, "critical"))

DEFAULT_WEIGHTS = {
    "supervised": 0.50,
    "rules": 0.25,
    "anomaly": 0.15,
    "evv_inconsistency": 0.10,
}


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def evv_inconsistency_score(features: pd.DataFrame) -> np.ndarray:
    """A compact EVV-only score, used as a fusion component and reported alone
    in the ablation study."""
    g = lambda c, d=0.0: (features[c].to_numpy(dtype=float)  # noqa: E731
                          if c in features.columns else np.full(len(features), d))
    parts = [
        np.clip(np.abs(g("evv_billing_duration_diff")) / 4.0, 0, 1),
        1.0 - np.clip(g("evv_present", 1.0), 0, 1),
        np.clip(g("implied_speed_kmh") / 300.0, 0, 1),
        np.clip(g("overlap_minutes_with_prev") / 60.0, 0, 1),
        np.clip(g("checkin_distance_from_home_m") / 5000.0, 0, 1),
    ]
    return np.nanmean(np.vstack(parts), axis=0)


@dataclass
class HybridRiskModel:
    """Combines component scores into a 0-100 review-priority score."""

    strategy: str = "weighted"
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    meta: LogisticRegression | None = None
    scaler: StandardScaler | None = None

    COMPONENTS = ("supervised", "rules", "anomaly", "evv_inconsistency")

    def _matrix(self, components: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([np.asarray(components[c], dtype=float)
                                for c in self.COMPONENTS])

    def fit(self, components: dict[str, np.ndarray], y=None) -> HybridRiskModel:
        if self.strategy == "stacked":
            if y is None:
                raise ValueError("stacked fusion requires labels from the validation window")
            X = self._matrix(components)
            self.scaler = StandardScaler().fit(X)
            self.meta = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.meta.fit(self.scaler.transform(X), y)
        return self

    def raw_score(self, components: dict[str, np.ndarray]) -> np.ndarray:
        """Component fusion in [0, 1] before the 0-100 rescale."""
        if self.strategy == "stacked":
            if self.meta is None:
                raise RuntimeError("stacked fusion not fitted")
            return self.meta.predict_proba(self.scaler.transform(self._matrix(components)))[:, 1]
        total = sum(self.weights.get(c, 0.0) for c in self.COMPONENTS)
        acc = np.zeros(len(next(iter(components.values()))), dtype=float)
        for c in self.COMPONENTS:
            acc += self.weights.get(c, 0.0) * np.clip(np.asarray(components[c], float), 0, 1)
        return acc / total if total else acc

    def risk_score(self, components: dict[str, np.ndarray]) -> np.ndarray:
        """0-100 review priority. Not a probability - see `calibrated_probability`."""
        return 100.0 * _minmax(self.raw_score(components))

    @staticmethod
    def priority_band(score) -> np.ndarray:
        s = np.asarray(score, dtype=float)
        out = np.empty(len(s), dtype=object)
        for lo, hi, name in PRIORITY_BANDS:
            out[(s >= lo) & (s < hi)] = name
        return out


def select_final_model(results: pd.DataFrame, selection_cfg) -> tuple[str, pd.DataFrame]:
    """Pick the deployed model. Returns (winner_name, annotated table).

    Rule:
      1. Exclude models whose Brier score exceeds the calibration floor. A model
         that ranks well but reports 0.9 for a 0.3-risk claim cannot drive the
         priority bands or the impact simulator.
      2. Among the survivors, maximise PR-AUC.
      3. Break ties (within 0.005 PR-AUC) on Precision@5%, the operational
         metric that matches realistic investigator capacity.

    Accuracy is reported in the table but never used to select, because at this
    prevalence the trivial all-negative classifier outscores every useful model
    on it.
    """
    table = results.copy()
    primary = selection_cfg.primary_metric
    tiebreak = selection_cfg.tiebreak_metric

    table["calibration_ok"] = table["brier"] <= selection_cfg.max_brier
    table["beats_trivial_accuracy"] = table["accuracy"] > table["accuracy_trivial_baseline"]

    pool = table[table["calibration_ok"]] if table["calibration_ok"].any() else table
    top = pool[primary].max()
    near = pool[pool[primary] >= top - 0.005]
    winner = near.sort_values([tiebreak, primary], ascending=False).index[0]

    table["selected"] = table.index == winner
    return str(winner), table

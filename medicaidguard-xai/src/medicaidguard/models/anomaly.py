"""Model C: unsupervised anomaly detection.

These models never see a label. They answer a different question from the
supervised models: not "does this look like known suspicious activity" but
"does this look unlike anything else". That matters for novel schemes, which by
definition have no training labels, and it is the reason the unsupervised track
is kept even though it scores lower on the labelled benchmark.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IsolationForestScorer:
    name = "isolation_forest"

    def __init__(self, seed: int = 0, contamination: float = 0.05, n_jobs: int = 1):
        self.pipe = Pipeline([
            ("scale", StandardScaler()),
            ("iso", IsolationForest(n_estimators=300, contamination=contamination,
                                    random_state=seed, n_jobs=n_jobs)),
        ])

    def fit(self, X, y=None):
        self.pipe.fit(X)
        return self

    def score(self, X) -> np.ndarray:
        # score_samples: higher = more normal. Invert and min-max to [0, 1] so
        # the value is orientable with the supervised probabilities. This is a
        # ranking score, NOT a probability, and is never reported as one.
        raw = -self.pipe.named_steps["iso"].score_samples(
            self.pipe.named_steps["scale"].transform(X))
        lo, hi = np.min(raw), np.max(raw)
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)


class LOFScorer:
    name = "local_outlier_factor"

    def __init__(self, n_neighbors: int = 35, contamination: float = 0.05, n_jobs: int = 1):
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True,
                                      contamination=contamination, n_jobs=n_jobs)

    def fit(self, X, y=None):
        self.lof.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X) -> np.ndarray:
        raw = -self.lof.score_samples(self.scaler.transform(X))
        lo, hi = np.min(raw), np.max(raw)
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

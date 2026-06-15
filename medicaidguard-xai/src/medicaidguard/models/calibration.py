"""Probability calibration.

A raw classifier margin is not a probability. Calibration matters here for a
concrete operational reason: the impact simulator and the review-priority bands
both interpret the score as "expected share of alerts at this level that are
genuine". If the score is uncalibrated those numbers are meaningless.

Calibration is fitted on the validation window, never on training (which the
model has already memorised) and never on test (which would leak).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "platt", "none"}:
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method
        self.model = None

    def fit(self, p_raw, y):
        p_raw = np.asarray(p_raw, dtype=float).reshape(-1)
        y = np.asarray(y)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(p_raw, y)
        elif self.method == "platt":
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(p_raw.reshape(-1, 1), y)
        return self

    def transform(self, p_raw) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype=float).reshape(-1)
        if self.method == "none" or self.model is None:
            return np.clip(p_raw, 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p_raw), 0.0, 1.0)
        return self.model.predict_proba(p_raw.reshape(-1, 1))[:, 1]

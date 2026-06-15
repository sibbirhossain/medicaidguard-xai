"""SHAP-based global and local explanations.

Interpretation boundary
-----------------------
SHAP attributes a model's *output* to its inputs. It does not establish that a
feature caused the underlying behaviour, and it says nothing about whether the
behaviour was fraudulent. Every explanation surfaced to a reviewer therefore
carries the limitation string below, and the dashboard renders it verbatim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHAP_LIMITATION = (
    "Feature attributions describe how this model reached its score. They are "
    "not evidence of intent, causation, or wrongdoing, and they do not "
    "constitute a finding of fraud."
)


class ShapExplainer:
    def __init__(self, model, background: pd.DataFrame, max_background: int = 200,
                 seed: int = 0):
        import shap

        self.columns = list(background.columns)
        bg = background
        if len(bg) > max_background:
            bg = bg.sample(max_background, random_state=seed)
        self._shap = shap
        try:
            # TreeExplainer is exact and fast for the tree models in the zoo.
            self.explainer = shap.TreeExplainer(model)
            self.kind = "tree"
        except Exception:
            f = getattr(model, "predict_proba", model.predict)
            self.explainer = shap.Explainer(
                lambda d: np.asarray(f(pd.DataFrame(d, columns=self.columns)))[:, -1],
                np.asarray(bg))
            self.kind = "kernel"

    def _values(self, X: pd.DataFrame) -> np.ndarray:
        v = self.explainer.shap_values(X) if self.kind == "tree" else self.explainer(np.asarray(X)).values
        v = np.asarray(v)
        if v.ndim == 3:            # (n, features, classes)
            v = v[:, :, -1]
        return v

    def global_importance(self, X: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
        vals = self._values(X)
        imp = np.abs(vals).mean(axis=0)
        return (pd.DataFrame({"feature": self.columns, "mean_abs_shap": imp})
                .sort_values("mean_abs_shap", ascending=False)
                .head(top_n).reset_index(drop=True))

    def local_explanation(self, X: pd.DataFrame, row: int, top_n: int = 8) -> list[dict]:
        vals = self._values(X.iloc[[row]])[0]
        order = np.argsort(-np.abs(vals))[:top_n]
        return [{
            "feature": self.columns[i],
            "value": float(X.iloc[row, i]),
            "shap_value": float(vals[i]),
            "direction": "increases risk" if vals[i] > 0 else "decreases risk",
        } for i in order]


def permutation_importance_table(model, X, y, n_repeats: int = 5, seed: int = 0,
                                 scoring: str = "average_precision") -> pd.DataFrame:
    """Model-agnostic cross-check on SHAP. If the two disagree sharply, the
    explanation is not stable enough to show a reviewer."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=seed, scoring=scoring, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns,
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))

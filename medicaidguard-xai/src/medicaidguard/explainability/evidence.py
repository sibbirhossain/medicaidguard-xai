"""Alert assembly: one structured, reviewable explanation per flagged claim."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..models.rules import rule_evidence
from .shap_explainer import SHAP_LIMITATION

DISCLAIMER = ("Synthetic/public research demonstration. Alerts are review "
              "recommendations, not findings of fraud.")


@dataclass
class Alert:
    alert_id: str
    claim_id: str
    provider_id: str
    caregiver_id: str
    patient_id: str
    service_date: str
    risk_score: float
    review_priority: str
    model_probability: float
    calibration_note: str
    rule_evidence: list = field(default_factory=list)
    feature_evidence: list = field(default_factory=list)
    peer_comparison: dict = field(default_factory=dict)
    explanation_limitations: str = SHAP_LIMITATION
    disclaimer: str = DISCLAIMER
    review_status: str = "pending"
    investigator_notes: str = ""

    def to_dict(self):
        return asdict(self)

    def narrative(self) -> str:
        lines = [
            f"Alert ID: {self.alert_id}",
            f"Review Priority: {self.review_priority.title()}",
            f"Risk Score: {self.risk_score:.0f}/100",
            f"Calibrated probability: {self.model_probability:.3f} ({self.calibration_note})",
            "",
            "Evidence:",
        ]
        n = 0
        for e in self.rule_evidence:
            n += 1
            lines.append(f"  {n}. {e['statement']} (severity {e['severity']:.2f})")
        for e in self.feature_evidence[:4]:
            n += 1
            lines.append(f"  {n}. {e['feature']} = {e['value']:.3g} "
                         f"({e['direction']}, SHAP {e['shap_value']:+.3f})")
        lines += ["", self.explanation_limitations, self.disclaimer]
        return "\n".join(lines)


def build_alerts(claims: pd.DataFrame, features: pd.DataFrame, risk_score: np.ndarray,
                 probability: np.ndarray, priority: np.ndarray,
                 explainer=None, top_n: int | None = None,
                 calibration_note: str = "isotonic, fitted on validation window",
                 min_priority: tuple[str, ...] = ("high", "critical")) -> list[Alert]:
    """Assemble alerts for the claims worth a reviewer's time."""
    X = features.drop(columns=["claim_id"], errors="ignore")
    idx = np.flatnonzero(np.isin(priority, min_priority))
    if top_n:
        idx = idx[np.argsort(-np.asarray(risk_score)[idx])][:top_n]

    alerts = []
    for i in idx:
        c = claims.iloc[i]
        fe = explainer.local_explanation(X, int(i)) if explainer is not None else []
        alerts.append(Alert(
            alert_id=f"SYN-{int(i):06d}",
            claim_id=str(c["claim_id"]),
            provider_id=str(c["provider_id"]),
            caregiver_id=str(c["caregiver_id"]),
            patient_id=str(c["patient_id"]),
            service_date=str(pd.Timestamp(c["service_date"]).date()),
            risk_score=float(risk_score[i]),
            review_priority=str(priority[i]),
            model_probability=float(probability[i]),
            calibration_note=calibration_note,
            rule_evidence=rule_evidence(X, X.index[i]),
            feature_evidence=fe,
            peer_comparison={
                "billed_hours": float(c["billed_hours"]),
                "authorized_hours": float(c["authorized_hours"]),
                "peer_median_billed_hours": float(claims["billed_hours"].median()),
            },
        ))
    return alerts

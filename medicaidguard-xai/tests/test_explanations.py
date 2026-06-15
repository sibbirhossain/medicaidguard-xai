import numpy as np
import pandas as pd

from medicaidguard.explainability.evidence import DISCLAIMER, Alert, build_alerts
from medicaidguard.explainability.shap_explainer import SHAP_LIMITATION
from medicaidguard.models.rules import rule_evidence


def test_rule_evidence_is_structured_and_readable(tables):
    from medicaidguard.features.pipeline import FeaturePipeline
    p = FeaturePipeline()
    mask = pd.Series(True, index=range(len(tables["claims"])))
    X = p.fit_transform(tables["claims"], tables["evv_events"],
                        tables["patients"], mask).drop(columns=["claim_id"])
    found = 0
    for i in X.index[:400]:
        ev = rule_evidence(X, i)
        for e in ev:
            assert set(e) == {"rule", "severity", "weight", "statement"}
            assert 0 <= e["severity"] <= 1
            assert e["statement"].endswith(".")
            found += 1
    assert found > 0, "no rule fired on any of 400 claims — rules are inert"


def test_alert_narrative_carries_disclaimers():
    a = Alert(alert_id="SYN-000001", claim_id="C1", provider_id="P", caregiver_id="CG",
              patient_id="PT", service_date="2024-05-01", risk_score=87.0,
              review_priority="high", model_probability=0.42,
              calibration_note="isotonic",
              rule_evidence=[{"rule": "r", "severity": 0.8, "weight": 1.0,
                              "statement": "Billed hours exceed authorisation."}])
    text = a.narrative()
    assert "SYN-000001" in text and "87/100" in text
    assert DISCLAIMER in text
    assert SHAP_LIMITATION in text


def test_build_alerts_only_returns_requested_priorities(tables):
    from medicaidguard.features.pipeline import FeaturePipeline
    p = FeaturePipeline()
    mask = pd.Series(True, index=range(len(tables["claims"])))
    feats = p.fit_transform(tables["claims"], tables["evv_events"],
                            tables["patients"], mask)
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 100, len(claims))
    from medicaidguard.models.fusion import HybridRiskModel
    band = HybridRiskModel.priority_band(score)
    alerts = build_alerts(claims, feats, score, score / 100, band, top_n=20)
    assert len(alerts) <= 20
    assert all(a.review_priority in ("high", "critical") for a in alerts)

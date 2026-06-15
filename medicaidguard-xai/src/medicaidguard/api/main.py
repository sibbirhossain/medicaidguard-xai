"""FastAPI service for MedicaidGuard-XAI.

Deployment posture: this service scores synthetic research data and returns
review recommendations. It does not make eligibility or payment decisions, does
not deny claims, and exposes no write endpoint that would alter a claim's
status. Every scoring response carries the demonstration disclaimer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from ..config import ARTIFACTS_DIR, RESULTS_DIR, SYNTHETIC_DIR, load_settings
from ..features.pipeline import FeaturePipeline
from ..impact.simulator import SimulationInputs, simulate
from ..models.fusion import HybridRiskModel
from ..models.rules import rule_evidence, rule_score
from .schemas import (
    DISCLAIMER,
    AlertSummary,
    BatchPredictRequest,
    ClaimInput,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    ScenarioSimulationRequest,
)

logging.basicConfig(level=logging.INFO,
                    format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
log = logging.getLogger("medicaidguard.api")

VERSION = "0.1.0"
app = FastAPI(
    title="MedicaidGuard-XAI",
    version=VERSION,
    description=("Explainable prioritisation of suspicious Medicaid billing and EVV "
                 "activity for human review. " + DISCLAIMER),
)

STATE: dict = {"model": None, "tables": None, "scored": None, "pipeline": None}


def _load():
    """Lazy load so the service starts even before a model has been trained."""
    if STATE["model"] is None:
        p = ARTIFACTS_DIR / "final_model.joblib"
        if p.exists():
            import joblib
            STATE["model"] = joblib.load(p)
    if STATE["tables"] is None and (SYNTHETIC_DIR / "claims.parquet").exists():
        STATE["tables"] = {
            n: pd.read_parquet(SYNTHETIC_DIR / f"{n}.parquet")
            for n in ("claims", "evv_events", "patients", "providers",
                      "caregivers", "scenario_labels")
        }
    return STATE


def _scored() -> pd.DataFrame:
    """Score the demo dataset once and cache it."""
    s = _load()
    if STATE["scored"] is not None:
        return STATE["scored"]
    if s["model"] is None or s["tables"] is None:
        raise HTTPException(503, "model or dataset not available; run scripts/train.py")

    settings = load_settings()
    tables = s["tables"]
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    from ..evaluation.splits import time_split
    split = time_split(claims, settings.split)
    pipe = FeaturePipeline()
    pipe.fit(claims, tables["evv_events"], tables["patients"],
             pd.Series((split == "train").to_numpy(), index=claims.index))
    feats = pipe.transform(claims, tables["evv_events"], tables["patients"])
    X = feats.drop(columns=["claim_id"])[s["model"]["columns"]]

    prob = s["model"]["model"].predict_proba(X)[:, 1]
    prob = s["model"]["calibrator"].transform(prob)
    rs = rule_score(X).to_numpy() / 100.0
    score = 100.0 * (0.75 * prob / max(prob.max(), 1e-9) + 0.25 * rs)
    score = np.clip(score, 0, 100)

    out = claims[["claim_id", "provider_id", "caregiver_id", "patient_id",
                  "service_date", "billed_hours", "billed_amount"]].copy()
    out["risk_score"] = score
    out["model_probability"] = prob
    out["review_priority"] = HybridRiskModel.priority_band(score)
    out["alert_id"] = [f"SYN-{i:06d}" for i in range(len(out))]
    out["split"] = split.to_numpy()
    STATE["scored"] = out
    STATE["pipeline"] = (pipe, feats)
    return out


@app.get("/health", response_model=HealthResponse)
def health():
    s = _load()
    return HealthResponse(
        status="ok" if s["model"] and s["tables"] else "degraded",
        model_loaded=s["model"] is not None,
        dataset_loaded=s["tables"] is not None,
        version=VERSION,
    )


@app.get("/metrics")
def metrics():
    p = RESULTS_DIR / "run_summary.json"
    if not p.exists():
        raise HTTPException(404, "no run summary; run scripts/train.py")
    return json.loads(p.read_text())


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    s = _load()
    if s["model"] is None:
        raise HTTPException(503, "no trained model")
    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text())
    winner = summary["winner"]
    return ModelInfo(
        selected_model=winner,
        selection_rule=("max PR-AUC among models meeting the Brier calibration "
                        "floor, ties broken on Precision@5%. Accuracy is reported "
                        "but never used to select, because the trivial all-negative "
                        "classifier outscores every useful model on it at this prevalence."),
        primary_metric="pr_auc",
        test_metrics=summary["results"][winner],
        feature_count=len(s["model"]["columns"]),
        calibration="isotonic, fitted on the validation window",
    )


@app.get("/alerts", response_model=list[AlertSummary])
def alerts(priority: str | None = None, provider_id: str | None = None,
           caregiver_id: str | None = None, min_score: float = 0.0,
           limit: int = Query(100, le=1000)):
    df = _scored()
    if priority:
        df = df[df["review_priority"] == priority]
    if provider_id:
        df = df[df["provider_id"] == provider_id]
    if caregiver_id:
        df = df[df["caregiver_id"] == caregiver_id]
    df = df[df["risk_score"] >= min_score].nlargest(limit, "risk_score")
    return [AlertSummary(alert_id=r.alert_id, claim_id=r.claim_id,
                         provider_id=r.provider_id, caregiver_id=r.caregiver_id,
                         risk_score=float(r.risk_score),
                         review_priority=str(r.review_priority),
                         service_date=str(pd.Timestamp(r.service_date).date()))
            for r in df.itertuples()]


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: str):
    df = _scored()
    row = df[df["alert_id"] == alert_id]
    if row.empty:
        raise HTTPException(404, f"unknown alert {alert_id}")
    i = row.index[0]
    _, feats = STATE["pipeline"]
    X = feats.drop(columns=["claim_id"])
    return {
        **row.iloc[0].to_dict(),
        "service_date": str(pd.Timestamp(row.iloc[0]["service_date"]).date()),
        "rule_evidence": rule_evidence(X, X.index[i]),
        "calibration_note": "isotonic, fitted on the validation window",
        "disclaimer": DISCLAIMER,
    }


@app.get("/explanations/{alert_id}")
def explanation(alert_id: str, top_n: int = 8):
    df = _scored()
    row = df[df["alert_id"] == alert_id]
    if row.empty:
        raise HTTPException(404, f"unknown alert {alert_id}")
    s = _load()
    _, feats = STATE["pipeline"]
    X = feats.drop(columns=["claim_id"])[s["model"]["columns"]]
    from ..explainability.shap_explainer import SHAP_LIMITATION, ShapExplainer
    ex = ShapExplainer(s["model"]["model"], X.sample(min(200, len(X)), random_state=0))
    return {
        "alert_id": alert_id,
        "local_explanation": ex.local_explanation(X, int(row.index[0]), top_n=top_n),
        "limitations": SHAP_LIMITATION,
        "disclaimer": DISCLAIMER,
    }


@app.get("/providers/{provider_id}")
def provider(provider_id: str):
    df = _scored()
    sub = df[df["provider_id"] == provider_id]
    if sub.empty:
        raise HTTPException(404, f"unknown provider {provider_id}")
    return {
        "provider_id": provider_id,
        "claims": int(len(sub)),
        "total_billed_hours": float(sub["billed_hours"].sum()),
        "mean_risk_score": float(sub["risk_score"].mean()),
        "high_priority_alerts": int((sub["review_priority"].isin(["high", "critical"])).sum()),
        "peer_mean_risk_score": float(df["risk_score"].mean()),
        "disclaimer": DISCLAIMER,
    }


@app.get("/caregivers/{caregiver_id}")
def caregiver(caregiver_id: str):
    df = _scored()
    sub = df[df["caregiver_id"] == caregiver_id]
    if sub.empty:
        raise HTTPException(404, f"unknown caregiver {caregiver_id}")
    return {
        "caregiver_id": caregiver_id,
        "claims": int(len(sub)),
        "mean_risk_score": float(sub["risk_score"].mean()),
        "high_priority_alerts": int((sub["review_priority"].isin(["high", "critical"])).sum()),
        "disclaimer": DISCLAIMER,
    }


@app.get("/evv/{event_id}")
def evv_event(event_id: str):
    s = _load()
    if s["tables"] is None:
        raise HTTPException(503, "dataset not loaded")
    e = s["tables"]["evv_events"]
    row = e[e["evv_event_id"] == event_id]
    if row.empty:
        raise HTTPException(404, f"unknown EVV event {event_id}")
    return json.loads(row.iloc[[0]].to_json(orient="records"))[0]


@app.get("/data-quality")
def data_quality():
    s = _load()
    if s["tables"] is None:
        raise HTTPException(503, "dataset not loaded")
    from ..data.validation import validate_all
    return validate_all(s["tables"])


@app.get("/experiments")
def experiments():
    from ..config import TABLES_DIR
    out = {}
    for p in sorted(Path(TABLES_DIR).glob("*.csv")):
        out[p.stem] = json.loads(pd.read_csv(p).to_json(orient="records"))
    if not out:
        raise HTTPException(404, "no experiment tables; run scripts/run_experiments.py")
    return out


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput):
    return batch_predict(BatchPredictRequest(claims=[claim]))[0]


@app.post("/batch-predict", response_model=list[PredictionResponse])
def batch_predict(req: BatchPredictRequest):
    s = _load()
    if s["model"] is None or s["tables"] is None:
        raise HTTPException(503, "model or dataset not available")

    rows = [c.model_dump() for c in req.claims]
    new = pd.DataFrame(rows)
    for c in ("service_date", "billing_start_time", "billing_end_time",
              "submission_timestamp", "check_in_time", "check_out_time"):
        if c in new.columns:
            new[c] = pd.to_datetime(new[c], errors="coerce")

    # History matters: rolling and peer features need the surrounding population,
    # so incoming claims are appended to the reference set before featurising.
    hist = s["tables"]["claims"]
    claim_cols = [c for c in hist.columns if c in new.columns]
    combined = pd.concat([hist, new[claim_cols]], ignore_index=True)
    combined["authorization_id"] = combined.get("authorization_id", "UNKNOWN")
    combined = combined.sort_values(["service_date", "claim_id"]).reset_index(drop=True)

    evv_new = new[[c for c in ("claim_id", "caregiver_id", "patient_id", "check_in_time",
                               "check_out_time", "check_in_latitude", "check_in_longitude",
                               "check_out_latitude", "check_out_longitude", "device_type",
                               "verification_method", "location_accuracy_m")
                   if c in new.columns]].dropna(subset=["check_in_time"])
    evv_new = evv_new.assign(evv_event_id=lambda d: "EVVNEW" + d.index.astype(str),
                             event_status="complete")
    evv = pd.concat([s["tables"]["evv_events"], evv_new], ignore_index=True)

    pipe = FeaturePipeline()
    mask = pd.Series(~combined["claim_id"].isin(new["claim_id"]).to_numpy(),
                     index=combined.index)
    pipe.fit(combined, evv, s["tables"]["patients"], mask)
    feats = pipe.transform(combined, evv, s["tables"]["patients"])
    sel = feats["claim_id"].isin(new["claim_id"]).to_numpy()
    X = feats.drop(columns=["claim_id"]).loc[sel]
    for c in s["model"]["columns"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[s["model"]["columns"]]

    prob = s["model"]["calibrator"].transform(s["model"]["model"].predict_proba(X)[:, 1])
    rs = rule_score(X).to_numpy() / 100.0
    score = np.clip(100.0 * (0.75 * prob + 0.25 * rs), 0, 100)
    band = HybridRiskModel.priority_band(score)

    ids = feats.loc[sel, "claim_id"].tolist()
    return [PredictionResponse(
        claim_id=str(cid), risk_score=float(score[i]),
        review_priority=str(band[i]), model_probability=float(prob[i]),
        calibration_note="isotonic, fitted on the validation window",
        rule_evidence=rule_evidence(X, X.index[i]),
    ) for i, cid in enumerate(ids)]


@app.post("/scenario-simulation")
def scenario_simulation(req: ScenarioSimulationRequest):
    return simulate(SimulationInputs(**req.model_dump()))

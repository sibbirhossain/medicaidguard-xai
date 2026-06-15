"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DISCLAIMER = ("Synthetic/public research demonstration. Alerts are review "
              "recommendations, not findings of fraud.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    dataset_loaded: bool
    version: str


class ClaimInput(BaseModel):
    """One claim submitted for scoring. Field names mirror the data model."""

    claim_id: str
    provider_id: str
    caregiver_id: str
    patient_id: str
    service_date: str
    billing_start_time: str
    billing_end_time: str
    billed_hours: float = Field(gt=0, le=24)
    authorized_hours: float = Field(ge=0, le=24)
    service_code: str
    billed_amount: float = Field(ge=0)
    claim_status: str = "pending"
    submission_timestamp: str
    check_in_time: str | None = None
    check_out_time: str | None = None
    check_in_latitude: float | None = Field(default=None, ge=-90, le=90)
    check_in_longitude: float | None = Field(default=None, ge=-180, le=180)
    check_out_latitude: float | None = Field(default=None, ge=-90, le=90)
    check_out_longitude: float | None = Field(default=None, ge=-180, le=180)
    device_type: str | None = None
    verification_method: str | None = None
    location_accuracy_m: float | None = None


class PredictionResponse(BaseModel):
    claim_id: str
    risk_score: float = Field(ge=0, le=100)
    review_priority: Literal["low", "medium", "high", "critical"]
    model_probability: float = Field(ge=0, le=1)
    calibration_note: str
    rule_evidence: list[dict[str, Any]] = []
    disclaimer: str = DISCLAIMER


class BatchPredictRequest(BaseModel):
    claims: list[ClaimInput] = Field(min_length=1, max_length=1000)


class AlertSummary(BaseModel):
    alert_id: str
    claim_id: str
    provider_id: str
    caregiver_id: str
    risk_score: float
    review_priority: str
    service_date: str


class ModelInfo(BaseModel):
    selected_model: str
    selection_rule: str
    primary_metric: str
    test_metrics: dict[str, Any]
    feature_count: int
    calibration: str
    disclaimer: str = DISCLAIMER


class ScenarioSimulationRequest(BaseModel):
    claims_reviewed_period: int = Field(default=100_000, gt=0)
    review_capacity_pct: float = Field(default=0.05, gt=0, le=1)
    investigator_hours_per_review: float = Field(default=1.5, gt=0)
    investigator_cost_per_hour: float = Field(default=65.0, ge=0)
    precision_at_capacity: float = Field(default=0.5, ge=0, le=1)
    recall_at_capacity: float = Field(default=0.7, ge=0, le=1)
    hypothetical_improper_rate: float = Field(default=0.03, ge=0, le=1)
    hypothetical_amount_per_case: float = Field(default=1200.0, ge=0)
    recovery_realisation_rate: float = Field(default=0.35, ge=0, le=1)

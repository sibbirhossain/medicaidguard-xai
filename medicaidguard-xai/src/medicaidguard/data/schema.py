"""Relational schema for the synthetic Medicaid + EVV data model.

Two things live here:

1. Column contracts for each table, used by `validation.py` to fail fast when a
   table drifts from what downstream code expects.
2. The split between *feature-safe* columns and *ground-truth / provenance*
   columns. Scenario metadata (which simulated scenario was injected, and with
   what severity) is generated alongside the data but must never reach the
   model. `LABEL_COLUMNS` and `LEAKAGE_COLUMNS` are the enforcement point.
"""

from __future__ import annotations

PROVIDERS = {
    "provider_id": "string",
    "provider_type": "string",
    "specialty": "string",
    "enrollment_date": "datetime64[ns]",
    "service_region": "string",
    "active_status": "bool",
    "historical_claim_volume": "int64",
}

CAREGIVERS = {
    "caregiver_id": "string",
    "provider_id": "string",
    "employment_start_date": "datetime64[ns]",
    "service_region": "string",
    "qualification_status": "string",
    "historical_visit_volume": "int64",
}

PATIENTS = {
    "patient_id": "string",
    "service_region": "string",
    "demographic_group": "string",   # synthetic cohort label, fairness diagnostics only
    "age_band": "string",
    "authorized_hours_weekly": "float64",
    "service_type": "string",
    "eligibility_status": "string",
}

AUTHORIZATIONS = {
    "authorization_id": "string",
    "patient_id": "string",
    "authorized_service_code": "string",
    "authorized_start_date": "datetime64[ns]",
    "authorized_end_date": "datetime64[ns]",
    "authorized_hours_weekly": "float64",
    "authorization_status": "string",
}

CLAIMS = {
    "claim_id": "string",
    "provider_id": "string",
    "caregiver_id": "string",
    "patient_id": "string",
    "authorization_id": "string",
    "service_date": "datetime64[ns]",
    "billing_start_time": "datetime64[ns]",
    "billing_end_time": "datetime64[ns]",
    "billed_hours": "float64",
    "authorized_hours": "float64",
    "service_code": "string",
    "billed_amount": "float64",
    "claim_status": "string",
    "submission_timestamp": "datetime64[ns]",
}

EVV_EVENTS = {
    "evv_event_id": "string",
    "claim_id": "string",
    "caregiver_id": "string",
    "patient_id": "string",
    "check_in_time": "datetime64[ns]",
    "check_out_time": "datetime64[ns]",
    "check_in_latitude": "float64",
    "check_in_longitude": "float64",
    "check_out_latitude": "float64",
    "check_out_longitude": "float64",
    "device_type": "string",
    "verification_method": "string",
    "location_accuracy_m": "float64",
    "event_status": "string",
}

# Ground truth produced by the generator. Kept in a separate table.
SCENARIO_LABELS = {
    "claim_id": "string",
    "is_suspicious": "int64",
    "scenario": "string",
    "severity": "float64",
}

TABLES = {
    "providers": PROVIDERS,
    "caregivers": CAREGIVERS,
    "patients": PATIENTS,
    "authorizations": AUTHORIZATIONS,
    "claims": CLAIMS,
    "evv_events": EVV_EVENTS,
    "scenario_labels": SCENARIO_LABELS,
}

PRIMARY_KEYS = {
    "providers": "provider_id",
    "caregivers": "caregiver_id",
    "patients": "patient_id",
    "authorizations": "authorization_id",
    "claims": "claim_id",
    "evv_events": "evv_event_id",
}

FOREIGN_KEYS = [
    ("caregivers", "provider_id", "providers", "provider_id"),
    ("authorizations", "patient_id", "patients", "patient_id"),
    ("claims", "provider_id", "providers", "provider_id"),
    ("claims", "caregiver_id", "caregivers", "caregiver_id"),
    ("claims", "patient_id", "patients", "patient_id"),
    ("evv_events", "claim_id", "claims", "claim_id"),
]

# --- Leakage control -------------------------------------------------------

LABEL_COLUMN = "is_suspicious"

# Columns that encode the answer, directly or by construction. A feature matrix
# containing any of these is rejected by evaluation.leakage.assert_no_leakage.
LEAKAGE_COLUMNS = frozenset({
    "is_suspicious",
    "scenario",
    "severity",
    "claim_id",
    "evv_event_id",
    "authorization_id",
})

# Identifier columns that are legitimate for grouping/splitting but are not
# model inputs (they would let a tree memorise specific entities).
GROUP_COLUMNS = frozenset({"provider_id", "caregiver_id", "patient_id"})

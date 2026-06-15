# Data Dictionary

Auto-generated from the actual generated tables at seed 20260913. Regenerate with `python scripts/generate_data.py`.

> All data is synthetic. No identifier corresponds to a real person or organisation.

## Leakage blocklist

The following columns are rejected by `evaluation.leakage.assert_no_leakage_columns` if they appear in a feature matrix:

```
authorization_id, claim_id, evv_event_id, is_suspicious, scenario, severity
```

Entity identifiers (`provider_id`, `caregiver_id`, `patient_id`) are valid for grouping and splitting but are excluded from model inputs so trees cannot memorise a specific entity.


## `regions`

6 rows × 3 columns · primary key: `—`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `service_region` | str | 0.0% | 6 | R00 | Region code R00-R05 |
| `center_lat` | float64 | 0.0% | 6 | 41.03610530747552 | Region centroid latitude |
| `center_lon` | float64 | 0.0% | 6 | -73.41808814197482 | Region centroid longitude |

## `providers`

60 rows × 7 columns · primary key: `provider_id`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `provider_id` | str | 0.0% | 60 | PRV0000 | Synthetic agency identifier (primary key) |
| `provider_type` | str | 0.0% | 4 | FI | Agency type: LHCSA, FI, CHHA, MLTC-Contracted |
| `specialty` | str | 0.0% | 4 | home_health_aide | Primary service line |
| `enrollment_date` | datetime64[us] | 0.0% | 59 | 2023-09-21 00:00:00 | Date the agency entered the programme |
| `service_region` | str | 0.0% | 6 | R03 | Region code R00-R05 |
| `active_status` | bool | 0.0% | 2 | True | Whether the agency is currently active |
| `historical_claim_volume` | int64 | 0.0% | 60 | 14627 | Prior-period claim count (entity attribute, not a feature) |

## `caregivers`

900 rows × 6 columns · primary key: `caregiver_id`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `caregiver_id` | str | 0.0% | 900 | CG00000 | Synthetic caregiver/PCA identifier (primary key) |
| `provider_id` | str | 0.0% | 60 | PRV0040 | Synthetic agency identifier (primary key) |
| `employment_start_date` | datetime64[us] | 0.0% | 723 | 2023-06-23 00:00:00 | Date the caregiver began with this agency |
| `service_region` | str | 0.0% | 6 | R04 | Region code R00-R05 |
| `qualification_status` | str | 0.0% | 3 | certified | certified, provisional or lapsed |
| `historical_visit_volume` | int64 | 0.0% | 803 | 395 | Prior-period visit count |

## `patients`

2,200 rows × 9 columns · primary key: `patient_id`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `patient_id` | str | 0.0% | 2,200 | PT000000 | Synthetic beneficiary identifier (primary key) |
| `service_region` | str | 0.0% | 6 | R04 | Region code R00-R05 |
| `demographic_group` | str | 0.0% | 3 | group_c | Randomly assigned synthetic cohort. Fairness diagnostics ONLY; no real-world meaning |
| `age_band` | str | 0.0% | 4 | 65-79 | Synthetic age band |
| `authorized_hours_weekly` | float64 | 0.0% | 36 | 12.0 | Weekly service hours authorised |
| `service_type` | str | 0.0% | 3 | personal_care | personal_care, home_health or respite |
| `eligibility_status` | str | 0.0% | 2 | active | active or pending |
| `home_lat` | float64 | 0.0% | 2,200 | 40.96476948205217 | Synthetic residence latitude (generator-internal; never a model feature) |
| `home_lon` | float64 | 0.0% | 2,200 | -74.81480944467228 | Synthetic residence longitude (generator-internal; never a model feature) |

## `authorizations`

2,200 rows × 7 columns · primary key: `authorization_id`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `authorization_id` | str | 0.0% | 2,200 | AUTH000000 | Authorisation identifier (primary key) |
| `patient_id` | str | 0.0% | 2,200 | PT000000 | Synthetic beneficiary identifier (primary key) |
| `authorized_service_code` | str | 0.0% | 5 | T1019 | HCPCS-style code the authorisation covers |
| `authorized_start_date` | datetime64[us] | 0.0% | 150 | 2023-10-06 00:00:00 | Authorisation window start |
| `authorized_end_date` | datetime64[us] | 0.0% | 339 | 2024-11-08 00:00:00 | Authorisation window end |
| `authorized_hours_weekly` | float64 | 0.0% | 36 | 12.0 | Weekly service hours authorised |
| `authorization_status` | str | 0.0% | 2 | expired | approved or expired |

## `claims`

45,297 rows × 14 columns · primary key: `claim_id`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `patient_id` | str | 0.0% | 2,199 | PT002179 | Synthetic beneficiary identifier (primary key) |
| `caregiver_id` | str | 0.0% | 890 | CG00138 | Synthetic caregiver/PCA identifier (primary key) |
| `provider_id` | str | 0.0% | 60 | PRV0017 | Synthetic agency identifier (primary key) |
| `authorization_id` | str | 0.0% | 2,199 | AUTH002179 | Authorisation identifier (primary key) |
| `service_code` | str | 0.0% | 5 | T1020 | HCPCS-style service code billed |
| `authorized_hours` | float64 | 0.0% | 60 | 2.0 | Authorised daily-equivalent hours for comparison |
| `service_date` | datetime64[us] | 0.0% | 312 | 2024-01-01 00:00:00 | Calendar date of service; drives the time-based split |
| `billing_start_time` | datetime64[us] | 0.0% | 11,555 | 2024-01-01 00:09:15 | Claimed service start timestamp |
| `billing_end_time` | datetime64[us] | 0.0% | 14,670 | 2024-01-01 00:11:15 | Claimed service end timestamp |
| `billed_hours` | float64 | 0.0% | 294 | 2.0 | Hours billed on this claim |
| `billed_amount` | float64 | 0.0% | 17,362 | 62.52 | Amount billed (synthetic currency units) |
| `claim_status` | str | 0.0% | 3 | paid | paid, pending or denied |
| `submission_timestamp` | datetime64[us] | 0.0% | 44,735 | 2024-01-05 03:14:15 | When the claim was submitted; lag is a feature |
| `claim_id` | str | 0.0% | 45,297 | CLM0000000 | Claim identifier (primary key). Excluded from features |

## `evv_events`

43,106 rows × 14 columns · primary key: `evv_event_id`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `evv_event_id` | str | 0.0% | 43,106 | EVV0000000 | EVV record identifier (primary key). Excluded from features |
| `claim_id` | str | 0.0% | 43,106 | CLM0000000 | Claim identifier (primary key). Excluded from features |
| `caregiver_id` | str | 0.0% | 890 | CG00138 | Synthetic caregiver/PCA identifier (primary key) |
| `patient_id` | str | 0.0% | 2,198 | PT002179 | Synthetic beneficiary identifier (primary key) |
| `check_in_time` | datetime64[us] | 0.0% | 40,868 | 2024-01-01 00:12:25 | Verified arrival timestamp |
| `check_out_time` | datetime64[us] | 0.0% | 42,957 | 2024-01-01 02:10:47 | Verified departure timestamp |
| `check_in_latitude` | float64 | 29.6% | 30,328 | 41.105800819185596 | Arrival latitude; null for telephony and manual entry |
| `check_in_longitude` | float64 | 29.6% | 30,328 | -75.47025863094517 | Arrival longitude; null for telephony and manual entry |
| `check_out_latitude` | float64 | 29.6% | 30,328 | 41.107316684193194 | Departure latitude; null for non-GPS methods |
| `check_out_longitude` | float64 | 29.6% | 30,328 | -75.47036611719064 | Departure longitude; null for non-GPS methods |
| `device_type` | str | 0.0% | 4 | mobile_app | mobile_app, telephony_ivr, fob or web_portal |
| `verification_method` | str | 0.0% | 4 | gps | gps, telephony, fixed_device or manual_entry |
| `location_accuracy_m` | float64 | 29.7% | 29,921 | 40.968671168326246 | Reported GPS accuracy radius in metres |
| `event_status` | str | 0.0% | 1 | complete | EVV record status |

## `scenario_labels`

45,297 rows × 4 columns · primary key: `—`

| Column | Type | Missing | Unique | Example | Description |
|---|---|---|---|---|---|
| `claim_id` | str | 0.0% | 45,297 | CLM0013099 | Claim identifier (primary key). Excluded from features |
| `is_suspicious` | int64 | 0.0% | 2 | 1 | SIMULATED label. Ground truth only; blocklisted from features |
| `scenario` | str | 0.0% | 17 | overlapping_visits | Which injected scenario produced the label; blocklisted from features |
| `severity` | float64 | 0.0% | 1,577 | 0.6780180845185421 | Injected scenario intensity in [0,1]; blocklisted from features |

## Foreign keys

- `caregivers.provider_id` → `providers.provider_id`
- `authorizations.patient_id` → `patients.patient_id`
- `claims.provider_id` → `providers.provider_id`
- `claims.caregiver_id` → `caregivers.caregiver_id`
- `claims.patient_id` → `patients.patient_id`
- `evv_events.claim_id` → `claims.claim_id`


## Indexes (relational store)

- `ix_claims_caregiver_date` on `(caregiver_id, service_date)` — matches the dominant access pattern for rolling features and the overlap/travel checks
- Single-column indexes on every foreign key and on `service_date`, `service_code`, `review_priority`

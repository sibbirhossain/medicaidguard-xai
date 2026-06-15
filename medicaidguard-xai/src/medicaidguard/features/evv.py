"""EVV consistency features.

These are the features that do not exist in any public claims dataset, and they
are the reason this project needs a synthetic track at all. Four families:

* completeness  - is there a verification record for the billed service?
* duration      - does the verified visit length match the billed length?
* location      - was the check-in near the patient, and is the implied travel
                  from the previous visit physically possible?
* concurrency   - does this caregiver's visit overlap another of their visits?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.generator import haversine_m

# A generous urban ceiling. Anything faster implies the two check-ins could not
# both have been performed by the same person.
MAX_PLAUSIBLE_SPEED_KMH = 110.0


def build_evv_features(claims: pd.DataFrame, evv: pd.DataFrame,
                       patients: pd.DataFrame) -> pd.DataFrame:
    df = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    e = evv.drop_duplicates("claim_id").set_index("claim_id")
    joined = df.join(e.drop(columns=["caregiver_id", "patient_id"], errors="ignore"),
                     on="claim_id")
    # A left join against an empty or all-missing EVV table yields object-dtype
    # columns, on which .dt raises. Coerce explicitly so that "no EVV at all" is
    # a supported state rather than a crash: it is exactly the state a new state
    # programme or an outage produces.
    for c in ("check_in_time", "check_out_time"):
        joined[c] = pd.to_datetime(joined.get(c), errors="coerce")
    for c in ("check_in_latitude", "check_in_longitude", "check_out_latitude",
              "check_out_longitude", "location_accuracy_m"):
        joined[c] = pd.to_numeric(joined.get(c), errors="coerce")
    for c in ("device_type", "verification_method"):
        if c not in joined.columns:
            joined[c] = np.nan

    f = pd.DataFrame(index=df.index)
    f["claim_id"] = df["claim_id"]

    has_evv = joined["check_in_time"].notna()
    f["evv_present"] = has_evv.astype("float64")

    evv_hours = (
        (joined["check_out_time"] - joined["check_in_time"]).dt.total_seconds() / 3600.0
    )
    f["evv_duration_hours"] = evv_hours
    f["evv_billing_duration_diff"] = df["billed_hours"] - evv_hours
    f["evv_billing_duration_ratio"] = df["billed_hours"] / evv_hours.replace(0, np.nan)
    f["evv_billing_duration_abs_diff"] = f["evv_billing_duration_diff"].abs()

    # Clock alignment: legitimate records differ by minutes, batch-entered
    # records align to the second.
    f["checkin_billing_offset_min"] = (
        (joined["check_in_time"] - df["billing_start_time"]).dt.total_seconds() / 60.0
    )
    f["checkout_billing_offset_min"] = (
        (joined["check_out_time"] - df["billing_end_time"]).dt.total_seconds() / 60.0
    )
    f["evv_clock_exact_match"] = (
        (f["checkin_billing_offset_min"].abs() < 0.02)
        & (f["checkout_billing_offset_min"].abs() < 0.02)
    ).astype("float64")

    # Distance from the patient's registered home.
    homes = patients.set_index("patient_id")[["home_lat", "home_lon"]]
    hl = homes.reindex(df["patient_id"]).to_numpy()
    f["checkin_distance_from_home_m"] = haversine_m(
        joined["check_in_latitude"].to_numpy(), joined["check_in_longitude"].to_numpy(),
        hl[:, 0], hl[:, 1],
    )
    f["checkin_checkout_distance_m"] = haversine_m(
        joined["check_in_latitude"].to_numpy(), joined["check_in_longitude"].to_numpy(),
        joined["check_out_latitude"].to_numpy(), joined["check_out_longitude"].to_numpy(),
    )
    f["location_accuracy_m"] = joined["location_accuracy_m"]
    f["has_gps"] = (joined["verification_method"] == "gps").astype("float64")
    f["device_type"] = joined["device_type"].fillna("missing").astype(str)
    f["verification_method"] = joined["verification_method"].fillna("missing").astype(str)

    f = pd.concat([f, _travel_and_overlap(joined)], axis=1)
    return f


def _travel_and_overlap(joined: pd.DataFrame) -> pd.DataFrame:
    """Sequential per-caregiver features: implied travel speed and overlap."""
    cols = ["caregiver_id", "check_in_time", "check_out_time",
            "check_in_latitude", "check_in_longitude",
            "billing_start_time", "billing_end_time"]
    d = joined[cols].copy()
    for c in ("check_in_time", "check_out_time", "billing_start_time", "billing_end_time"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    for c in ("check_in_latitude", "check_in_longitude"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_pos"] = np.arange(len(d))
    d = d.sort_values(["caregiver_id", "billing_start_time"])
    g = d.groupby("caregiver_id", observed=True, sort=False)

    prev_out = g["check_out_time"].shift(1)
    prev_lat = g["check_in_latitude"].shift(1)
    prev_lon = g["check_in_longitude"].shift(1)
    prev_bill_end = g["billing_end_time"].shift(1)

    gap_h = (d["check_in_time"] - prev_out).dt.total_seconds() / 3600.0
    dist_km = haversine_m(prev_lat.to_numpy(), prev_lon.to_numpy(),
                          d["check_in_latitude"].to_numpy(),
                          d["check_in_longitude"].to_numpy()) / 1000.0

    # Guard the divide: a zero or negative gap with real distance is itself the
    # strongest possible signal, so map it to a large finite speed.
    safe_gap = gap_h.to_numpy()
    speed = np.where(safe_gap > 1e-3, dist_km / np.maximum(safe_gap, 1e-3),
                     np.where(dist_km > 0.5, 9_999.0, 0.0))

    out = pd.DataFrame(index=d.index)
    out["travel_gap_hours"] = gap_h
    out["travel_distance_km"] = dist_km
    out["implied_speed_kmh"] = speed
    out["implausible_travel_flag"] = (speed > MAX_PLAUSIBLE_SPEED_KMH).astype("float64")

    # Overlap: this visit starts before the previous one was billed as finished.
    overlap_min = (prev_bill_end - d["billing_start_time"]).dt.total_seconds() / 60.0
    out["overlap_minutes_with_prev"] = overlap_min.clip(lower=0).fillna(0.0)
    out["visit_overlaps_prev"] = (overlap_min > 1.0).astype("float64")

    out["_pos"] = d["_pos"].to_numpy()
    out = out.sort_values("_pos").drop(columns=["_pos"]).reset_index(drop=True)
    return out

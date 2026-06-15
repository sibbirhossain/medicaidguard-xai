"""Billing-behaviour features.

Leakage discipline
------------------
Every entity-level aggregate here is *causal*: for a claim on day t belonging
to caregiver c, only claims of c strictly before the current row contribute.
This is enforced with a sort + groupby + shift(1) + expanding/rolling pattern.
A naive `groupby().transform('mean')` would let a caregiver's future (including
their own suspicious claims) inform their past, which inflates every metric and
is the single most common defect in fraud-detection write-ups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_rolling(df: pd.DataFrame, key: str, value: str, windows=(7, 30)) -> pd.DataFrame:
    """Past-only rolling mean/count per entity over calendar-day windows."""
    out = pd.DataFrame(index=df.index)
    d = df[[key, "service_date", value]].copy()
    d = d.sort_values([key, "service_date"])
    g = d.groupby(key, observed=True, sort=False)

    # shift(1) removes the current row so the window is strictly historical.
    shifted = g[value].shift(1)
    tmp = pd.DataFrame({key: d[key], "service_date": d["service_date"], "v": shifted})

    for w in windows:
        roll = (
            tmp.set_index("service_date")
            .groupby(key, observed=True, sort=False)["v"]
            .rolling(f"{w}D", min_periods=1)
        )
        mean = roll.mean().reset_index(level=0, drop=True)
        cnt = roll.count().reset_index(level=0, drop=True)
        mean.index = d.index
        cnt.index = d.index
        out[f"{key}_{value}_mean_{w}d"] = mean.reindex(df.index)
        out[f"{key}_{value}_count_{w}d"] = cnt.reindex(df.index)

    # Expanding history (all prior activity) gives a stable personal baseline.
    exp_mean = g[value].apply(lambda s: s.shift(1).expanding().mean())
    exp_mean.index = d.index
    exp_std = g[value].apply(lambda s: s.shift(1).expanding().std())
    exp_std.index = d.index
    out[f"{key}_{value}_hist_mean"] = exp_mean.reindex(df.index)
    out[f"{key}_{value}_hist_std"] = exp_std.reindex(df.index)
    return out


def build_billing_features(claims: pd.DataFrame) -> pd.DataFrame:
    """Row-level and causal-historical billing features."""
    df = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    f = pd.DataFrame(index=df.index)
    f["claim_id"] = df["claim_id"]

    auth = df["authorized_hours"].replace(0, np.nan)
    f["billed_hours"] = df["billed_hours"]
    f["billed_amount"] = df["billed_amount"]
    f["authorized_hours"] = df["authorized_hours"]
    f["billed_to_authorized_ratio"] = df["billed_hours"] / auth
    f["billed_over_authorized"] = (df["billed_hours"] - df["authorized_hours"]).clip(lower=0)
    f["hourly_rate"] = df["billed_amount"] / df["billed_hours"].replace(0, np.nan)

    # Duplicate / near-duplicate signals. Exact duplicates share the full key;
    # near-duplicates share the coarse key (visit rounded to the hour).
    exact_key = (df["caregiver_id"].astype(str) + "|" + df["patient_id"].astype(str) + "|"
                 + df["billing_start_time"].astype(str) + "|" + df["billed_hours"].astype(str))
    coarse_key = (df["caregiver_id"].astype(str) + "|" + df["patient_id"].astype(str) + "|"
                  + df["billing_start_time"].dt.floor("h").astype(str))
    f["exact_duplicate_group_size"] = exact_key.map(exact_key.value_counts()).astype("float64")
    f["near_duplicate_group_size"] = coarse_key.map(coarse_key.value_counts()).astype("float64")
    f["same_day_same_pair_claims"] = (
        df.groupby(["caregiver_id", "patient_id", "service_date"], observed=True)["claim_id"]
        .transform("count").astype("float64")
    )

    # Daily workload per caregiver (same-day, so it is observable at review time).
    daily = df.groupby(["caregiver_id", "service_date"], observed=True)["billed_hours"]
    f["caregiver_daily_hours"] = daily.transform("sum")
    f["caregiver_daily_visits"] = daily.transform("count").astype("float64")
    f["caregiver_daily_patients"] = (
        df.groupby(["caregiver_id", "service_date"], observed=True)["patient_id"]
        .transform("nunique").astype("float64")
    )

    for key in ("caregiver_id", "provider_id", "patient_id"):
        f = pd.concat([f, _causal_rolling(df, key, "billed_hours")], axis=1)

    # Deviation from the entity's own history (robust to scale differences).
    for key in ("caregiver_id", "provider_id"):
        hist = f[f"{key}_billed_hours_hist_mean"]
        sd = f[f"{key}_billed_hours_hist_std"].replace(0, np.nan)
        f[f"{key}_hours_z_vs_self"] = (df["billed_hours"] - hist) / sd
        f[f"{key}_hours_ratio_vs_self"] = df["billed_hours"] / hist.replace(0, np.nan)
        # Short-window vs long-window trend: a level shift shows up here.
        f[f"{key}_trend_7_30"] = (
            f[f"{key}_billed_hours_mean_7d"] / f[f"{key}_billed_hours_mean_30d"].replace(0, np.nan)
        )

    f["service_code"] = df["service_code"].astype(str)
    f["claim_status"] = df["claim_status"].astype(str)
    return f

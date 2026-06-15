"""Temporal and robust-statistical features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_temporal_features(claims: pd.DataFrame) -> pd.DataFrame:
    df = claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    f = pd.DataFrame(index=df.index)
    f["claim_id"] = df["claim_id"]

    st = df["billing_start_time"]
    hour = st.dt.hour + st.dt.minute / 60.0
    f["start_hour"] = hour
    f["day_of_week"] = st.dt.dayofweek.astype("float64")
    f["is_weekend"] = (st.dt.dayofweek >= 5).astype("float64")
    f["is_after_hours"] = ((hour < 6.0) | (hour >= 21.0)).astype("float64")
    f["is_overnight"] = ((hour >= 0.0) & (hour < 5.0)).astype("float64")
    f["month"] = st.dt.month.astype("float64")
    f["day_of_month"] = st.dt.day.astype("float64")

    # Cyclical encodings keep 23:00 and 01:00 close together for linear models.
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # "Roundness": human-observed visits rarely start exactly on the hour with a
    # whole-number duration. Batch data entry does.
    f["start_minute_offset"] = st.dt.minute.astype("float64")
    f["starts_on_hour"] = (st.dt.minute == 0).astype("float64") * (st.dt.second == 0).astype("float64")
    f["duration_is_whole_hours"] = (
        np.isclose(df["billed_hours"] % 1.0, 0.0)
    ).astype("float64")

    lag_h = (df["submission_timestamp"] - df["billing_end_time"]).dt.total_seconds() / 3600.0
    f["submission_lag_hours"] = lag_h
    f["submission_same_day"] = (lag_h < 24).astype("float64")
    f["submission_immediate"] = (lag_h < 0.25).astype("float64")
    return f


def robust_z(values: pd.Series, center: float, scale: float) -> pd.Series:
    """Median/MAD z-score. MAD is scaled to be consistent with the SD for
    normal data (0.6745 factor), and a zero MAD falls back to NaN rather than
    producing infinities."""
    scale = scale if scale and scale > 0 else np.nan
    return 0.6745 * (values - center) / scale


class PeerDeviationTransformer:
    """Peer-group robust deviations, fitted on the training window only.

    Fitting on the full dataset would let test-period behaviour define the
    "normal" baseline that test-period rows are then scored against — a subtle
    but real leak. `fit` must be called with training rows only.
    """

    def __init__(self, group_cols=("service_code",),
                 value_cols=("billed_hours", "billed_amount", "hourly_rate")):
        self.group_cols = list(group_cols)
        self.value_cols = list(value_cols)
        self.stats_: dict[str, pd.DataFrame] = {}
        self.global_: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> PeerDeviationTransformer:
        for col in self.value_cols:
            if col not in df.columns:
                continue
            grp = df.groupby(self.group_cols, observed=True)[col]
            med = grp.median()
            mad = grp.apply(lambda s: (s - s.median()).abs().median())
            self.stats_[col] = pd.DataFrame({"median": med, "mad": mad})
            self.global_[col] = (float(df[col].median()),
                                 float((df[col] - df[col].median()).abs().median()))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col, stats in self.stats_.items():
            if col not in df.columns:
                continue
            keys = df[self.group_cols[0]] if len(self.group_cols) == 1 else \
                pd.MultiIndex.from_frame(df[self.group_cols])
            med = stats["median"].reindex(keys).to_numpy()
            mad = stats["mad"].reindex(keys).to_numpy()
            gmed, gmad = self.global_[col]
            med = np.where(np.isnan(med), gmed, med)
            mad = np.where(np.isnan(mad) | (mad <= 0), gmad if gmad > 0 else np.nan, mad)
            out[f"{col}_peer_robust_z"] = 0.6745 * (df[col].to_numpy() - med) / mad
            out[f"{col}_peer_ratio"] = df[col].to_numpy() / np.where(med == 0, np.nan, med)
        return out

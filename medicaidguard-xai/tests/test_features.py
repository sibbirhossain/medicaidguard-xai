import numpy as np
import pandas as pd

from medicaidguard.features.billing import build_billing_features
from medicaidguard.features.evv import MAX_PLAUSIBLE_SPEED_KMH, build_evv_features
from medicaidguard.features.pipeline import FeaturePipeline
from medicaidguard.features.temporal import build_temporal_features


def test_billing_features_align_to_claims(tables):
    f = build_billing_features(tables["claims"])
    assert len(f) == len(tables["claims"])
    assert f["claim_id"].is_unique


def test_authorized_ratio_is_correct(tables):
    f = build_billing_features(tables["claims"])
    c = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    expected = c["billed_hours"] / c["authorized_hours"].replace(0, np.nan)
    pd.testing.assert_series_equal(f["billed_to_authorized_ratio"], expected,
                                   check_names=False)


def test_evv_features_present(tables):
    f = build_evv_features(tables["claims"], tables["evv_events"], tables["patients"])
    for col in ("evv_present", "evv_billing_duration_diff", "implied_speed_kmh",
                "overlap_minutes_with_prev", "checkin_distance_from_home_m"):
        assert col in f.columns


def test_missing_evv_is_flagged(tables):
    f = build_evv_features(tables["claims"], tables["evv_events"], tables["patients"])
    n_missing = int((f["evv_present"] == 0).sum())
    assert n_missing == len(tables["claims"]) - tables["evv_events"]["claim_id"].nunique()


def test_travel_plausibility_threshold_fires():
    """A 200 km jump in 10 minutes must be flagged."""
    base = pd.Timestamp("2024-03-01 08:00")
    claims = pd.DataFrame({
        "claim_id": ["A", "B"], "provider_id": ["P", "P"],
        "caregiver_id": ["C", "C"], "patient_id": ["X", "Y"],
        "service_date": [base.normalize()] * 2,
        "billing_start_time": [base, base + pd.Timedelta(hours=1, minutes=10)],
        "billing_end_time": [base + pd.Timedelta(hours=1),
                             base + pd.Timedelta(hours=2)],
        "billed_hours": [1.0, 1.0], "authorized_hours": [1.0, 1.0],
        "service_code": ["T1019"] * 2, "billed_amount": [27.5, 27.5],
        "claim_status": ["paid"] * 2,
        "submission_timestamp": [base + pd.Timedelta(days=1)] * 2,
    })
    evv = pd.DataFrame({
        "evv_event_id": ["E1", "E2"], "claim_id": ["A", "B"],
        "caregiver_id": ["C", "C"], "patient_id": ["X", "Y"],
        "check_in_time": claims["billing_start_time"],
        "check_out_time": claims["billing_end_time"],
        "check_in_latitude": [40.0, 41.8], "check_in_longitude": [-74.0, -74.0],
        "check_out_latitude": [40.0, 41.8], "check_out_longitude": [-74.0, -74.0],
        "device_type": ["mobile_app"] * 2, "verification_method": ["gps"] * 2,
        "location_accuracy_m": [10.0, 10.0], "event_status": ["complete"] * 2,
    })
    patients = pd.DataFrame({"patient_id": ["X", "Y"],
                             "home_lat": [40.0, 41.8], "home_lon": [-74.0, -74.0]})
    f = build_evv_features(claims, evv, patients)
    assert f["implied_speed_kmh"].max() > MAX_PLAUSIBLE_SPEED_KMH
    assert f["implausible_travel_flag"].sum() == 1


def test_overlap_detection():
    base = pd.Timestamp("2024-03-01 08:00")
    claims = pd.DataFrame({
        "claim_id": ["A", "B"], "provider_id": ["P", "P"],
        "caregiver_id": ["C", "C"], "patient_id": ["X", "Y"],
        "service_date": [base.normalize()] * 2,
        "billing_start_time": [base, base + pd.Timedelta(hours=1)],
        "billing_end_time": [base + pd.Timedelta(hours=4),
                             base + pd.Timedelta(hours=3)],
        "billed_hours": [4.0, 2.0], "authorized_hours": [4.0, 2.0],
        "service_code": ["T1019"] * 2, "billed_amount": [110.0, 55.0],
        "claim_status": ["paid"] * 2,
        "submission_timestamp": [base + pd.Timedelta(days=1)] * 2,
    })
    evv = pd.DataFrame(columns=["evv_event_id", "claim_id", "caregiver_id", "patient_id",
                                "check_in_time", "check_out_time", "check_in_latitude",
                                "check_in_longitude", "check_out_latitude",
                                "check_out_longitude", "device_type",
                                "verification_method", "location_accuracy_m",
                                "event_status"])
    patients = pd.DataFrame({"patient_id": ["X", "Y"], "home_lat": [40.0, 40.0],
                             "home_lon": [-74.0, -74.0]})
    f = build_evv_features(claims, evv, patients)
    assert f["visit_overlaps_prev"].sum() == 1
    assert f["overlap_minutes_with_prev"].max() >= 179


def test_temporal_flags():
    base = pd.Timestamp("2024-03-02 02:00")  # a Saturday, 2am
    claims = pd.DataFrame({
        "claim_id": ["A"], "provider_id": ["P"], "caregiver_id": ["C"],
        "patient_id": ["X"], "service_date": [base.normalize()],
        "billing_start_time": [base], "billing_end_time": [base + pd.Timedelta(hours=2)],
        "billed_hours": [2.0], "authorized_hours": [2.0], "service_code": ["T1019"],
        "billed_amount": [55.0], "claim_status": ["paid"],
        "submission_timestamp": [base + pd.Timedelta(hours=2, minutes=5)],
    })
    f = build_temporal_features(claims)
    assert f["is_weekend"].iloc[0] == 1.0
    assert f["is_overnight"].iloc[0] == 1.0
    assert f["duration_is_whole_hours"].iloc[0] == 1.0


def test_pipeline_requires_fit_before_transform(tables):
    p = FeaturePipeline()
    try:
        p.transform(tables["claims"], tables["evv_events"], tables["patients"])
    except RuntimeError:
        return
    raise AssertionError("transform should refuse to run before fit")


def test_pipeline_output_has_no_nan_or_inf(tables):
    import numpy as np
    p = FeaturePipeline()
    mask = pd.Series(True, index=range(len(tables["claims"])))
    out = p.fit_transform(tables["claims"], tables["evv_events"],
                          tables["patients"], mask)
    num = out.drop(columns=["claim_id"]).select_dtypes(include=[np.number])
    assert np.isfinite(num.to_numpy()).all()

import pandas as pd

from medicaidguard.data.generator import SyntheticMedicaidEVVGenerator, haversine_m


def test_reproducible_with_same_seed(small_cfg):
    a = SyntheticMedicaidEVVGenerator(small_cfg).generate()
    b = SyntheticMedicaidEVVGenerator(small_cfg).generate()
    pd.testing.assert_frame_equal(a["claims"], b["claims"])
    pd.testing.assert_frame_equal(a["scenario_labels"], b["scenario_labels"])


def test_different_seed_changes_data(small_cfg):
    import dataclasses
    other = dataclasses.replace(small_cfg, seed=small_cfg.seed + 1)
    a = SyntheticMedicaidEVVGenerator(small_cfg).generate()
    b = SyntheticMedicaidEVVGenerator(other).generate()
    assert not a["claims"]["billed_hours"].equals(b["claims"]["billed_hours"])


def test_prevalence_is_configurable(small_cfg):
    import dataclasses
    lo = SyntheticMedicaidEVVGenerator(
        dataclasses.replace(small_cfg, suspicious_prevalence=0.01)).generate()
    hi = SyntheticMedicaidEVVGenerator(
        dataclasses.replace(small_cfg, suspicious_prevalence=0.08)).generate()
    assert lo["scenario_labels"]["is_suspicious"].mean() < \
           hi["scenario_labels"]["is_suspicious"].mean()


def test_every_claim_has_a_label(tables):
    assert set(tables["claims"]["claim_id"]) == set(tables["scenario_labels"]["claim_id"])


def test_labels_are_not_in_the_claims_table(tables):
    assert "is_suspicious" not in tables["claims"].columns
    assert "scenario" not in tables["claims"].columns


def test_haversine_known_distance():
    # ~111.2 km per degree of latitude at the equator.
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110_000 < d < 112_000

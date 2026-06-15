from medicaidguard.data.validation import (
    validate_all,
    validate_keys,
    validate_plausibility,
)


def test_schema_and_keys_are_clean(tables):
    report = validate_all(tables)
    assert report["key_issues"] == []
    assert report["plausibility_issues"] == []
    assert report["passed"]


def test_foreign_keys_resolve(tables):
    assert validate_keys(tables) == []


def test_plausibility_catches_broken_times(tables):
    bad = {k: v.copy() for k, v in tables.items()}
    bad["claims"].loc[0, "billing_end_time"] = bad["claims"].loc[0, "billing_start_time"] - \
        __import__("pandas").Timedelta(hours=2)
    assert any("before" in m for m in validate_plausibility(bad))

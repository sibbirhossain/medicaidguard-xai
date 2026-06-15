"""Schema, referential-integrity and plausibility validation."""

from __future__ import annotations

import pandas as pd

from . import schema


class ValidationError(AssertionError):
    pass


def validate_columns(df: pd.DataFrame, name: str, strict: bool = False) -> list[str]:
    expected = schema.TABLES.get(name)
    if expected is None:
        return []
    missing = [c for c in expected if c not in df.columns]
    if missing and strict:
        raise ValidationError(f"{name}: missing columns {missing}")
    return missing


def validate_keys(tables: dict[str, pd.DataFrame]) -> list[str]:
    issues = []
    for tbl, pk in schema.PRIMARY_KEYS.items():
        if tbl not in tables:
            continue
        df = tables[tbl]
        if pk in df.columns and df[pk].duplicated().any():
            issues.append(f"{tbl}: duplicate primary keys on {pk}")
    for child, ccol, parent, pcol in schema.FOREIGN_KEYS:
        if child not in tables or parent not in tables:
            continue
        c, p = tables[child], tables[parent]
        if ccol not in c.columns or pcol not in p.columns:
            continue
        orphan = ~c[ccol].isin(set(p[pcol]))
        if orphan.any():
            issues.append(f"{child}.{ccol}: {int(orphan.sum())} rows with no matching "
                          f"{parent}.{pcol}")
    return issues


def validate_plausibility(tables: dict[str, pd.DataFrame]) -> list[str]:
    issues = []
    c = tables.get("claims")
    if c is not None:
        if (c["billed_hours"] <= 0).any():
            issues.append("claims: non-positive billed_hours present")
        if (c["billing_end_time"] < c["billing_start_time"]).any():
            issues.append("claims: billing_end_time before billing_start_time")
        if (c["submission_timestamp"] < c["billing_end_time"]).any():
            issues.append("claims: submission before service end")
    e = tables.get("evv_events")
    if e is not None and len(e):
        bad = (e["check_out_time"] < e["check_in_time"])
        if bad.any():
            issues.append(f"evv_events: {int(bad.sum())} rows check out before check in")
        lat = e["check_in_latitude"].dropna()
        if len(lat) and ((lat < -90) | (lat > 90)).any():
            issues.append("evv_events: latitude out of range")
    return issues


def missing_value_report(tables: dict[str, pd.DataFrame]) -> dict:
    return {name: {c: round(float(df[c].isna().mean()), 4)
                   for c in df.columns if df[c].isna().any()}
            for name, df in tables.items()}


def validate_all(tables: dict[str, pd.DataFrame]) -> dict:
    report = {
        "missing_columns": {n: validate_columns(df, n) for n, df in tables.items()},
        "key_issues": validate_keys(tables),
        "plausibility_issues": validate_plausibility(tables),
        "missing_values": missing_value_report(tables),
        "row_counts": {n: int(len(df)) for n, df in tables.items()},
    }
    report["passed"] = not (report["key_issues"] or report["plausibility_issues"])
    return report

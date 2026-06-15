"""Model A: transparent rules.

This is the baseline any ML model has to beat, and it is also the evidence
layer of the final system. Rules are cheap, auditable, and defensible in an
appeal, which matters more in a program-integrity setting than a marginal AUC
gain. Each rule emits a severity in [0, 1] and a human-readable sentence.

Thresholds are policy parameters, not learned values. They are set here to
defensible operational defaults and exposed in configs/model_config.yaml so a
reviewer can see and change them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    weight: float
    description: str


RULES = (
    Rule("billed_exceeds_authorization", 1.00,
         "Billed hours exceed the authorised daily hours by a material margin."),
    Rule("visit_overlap", 1.00,
         "This visit overlaps another visit billed by the same caregiver."),
    Rule("implausible_travel", 0.95,
         "Implied travel speed from the previous visit is physically impossible."),
    Rule("evv_duration_mismatch", 0.85,
         "Verified EVV visit length differs materially from the hours billed."),
    Rule("missing_evv", 0.55,
         "No EVV verification record is linked to this billed service."),
    Rule("exact_duplicate", 0.90,
         "An identical claim exists for the same caregiver, patient and time."),
    Rule("near_duplicate", 0.70,
         "A near-identical claim exists within the same hour."),
    Rule("timestamp_pattern", 0.60,
         "Clock times align exactly and submission was near-instant, "
         "consistent with batch entry rather than observed care."),
    Rule("excessive_daily_workload", 0.75,
         "Caregiver's billed hours for the day exceed a plausible shift."),
    Rule("after_hours", 0.35,
         "Service billed during overnight hours."),
)

RULES_BY_NAME = {r.name: r for r in RULES}


def _sev(x, lo, hi):
    """Ramp a value into [0, 1] between lo and hi."""
    return np.clip((np.asarray(x, dtype=float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def apply_rules(features: pd.DataFrame) -> pd.DataFrame:
    """Return one severity column per rule, aligned to `features`."""
    f = features
    out = pd.DataFrame(index=f.index)
    g = lambda c, d=0.0: f[c] if c in f.columns else pd.Series(d, index=f.index)  # noqa: E731

    ratio = g("billed_to_authorized_ratio", 1.0)
    out["billed_exceeds_authorization"] = _sev(ratio, 1.25, 3.0)

    out["visit_overlap"] = _sev(g("overlap_minutes_with_prev"), 1.0, 60.0)

    out["implausible_travel"] = _sev(g("implied_speed_kmh"), 110.0, 400.0)

    dur_ratio = g("evv_billing_duration_ratio", 1.0)
    out["evv_duration_mismatch"] = np.maximum(
        _sev(dur_ratio, 1.25, 2.5), _sev(1.0 / np.maximum(dur_ratio, 1e-6), 1.25, 2.5)
    )

    out["missing_evv"] = (1.0 - g("evv_present", 1.0)).clip(0, 1)

    out["exact_duplicate"] = _sev(g("exact_duplicate_group_size", 1.0), 1.0, 2.0)
    out["near_duplicate"] = _sev(g("near_duplicate_group_size", 1.0), 1.0, 3.0)

    out["timestamp_pattern"] = (
        g("evv_clock_exact_match") * 0.5
        + g("starts_on_hour") * 0.25
        + g("submission_immediate") * 0.25
    )

    out["excessive_daily_workload"] = _sev(g("caregiver_daily_hours"), 14.0, 26.0)

    out["after_hours"] = g("is_overnight").clip(0, 1)

    return out.fillna(0.0)


def rule_score(features: pd.DataFrame) -> pd.Series:
    """Weighted rule score in [0, 100].

    Weights are combined with a noisy-OR rather than a sum so that one severe
    rule already produces a high score and ten weak rules do not.
    """
    sev = apply_rules(features)
    prob_not = np.ones(len(sev))
    for name, col in sev.items():
        w = RULES_BY_NAME[name].weight
        prob_not *= (1.0 - np.clip(col.to_numpy() * w, 0.0, 0.999))
    return pd.Series(100.0 * (1.0 - prob_not), index=features.index)


def rule_evidence(features: pd.DataFrame, row_index, min_severity: float = 0.2) -> list[dict]:
    """Structured, human-readable evidence for one claim."""
    sev = apply_rules(features.loc[[row_index]]).iloc[0]
    ev = []
    for name, value in sev.items():
        if value >= min_severity:
            r = RULES_BY_NAME[name]
            ev.append({
                "rule": name,
                "severity": round(float(value), 3),
                "weight": r.weight,
                "statement": r.description,
            })
    return sorted(ev, key=lambda d: -d["severity"] * d["weight"])

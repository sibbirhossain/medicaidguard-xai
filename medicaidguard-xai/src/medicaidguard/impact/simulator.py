"""Scenario-based operational and economic simulator.

What this is
------------
A transparent arithmetic model that turns *assumptions you supply* into review
workload and an illustrative avoided-loss range. It is a sensitivity tool for
reasoning about review capacity, not a measurement.

What this is not
----------------
It is not an estimate of Medicaid savings. No number produced here has been
validated against a real recovery, and the detection performance it consumes
comes from synthetic data. Every output carries `DISCLAIMER`, and the dashboard
renders it next to every figure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DISCLAIMER = "Illustrative scenario-based estimate—not measured real-world savings."


@dataclass
class SimulationInputs:
    claims_reviewed_period: int = 100_000
    review_capacity_pct: float = 0.05
    investigator_hours_per_review: float = 1.5
    investigator_cost_per_hour: float = 65.0
    precision_at_capacity: float = 0.50
    recall_at_capacity: float = 0.70
    hypothetical_improper_rate: float = 0.03
    hypothetical_amount_per_case: float = 1_200.0
    recovery_realisation_rate: float = 0.35

    def to_dict(self):
        return asdict(self)


def simulate(inp: SimulationInputs) -> dict:
    """One scenario. All quantities are derived, none are measured."""
    alerts = int(round(inp.claims_reviewed_period * inp.review_capacity_pct))
    true_positives = alerts * inp.precision_at_capacity
    false_positives = alerts - true_positives

    review_hours = alerts * inp.investigator_hours_per_review
    review_cost = review_hours * inp.investigator_cost_per_hour
    wasted_review_cost = (false_positives * inp.investigator_hours_per_review
                          * inp.investigator_cost_per_hour)

    exposure = (inp.claims_reviewed_period * inp.hypothetical_improper_rate
                * inp.hypothetical_amount_per_case)
    addressed = true_positives * inp.hypothetical_amount_per_case
    realised = addressed * inp.recovery_realisation_rate

    return {
        "alerts_generated": alerts,
        "expected_true_positives": round(true_positives, 1),
        "expected_false_positives": round(false_positives, 1),
        "investigator_hours": round(review_hours, 1),
        "investigator_fte_equivalent": round(review_hours / 1_700, 2),
        "review_cost": round(review_cost, 2),
        "wasted_review_cost_on_false_positives": round(wasted_review_cost, 2),
        "hypothetical_period_exposure": round(exposure, 2),
        "hypothetical_amount_addressed": round(addressed, 2),
        "hypothetical_realised_recovery": round(realised, 2),
        "net_illustrative_position": round(realised - review_cost, 2),
        "break_even_precision": round(
            (inp.investigator_hours_per_review * inp.investigator_cost_per_hour)
            / max(inp.hypothetical_amount_per_case * inp.recovery_realisation_rate, 1e-9), 4),
        "disclaimer": DISCLAIMER,
    }


def sensitivity(inp: SimulationInputs, parameter: str,
                values) -> pd.DataFrame:
    """Sweep one assumption, holding everything else fixed."""
    rows = []
    for v in values:
        kwargs = inp.to_dict()
        kwargs[parameter] = v
        r = simulate(SimulationInputs(**kwargs))
        r[parameter] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs["disclaimer"] = DISCLAIMER
    return df


def capacity_curve(y_true, y_score, capacities=np.linspace(0.005, 0.25, 25),
                   inp: SimulationInputs | None = None) -> pd.DataFrame:
    """Join measured ranking performance to the workload model.

    Precision/recall at each capacity come from the actual held-out test
    predictions; everything downstream of them is assumption-driven.
    """
    from ..evaluation.metrics import precision_at_k, recall_at_k

    inp = inp or SimulationInputs()
    rows = []
    for cap in capacities:
        p = precision_at_k(y_true, y_score, float(cap))
        r = recall_at_k(y_true, y_score, float(cap))
        kwargs = inp.to_dict()
        kwargs.update(review_capacity_pct=float(cap),
                      precision_at_capacity=p, recall_at_capacity=r)
        out = simulate(SimulationInputs(**kwargs))
        out.update(capacity_pct=float(cap), measured_precision=p, measured_recall=r)
        rows.append(out)
    df = pd.DataFrame(rows)
    df.attrs["disclaimer"] = DISCLAIMER
    return df

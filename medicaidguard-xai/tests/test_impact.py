from medicaidguard.impact.simulator import (
    DISCLAIMER,
    SimulationInputs,
    sensitivity,
    simulate,
)


def test_simulation_arithmetic():
    out = simulate(SimulationInputs(
        claims_reviewed_period=100_000, review_capacity_pct=0.05,
        precision_at_capacity=0.4, investigator_hours_per_review=2.0,
        investigator_cost_per_hour=50.0))
    assert out["alerts_generated"] == 5000
    assert out["expected_true_positives"] == 2000.0
    assert out["investigator_hours"] == 10_000.0
    assert out["review_cost"] == 500_000.0


def test_disclaimer_is_always_attached():
    assert simulate(SimulationInputs())["disclaimer"] == DISCLAIMER
    df = sensitivity(SimulationInputs(), "precision_at_capacity", [0.2, 0.4])
    assert (df["disclaimer"] == DISCLAIMER).all()


def test_break_even_precision_is_meaningful():
    out = simulate(SimulationInputs(investigator_hours_per_review=1.0,
                                    investigator_cost_per_hour=100.0,
                                    hypothetical_amount_per_case=1000.0,
                                    recovery_realisation_rate=0.5))
    # 100 of cost against 500 of expected realised value per true positive.
    assert abs(out["break_even_precision"] - 0.2) < 1e-9

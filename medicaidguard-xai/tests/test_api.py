import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from medicaidguard.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_always_responds(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body


def test_openapi_schema_is_generated(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for p in ("/health", "/alerts", "/predict", "/batch-predict",
              "/model-info", "/scenario-simulation", "/data-quality"):
        assert p in paths


def test_scenario_simulation_is_pure_arithmetic(client):
    r = client.post("/scenario-simulation", json={
        "claims_reviewed_period": 10_000, "review_capacity_pct": 0.05,
        "precision_at_capacity": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["alerts_generated"] == 500
    assert body["expected_true_positives"] == 250.0
    assert "not measured real-world savings" in body["disclaimer"]


def test_invalid_claim_is_rejected(client):
    r = client.post("/predict", json={"claim_id": "X"})
    assert r.status_code == 422


def test_missing_alert_returns_404_or_503(client):
    r = client.get("/alerts/NOPE-999999")
    assert r.status_code in (404, 503)

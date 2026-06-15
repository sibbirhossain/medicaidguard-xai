"""Repository helpers: load the synthetic tables into the relational store."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .models import Base
from .session import get_engine, init_db

LOAD_ORDER = ["providers", "caregivers", "patients", "authorizations",
              "claims", "evv_events"]


def load_tables(tables: dict[str, pd.DataFrame], url: str | None = None,
                if_exists: str = "replace") -> dict[str, int]:
    """Bulk-load with pandas.to_sql, respecting foreign-key order."""
    engine = init_db(url)
    written = {}
    for name in LOAD_ORDER:
        if name not in tables:
            continue
        df = tables[name]
        cols = [c.name for c in Base.metadata.tables[name].columns]
        df = df[[c for c in cols if c in df.columns]]
        df.to_sql(name, engine, if_exists=if_exists, index=False, chunksize=5000)
        written[name] = len(df)
    return written


def store_predictions(scored: pd.DataFrame, model_version: str,
                      url: str | None = None) -> int:
    engine = init_db(url)
    out = scored[["claim_id", "risk_score", "model_probability",
                  "review_priority"]].copy()
    out["model_version"] = model_version
    out["scored_at"] = datetime.now(timezone.utc)
    out.to_sql("predictions", engine, if_exists="append", index=False, chunksize=5000)
    return len(out)


def top_alerts(limit: int = 100, url: str | None = None) -> pd.DataFrame:
    engine = get_engine(url)
    return pd.read_sql(
        "SELECT p.claim_id, p.risk_score, p.review_priority, c.provider_id, "
        "c.caregiver_id, c.service_date FROM predictions p "
        "JOIN claims c ON c.claim_id = p.claim_id "
        "ORDER BY p.risk_score DESC LIMIT ?", engine, params=(limit,))

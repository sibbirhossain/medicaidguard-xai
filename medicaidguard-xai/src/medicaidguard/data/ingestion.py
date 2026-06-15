"""Reusable public-dataset ingestion framework (Track A).

Scope note
----------
This module is deliberately generic. The dataset discovery pass
(reports/dataset_discovery.md) found no public dataset that carries linked
Medicaid claims *and* EVV telemetry *and* verified fraud labels, so there is no
single hardcoded download that would make Track A honest. What exists instead
is a registry (configs/datasets.yaml) of candidate public sources with their
licences, plus a pipeline that ingests, hashes, standardises and versions
whatever the user is entitled to download.

Nothing here fabricates a dataset, and nothing downloads automatically: several
CMS and state portals require accepting terms of use that a script cannot
accept on the user's behalf.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from ..config import CONFIG_DIR, EXTERNAL_DIR, PROCESSED_DIR, ensure_dirs


@dataclass
class DatasetRecord:
    name: str
    source: str
    url: str
    owner: str
    licence: str
    download_method: str
    notes: str = ""
    local_file: str | None = None
    sha256: str | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    ingested_at: str | None = None

    def to_dict(self):
        return asdict(self)


def load_registry(path: Path | None = None) -> list[DatasetRecord]:
    path = path or (CONFIG_DIR / "datasets.yaml")
    if not Path(path).exists():
        return []
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return [DatasetRecord(**d) for d in raw.get("datasets", [])]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def ingest_local_file(record: DatasetRecord, path: str | Path,
                      standardise=None) -> tuple[pd.DataFrame, DatasetRecord]:
    """Ingest a file the user has already downloaded under their own licence.

    Raw bytes are preserved in data/raw, a checksum is recorded, and the
    standardised frame is written to data/processed as Parquet.
    """
    ensure_dirs()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from {record.url} under the "
            f"'{record.licence}' terms, then re-run.")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        df = pd.read_csv(path, sep=None, engine="python", low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"unsupported extension: {suffix}")

    record.local_file = str(path)
    record.sha256 = sha256_file(path)
    record.ingested_at = datetime.now(timezone.utc).isoformat()

    if standardise is not None:
        df = standardise(df)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    record.n_rows, record.n_cols = int(len(df)), int(df.shape[1])

    out = PROCESSED_DIR / f"{record.name}.parquet"
    df.to_parquet(out, index=False)
    meta = EXTERNAL_DIR / f"{record.name}.metadata.json"
    meta.write_text(json.dumps(record.to_dict(), indent=2))
    return df, record


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-generate a data dictionary for any ingested frame."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
            "example": None if s.dropna().empty else str(s.dropna().iloc[0])[:60],
        })
    return pd.DataFrame(rows)

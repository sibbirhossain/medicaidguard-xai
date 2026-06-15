#!/usr/bin/env python3
"""Generate the synthetic Medicaid + EVV dataset and write Parquet tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicaidguard.config import SYNTHETIC_DIR, ensure_dirs, load_settings  # noqa: E402
from medicaidguard.data.generator import generate_and_save  # noqa: E402
from medicaidguard.data.validation import validate_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=int, default=None)
    ap.add_argument("--prevalence", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    ensure_dirs()
    s = load_settings(args.config)
    if args.claims:
        s.generator.n_claims = args.claims
    if args.prevalence is not None:
        s.generator.suspicious_prevalence = args.prevalence
    if args.seed is not None:
        s.generator.seed = args.seed

    tables = generate_and_save(s.generator)
    report = validate_all(tables)
    (SYNTHETIC_DIR / "generation_metadata.json").write_text(json.dumps({
        "config": s.generator.__dict__,
        "row_counts": {k: int(len(v)) for k, v in tables.items()},
        "validation": report,
    }, indent=2, default=str))

    for k, v in tables.items():
        print(f"{k:20s} {len(v):>8,} rows")
    print(f"\nsimulated suspicious prevalence: "
          f"{tables['scenario_labels']['is_suspicious'].mean():.4f}")
    print(f"written to {SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()

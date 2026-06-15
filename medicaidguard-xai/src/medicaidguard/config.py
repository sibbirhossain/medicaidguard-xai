"""Central configuration for MedicaidGuard-XAI.

All paths are resolved relative to the repository root so the project runs
unchanged on a laptop, in Docker, or in Google Colab. No personal paths are
hardcoded; override the root with the MEDICAIDGUARD_ROOT environment variable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    env = os.environ.get("MEDICAIDGUARD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py -> medicaidguard -> src -> repo root
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
RESULTS_DIR = REPORTS_DIR / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_DIR = ROOT / "configs"

GLOBAL_SEED = 20260913


def ensure_dirs() -> None:
    for d in (
        RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, EXTERNAL_DIR,
        FIGURES_DIR, TABLES_DIR, RESULTS_DIR, ARTIFACTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class GeneratorConfig:
    """Controls the synthetic Medicaid + EVV generator.

    `suspicious_prevalence` is the share of *claims* that carry at least one
    simulated suspicious scenario. Real program-integrity base rates are not
    publicly established at the claim level, so this is a controlled
    experimental parameter, not an estimate of reality.
    """

    n_providers: int = 60
    n_caregivers: int = 900
    n_patients: int = 2200
    n_claims: int = 45_000
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    n_regions: int = 6
    suspicious_prevalence: float = 0.035
    missing_evv_rate: float = 0.04          # benign data-quality gaps
    coordinate_noise_m: float = 45.0        # normal GPS jitter, metres
    seed: int = GLOBAL_SEED
    scenarios: list[str] = field(default_factory=lambda: list(ALL_SCENARIOS))


@dataclass
class SplitConfig:
    """Time-based split boundaries (inclusive of start, exclusive of end)."""

    train_end: str = "2024-08-31"
    valid_end: str = "2024-10-31"
    # test is everything after valid_end


@dataclass
class SelectionConfig:
    """How the *final* model is chosen.

    Accuracy is deliberately NOT the primary criterion. Under heavy class
    imbalance a constant "not suspicious" predictor achieves near-perfect
    accuracy while detecting nothing, so accuracy cannot discriminate between
    useful and useless models. It is still computed and reported.
    """

    primary_metric: str = "pr_auc"
    tiebreak_metric: str = "precision_at_5pct"
    max_brier: float = 0.10           # calibration floor; models above are excluded
    review_capacity_pcts: tuple[float, ...] = (0.01, 0.05, 0.10)


@dataclass
class Settings:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    seed: int = GLOBAL_SEED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALL_SCENARIOS = (
    "overlapping_visits",
    "authorization_exceedance",
    "implausible_travel",
    "evv_duration_mismatch",
    "missing_evv",
    "duplicate_claim",
    "near_duplicate_claim",
    "billing_spike",
    "excessive_workload",
    "provider_caregiver_concentration",
    "after_hours_activity",
    "weekend_activity",
    "provider_billing_shift",
    "suspicious_timestamps",
    "coordinated_activity",
    "authorization_inconsistency",
)

# Scenarios withheld from training in Experiment 5 (scenario generalization).
HELDOUT_SCENARIOS = ("coordinated_activity", "near_duplicate_claim", "suspicious_timestamps")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from YAML, falling back to dataclass defaults."""
    settings = Settings()
    if path is None:
        path = CONFIG_DIR / "base.yaml"
    path = Path(path)
    if not path.exists():
        return settings
    raw = yaml.safe_load(path.read_text()) or {}
    if "generator" in raw:
        settings.generator = GeneratorConfig(**raw["generator"])
    if "split" in raw:
        settings.split = SplitConfig(**raw["split"])
    if "selection" in raw:
        sel = dict(raw["selection"])
        if "review_capacity_pcts" in sel:
            sel["review_capacity_pcts"] = tuple(sel["review_capacity_pcts"])
        settings.selection = SelectionConfig(**sel)
    settings.seed = raw.get("seed", GLOBAL_SEED)
    return settings

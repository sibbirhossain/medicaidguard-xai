# Implementation Evidence

> Technical supporting material, not legal advice.

## Verification procedure

Any reviewer can reproduce every claim below:

```bash
git clone https://github.com/YOUR-USERNAME/medicaidguard-xai.git
cd medicaidguard-xai
pip install -r requirements.txt && pip install -e .
make all
```

Runtime roughly 20 minutes on a single CPU core. Compare the regenerated
`reports/tables/` and `reports/results/` against the committed versions; only
the timestamp and hardware fields in `run_summary.json` should differ.

## Codebase

| Area | Modules | Notes |
|---|---|---|
| Configuration | `config.py` | Dataclass settings, YAML loading, path resolution via `MEDICAIDGUARD_ROOT` |
| Data | `data/schema.py`, `generator.py`, `validation.py`, `ingestion.py`, `metadata.py` | Schema contracts, 16-scenario generator, validation, licensed ingestion, run provenance |
| Features | `features/billing.py`, `evv.py`, `temporal.py`, `network.py`, `pipeline.py` | 101 features, fit/transform contract |
| Models | `models/rules.py`, `supervised.py`, `anomaly.py`, `fusion.py`, `calibration.py` | Rules, four learners, two detectors, two fusion strategies, isotonic calibration |
| Evaluation | `evaluation/metrics.py`, `splits.py`, `leakage.py`, `confidence_intervals.py`, `experiments.py` | Metrics, splitting, leakage guards, statistics, experiment suite |
| Explainability | `explainability/shap_explainer.py`, `evidence.py` | SHAP wrapper, alert assembly |
| Impact | `impact/simulator.py` | Scenario-based workload and economics |
| API | `api/main.py`, `schemas.py` | 14 endpoints, Pydantic contracts |
| Database | `database/models.py`, `session.py`, `repository.py` | SQLAlchemy ORM, PostgreSQL-compatible |
| Dashboard | `dashboard/app.py` | Eight pages |

## Test suite

43 tests, all passing. Coverage by concern:

| Module | Concern |
|---|---|
| `test_generator.py` | Seed reproducibility, prevalence configurability, label separation |
| `test_validation.py` | Schema, referential integrity, plausibility |
| `test_features.py` | Feature correctness, travel plausibility, overlap detection, no NaN/inf, fit-before-transform contract |
| `test_no_leakage.py` | Column blocklist, planted-label rejection, no perfect predictor, chronological split, train-only statistic fitting |
| `test_models.py` | Precision@K correctness, calibration improves ECE, rule bounds, priority band coverage, selection rule ignores accuracy and enforces the calibration floor |
| `test_explanations.py` | Evidence structure, disclaimer propagation, priority filtering |
| `test_api.py` | Health, OpenAPI schema, validation rejection, simulator arithmetic |
| `test_impact.py` | Simulator arithmetic, disclaimer attachment, break-even derivation |

**The tests found two real defects during development.** An independently drawn
EVV check-out offset was inverting 36% of visit intervals on short visits, and
an empty EVV table crashed feature construction through a dtype path. Both were
fixed and all results regenerated afterwards. This is recorded because a test
suite that has never caught anything is not evidence of quality.

## Continuous integration

`.github/workflows/ci.yml` runs on push and pull request across Python 3.10 and
3.12: lint (`ruff`), full test suite, an end-to-end pipeline smoke run, results
upload, dependency audit (`pip-audit`), and a check that fails the build if any
data file is tracked outside `reports/`.

## Reproducibility controls

- Single global seed in `configs/base.yaml`; `numpy.random.default_rng` with no
  reliance on global state; explicit `random_state` on every estimator.
- `n_jobs=1` by default, trading wall-clock time for exact reproducibility.
- Run provenance captured automatically: git commit, Python version, platform,
  and the version of every relevant package.
- Second-resolution timestamps so the generator is stable across pandas
  microsecond and nanosecond datetime regimes.

## Documentation

Roughly 6,000 words across README, methodology, architecture, data dictionary
(auto-generated from the real tables), privacy and responsible AI, deployment,
reproducibility, dataset discovery report, research report, limitations,
security policy, contributing guide, code of conduct and changelog.

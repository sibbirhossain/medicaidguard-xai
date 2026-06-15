# Technical Project Summary

**Project:** MedicaidGuard-XAI
**Author:** Md Sibbir Hossain · ORCID 0009-0002-0795-4512
**Repository:** https://github.com/YOUR-USERNAME/medicaidguard-xai
**License:** MIT

> **These are technical supporting materials, not legal advice.** They describe
> what was built and what it demonstrates. They do not assert, and must not be
> used to assert, that this project alone establishes eligibility for any
> immigration classification. That determination rests with counsel and USCIS.

---

## 1. What was built

A complete, reproducible machine learning platform that prioritises suspicious
Medicaid personal-care billing and Electronic Visit Verification activity for
human review.

| Component | Description |
|---|---|
| Synthetic data generator | Linked provider/caregiver/patient/authorisation/claim/EVV schema, 16 configurable suspicious scenarios, reproducible seeds |
| Public-data ingestion framework | Licence tracking, checksums, versioning, auto-generated data dictionaries |
| Feature pipeline | 101 features across five families, fitted on the training window only |
| Detection engine | Rule engine, four supervised learners, two unsupervised detectors, two fusion strategies |
| Calibration layer | Isotonic regression fitted on validation |
| Statistical validation | Bootstrap, cluster bootstrap, paired bootstrap, DeLong, McNemar |
| Explainability | SHAP global/local, structured rule evidence, permutation importance cross-check |
| Impact simulator | Scenario-based workload and illustrative economics |
| Backend | FastAPI, 14 endpoints, Pydantic validation |
| Dashboard | Eight-page Streamlit investigator interface |
| Quality | 43 automated tests, GitHub Actions CI, Docker, Colab notebook |

Scale: 45,297 synthetic claims, 43,106 EVV events, 101 features, nine models
compared under time-based validation.

## 2. Measured results

Produced by `make all` at seed 20260913 on synthetic data with simulated labels.

- Selected model (LightGBM) PR-AUC **0.889**, caregiver-clustered bootstrap CI
  [0.860, 0.917]; ROC-AUC 0.985 [0.977, 0.992].
- Transparent rule baseline PR-AUC 0.354; difference +0.535, CI [0.481, 0.585].
- Feature ablation: billing-only 0.437, EVV-only 0.593, combined **0.740**.
- Calibration after isotonic regression: Brier 0.0088, ECE 0.0028.
- Efficiency: 5.6 s training, 0.86 s inference on 7,233 claims, single CPU core.

Two results contradicted the framework's own hypotheses and are reported as
negative findings rather than suppressed: hybrid fusion did not beat the best
individual model, and the selected model is not statistically separable from a
random forest.

## 3. Technical skills demonstrated

- **Applied machine learning:** multi-model comparison, class-imbalance
  handling, probability calibration, hyperparameter selection under an explicit
  trade-off between detection quality, calibration and compute cost.
- **Research methodology:** time-based and group-aware validation, three-layer
  leakage control with automated verification, bootstrap and cluster-bootstrap
  confidence intervals, DeLong and McNemar tests, ablation design, held-out
  scenario generalisation.
- **Data engineering:** relational schema design with referential integrity,
  Parquet storage, vectorised feature engineering at 3.6 s per 45k rows,
  PostgreSQL-compatible ORM layer.
- **Explainable AI:** SHAP integration with an explicit interpretation boundary,
  structured evidence generation, model-agnostic cross-checking.
- **Software engineering:** modular package layout, 43 automated tests, CI
  across two Python versions, containerisation, typed API contracts.
- **Responsible AI:** privacy threat model, documented decision boundaries,
  human-in-the-loop design, bias analysis, foreseeable-misuse register.

## 4. Domain grounding

The author works in billing and technology administration at a New York home
care agency, covering Medicaid billing automation, EVV compliance tracking,
multi-payer authorisation workflows and HHAeXchange platform management.

That background shaped specific design decisions that are visible in the code:
the 16 scenarios reflect documentation patterns that occur in practice; the
missing-EVV rule carries a deliberately low weight (0.55 against 0.85–1.00 for
hard rules) because EVV gaps track connectivity and device access more than
misconduct; and the rule layer is retained despite weak standalone performance
because rule evidence is what survives an appeal.

**No employer data, no PHI, and no proprietary information is present in this
repository.** All data is synthetic. This separation is deliberate and
documented in `SECURITY.md` and `docs/privacy.md`.

## 5. Verifiable artefacts

| Artefact | Location |
|---|---|
| Source code | `src/medicaidguard/` — 25 modules |
| Test suite | `tests/` — 43 tests, all passing |
| Reproducibility guide | `docs/reproducibility.md` |
| Dataset discovery report | `reports/dataset_discovery.md` |
| Research report | `reports/research_report.md` |
| Results tables | `reports/tables/`, `reports/results/` |
| Run provenance | `reports/results/run_summary.json` |
| Colab notebook | `notebooks/MedicaidGuard_XAI_Colab.ipynb` |
| CI configuration | `.github/workflows/ci.yml` |

Every claim above is checkable by running `make all` and comparing against the
committed tables.

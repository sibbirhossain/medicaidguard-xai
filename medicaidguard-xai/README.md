# MedicaidGuard-XAI

**Safeguarding Public Healthcare Spending with Explainable AI: A Statistically Validated Machine Learning Framework for Medicaid Fraud Detection and Electronic Visit Verification**

> **Synthetic/public research demonstration. Alerts are review recommendations, not findings of fraud.**
> Every metric below was produced on synthetic data with **simulated** labels. Nothing here measures real-world Medicaid fraud detection, and no dollar figure in this repository is a measured saving.

---

## What this is

A research prototype that prioritises suspicious Medicaid personal-care billing and Electronic Visit Verification (EVV) activity for human review. It combines transparent rules, supervised learning, unsupervised anomaly detection and hybrid risk fusion, with probability calibration, SHAP explanations, and statistical validation that reports confidence intervals rather than point estimates.

It is built to be defensible under scrutiny, which mostly means it is built to avoid the four ways this kind of project usually goes wrong: label leakage, random splits on time-ordered data, uncalibrated scores presented as probabilities, and accuracy reported without its trivial baseline.

**Run it now:** [`notebooks/MedicaidGuard_XAI_Colab.ipynb`](notebooks/MedicaidGuard_XAI_Colab.ipynb) runs the whole pipeline on Colab's free CPU tier in 10–20 minutes.

---

## Research question

> Can an explainable hybrid framework combining billing behaviour, EVV consistency, temporal patterns, statistical deviations and provider–caregiver relationships improve the prioritisation of suspicious Medicaid-related activity compared with conventional rules and individual machine learning models, while maintaining calibration, transparency, robustness and computational efficiency?

### Hypotheses and what actually happened

| # | Hypothesis | Result |
|---|---|---|
| H1 | Learned models beat transparent rules | **Supported.** PR-AUC 0.889 vs 0.354; difference +0.535, 95% CI [0.481, 0.585] |
| H2 | Billing + EVV features beat either family alone | **Supported, and it is the strongest result here.** 0.437 (billing) / 0.593 (EVV) → 0.740 combined |
| H3 | Hybrid fusion beats the best individual model | **Not supported.** Best hybrid 0.865 vs LightGBM 0.889, difference +0.024 favouring the single model, CI [0.013, 0.037] |
| H4 | Scores can be calibrated to usable probabilities | **Supported.** Brier 0.0088, ECE 0.0028 after isotonic calibration on the validation window |
| H5 | Performance degrades on scenarios never labelled in training | **Supported.** Held-out scenarios recover at 0.42–0.90 recall@5% vs 1.00 for most trained scenarios |

H3 is a negative result on the project's headline claim. It is reported as such. The hybrid framework's case has to rest on rule evidence and appeal defensibility, not on ranking performance.

---

## Results

All figures: 45,297 synthetic claims, 43,106 EVV events, 101 features, time-based split (train ≤ 2024-08-31, validate ≤ 2024-10-31, test thereafter). Test window: 7,233 claims at 3.97% prevalence. Seed 20260913.

### Model comparison — held-out test window

| Model | PR-AUC | ROC-AUC | P@5% | R@5% | Accuracy | Trivial baseline | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** *(selected)* | **0.889** | 0.985 | 0.674 | 0.850 | 0.988 | 0.960 | 0.0088 | 0.0028 |
| Random forest | 0.879 | 0.988 | 0.671 | 0.847 | 0.987 | 0.960 | 0.0103 | 0.0018 |
| Hybrid (stacked) | 0.865 | 0.983 | 0.669 | 0.843 | 0.989 | 0.960 | 0.0088 | 0.0032 |
| Hybrid (weighted) | 0.846 | 0.978 | 0.657 | 0.829 | 0.990 | 0.960 | 0.0090 | 0.0015 |
| Gradient boosting | 0.835 | 0.987 | **0.682** | **0.861** | 0.985 | 0.960 | 0.0112 | 0.0022 |
| Logistic regression | 0.788 | 0.974 | 0.624 | 0.787 | 0.983 | 0.960 | 0.0137 | 0.0031 |
| Local Outlier Factor | 0.407 | 0.757 | 0.373 | 0.470 | 0.965 | 0.960 | — | — |
| Rules only | 0.354 | 0.843 | 0.354 | 0.446 | 0.938 | 0.960 | 0.0290 | 0.0014 |
| Isolation Forest | 0.211 | 0.823 | 0.246 | 0.310 | 0.933 | 0.960 | — | — |

Selected model PR-AUC with caregiver-clustered bootstrap CI: **0.889 [0.860, 0.917]** (300 resamples).
ROC-AUC: 0.985 [0.977, 0.992] (500 resamples).

### Read the accuracy column before drawing any conclusion from it

At 3.97% prevalence, a model that predicts "not suspicious" for every claim scores **0.960 accuracy while detecting nothing**. Every learned model above sits between 0.983 and 0.990 — a spread of 0.7 percentage points across models whose PR-AUC ranges from 0.788 to 0.889. Meanwhile the rules baseline *underperforms* the do-nothing classifier on accuracy (0.938) while being genuinely useful.

Selecting on accuracy would have picked **hybrid-weighted** (0.990), which ranks fourth on PR-AUC. This project therefore selects on PR-AUC subject to a Brier calibration floor, with Precision@5% as tiebreaker. Accuracy is reported in every table beside its trivial baseline so the comparison cannot be skipped.

### Is the winner actually better?

| Comparison | ΔPR-AUC | 95% CI | DeLong *p* (ROC) |
|---|---|---|---|
| LightGBM vs rules only | +0.535 | [0.481, 0.585] | <0.001 |
| LightGBM vs Isolation Forest | +0.679 | [0.622, 0.719] | <0.001 |
| LightGBM vs Local Outlier Factor | +0.482 | [0.417, 0.541] | <0.001 |
| LightGBM vs logistic regression | +0.101 | [0.069, 0.139] | 0.044 |
| LightGBM vs hybrid (weighted) | +0.043 | [0.028, 0.060] | 0.043 |
| LightGBM vs hybrid (stacked) | +0.024 | [0.013, 0.037] | 0.230 |
| LightGBM vs gradient boosting | +0.054 | [0.028, 0.085] | 0.460 |
| **LightGBM vs random forest** | **+0.010** | **[−0.002, 0.021]** | **0.242** |

**The selected model is not significantly better than random forest.** The PR-AUC interval crosses zero and DeLong finds no ROC difference. The defensible reason to deploy LightGBM here is not accuracy — it is that it trains in 5.6 s versus 57.9 s for random forest and 79.3 s for gradient boosting, at equivalent detection quality. State the speed argument, not an accuracy argument.

### Feature ablation — the strongest finding

| Feature set | Features | PR-AUC | ROC-AUC | P@5% | R@5% |
|---|---|---|---|---|---|
| Billing only | 44 | 0.437 | 0.788 | 0.354 | 0.446 |
| EVV only | 36 | 0.593 | 0.838 | 0.492 | 0.620 |
| Billing + EVV | 72 | 0.740 | 0.935 | 0.583 | 0.735 |
| + temporal | 87 | 0.836 | 0.976 | 0.638 | 0.805 |
| + statistical | 93 | 0.839 | 0.976 | 0.638 | 0.805 |
| + network (full) | 101 | **0.889** | 0.985 | 0.674 | 0.850 |

Billing and EVV each carry real but partial signal; combined they gain +0.147 over the better half alone. That is the substantive claim this project supports. Note also that the statistical block adds almost nothing (+0.003) once temporal features are present — an honest finding that argues for dropping it, not for keeping it because it sounds sophisticated.

### Scenario generalisation

Three scenarios were relabelled negative during training, simulating a scheme nobody has investigated.

| Scenario | Held out | Recall@5% |
|---|---|---|
| Excessive workload, authorisation exceedance, billing spike, duplicate claim, EVV duration mismatch, provider concentration, weekend, after-hours | No | 1.000 |
| Implausible travel | No | 0.905 |
| **Suspicious timestamps** | **Yes** | **0.900** |
| Provider billing shift | No | 0.889 |
| **Coordinated activity** | **Yes** | **0.769** |
| Authorisation inconsistency | No | 0.722 |
| Overlapping visits | No | 0.500 |
| **Near-duplicate claim** | **Yes** | **0.417** |
| Missing EVV | No | 0.188 |

Held-out scenarios are recovered, but unevenly and worse than trained ones. Missing EVV at 0.188 is the operationally important number: a missing verification record is weak evidence, and the model has correctly learned not to over-weight it — which is the behaviour you want, since EVV gaps track smartphone access and connectivity more than misconduct.

### Class imbalance

| Injected prevalence | Observed | PR-AUC | P@5% | R@5% | Brier |
|---|---|---|---|---|---|
| 1.0% | 0.0115 | 0.843 | 0.206 | 0.892 | 0.0027 |
| 2.0% | 0.0212 | 0.889 | 0.388 | 0.915 | 0.0039 |
| 3.5% | 0.0397 | 0.889 | 0.674 | 0.850 | 0.0088 |
| 6.0% | 0.0674 | 0.918 | 0.995 | 0.737 | 0.0133 |

Ranking quality holds down to 1% prevalence. Precision@5% moves almost entirely with base rate, which is arithmetic rather than a property of the model — worth knowing before quoting a precision figure without its prevalence.

### Operational capacity (Experiment 7)

| Reviewed | Precision | Recall | Alerts / 100k claims | Investigator FTE |
|---|---|---|---|---|
| 0.5% | 1.000 | 0.125 | 500 | 0.44 |
| 2.5% | 1.000 | 0.641 | 2,542 | 2.24 |
| 4.6% | 0.732 | 0.847 | 4,583 | 4.04 |
| 9.7% | 0.382 | 0.934 | 9,688 | 8.55 |
| 25.0% | 0.157 | 0.990 | 25,000 | 22.06 |

FTE figures assume 1.5 review hours per alert and 1,700 productive hours per year — **assumptions, not measurements**.

### Fairness diagnostics (Experiment 8)

Across synthetic cohorts, PR-AUC ranges 0.865–0.932 and FPR ranges 0.031–0.047. **These numbers say nothing about real populations** — the cohort labels are randomly assigned by the generator. The experiment validates that the measurement pipeline works, so it can be pointed at real cohorts under appropriate governance. See [`docs/privacy.md`](docs/privacy.md) for the bias risks that would be real in deployment.

### Computational efficiency (Experiment 10)

Single CPU core, 45,297 claims × 101 features.

| Stage | Seconds |
|---|---|
| Feature engineering (all 45k rows) | 3.62 |
| Rule screening | 0.01 |
| LightGBM training | 5.61 |
| Random forest training | 57.89 |
| Gradient boosting training | 79.34 |
| Isolation Forest fit | 1.51 |
| LightGBM inference (7,233 claims) | 0.86 |

---

## Data

**No public dataset links Medicaid personal-care claims to EVV telemetry with verified fraud labels.** EVV data is operational PHI held by state Medicaid agencies and their aggregators; fraud determinations are legal outcomes published as case narratives and exclusion lists, not row-level flags; and public CMS claims files are aggregated by design so members cannot be re-identified.

[`reports/dataset_discovery.md`](reports/dataset_discovery.md) documents the full search across CMS Data, Data.Medicaid.gov, Data.gov, HHS OIG, state portals, Kaggle, UCI, PhysioNet, Harvard Dataverse, Zenodo and Figshare, and explains why the two closest candidates (the OIG LEIE exclusions list and the popular Kaggle provider-fraud datasets) were rejected as primary sources rather than used because they were convenient.

| Track | Status |
|---|---|
| A — Real public data | Ingestion framework built (licences, checksums, versioning, data dictionaries). Public CMS material informs feature *design* and background framing only; no public row contributes to any reported result. |
| B — Synthetic Medicaid + EVV | **Primary.** 16 configurable scenarios, reproducible seeds, scenario metadata in a separate table. |
| C — Cross-dataset validation | **Not performed.** No compatible dataset pair exists. Merging incompatible sources to manufacture a result would be worse than reporting its absence. |

---

## Install and run

```bash
git clone https://github.com/sibbirhossain/medicaidguard-xai.git
cd medicaidguard-xai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

make data         # generate the synthetic dataset
make train        # train, compare, select, calibrate
make experiments  # full experiment suite -> reports/tables/
make test         # 43 tests
```

Or: `make all`.

### Dashboard and API

```bash
make dashboard    # http://localhost:8501
make api          # http://localhost:8000/docs
```

### Docker

```bash
docker compose up --build
```

Generates data, trains, then serves the API on 8000 and the dashboard on 8501.

### Google Colab

Open [`notebooks/MedicaidGuard_XAI_Colab.ipynb`](notebooks/MedicaidGuard_XAI_Colab.ipynb) in Colab. It installs dependencies, runs the full pipeline, renders the curves and explanations, runs the tests, and includes a cell that pushes the repository to GitHub using a token entered through `getpass` so it is never written into the saved notebook.

---

## API

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "claim_id":"TEST001","provider_id":"PRV0001","caregiver_id":"CG00001",
  "patient_id":"PT000001","service_date":"2024-11-15",
  "billing_start_time":"2024-11-15T09:00:00","billing_end_time":"2024-11-15T17:00:00",
  "billed_hours":8.0,"authorized_hours":4.0,"service_code":"T1019",
  "billed_amount":220.0,"submission_timestamp":"2024-11-15T17:01:00"}'
```

Endpoints: `/health` `/metrics` `/alerts` `/alerts/{id}` `/providers/{id}` `/caregivers/{id}` `/evv/{id}` `/predict` `/batch-predict` `/model-info` `/explanations/{id}` `/experiments` `/data-quality` `/scenario-simulation`.

No endpoint changes a claim's status, denies a service, or alters an authorisation. That is deliberate and documented in [`SECURITY.md`](SECURITY.md).

---

## How leakage is prevented

The failure mode that invalidates most fraud-detection results, addressed three ways and tested:

1. **Column-level** — label and provenance columns are blocklisted in `schema.LEAKAGE_COLUMNS`; entity IDs are excluded from features so trees cannot memorise a caregiver.
2. **Temporal** — every entity aggregate is strictly backward-looking (`sort → groupby → shift(1) → rolling/expanding`). A plain `groupby().transform()` would let a caregiver's later claims inform their earlier ones.
3. **Fit-window** — peer medians/MADs, network degrees, categorical vocabularies and imputation values are fitted on the training window only. `test_peer_statistics_are_fitted_on_train_only` refits on a truncated dataset and requires identical statistics.

A diagnostic guard flags any single feature reaching AUC ≥ 0.999 on training. On the current run: **none**.

---

## Explainability

Every alert carries structured rule evidence in plain sentences, the calibrated probability with its calibration method, SHAP local attributions, and this limitation, verbatim:

> Feature attributions describe how this model reached its score. They are not evidence of intent, causation, or wrongdoing, and they do not constitute a finding of fraud.

The rule layer is retained despite scoring a third of the best model's PR-AUC, because rule evidence is what a reviewer can restate as a factual assertion about the claim, and what survives an appeal.

---

## Limitations

Read these before citing anything above.

1. **Synthetic labels.** Results measure recovery of injected behavioural patterns, not detection of real fraud. No claim about real-world performance is supported.
2. **Shared authorship of generator and detector.** Scenarios are detectable partly because they were designed in a representable form. Experiment 5 is a partial defence, not a complete one.
3. **The hybrid framework did not beat the best single model** (H3 unsupported).
4. **The selected model is not significantly better than random forest**; the speed difference is the real argument.
5. **Single simulated year.** Multi-year drift untested.
6. **Cross-dataset validation not performed**, for the reasons in the discovery report.
7. **Fairness results are about synthetic cohorts** and transfer to no real population.
8. **Impact simulator outputs are arithmetic on your assumptions**, labelled "Illustrative scenario-based estimate—not measured real-world savings" on every field.
9. **Ablation uses a restricted learner set** for runtime; recorded in `configs/experiments.yaml`.

See [`reports/limitations.md`](reports/limitations.md) and [`docs/methodology.md`](docs/methodology.md).

---

## Documentation

| Document | Contents |
|---|---|
| [`reports/dataset_discovery.md`](reports/dataset_discovery.md) | Full public-dataset search, evaluation, and selection rationale |
| [`reports/research_report.md`](reports/research_report.md) | Research report built from the actual results |
| [`reports/limitations.md`](reports/limitations.md) | Consolidated limitations |
| [`docs/methodology.md`](docs/methodology.md) | Problem formulation, splitting, leakage control, statistics |
| [`docs/architecture.md`](docs/architecture.md) | Pipeline diagrams and module map |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every table and field |
| [`docs/privacy.md`](docs/privacy.md) | Privacy threat model, responsible AI, bias analysis, misuse |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeds, pinning, nondeterminism, verification |
| [`docs/deployment.md`](docs/deployment.md) | Local, Docker, and production considerations |
| [`SECURITY.md`](SECURITY.md) | Threat model and disclosure |

---

## Citation

```bibtex
@software{hossain_medicaidguard_xai_2026,
  author  = {Hossain, Md Sibbir},
  title   = {MedicaidGuard-XAI: An Explainable, Statistically Validated Framework
             for Prioritising Suspicious Medicaid Billing and EVV Activity},
  year    = {2026},
  url     = {https://github.com/sibbirhossain/medicaidguard-xai},
  license = {MIT},
  note    = {Results produced on synthetic data with simulated labels}
}
```

See [`CITATION.cff`](CITATION.cff). ORCID: [0009-0002-0795-4512](https://orcid.org/0009-0002-0795-4512).

---

## License

MIT — see [`LICENSE`](LICENSE).

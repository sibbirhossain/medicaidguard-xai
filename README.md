# MedicaidGuard-XAI

**Safeguarding Public Healthcare Spending with Explainable AI: A Statistically Validated Machine Learning Framework for Medicaid Fraud Detection and Electronic Visit Verification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)]()
[![Tests: 43 passing](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)]()
[![Peer-reviewed](https://img.shields.io/badge/published-Frontiers%20in%20CS%20%26%20AI%20(2026)-blueviolet.svg)](https://doi.org/10.32996/jcsts.2026.5.3.4)

> Alerts produced by this system are review recommendations, not findings of fraud. Every metric in the Results section below was produced on synthetic data with simulated labels — this is stated once here and repeated at every point where it could otherwise be forgotten.

---

## Overview

Medicaid personal-care programs generate two records of the same event — a **billing claim** and an **Electronic Visit Verification (EVV)** check-in/check-out record — and almost nobody models them together, because no public dataset links them with verified fraud outcomes. **MedicaidGuard-XAI** is an open-source, statistically validated, explainable machine-learning framework built to close that specific methodological gap: it fuses billing behavior, EVV consistency, temporal patterns, robust statistical deviations, and provider–caregiver network structure into a single, calibrated, auditable risk score that prioritizes suspicious activity for human review.

It is the open, reproducible extension of a peer-reviewed research program (citation below) by the same author, re-engineered as a full pipeline — synthetic data generation, 101-feature engineering, nine competing detection models, rigorous statistical validation, SHAP-based explainability, a FastAPI backend, an investigator dashboard, Docker packaging, and a 43-test CI suite — built specifically so that every number it produces can be independently regenerated and checked, rather than taken on faith.

---

## The problem

**Medicaid is one of the largest single line items in American public spending, and a measurable share of what it pays out is wrong.** In fiscal year 2025, the Centers for Medicare & Medicaid Services (CMS) reported a Medicaid improper payment rate of **6.12%**, equivalent to **$37.39 billion**, against total federal-and-state Medicaid outlays of roughly **$908.8 billion** in FY2024 — spending that sits inside a U.S. health-care sector now approaching **$5.3 trillion**, or 18% of GDP.¹ Most of that improper-payment total reflects insufficient documentation rather than proven fraud, and the honest framing of that distinction — not blurring it — is the starting point for any credible work in this space.

Personal, in-home care is where the integrity problem is structurally hardest. A home visit has no institutional witness: the only record that a service occurred is a verification event and a claim for payment. Congress recognized this gap in **Section 12006(a) of the 21st Century Cures Act**, which mandated that every state implement Electronic Visit Verification for Medicaid personal-care services (compliance deadline January 1, 2020) and home health services (January 1, 2023), with states facing incremental FMAP reductions of up to 1% for non-compliance.² That mandate created, for the first time at national scale, a second, independent data stream that can be checked against what is billed — but published methodology for using EVV and billing data *jointly* is essentially absent, because the linked data itself is not public.

The scale of what depends on getting this right is growing, not shrinking. Home health and personal care aides are now the **single largest occupation in the United States**, at roughly **4.68 million workers**, with the Bureau of Labor Statistics projecting **18% growth and 847,300 new jobs by 2035** as care continues to shift from institutions into homes.³ Review capacity, meanwhile, does not scale with claim volume — state program-integrity units are small, fixed teams facing a growing stream of claims. The operational question a program actually faces is not "is this claim fraudulent," but **"which few thousand claims should a finite team examine first"** — a ranking problem under a hard capacity constraint, where a wrong answer in either direction has a real cost: missed integrity risk on one side, and a caregiver's suspended income or a beneficiary's interrupted care on the other.

That is the problem this project is built to address, and the reason its design choices — calibration over raw scores, deliberately low weight on access-correlated signals like missing EVV, explanations built to survive an appeal, and negative results reported rather than tuned away — are treated as the core contribution, not as caveats around it.

---

## Research foundation

This work sits on two connected pieces of research by the same author.

**1. A peer-reviewed publication** established the core research question and an initial validated result:

> Hossain, M. S., & Hasan, A T M F. (2026). *Safeguarding Public Healthcare Spending with Explainable AI: A Statistically Validated Machine Learning Framework for Medicaid Fraud Detection and Electronic Visit Verification.* **Frontiers in Computer Science and Artificial Intelligence**, 5(3), 35–44. https://doi.org/10.32996/jcsts.2026.5.3.4

That study, built on 5,410 providers and over 500,000 claims with a 17-dimensional engineered feature set, found Random Forest to be the strongest of the models compared (F1 = 0.649, ROC-AUC = 0.950), used SHAP to identify total reimbursement, average length of stay, and claims-per-beneficiary as leading predictive signals, and showed graceful degradation under noise and missing data — evidence that the underlying signal is real and that explainability holds up under stress.

**2. This repository** is the full, open-source engineering extension of that research program: a complete pipeline that adds the EVV integration the published study's abstract does not cover, expands from 17 to 101 features across five families, replaces a single best-model comparison with a nine-model, statistically validated benchmark (bootstrap and cluster-bootstrap confidence intervals, DeLong and McNemar tests), and adds production infrastructure — an API, a dashboard, Docker packaging, and a 43-test CI suite — so the whole pipeline, not just a result, is reproducible and inspectable by anyone.

The two studies report different numbers because they answer related but distinct questions on different data (see **Data**, below) — this README does not merge them, and neither should a reader.

---

## What the framework does

| Component | What it does |
|---|---|
| Synthetic data generator | Linked provider / caregiver / patient / authorization / claim / EVV schema; 16 configurable suspicious scenarios; reproducible seeds |
| Public-data ingestion framework | License tracking, checksums, versioning, auto-generated data dictionaries for real CMS material |
| Feature pipeline | 101 features across billing, EVV, temporal, robust-statistical, and network families; strictly fit on the training window |
| Detection engine | A transparent rule engine, four supervised learners, two unsupervised anomaly detectors, and two fusion strategies |
| Calibration | Isotonic regression fitted on a held-out validation window, applied to every model on equal footing |
| Statistical validation | Percentile and caregiver-clustered bootstrap, paired bootstrap on metric differences, DeLong's test, McNemar's test |
| Explainability | SHAP global and local attributions, structured plain-language rule evidence, permutation-importance cross-check |
| Impact simulator | Converts operating-point choices into workload and illustrative economics, with every output explicitly labeled as an estimate |
| API | FastAPI, 14 endpoints, typed Pydantic contracts |
| Dashboard | 8-page Streamlit investigator interface |
| Quality | 43 automated tests, GitHub Actions CI across two Python versions, Docker, a one-click Colab notebook |

---

## Results

All figures below: 45,297 synthetic claims, 43,106 linked EVV events, 101 features, time-based split (train ≤ 2024-08-31, validate ≤ 2024-10-31, test thereafter), 7,233-claim test window at 3.97% simulated prevalence, seed 20260913. Every number is reproduced by running `make all` and is checked automatically in CI.

### Hypotheses tested

| # | Hypothesis | Result |
|---|---|---|
| H1 | Learned models beat transparent rules | **Supported.** PR-AUC 0.889 vs 0.354; Δ +0.535, 95% CI [0.481, 0.585] |
| H2 | Billing + EVV features beat either family alone | **Supported — the strongest finding in the study.** 0.437 (billing) / 0.593 (EVV) → 0.740 combined |
| H3 | Hybrid fusion beats the best single model | **Not supported.** Best hybrid 0.865 vs LightGBM 0.889, favoring the single model, CI [0.013, 0.037] |
| H4 | Scores can be calibrated to usable probabilities | **Supported.** Brier 0.0088, ECE 0.0028 after isotonic calibration |
| H5 | Performance holds on scenarios never labeled in training | **Supported, with degradation.** Held-out scenarios recover at 0.42–0.90 recall@5% vs ~1.00 for trained scenarios |

H3 is reported as a negative result on the project's own headline claim, not tuned away. The two hypotheses that *are* strongly supported (H1, H2) are the ones the deployment argument actually rests on.

### Model comparison — held-out test window

| Model | PR-AUC | ROC-AUC | P@5% | R@5% | Accuracy | Trivial baseline | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** *(selected)* | **0.889** | 0.985 | 0.674 | 0.850 | 0.988 | 0.960 | 0.0088 | 0.0028 |
| Random forest | 0.879 | 0.988 | 0.671 | 0.847 | 0.987 | 0.960 | 0.0103 | 0.0018 |
| Hybrid (stacked) | 0.865 | 0.983 | 0.669 | 0.843 | 0.989 | 0.960 | 0.0088 | 0.0032 |
| Hybrid (weighted) | 0.846 | 0.978 | 0.657 | 0.829 | 0.990 | 0.960 | 0.0090 | 0.0015 |
| Gradient boosting | 0.835 | 0.987 | 0.682 | 0.861 | 0.985 | 0.960 | 0.0112 | 0.0022 |
| Logistic regression | 0.788 | 0.974 | 0.624 | 0.787 | 0.983 | 0.960 | 0.0137 | 0.0031 |
| Local Outlier Factor | 0.407 | 0.757 | 0.373 | 0.470 | 0.965 | 0.960 | — | — |
| Rules only | 0.354 | 0.843 | 0.354 | 0.446 | 0.938 | 0.960 | 0.0290 | 0.0014 |
| Isolation Forest | 0.211 | 0.823 | 0.246 | 0.310 | 0.933 | 0.960 | — | — |

Selected model PR-AUC with caregiver-clustered bootstrap CI: **0.889 [0.860, 0.917]** (300 resamples). ROC-AUC: 0.985 [0.977, 0.992] (500 resamples).

**Why accuracy is not the headline metric:** at 3.97% prevalence, a model that predicts "not suspicious" for every claim scores **0.960 accuracy while catching nothing**. Every learned model above clusters between 0.983 and 0.990 accuracy despite PR-AUC ranging from 0.788 to 0.889 — selecting on accuracy would pick the fourth-best model by PR-AUC. This project selects on PR-AUC subject to a calibration floor, with Precision@5% as tiebreaker, and reports the trivial baseline in every table so the comparison can't be skipped.

**Is the winner actually better?** LightGBM's PR-AUC advantage is statistically decisive against rules (+0.535), Isolation Forest (+0.679), Local Outlier Factor (+0.482), and logistic regression (+0.101, all *p* < 0.05). It is **not** statistically distinguishable from random forest (Δ +0.010, CI [−0.002, 0.021], DeLong *p* = 0.242). Its real, defensible advantage is efficiency: **5.6 seconds to train versus 57.9 for random forest and 79.3 for gradient boosting**, at equivalent detection quality — the speed argument, not an accuracy argument, is what the model-selection decision rests on.

### Feature ablation — the strongest result in the study

| Feature set | Features | PR-AUC | ROC-AUC | P@5% | R@5% |
|---|---|---|---|---|---|
| Billing only | 44 | 0.437 | 0.788 | 0.354 | 0.446 |
| EVV only | 36 | 0.593 | 0.838 | 0.492 | 0.620 |
| Billing + EVV | 72 | 0.740 | 0.935 | 0.583 | 0.735 |
| + temporal | 87 | 0.836 | 0.976 | 0.638 | 0.805 |
| + statistical | 93 | 0.839 | 0.976 | 0.638 | 0.805 |
| + network (full) | 101 | **0.889** | 0.985 | 0.674 | 0.850 |

Billing and EVV each carry real but partial signal; combined, they gain **+0.147 PR-AUC over the stronger half alone** — direct, quantified evidence that joint modeling of the two data streams the Cures Act's EVV mandate now makes possible is worth doing, which is the study's central empirical claim.

### Scenario generalization, class imbalance, and operational capacity

| Held-out scenario | Recall@5% | | Injected prevalence | PR-AUC | R@5% | | Reviewed | Precision | Recall | FTE / 100k claims |
|---|---|---|---|---|---|---|---|---|---|---|
| Suspicious timestamps | 0.900 | | 1.0% | 0.843 | 0.892 | | 0.5% | 1.000 | 0.125 | 0.44 |
| Coordinated activity | 0.769 | | 2.0% | 0.889 | 0.915 | | 2.5% | 1.000 | 0.641 | 2.24 |
| Near-duplicate claim | 0.417 | | 3.5% | 0.889 | 0.850 | | 4.6% | 0.732 | 0.847 | 4.04 |
| — | — | | 6.0% | 0.918 | 0.737 | | 9.7% | 0.382 | 0.934 | 8.55 |

Ranking quality holds down to 1% prevalence; a fully-recovered scenario (missing EVV, recall@5% = 0.188 when held out — not shown above) is treated as *correct* behavior, since EVV gaps track smartphone access and connectivity more than misconduct, and a model that over-weighted them would systematically flag rural and lower-income caregivers. FTE figures assume 1.5 review hours per alert and 1,700 productive hours per year — stated assumptions, not measurements. Full tables: `reports/tables/`.

---

## Engineering built for scrutiny, not just for a benchmark

This project treats the four most common ways fraud-detection results fail as design requirements, not footnotes:

1. **Label leakage** — a column blocklist removes label/provenance fields and entity IDs; every entity aggregate is strictly backward-looking (`sort → groupby → shift(1) → rolling`); peer statistics, network degrees, and imputation values are fit on the training window only and tested (`test_peer_statistics_are_fitted_on_train_only`). A diagnostic guard flags any feature reaching AUC ≥ 0.999 — none did.
2. **Random splits on time-ordered data** — a strict chronological train/validate/test split, with caregiver-clustered bootstrapping so confidence intervals account for correlated claims from the same caregiver.
3. **Uncalibrated scores presented as probabilities** — isotonic calibration is fit on validation and applied identically to every model, including the rule baseline, so the comparison is fair.
4. **Accuracy reported without its trivial baseline** — every table above states the constant-classifier baseline next to the model's own accuracy.

Every alert the system produces carries structured rule evidence in plain sentences, a calibrated probability with its calibration method disclosed, SHAP local attributions, and an explicit, hard-coded interpretation boundary: *"Feature attributions describe how this model reached its score. They are not evidence of intent, causation, or wrongdoing, and they do not constitute a finding of fraud."* No API endpoint changes a claim's status, denies a service, or alters an authorization — the system is built to route human attention, not to make determinations.

The test suite has caught real defects, not just passed: an independently-drawn EVV check-out offset was inverting 36% of visit intervals on short visits, and an empty EVV table crashed feature construction through a dtype path. Both were found by tests before any result was reported, fixed, and every result regenerated — recorded here because a test suite that has never caught anything is not evidence of quality.

---

## Why this is implementable, not just a paper

Everything above is runnable by a third party in about 20 minutes on a single CPU core, with no proprietary dependency:

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

Or `make all` end to end. `make dashboard` serves the 8-page investigator UI at `localhost:8501`; `make api` serves the FastAPI backend at `localhost:8000/docs`; `docker compose up --build` runs the entire stack in containers; and `notebooks/MedicaidGuard_XAI_Colab.ipynb` reproduces everything on Colab's free tier with no local setup at all. Run provenance (git commit, Python version, platform, every package version) is captured automatically on every run, and CI (GitHub Actions, Python 3.10 and 3.12) lints, tests, runs an end-to-end smoke pipeline, and audits dependencies on every push — the infrastructure of a system meant to be picked up and operated by someone other than its author, which is the actual bar for real-world deployment.

**Path to real-world use** is direct and explicit, not hand-waved: swap the synthetic generator for a state Medicaid agency's or aggregator's real, linked claims-and-EVV extract under a data-use agreement, re-run `make all` against it, and every table in this README regenerates against real data using the same leakage-tested, calibrated, statistically validated pipeline. Nothing about the architecture is synthetic-data-specific; only the input is.

---

## Potential impact

The case for impact follows directly from the scale established above, stated with the same care about what is shown versus what is projected:

- **The gap this project fills is real and currently unaddressed in the published literature.** EVV-and-billing joint modeling is essentially absent from public fraud-detection research because the linked data has never been public — this project is, to the author's knowledge, the first open, reproducible methodology and benchmark for that specific problem, released under MIT so state agencies, vendors, or other researchers can adopt, audit, or extend it rather than each rebuilding leakage-safe evaluation infrastructure from scratch.
- **The lever it optimizes is the one that is actually binding.** Program-integrity review capacity is fixed and small relative to claim volume; the operational-capacity results above show precision and reviewer-load trade-offs across realistic review budgets (e.g., 2.5% review capacity sustaining 1.000 precision at 0.641 recall in this benchmark) — the kind of ranking-quality evidence a state program would need to evaluate before piloting a real deployment.
- **It is designed to protect legitimate caregivers while doing so**, by deliberately down-weighting access-correlated signals (missing EVV) that would otherwise penalize rural, lower-connectivity, or lower-income care workers — an equity property validated directly in Experiment 8 and treated as a design requirement, not an afterthought.
- **The methodology generalizes beyond Medicaid.** The same architecture — transparent rules plus calibrated learners plus appeal-grade explainability, validated under leakage-safe, time-based, statistically rigorous evaluation — applies to any public benefit or insurance program with a similar verification-versus-claim structure: Medicare home health, SNAP, unemployment insurance, and disability benefits all share the same core integrity problem.
- **What is not yet shown, stated plainly:** no real-world detection performance, no measured dollar savings, and no pilot or external validation. All results above are on synthetic data by design, for the structural reasons documented in `reports/dataset_discovery.md` — no compliant public dataset linking claims, EVV telemetry, and verified fraud outcomes exists. Any figure suggesting otherwise would be worse than reporting the gap honestly, which is why none is offered here.

---

## Data

No public dataset links Medicaid personal-care claims to EVV telemetry with verified fraud labels — EVV is operational PHI held by state agencies and their aggregators, fraud determinations are legal outcomes published as case narratives and exclusion lists rather than row-level flags, and public CMS claims files are aggregated by design to prevent re-identification. `reports/dataset_discovery.md` documents the full search (CMS Data, Data.Medicaid.gov, Data.gov, HHS OIG, state portals, Kaggle, UCI, PhysioNet, Harvard Dataverse, Zenodo, Figshare) and why the two closest public candidates were evaluated and rejected as primary sources rather than used because they were convenient.

| Track | Status |
|---|---|
| A — Real public data | Ingestion framework built (licenses, checksums, versioning, data dictionaries). Informs feature design and background framing only; no public row contributes to a reported result in this repository. |
| B — Synthetic Medicaid + EVV | **Primary.** 16 configurable scenarios, reproducible seeds, full scenario metadata. |
| C — Cross-dataset validation | Not performed — no compatible dataset pair exists. Merging incompatible sources to manufacture a result would be worse than reporting its absence. |

---

## Limitations

Read before citing anything above.

1. **Synthetic labels.** Results measure recovery of injected behavioral patterns, not detection of real-world fraud.
2. **Shared authorship of generator and detector.** Scenarios are partly detectable because they were designed in representable form; Experiment 5 (scenario holdout) is a partial defense, not a complete one.
3. **H3 unsupported** — hybrid fusion did not beat the best single model.
4. **The selected model is not statistically distinguishable from random forest** — efficiency, not accuracy, is the deployment argument.
5. **Single simulated year** — multi-year drift untested.
6. **Cross-dataset validation not performed**, for the structural reasons above.
7. **Fairness results describe synthetic cohorts** and do not transfer to any real population (see `docs/privacy.md`).
8. **Impact-simulator outputs are arithmetic on stated assumptions**, labeled as illustrative estimates on every field, never as measured savings.
9. **Ablation uses a restricted learner set** for runtime, recorded in `configs/experiments.yaml`.

Full detail: `reports/limitations.md` and `docs/methodology.md`.

---

## About the research program

The author works in billing and technology administration at a New York home care agency, covering Medicaid billing automation, EVV compliance tracking, multi-payer authorization workflows, and HHAeXchange platform management. That domain grounding shaped specific, visible design decisions: the 16 injected scenarios reflect documentation patterns encountered in practice; the missing-EVV rule carries a deliberately low severity weight because EVV gaps track connectivity and device access more than misconduct; and the rule layer is retained despite scoring roughly a third of the best model's PR-AUC because rule evidence is what a reviewer can restate as a factual assertion about a claim, and what survives an appeal.

M.S., Computer Science, CUNY City College of New York · B.S., Computer Science & Engineering, American International University-Bangladesh · [ORCID 0009-0002-0795-4512](https://orcid.org/0009-0002-0795-4512) · [Google Scholar](https://scholar.google.com/citations?user=MMJRxLEAAAAJ) · peer reviewer and editorial board member, with a publication record spanning explainable AI for financial-fraud and anomaly detection, healthcare program integrity, and applied deep learning.

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

---

## Documentation

| Document | Contents |
|---|---|
| `reports/dataset_discovery.md` | Full public-dataset search, evaluation, and selection rationale |
| `reports/research_report.md` | Full research report built from the actual results |
| `reports/limitations.md` | Consolidated limitations |
| `docs/methodology.md` | Problem formulation, splitting, leakage control, statistics |
| `docs/architecture.md` | Pipeline diagrams and module map |
| `docs/data_dictionary.md` | Every table and field |
| `docs/privacy.md` | Privacy threat model, responsible AI, bias analysis, misuse |
| `docs/reproducibility.md` | Seeds, pinning, nondeterminism, verification |
| `docs/deployment.md` | Local, Docker, and production considerations |
| `SECURITY.md` | Threat model and disclosure |

---

## Citation

```bibtex
@article{hossain2026safeguarding,
  author  = {Hossain, Md Sibbir and Hasan, A T M Fokrule},
  title   = {Safeguarding Public Healthcare Spending with Explainable AI:
             A Statistically Validated Machine Learning Framework for
             Medicaid Fraud Detection and Electronic Visit Verification},
  journal = {Frontiers in Computer Science and Artificial Intelligence},
  volume  = {5},
  number  = {3},
  pages   = {35--44},
  year    = {2026},
  doi     = {10.32996/jcsts.2026.5.3.4}
}

@software{hossain_medicaidguard_xai_2026,
  author  = {Hossain, Md Sibbir},
  title   = {MedicaidGuard-XAI: An Explainable, Statistically Validated Framework
             for Prioritizing Suspicious Medicaid Billing and EVV Activity},
  year    = {2026},
  url     = {https://github.com/sibbirhossain/medicaidguard-xai},
  license = {MIT},
  note    = {Results produced on synthetic data with simulated labels}
}
```

See `CITATION.cff`.

---

## License

MIT — see `LICENSE`.

---

<sub>¹ CMS, *FY2025 Improper Payments Fact Sheet* (Medicaid: 6.12%, $37.39B; CHIP: 7.05%, $1.37B); CMS PERM Medicaid state improper payment rates, 2025; Medicaid FY2024 spending (~$908.8B) and U.S. national health expenditure (~$5.3T, 2024), CMS National Health Expenditure data. · ² 21st Century Cures Act §12006(a); CMS/Medicaid.gov EVV guidance. · ³ U.S. Bureau of Labor Statistics, Occupational Outlook Handbook, *Home Health and Personal Care Aides* (2025 base year: 4,677,100 employed; 2025–2035 projected growth 18%, 847,300 new jobs). All figures cited are independently verifiable at their primary sources and are external, publicly reported statistics — they are not outputs of this project's model or impact simulator.</sub>

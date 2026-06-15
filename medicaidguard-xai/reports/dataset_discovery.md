# Dataset Discovery Report

**Project:** MedicaidGuard-XAI
**Purpose:** Document the search for public healthcare datasets suitable for
Medicaid billing and EVV anomaly detection, and justify the data strategy that
follows from what was actually found.

This report exists because the build specification requires dataset discovery
*before* model development. The conclusion below is not the convenient one, and
it constrains every claim the rest of the project is allowed to make.

---

## 1. Headline finding

**No public dataset exists that contains linked Medicaid personal-care claims,
Electronic Visit Verification telemetry, and verified fraud labels.**

This is not a gap in the search. It follows from how the data is produced and
governed:

- **EVV data is operational PHI.** Under the 21st Century Cures Act, states must
  collect EVV for Medicaid personal care and home health services. The records
  sit with state Medicaid agencies and their aggregator vendors (HHAeXchange,
  Sandata, Tellus and others). They contain member identity, home location and
  minute-level presence data for people receiving care in their homes. There is
  no public-interest argument for publishing them and no mechanism that does.
- **Fraud labels are legal outcomes, not data fields.** A determination of fraud
  comes from a Medicaid Fraud Control Unit investigation, an administrative
  action, or a court. These outcomes are published as *case narratives and
  exclusion lists*, not as row-level flags attached to claims.
- **Public claims data is aggregated by design.** CMS public-use files are
  released at provider-year or state-quarter granularity precisely so that
  individual members cannot be re-identified. Aggregation destroys the
  within-visit structure that EVV anomaly detection depends on.

The honest consequence: **the EVV research track must run on synthetic data**,
and every result in this repository is a result about simulated labels. That
limitation is stated in the README, the research report, the dashboard and the
API responses.

---

## 2. Sources searched

| Category | Sources examined |
|---|---|
| U.S. federal healthcare | CMS Data portal, Data.Medicaid.gov, Medicaid.gov, Data.gov, CMS public-use files, CMS Open Payments, HHS OIG |
| Program integrity | HHS OIG LEIE exclusions database, OIG enforcement actions, CMS PERM improper-payment program, state MFCU reporting |
| State open data | State Medicaid open-data portals for the larger EVV-implementing states |
| Research repositories | Kaggle, UCI Machine Learning Repository, PhysioNet, Harvard Dataverse, Zenodo, Figshare |
| Code and supplements | Public GitHub repositories with explicit licences, peer-reviewed paper supplements |

Search terms: Medicaid claims, healthcare claims, healthcare fraud detection,
insurance fraud, medical billing anomalies, healthcare utilization, provider
billing, electronic visit verification, EVV, home healthcare, personal care
services, healthcare anomaly detection.

---

## 3. Candidate evaluation

| Dataset | Source | Data type | Granularity | Labels | EVV fields | Licence | Relevance | Selected |
|---|---|---|---|---|---|---|---|---|
| Medicaid State Drug Utilization | Data.Medicaid.gov | Utilisation | State × quarter × drug | None | None | US Gov public domain | Low — no claims, no providers | No |
| Medicare Physician & Other Practitioners PUF | CMS Data | Billing | Provider (NPI) × HCPCS × year | None | None | US Gov public domain | Medium — best public analogue for provider peer-deviation features | Context only |
| Medicare Part D Prescriber PUF | CMS Data | Billing | Prescriber × drug × year | None | None | US Gov public domain | Low — pharmacy domain, wrong service type | No |
| HHS OIG LEIE | HHS OIG | Exclusion list | Individual / entity | Adverse action | None | US Gov public domain | Medium — a real label, but see §4 | Context only |
| CMS Open Payments | CMS | Industry payments | Physician × payment | None | None | US Gov public domain | Low — conflict of interest, not billing integrity | No |
| CMS PERM improper-payment reports | CMS | Aggregate statistics | National / state | Rate estimates | None | US Gov public domain | Context — background only, see §5 | Context only |
| Kaggle healthcare-provider fraud datasets | Kaggle | Claims | Claim-level | Provider-level flag | None | Varies; frequently unclear | See §4.2 | No |
| Insurance fraud tabular datasets (UCI / Kaggle) | Various | Auto/general insurance | Claim-level | Binary | None | Varies | Low — different domain and generating process | No |
| Synthea synthetic EHR | MITRE / open source | Synthetic EHR | Patient-level | None | None | Apache 2.0 | Low — clinical encounters, no billing-integrity structure or EVV | No |

---

## 4. Why the two closest candidates were not used as the primary dataset

### 4.1 HHS OIG LEIE

The LEIE is a genuine, verified, government-maintained list of individuals and
entities excluded from federal healthcare programs. It is tempting as a label
source and should be resisted for three reasons:

1. **Exclusion is not a claim-level fraud finding.** Many exclusions are
   mandatory consequences of a conviction, a licence revocation, or a defaulted
   health education loan. Treating an excluded NPI as "fraud" mislabels the
   construct.
2. **It carries no billing rows.** Joining it to a public billing file gives a
   provider-year label attached to aggregated counts, not a per-claim target.
3. **Temporal contamination.** The exclusion date typically postdates the
   conduct by years. Any model trained on post-exclusion billing learns the
   consequences of the exclusion, not its antecedents.

A defensible use exists — as a weak, explicitly-defined provider-level adverse
outcome for a separate research question — but it does not support the research
question this project asks.

### 4.2 Kaggle "healthcare provider fraud" datasets

Several widely-used Kaggle datasets carry claim-level rows with a
`PotentialFraud` provider flag. They are popular in the literature. They were
not selected because:

- The provenance of the fraud flag is not documented to a verifiable source, so
  the label's construct validity cannot be established.
- The licence terms are frequently unstated or non-redistributable.
- They contain no EVV fields whatsoever, so they cannot address the part of the
  research question that is actually novel.

Using one of these and calling the result "Medicaid fraud detection" would be
the single easiest way to make this project look stronger and be weaker. The
specification forbids it and so does good practice.

---

## 5. Use of public data for context

Public CMS material is used in this project for **background framing only** —
never as a source of results:

- CMS PERM publishes measured improper-payment rates for Medicaid. These are
  *payment-level* estimates driven largely by documentation and eligibility
  errors, not fraud determinations. They are cited in the README to motivate the
  problem and are explicitly **not** used to calibrate the generator's
  suspicious-activity prevalence, which is a controlled experimental parameter.
- The Medicare Physician PUF informed the *design* of the provider peer-deviation
  features (which billing aggregates are meaningful at provider level), without
  contributing any training row.

Any specific figure quoted from these sources in the paper must be checked
against the current CMS publication at the time of writing, with the release
year stated. Government statistics are not invented or approximated here.

---

## 6. Data strategy that follows

| Track | Status | Content |
|---|---|---|
| **A — Real public data** | *Framework built, not the basis of results* | `src/medicaidguard/data/ingestion.py` ingests any dataset the user is licensed to download, with checksums, licence records, versioning and auto-generated data dictionaries. `configs/datasets.yaml` is the registry. Nothing downloads automatically, because several portals require accepting terms a script cannot accept on a user's behalf. |
| **B — Synthetic Medicaid + EVV** | **Primary, and the basis of all reported results** | `src/medicaidguard/data/generator.py`. 16 configurable suspicious scenarios, reproducible seeds, scenario metadata held in a separate table to prevent leakage. |
| **C — Cross-dataset validation** | **Not performed** | Requires two datasets with compatible schemas and comparable labels. No such pair exists for this problem. Merging incompatible sources to manufacture a cross-validation result would be worse than reporting its absence. |

---

## 7. What this means for the claims this project may make

Permitted:

- That the method recovers injected behavioural patterns under time-based
  validation, with stated confidence intervals.
- That combining billing and EVV feature families measurably outperforms either
  alone *on this data*.
- That the engineering is reproducible, calibrated and leakage-checked.

Not permitted:

- Any statement about real-world Medicaid fraud detection performance.
- Any dollar figure presented as savings.
- Any implication that a real provider, caregiver or beneficiary has been
  identified.
- Any claim of external adoption or validation that has not occurred.

---

## 8. Reviewer note on the synthetic track

The generator and the detector were written by the same author. Injected
scenarios are therefore detectable partly because they were designed in a
representable form. Two partial defences are implemented:

1. **Scenario generalisation (Experiment 5).** Selected scenarios are relabelled
   as negative during training, simulating a scheme that has never been
   investigated and so carries no label. Performance on those held-out scenarios
   is reported separately and is substantially worse — which is the expected and
   honest result.
2. **Realistic normal behaviour.** Baseline claims carry irregular hours, GPS
   drift, late submissions, benign missing EVV and genuine duration variance. If
   "normal" were too clean, every model would score near 1.0 and the benchmark
   would be meaningless.

Neither defence makes synthetic results transferable to real data. The
limitation is structural and belongs in the abstract of any paper using this
work, not in a footnote.

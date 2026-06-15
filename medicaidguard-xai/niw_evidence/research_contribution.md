# Research Contribution Statement

> Technical supporting material, not legal advice. This document describes a
> research contribution. It does not assert eligibility for any immigration
> classification.

## 1. The gap addressed

Published healthcare fraud detection research concentrates on Medicare claims,
where public provider-level files exist. Electronic Visit Verification — the
check-in/check-out telemetry that federal law requires states to collect for
Medicaid personal care and home health services — is essentially absent from the
modelling literature, because the data is operational PHI and is not published.

This project documents that absence rigorously (`reports/dataset_discovery.md`,
a search across CMS Data, Data.Medicaid.gov, Data.gov, HHS OIG, state portals,
Kaggle, UCI, PhysioNet, Harvard Dataverse, Zenodo and Figshare), then builds a
methodology and a reproducible open-source implementation for the problem the
absent data would otherwise support.

## 2. Specific contributions

**2.1 A quantified answer to whether EVV adds signal over billing alone.**
Feature ablation on matched data gives billing-only PR-AUC 0.437, EVV-only
0.593, and combined 0.740 — a gain of 0.147 over the better half alone. This is
the substantive empirical claim, and it is the kind of question that cannot be
asked at all without linked data.

**2.2 An evaluation harness that resists the field's common defects.**
Three-layer leakage control (column blocklist, strictly backward-looking
aggregates, train-window-only statistic fitting) with automated verification;
time-based rather than random splitting; accuracy always reported beside its
trivial baseline; calibration applied to the rule baseline on the same footing
as learned models so the comparison is fair.

**2.3 An open, reproducible synthetic benchmark for EVV anomaly detection.**
Sixteen behaviourally-injected scenarios with configurable prevalence and
severity, seeded reproducibility, and scenario metadata held outside the feature
matrix. Released under MIT so others can extend or dispute it.

**2.4 Explainability designed as reviewable evidence rather than retrofitted.**
Structured rule evidence in plain sentences with severities, alongside SHAP
attributions carrying an explicit interpretation boundary. The rule layer is
retained despite scoring a third of the best model's PR-AUC, on the argument
that a score which cannot be restated as a factual assertion about the claim is
not usable where alerts affect livelihoods.

**2.5 Negative results reported.** Hybrid fusion did not beat the best single
model (0.865 vs 0.889). The selected model is not statistically separable from a
random forest (ΔPR-AUC +0.010, CI [−0.002, 0.021]). The robust-statistical
feature block adds +0.003 and should be dropped. These are recorded in the
README, the research report and the limitations document rather than tuned away.

## 3. Relationship to the author's prior published work

This implementation is the engineering and experimental counterpart to the
author's peer-reviewed research programme on AI-driven fraud and anomaly
detection, which spans financial fraud detection using graph neural networks,
deep learning applications, and healthcare program integrity.

**Anyone assembling an evidence package must verify independently:** publication
venue, DOI, acceptance status, citation counts, h-index, editorial and peer
review service, and any adoption or external interest. Those are matters of
record that this repository cannot and does not attest to. This document asserts
only what the code and the committed results demonstrate.

## 4. What this contribution does not establish

- It does not demonstrate real-world Medicaid fraud detection performance. All
  results use simulated labels.
- It does not establish state-of-the-art performance; no shared benchmark exists
  for this problem.
- It does not demonstrate operational adoption. No pilot has been conducted.
- It does not, alone, establish national importance or substantial merit. Those
  arguments require external evidence — citations, adoption, collaborations,
  independent validation — that exists outside this repository or does not yet
  exist.

## 5. Reproducibility as the basis of the claim

Every number in §2 regenerates from `make all` at seed 20260913, on a laptop or
a free Colab instance, in under twenty minutes. The evidence is the artefact,
not a description of it.

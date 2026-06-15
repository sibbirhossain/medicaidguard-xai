# Limitations

Ordered by how much they constrain what may be claimed.

## 1. Synthetic labels (binding)

No result here measures real-world Medicaid fraud detection. Every label is an
injected scenario in generated data. The correct statement of what was measured
is: *the method recovers injected behavioural patterns under time-based
validation with stated confidence intervals.*

What this forbids: any performance claim about real Medicaid data, any
comparison against a published real-data benchmark, any implication that a real
provider, caregiver or beneficiary has been identified.

## 2. Generator and detector share an author (binding)

Scenarios are detectable partly because they were designed in a form the feature
set can represent. This inflates every metric by an unknown amount.

Partial defence: Experiment 5 relabels three scenarios as negative during
training and measures recovery separately. Held-out recall@5% (0.417–0.900) is
materially worse than trained-scenario recall (1.000 for eight scenarios), which
is evidence the inflation is real. It does not quantify it.

The only real fix is a generator authored independently of the detector, or
evaluation on real data.

## 3. Hypothesis 3 was not supported

Hybrid fusion did not beat the best individual model (0.865 vs 0.889). The
framework's motivating claim is therefore unsupported on ranking performance.
Its remaining justification — rule evidence usable in an appeal — is a design
argument, not an empirical one.

## 4. The selected model is not statistically separable from random forest

ΔPR-AUC +0.010, CI [−0.002, 0.021]; DeLong *p* = 0.242. Any presentation of
LightGBM as "the most accurate model" overstates the evidence. The supportable
claim is equivalent detection quality at roughly one tenth the training time.

## 5. Single simulated year

One 12-month period. Temporal robustness is tested within it (train on months
1–8, test on 11–12) but multi-year drift, policy changes, rate changes and
seasonal programme effects are untested.

## 6. No cross-dataset validation

No pair of datasets with compatible schemas and comparable labels exists for
this problem. Merging incompatible sources to produce a cross-validation number
would be worse than its absence. See `reports/dataset_discovery.md`.

## 7. Fairness results describe synthetic cohorts

Cohort labels are randomly assigned by the generator. The reported PR-AUC range
(0.865–0.932) and FPR range (0.031–0.047) across groups say nothing about any
real population. The experiment validates the measurement code. Real bias risks
are analysed in `docs/privacy.md` and are not measured here.

## 8. Impact simulator outputs are not measurements

Precision and recall come from the held-out synthetic test window; everything
downstream — improper-payment rate, amount per case, realisation rate, review
cost — is a user-supplied assumption. Outputs are labelled accordingly on every
field.

## 9. Methodological details worth disclosing

- **Claim-level bootstrap understates dependence** between claims from the same
  caregiver. The caregiver-clustered bootstrap is reported for the selected
  model and is the wider, more defensible interval.
- **The paired-bootstrap *p*-like quantity** is the doubled proportion of
  resamples on the opposite side of zero. It is descriptive, not an exact test.
- **Ablation uses a restricted learner set** (LightGBM + logistic regression) for
  runtime. The question asked is which feature families matter, not which
  learner wins, but the restriction is recorded in `configs/experiments.yaml`.
- **Threshold selection uses the validation window**, never test. Reported
  threshold-dependent metrics inherit validation-set variance.
- **LOF training is subsampled** to 12,000 rows for tractability in high
  dimension.
- **The robust-statistical feature block adds +0.003 PR-AUC** once temporal
  features are present. It is retained for ablation reproducibility, not because
  it earns its place.

## 10. Engineering limitations

- No authentication on the API; it must not face an untrusted network with real
  data loaded.
- Investigator notes and review status in the dashboard are session-local and
  not persisted.
- Multi-threaded training is disabled for exact reproducibility, at a cost in
  wall-clock time.

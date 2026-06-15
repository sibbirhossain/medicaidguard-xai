# Methodology

## 1. Problem formulation

Let a claim be $c_i$ with features $x_i \in \mathbb{R}^d$ derived from billing
records, EVV telemetry, temporal context, robust peer statistics and the
provider–caregiver–patient interaction graph. Let $y_i \in \{0, 1\}$ indicate
whether the claim carries an injected suspicious scenario.

The task is **ranking under a capacity constraint**, not classification. Given a
review budget of $K$ claims, the objective is

$$\max_{s} \; \frac{1}{K}\sum_{i \in \text{top-}K(s)} y_i$$

where $s: \mathbb{R}^d \to \mathbb{R}$ is the scoring function. This framing is
the reason PR-AUC and Precision@K are the reported headline metrics: an
investigator unit can review a fixed volume per period, and what matters is how
many genuine cases sit in that volume.

## 2. Why not accuracy

At prevalence $\pi \approx 0.04$, the constant predictor $s(x) = 0$ achieves
accuracy $1 - \pi \approx 0.96$ while achieving recall $0$. Accuracy is
therefore not a discriminating statistic in this regime. Empirically, across the
nine models compared here, accuracy spans roughly one percentage point while
PR-AUC spans a factor of five.

Every results table in this project reports `accuracy` beside
`accuracy_trivial_baseline` $= \max(\pi, 1-\pi)$ so the comparison is forced
rather than optional. Selection uses PR-AUC subject to a calibration floor
(`SelectionConfig.max_brier`), with Precision@5% as the tiebreaker.

## 3. Splitting

**Time-based.** Train through `split.train_end`, validate through
`split.valid_end`, test thereafter. Program-integrity models are deployed
forward in time; a random split reports a number the deployed system cannot
achieve. `assert_temporal_order` enforces that no training claim postdates the
first test claim.

**Group-aware folds** are available (`splits.group_aware_folds`) for
cross-validated variants, keeping each entity wholly on one side so the model
cannot memorise a caregiver instead of a behaviour.

## 4. Leakage control

Three distinct mechanisms, because leakage arrives by three distinct routes:

1. **Column-level.** `schema.LEAKAGE_COLUMNS` lists label and provenance fields.
   `assert_no_leakage_columns` rejects any feature matrix containing them. Entity
   identifiers are excluded separately via `GROUP_COLUMNS`.
2. **Temporal.** All entity aggregates are strictly backward-looking: sort by
   entity and date, `shift(1)`, then roll or expand. A `groupby().transform()`
   would let a caregiver's later claims inform their earlier ones.
3. **Fit-window.** Anything that learns a statistic — peer medians and MADs,
   network degrees and concentrations, categorical vocabularies, imputation
   medians — is fitted on the training window only and applied as a lookup
   thereafter. `test_peer_statistics_are_fitted_on_train_only` verifies this by
   refitting on a truncated dataset and requiring identical statistics.

A fourth check is diagnostic rather than structural:
`assert_no_perfect_predictor` flags any single feature reaching AUC ≥ 0.999 on
the training window. A feature that alone separates the classes is almost always
a leak.

## 5. Feature families

| Family | Count (approx.) | Representative features |
|---|---|---|
| Billing | ~44 | billed-to-authorised ratio, duplicate group sizes, caregiver daily hours, causal rolling means at 7/30 days, deviation from the entity's own history |
| EVV | ~36 | EVV presence, billed-minus-verified duration, clock-exactness, check-in distance from home, implied travel speed between consecutive visits, overlap minutes |
| Temporal | ~15 | hour-of-day (cyclically encoded), weekend, overnight, duration roundness, submission lag |
| Statistical | ~6 | robust peer z-scores and ratios by service code, using median/MAD fitted on training |
| Network | ~8 | caregiver and patient degree, caregiver share of agency volume, pair persistence, relationship novelty |

## 6. Detection engine

- **Model A — rules.** Ten transparent rules, each with a severity ramp in
  $[0,1]$ and a policy weight. Combined by noisy-OR:
  $S = 100\left(1 - \prod_r (1 - w_r \sigma_r)\right)$, so one severe rule
  produces a high score and ten weak rules do not. Thresholds are policy
  parameters documented in `configs/model_config.yaml`.
- **Model B — supervised.** Logistic regression, random forest, gradient
  boosting, LightGBM. Class imbalance handled by `class_weight="balanced"`
  rather than resampling: reweighting changes the loss without duplicating rows,
  so it cannot place near-copies of a positive claim on both sides of a fold
  boundary.
- **Model C — unsupervised.** Isolation Forest and Local Outlier Factor, fitted
  on training rows without labels. They answer a different question — "is this
  unlike anything else" rather than "does this resemble known suspicious
  activity" — which is the only question available for a scheme that has never
  been investigated.
- **Model D — hybrid fusion.** Two strategies compared rather than assumed: a
  fixed convex combination of normalised components (transparent, auditable) and
  a logistic meta-learner fitted on validation (usually stronger, less
  transparent). Components are the best supervised probability, the rule score,
  the anomaly score and a compact EVV-inconsistency score.

## 7. Calibration

Isotonic regression fitted on the **validation** window, never on training
(already memorised) and never on test (leakage). This matters operationally
rather than cosmetically: the priority bands and the impact simulator both
interpret the score as an expected share of genuine cases, and an uncalibrated
score makes those numbers meaningless.

The rule score and the stacked fusion score are calibrated on the same footing
as the learned models. Without this, the rule baseline would be disqualified by
the Brier floor for a reason unrelated to its ranking quality — an unfair
comparison in the baseline's disfavour.

Unsupervised scores are min-max normalised ranks. They are reported with
calibration metrics as `NaN` rather than presented as probabilities.

## 8. Statistical validation

- **Bootstrap CIs** (percentile, 1000 resamples) at claim level.
- **Cluster bootstrap** over caregivers, which is the more defensible interval
  because claims from one caregiver are not independent. It is wider, and the
  wider one is reported for the selected model.
- **Paired bootstrap** on PR-AUC differences: both models are scored on the same
  resampled rows so shared sampling noise cancels. The accompanying $p$-like
  quantity is the doubled proportion of resamples on the opposite side of zero —
  a descriptive statistic, not an exact test.
- **DeLong** for correlated ROC-AUCs.
- **McNemar** (exact binomial) on discordant pairs at the 5% review threshold.

Reported for every comparison: point estimate, interval, sample size, seed,
methodology, and whether the difference is practically meaningful at realistic
review capacity. A statistically detectable PR-AUC gap of 0.02 that does not
change how many cases a unit finds in its top 5% is reported as such.

## 9. Known methodological limitations

1. **Synthetic labels.** Results measure recovery of injected patterns, not
   detection of real fraud.
2. **Shared authorship of generator and detector.** Scenarios are detectable
   partly because they were designed in a representable form. Experiment 5
   (held-out scenarios) is a partial, not complete, defence.
3. **Claim-level bootstrap understates dependence** where used; the cluster
   bootstrap is provided for this reason.
4. **Single time period.** One simulated year. Multi-year drift is untested.
5. **Ablation uses a restricted learner set** for runtime. The question it asks
   is which feature families matter, not which learner wins, but the restriction
   is recorded.

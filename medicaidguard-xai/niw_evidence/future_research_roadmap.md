# Future Research Roadmap

> Technical supporting material, not legal advice. These are research plans, not
> completed work, and are labelled as such.

## Near term — closing the credibility gaps in the current results

**R1. Independently authored generator.** The binding limitation of the present
work is that generator and detector share an author, which inflates every metric
by an unknown amount. A generator specified by one party against behavioural
descriptions and implemented by another, or a scenario set contributed by
practitioners without sight of the feature code, would quantify that inflation.

**R2. Graph-native modelling for coordinated activity.** Coordinated rings
generalised worst among held-out scenarios (recall@5% 0.769) because ring
membership is a property of the relationship structure, not of any individual
claim. Current network features are aggregate degrees and concentrations.
Community detection or a graph neural network operating on the
provider–caregiver–patient graph is the natural next step, and connects directly
to the author's prior work on graph neural networks for financial fraud.

**R3. Drop the statistical block, test what replaces it.** Robust peer
deviations add +0.003 PR-AUC once temporal features are present. Removing them
and testing change-point detection on provider billing trajectories in their
place is a cheap, falsifiable experiment.

**R4. Multi-year drift.** The present evaluation covers one simulated year.
Extending the generator to multiple years with rate changes, policy shifts and
seasonal effects would test whether calibration survives, and how often
recalibration is needed.

## Medium term — contact with reality

**R5. Evaluation on real EVV data under a data-use agreement.** The decisive
step. Requires a state Medicaid agency or aggregator partnership, IRB review
where applicable, and a protocol that keeps member-level data inside the
partner's environment. Even a retrospective study on a single state's
personal-care segment would convert every claim in this work from "recovers
injected patterns" to a real finding.

**R6. Control-stream design.** Any real deployment needs a randomly sampled
review stream outside the model's ranking. Without it, reviewers investigate
where the model points, labels record what they found, and the next model learns
reviewer behaviour. Designing and costing that control stream is a research
contribution in itself and is largely absent from the applied literature.

**R7. Real fairness evaluation.** The current fairness pipeline is validated but
points at randomly assigned synthetic cohorts. Applied to real data it would
test the specific hypotheses stated in `docs/privacy.md`: that documentation
capacity, technology access and connectivity drive apparent risk independent of
misconduct.

**R8. Reviewer study on explanation quality.** Whether rule evidence actually
improves reviewer decisions, or merely improves their confidence, is an
empirical question that has not been asked in this domain. A controlled study
with experienced reviewers would be a distinctive contribution.

## Longer term

**R9. Cross-state generalisation.** Different states use different EVV
aggregators with different data models. Whether a model trained in one state
transfers is unknown and matters for whether this is a per-state or a shared
capability.

**R10. Prospective evaluation.** Deploy in shadow mode alongside existing
processes, measure whether the ranking changes what investigators find, and
publish the result including if it does not.

**R11. Appeal-aware modelling.** Constraining the model so every alert above a
priority threshold is accompanied by at least one rule-based factual assertion
about the claim. This trades ranking performance for defensibility and the
trade-off has not, to the author's knowledge, been quantified.

## Dependencies and honest sequencing

R1, R2, R3 and R4 are achievable with the current repository and no external
dependency. R5 gates R6, R7, R9 and R10 — none of them can proceed without a
data partnership, and that partnership is the single highest-value thing to
pursue. R8 requires access to practising reviewers but not to their data.

Nothing in this roadmap has been completed. Anyone citing it should describe it
as planned work.

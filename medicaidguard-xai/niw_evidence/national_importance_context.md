# U.S. Healthcare Technology Relevance — Context Notes

> **Technical supporting material, not legal advice.** This document provides
> context for why the problem domain matters. It does not assert that this
> project establishes national importance, and it deliberately contains **no
> government statistics**, because inventing or approximating one would be worse
> than omitting it.

## How to use this document

Every factual claim about federal programs, spending, improper-payment rates or
policy below is stated **as a category of fact to be verified**, not as a
figure. If you are assembling an evidence package, you must:

1. Retrieve the current figure from the primary government source.
2. Cite the source, the publication year, and the exact measure.
3. State the measure correctly. CMS PERM reports an **improper payment rate**,
   which is dominated by documentation and eligibility errors and is **not** a
   fraud rate. Presenting one as the other is the most common error in this
   space and it is the kind of error that damages credibility on review.

## Verifiable context categories

| Claim category | Primary source to verify against |
|---|---|
| Total federal and state Medicaid expenditure | CMS National Health Expenditure data; CMS Financial Management Reports |
| Medicaid improper payment rate and methodology | CMS PERM program reports |
| Federal EVV mandate scope and deadlines | 21st Century Cures Act, Section 12006; CMS EVV guidance to states |
| Number of beneficiaries receiving personal care / home health services | CMS Medicaid enrollment and utilisation reports |
| Medicaid Fraud Control Unit activity and outcomes | HHS OIG annual MFCU statistical reports |
| Exclusion actions | HHS OIG LEIE |
| Home care workforce size and growth projections | Bureau of Labor Statistics Occupational Outlook |

## The substantive argument, stated without numbers

**The problem is structural, not incidental.** Personal care is delivered in
private residences with no institutional witness. The only evidence a service
occurred is a verification record and a claim. This is a materially harder
integrity environment than facility-based care, and it is growing as
home-and-community-based services expand relative to institutional care.

**Federal policy has created the data but not the analytics.** The Cures Act
requires states to collect EVV. States and their aggregator vendors now hold
large volumes of check-in/check-out telemetry. Published methodology for using
that telemetry jointly with billing data is thin, because the data itself is not
public — which is precisely the gap this project addresses with an open,
reproducible methodology and synthetic benchmark.

**Review capacity is the binding constraint.** Investigative units are small
relative to claim volume. Improving the *ordering* of a review queue is
therefore a more tractable lever than improving classification accuracy, and it
is what this project optimises.

**Getting it wrong has asymmetric cost.** A false positive against a home care
caregiver can suspend their income during review; against an agency it can
trigger a payment hold reaching the beneficiaries in its care. This is why the
project emphasises calibration, explainability, low weight on access-correlated
signals like missing EVV, and human-in-the-loop review — and why those choices
are defensible as contributions rather than as caution.

## What this project contributes to that context

- An open-source methodology for a problem where the data is not public.
- A reproducible synthetic benchmark others can extend or dispute.
- An evaluation harness that resists leakage, temporal contamination and
  accuracy-under-imbalance errors.
- Explainability designed for appeal defensibility.
- Honest reporting, including negative results.

## What it does not contribute

- No real-world detection performance.
- No measured savings. The impact simulator converts your assumptions into
  arithmetic and labels every output accordingly.
- No adoption, pilot, or external validation. None has occurred.
- No claim that open-sourcing a methodology constitutes national benefit on its
  own. That argument, if made, needs external evidence.

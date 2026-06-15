# Security policy

## Reporting a vulnerability

Open a private security advisory through the repository's Security tab. Please
do not open a public issue for a vulnerability.

## Scope and threat model

This repository holds a research prototype that operates on **synthetic** data.
It is not hardened for production use against real Protected Health Information.

### What the project deliberately does not do

* It does not read, store or transmit real PHI. The synthetic generator
  produces fictional identifiers only.
* It has no authentication layer, because it has no real data to protect. Do
  not expose the API or dashboard to an untrusted network with real data
  loaded; put an authenticating reverse proxy in front of it first.
* It does not make eligibility, payment or denial decisions, and offers no
  endpoint that changes a claim's status.

### Threats considered

| Threat | Mitigation in this repository |
|---|---|
| Real PHI committed to Git | `.gitignore` excludes all of `data/`; CI fails if a `.parquet` or `.csv` outside `reports/` is tracked |
| Secrets in source | No credentials in code; `.env.example` documents variables, `.env` is ignored |
| Location data exposure | EVV coordinates are synthetic; the API returns distance-derived features rather than raw coordinates in scoring responses |
| Model output misread as a finding | Every scoring response and every dashboard page carries the review-recommendation disclaimer |
| Dependency vulnerabilities | `pip-audit` runs in CI |
| Container running as root | Dockerfile drops to an unprivileged UID |

### If you adapt this for real data

You are taking on obligations this repository does not discharge: a HIPAA
business-associate agreement where applicable, minimum-necessary access
controls, encryption at rest and in transit, audit logging of every access to
member-level records, a documented appeal path for anyone affected by an alert,
and your state Medicaid agency's own EVV data-use terms. Treat the code here as
a methodology reference, not a compliant system.

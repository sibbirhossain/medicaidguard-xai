# Architecture

## Pipeline

```mermaid
flowchart TD
    A[Dataset discovery<br/>reports/dataset_discovery.md] --> B[Public data ingestion<br/>data/ingestion.py]
    A --> C[Synthetic generator<br/>data/generator.py]
    B --> D[Validation<br/>data/validation.py]
    C --> D
    D --> E[Time-based split<br/>evaluation/splits.py]
    E --> F[Feature pipeline<br/>features/pipeline.py]
    F --> G{Leakage guards<br/>evaluation/leakage.py}
    G -->|pass| H[Rule screening<br/>models/rules.py]
    G -->|pass| I[Supervised models<br/>models/supervised.py]
    G -->|pass| J[Anomaly detection<br/>models/anomaly.py]
    H --> K[Hybrid fusion<br/>models/fusion.py]
    I --> K
    J --> K
    K --> L[Calibration<br/>models/calibration.py]
    L --> M[Explainability<br/>explainability/]
    L --> N[Statistical validation<br/>evaluation/]
    M --> O[Alert assembly<br/>explainability/evidence.py]
    N --> P[Reports and tables<br/>reports/]
    O --> Q[FastAPI<br/>api/main.py]
    O --> R[Streamlit dashboard<br/>dashboard/app.py]
    Q --> S[Docker / CI]
    R --> S
```

## Feature construction

```mermaid
flowchart LR
    C[claims] --> B[billing.py<br/>row-level + causal rolling]
    C --> T[temporal.py<br/>time of day, roundness, lag]
    E[evv_events] --> V[evv.py<br/>duration, location, travel, overlap]
    C --> V
    P[patients] --> V
    C --> N[network.py<br/>degrees, concentration]
    B --> S[temporal.py<br/>PeerDeviationTransformer]
    B --> PIPE[pipeline.py]
    V --> PIPE
    T --> PIPE
    S --> PIPE
    N --> PIPE
    PIPE --> X[feature matrix]

    subgraph fitted on TRAIN WINDOW only
      S
      N
    end
```

## Module map

| Path | Responsibility |
|---|---|
| `config.py` | Paths, seeds, generator/split/selection configuration, YAML loading |
| `data/schema.py` | Table contracts, keys, and the leakage-column blocklist |
| `data/generator.py` | Synthetic Medicaid + EVV generation, 16 scenarios |
| `data/validation.py` | Schema, referential integrity, plausibility, missingness |
| `data/ingestion.py` | Public-dataset ingestion with licences and checksums |
| `data/metadata.py` | Run provenance: git commit, package versions, platform |
| `features/*` | Five feature families plus the fit/transform pipeline |
| `models/rules.py` | Transparent rule engine and structured evidence |
| `models/supervised.py` | Candidate learner zoo |
| `models/anomaly.py` | Isolation Forest, LOF |
| `models/fusion.py` | Two fusion strategies, priority bands, selection rule |
| `models/calibration.py` | Isotonic / Platt calibration |
| `evaluation/metrics.py` | PR-AUC, Precision@K, calibration error, workload |
| `evaluation/confidence_intervals.py` | Bootstrap, cluster bootstrap, DeLong, McNemar |
| `evaluation/leakage.py` | Runtime leakage assertions |
| `evaluation/experiments.py` | `run_pipeline` plus the experiment suite |
| `explainability/*` | SHAP wrapper, alert assembly and narratives |
| `impact/simulator.py` | Scenario-based workload and illustrative economics |
| `api/*` | FastAPI service |
| `database/*` | SQLAlchemy schema, session, repository |

## Design decisions worth defending

**One `run_pipeline` function, used everywhere.** The scripts, the notebooks and
the tests all call it. This is why the numbers in the report, the dashboard and
the API cannot drift apart — there is no second code path that could disagree.

**Rules retained despite weak standalone performance.** Rule PR-AUC is roughly a
third of the best model's. They stay because rule evidence is what a reviewer
can restate as a factual assertion about the claim, and what survives an appeal.

**Fusion strategies compared, not assumed.** Averaging component scores with
arbitrary weights is the common shortcut. Both a transparent weighted scheme and
a learned stacked scheme are implemented and evaluated, and the selected one is
recorded.

**Unsupervised track kept despite low labelled scores.** It is the only
component that can respond to a scheme with no training labels, which is the
realistic case for novel fraud.

# Deployment

## Local

```bash
pip install -r requirements.txt && pip install -e .
make data && make train
make api          # http://localhost:8000/docs
make dashboard    # http://localhost:8501
```

## Docker

```bash
docker compose up --build
```

Three services: `bootstrap` generates data and trains, then `api` (8000) and
`dashboard` (8501) start once it completes successfully. Volumes mount `data/`,
`artifacts/` and `reports/` so results survive container restarts.

The container runs as UID 10001, not root, and declares a health check against
`/health`.

## Google Colab

`notebooks/MedicaidGuard_XAI_Colab.ipynb` runs everything on the free CPU tier
in 10–20 minutes. No GPU is needed; these are tabular models and a GPU would not
help. The notebook includes a GitHub push cell that takes the token through
`getpass` and scrubs the credential-bearing remote afterwards.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MEDICAIDGUARD_ROOT` | repository root | Base path for data, artifacts, reports |
| `MEDICAIDGUARD_DB_URL` | `sqlite:///./data/medicaidguard.db` | Database; PostgreSQL URLs supported |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Copy `.env.example` to `.env`. Never commit `.env`.

## PostgreSQL

The SQLAlchemy metadata in `database/models.py` avoids SQLite-only constructs,
so the same schema creates on PostgreSQL:

```bash
export MEDICAIDGUARD_DB_URL="postgresql+psycopg://user:pass@host:5432/medicaidguard"
python -c "from medicaidguard.database.session import init_db; init_db()"
```

## Before putting this anywhere real

This is a research prototype. A deployment against real data would need, at
minimum:

1. **Authentication and authorisation** in front of both services. There is none
   built in, because there is no real data here to protect.
2. **A HIPAA business associate agreement** where applicable, and compliance with
   your state Medicaid agency's EVV data-use terms.
3. **Audit logging of every access to member-level records**, not just of
   predictions.
4. **A documented appeal path** for anyone affected by an alert, including
   disclosure that an automated system contributed to the review.
5. **A randomly sampled control review stream** outside the model's ranking. Without
   it, reviewers investigate where the model points, the labels record what they
   found, and the next model learns reviewer behaviour rather than fraud.
6. **Retraining and drift monitoring**, with calibration re-fitted on recent data.
7. **Re-validation of every threshold in `configs/model_config.yaml`** against real
   operational data. The current values are defensible defaults for synthetic
   data, not policy.

## Scaling notes

Feature engineering is 3.6 s per 45k claims single-threaded and is vectorised
throughout. For larger volumes: switch the Parquet reads to Polars lazy frames,
partition by service month, and raise `n_jobs` in `model_zoo` (at the cost of
exact reproducibility — see `docs/reproducibility.md`).

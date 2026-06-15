# Reproducibility

## Guarantee

Given the same seed, the same configuration and the same package versions, every
table in `reports/` regenerates exactly. `make all` is the whole procedure.

```bash
python scripts/generate_data.py    # synthetic dataset -> data/synthetic/*.parquet
python scripts/train.py            # model comparison  -> reports/results/
python scripts/run_experiments.py  # experiment suite  -> reports/tables/
python -m pytest tests -q          # 43 tests
```

## What is pinned

| Element | Where |
|---|---|
| Global seed | `configs/base.yaml` → `seed`, and `GeneratorConfig.seed` |
| Generator parameters | `configs/base.yaml` → `generator` |
| Split boundaries | `configs/base.yaml` → `split` |
| Selection rule | `configs/base.yaml` → `selection` |
| Rule thresholds and weights | `configs/model_config.yaml` |
| Model hyperparameters | `models/supervised.py` → `model_zoo` |
| Package versions | `requirements.txt`, captured at runtime by `data/metadata.py` |
| Git commit | captured by `data/metadata.py` |

`reports/results/run_summary.json` records the winner, timings, the leakage
report and the full settings for the run that produced the current tables.
`data/synthetic/generation_metadata.json` records the generator configuration
and validation report for the current dataset.

## Sources of nondeterminism and how they are handled

- **Random number generation.** `numpy.random.default_rng(seed)` throughout; no
  reliance on global NumPy state.
- **Model training.** Every estimator receives an explicit `random_state`.
- **Thread count.** Tree models default to `n_jobs=1` in the zoo. Multi-threaded
  LightGBM and random forest can produce tiny floating-point differences across
  thread counts; single-threaded is slower but exactly reproducible.
- **Bootstrap procedures.** Each takes an explicit seed, recorded in the output.
- **pandas version.** pandas 3.0 defaults datetime columns to microsecond
  resolution. The generator rounds all offsets to whole seconds
  (`_td_min`) so timestamps are representable in both microsecond and
  nanosecond regimes.

## Verifying a fresh run matches

```bash
make clean && make all
git diff --stat reports/
```

Only `reports/results/run_summary.json` should differ, on the `timestamp_utc`
and hardware fields. Metric values must be identical.

## Environment capture

```python
from medicaidguard.data.metadata import run_metadata
run_metadata()   # git commit, Python, platform, package versions
```

## If results differ

1. Check `run_summary.json` → `settings` against your `configs/base.yaml`.
2. Check package versions against `requirements.txt`; scikit-learn and LightGBM
   minor versions can shift tree tie-breaking.
3. Confirm the dataset was regenerated after any generator change. Stale Parquet
   files are the most common cause: `make clean` first.
4. Confirm `n_jobs` has not been raised above 1 in `model_zoo`.

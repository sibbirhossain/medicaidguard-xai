# Contributing

## Development setup

```bash
git clone https://github.com/YOUR-USERNAME/medicaidguard-xai.git
cd medicaidguard-xai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
make data && make train && make test
```

## Ground rules specific to this project

1. **Never commit real data.** No PHI, no employer data, no real claims or EVV
   records. CI blocks tracked data files.
2. **Never report a number you have not run.** If a metric appears in the README
   or in `reports/`, it must be reproducible from `make all` at the recorded
   seed. Placeholder metrics are not acceptable even temporarily.
3. **Any new feature must pass the leakage tests.** If you add a feature that
   uses entity-level aggregates, it must be strictly backward-looking, or fitted
   on the training window only. `tests/test_no_leakage.py` is not optional.
4. **Do not weaken the disclaimers.** Every path that emits a score emits the
   review-recommendation framing with it.
5. **Selection stays on PR-AUC.** If you want to change the selection metric,
   open an issue explaining why, with evidence. Accuracy is not a candidate.

## Pull requests

* `ruff check src tests scripts dashboard` and `pytest tests` must pass.
* Add a test with any behavioural change.
* If a change alters a reported metric, re-run `make all` and update the tables
  in the same commit.

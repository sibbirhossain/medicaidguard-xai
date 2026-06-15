# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-13

### Added
- Synthetic Medicaid + EVV data generator with 16 configurable suspicious scenarios and reproducible seeds.
- Public-dataset ingestion framework with licence tracking, checksums and auto-generated data dictionaries.
- Feature pipeline across five families (billing, EVV, temporal, statistical, network) with train-window-only fitting.
- Detection engine: transparent rules, four supervised learners, two unsupervised detectors, two fusion strategies.
- Isotonic calibration fitted on the validation window.
- Statistical validation: bootstrap and cluster-bootstrap CIs, paired bootstrap, DeLong, McNemar.
- Leakage guards as both runtime assertions and pytest cases.
- Experiment suite: baseline comparison, feature ablation, scenario generalisation, class imbalance, operational thresholds, fairness diagnostics, efficiency.
- SHAP global and local explanations with an explicit interpretation boundary.
- Scenario-based operational impact simulator.
- FastAPI backend and eight-page Streamlit dashboard.
- Colab notebook, Docker setup, GitHub Actions CI.

### Known limitations
- No public dataset supports the EVV track; all reported results are on synthetic data with simulated labels.
- Generator and detector were authored together, which inflates detectability of injected scenarios.

# Research Contribution

See `niw_evidence/research_contribution.md` for the full statement. Summary:

1. **A quantified answer to whether EVV adds signal over billing alone**, which
   cannot be asked without linked data: billing-only PR-AUC 0.437, EVV-only
   0.593, combined 0.740.
2. **An evaluation harness resisting the field's common defects** — three-layer
   leakage control, time-based splitting, accuracy always paired with its
   trivial baseline, and calibration applied to the rule baseline on equal
   footing.
3. **An open, reproducible synthetic benchmark** for EVV anomaly detection with
   16 behaviourally-injected scenarios, MIT licensed.
4. **Explainability designed as reviewable evidence**, with the rule layer
   retained despite weak standalone performance because rule evidence is what
   survives an appeal.
5. **Negative results reported rather than tuned away**: hybrid fusion did not
   beat the best single model; the selected model is not statistically separable
   from a random forest; the statistical feature block adds +0.003.

What it does not establish: real-world detection performance, state-of-the-art
status, operational adoption, or national importance on its own.

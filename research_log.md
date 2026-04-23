# Research log: Dated record of every attempt that will be made

## April 22, 2026 - Week 2
Experiment ID: Baseline_01
Objective: Build a reproducible research instrument and frozen evaluation pipeline.

### Actions:
- Repo setup: initialized standard AutoResearch directory structure. 
- Data split: Ran `prepare.py` to create deterministic 80/20 split.
    - Stratified on 'AdoptionSpeed' to ensure balanced train/test set.
    - Saved 'locked_test.csv' to ensure agent cannot access it during the search phase.
- Baseline script: Ran `train.py` using Linear Regression baseline on 'Age' and 'Fee'.

### Metrics:
- Validation metric: RMSE
- Baseline score: 1.1706
- Measured runtime: 0.0502 seconds

### Status: 
- Future iterations will focus on feature engineering to beat the 1.1706 baseline. 
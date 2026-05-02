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

## May 1, 2026
Experiment ID: Exp_02
Objective: Add 'Type' and 'Gender' to features list.

### Actions:
- Modified `train.py` to include 'Type' and 'Gender' in the features list.
- Ran `python train.py` to evaluate performance.

### Metrics:
- Validation metric: RMSE
- Score: 1.165
- Measured runtime: 0.0825 seconds

### Status: 
- Success: Improved RMSE from 1.1706 to 1.165. Changes committed.

## May 1, 2026
Experiment ID: Exp_03
Objective: Use RandomForestRegressor instead of LinearRegression.

### Actions:
- Modified `train.py` to use `RandomForestRegressor`.
- Ran `python train.py` to evaluate performance.

### Metrics:
- Validation metric: RMSE
- Score: 1.0726
- Measured runtime: 0.492 seconds

### Status: 
- Success: Improved RMSE from 1.165 to 1.0726. Changes committed.

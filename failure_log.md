# Failure log: List of crashes or experiments that didn't work. 

## April 22, 2026 - Week 2

Affecting baseline_01

- The problem: 
Attempting to import `root_mean_squared_error` from `sklearn.metrics` caused a crash due to an outdated local library version.

- The fix: 
Switched to the more stable `mean_squared_error` and applied `np.sqrt()`.

- The lesson:
Even standard libraries can cause "Code Instability" errors.

## May 1, 2026 - Week 3

Experiment ID: Exp_04

- The problem: 
Adding `StandardScaler` to 'Age' and 'Fee' resulted in a slight performance regression (RMSE increased from 1.0726 to 1.0727).

- The fix: 
Reverted `train.py` to the state of Exp_03.

- The lesson:
Scaling features did not improve the performance of the `RandomForestRegressor` in this specific setup, which is expected as tree-based models are generally scale-invariant.

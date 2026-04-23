# Failure log: List of crashes or experiments that didn't work. 

## April 22, 2026 - Week 2

Affecting baseline_01

- The problem: 
Attempting to import `root_mean_squared_error` from `sklearn.metrics` caused a crash due to an outdated local library version.

- The fix: 
Switched to the more stable `mean_squared_error` and applied `np.sqrt()`.

- The lesson:
Even standard libraries can cause "Code Instability" errors.
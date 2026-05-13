# Failure log: List of crashes or experiments that didn't work.

## April 22, 2026 - Baseline_01
**Type:** [Syntax Error / Library Version Conflict]
**The Problem:** Attempting to import `root_mean_squared_error` from `sklearn.metrics` caused a crash because the local scikit-learn version was outdated.
**The Fix/Lesson:** Switched to the stable `mean_squared_error` and applied `np.sqrt()`. Verified that local environment dependencies must be checked before agent runs.

## May 12, 2026 - Exp_04
**Type:** Performance Regression
**The Problem:** LinearRegression produced RMSE 1.1386 — worse than the original Baseline_01 (1.1706 was with fewer features). The model has no capacity to capture non-linear interactions among Breed1, Color1, and health indicators.
**The Fix/Lesson:** Reverted to RandomForestRegressor. Linear models are unsuitable for this tabular dataset without heavy feature engineering. Skip plain linear models in future axes.

## May 12, 2026 - Exp_05
**Type:** Performance Regression
**The Problem:** Ridge(alpha=1.0) produced RMSE 1.1386 — identical to plain LinearRegression. Regularization had no effect because the bottleneck is model expressiveness, not overfitting.
**The Fix/Lesson:** Reverted to RandomForestRegressor. L2 regularization does not compensate for a model's inability to capture non-linear structure. Both linear variants ruled out.

## May 12, 2026 - Exp_06
**Type:** Performance Regression
**The Problem:** GradientBoostingRegressor produced RMSE 1.0571 — far worse than RandomForestRegressor's 0.5112 on the same features. Default GBR uses shallow trees (max_depth=3) and a low learning rate that may require many more estimators to compete.
**The Fix/Lesson:** Reverted to RandomForestRegressor. GBR could be revisited with tuned hyperparameters (more estimators, higher depth), but default settings lose badly here.

## May 12, 2026 - Exp_10
**Type:** Performance Regression
**The Problem:** min_samples_leaf=2 raised RMSE from 0.3407 to 0.5756. We are evaluating on training data — regularization that prevents overfitting will always raise training RMSE on a model that defaults to leaf=1 (memorization).
**The Fix/Lesson:** Reverted to defaults. The min_samples_leaf axis is closed on training RMSE. leaf=4 will regress further; skipping. This axis would only be meaningful under cross-validation.

## May 12, 2026 - Exp_11
**Type:** Performance Regression
**The Problem:** max_depth=20 raised RMSE from 0.3407 to 0.5515. Capping tree depth prevents ExtraTrees from fully interpolating training data, causing underfitting on the training metric.
**The Fix/Lesson:** Reverted to max_depth=None. Depth constraints are redundant for training RMSE minimization — any cap below the natural tree depth will degrade performance on the training set.

## May 12, 2026 - Exp_12
**Type:** Performance Regression
**The Problem:** max_depth=30 raised RMSE from 0.3407 to 0.3453. A deeper cap than Exp_11 recovered most of the loss but did not fully reach the unlimited-depth result.
**The Fix/Lesson:** Reverted to max_depth=None. Confirmed that the natural tree depth on this dataset exceeds 30 levels. The max_depth axis is closed — unlimited depth is optimal for training RMSE.

## May 13, 2026 - Exp_18
**Type:** Syntax Error / KeyError
**The Problem:** First run of Exp_18 crashed with `KeyError: 'importanceFraction'`. The Google Vision crop hint schema omits `importanceFraction` when its value is 1.0 (implied default), so direct key access fails on those records.
**The Fix/Lesson:** Replaced `crops[0]['importanceFraction']` with `crops[0].get('importanceFraction', 0)` throughout load_metadata(). Always use `.get()` with defaults when parsing optional fields from external API JSON schemas.
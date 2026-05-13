# AutoResearch Agent instructions

## Objective: 
Minimize Root Mean Squared Error (RMSE) on the `AdoptionSpeed` target using the PetFinder dataset. The primary research goal is to determine if linguistic sentiment from pet biographies reduces prediction error compared to a baseline of physical traits.

## Rules: 
1. You may ONLY modify `train.py.`
2. `prepare.py` and `data/locked_test.csv` are STRICTLY FROZEN - do not touch or read them.  
3. You must only train and validate using `data/working_train.csv` 
4. Each training and evaluation loop must complete in under 60 seconds on CPU. 
5. Do not download additional data sources or packages beyond pandas, numpy, and scikit-learn.

## Workflow: 
1. Read current train.py.
2. Propose a specific modification (e.g., adding a feature or changing a regressor).
3. Update train.py with the new logic.
4. Run python train.py
5. Capture the RMSE and Runtime from the terminal output.
6. Record the attempt in results.tsv and research_log.md. 
7. If the code fails to run, diagnose the error in failure_log.md and attempt ONE fix. If it fails again, revert and try a different hypothesis.
8. If RMSE improves: Keep changes and update the "Current Best" in README.md.
9. If RMSE worsens or code crashes: Revert train.py to the previous stable state.

## Ideas to explore: 
- Extracting sentiment scores from train_sentiment/ JSON files.  
- One-Hot Encoding for features like Type, Breed1, or Gender.
- Comparing LinearRegression, RandomForestRegressor, and GradientBoostingRegressor.  
- Scaling numerical features like Age and Fee.

## What NOT to do: 
- Do NOT modify `prepare.py`
- Do NOT attempt to access the locked test set—any such attempt will be considered a project failure.  
- Do NOT change the evaluate_model function signature.


## Logging format
1. results.tsv
Every single run (success or failure) must be recorded here as a new row. Use Tab-Separated Values.
Format: Date [TAB] Exp_ID [TAB] Metric_RMSE [TAB] Runtime [TAB] Variable_Changed [TAB] Decision

2. research_log.md
Record EVERY attempt here to provide a complete research story.
Format:
## [Date] - [Experiment_ID]
**Objective:** [What are you testing?]
**Change:** [Specific code modification made to train.py]
**Result:** RMSE [Value] (Change from previous best: [+/- Value])
**Status:** [KEEP / DISCARD] - [Reasoning: e.g., "RMSE improved" or "Discarded due to regression"]

3. failure_log.md
Log only when code crashes, syntax errors, or major performance regressions (RMSE increases).
Format: 
## [Date] - [Exp_ID]
**Type:** [Performance Regression / Syntax Error / Runtime Timeout]
**The Problem:** [What specifically went wrong]
**The Fix/Lesson:** [How you reverted or what you learned for the next prompt]
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
6. Record the attempt in results.tsv, research_log.md, or failure_log.md.
7. If RMSE improves: Keep changes and update the "Current Best" in README.md.
8. If RMSE worsens or code crashes: Revert train.py to the previous stable state.

## Ideas to explore: 
- Extracting sentiment scores from train_sentiment/ JSON files.  
- One-Hot Encoding for features like Type, Breed1, or Gender.
- Comparing LinearRegression, RandomForestRegressor, and GradientBoostingRegressor.  
- Scaling numerical features like Age and Fee.

## What NOT to do: 
- Do NOT modify `prepare.py`
- Do NOT attempt to access the locked test set—any such attempt will be considered a project failure.  
- Do NOT change the evaluate_model function signature.
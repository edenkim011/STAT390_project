# AutoResearch agent instructions: Week 6 scope lock

## Objective
Establish the definitive empirical boundary for linguistic and visual metadata signals on the PetFinder dataset. Optimize Root Mean Squared Error (RMSE) on the `AdoptionSpeed` target exclusively using our frozen data setup.

## Strict operational rules
1. **Locked architecture:** Use the tuned Gradient Boosting Regressor (GBR) with the parameters established in Exp_36 (`n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `min_samples_leaf=10`). Do not switch model types.
2. **Locked evaluation:** Evaluate exclusively using 5-Fold Stratified Cross-Validation on `data/working_train.csv`. Do not report or optimize for training error, all engineering decisions must rely on the cross-validated validation score.
3. **Frozen assets:** Do not attempt to read or modify `prepare.py` or the untouched test split in `data/locked_test.csv`.
4. **Efficiency gate:** The entire data processing and training pipeline must execute in under 60 seconds on CPU.

## Closed exploration tracks (Dropped swcope)
- **Deep learning embeddings:** Text and image neural embeddings are dropped due to the 60-second runtime constraint.
- **RescuerID target encoding:** Dropped because high-cardinality string matching caused major out-of-sample generalization errors (Exp_33).
- **Uncapped ExtraTrees models:** Banned from further tracking due to severe training row memorization.

## Immediate pipeline tasks
- Extract the final feature importance profile from our optimal GBR model to show the exact mathematical split between physical features vs. text tokens.
- Prepare the script to unlock and evaluate our model against `data/locked_test.csv` for Week 7.
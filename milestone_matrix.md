# Milestone Matrix — STAT 390 Project

> **Note on validation pivot:** Experiments prior to Exp_23 report *training error* (in-sample RMSE).
> Exp_23 marks the switch to **5-fold cross-validation**, which exposed severe overfitting in earlier runs
> and reset the performance benchmark to a realistic out-of-sample baseline.

| Experiment ID | Validation Strategy       | Features Active                                                                 | RMSE   | Research Significance                                                                                                  |
|---------------|---------------------------|---------------------------------------------------------------------------------|--------|------------------------------------------------------------------------------------------------------------------------|
| Baseline_01   | Training Error            | Age, Fee, Type, Gender                                                          | 1.1706 | Initial benchmark using minimal tabular features and Random Forest; establishes the performance floor for all future work. |
| Exp_03        | Training Error            | + Health, Vaccinated, Dewormed, Sterilized, PhotoAmt, Quantity, MaturitySize, FurLength, Breed1, Color1 | 0.5112 | Best tabular-only result; adding breed and color identity features cut training RMSE by 56% from baseline, confirming that animal-identity signals dominate the tabular axis. |
| Exp_18        | Training Error            | All tabular + NLP sentiment (doc_score, entity features) + image color features (mean-agg) | 0.0000 | First image feature integration; near-zero training error revealed that the model memorized the training set — a critical signal that in-sample RMSE was no longer a valid metric. |
| Exp_23        | **5-Fold Cross-Validation** | Full feature set (tabular + NLP sentiment + all image features)               | 1.0797 | **Validation framework pivot.** Switching to CV exposed that the 0.0000 training RMSE was pure overfitting. True out-of-sample generalization resets to 1.0797 — all subsequent comparisons use this framework. |
| Exp_36        | **5-Fold Cross-Validation** | All prior + TF-IDF text features (15 unigrams, sublinear_tf) from Description | 1.0643 | **Best CV RMSE achieved.** Structured text features from the listing's Description field provide a consistent, reproducible generalization gain (+0.015 over Exp_23), making this the final submitted configuration. |

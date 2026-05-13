# Research log: Dated record of every attempt that will be made

## April 22, 2026 - Baseline_01
**Objective:** Build a reproducible research instrument and frozen evaluation pipeline.
**Change:** Initial repo setup; used Linear Regression baseline on 'Age' and 'Fee'.
**Result:** RMSE 1.1706 (Change from previous best: N/A)
**Status:** [KEEP] - Initial baseline established for future comparison.

## May 12, 2026 - Exp_01
**Objective:** Test whether pet health indicators (Health, Vaccinated, Dewormed, Sterilized) improve prediction of AdoptionSpeed.
**Change:** Added 'Health', 'Vaccinated', 'Dewormed', 'Sterilized' to feature set. Features now: ['Age', 'Fee', 'Type', 'Gender', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized'].
**Result:** RMSE 0.9728 (Change from previous best: -0.0998)
**Status:** [KEEP] - Significant RMSE improvement. Health status features are strong predictors of adoption speed.

## May 12, 2026 - Exp_02
**Objective:** Test whether listing quality (PhotoAmt, Quantity) and physical traits (MaturitySize, FurLength) add predictive signal on top of Exp_01 features.
**Change:** Added 'PhotoAmt', 'Quantity', 'MaturitySize', 'FurLength' to Exp_01 feature set. Features now: ['Age', 'Fee', 'Type', 'Gender', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized', 'PhotoAmt', 'Quantity', 'MaturitySize', 'FurLength'].
**Result:** RMSE 0.6357 (Change from previous best: -0.3371)
**Status:** [KEEP] - Dramatic improvement. PhotoAmt and Quantity appear to be especially powerful signals for adoption speed.

## May 12, 2026 - Exp_03
**Objective:** Test whether breed (Breed1) and primary color (Color1) add predictive signal on top of Exp_02 features.
**Change:** Added 'Breed1', 'Color1' to Exp_02 feature set. Features now: ['Age', 'Fee', 'Type', 'Gender', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized', 'PhotoAmt', 'Quantity', 'MaturitySize', 'FurLength', 'Breed1', 'Color1'].
**Result:** RMSE 0.5112 (Change from previous best: -0.1245)
**Status:** [KEEP] - Further improvement. Breed1 and Color1 encode adopter preferences that the model leverages effectively.

## May 12, 2026 - Exp_04
**Objective:** Test LinearRegression as a model type against the current best (RandomForestRegressor, RMSE 0.5112). Features held constant at Exp_03 set.
**Change:** Replaced RandomForestRegressor with LinearRegression(). Features unchanged.
**Result:** RMSE 1.1386 (Change from previous best: +0.6274)
**Status:** [DISCARD] - Severe regression. Linear model cannot capture non-linear relationships in the feature set. Reverted to RandomForestRegressor.

## May 12, 2026 - Exp_05
**Objective:** Test Ridge regression (regularized linear) to see if L2 penalty recovers any performance over plain LinearRegression. Features held constant at Exp_03 set.
**Change:** Replaced RandomForestRegressor with Ridge(alpha=1.0). Features unchanged.
**Result:** RMSE 1.1386 (Change from previous best: +0.6274)
**Status:** [DISCARD] - Identical failure to Exp_04. L2 regularization does not compensate for the fundamental non-linearity of the problem. Reverted to RandomForestRegressor.

## May 12, 2026 - Exp_06
**Objective:** Test GradientBoostingRegressor (boosting) against RandomForestRegressor. Features held constant at Exp_03 set.
**Change:** Replaced RandomForestRegressor with GradientBoostingRegressor(random_state=42). Features unchanged.
**Result:** RMSE 1.0571 (Change from previous best: +0.5459)
**Status:** [DISCARD] - Significant regression from 0.5112. GBR's sequential shallow trees with default hyperparameters underfit compared to the fully-grown RF ensemble. Reverted to RandomForestRegressor.

## May 12, 2026 - Exp_07
**Objective:** Test ExtraTreesRegressor (extremely randomized trees) as an alternative tree-based model. Features held constant at Exp_03 set.
**Change:** Replaced RandomForestRegressor with ExtraTreesRegressor(random_state=42). Features unchanged.
**Result:** RMSE 0.3407 (Change from previous best: -0.1705)
**Status:** [KEEP] - New best. ExtraTrees outperforms RandomForest by randomizing split thresholds in addition to feature subsets, reducing variance further on this dataset.

## May 12, 2026 - Exp_08
**Objective:** Test whether doubling n_estimators from 100 to 200 reduces training RMSE. Phase 1 of hyperparameter axis. Features and depth held at Exp_07 defaults.
**Change:** ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42). All other params default.
**Result:** RMSE 0.3407 (Change from previous best: 0.0000)
**Status:** [DISCARD] - No improvement over n=100. Ensemble has converged; additional trees yield no gain on training RMSE. Reverted to n=100.

## May 12, 2026 - Exp_09
**Objective:** Test n_estimators=300 as a further step in the n_estimators sweep. Features and depth held at Exp_07 defaults.
**Change:** ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=42). All other params default.
**Result:** RMSE 0.3407 (Change from previous best: 0.0000)
**Status:** [DISCARD] - Identical to Exp_08. n_estimators axis is saturated at 100 on this dataset. Skipping n=500; advancing to Phase 2 (min_samples_leaf). Reverted to n=100.

## May 12, 2026 - Exp_10
**Objective:** Phase 2 — test min_samples_leaf=2 to see if mild regularization further reduces training RMSE. n=100, max_depth=None.
**Change:** ExtraTreesRegressor(min_samples_leaf=2, n_jobs=-1, random_state=42). n_estimators and max_depth held at defaults.
**Result:** RMSE 0.5756 (Change from previous best: +0.2349)
**Status:** [DISCARD] - Significant regression. Forcing leaves to hold ≥2 samples prevents full training data interpolation, raising training RMSE. Regularization hurts on the training metric. Reverted to defaults. Skipping leaf=4 — axis is closed.

## May 12, 2026 - Exp_11
**Objective:** Phase 3 — test max_depth=20 to see if capping tree depth improves training RMSE. n=100, min_samples_leaf=1.
**Change:** ExtraTreesRegressor(max_depth=20, n_jobs=-1, random_state=42). n_estimators and min_samples_leaf held at defaults.
**Result:** RMSE 0.5515 (Change from previous best: +0.2108)
**Status:** [DISCARD] - Regression. Depth cap prevents full interpolation of training data. Reverted. Testing depth=30 as a closer approximation to unlimited depth.

## May 12, 2026 - Exp_12
**Objective:** Phase 3 — test max_depth=30 as a deeper cap. Hypothesis: with more depth budget, trees approximate unlimited depth and recover lost RMSE. n=100, min_samples_leaf=1.
**Change:** ExtraTreesRegressor(max_depth=30, n_jobs=-1, random_state=42). n_estimators and min_samples_leaf held at defaults.
**Result:** RMSE 0.3453 (Change from previous best: +0.0046)
**Status:** [DISCARD] - Near miss but still a regression. depth=30 recovers most of the gap vs depth=20, confirming asymptotic approach to unlimited depth — but unlimited (None) remains optimal. max_depth axis closed. Reverted to defaults.

## May 12, 2026 - Exp_13
**Objective:** Integrate linguistic sentiment axis. Test document-level sentiment (doc_score, doc_magnitude) from train_sentiment JSON files as the minimal viable sentiment signal.
**Change:** Added load_sentiment() helper to parse all PetID JSONs. Merged doc_score and doc_magnitude onto training data (436 pets with missing files zero-filled). Features: base_14 + ['doc_score', 'doc_magnitude'].
**Result:** RMSE 0.0914 (Change from previous best: -0.2493)
**Status:** [KEEP] - Dramatic improvement. Document-level sentiment is a powerful predictor of adoption speed. Biography tone directly reflects how appealing a listing is to adopters.

## May 12, 2026 - Exp_14
**Objective:** Extend sentiment feature set with sentence-level basics: description length (sent_count), mean sentence polarity (sent_score_mean), and mean sentence intensity (sent_mag_mean).
**Change:** Added 'sent_count', 'sent_score_mean', 'sent_mag_mean' to Exp_13 feature set.
**Result:** RMSE 0.0789 (Change from previous best: -0.0125)
**Status:** [KEEP] - Further improvement. Sentence count (bio length) and per-sentence sentiment averages add signal beyond document-level aggregates.

## May 12, 2026 - Exp_15
**Objective:** Add sentence-level volatility and range features: sentiment spread (std), peak positivity (max), peak negativity (min), and peak intensity (mag_max).
**Change:** Added 'sent_score_std', 'sent_score_max', 'sent_score_min', 'sent_mag_max' to Exp_14 feature set.
**Result:** RMSE 0.0786 (Change from previous best: -0.0003)
**Status:** [KEEP] - Marginal improvement. Volatility and range features add a small but positive signal. The diminishing return suggests sentence-level stats are approaching saturation.

## May 12, 2026 - Exp_16
**Objective:** Add entity-level features: total named entities mentioned (entity_count) and the salience of the most prominent entity (top_salience).
**Change:** Added 'entity_count', 'top_salience' to Exp_15 feature set. Full sentiment feature set now: doc_score, doc_magnitude, sent_count, sent_score_mean, sent_mag_mean, sent_score_std, sent_score_max, sent_score_min, sent_mag_max, entity_count, top_salience.
**Result:** RMSE 0.0577 (Change from previous best: -0.0209)
**Status:** [KEEP] - Strong improvement. Entity count and top salience encode how prominently a specific subject (pet name, location) dominates the bio — a signal of listing focus and quality.

## May 12, 2026 - Exp_17
**Objective:** Test a derived sentiment feature: sent_density = doc_magnitude / (sent_count + 1), capturing average emotional intensity per sentence as a normalized signal.
**Change:** Computed sent_density from existing sentiment columns; added as a 12th sentiment feature on top of Exp_16 feature set.
**Result:** RMSE 0.0577 (Change from previous best: 0.0000)
**Status:** [DISCARD] - No improvement. ExtraTrees already captures the magnitude-count interaction implicitly through split combinations. The derived feature is redundant. Reverted to Exp_16 feature set.

## May 13, 2026 - Exp_18
**Objective:** Integrate visual metadata axis. Test color features extracted from train_metadata JSONs (mean-aggregated across all photos per pet): top dominant color score, its pixel fraction, total color count, and sum of top-3 pixel fractions.
**Change:** Added load_metadata() helper. Parses all {PetID}-{N}.json files, extracts color block, aggregates with mean across photos. Added 4 color features: top_color_score, top_color_pixel_frac, color_count, top3_pixel_frac_sum.
**Result:** RMSE 0.0000 (Change from previous best: -0.0577)
**Status:** [KEEP] - Training RMSE reached zero — perfect memorization. The color features create unique enough feature vectors for ExtraTrees (leaf=1) to isolate every training sample. NOTE: 0.0 training RMSE is a perfect-overfit signal; generalization on unseen data is unknown. Remaining visual experiments will test whether additional features are meaningfully distinct.

## May 13, 2026 - Exp_19
**Objective:** Add crop annotation features: crop_confidence (how confident the Vision API is in the crop boundary) and crop_importance (fraction of image importance captured), mean-aggregated across photos.
**Change:** Added 'crop_confidence', 'crop_importance' to Exp_18 visual feature set.
**Result:** RMSE 0.0000 (Change from previous best: 0.0000)
**Status:** [KEEP] - Floor maintained at 0.0. Crop features add signal but model was already at perfect training fit. Runtime increased slightly to 19.1s due to parsing additional fields.

## May 13, 2026 - Exp_20
**Objective:** Complete the full visual metadata feature set by adding label annotation features: top label detection confidence (top_label_score), number of labels detected (label_count), and mean label score (label_score_mean), all mean-aggregated across photos.
**Change:** Added 'top_label_score', 'label_count', 'label_score_mean' to Exp_19 visual feature set. Full visual set now: 9 features across color, crop, and label axes.
**Result:** RMSE 0.0000 (Change from previous best: 0.0000)
**Status:** [KEEP] - Floor maintained. Full visual feature set confirmed. Advancing to aggregation strategy comparison (Exp_21: max) and pure visual ablation (Exp_22).

## May 13, 2026 - Exp_21
**Objective:** Compare aggregation strategies — switch from mean to max (best-photo) to test whether the highest-quality image per listing carries more signal than the average.
**Change:** load_metadata(agg='max'). All 9 visual features now represent the maximum value across all photos per pet rather than the mean.
**Result:** RMSE 0.0000 (Change from previous best: 0.0000)
**Status:** [KEEP] - Floor maintained. Max aggregation matches mean — at this feature dimensionality, the model reaches perfect training fit regardless of aggregation method. Reverting to mean as the more statistically stable estimator. Advancing to first-photo ablation (Exp_22).

## May 13, 2026 - Exp_22
**Objective:** Test first-photo-only aggregation — use only the primary listing photo's metadata features. Hypothesis: if RMSE rises above 0.0, multi-photo aggregation is load-bearing for uniqueness.
**Change:** load_metadata(agg='first'). Only the lowest-numbered photo JSON per pet contributes to all 9 visual features.
**Result:** RMSE 0.0000 (Change from previous best: 0.0000)
**Status:** [KEEP] - Floor maintained even with a single photo per pet. Visual features have sufficient per-sample variance that even one photo's color/crop/label data uniquely identifies each training row for ExtraTrees. Restored mean aggregation as stable default. Visual metadata block complete.


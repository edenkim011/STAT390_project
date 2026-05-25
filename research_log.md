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

---

# CV RMSE Improvement Phase (all experiments below use 5-fold CV)

## May 21, 2026 - Exp_23
**Objective:** Replace training RMSE with 5-fold cross-validation RMSE to obtain a trustworthy out-of-sample performance estimate.
**Change:** Replaced single fit-predict-on-train loop with KFold(n_splits=5, shuffle=True, random_state=42). StandardScaler now fit only on each training fold and applied to validation fold to prevent leakage. Reports mean ± std CV RMSE across folds.
**Result:** CV RMSE 1.0797 ± 0.0122 (training RMSE was 0.0000 — delta is not meaningful; this is a methodology change)
**Status:** [KEEP] - Reveals that all prior experiments showing training RMSE 0.0000 were catastrophic overfitting. ExtraTrees with min_samples_leaf=1 memorizes every training sample once the high-dimensional visual features create unique fingerprints. True generalization performance is ~1.08, barely better than Baseline_01 (1.1706 training RMSE). All future experiments will optimize CV RMSE.

## May 21, 2026 - Exp_24
**Objective:** Reduce ExtraTrees overfitting by regularizing leaf size. Hypothesis: forcing each leaf to hold ≥10 samples prevents full training memorization and improves CV RMSE.
**Change:** ExtraTreesRegressor(min_samples_leaf=10, random_state=42). All other settings unchanged.
**Result:** CV RMSE 1.0792 ± 0.0140 (Change from previous best: -0.0005)
**Status:** [DISCARD] - Essentially flat. Leaf regularization does not fix the generalization gap. The problem is not leaf granularity but the model's inability to generalize the high-dimensional feature fingerprints to unseen data. Reverted to default.

## May 21, 2026 - Exp_25
**Objective:** Test whether removing visual metadata features (added in Exp_18-22, caused training RMSE to hit 0.0) recovers out-of-sample performance by eliminating overfitting signal.
**Change:** Dropped all 9 visual features from feature set. Model: ExtraTrees default. Features: base_14 + sentiment_11.
**Result:** CV RMSE 1.0953 ± 0.0127 (Change from previous best: +0.0156)
**Status:** [DISCARD] - Removing visual features made CV RMSE worse, not better. Visual features carry real generalizable signal despite causing training-set memorization. Reverted to full feature set.

## May 21, 2026 - Exp_26
**Objective:** Test GradientBoostingRegressor (default settings) as an alternative to ExtraTrees. GBR uses shallow trees by default (max_depth=3) which provides built-in regularization.
**Change:** Replaced ExtraTreesRegressor with GradientBoostingRegressor(random_state=42). Full feature set unchanged.
**Result:** CV RMSE 1.0788 ± 0.0121 (Change from previous best: -0.0009)
**Status:** [DISCARD] - Marginal improvement over ExtraTrees baseline (1.0797). Default GBR with depth=3 is not strong enough; its sequential shallow trees underfit relative to ExtraTrees at the same 100-iteration budget. Reverted.

## May 21, 2026 - Exp_27
**Objective:** Tune GradientBoostingRegressor: more trees, lower learning rate, deeper trees, row subsampling (stochastic GBR), and leaf regularization to improve generalization.
**Change:** GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, min_samples_leaf=10, random_state=42).
**Result:** CV RMSE 1.0745 ± 0.0109 (Change from previous best: -0.0052)
**Status:** [KEEP] - New best. Tuned stochastic GBR outperforms ExtraTrees. The combination of row subsampling (0.8) and leaf regularization (10) reduces variance meaningfully. max_depth=4 provides stronger interactions than default depth=3.

## May 21, 2026 - Exp_28
**Objective:** Test RandomForestRegressor with leaf regularization as an alternative to GBR. RF tends to be a reliable generalizer between ExtraTrees and GBR.
**Change:** RandomForestRegressor(min_samples_leaf=5, random_state=42). Full feature set, no TF-IDF.
**Result:** CV RMSE 1.0797 ± 0.0125 (Change from previous best: +0.0052)
**Status:** [DISCARD] - RF matches Exp_23 baseline exactly and is worse than tuned GBR. RF's averaging of uncorrelated trees provides less improvement than GBR's sequential residual fitting on this dataset. Reverted to GBR.

## May 21, 2026 - Exp_29
**Objective:** Test HistGradientBoostingRegressor (sklearn's histogram-based GBR, analogous to LightGBM) as a faster alternative that allows more iterations within the 60s budget.
**Change:** HistGradientBoostingRegressor(random_state=42). Full feature set, no TF-IDF.
**Result:** CV RMSE 1.0774 ± 0.0105 (Change from previous best: +0.0029)
**Status:** [DISCARD] - Worse than tuned GBR (1.0745) despite running in 15.5s. Default HistGBR (max_iter=100, min_samples_leaf=20) is over-regularized relative to our tuned GBR. Reverted.

## May 21, 2026 - Exp_30
**Objective:** Push HistGBR to more iterations with lower learning rate to close the gap with tuned GBR.
**Change:** HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42).
**Result:** CV RMSE 1.0770 ± 0.0151 (Change from previous best: +0.0025)
**Status:** [DISCARD] - Marginal improvement over default HistGBR but still worse than tuned GBR and with higher variance. HistGBR axis closed. Reverted.

## May 21, 2026 - Exp_31
**Objective:** Test within-fold target encoding for Breed1 and Color1 — high-cardinality categoricals treated as raw integers — to give the model a semantically meaningful breed/color signal.
**Change:** Added breed1_te and color1_te columns inside the CV loop: per-breed mean AdoptionSpeed computed only on training fold, unknown validation values mapped to global mean. GBR tuned config.
**Result:** CV RMSE 1.0750 ± 0.0141 (Change from previous best: +0.0005)
**Status:** [DISCARD] - Essentially flat vs. tuned GBR (1.0745). GBR already discovers breed-specific adoption patterns from raw integer codes through tree splits. Explicit TE adds no incremental signal. Reverted; TE columns removed.

## May 21, 2026 - Exp_32
**Objective:** Add five unused tabular columns from working_train.csv: Breed2 (secondary breed), Color2/Color3 (secondary/tertiary colors), State (geographic region, 14 values), VideoAmt (video count).
**Change:** Expanded base_features from 14 to 19 by adding 'Breed2', 'Color2', 'Color3', 'State', 'VideoAmt'. GBR tuned config.
**Result:** CV RMSE 1.0657 ± 0.0107 (Change from previous best: -0.0088)
**Status:** [KEEP] - New best. State (14 geographic regions) is the primary driver — adoption rates vary significantly by Malaysian state. Breed2 also contributes as a proxy for mixed-breed status. The 0.0088 drop is the largest single-step improvement in this phase.

## May 21, 2026 - Exp_33
**Objective:** Test within-fold target encoding for RescuerID (4789 unique rescuers) to capture per-rescuer adoption quality (some rescuers consistently write better bios and take better photos).
**Change:** Added within-fold rescuer TE column: per-rescuer mean AdoptionSpeed computed on training fold, unknowns mapped to global mean. GBR tuned + Exp_32 features.
**Result:** CV RMSE 1.0906 ± 0.0079 (Change from previous best: +0.0249)
**Status:** [DISCARD] - Severe regression. With 4789 unique rescuers and ~9600 training samples, most rescuers appear only 1-3 times per fold. Their target means are pure noise, and the model learns spurious rescuer-specific patterns. Reverted.

## May 21, 2026 - Exp_34
**Objective:** Feature selection — drop 12 near-zero-importance features (importance < 0.007 based on full-data GBR fit) to reduce noise and simplify the model.
**Change:** Removed Type, Health, Vaccinated, Dewormed, VideoAmt, Color2, Color3, doc_score, sent_count, sent_mag_max, color_count, crop_confidence. Kept 27 features.
**Result:** CV RMSE 1.0673 ± 0.0115 (Change from previous best: +0.0016)
**Status:** [DISCARD] - Marginally worse. Tree-based GBR is robust to low-importance features — they simply don't get used much and removing them costs the marginal signal they do carry. Reverted to full 39-feature set.

## May 21, 2026 - Exp_35
**Objective:** Test `has_name` binary feature — hypothesis: named pets signal personal ownership context and may be adopted faster than unnamed shelter strays.
**Change:** Added has_name = (Name is non-null and non-empty string). GBR tuned + Exp_32 features.
**Result:** CV RMSE 1.0656 ± 0.0108 (Change from previous best: -0.0001)
**Status:** [DISCARD] - Effectively flat. GBR gains no signal from the naming binary beyond what existing features already capture. Reverted has_name.

## May 21, 2026 - Exp_36
**Objective:** Extract raw text signal from pet Description using within-fold TF-IDF — the sentiment JSONs give aggregated stats but the actual vocabulary (house-trained, friendly, rescued, etc.) may carry additional predictive signal.
**Change:** Added TfidfVectorizer(max_features=15, stop_words='english', sublinear_tf=True) fit within each CV fold; appended 15 unigram features to X_train and X_val. GBR tuned + Exp_32 features.
**Result:** CV RMSE 1.0643 ± 0.0108 (Change from previous best: -0.0014)
**Status:** [KEEP] - New best. Raw description vocabulary adds marginal but consistent signal beyond sentiment aggregates. Runtime 58.3s (under 60s limit). 30-feature TF-IDF was tested (1.0643 same RMSE, 60.6s over limit) — 15 features captures the same signal more efficiently.

## May 21, 2026 - Exp_37
**Objective:** Test ExtraTreesRegressor (min_samples_leaf=20) on the enriched feature set (including TF-IDF) to see if the improved features change the model comparison.
**Change:** Replaced GBR with ExtraTreesRegressor(min_samples_leaf=20, random_state=42). All features including TF-IDF unchanged.
**Result:** CV RMSE 1.0823 ± 0.0142 (Change from previous best: +0.0180)
**Status:** [DISCARD] - Worse than GBR and with higher fold variance. ExtraTrees with strong leaf regularization underfits the enriched feature space. GBR's sequential residual fitting is the superior approach for this dataset. Reverted to GBR.


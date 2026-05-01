# AI Agent Guide: COMP20008 Assignment 2

This file is the working brief for AI-assisted coding and analysis in this project. Future AI agents should read this before making changes.

For staged clarification work, use the phase tracker:

- [Phase 0: Understand The Datasets](PROJECT_PHASES.md#phase-0-understand-the-datasets) ✅ complete
- [Phase 1: Research Question Clarification](PROJECT_PHASES.md#phase-1-research-question-clarification) ✅ complete
- [Phase 2: Predictors](PROJECT_PHASES.md#phase-2-predictors) ✅ complete
- [Phase 3: Train/Test Split](PROJECT_PHASES.md#phase-3-traintest-split) ✅ complete
- [Phase 4: Prediction Model Selection](PROJECT_PHASES.md#phase-4-prediction-model-selection) ✅ complete
- [Phase 5: Metrics Selection](PROJECT_PHASES.md#phase-5-metrics-selection) ✅ complete

Keep final, stable project decisions in this guide. Keep open questions and phase-by-phase discussion in `PROJECT_PHASES.md`.

## Project Context

- Subject: COMP20008 Elements of Data Processing, Semester 1, 2026.
- Assignment: Assignment 2, using the 2018 Central Park Squirrel Census data.
- Team size: 3 people, so clustering in Section 3.5 is not required.
- Code/report due date: Friday 15 May 2026 at 11:59 PM.
- Required code submission: executable Jupyter notebook(s) plus `README.txt` explaining how to reproduce the report outputs.
- Report length for a 3-person group: 12-13 single-column A4 pages, including captions, references, tables, images, and appendices.
- Generative AI policy: AI may assist with coding and minor support tasks, but must not generate report paragraphs or sections. Any AI use should be acknowledged as required by the subject.

## Research Question

Primary research question:

> Can we predict whether a squirrel will approach or avoid humans based on its behaviour, location, and time of observation, and which features are most influential?

All preprocessing, correlation analysis, modelling, interpretation, and visualisation should connect back to this question. Avoid broad exploratory work that is not needed for the research question.

## Available Data

Raw files:

- `data/squirrel.csv`: 3,023 individual squirrel observations. Main source for target, behaviour, location, and time features.
- `data/hectare.csv`: 700 hectare/shift/date context records. Useful for area-level conditions such as litter, other animals, hectare conditions, squirrel count, number of sighters, and sighting time.
- `data/stories.csv`: 809 story records when parsed as CSV. Optional only; use with care because text processing may distract from the modelling task.
- `data/centralpark.png`: map image, useful for report context or spatial visualisation.

Important columns in `squirrel.csv`:

- Target-related: `Approaches`, `Indifferent`, `Runs from`.
- Behaviour: `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
- Vocalisation/signals: `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
- Time/location: `Shift`, `Date`, `Hectare`, `X`, `Y`, `Location`, `Above Ground Sighter Measurement`.
- Other possible context: `Age`, `Primary Fur Color`, `Highlight Fur Color`.

Important columns in `hectare.csv`:

- Join key: `Hectare`, `Shift`, `Date`.
- Context features: `Sighter Observed Weather Data`, `Litter`, `Other Animal Sightings`, `Hectare Conditions`, `Number of sighters`, `Number of Squirrels`, `Total Time of Sighting`.

## Prediction Target

Use a clean binary target for the main model:

- `approach`: `Approaches == true`, `Runs from == false`, and `Indifferent == false`.
- `avoid`: `Runs from == true`, `Approaches == false`, and `Indifferent == false`.
- Exclude rows with no recorded interaction, `Indifferent == true`, or multiple conflicting interaction flags from the primary binary model.

Current target facts from `squirrel.csv`:

- Exact `approach`: 143 rows.
- Exact `avoid`: 630 rows.
- Exact `indifferent`: 1,407 rows.
- No recorded interaction: 780 rows.
- Multiple interaction flags: 63 rows.

This binary target is imbalanced, so evaluation must not rely on accuracy alone. A useful sensitivity analysis is a 3-class model over exact `approach`, `indifferent`, and `avoid` rows if time permits, but the main result should answer the stated binary research question.

## Feature Rules

Never use these as predictors in the main models:

- `Approaches`, `Indifferent`, `Runs from`: these define the target.
- `Other Interactions`: likely leaks target-like information through free text.
- `Unique Squirrel ID`, `Lat/Long`, and `Combination of Primary and Highlight Color`: ID/duplicate/composite fields that add little interpretable value.
- `Hectare Squirrel Number`: observation ordering within hectare, not a meaningful behavioural cause.
- Raw `Hectare` ID: high-cardinality + within-hectare leakage on a random split (Phase 0).
- Raw `Date` and `day_of_week`: only `is_weekend` is retained (Phase 2).
- Raw `Location`, raw `Above Ground Sighter Measurement`, raw `Sighter Observed Weather Data`, raw `Other Animal Sightings`, raw `Highlight Fur Color`: replaced by engineered features (Phase 2).
- High-null free text: `Color notes`, `Other Activities`, `Specific Location`.
- `activity_count`, `is_active`, `vocalisation_count`, `tail_signal_count`: summed counts are perfect linear combinations of the underlying flags; flags-only chosen (Phase 2).

Final predictor groups (Phase 2 decisions; see [PROJECT_PHASES.md#phase-2-predictors](PROJECT_PHASES.md#phase-2-predictors) for full rationale):

- Behaviour flags (kept individually, no summed counts): `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
- Signal flags (kept individually, no summed counts): `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
- Time: `Shift`, `is_weekend` (raw `Date` and `day_of_week` excluded).
- Spatial: `X`, `Y` (standardised, train-only fit), `is_above_ground`, `above_ground_numeric`, `location_missing`. Raw `Hectare`, raw `Location`, raw `Above Ground Sighter Measurement` excluded.
- Squirrel characteristics: `Age` (treat `"?"` as missing), `Primary Fur Color`, plus per-colour highlight flags `highlight_gray`, `highlight_white`, `highlight_cinnamon`, `highlight_black`, `highlight_missing` (raw `Highlight Fur Color` excluded).
- Hectare context after joining on `Hectare`, `Shift`, `Date`: `Litter` (encode missing as `"Unknown"`), `Hectare Conditions` (fold `"Medium"` typo into `"Moderate"`), `Number of sighters`, `Number of Squirrels`, `Total Time of Sighting`, parsed `temperature_f` and `sky_condition` from `Sighter Observed Weather Data`, animal-keyword flags from `Other Animal Sightings` (final keyword set TBD by the preprocessing teammate; `animals_humans_present` is now included, and `animals_data_missing` covers nulls), and `squirrel_density_proxy`.

Engineered feature definitions:

- `is_above_ground`: `True` if `Location == "Above Ground"`, else `False`.
- `above_ground_numeric`: numeric height parsed from `Above Ground Sighter Measurement` when `Location == "Above Ground"`; `0` when `Location == "Ground Plane"`; median-impute (train-only) when missing.
- `location_missing`: `True` if `Location` is null, else `False`.
- `is_weekend`: `True` if the parsed date is Saturday/Sunday, else `False`. Mechanism (weekend visitor density → behaviour shift) is inferred, not stated in the spec — flag as an assumption in the report.
- `highlight_gray` / `highlight_white` / `highlight_cinnamon` / `highlight_black`: `True` if the colour name appears in `Highlight Fur Color`, else `False`. `highlight_missing`: `True` if `Highlight Fur Color` is null, else `False`. The four colours are the closed vocabulary documented in the data dictionary.
- Animal-keyword flags from `Other Animal Sightings`: case-insensitive substring match per keyword; `True` if the keyword appears, else `False`. `animals_humans_present` is included (re-added Phase 2 revisit). `animals_data_missing`: `True` if `Other Animal Sightings` is null, else `False`. Final keyword set (beyond `human`) is delegated to the preprocessing teammate, who will inspect the free-text vocabulary across the full dataset and decide which species to flag.
- `temperature_f`: numeric temperature parsed from `Sighter Observed Weather Data` (regex `\d+`); median-impute (train-only) for nulls and parse failures.
- `sky_condition`: coarse keyword match on `Sighter Observed Weather Data` (`clear`, `overcast`, `cloudy`, `rain`); missing → `"Unknown"`; one-hot encode.
- `squirrel_density_proxy`: `Number of Squirrels` / `Total Time of Sighting` (squirrels per minute); median-impute (train-only) where undefined.

## Preprocessing Expectations

Preprocessing must be justified in relation to the research question, not presented as generic cleaning.

Minimum expected steps:

- Parse booleans from string values `"true"` and `"false"`.
- Parse dates from `MMDDYYYY`.
- Convert numeric fields from strings, handling missing or malformed values.
- Treat `Age == "?"` as missing.
- Impute missing categorical values as `"Unknown"` where the missingness itself may be meaningful.
- Impute numeric values using a train-only median imputer inside a scikit-learn pipeline.
- One-hot encode categorical variables inside a pipeline.
- Standardise numeric variables for models that need scaling, such as logistic regression or k-nearest neighbours.
- Merge `squirrel.csv` with `hectare.csv` using `Hectare`, `Shift`, and `Date`; document that one squirrel join key currently has no matching hectare row.
- Build `session_id = Hectare + "_" + Shift + "_" + Date` during preprocessing. Used as the `groups` argument to `StratifiedGroupKFold`. **Not** a predictor.

Avoid preprocessing leakage: fit imputers, encoders, scalers, and feature selection only on training data.

**Task allocation between teammates:**

- **Data preprocessing teammate:** deterministic rewrites of raw data — boolean/date parsing, target relabelling (A+I → approach, I+R → avoid) and conflict-row drop (A+R, A+I+R), free-text parsing into keyword flags, joining squirrel ↔ hectare, type fixes (`Age == "?"` → NaN, parsing `Above Ground Sighter Measurement`), folding `Hectare Conditions` `"Medium"` → `"Moderate"`, building `session_id`. Output: a single clean modelling table.
- **Modelling teammate:** the sklearn `Pipeline` / `ColumnTransformer` — imputers (`SimpleImputer(strategy="median")` numeric, `SimpleImputer(strategy="constant", fill_value="Unknown")` categorical), `OneHotEncoder(handle_unknown="ignore")`, and `StandardScaler` for LR/KNN (not for tree models). Rule of thumb: anything where "fit on train, transform test" matters lives in modelling code.

## Supervised Learning Plan

The assignment requires at least two supervised learning models and a meaningful comparison. Final model set fixed in Phase 4: **Logistic Regression + Random Forest**, plus a `most_frequent` dummy baseline.

Model set:

1. **Baseline: `DummyClassifier(strategy="most_frequent")`.** Predicts the larger class (`avoid`) for every row. Deterministic; recall on `approach` = 0; accuracy ≈ 81%. Reference floor only — not one of the two reported models. `strategy="stratified"` rejected (adds noise without changing the accuracy floor).
2. **Logistic Regression.** Linear, interpretable via signed standardised coefficients — directly answers the "which features are most influential" half of the research question. `class_weight="balanced"` for imbalance.
3. **Random Forest.** Nonlinear, captures behaviour × location × context interactions, supports both impurity-based and permutation feature importance. `class_weight="balanced"`. Chosen over single decision tree (lower variance, more reliable importances), SVC-RBF (interpretation-hostile, expensive), and KNN (no `class_weight`, weak in high-D OHE space).

Optional discussion exhibit: a small `DecisionTreeClassifier(max_depth=3)` may be fitted *post hoc* purely as a visual rule diagram for the discussion section — not an evaluated model.

Evaluation setup:

- **Split: 80/20, stratified by class AND grouped by `session_id`** (not row-level random). Each (Hectare, Shift, Date) session contributes multiple squirrel rows that share identical hectare-context features; a row-level random split would leak session context across train/test. Use `StratifiedGroupKFold` with `groups=session_id`, `random_state=42`. Expected sizes: ~643 train / ~161 test, with ~32 approach in test.
- **Inner CV: 5-fold `StratifiedGroupKFold`** on the 80% training portion, inside `GridSearchCV`, for hyperparameter selection. **Scorer: `average_precision`** (threshold-independent ranking quality, minority-class-sensitive). Rejected: F1 (needs a threshold, contaminates the later threshold sweep at default 0.5); Brier (measures calibration not ranking — we only need ranking quality so the threshold sweep can find a good cut). GridSearchCV averages fold scores per candidate and refits the winner on the full 80% train.
- **Threshold tuning** (real models only, not dummy): after hyperparameter selection, use `cross_val_predict(method="predict_proba", cv=StratifiedGroupKFold(5))` on the training set to obtain out-of-fold probabilities, sweep thresholds (e.g. 0.05–0.95 in 0.01 steps), pick the one maximising macro-F1, then apply once to the test set.
- **Confidence intervals: bootstrap-resample the test set** (~1,000 resamples with replacement; report 2.5th/97.5th percentiles) for each point-estimate metric. **Use the same bootstrap indices across all metrics** so CIs are paired and directly comparable. Small test set (~32 approach rows) is acknowledged as a report limitation.
- Use a fixed random seed (`random_state=42`) throughout for reproducibility.
- **Class imbalance**: use `class_weight="balanced"` on LR and RF. **Do not** use SMOTE or other resampling — most predictors are sparse booleans / OHE'd categoricals, where synthetic interpolation injects noise. `class_weight` and stratification are complementary: stratification controls fold composition, `class_weight` controls training behaviour. Both required.
- See **Metrics and Evaluation** below for the full reporting plan (Phase 5 decisions).

Hyperparameter grids (kept small — ~35 total fits across both models):

- **Logistic Regression** — tuned: `C ∈ {0.01, 0.1, 1, 10}`. Fixed: `penalty="l1"` (sparser, easier interpretation), `solver="liblinear"` (l1-compatible), `max_iter=2000`, `class_weight="balanced"`, `random_state=42`. 4 × 5 = 20 fits.
- **Random Forest** — tuned: `max_depth ∈ {None, 10, 20}`. Fixed: `n_estimators=500` (monotonic; gains plateau on 643 rows), `max_features="sqrt"`, `min_samples_leaf=1` (forcing larger leaves on small imbalanced data risks erasing the minority signal after bagging), `class_weight="balanced"`, `random_state=42`, `n_jobs=-1`. 3 × 5 = 15 fits.

Pipeline scaling rule:

- **LR pipeline must standardise all numeric features** (`StandardScaler` in the `ColumnTransformer`, fit on train only). Required so coefficient magnitudes are comparable for the feature-influence analysis — without standardisation a coefficient reflects scale (`temperature_f` ~30–90 vs `Number of Squirrels` ~1–15 vs 0/1 flags), not influence. After standardisation, β = "log-odds shift in `approach` per 1-SD change in this feature."
- **RF pipeline does NOT include `StandardScaler`** — tree splits are scale-invariant, importances unaffected. RF preprocessing = imputers + OHE only.

## Metrics and Evaluation

Phase 5 decisions. All metrics are computed on the held-out 20% test set unless stated otherwise.

**Primary metric:** macro-F1. Matches the threshold-tuning objective from Phase 3, weights both classes equally under the 82/18 imbalance, interpretable to non-ML readers.

**Secondary test-set metrics:** balanced accuracy, per-class precision and recall on `approach`, PR-AUC (average precision), confusion matrix. Accuracy reported only as the dummy-baseline floor (~81%).

**Not reported:** ROC-AUC (inflated under 82/18 imbalance), weighted-F1 (collapses to majority class), accuracy as a headline.

**Uncertainty (test set):** 95% bootstrap CIs (~1,000 resamples, 2.5/97.5 percentiles) on macro-F1, balanced accuracy, per-class precision/recall on `approach`, and PR-AUC. **Same bootstrap indices reused across all metrics** for paired comparability. No CI on confusion-matrix counts. No inner-CV std reported. Wide CIs (~32 `approach` rows in test) flagged as a report limitation.

**Generalisation check:** report training-set point estimates of macro-F1, balanced accuracy, and PR-AUC alongside test results. Train inside or near test bootstrap CI = healthy generalisation; train far above test CI upper bound = overfitting gap to discuss. **No bootstrap CI on training metrics** — resubstitution scores measure uncertainty on data the model has already seen.

**Hyperparameter selection evidence:** include a small grid-search table from `GridSearchCV.cv_results_` showing mean inner-CV `average_precision` for all 7 candidates (4 LR `C` values + 3 RF `max_depth` values). Methods or appendix placement, no std. This justifies the chosen hyperparameters; the inner-CV-vs-test PR-AUC comparison is **not** reported (generalisation argument rests on train-vs-test alone).

**Confusion matrix interpretation:** `approach` as positive class. Discuss FN (true approach predicted as avoid — model misses an approacher, low cost) vs FP (true avoid predicted as approach — model overclaims, more misleading for the research question) qualitatively. No numerical cost weights.

**Runtime / storage:** one-line methodology mention only ("LR and RF both train in <1 minute on a laptop; runtime is not a differentiator at this scale"). Not a results-section topic.

## Feature Influence

The research question explicitly asks which features are most influential, so feature interpretation is mandatory. Three complementary methods, all aggregating OHE dummies back to parent features for cross-method comparison:

- **LR standardised coefficients** (primary for LR): grouped from OHE back to readable feature names. Requires `StandardScaler` on numeric inputs (already in the LR pipeline) so magnitudes are comparable. β interpreted as "log-odds shift in `approach` per 1-SD change."
- **RF impurity-based importance** (secondary for RF, structural view): free during training, stable across 500 trees, but biased toward high-cardinality / continuous features. Already on a comparable scale (sums to 1).
- **RF permutation importance on the held-out test set** (primary for RF, predictive view): ~30 repeats, model-agnostic, no cardinality bias, directly tied to held-out predictive performance. Understates correlated features (each shuffled alone leaves the other intact).

Reporting plan: present LR coefficients and RF permutation importance side-by-side as the primary cross-model comparison. RF impurity importance kept as a secondary check (final decision on whether it appears in the main report or appendix is deferred to writing time). Tree splits and permutation are both scale-invariant, so the RF pipeline does not include `StandardScaler`.

Interpret feature influence cautiously:

- Use association language, not causal language.
- Compare whether the models agree on influential features. Convergence between LR coefficients and RF permutation importance is a strong, defensible result; divergence triggers a discussion of cardinality bias (impurity) or feature correlation (permutation).
- Mention correlated features and possible confounding, such as location, human presence, food availability, and observer effects.

## Visualisation Checklist

Figures should directly support the research question and be labelled clearly.

Useful visuals for the modelling section:

- Target class distribution.
- Confusion matrices for each main model.
- Bar chart of top feature importances or coefficients.
- Optional spatial scatter/map of approach versus avoid observations.
- Optional metric comparison table across models.

Do not include screenshots of code in the report.

## Rubric-Driven Priorities

The rubric rewards depth, justification, and interpretation over many techniques. Future work should prioritise:

- Clear target definition, split strategy, metrics, and model justification.
- Correct implementation using pipelines to prevent leakage.
- Model comparison that goes beyond scores.
- Explanation of why models differ and what that means for the research question.
- Specific limitations: class imbalance, excluded indifferent/no-interaction rows, observer bias, missingness, non-causal inference, and limited time period.
- Clear visualisations with captions and report references.

## Suggested Repository Structure

Keep raw data unchanged.

Suggested files:

- `code.ipynb`: final reproducible notebook for submission.
- `README.txt`: concise run instructions and expected outputs.
- `outputs/figures/`: generated report figures.
- `outputs/tables/`: generated metric tables or CSV summaries.
- `AI_AGENT_GUIDE.md`: this project brief.
- `PROJECT_PHASES.md`: staged clarification tracker and current open questions.

If creating helper scripts, keep them small and make sure the notebook remains the main reproducible artefact.

## AI Agent Operating Instructions

When helping with this project:

- Read this file before changing code.
- Keep work tightly aligned with the research question.
- Do not generate final report prose. Provide code, result summaries, outlines, or bullet notes that the student can write from.
- Prefer reproducible notebook code using standard Python data science libraries.
- Do not modify raw data files.
- Preserve a fixed random seed in modelling code.
- Use Australian English in report-facing notes.
- Whenever reporting results, include enough context for the student to defend the method in the oral assessment.
- If a modelling result is weak, explain the likely reason instead of hiding it.
- Keep all claims evidence-based and avoid causal wording unless the analysis genuinely supports causality.

## Report Placement Notes

For the modelling workstream:

- Methodology: target construction, feature groups, preprocessing, split strategy, model choices, hyperparameters, evaluation metrics, alternatives considered.
- Results Exploration and Analysis: class distribution, model metric table, confusion matrices, feature importance plots.
- Discussion and Interpretation: what the best model suggests about approach/avoid behaviour, why features may matter, model trade-offs, and what errors reveal.
- Limitations and Improvement Opportunities: imbalance, excluded classes, missing/contextual features, observational bias, non-causal associations, and possible future multiclass or spatial models.

Dataset citation required in the final report:

City of New York. (2019). 2018 Central Park Squirrel Census - Squirrel Data. NYC Open Data. https://data.cityofnewyork.us/Environment/2018-Central-Park-Squirrel-Census-Squirrel-Data/vfnx-vebw

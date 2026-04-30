# AI Agent Guide: COMP20008 Assignment 2

This file is the working brief for AI-assisted coding and analysis in this project. Future AI agents should read this before making changes.

For staged clarification work, use the phase tracker:

- [Phase 0: Understand The Datasets](PROJECT_PHASES.md#phase-0-understand-the-datasets) ✅ complete
- [Phase 1: Research Question Clarification](PROJECT_PHASES.md#phase-1-research-question-clarification) ✅ complete
- [Phase 2: Predictors](PROJECT_PHASES.md#phase-2-predictors) ✅ complete
- [Phase 3: Train/Test Split](PROJECT_PHASES.md#phase-3-traintest-split) ✅ complete
- [Phase 4: Prediction Model Selection](PROJECT_PHASES.md#phase-4-prediction-model-selection)
- [Phase 5: Metrics Selection](PROJECT_PHASES.md#phase-5-metrics-selection)

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
- `animals_humans_present`: every observation was made by a human sighter, so the flag is near-constant and risks being read as causal (Phase 2).

Final predictor groups (Phase 2 decisions; see [PROJECT_PHASES.md#phase-2-predictors](PROJECT_PHASES.md#phase-2-predictors) for full rationale):

- Behaviour flags (kept individually, no summed counts): `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
- Signal flags (kept individually, no summed counts): `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
- Time: `Shift`, `is_weekend` (raw `Date` and `day_of_week` excluded).
- Spatial: `X`, `Y` (standardised, train-only fit), `is_above_ground`, `above_ground_numeric`, `location_missing`. Raw `Hectare`, raw `Location`, raw `Above Ground Sighter Measurement` excluded.
- Squirrel characteristics: `Age` (treat `"?"` as missing), `Primary Fur Color`, plus per-colour highlight flags `highlight_white`, `highlight_cinnamon`, `highlight_black`, `highlight_missing` (raw `Highlight Fur Color` excluded).
- Hectare context after joining on `Hectare`, `Shift`, `Date`: `Litter` (encode missing as `"Unknown"`), `Hectare Conditions` (fold `"Medium"` typo into `"Moderate"`), `Number of sighters`, `Number of Squirrels`, `Total Time of Sighting`, parsed `temperature_f` and `sky_condition` from `Sighter Observed Weather Data`, animal-keyword flags `animals_dogs_present`, `animals_cats_present`, `animals_hawks_present`, `animals_pigeons_present`, `animals_data_missing`, and `squirrel_density_proxy`. `animals_humans_present` is excluded as a collection-bias artefact.

Engineered feature definitions:

- `is_above_ground`: `True` if `Location == "Above Ground"`, else `False`.
- `above_ground_numeric`: numeric height parsed from `Above Ground Sighter Measurement` when `Location == "Above Ground"`; `0` when `Location == "Ground Plane"`; median-impute (train-only) when missing.
- `location_missing`: `True` if `Location` is null, else `False`.
- `is_weekend`: `True` if the parsed date is Saturday/Sunday, else `False`. Mechanism (weekend visitor density → behaviour shift) is inferred, not stated in the spec — flag as an assumption in the report.
- `highlight_white` / `highlight_cinnamon` / `highlight_black`: `True` if the colour name appears in `Highlight Fur Color`, else `False`. `highlight_missing`: `True` if `Highlight Fur Color` is null, else `False`.
- `animals_dogs_present` / `animals_cats_present` / `animals_hawks_present` / `animals_pigeons_present`: `True` if the keyword (case-insensitive) appears in `Other Animal Sightings`, else `False`. `animals_data_missing`: `True` if `Other Animal Sightings` is null, else `False`.
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

The assignment requires at least two supervised learning models and a meaningful comparison.

Recommended model set:

1. Dummy baseline: use `DummyClassifier(strategy="most_frequent")` or `strategy="stratified")` only as a reference point.
2. Logistic regression: interpretable linear model with `class_weight="balanced"`. Use coefficients to discuss feature influence.
3. Decision tree or random forest: non-linear model that can capture interactions between behaviour, location, and context. Use class weighting where available and constrain complexity to reduce overfitting.

At least two real supervised models must be reported. The dummy model is a baseline, not one of the two main models.

Recommended evaluation setup:

- **Split: 80/20, stratified by class AND grouped by `session_id`** (not row-level random). Each (Hectare, Shift, Date) session contributes multiple squirrel rows that share identical hectare-context features; a row-level random split would leak session context across train/test. Use `StratifiedGroupKFold` with `groups=session_id`, `random_state=42`. Expected sizes: ~643 train / ~161 test, with ~32 approach in test.
- **Inner CV: 5-fold `StratifiedGroupKFold`** on the 80% training portion, inside `GridSearchCV`, for hyperparameter selection. Inner-CV scoring uses a `predict_proba`-based metric (e.g. `average_precision` or `roc_auc`) so hyperparameter choice is threshold-independent.
- **Threshold tuning** (real models only, not dummy): after hyperparameter selection, use `cross_val_predict(method="predict_proba", cv=StratifiedGroupKFold(5))` on the training set to obtain out-of-fold probabilities, sweep thresholds (e.g. 0.05–0.95 in 0.01 steps), pick the one maximising macro-F1, then apply once to the test set.
- **Confidence intervals: bootstrap-resample the test set** (~1,000 resamples with replacement; report 2.5th/97.5th percentiles) for each metric. Small test set (~32 approach rows) is acknowledged as a report limitation.
- Use a fixed random seed (`random_state=42`) throughout for reproducibility.
- **Class imbalance**: use `class_weight="balanced"` on LR / decision tree / random forest. **Do not** use SMOTE or other resampling — most predictors are sparse booleans / OHE'd categoricals, where synthetic interpolation injects noise. `class_weight` and stratification are complementary: stratification controls fold composition, `class_weight` controls training behaviour. Both required.
- Primary metrics: macro F1, balanced accuracy, precision, recall, and confusion matrix.
- If using probability outputs, include ROC-AUC or PR-AUC, but explain what they mean in the imbalanced setting.
- Discuss false positives and false negatives in context: predicting approach incorrectly is different from predicting avoid incorrectly.

Hyperparameters to justify:

- Logistic regression: regularisation strength `C`, solver, maximum iterations, class weighting.
- Decision tree: `max_depth`, `min_samples_leaf`, class weighting.
- Random forest: number of trees, `max_depth`, `min_samples_leaf`, class weighting, random seed.

## Feature Influence

The research question explicitly asks which features are most influential, so feature interpretation is mandatory.

Use at least two complementary methods where possible:

- Logistic regression coefficients after preprocessing, grouped back to readable feature names.
- Tree-based feature importance for decision tree or random forest.
- Permutation importance on the held-out test set for model-agnostic comparison.

Interpret feature influence cautiously:

- Use association language, not causal language.
- Compare whether the models agree on influential features.
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

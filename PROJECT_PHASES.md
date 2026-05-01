# Project Phase Clarification Plan

This file tracks the staged clarification process for the prediction-model workstream. Use it alongside `AI_AGENT_GUIDE.md`.

Working mode:

- Work through one phase at a time with an AI agent.
- Keep open questions in this file until they are resolved.
- When a decision becomes stable, copy the final decision back into `AI_AGENT_GUIDE.md` if it affects project rules, modelling design, or report structure.
- Keep discussion focused on the current phase unless a later phase is blocking the decision.

This is a good mode for this project because the assignment rewards depth and justification. Separating phases keeps each discussion narrow, while `AI_AGENT_GUIDE.md` remains the concise source of agreed project decisions.

## Phase 0: Understand The Datasets

Goal: understand what each raw file represents, what the rows mean, and which joins are valid before modelling.

### Decisions

- **Q1 (closed): Why does `hectare.csv` contain a `Date` feature?**
  - The row unit is a *sighting session* keyed by (Hectare, Shift, Date). The census ran across 11 distinct dates (6–20 Oct 2018), so Date is part of the natural session key and is needed to align hectare context (weather, litter, conditions) with squirrel observations from the same session. Join key stays (Hectare, Shift, Date).

- **Q2 (closed): Is each hectare-shift visited only once?**
  - Yes. All 700 (Hectare, Shift) pairs are unique; 349 hectares have both AM and PM, 2 have only one shift. 86 hectare sessions have `Number of Squirrels == 0` (env data only, no squirrels). Use a **left join from squirrel → hectare** so behaviour rows drive the modelling table.
  - Note: 1 squirrel row has no matching hectare row (09I, AM, 10142018). Drop-vs-impute decision is delegated to the data preprocessing teammate.

- **Q3 (closed): Should raw `Hectare` ID be a predictor?**
  - **No.** Reasons: (a) ~339 hectares for ~2,200 modelable rows → high-cardinality OHE, overfitting risk; (b) random split causes within-hectare leakage because each hectare appears in 1–2 sessions and would land in both train and test; (c) using an opaque ID hides the spatial drivers the research question is asking about.

- **Q4 (closed): What is the difference between `Hectare Squirrel Number` (squirrel.csv) and `Number of Squirrels` (hectare.csv)?**
  - `Hectare Squirrel Number` is a per-row sequential index of the squirrel within its session (1..23). `Number of Squirrels` is a per-session count. Empirically, `max(Hectare Squirrel Number)` per session equals `Number of Squirrels` in 610/613 matched sessions.
  - Decision: exclude `Hectare Squirrel Number` (ordering index, not behaviour). Build a session-level density predictor from `Number of Squirrels / Total Time of Sighting` (in minutes, mean ≈ 24.9, range 1–70, 20 nulls / 700 rows).

### Data observations

Catalogued so they are not surprises in later phases:

- **Missingness in `squirrel.csv`** (n=3,023):
  - High-null free-text fields: `Color notes` (2,841), `Other Interactions` (2,783), `Other Activities` (2,586), `Specific Location` (2,547) — excluded from predictors.
  - Modelling-relevant nulls: `Highlight Fur Color` (1,086), `Age` (121, plus 4 literal `"?"` to treat as missing), `Above Ground Sighter Measurement` (114), `Location` (64), `Primary Fur Color` (55).
- **`Above Ground Sighter Measurement` is mixed-type**: 2,116 rows hold the literal string `"FALSE"` (= ground plane), the rest are numeric heights. Redundant with `Location` (`FALSE` ↔ `Ground Plane`); the 64 missing `Location` rows align with 114 missing height rows.
- **`Highlight Fur Color` has 10 multi-valued combinations** (e.g., `"Cinnamon, White"`, `"Black, Cinnamon, White"`). Direct one-hot is messy.
- **Date span is short and uneven**: 11 dates, 6–20 Oct 2018. Per-day squirrel counts range from 67 (10/20) to 434 (10/13) — heavy bias toward weekend dates. Calendar feature only; not enough span for a temporal split.
- **Shift imbalance in `squirrel.csv`**: AM 1,347 vs PM 1,676 even though `hectare.csv` is balanced 350/350 — PM sessions yielded more squirrels per session.
- **`hectare.csv` missingness**: `Litter` (381 — over half), `Hectare Conditions` (40), `Total Time of Sighting` (20), `Sighter Observed Weather Data` (19), `Other Animal Sightings` (32). Encode missingness explicitly where it may be informative.
- **`Hectare Conditions`** has a typo level `"Medium"` (n=1) — fold into `"Moderate"` during cleaning.
- **`Sighter Observed Weather Data` is free text**, 498 unique strings (e.g., `"57º F, overcast"`).
- **`Other Animal Sightings`**: comma-separated free text (`"Humans, Dogs, Pigeons, Sparrows"` etc.).
- **5 duplicate `Unique Squirrel ID`s** (3,023 rows but 3,018 unique IDs). Preprocessing teammate to decide whether to dedupe.
- **Observer effects**: 235 unique sighters, median 2 sessions per sighter, max 19. Not in the predictor set; mention as a limitation.

Decision status: Q1–Q4 closed. Phase 0 complete.

## Phase 1: Research Question Clarification

Goal: decide the exact modelling target and make sure it matches the research question.

Research question:

> Can we predict whether a squirrel will approach or avoid humans based on its behaviour, location, and time of observation, and which features are most influential?

Verified interaction-flag counts in `squirrel.csv` (3,023 rows total):

| Category | Logical rule | Count | % |
| --- | --- | ---: | ---: |
| Exact approach | `A & ~I & ~R` | 143 | 4.7% |
| Exact avoid (runs from) | `~A & ~I & R` | 630 | 20.8% |
| Exact indifferent | `~A & I & ~R` | 1,407 | 46.5% |
| No interaction recorded | `~A & ~I & ~R` | 780 | 25.8% |
| Multi-flag (≥2 of A/I/R) | — | 63 | 2.1% |

Multi-flag breakdown: `A+I` = 15, `A+R` = 16, `I+R` = 28, `A+I+R` = 4.

### Decisions

- **Q1 (closed): Binary `approach` vs `avoid` is the only target. Three-class will not be performed.**
  - Reason: the research question is binary by wording. `indifferent` is a "no decision" outcome, and keeping it as a third class lets a trivial classifier score high accuracy by always predicting indifferent, which masks whether the model has actually learned approach/avoid drivers.

- **Q2 (closed): Relabel resolvable multi-flag rows; drop genuine conflicts.**
  - `A+I` (15 rows) → **approach** (active behaviour wins over neutral).
  - `I+R` (28 rows) → **avoid** (same logic; runs-from is the active behaviour).
  - `A+R` (16 rows) and `A+I+R` (4 rows) → **drop** as genuine conflicts.
  - Net effect: binary pool grows from 773 → **804** rows (158 approach, 658 avoid; imbalance ≈ 1 : 4.16).
  - Owner: data preprocessing teammate applies relabelling and conflict-row removal during target construction, before any feature preprocessing or train/test split.

- **Q3 (closed): Exclude rows with no recorded interaction (780 rows).**
  - Reason: "no interaction recorded" is ambiguous — it could mean the squirrel never encountered a human, or the interaction simply was not logged. Treating it as `approach` or `avoid` is wrong by definition; treating it as `indifferent` would conflate "saw a human and did not react" with "no human was present".

- **Q4 (closed): Data-sufficiency concern resolved by the Phase 3 protocol.**
  - The 804-row binary pool is small but defensible. Resolution: stratified-grouped 80/20 split + 5-fold inner CV, `class_weight="balanced"`, low-complexity models, and bootstrap CIs on test metrics. Small test set (~32 approach rows) flagged as a report limitation.

### Final target rule

- `y = 1` (approach): `Approaches & ~Runs from` after the Q2 relabel resolves `A+I` to approach.
- `y = 0` (avoid): `Runs from & ~Approaches` after the Q2 relabel resolves `I+R` to avoid.
- Drop: exact indifferent (1,407), no interaction recorded (780), `A+R` (16), `A+I+R` (4) — total 2,207 rows excluded; binary modelling pool = **804 rows** (158 approach, 658 avoid).

Decision status: Q1–Q4 closed. Phase 1 complete.

## Phase 2: Predictors

Goal: decide which raw and engineered predictors belong in the model.

`stories.csv` is not used. `squirrel.csv` and `hectare.csv` are joined on `Hectare`, `Shift`, and `Date`.

### Decisions

- **Q1 (closed): Include both fur-colour fields, but simplify `Highlight Fur Color`.**
  - `Primary Fur Color`: keep raw → one-hot encode (3 levels, 55 nulls).
  - `Highlight Fur Color`: split the multi-valued string into per-colour boolean flags `highlight_gray`, `highlight_white`, `highlight_cinnamon`, `highlight_black`, plus `highlight_missing` for the 1,086 nulls. The four colours are the closed vocabulary from the data dictionary.
  - Reason: the research question explicitly asks which features are most influential, so excluding fur on conceptual grounds prevents the model from answering. Per-colour flags handle the multi-valued combinations cleanly.

- **Q2 (closed): `X` and `Y` are kept as standardised continuous predictors; no spatial zoning.**
  - Reason: trees use raw coordinates natively; LR gets a weak linear approximation, which is acceptable. Coarse zones inflate dimensionality on the 804-row pool without clear benefit. Map visualisation is for the report, not the model.

- **Q3 (closed): Drop raw `Location` and raw `Above Ground Sighter Measurement`; keep three engineered features.**
  - `is_above_ground` = `True` if `Location == "Above Ground"`, else `False`.
  - `above_ground_numeric` = parsed numeric height when `Location == "Above Ground"`; `0` when `Location == "Ground Plane"`; median-impute (train-only) when missing.
  - `location_missing` = `True` if `Location` is null, else `False` (covers the 64 null `Location` / 114 null height rows).

- **Q4 (closed): Date is reduced to `is_weekend` only; drop raw date and `day_of_week`.**
  - `is_weekend` = `True` if the parsed date falls on Saturday/Sunday, else `False`.
  - Reason: heavy weekend bias in squirrel counts (10/13 Sat = 434, 10/20 Sat = 67, midweek dates 200–300). Plausible mechanism: more park visitors on weekends → shifted approach/avoid base rates. **Caveat: the visitor-density mechanism is inferred, not stated in the assignment spec** — flag as an assumption in the report. Raw date and `day_of_week` are dropped because 11 unevenly distributed dates over 14 days are too sparse to support more granular calendar features.

- **Q5 (revised): Engineer animal-keyword flags from `Other Animal Sightings`; final keyword set delegated to the preprocessing teammate.**
  - Each flag = `True` if the keyword (case-insensitive) appears in the comma-separated string, else `False`.
  - Confirmed flags: `animals_humans_present` (keyword `"human"`), `animals_dogs_present` (keyword `"dog"`), and `animals_data_missing` for the 32 null rows in `hectare.csv`.
  - **Open**: which additional species keywords to flag. Delegated to the preprocessing teammate, who will inspect the free-text vocabulary across the full dataset and propose the final keyword list before the modelling table is finalised.
  - Reason for re-including `animals_humans_present`: although every sighter is human, the flag captures whether *additional* humans were noted in the area beyond the sighter, which is a plausible behavioural driver. Earlier exclusion was overcautious; we will check post-hoc whether the column is near-constant and revisit if so.

- **Q6 (closed): Parse `Sighter Observed Weather Data` into two engineered features; add density proxy; exclude all summed activity/signal counts.**
  - **Removed**: `activity_count`, `is_active`, `vocalisation_count`, `tail_signal_count` — perfect linear combinations of the underlying flags (Q7 keeps flags-only); collinear with no information gain.
  - `temperature_f` — numeric (regex `\d+` on the temperature substring); median-impute (train-only) for nulls/parse failures.
  - `sky_condition` — coarse keyword match (`clear`, `overcast`, `cloudy`, `rain`) → one-hot encode; missing → `"Unknown"`.
  - `squirrel_density_proxy` = `Number of Squirrels` / `Total Time of Sighting` (squirrels per minute); median-impute (train-only) where undefined.
  - `Specific Location` remains excluded (no parsing attempt).

- **Q7 (closed): Keep individual behaviour and signal flags only — no summed counts.**
  - Behaviour flags retained: `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
  - Signal flags retained: `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
  - Reason: a sum is a perfect linear combination of its component flags; keeping both forms introduces collinearity with no extra information. Flags-only gives interpretable per-behaviour importance.

### Final predictor list

From `squirrel.csv`:

- Behaviour flags (5): `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
- Signal flags (5): `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
- Time (2): `Shift` (OHE), `is_weekend`.
- Spatial (5): `X`, `Y` (standardised), `is_above_ground`, `above_ground_numeric`, `location_missing`.
- Individual (2 raw + 5 engineered): `Age` (OHE; treat `"?"` as missing), `Primary Fur Color` (OHE), `highlight_gray`, `highlight_white`, `highlight_cinnamon`, `highlight_black`, `highlight_missing`.

From `hectare.csv` (joined on `Hectare`, `Shift`, `Date`):

- Hectare context (5): `Litter` (OHE; encode missing as `"Unknown"` — 381 nulls), `Hectare Conditions` (OHE; fold `"Medium"` typo into `"Moderate"`), `Number of sighters`, `Number of Squirrels`, `Total Time of Sighting`.
- Animal flags (engineered): `animals_humans_present`, `animals_dogs_present`, plus additional species flags TBD by the preprocessing teammate after free-text inspection, plus `animals_data_missing` for null rows.
- Weather (2 engineered): `temperature_f`, `sky_condition` (OHE).
- Density (1 engineered): `squirrel_density_proxy`.

### Excluded fields

- **Target leakers**: `Approaches`, `Indifferent`, `Runs from`, `Other Interactions`.
- **ID / duplicate / composite**: `Unique Squirrel ID`, `Lat/Long`, `Combination of Primary and Highlight Color`, `Hectare Squirrel Number`.
- **High-null free text**: `Color notes`, `Other Activities`, `Specific Location`.
- **Spatial leakage**: raw `Hectare` ID (Phase 0).
- **Calendar redundancy**: raw `Date`, `day_of_week` (Q4 — only `is_weekend` retained).
- **Replaced by engineered features**: raw `Location`, raw `Above Ground Sighter Measurement` (Q3), raw `Sighter Observed Weather Data` (Q6), raw `Other Animal Sightings` (Q5), raw `Highlight Fur Color` (Q1).
- **Collinear sums**: `activity_count`, `is_active`, `vocalisation_count`, `tail_signal_count` (Q7).

Decision status: Q1–Q7 closed. Phase 2 complete.

## Phase 3: Train/Test Split

Goal: decide how to split data and prevent leakage.

### Decisions

- **Q1 (closed): Random split, not temporal.**
  - Reason: the 11-day span is too short for a forecasting framing; the research question is about behavioural/contextual drivers, not future prediction. Per-day counts are heavily skewed (10/13: 434 vs 10/20: 67), so a chronological cut would over/under-represent specific weekends in either side.

- **Q2 (closed): Session-grouped splitting to prevent within-session leakage.**
  - Each (Hectare, Shift, Date) session contributes multiple squirrel rows that share identical hectare-context features (litter, weather, density proxy, other-animal flags, total sighting time). A row-level random split would put squirrels from the same session into both train and test, letting the model memorise session context.
  - Mitigation: build `session_id = Hectare + "_" + Shift + "_" + Date` and pass it as the `groups` argument to `StratifiedGroupKFold`. `session_id` is **not** a predictor — splitter input only.
  - Owner: data preprocessing teammate builds `session_id` (deterministic concat; no fitting required).

- **Q3 (closed): Single 80/20 hold-out + inner 5-fold CV.**
  - Steps:
    1. Outer split: one stratified-grouped 80/20 split over the 804-row binary pool, stratified on `y`, grouped by `session_id`, `random_state=42`.
    2. On the 80% training portion: 5-fold `StratifiedGroupKFold` (same stratify + group rules) inside `GridSearchCV` to choose model hyperparameters. Inner-CV scoring uses a `predict_proba`-based metric (e.g. `average_precision` or `roc_auc`) so hyperparameter selection is threshold-independent.
    3. Refit the best pipeline on the full 80% training set.
    4. Threshold tuning (real models only): use `cross_val_predict(method="predict_proba", cv=StratifiedGroupKFold(5))` on the training set, sweep thresholds (0.05–0.95 in 0.01 steps), pick the one maximising macro-F1.
    5. Apply the chosen threshold to the held-out 20% test-set probabilities and compute metrics.
    6. **CI**: bootstrap-resample the test set (~1,000 resamples with replacement; report 2.5th/97.5th percentiles for each metric).
  - Expected sizes: ~643 train / ~161 test rows, with ~32 approach in the test set. Small-test-set caveat flagged as a report limitation; bootstrap CI makes it defensible.
  - Why not nested CV: the assignment expects a clear single train/test split + CV; one held-out test set is easier to defend in the report and oral; outer-fold scores from 161-row folds would be just as noisy in practice.

- **Q4 (closed): Preprocessing fitted only on training data, inside an sklearn `Pipeline` / `ColumnTransformer`.**
  - **Inside the pipeline (fit on training fold only)**: `SimpleImputer(strategy="median")` on numeric, `SimpleImputer(strategy="constant", fill_value="Unknown")` on categoricals where missingness may be informative, `OneHotEncoder(handle_unknown="ignore")` on categoricals, `StandardScaler` on numerics for LR/KNN (not for tree models).
  - **Outside the pipeline (deterministic, fine on full data)**: boolean/date parsing, target relabelling and conflict-row drop, free-text parsing into keyword flags, joining squirrel ↔ hectare, type fixes (`Age == "?"` → NaN, parsing `Above Ground Sighter Measurement`), folding `Hectare Conditions` `"Medium"` → `"Moderate"`, building `session_id`.
  - **Task allocation**:
    - Data preprocessing teammate: everything outside the pipeline plus `session_id` construction.
    - Modelling teammate: the `Pipeline` / `ColumnTransformer`, including imputers, encoders, and `StandardScaler`. Rule of thumb: anything where "fit on train, transform test" matters lives in the modelling code.

- **Q5 (closed): Class imbalance handled by `class_weight="balanced"` + threshold tuning. No resampling.**
  - **`class_weight="balanced"`**: scikit-learn argument on `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`. Re-weights the loss so each class contributes equally regardless of size (with 158 approach / 658 avoid, every approach row counts ~4.16× as much during training). Complementary to stratification: stratification controls *how imbalance is distributed across folds*, `class_weight` controls *how the model treats the imbalance during training*. Both required.
  - **Threshold tuning** (real models only, not a dummy baseline): chosen on training-set out-of-fold probabilities (Q3 step 4), applied once on the test set. Not a `GridSearchCV` hyperparameter — separate post-fit step using `predict_proba` instead of `predict`.
  - **No SMOTE or other resampling**: most predictors are sparse booleans / OHE'd categoricals, where synthetic interpolation injects noise.

Decision status: Q1–Q5 closed. Phase 3 complete.

## Phase 4: Prediction Model Selection

Goal: choose the supervised models and their tuning ranges.

### Decisions

- **Q1+Q2 (closed): Two main supervised models — Logistic Regression and Random Forest — plus a `most_frequent` dummy baseline.**
  - **Baseline: `DummyClassifier(strategy="most_frequent")`.** Predicts `avoid` for every row (the larger class). Deterministic, recall on `approach` = 0, accuracy ≈ 81%. Reference floor only — any reported model must beat 0 minority recall to be worth keeping. `strategy="stratified"` rejected because it adds noise without changing the accuracy floor.
  - **Model 1: Logistic Regression.** Linear, interpretable via signed standardised coefficients — directly answers the "which features are most influential" half of the research question. `class_weight="balanced"` handles the imbalance.
  - **Model 2: Random Forest.** Nonlinear, captures behaviour × location × context interactions, gives both impurity-based and permutation feature importance. Compared against alternatives:
    - **vs single Decision Tree**: RF has lower variance, more reliable importances (single-tree importances flip across reseeds), better predictive performance on 643 train × ~50 OHE'd features. A small `DecisionTreeClassifier(max_depth=3)` may be fitted *post hoc* purely as a visual rule-extraction exhibit for the discussion section, but is not an evaluated model.
    - **vs SVC (RBF kernel)**: SVC is interpretation-hostile (no per-feature signal natively), expensive to tune (`C` × `gamma`), and struggles with high-dimensional sparse OHE inputs.
    - **vs KNN**: KNN suffers in high-dimensional space (distance loses meaning), has no native `class_weight` support, and provides nothing for the feature-influence half of the research question.
  - Two models, not three: the rubric rewards depth over count, and RF subsumes a single tree on performance. A third evaluated model would dilute the analysis.

- **Q3 (closed): Hyperparameter grids — small, justified, ~35 total fits.**
  - **Inner CV scorer: `average_precision`** for both LR and RF. Threshold-independent (uses `predict_proba` directly), minority-class-sensitive, decouples hyperparameter selection from the later threshold-tuning step. Rejected alternatives: F1 (requires a threshold, contaminates the later threshold sweep at default 0.5); Brier (measures calibration not ranking — we only need ranking quality so the threshold sweep can find a good cut).
  - **Logistic Regression grid:**
    - Tuned: `C ∈ {0.01, 0.1, 1, 10}` — 4 candidates.
    - Fixed: `penalty="l1"` (sparser coefficients, automatic feature selection, easier interpretation), `solver="liblinear"` (l1-compatible), `max_iter=2000`, `class_weight="balanced"`, `random_state=42`.
    - 4 candidates × 5 folds = 20 fits.
  - **Random Forest grid:**
    - Tuned: `max_depth ∈ {None, 10, 20}` — 3 candidates. `None` relies on bagging for regularisation; `20` is bounded but generous; `10` is shallower bias/variance trade-off.
    - Fixed: `n_estimators=500` (monotonic — more trees rarely hurt; gains plateau well before this on 643 rows), `max_features="sqrt"` (standard for classification RF), `min_samples_leaf=1` (forcing larger leaves on small imbalanced data risks erasing the minority signal after bagging — `class_weight="balanced"` and `max_depth` are the regularisation knobs), `class_weight="balanced"`, `random_state=42`, `n_jobs=-1`.
    - 3 candidates × 5 folds = 15 fits.
  - **Total: ~35 fits**, very cheap.
  - **GridSearchCV mechanics**: each candidate scored on every fold, fold scores **averaged** per candidate, candidate with the highest mean wins, refit once on the full 80% training set. Not majority-vote across folds.

- **Q4 (closed): Feature influence — three complementary methods.**
  - **LR standardised coefficients (primary for LR)**: requires standardising all numeric features in the LR `ColumnTransformer` so coefficient magnitudes are comparable across features. Without standardisation, magnitudes reflect feature scales, not influence. After standardisation, β = "log-odds shift in `approach` per 1-SD change in this feature." OHE dummies are aggregated back to parent feature (e.g. sum `|β|` across `Primary Fur Color` levels).
  - **RF impurity-based importance (primary for RF, structural view)**: free during training, stable across 500 trees, but biased toward high-cardinality features (continuous + many-level OHE). Already on a comparable scale (sums to 1). RF does **not** require standardisation — tree splits are scale-invariant.
  - **RF permutation importance on the held-out test set (primary for RF, predictive view)**: shuffles each column ~30 times, measures held-out AP drop. Model-agnostic, no cardinality bias, directly tied to predictive performance, but understates correlated features (each shuffled alone leaves the other intact). Already on a comparable scale (units of metric drop). Also scale-invariant.
  - **Reporting plan**: present LR coefficients and RF permutation importance side-by-side as the primary cross-model comparison. RF impurity importance kept as a secondary check (appendix or sensitivity figure) — agreement between impurity and permutation strengthens claims, disagreement triggers a discussion of cardinality bias or feature correlation. Final decision on whether both RF methods appear in the main report is deferred to writing time.

Decision status: Q1–Q4 closed. Phase 4 complete.

## Phase 5: Metrics Selection

Goal: decide how model performance will be evaluated and communicated.

### Decisions

- **Q1 (closed): Which metrics are most suitable for the research question?**
  - **Primary: macro-F1.** Matches the threshold-tuning objective from Phase 3, weights both classes equally under the 82/18 imbalance, and is interpretable to non-ML readers.
  - **Secondary on test set**: balanced accuracy, per-class precision and recall on `approach`, PR-AUC (average precision), and the confusion matrix. Accuracy shown only as the dummy-baseline floor (~81%).
  - **Not used**: ROC-AUC (inflated under 82/18 imbalance because the FPR denominator is large); weighted-F1 (collapses to majority-class performance); accuracy as a headline (dummy already achieves ~81%).

- **Q2 (closed): How should metrics reflect class imbalance?** Removed — already covered by the metric choices in Q1 (macro-F1, balanced accuracy, PR-AUC, per-class breakdown).

- **Q3 (closed): With cross-validation, should we report confidence intervals, standard deviations, or error bars?**
  - **95% bootstrap CIs on the test set** (~1,000 resamples, 2.5/97.5 percentiles) for macro-F1, balanced accuracy, per-class precision/recall on `approach`, and PR-AUC. Use the **same bootstrap indices across all metrics** so CIs are paired and directly comparable.
  - **No CI on the confusion matrix** — report raw counts; per-class precision/recall CIs already capture the relevant uncertainty.
  - **No inner-CV std reported.** 5 fold scores are too thin a basis for a meaningful std, and inner-CV measures training-fold stability rather than generalisation.
  - Acknowledge in the report that with ~32 `approach` rows in test, bootstrap CIs will be wide — this is a stated limitation, not a hidden one.

- **Q4 (closed): Is model training cost worth discussing?** No. ~35 total fits on 643 rows runs in seconds-to-minutes; storage is trivial. One sentence in methodology ("LR and RF both train in <1 minute on a laptop; runtime is not a differentiator at this scale") and nothing more.

- **Q5 (closed): How do we evaluate generalisation of the model?**
  - **Train-vs-test point comparison.** Report training-set point estimates of macro-F1, balanced accuracy, and PR-AUC alongside the test results. If the train point estimate sits inside or near the test bootstrap CI, generalisation looks healthy; if train sits far above the test CI upper bound, treat the gap as an overfitting signal in the discussion.
  - **No bootstrap CI on training metrics.** Resubstitution scores measure uncertainty on data the model has already seen — comparing a "seen-data" CI against a "unseen-data" CI invites overinterpretation. Point estimate vs test CI is the right comparison.
  - **Hyperparameter selection evidence (separate from generalisation):** include a small grid-search results table from `cv_results_` showing inner-CV mean `average_precision` for all 7 candidates (4 LR `C` values + 3 RF `max_depth` values), to justify the chosen hyperparameters. Methods or appendix placement, no std.
  - Inner-CV PR-AUC vs test PR-AUC comparison **not reported** — generalisation argument rests on the train-vs-test point comparison alone.

- **Confusion matrix interpretation plan:** `approach` as the positive class. Discuss FN (true approach predicted as avoid — low real-world cost, model misses an approacher) vs FP (true avoid predicted as approach — more misleading for the research question, model overclaims approachers). Qualitative discussion only; no numerical cost weights.

Decision status: Q1–Q5 closed. Phase 5 complete.

## Phase Decision Log

- **Phase 0**: complete. Join key = (Hectare, Shift, Date) via left join from squirrel → hectare. Raw `Hectare` ID excluded as predictor; use `X`, `Y` and hectare-level context. `Hectare Squirrel Number` excluded; build a session-level density predictor = `Number of Squirrels / Total Time of Sighting`.
- **Phase 1**: complete. Target = binary `approach` (158) vs `avoid` (658), 804 rows after relabelling `A+I → approach` and `I+R → avoid`. Three-class out of scope. Excluded: exact indifferent (1,407), no-interaction (780), genuine conflicts `A+R` + `A+I+R` (20).
- **Phase 2**: complete. Final predictor set spans behaviour flags (5), signal flags (5), time (`Shift`, `is_weekend`), spatial (`X`, `Y` standardised, `is_above_ground`, `above_ground_numeric`, `location_missing`), individual (`Age`, `Primary Fur Color`, five `highlight_*` flags for gray/white/cinnamon/black/missing), and hectare context (`Litter`, `Hectare Conditions`, animal-keyword flags including `animals_humans_present` and `animals_dogs_present` plus additional species TBD by the preprocessing teammate + `animals_data_missing`, `Number of sighters`, `Number of Squirrels`, `Total Time of Sighting`, `temperature_f`, `sky_condition`, `squirrel_density_proxy`). Excluded: target leakers, ID/composite fields, raw `Hectare`, raw `Date` and `day_of_week`, raw `Location`/`Above Ground`/`Weather`/`Animal Sightings`/`Highlight` (replaced by engineered features), all summed activity/signal counts (flags-only), and `Specific Location`.
- **Phase 3**: complete. Random (not temporal) split. Single 80/20 stratified-grouped hold-out (`random_state=42`), grouped by `session_id`. 5-fold `StratifiedGroupKFold` `GridSearchCV` on the 80% training portion using a `predict_proba`-based metric. Threshold tuned on training out-of-fold probabilities (real models only) to maximise macro-F1, applied once to the test set. Bootstrap CIs (~1,000 resamples) on test metrics. Preprocessing fitted only on training data inside an sklearn `Pipeline` / `ColumnTransformer` (median/constant imputers, OHE, `StandardScaler` for LR/KNN). Class imbalance handled via `class_weight="balanced"` + threshold tuning; no SMOTE.
- **Phase 4**: complete. Two main models — Logistic Regression and Random Forest — plus a `most_frequent` dummy baseline. LR: l1 penalty (fixed), liblinear solver, `class_weight="balanced"`, grid `C ∈ {0.01, 0.1, 1, 10}`, standardised numeric inputs (required for comparable coefficients). RF: `n_estimators=500`, `max_features="sqrt"`, `min_samples_leaf=1`, `class_weight="balanced"` all fixed, grid `max_depth ∈ {None, 10, 20}`, no scaling. Inner-CV scorer `average_precision` (ranking quality, threshold-free, minority-aware). ~35 total fits across both models. Feature influence via LR standardised coefficients + RF permutation importance (primary) with RF impurity importance as a secondary cross-check.
- **Phase 5**: complete. Primary metric = macro-F1 (matches threshold-tuning objective). Secondary test-set metrics = balanced accuracy, per-class precision/recall on `approach`, PR-AUC, confusion matrix. Accuracy shown only as the dummy floor. ROC-AUC and weighted-F1 not reported. Uncertainty = 95% bootstrap CIs (~1,000 resamples, paired indices across metrics) on all point-estimate metrics; no CI on confusion-matrix counts; no inner-CV std. Generalisation evaluated via train-set point estimates of macro-F1 / balanced accuracy / PR-AUC compared against test bootstrap CIs (no train CI). Hyperparameter selection justified via a `cv_results_` grid-search table (mean inner-CV `average_precision` for all 7 candidates), methods or appendix. Confusion matrix interpreted with `approach` as positive class, qualitative FN-vs-FP discussion. Runtime/storage = one-line methodology mention only.

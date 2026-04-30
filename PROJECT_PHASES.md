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

Current questions:

- Q1 (closed): In `hectare.csv`, why does it contain a `Date` feature?
  - Resolution: the row unit in `hectare.csv` is a *sighting session* keyed by (Hectare, Shift, Date). The census ran across 11 distinct dates (6–20 Oct 2018), so Date is part of the natural session key and is needed to align hectare context (weather, litter, conditions) with squirrel observations from the same session. Join key stays (Hectare, Shift, Date).
- Q2 (closed): For a single hectare-shift, is that hectare visited only once?
  - Resolution: yes. All 700 (Hectare, Shift) pairs are unique; 349 hectares have both AM and PM, 2 have only one shift. 86 hectare sessions have `Number of Squirrels == 0` (env data only, no squirrels). Use a **left join from squirrel → hectare** so behaviour rows drive the modelling table.
  - Note: 1 squirrel row has no matching hectare row (09I, AM, 10142018). Drop-vs-impute decision is delegated to the teammate owning data preprocessing.
- Q3 (closed): If a hectare-shift is observed once, should `Hectare` itself be a predictor?
  - Resolution: **do NOT use raw `Hectare` ID as a predictor.** Reasons: (a) ~339 hectares for ~2,200 modelable rows → high-cardinality OHE, overfitting risk; (b) random split causes within-hectare leakage because each hectare appears in 1–2 sessions and would land in both train and test; (c) using an opaque ID hides the spatial drivers the research question is asking about.
  - Use instead: hectare-level **context** features (litter, hectare conditions, number of sighters, number of squirrels, total sighting time, other animal sightings) plus continuous spatial coordinates `X`, `Y`. Coarser spatial zones are optional.
- Q4 (closed): What is the difference between `Hectare Squirrel Number` in `squirrel.csv` and `Number of Squirrels` in `hectare.csv`?
  - Resolution: `Hectare Squirrel Number` (squirrel.csv) is a per-row sequential index of the squirrel within its session (1..23). `Number of Squirrels` (hectare.csv) is a per-session count of squirrels seen in that session. Empirically, `max(Hectare Squirrel Number)` per session equals `Number of Squirrels` in 610/613 matched sessions.
  - Decision: exclude `Hectare Squirrel Number` (ordering index, not behaviour). **Use a session-level density predictor** built from `Number of Squirrels / Total Time of Sighting` (`Total Time of Sighting` confirmed present, in minutes, mean ≈ 24.9, range 1–70, 20 nulls / 700 rows).

Expected output of this phase:

- A short data dictionary for the fields relevant to the research question.
- A clear explanation of the row unit for `squirrel.csv` and `hectare.csv`.
- A decision about whether `Hectare`, `Date`, and hectare-level count features are valid predictors.

Decision status: Q1–Q4 all closed. Phase 0 complete.

Additional data-understanding observations (recorded so they are not surprises in later phases):

- **Missingness in `squirrel.csv`** (counts of nulls, n=3,023):
  - High-null free-text fields: `Color notes` (2,841), `Other Interactions` (2,783), `Other Activities` (2,586), `Specific Location` (2,547) — already excluded from predictors.
  - Modelling-relevant nulls: `Highlight Fur Color` (1,086), `Age` (121, plus 4 literal `"?"` to treat as missing), `Above Ground Sighter Measurement` (114), `Location` (64), `Primary Fur Color` (55).
- **`Above Ground Sighter Measurement` is mixed-type**: 2,116 rows hold the literal string `"FALSE"` (= ground plane), the rest are numeric heights (e.g., 10, 20, 15…). It is **redundant with `Location`** (`FALSE` ↔ `Ground Plane`), and the 64 missing `Location` rows align with 114 missing height rows. Parsing rule: `Location == "Above Ground"` → numeric height; else 0; missing → impute. Already noted in [AI_AGENT_GUIDE.md:98](AI_AGENT_GUIDE.md#L98).
- **`Highlight Fur Color` has 10 multi-valued combinations** (e.g., `"Cinnamon, White"`, `"Black, Cinnamon, White"`). Direct one-hot is messy; split into per-colour boolean flags or simplify before encoding. Phase 2 decision.
- **Date span is short and uneven**: 11 dates, 6–20 Oct 2018. Per-day squirrel counts range from 67 (10/20) to 434 (10/13) — heavy bias toward weekend dates. Treat as calendar feature only; not enough span for a temporal split.
- **Shift imbalance in `squirrel.csv`**: AM 1,347 vs PM 1,676 even though `hectare.csv` is balanced 350/350 — PM sessions yielded more squirrels per session. Worth noting in interpretation.
- **`hectare.csv` missingness**: `Litter` (381 nulls — over half), `Hectare Conditions` (40), `Total Time of Sighting` (20), `Sighter Observed Weather Data` (19), `Other Animal Sightings` (32). Encode missingness explicitly (e.g., `"Unknown"`) where it may be informative.
- **`Hectare Conditions`** has a typo level `"Medium"` (n=1) — fold into `"Moderate"` during cleaning.
- **`Sighter Observed Weather Data` is free text**, 498 unique strings (e.g., `"57º F, overcast"`). Probably not worth using directly; if needed, parse temperature and a coarse sky-condition flag. Likely excluded for the main model.
- **`Other Animal Sightings`**: confirmed comma-separated free text (`"Humans, Dogs, Pigeons, Sparrows"` etc.). Parsing approach for Phase 2 is keyword flags (humans / dogs / cats / hawks / pigeons / etc.).
- **5 duplicate `Unique Squirrel ID`s** (3,023 rows but 3,018 unique IDs). Minor — flag but not blocking; preprocessing teammate to decide whether to dedupe.
- **Observer effects**: 235 unique sighters, median 2 sessions per sighter, max 19. Could induce observer bias in `Approaches` / `Runs from` labelling but is not in the predictor set; mention as a limitation.

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
  - Reason: the research question is binary by wording. `indifferent` is a "no decision" outcome, and keeping it as a third class lets a trivial classifier score high accuracy by always predicting indifferent, which masks whether the model has actually learned approach/avoid drivers. A three-class sensitivity run is explicitly out of scope for this project — focus stays on a single, well-justified binary model.

- **Q2 (closed): Relabel resolvable multi-flag rows; drop genuine conflicts.**
  - `A+I` (15 rows) → **approach** (active behaviour wins over neutral).
  - `I+R` (28 rows) → **avoid** (same logic; runs-from is the active behaviour).
  - `A+R` (16 rows) and `A+I+R` (4 rows) → **drop** as genuine conflicts that cannot be resolved without guessing.
  - Net effect: binary pool grows from 773 → **804** rows (158 approach, 658 avoid; imbalance ≈ 1 : 4.16, essentially unchanged from 1 : 4.4).
  - **Note for the data preprocessing teammate**: this relabelling rule is applied during target construction, before any feature preprocessing or train/test split. The 20 conflict rows (`A+R`, `A+I+R`) must be removed at the same step.

- **Q3 (closed): Exclude rows with no recorded interaction (780 rows).**
  - Reason: "no interaction recorded" is ambiguous — it could mean the squirrel never encountered a human, or the interaction simply was not logged. Treating it as `approach` or `avoid` is wrong by definition, and treating it as `indifferent` would conflate "saw a human and did not react" with "no human was present", breaking the meaning of the target.

- **Q4 (open with mitigation plan): Data-sufficiency concern acknowledged.**
  - The binary pool of ~804 rows is small. To make this defensible, the project will:
    - Use stratified k-fold cross-validation (k = 5) on the training portion so every row contributes to evaluation and variance is controlled.
    - Use class-weighted models (`class_weight="balanced"`) so the 158 approach rows are not drowned by the 658 avoid rows.
    - Restrict to regularised, low-complexity models (logistic regression, shallow decision tree, small random forest) — appropriate for ~800 rows with ~30–40 features.
  - Final adequacy of the binary pool will be revisited after Phase 3 (split strategy) and Phase 5 (metric choice). If 804 rows still appears insufficient at that stage, the response will be to restrict the feature set or simplify the model further — three-class is not a fallback option.

### Final target rule

- `y = 1` (approach): `Approaches & ~Runs from` after the Q2 relabel resolves `A+I` to approach.
- `y = 0` (avoid): `Runs from & ~Approaches` after the Q2 relabel resolves `I+R` to avoid.
- Drop: exact indifferent (1,407), no interaction recorded (780), `A+R` (16), `A+I+R` (4) — total 2,207 rows excluded; binary modelling pool = 816 candidate rows, **804 after the relabel rule** (158 approach, 658 avoid).

### Expected output of this phase

- Final target definition. ✅
- Inclusion/exclusion rule for ambiguous or missing interaction rows. ✅
- Short justification that can be defended in the report methodology and oral assessment. ✅

Decision status: Q1, Q2, Q3 closed; Q4 open pending Phase 3/5 confirmation.

## Phase 2: Predictors

Goal: decide which raw and engineered predictors belong in the model.

Current assumptions:

- `stories.csv` is probably not used for the main modelling task.
- `squirrel.csv` and `hectare.csv` are combined using `Hectare`, `Shift`, and `Date`.

### Decisions

- **Q1 (closed): Include both fur-colour fields, but simplify `Highlight Fur Color`.**
  - `Primary Fur Color`: keep raw → one-hot encode (3 levels, 55 nulls).
  - `Highlight Fur Color`: split the multi-valued string into per-colour boolean flags `highlight_white`, `highlight_cinnamon`, `highlight_black`, plus `highlight_missing` for the 1,086 nulls.
  - Reason: the research question explicitly asks which features are most influential, so excluding fur on conceptual grounds prevents the model from answering. Per-colour flags handle the 10 multi-valued combinations cleanly.
  - **Note for the data-preprocessing teammate**: this parsing happens during feature construction.

- **Q2 (closed): `X` and `Y` are kept as standardised continuous predictors; no spatial zoning.**
  - Fit `StandardScaler` on training data only and apply the same transform to the test set and each CV fold's holdout. Wrap inside the same scikit-learn `Pipeline` as the model so `cross_val_score` / `GridSearchCV` handle fit/transform correctly. This follows the leakage rule already in [AI_AGENT_GUIDE.md:119](AI_AGENT_GUIDE.md#L119).
  - Reason: trees use raw coordinates natively; LR gets a weak linear approximation, which is acceptable. Coarse zones inflate dimensionality on the 804-row pool without clear benefit. Map visualisation is for the report, not the model.

- **Q3 (closed): Drop raw `Location` and raw `Above Ground Sighter Measurement`; keep three engineered features.**
  - `is_above_ground` = `True` if `Location == "Above Ground"`, else `False`.
  - `above_ground_numeric` = parsed numeric height from `Above Ground Sighter Measurement` when `Location == "Above Ground"`; `0` when `Location == "Ground Plane"`; median-impute (train-only) when missing.
  - `location_missing` = `True` if `Location` is null, else `False` (covers the 64 null `Location` rows / 114 null height rows).
  - **Note for the data-preprocessing teammate**: this parsing matches the rule already noted in [AI_AGENT_GUIDE.md:98](AI_AGENT_GUIDE.md#L98).

- **Q4 (closed): Date is reduced to `is_weekend` only; drop raw date and `day_of_week`.**
  - `is_weekend` = `True` if the parsed date falls on Saturday/Sunday, else `False`.
  - Reason: the dataset shows a heavy weekend bias in squirrel counts (10/13 Sat = 434, 10/20 Sat = 67, midweek dates 200–300). Plausible mechanism: more park visitors on weekends → shifted approach/avoid base rates. **Caveat: the visitor-density mechanism is inferred, not stated in the assignment spec or rubric** — flag as an assumption in the report. Raw date and `day_of_week` are dropped because 11 unevenly distributed dates over 14 days are too sparse to support more granular calendar features.

- **Q5 (closed): Drop `animals_humans_present`; keep the other animal-keyword flags.**
  - Final flags from `Other Animal Sightings`: `animals_dogs_present`, `animals_cats_present`, `animals_hawks_present`, `animals_pigeons_present`, plus `animals_data_missing` for the 32 null rows.
  - Each flag = `True` if the keyword (case-insensitive) appears in the comma-separated string.
  - Reason: every observation was made by a human sighter, so `animals_humans_present` is near-constant in the predictor and risks being read as causal when it just reflects the data-collection setup. (Supersedes the suggestion in [AI_AGENT_GUIDE.md:100](AI_AGENT_GUIDE.md#L100).)

- **Q6 (closed): Final engineered-feature list and weather-data inclusion.**
  - **Removed** from the engineered set: `activity_count`, `is_active`, `vocalisation_count`, `tail_signal_count`. Reason: they are perfect linear combinations of the underlying behaviour and signal flags (Q7 chose flags-only); keeping both forms creates collinearity for no information gain.
  - **Kept** engineered features (with definitions):
    - `is_above_ground` — see Q3.
    - `above_ground_numeric` — see Q3.
    - `location_missing` — see Q3.
    - `is_weekend` — see Q4.
    - `highlight_white` / `highlight_cinnamon` / `highlight_black` — `True` if the colour name appears in `Highlight Fur Color`, else `False`.
    - `highlight_missing` — `True` if `Highlight Fur Color` is null, else `False`.
    - `animals_dogs_present` / `animals_cats_present` / `animals_hawks_present` / `animals_pigeons_present` — `True` if the keyword (case-insensitive) appears in `Other Animal Sightings`, else `False`.
    - `animals_data_missing` — `True` if `Other Animal Sightings` is null, else `False`.
    - `temperature_f` — see weather note below.
    - `sky_condition` — see weather note below.
    - `squirrel_density_proxy` = `Number of Squirrels` / `Total Time of Sighting` (squirrels per minute); median-impute (train-only) where undefined.
  - **`Sighter Observed Weather Data` is included as a predictor.** **Note for the data-preprocessing teammate**: parse the free-text field (498 unique strings, e.g., `"57º F, overcast"`, 19 nulls / 700 rows) into:
    - `temperature_f` — numeric (regex `\d+` on the temperature substring); median-impute (train-only) for nulls/parse failures.
    - `sky_condition` — coarse keyword match (`clear`, `overcast`, `cloudy`, `rain`) → one-hot encode; missing → `"Unknown"`.
  - `Specific Location` remains excluded (no parsing attempt).

- **Q7 (closed): Keep individual behaviour and signal flags only — do not also keep their summed counts.**
  - Behaviour flags retained: `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
  - Signal flags retained: `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
  - Reason: a sum is a perfect linear combination of its component flags; keeping both forms introduces collinearity with no extra information. Flags-only is the cleaner default and gives interpretable per-behaviour importance.

### Final predictor list

From `squirrel.csv`:

- Behaviour flags (5): `Running`, `Chasing`, `Climbing`, `Eating`, `Foraging`.
- Signal flags (5): `Kuks`, `Quaas`, `Moans`, `Tail flags`, `Tail twitches`.
- Time (2): `Shift` (OHE), `is_weekend`.
- Spatial (5): `X`, `Y` (standardised), `is_above_ground`, `above_ground_numeric`, `location_missing`.
- Individual (3 raw + 4 engineered): `Age` (OHE; treat `"?"` as missing), `Primary Fur Color` (OHE), `highlight_white`, `highlight_cinnamon`, `highlight_black`, `highlight_missing`.

From `hectare.csv` (joined on `Hectare`, `Shift`, `Date`):

- Hectare context (3 raw): `Litter` (OHE; encode missing as `"Unknown"` — 381 nulls), `Hectare Conditions` (OHE; fold `"Medium"` typo into `"Moderate"`), `Number of sighters`, `Number of Squirrels`, `Total Time of Sighting`.
- Animal flags (5 engineered): `animals_dogs_present`, `animals_cats_present`, `animals_hawks_present`, `animals_pigeons_present`, `animals_data_missing`.
- Weather (2 engineered): `temperature_f`, `sky_condition` (OHE).
- Density (1 engineered): `squirrel_density_proxy`.

### Excluded fields

- **Target leakers**: `Approaches`, `Indifferent`, `Runs from`, `Other Interactions`.
- **ID / duplicate / composite**: `Unique Squirrel ID`, `Lat/Long`, `Combination of Primary and Highlight Color`, `Hectare Squirrel Number`.
- **High-null free text**: `Color notes` (2,841 nulls), `Other Activities` (2,586), `Specific Location` (2,547).
- **Spatial leakage**: raw `Hectare` ID (Phase 0 decision — high cardinality + within-hectare leakage on a random split).
- **Calendar redundancy**: raw `Date`, `day_of_week` (Q4 — only `is_weekend` retained).
- **Replaced by engineered features**: raw `Location`, raw `Above Ground Sighter Measurement` (Q3), raw `Sighter Observed Weather Data` (Q6 — replaced by `temperature_f` + `sky_condition`), raw `Other Animal Sightings` (Q5 — replaced by per-keyword flags), raw `Highlight Fur Color` (Q1 — replaced by per-colour flags).
- **Collinear sums**: `activity_count`, `is_active`, `vocalisation_count`, `tail_signal_count` (Q7 — flags-only chosen).
- **Single-flag exclusion**: `animals_humans_present` (Q5 — collection-bias artefact).

### Expected output of this phase

- Final predictor list grouped by behaviour, time, spatial, individual characteristics, and hectare context. ✅
- Explicit list of excluded fields and why. ✅
- Feature-engineering plan. ✅

Decision status: Q1–Q7 all closed. Phase 2 complete.

## Phase 3: Train/Test Split

Goal: decide how to split data and prevent leakage.

### Decisions

- **Q1 (closed): Random split, not temporal.**
  - Reason: 11-day span (6–20 Oct 2018) is too short for a forecasting framing; the research question is about behavioural/contextual drivers, not future prediction. Per-day counts are heavily skewed (10/13: 434 vs 10/20: 67), so a chronological cut would over/under-represent specific weekends in either side.

- **Q2 (closed): Session-grouped splitting to prevent within-session leakage.**
  - Each (Hectare, Shift, Date) session contributes multiple squirrel rows that share identical hectare-context features (litter, weather, density proxy, other-animal flags, total sighting time). A row-level random split would put squirrels from the same session into both train and test, letting the model memorise session context.
  - Mitigation: build `session_id = Hectare + "_" + Shift + "_" + Date` and pass it as the `groups` argument to `StratifiedGroupKFold`. `session_id` is **not** a predictor — splitter input only.
  - **Owner: data preprocessing teammate** builds `session_id` (deterministic concat; no fitting required).
  - `Date` itself is already excluded as a predictor (Phase 2 Q4 — only `is_weekend` retained).

- **Q3 (closed): Single 80/20 hold-out + inner 5-fold CV (Option A).**
  - Steps:
    1. Outer split: one stratified-grouped 80/20 split over the 804-row binary pool, stratified on `y`, grouped by `session_id`, `random_state=42`.
    2. On the 80% training portion: 5-fold `StratifiedGroupKFold` (same stratify + group rules) inside `GridSearchCV` to choose model hyperparameters. Inner-CV scoring uses a `predict_proba`-based metric (e.g. `average_precision` or `roc_auc`) so hyperparameter selection is threshold-independent.
    3. Refit the best pipeline on the full 80% training set.
    4. Threshold tuning (real models only — see Q5): use `cross_val_predict(method="predict_proba", cv=StratifiedGroupKFold(5))` on the training set to obtain out-of-fold probabilities, sweep thresholds (e.g. 0.05–0.95 in 0.01 steps), pick the one maximising macro-F1.
    5. Apply the chosen threshold to the held-out 20% test set probabilities and compute metrics.
    6. **CI:** bootstrap-resample the test set (~1,000 resamples with replacement; report 2.5th/97.5th percentiles for each metric) to give a CI on held-out performance.
  - Expected sizes: ~643 train / ~161 test rows, with ~32 approach in the test set. Small-test-set caveat flagged as a report limitation; bootstrap CI makes it defensible.
  - Why not nested CV: the assignment expects a clear single train/test split + CV; one held-out test set is easier to defend in the report and oral; outer-fold scores from 161-row folds would be just as noisy in practice.

- **Q4 (closed): Preprocessing fitted only on training data, inside an sklearn `Pipeline` / `ColumnTransformer`.**
  - **Inside the pipeline (fit on training fold only):** `SimpleImputer(strategy="median")` on numeric, `SimpleImputer(strategy="constant", fill_value="Unknown")` on categoricals where missingness may be informative, `OneHotEncoder(handle_unknown="ignore")` on categoricals, `StandardScaler` on numerics for LR/KNN (not needed for tree models).
  - **Outside the pipeline (deterministic, fine on full data):** boolean/date parsing, target relabelling (A+I → approach, I+R → avoid), conflict-row drop (A+R, A+I+R), free-text parsing into keyword flags, joining squirrel ↔ hectare, basic type fixes (`Age == "?"` → NaN, parsing `Above Ground Sighter Measurement`), folding `Hectare Conditions` `"Medium"` → `"Moderate"`, building `session_id`.
  - **Task allocation:**
    - Data preprocessing teammate: everything outside the pipeline (deterministic rewrites of raw data) plus `session_id` construction.
    - Modelling teammate: the `Pipeline` / `ColumnTransformer`, including imputers, encoders, and `StandardScaler`. Rule of thumb: anything where "fit on train, transform test" matters lives in the modelling code.

- **Q5 (closed): Class imbalance handled by `class_weight="balanced"` + threshold tuning. No resampling.**
  - **`class_weight="balanced"`**: scikit-learn argument on `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`. Re-weights the loss so each class contributes equally regardless of size (with 158 approach / 658 avoid, every approach row counts ~4.16× as much during training). Complementary to stratification: stratification controls *how imbalance is distributed across folds*, `class_weight` controls *how the model treats the imbalance during training*. Both are required.
  - **Threshold tuning** (real models only, not dummy): chosen on training-set out-of-fold probabilities (Q3 step 4), applied once on the test set. Not a `GridSearchCV` hyperparameter — separate post-fit step using `predict_proba` instead of `predict`.
  - **Dummy baseline**: `DummyClassifier(strategy="most_frequent", random_state=42)` reported as a floor in the metrics table; optional second row with `strategy="stratified"` as a chance-level anchor. No hyperparameter tuning, no threshold tuning. Not counted as one of the two real supervised models — it is the reference point that proves the real models have learned something from the predictors.

### Phase 1 Q4 (data sufficiency) revisit

Confirmed defensible under Option A: ~643 train / ~161 test, ~32 approach in test. The combination of stratified-grouped split, `class_weight="balanced"`, low-complexity models, and bootstrap CIs on test metrics meets the Phase 1 mitigation plan. Small-test-set caveat to be flagged as a limitation in the report.

### Expected output of this phase

- Final train/test split rule. ✅
- Cross-validation strategy. ✅
- Leakage-prevention checklist (session-grouped split, pipeline-only preprocessing, raw `Date` excluded). ✅
- Imbalance-handling decision (`class_weight="balanced"` + threshold tuning, no resampling). ✅

Decision status: Q1–Q5 all closed. Phase 3 complete.

## Phase 4: Prediction Model Selection

Goal: choose the supervised models and their tuning ranges.

Current questions:

- Should the project use two or three supervised models?
- Which models best balance interpretability, performance, and assignment scope?
- What hyperparameters should be tuned for each model?
- Which model is best for feature influence and explanation?

Expected output of this phase:

- Final model list.
- Hyperparameter grid or small set of justified values.
- Baseline model definition.
- Rationale for why each model is suitable for the research question.

Decision status: open.

## Phase 5: Metrics Selection

Goal: decide how model performance will be evaluated and communicated.

Current questions:

- Which metrics are most suitable for the research question?
- How should metrics reflect class imbalance?
- With cross-validation, should we report confidence intervals, standard deviations, or error bars?
- Is model training cost worth discussing, such as runtime and storage?

Expected output of this phase:

- Primary metric.
- Secondary metrics.
- Confusion matrix interpretation plan.
- Cross-validation uncertainty reporting plan.
- Decision about whether runtime/storage is relevant enough to report.

Decision status: open.

## Phase Decision Log

Use this section to record final decisions after each phase is discussed.

- Phase 0: complete. Join key = (Hectare, Shift, Date) via left join from squirrel → hectare. Raw `Hectare` ID excluded as predictor; use X, Y and hectare-level context. `Hectare Squirrel Number` excluded; use a session-level **density predictor** = `Number of Squirrels / Total Time of Sighting`. Known data caveats (missingness, mixed-type fields, multi-valued highlight colour, shift imbalance, observer effects) catalogued above. Date-as-predictor and `Other Animal Sightings` parsing deferred to Phase 2.
- Phase 1: Q1, Q2, Q3 closed. Target = binary `approach` (158) vs `avoid` (658), 804 rows after relabelling `A+I → approach` and `I+R → avoid`. Three-class is **out of scope** (no sensitivity run). Excluded: exact indifferent (1,407), no-interaction (780), genuine conflicts `A+R` + `A+I+R` (20). Q4 (data sufficiency) open — mitigated by stratified 5-fold CV, `class_weight="balanced"`, and low-complexity models; revisit after Phase 3/5. Multi-flag relabel/drop owned by the data preprocessing teammate.
- Phase 2: complete. Final predictor set spans behaviour flags (5), signal flags (5), time (Shift + is_weekend), spatial (X, Y standardised, is_above_ground, above_ground_numeric, location_missing), individual characteristics (Age, Primary Fur Color OHE, four highlight_* flags), and hectare context (Litter, Hectare Conditions, four animal-keyword flags + animals_data_missing, Number of sighters, Number of Squirrels, Total Time of Sighting, temperature_f, sky_condition, squirrel_density_proxy). Excluded: target leakers, ID/composite fields, raw Hectare, raw Date and day_of_week, raw Location/Above Ground/Weather/Animal Sightings/Highlight (replaced by engineered features), all summed activity/signal counts (flags-only), `animals_humans_present` (collection-bias artefact), and Specific Location. X/Y standardisation must be fit on training data only inside a scikit-learn Pipeline. Calendar mechanism (visitor density on weekends) is an inferred assumption — flag in the report.
- Phase 3: complete. Random (not temporal) split. Single 80/20 stratified-grouped hold-out (`random_state=42`), grouped by `session_id = Hectare + "_" + Shift + "_" + Date` to prevent within-session leakage; 5-fold `StratifiedGroupKFold` `GridSearchCV` on the 80% training portion using a `predict_proba`-based metric for hyperparameter selection. Threshold tuned on training out-of-fold probabilities (real models only, not dummy) to maximise macro-F1, then applied once to the test set. Bootstrap CIs (~1,000 resamples) on test metrics. Preprocessing fitted only on training data inside an sklearn `Pipeline` / `ColumnTransformer` (median/constant imputers, OHE, StandardScaler for LR/KNN). Owner split: preprocessing teammate builds `session_id` and deterministic rewrites; modelling teammate owns the pipeline including standardisation. Class imbalance handled via `class_weight="balanced"` + threshold tuning; no SMOTE/oversampling. `DummyClassifier(strategy="most_frequent")` as the floor baseline. Phase 1 Q4 (data sufficiency) confirmed defensible — flag small test set (~32 approach rows) as a report limitation.
- Phase 4: pending.
- Phase 5: pending.

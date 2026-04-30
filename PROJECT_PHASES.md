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

Current questions:

- Should `Primary Fur Color` and `Highlight Fur Color` be predictors?
- Intuition says fur colour may be irrelevant to approach/avoid behaviour, but should it be tested or excluded on conceptual grounds?
- Can `X` and `Y` be useful, and if so should they be used directly, transformed into spatial zones, or only visualised?
- `Location` and `Above Ground Sighter Measurement` seem useful. How should they be cleaned and encoded?
- Should `Date` be a predictor? Options: raw date, day index since start, day-of-week, exclude. (Carried over from Phase 0.)
- How should `Other Animal Sightings` be used as a predictor? It is a free-text comma-separated list (e.g., "Humans, Pigeons, Dogs") and needs parsing into keyword flags (e.g., `animals_humans_present`, `animals_dogs_present`) before encoding. (Carried over from Phase 0.)
- What is the final list of raw features used from each dataset?
- What engineered features should be created?

Expected output of this phase:

- Final predictor list grouped by behaviour, time, spatial, individual characteristics, and hectare context.
- Explicit list of excluded fields and why.
- Feature-engineering plan.

Decision status: open.

## Phase 3: Train/Test Split

Goal: decide how to split data and prevent leakage.

Current questions:

- Is this a temporal-series question? Current assumption: no, so random stratified splitting is acceptable.
- Is there any risk of future-looking leakage from using `Date` or hectare-level summaries?
- Since cross-validation is required for model selection, what is the exact split strategy?
- Which preprocessing steps are fitted only on training data?
- How should class imbalance be handled?

Expected output of this phase:

- Final train/test split rule.
- Cross-validation strategy.
- Leakage-prevention checklist.
- Imbalance-handling decision, such as class weighting, resampling, threshold tuning, or metric choice.

Decision status: open.

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
- Phase 2: pending.
- Phase 3: pending.
- Phase 4: pending.
- Phase 5: pending.

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

- In `hectare.csv`, why does it contain a `Date` feature?
- For a single hectare-shift, is that hectare visited only once?
- If a hectare-shift is observed once, should `Hectare` itself be a predictor?
- What is the difference between `Hectare Squirrel Number` in `squirrel.csv` and `Number of Squirrels` in `hectare.csv`?

Expected output of this phase:

- A short data dictionary for the fields relevant to the research question.
- A clear explanation of the row unit for `squirrel.csv` and `hectare.csv`.
- A decision about whether `Hectare`, `Date`, and hectare-level count features are valid predictors.

Decision status: open.

## Phase 1: Research Question Clarification

Goal: decide the exact modelling target and make sure it matches the research question.

Research question:

> Can we predict whether a squirrel will approach or avoid humans based on its behaviour, location, and time of observation, and which features are most influential?

Current questions:

- The raw interaction counts are roughly: approach 179, indifferent 1455, run away 679.
- Should the target `y` be categorical with three classes: `approach`, `indifferent`, and `run away`?
- Or should the main target be binary: `approach` versus `avoid`, with `indifferent` excluded or used in a secondary analysis?
- How should rows with multiple interaction flags be handled?
- How should rows with no recorded interaction be handled?

Expected output of this phase:

- Final target definition.
- Inclusion/exclusion rule for ambiguous or missing interaction rows.
- Short justification that can be defended in the report methodology and oral assessment.

Decision status: open.

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

- Phase 0: pending.
- Phase 1: pending.
- Phase 2: pending.
- Phase 3: pending.
- Phase 4: pending.
- Phase 5: pending.

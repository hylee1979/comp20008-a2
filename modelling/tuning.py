from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

from modelling.config import (
    INNER_CV_SCORER, LR_GRID, N_SPLITS_INNER, RANDOM_STATE,
    RF_GRID, THRESHOLD_RANGE,
)
from modelling.threshold_plots import plot_pr_curve


@dataclass
class TunedModel:
    name: str
    pipeline: object
    best_params: dict
    cv_results: pd.DataFrame
    threshold: float


def inner_cv():
    return StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)


def grid_search(pipeline, grid, X, y, groups, name):
    print(f"[{name}] GridSearchCV  grid={grid}  scorer={INNER_CV_SCORER}")
    gs = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=INNER_CV_SCORER,
        cv=inner_cv(),
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    gs.fit(X, y, groups=groups)
    cv_df = pd.DataFrame(gs.cv_results_) #TODO: check what is in cv_results_ and if we can use it for analysis later. Can we plot PR curves?
    print(f"[{name}] best_params={gs.best_params_}  best_inner_CV_AP={gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, cv_df


def tune_threshold(pipeline, X, y, groups, name, output_dir=None):
    proba_oof = cross_val_predict( #TODO: what does cross_val_predict return when method="predict_proba"? check if the columns are in the same order as the classes in y, and if it returns probabilities for both classes or just the positive class
        pipeline, X, y,
        groups=groups,
        cv=inner_cv(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    scores = [
        f1_score(y, (proba_oof >= t).astype(int), average="macro", zero_division=0)
        for t in THRESHOLD_RANGE
    ] #TODO: maybe can make a plot of threshold vs score to visualise how the threshold affects the performance, and if there are multiple thresholds that give similar performance. or make PR curve and indicate the chosen threshold on the curve.
    best_idx = int(np.argmax(scores))
    best_t = float(THRESHOLD_RANGE[best_idx])
    if output_dir is not None:
        plot_pr_curve(y, proba_oof, best_t, name, output_dir)
    print(f"[{name}] best_threshold={best_t:.2f}  train OOF macro-F1={scores[best_idx]:.4f}")
    return best_t


def tune_model(pipeline, grid, X, y, groups, name, output_dir=None):
    # inner cv for hyperparameter tuning, then refit on the whole training set
    refit_pipeline, best_params, cv_df = grid_search(pipeline, grid, X, y, groups, name)
    # inner cv for threshold tuning, using the refitted pipeline to get OOF probabilities
    threshold = tune_threshold(refit_pipeline, X, y, groups, name, output_dir=output_dir)
    return TunedModel(
        name=name,
        pipeline=refit_pipeline,
        best_params=best_params,
        cv_results=cv_df,
        threshold=threshold,
    )


def fit_dummy(pipeline, X, y):
    pipeline.fit(X, y)
    return TunedModel(
        name="Dummy",
        pipeline=pipeline,
        best_params={},
        cv_results=pd.DataFrame(),
        threshold=0.5,
    )


def tune_all(split, build_lr, build_rf, build_dummy, output_dir=None):
    """Tune dummy / LR / RF on the training portion of an outer split."""
    tuned = {}

    print("--- Dummy (most_frequent) ---")
    tuned["Dummy"] = fit_dummy(build_dummy(), split.X_train, split.y_train)

    print("--- Logistic Regression ---")
    tuned["LR"] = tune_model(
        build_lr(split.X_train), LR_GRID,
        split.X_train, split.y_train, split.groups_train,
        name="LR",
        output_dir=output_dir,
    )

    print("--- Random Forest ---")
    tuned["RF"] = tune_model(
        build_rf(split.X_train), RF_GRID,
        split.X_train, split.y_train, split.groups_train,
        name="RF",
        output_dir=output_dir,
    )

    return tuned

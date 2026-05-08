import pandas as pd
from sklearn.inspection import permutation_importance

from modelling.config import (
    CATEGORICAL_COLS, INNER_CV_SCORER, PERM_N_REPEATS, RANDOM_STATE,
)
from modelling.pipelines import get_feature_names


def aggregate_to_parent(transformed_names, values, raw_categorical, aggfunc):
    """Sum OHE dummies back to the raw (parent) feature for cross-model comparison."""
    parents = []
    for name in transformed_names:
        rest = name.split("__", 1)[1] if "__" in name else name
        matched = None
        for cat in raw_categorical:
            if rest == cat or rest.startswith(cat + "_"):
                matched = cat
                break
        parents.append(matched or rest)

    df_out = pd.DataFrame({"feature": parents, "value": values})
    if aggfunc == "sum_abs":
        return df_out.assign(value=df_out["value"].abs()).groupby("feature", as_index=False)["value"].sum()
    return df_out.groupby("feature", as_index=False)["value"].sum()


def lr_coefficients(pipeline, raw_categorical):
    model = pipeline.named_steps["model"]
    feat_names = get_feature_names(pipeline)
    coefs = model.coef_.ravel()

    per_dummy = pd.DataFrame({"feature": feat_names, "coefficient": coefs})

    aggregated = aggregate_to_parent(feat_names, coefs, raw_categorical, "sum_abs")
    aggregated = aggregated.rename(columns={"value": "abs_coefficient_sum"})
    aggregated = aggregated.sort_values("abs_coefficient_sum", ascending=False).reset_index(drop=True)

    print("LR top-10 features (sum |coef|):")
    print(aggregated.head(10).to_string(index=False))
    return aggregated, per_dummy


def rf_impurity_importance(pipeline, raw_categorical):
    model = pipeline.named_steps["model"]
    feat_names = get_feature_names(pipeline)
    importances = model.feature_importances_

    aggregated = aggregate_to_parent(feat_names, importances, raw_categorical, "sum")
    aggregated = aggregated.rename(columns={"value": "impurity_importance"})
    aggregated = aggregated.sort_values("impurity_importance", ascending=False).reset_index(drop=True)

    print("RF impurity top-10 features:")
    print(aggregated.head(10).to_string(index=False))
    return aggregated


def model_permutation_importance(name, pipeline, X_test, y_test):
    print(f"{name} permutation importance: n_repeats={PERM_N_REPEATS}, scoring={INNER_CV_SCORER}")
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=PERM_N_REPEATS,
        scoring=INNER_CV_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    df_perm = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    print(f"{name} permutation top-10 features:")
    print(df_perm.head(10).to_string(index=False))
    return df_perm


def analyse_all(tuned, split):
    raw_cat = [c for c in CATEGORICAL_COLS if c in split.X_train.columns]

    lr_agg, lr_per_dummy = lr_coefficients(tuned["LR"].pipeline, raw_cat)
    rf_imp = rf_impurity_importance(tuned["RF"].pipeline, raw_cat)
    lr_perm = model_permutation_importance("LR", tuned["LR"].pipeline, split.X_test, split.y_test)
    rf_perm = model_permutation_importance("RF", tuned["RF"].pipeline, split.X_test, split.y_test)

    return {
        "lr_coefficients_aggregated": lr_agg,
        "lr_coefficients_per_dummy": lr_per_dummy,
        "lr_permutation": lr_perm,
        "rf_impurity": rf_imp,
        "rf_permutation": rf_perm,
    }

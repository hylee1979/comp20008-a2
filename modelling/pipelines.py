from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import numpy as np
import pandas as pd

from modelling.config import (
    BOOLEAN_FLAGS, CATEGORY_DICT, OHE_COLS, LR_FIXED, NUMERIC_COLS,
    RANDOM_STATE, RF_FIXED,
)


def resolve_columns(df):
    num = [c for c in NUMERIC_COLS if c in df.columns]
    ohe = [c for c in OHE_COLS if c in df.columns]
    hectare = "Hectare Conditions" if "Hectare Conditions" in df.columns else None
    flag = [c for c in BOOLEAN_FLAGS if c in df.columns]
    return num, ohe, hectare, flag


def numeric_branch(scale):
    if scale:
        return Pipeline([("scaler", StandardScaler())])
    return "passthrough"


def ohe_branch(ohe_cols):
    ohe = OneHotEncoder(
                    categories=[CATEGORY_DICT[c] for c in ohe_cols],
                    handle_unknown="ignore",
                    sparse_output=False,
                )
    return Pipeline([("ohe", ohe)])


def encode_hectare_conditions(X):
    """
    X: DataFrame with one column: 'Hectare Conditions'
       value examples:
       - "Busy, Calm"
       - "Moderate"
       - np.nan
       - ["Busy", "Calm"]  # also allowed
    return: DataFrame with fixed columns:
       hectare_condition_Busy / Calm / Moderate
    """
    col = X.iloc[:, 0]

    allowed = CATEGORY_DICT["Hectare Conditions"]
    out = pd.DataFrame(
        0,
        index=X.index,
        columns=[f"hectare_condition_{c}" for c in allowed],
    )

    for idx, value in col.items():
        if isinstance(value, list):
            labels = value
        elif pd.isna(value):
            labels = []
        else:
            labels = [item.strip() for item in str(value).split(",")]

        for label in labels:
            if label in allowed:
                out.loc[idx, f"hectare_condition_{label}"] = 1

    return out


def hectare_branch():
    hectare_transformer = FunctionTransformer(
        encode_hectare_conditions,
        validate=False,
        feature_names_out=lambda self, input_features: np.array(
            [f"hectare_condition_{c}" for c in CATEGORY_DICT["Hectare Conditions"]]
        ),
    )
    return Pipeline([
        ("hectare", hectare_transformer),
    ])

def build_lr_pipeline(df):
    num, ohe, hec, flag = resolve_columns(df)
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_branch(scale=True), num),
            ("ohe", ohe_branch(ohe), ohe),
            ("hectare", hectare_branch(), [hec] if hec else []),
            ("flag", "passthrough", flag)
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(**LR_FIXED))])


def build_rf_pipeline(df):
    num, ohe, hec, flag = resolve_columns(df)
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_branch(scale=False), num),
            ("ohe", ohe_branch(ohe), ohe),
            ("hectare", hectare_branch(), [hec] if hec else []),
            ("flag", "passthrough", flag)
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("pre", pre), ("model", RandomForestClassifier(**RF_FIXED))])


def build_dummy_pipeline():
    return Pipeline([
        ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
    ])


def get_feature_names(pipeline):
    pre = pipeline.named_steps.get("pre")
    if pre is None:
        return []
    return list(pre.get_feature_names_out())

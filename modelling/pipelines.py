from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelling.config import (
    BOOLEAN_FLAGS, CATEGORICAL_COLS, LR_FIXED, NUMERIC_COLS,
    RANDOM_STATE, RF_FIXED,
)


def resolve_columns(df):
    num = [c for c in NUMERIC_COLS if c in df.columns]
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    flag = [c for c in BOOLEAN_FLAGS if c in df.columns]
    return num, cat, flag


def numeric_branch(scale):
    steps = [("imputer", SimpleImputer(strategy="median"))] # TODO: should remove imputer, since my teammate has done the preprocessing
    if scale:
        steps.append(("scaler", StandardScaler())) # standard normal
    return Pipeline(steps)


def categorical_branch():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), # TODO: should remove imputer, since my teammate has done the preprocessing
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # TODO: check if i need this
    ])


def flag_branch():
    steps = [("imputer", SimpleImputer(strategy="most_frequent"))] # TODO: should remove imputer, since my teammate has done the preprocessing
    return Pipeline(steps)


def build_lr_pipeline(df):
    num, cat, flag = resolve_columns(df)
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_branch(scale=True), num),
            ("cat", categorical_branch(), cat),
            ("flag", flag_branch(scale=True), flag),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(**LR_FIXED))])


def build_rf_pipeline(df):
    num, cat, flag = resolve_columns(df)
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_branch(scale=False), num),
            ("cat", categorical_branch(), cat),
            ("flag", flag_branch(scale=False), flag),
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

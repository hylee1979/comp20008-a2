from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from modelling.config import ALL_PREDICTOR_COLS, RANDOM_STATE, SESSION_COL, TARGET_COL


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


def stratified_grouped_split(df):
    """80/20 stratified-grouped hold-out via the first fold of StratifiedGroupKFold(5)."""
    feature_cols = [c for c in ALL_PREDICTOR_COLS if c in df.columns]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)
    groups = df[SESSION_COL]

    # grouped by session_id, so that all rows from a session are in the same split. prevent leakage of session-specific info (e.g. location). stratified by target, to maintain class balance in both splits.
    stratified_folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(stratified_folds.split(X, y, groups=groups))

    s = SplitResult(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        groups_train=groups.iloc[train_idx].reset_index(drop=True),
        groups_test=groups.iloc[test_idx].reset_index(drop=True),
    )

    overlap = set(s.groups_train.unique()) & set(s.groups_test.unique())
    if overlap:
        raise RuntimeError(f"Session leakage: {len(overlap)} sessions in both splits")

    n_total = len(s.y_train) + len(s.y_test)
    print(f"Train: {len(s.y_train)} rows ({100 * len(s.y_train) / n_total:.1f}%)")
    print(f"Test:  {len(s.y_test)} rows ({100 * len(s.y_test) / n_total:.1f}%)")
    print(f"Train class balance: {s.y_train.value_counts().to_dict()}")
    print(f"Test class balance:  {s.y_test.value_counts().to_dict()}")
    print(f"Sessions: train={s.groups_train.nunique()} unique, test={s.groups_test.nunique()} unique")

    return s

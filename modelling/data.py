import pandas as pd

from modelling.config import ALL_PREDICTOR_COLS, SESSION_COL, TARGET_COL


def load_processed_data(path=None):
    df = pd.read_csv(path)

    expected = [TARGET_COL, SESSION_COL] + ALL_PREDICTOR_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns ({len(missing)}): {missing}")

    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {path}")
    print(f"Class balance: {df[TARGET_COL].value_counts().to_dict()}")

    return df

"""Synthetic processed-data stub for smoke-testing the modelling pipeline.

Generates a dataframe matching the expected post-preprocessing schema
(see AI_AGENT_GUIDE.md / PROJECT_PHASES.md Phase 2). Intentionally injects
weak signal into a few features so models do better than the dummy.

Run:

    PYENV_VERSION=3.13.2/envs/comp20008 python -m scripts.make_synthetic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed.csv"

RNG = np.random.default_rng(42)
N = 804
N_APPROACH = 158


def _rand_bool(p: float, n: int = N) -> np.ndarray:
    return (RNG.random(n) < p).astype(int)


def main() -> None:
    # Build session_id from ~250 sessions (3-4 squirrels per session).
    n_sessions = 250
    session_ids = np.array([f"H{(i % 250):03d}_AM_10142018" for i in range(n_sessions)])
    rows = RNG.choice(session_ids, size=N, replace=True)

    df = pd.DataFrame({"session_id": rows})

    # Numeric features
    df["X"] = RNG.normal(0, 1, N)
    df["Y"] = RNG.normal(0, 1, N)
    df["above_ground_numeric"] = np.clip(RNG.normal(2, 4, N), 0, None)
    df["Number of sighters"] = RNG.integers(1, 5, N)
    df["Number of Squirrels"] = RNG.integers(1, 12, N)
    df["Total Time of Sighting"] = RNG.integers(5, 60, N)
    df["temperature_f"] = RNG.normal(60, 10, N)
    df["squirrel_density_proxy"] = df["Number of Squirrels"] / df["Total Time of Sighting"]

    # Categorical features
    df["Shift"] = RNG.choice(["AM", "PM"], N)
    df["Age"] = RNG.choice(["Adult", "Juvenile", "Unknown"], N, p=[0.7, 0.2, 0.1])
    df["Primary Fur Color"] = RNG.choice(["Gray", "Cinnamon", "Black", "Unknown"], N, p=[0.7, 0.2, 0.05, 0.05])
    df["Litter"] = RNG.choice(["Some", "None", "Unknown"], N, p=[0.4, 0.4, 0.2])
    df["Hectare Conditions"] = RNG.choice(["Calm", "Busy", "Moderate", "Unknown"], N)
    df["sky_condition"] = RNG.choice(["clear", "overcast", "cloudy", "rain", "Unknown"], N)

    # Behaviour flags
    df["Running"] = _rand_bool(0.20)
    df["Chasing"] = _rand_bool(0.05)
    df["Climbing"] = _rand_bool(0.20)
    df["Eating"] = _rand_bool(0.25)
    df["Foraging"] = _rand_bool(0.40)

    # Signal flags
    df["Kuks"] = _rand_bool(0.05)
    df["Quaas"] = _rand_bool(0.03)
    df["Moans"] = _rand_bool(0.02)
    df["Tail flags"] = _rand_bool(0.05)
    df["Tail twitches"] = _rand_bool(0.10)

    # Highlight + animal + other flags
    for c in ("highlight_gray", "highlight_white", "highlight_cinnamon", "highlight_black"):
        df[c] = _rand_bool(0.15)
    df["highlight_missing"] = _rand_bool(0.30)
    df["animals_humans_present"] = _rand_bool(0.20)
    df["animals_data_missing"] = _rand_bool(0.05)
    df["is_weekend"] = _rand_bool(0.30)
    df["is_above_ground"] = _rand_bool(0.30)
    df["location_missing"] = _rand_bool(0.05)

    # Inject mild signal so LR/RF can beat dummy:
    # Approach more likely when Eating or Foraging, less likely when humans present.
    score = (
        0.6 * df["Eating"]
        + 0.4 * df["Foraging"]
        - 0.7 * df["animals_humans_present"]
        + 0.3 * (df["Primary Fur Color"] == "Cinnamon").astype(int)
        + 0.05 * RNG.normal(size=N)
    )
    # Pick top ~158 by score as approach to hit the ~20% imbalance.
    top_idx = np.argsort(-score)[:N_APPROACH]
    y = np.zeros(N, dtype=int)
    y[top_idx] = 1
    df["y"] = y

    # Inject some NaNs to exercise the imputers.
    nan_rows = RNG.choice(N, 30, replace=False)
    df.loc[nan_rows, "above_ground_numeric"] = np.nan
    df.loc[RNG.choice(N, 50, replace=False), "Litter"] = np.nan
    df.loc[RNG.choice(N, 20, replace=False), "temperature_f"] = np.nan

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df)} rows, {df['y'].mean():.1%} approach)")


if __name__ == "__main__":
    main()

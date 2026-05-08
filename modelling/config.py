from pathlib import Path

import numpy as np


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Reproducibility
RANDOM_STATE = 42

# Target / split
TARGET_COL = "y"
SESSION_COL = "session_id"
N_SPLITS_INNER = 5

# Predictor groups
NUMERIC_COLS = [
    "X", "Y",
    "above_ground_numeric",
    "Number of sighters", "Number of Squirrels", "Total Time of Sighting",
    "temperature",
    "squirrel_density_proxy",
]

CATEGORICAL_COLS = [
    "Shift", "Age", "Primary Fur Color",
    "Litter", "Hectare Conditions",
    "sky_condition",
]

BEHAVIOUR_FLAGS = ["Running", "Chasing", "Climbing", "Eating", "Foraging"]
SIGNAL_FLAGS = ["Kuks", "Quaas", "Moans", "Tail flags", "Tail twitches"]
HIGHLIGHT_FLAGS = [
    "highlight_gray", "highlight_white", "highlight_cinnamon",
    "highlight_black", "highlight_missing",
]
ANIMAL_FLAGS = ["animals_human_present", 'animals_dog_present', 'animals_pigeon_present',
                'animals_bird_present','animals_sparrow_present', 'animals_duck_present',
                'animals_data_missing']
OTHER_FLAGS = ["is_weekend", "is_above_ground", "location_missing"]

BOOLEAN_FLAGS = (
    BEHAVIOUR_FLAGS + SIGNAL_FLAGS + HIGHLIGHT_FLAGS + ANIMAL_FLAGS + OTHER_FLAGS
)

ALL_PREDICTOR_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BOOLEAN_FLAGS

# Hyperparameter grids
LR_GRID = {"model__C": [0.01, 0.1, 1, 10]}
LR_FIXED = {
    "penalty": "l1",
    "solver": "liblinear",
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_GRID = {"model__max_depth": [None, 10, 20]}
RF_FIXED = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Inner CV scorer + threshold sweep
INNER_CV_SCORER = "average_precision"
THRESHOLD_RANGE = np.arange(0.05, 0.951, 0.01)

# Bootstrap for test-set CIs
BOOTSTRAP_N = 1000
BOOTSTRAP_LOW_PCT = 2.5
BOOTSTRAP_HIGH_PCT = 97.5

# Permutation importance
PERM_N_REPEATS = 30

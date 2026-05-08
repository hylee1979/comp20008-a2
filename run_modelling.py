"""Run the squirrel approach/avoid modelling pipeline end to end.

Usage:

    python run_modelling.py
    python run_modelling.py --data data/processed.csv

Outputs are written under outputs/tables/ and outputs/figures/.
"""

import argparse
import time
from pathlib import Path

from modelling.data import load_processed_data
from modelling.evaluation import evaluate_all, metrics_to_frame
from modelling.feature_influence import analyse_all
from modelling.outputs import write_outputs
from modelling.pipelines import build_dummy_pipeline, build_lr_pipeline, build_rf_pipeline
from modelling.split import stratified_grouped_split
from modelling.tuning import tune_all


def parse_args():
    parser = argparse.ArgumentParser(description="Run the squirrel modelling pipeline.")
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=None,
        help="Path to the processed CSV. Defaults to data/processed.csv at the project root.",
    )
    return parser.parse_args()


def run(data_path=None):
    print("=" * 60)
    print("LOAD DATA")
    print("=" * 60)
    df = load_processed_data(data_path)

    print("=" * 60)
    print("OUTER 80/20 SPLIT")
    print("=" * 60)
    split = stratified_grouped_split(df)

    print("=" * 60)
    print("TUNING: DUMMY / LR / RF")
    print("=" * 60)
    tuned = tune_all(
        split,
        build_lr=build_lr_pipeline,
        build_rf=build_rf_pipeline,
        build_dummy=build_dummy_pipeline,
    )

    print("=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    eval_results = evaluate_all(tuned, split)
    metrics_df = metrics_to_frame(eval_results)

    print("=" * 60)
    print("FEATURE INFLUENCE")
    print("=" * 60)
    influence = analyse_all(tuned, split)

    print("=" * 60)
    print("WRITE OUTPUTS")
    print("=" * 60)
    write_outputs(tuned, eval_results, influence, metrics_df)


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    run(args.data)

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")

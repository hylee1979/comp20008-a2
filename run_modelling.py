"""Run the squirrel approach/avoid modelling pipeline end to end.

Usage:

    python run_modelling.py
    python run_modelling.py --data data/cleaned_table_2.parquet

Outputs are written under a timestamped outputs/experiment_YYYYMMDD_HHMMSS/ directory.
"""

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")

from modelling.data import load_processed_data
from modelling.evaluation import evaluate_all, metrics_to_frame
from modelling.feature_influence import analyse_all
from modelling.outputs import create_experiment_dir, relative_path, write_outputs
from modelling.pipelines import build_dummy_pipeline, build_lr_pipeline, build_rf_pipeline
from modelling.split import stratified_grouped_split
from modelling.tuning import tune_all


def parse_args():
    parser = argparse.ArgumentParser(description="Run the squirrel modelling pipeline.")
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default="data/cleaned_table_2.parquet",
        help="Path to the processed parquet file. Defaults to data/cleaned_table_2.parquet at the project root.",
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

    des_dir = create_experiment_dir()
    print(f"Experiment outputs: {relative_path(des_dir)}")

    print("=" * 60)
    print("TUNING: DUMMY / LR / RF")
    print("=" * 60)
    tuned = tune_all(
        split,
        build_lr=build_lr_pipeline,
        build_rf=build_rf_pipeline,
        build_dummy=build_dummy_pipeline,
        output_dir=des_dir,
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
    write_outputs(tuned, eval_results, influence, metrics_df, data_path=data_path, des_dir=des_dir)


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    run(args.data)

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")

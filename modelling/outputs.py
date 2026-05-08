from datetime import datetime
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelling import config as experiment_config
from modelling.config import OUTPUT_DIR, PROJECT_ROOT


CONFIG_SECTIONS = {
    "Paths": ["PROJECT_ROOT", "OUTPUT_DIR"],
    "Reproducibility": ["RANDOM_STATE"],
    "Target / split": ["TARGET_COL", "SESSION_COL", "N_SPLITS_INNER"],
    "Predictors": [
        "NUMERIC_COLS",
        "OHE_COLS",
        "MULTILABEL_COLS",
        "CATEGORICAL_COLS",
        "CATEGORY_DICT",
        "ALREADY_SPLIT_FLAG_GROUPS",
        "OTHER_FLAGS",
        "BOOLEAN_FLAGS",
        "ALL_PREDICTOR_COLS",
    ],
    "Hyperparameters": ["LR_GRID", "LR_FIXED", "RF_GRID", "RF_FIXED"],
    "Scoring / thresholding": ["INNER_CV_SCORER", "THRESHOLD_RANGE"],
    "Bootstrap": ["BOOTSTRAP_N", "BOOTSTRAP_LOW_PCT", "BOOTSTRAP_HIGH_PCT"],
    "Permutation importance": ["PERM_N_REPEATS"],
}


def create_experiment_dir():
    des_dir = OUTPUT_DIR / f"experiment_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    des_dir.mkdir(parents=True, exist_ok=True)
    return des_dir


def relative_path(path):
    path = Path(path)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT)
        except ValueError:
            return path
    return path


def format_config_value(value):
    if isinstance(value, Path):
        return str(relative_path(value))
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if value.size == 1:
            return pformat(value.tolist(), width=100, sort_dicts=False)
        step = value[1] - value[0]
        return (
            f"{value[0]:.4g} to {value[-1]:.4g} "
            f"(n={value.size}, step={step:.4g})"
        )
    return pformat(value, width=100, sort_dicts=False)


def write_experiment_config(tuned=None, data_path=None, des_dir=None, path=None):
    if path is not None:
        out = Path(path)
    else:
        out_dir = Path(des_dir) if des_dir is not None else create_experiment_dir()
        out = out_dir / "experiment_config.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Experiment configuration",
        f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Experiment directory: {relative_path(out.parent)}",
    ]
    if data_path is not None:
        lines.append(f"Data path: {relative_path(data_path)}")

    for section_name, names in CONFIG_SECTIONS.items():
        lines.extend(["", section_name, "-" * len(section_name)])
        for name in names:
            value = getattr(experiment_config, name)
            lines.append(f"{name}: {format_config_value(value)}")

    if tuned:
        section_name = "Tuned model settings"
        lines.extend(["", section_name, "-" * len(section_name)])
        for name, tm in tuned.items():
            lines.append(f"{name}:")
            lines.append(f"  threshold: {tm.threshold:.4g}")
            lines.append(f"  best_params: {format_config_value(tm.best_params)}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {relative_path(out)}")


def save_csv(df_out, path, index=False):
    df_out.to_csv(path, index=index)
    rel = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
    print(f"Wrote {rel} ({len(df_out)} rows)")


def save_metrics(metrics_df, des_dir):
    save_csv(metrics_df, des_dir / "metrics_summary.csv")


def plot_metrics_summary(metrics_df, des_dir, path=None):
    metric_order = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "precision_approach",
        "recall_approach",
        "pr_auc",
    ]
    metric_labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "balanced_accuracy": "Balanced accuracy",
        "precision_approach": "Precision (approach)",
        "recall_approach": "Recall (approach)",
        "pr_auc": "PR AUC",
    }

    metrics = [m for m in metric_order if m in metrics_df["metric"].unique()]
    if not metrics:
        return

    models = metrics_df["model"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.36

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()
    colors = {"train": "tab:blue", "test": "tab:orange"}

    for ax, metric in zip(axes, metrics):
        metric_df = metrics_df[metrics_df["metric"] == metric]
        y_max = 1.0

        for split_name, offset in (("train", -width / 2), ("test", width / 2)):
            split_df = (
                metric_df[metric_df["split"] == split_name]
                .set_index("model")
                .reindex(models)
            )
            points = split_df["point"].to_numpy(dtype=float)

            yerr = None
            if split_name == "test":
                lows = split_df["ci_low"].to_numpy(dtype=float)
                highs = split_df["ci_high"].to_numpy(dtype=float)
                valid_ci = ~np.isnan(lows) & ~np.isnan(highs)
                lower_err = np.where(valid_ci, np.maximum(points - lows, 0), 0)
                upper_err = np.where(valid_ci, np.maximum(highs - points, 0), 0)
                yerr = np.vstack([lower_err, upper_err])
                if np.any(valid_ci):
                    y_max = max(y_max, np.nanmax(highs))

            bars = ax.bar(
                x + offset,
                points,
                width,
                label=split_name.title(),
                color=colors[split_name],
                yerr=yerr,
                capsize=4 if yerr is not None else 0,
                error_kw={"elinewidth": 1, "capthick": 1},
            )
            y_max = max(y_max, np.nanmax(points))

            for i, bar in enumerate(bars):
                point = points[i]
                if np.isnan(point):
                    continue
                text_y = point
                ax.annotate(
                    f"{point:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, text_y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(metric_labels.get(metric, metric))
        ax.set_xticks(x, models)
        ax.set_ylim(0, min(1.15, y_max + 0.12))
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    axes[0].set_ylabel("Score")
    axes[3].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(path) if path is not None else Path(des_dir) / "metrics_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {relative_path(out)}")


def save_confusion_matrices(eval_results, des_dir):
    for r in eval_results:
        if r.confusion is None:
            continue
        cm_df = pd.DataFrame(
            r.confusion,
            index=["true_avoid", "true_approach"],
            columns=["pred_avoid", "pred_approach"],
        )
        save_csv(cm_df, des_dir / f"confusion_matrix_{r.name.lower()}.csv", index=True)


def save_grid_search_tables(tuned, des_dir):
    for name, tm in tuned.items():
        if tm.cv_results.empty:
            continue
        keep = [c for c in tm.cv_results.columns
                if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
        save_csv(tm.cv_results[keep], des_dir / f"grid_search_{name.lower()}.csv")


def save_feature_influence(influence, des_dir):
    for tag, df_out in influence.items():
        save_csv(df_out, des_dir / f"feature_influence_{tag}.csv")


def plot_confusion_matrices(eval_results, des_dir, path=None):
    fig, axes = plt.subplots(1, len(eval_results), figsize=(4 * len(eval_results), 3.5))
    if len(eval_results) == 1:
        axes = [axes]
    for ax, r in zip(axes, eval_results):
        if r.confusion is None:
            continue
        ax.imshow(r.confusion, cmap="Blues")
        ax.set_title(f"{r.name} (threshold={r.threshold:.2f})")
        ax.set_xticks([0, 1], ["pred avoid", "pred approach"])
        ax.set_yticks([0, 1], ["true avoid", "true approach"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, r.confusion[i, j],
                    ha="center", va="center",
                    color="white" if r.confusion[i, j] > r.confusion.max() / 2 else "black",
                )
    fig.tight_layout()
    out = Path(path) if path is not None else Path(des_dir) / "confusion_matrices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {relative_path(out)}")


def plot_feature_influence(influence, des_dir, top_n=15, path=None):
    lr_top = influence["lr_coefficients_aggregated"].head(top_n).iloc[::-1]
    rf_imp_top = influence["rf_impurity"].head(top_n).iloc[::-1]
    lr_perm_top = influence["lr_permutation"].head(top_n).iloc[::-1]
    rf_top = influence["rf_permutation"].head(top_n).iloc[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.barh(lr_top["feature"], lr_top["abs_coefficient_sum"], color="tab:blue")
    ax1.set_title(f"LR  top {top_n} by sum |coef|")
    ax1.set_xlabel("Sum of |standardised coefficients|")

    ax2.barh(
        rf_imp_top["feature"], rf_imp_top["impurity_importance"],
        color="tab:purple",
    )
    ax2.set_title(f"RF  top {top_n} impurity importance")
    ax2.set_xlabel("Mean decrease in impurity")

    ax3.barh(
        lr_perm_top["feature"], lr_perm_top["perm_importance_mean"],
        color="tab:orange", xerr=lr_perm_top["perm_importance_std"],
    )
    ax3.set_title(f"LR  top {top_n} permutation importance")
    ax3.set_xlabel("Mean drop in average_precision")

    ax4.barh(
        rf_top["feature"], rf_top["perm_importance_mean"],
        color="tab:green", xerr=rf_top["perm_importance_std"],
    )
    ax4.set_title(f"RF  top {top_n} permutation importance")
    ax4.set_xlabel("Mean drop in average_precision")

    fig.tight_layout()
    out = Path(path) if path is not None else Path(des_dir) / "feature_influence_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {relative_path(out)}")


def write_outputs(tuned, eval_results, influence, metrics_df, data_path=None, des_dir=None):
    if des_dir is None:
        des_dir = create_experiment_dir()
    print(f"Saving outputs to {relative_path(des_dir)}")

    write_experiment_config(tuned=tuned, data_path=data_path, des_dir=des_dir)
    save_metrics(metrics_df, des_dir)
    save_confusion_matrices(eval_results, des_dir)
    save_grid_search_tables(tuned, des_dir)
    save_feature_influence(influence, des_dir)

    plot_metrics_summary(metrics_df, des_dir)
    plot_confusion_matrices(eval_results, des_dir)
    plot_feature_influence(influence, des_dir)

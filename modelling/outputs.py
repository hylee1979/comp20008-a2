import matplotlib.pyplot as plt
import pandas as pd

from modelling.config import FIGURE_DIR, PROJECT_ROOT, TABLE_DIR


def save_csv(df_out, path, index=False):
    df_out.to_csv(path, index=index)
    rel = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
    print(f"Wrote {rel} ({len(df_out)} rows)")


def save_metrics(metrics_df):
    save_csv(metrics_df, TABLE_DIR / "metrics_summary.csv")


def save_confusion_matrices(eval_results):
    for r in eval_results:
        if r.confusion is None:
            continue
        cm_df = pd.DataFrame(
            r.confusion,
            index=["true_avoid", "true_approach"],
            columns=["pred_avoid", "pred_approach"],
        )
        save_csv(cm_df, TABLE_DIR / f"confusion_matrix_{r.name.lower()}.csv", index=True)


def save_grid_search_tables(tuned):
    for name, tm in tuned.items():
        if tm.cv_results.empty:
            continue
        keep = [c for c in tm.cv_results.columns
                if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
        save_csv(tm.cv_results[keep], TABLE_DIR / f"grid_search_{name.lower()}.csv")


def save_feature_influence(influence):
    for tag, df_out in influence.items():
        save_csv(df_out, TABLE_DIR / f"feature_influence_{tag}.csv")


def plot_confusion_matrices(eval_results, path=None):
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
    out = path or (FIGURE_DIR / "confusion_matrices.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


def plot_feature_influence(influence, top_n=15, path=None):
    lr_top = influence["lr_coefficients_aggregated"].head(top_n).iloc[::-1]
    rf_top = influence["rf_permutation"].head(top_n).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.barh(lr_top["feature"], lr_top["abs_coefficient_sum"], color="tab:blue")
    ax1.set_title(f"LR  top {top_n} by sum |coef|")
    ax1.set_xlabel("Sum of |standardised coefficients|")

    ax2.barh(
        rf_top["feature"], rf_top["perm_importance_mean"],
        color="tab:green", xerr=rf_top["perm_importance_std"],
    )
    ax2.set_title(f"RF  top {top_n} permutation importance")
    ax2.set_xlabel("Mean drop in average_precision")

    fig.tight_layout()
    out = path or (FIGURE_DIR / "feature_influence_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


def write_outputs(tuned, eval_results, influence, metrics_df):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics_df)
    save_confusion_matrices(eval_results)
    save_grid_search_tables(tuned)
    save_feature_influence(influence)

    plot_confusion_matrices(eval_results)
    plot_feature_influence(influence)

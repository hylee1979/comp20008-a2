from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from modelling.config import PROJECT_ROOT


def _relative_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def plot_pr_curve(y_true, proba, selected_threshold, name, output_dir):
    """Save an out-of-fold PR curve and mark the selected threshold."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    average_precision = average_precision_score(y_true, proba)

    selected_pred = (proba >= selected_threshold).astype(int)
    selected_precision = precision_score(
        y_true, selected_pred, pos_label=1, zero_division=0
    )
    selected_recall = recall_score(
        y_true, selected_pred, pos_label=1, zero_division=0
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pr_curve_{name.lower()}.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        recall,
        precision,
        linewidth=1.8,
        label=f"OOF PR curve (AP = {average_precision:.3f})",
    )
    ax.axhline(
        y_true.mean(),
        color="0.45",
        linestyle=":",
        linewidth=1.2,
        label="positive-class baseline",
    )
    ax.scatter(
        selected_recall,
        selected_precision,
        color="tab:red",
        zorder=3,
        label=f"selected threshold = {selected_threshold:.2f}",
    )

    ax.set_title(f"{name}: out-of-fold PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="both", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Wrote {_relative_path(out)}")
    return out

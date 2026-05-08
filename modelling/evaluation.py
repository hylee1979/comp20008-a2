from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from modelling.config import (
    BOOTSTRAP_HIGH_PCT, BOOTSTRAP_LOW_PCT, BOOTSTRAP_N, RANDOM_STATE,
)


METRIC_KEYS = [
    "accuracy", "macro_f1", "balanced_accuracy",
    "precision_approach", "recall_approach", "pr_auc",
]


@dataclass
class MetricResult:
    point: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")


@dataclass
class EvaluationResult:
    name: str
    threshold: float
    test_metrics: dict = field(default_factory=dict)
    train_metrics: dict = field(default_factory=dict)
    confusion: np.ndarray = None


def compute_metrics(y_true, y_proba, y_pred):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_approach": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_approach": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        out["pr_auc"] = float("nan")
    return out


def predict(pipeline, X, threshold):
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        return proba, preds
    return None, np.asarray(pipeline.predict(X), dtype=int)


def bootstrap_indices(n, n_resamples, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_metrics(y_true, y_proba, y_pred, indices):
    samples = {k: [] for k in METRIC_KEYS}

    for i in range(indices.shape[0]):
        idx = indices[i]
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_proba[idx] if y_proba is not None else None
        if len(np.unique(yt)) < 2: # skip when samples doesn't contain both classes
            continue
        m = compute_metrics(yt, ypr, yp)
        for k in METRIC_KEYS:
            samples[k].append(m[k])

    # point estimate on test set
    point = compute_metrics(y_true, y_proba, y_pred)

    out = {}
    for k in METRIC_KEYS:
        arr = np.array(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            out[k] = MetricResult(point=point[k])
        else:
            out[k] = MetricResult(
                point=point[k],
                ci_low=float(np.percentile(arr, BOOTSTRAP_LOW_PCT)),
                ci_high=float(np.percentile(arr, BOOTSTRAP_HIGH_PCT)),
            )
    return out


def evaluate(name, pipeline, threshold, X_train, y_train, X_test, y_test):
    y_train_arr = y_train.to_numpy(dtype=int) #TODO: do we need this?
    y_test_arr = y_test.to_numpy(dtype=int) #TODO: do we need this?

    proba_test, pred_test = predict(pipeline, X_test, threshold)
    proba_train, pred_train = predict(pipeline, X_train, threshold)

    indices = bootstrap_indices(len(y_test_arr), BOOTSTRAP_N, seed=RANDOM_STATE)
    test_metrics = bootstrap_metrics(y_test_arr, proba_test, pred_test, indices)

    train_point = compute_metrics(y_train_arr, proba_train, pred_train)
    train_metrics = {k: MetricResult(point=v) for k, v in train_point.items()}

    cm = confusion_matrix(y_test_arr, pred_test, labels=[0, 1])

    print(
        f"[{name}] threshold={threshold:.2f}  "
        f"TEST  macro-F1={test_metrics['macro_f1'].point:.3f} "
        f"[{test_metrics['macro_f1'].ci_low:.3f}, {test_metrics['macro_f1'].ci_high:.3f}]  "
        f"bal-acc={test_metrics['balanced_accuracy'].point:.3f}  "
        f"PR-AUC={test_metrics['pr_auc'].point:.3f}"
    )
    print(
        f"[{name}] threshold={threshold:.2f}  "
        f"TRAIN macro-F1={train_metrics['macro_f1'].point:.3f}  "
        f"bal-acc={train_metrics['balanced_accuracy'].point:.3f}  "
        f"PR-AUC={train_metrics['pr_auc'].point:.3f}"
    )
    print(f"[{name}] confusion matrix (rows=true, cols=pred, [0,1]):\n{cm}")

    return EvaluationResult(
        name=name,
        threshold=threshold,
        test_metrics=test_metrics,
        train_metrics=train_metrics,
        confusion=cm,
    )


def evaluate_all(tuned, split):
    return [
        evaluate(
            name=tm.name,
            pipeline=tm.pipeline,
            threshold=tm.threshold,
            X_train=split.X_train,
            y_train=split.y_train,
            X_test=split.X_test,
            y_test=split.y_test,
        )
        for tm in tuned.values()
    ]


def metrics_to_frame(results):
    rows = []
    for r in results:
        for split_name, store in (("test", r.test_metrics), ("train", r.train_metrics)):
            for metric, mr in store.items():
                rows.append({
                    "model": r.name,
                    "split": split_name,
                    "metric": metric,
                    "point": mr.point,
                    "ci_low": mr.ci_low,
                    "ci_high": mr.ci_high,
                    "threshold": r.threshold,
                })
    return pd.DataFrame(rows)

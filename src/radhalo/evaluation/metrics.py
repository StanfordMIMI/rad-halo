"""Detection metrics against adjudicated labels.

LLM judges output binary scores and are thresholded at 0.5. RadCheck outputs
probabilities; its threshold maximises F1 on the evaluated sentences.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def f1_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall), nan=0.0)
    if len(f1) == 0:
        return 0.5
    threshold = thresholds[np.argmax(f1)]
    return float(threshold) if np.isfinite(threshold) else 0.5


def detection_metrics(y_true: np.ndarray, y_score: np.ndarray, is_classifier: bool) -> dict:
    """Precision, recall and F1 (and AUROC/AUPRC for classifiers), rounded to 4 decimals."""
    n_errors = int(y_true.sum())
    auroc = auprc = None
    if is_classifier and 0 < n_errors < len(y_true):
        auroc = round(roc_auc_score(y_true, y_score), 4)
        auprc = round(average_precision_score(y_true, y_score), 4)

    threshold = f1_optimal_threshold(y_true, y_score) if is_classifier else 0.5
    y_pred = (y_score >= threshold).astype(int)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0.0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0.0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0.0), 4),
        "auroc": auroc,
        "auprc": auprc,
        "threshold": round(threshold, 4) if is_classifier else None,
        "num_sentences": len(y_true),
        "num_errors": n_errors,
    }

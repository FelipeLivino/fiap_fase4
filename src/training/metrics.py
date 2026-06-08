"""Metric utilities for binary classification experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


@dataclass(frozen=True)
class BinaryMetrics:
    """Core binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "threshold": self.threshold,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def probabilities_to_predictions(
    probabilities: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Convert probabilities into binary predictions."""
    return (probabilities >= threshold).astype(int)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> BinaryMetrics:
    """Compute binary metrics from labels and positive-class probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = probabilities_to_predictions(y_prob, threshold=threshold)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc = None

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc_roc=auc_roc,
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray] | None:
    """Compute ROC curve points when both classes are present."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

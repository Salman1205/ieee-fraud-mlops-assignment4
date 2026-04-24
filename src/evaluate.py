from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm,
        "threshold": threshold,
    }

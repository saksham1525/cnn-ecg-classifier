"""Metrics computation."""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score


def compute_metrics(y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray], threshold: float = 0.5) -> Dict[str, float]:
    """Compute multi-label classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {
        "f1_macro": f1_score(y_true, y_pred_binary, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred_binary, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred_binary, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred_binary, average="macro", zero_division=0),
    }
    
    # Add AUC and PR-AUC
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_pred, average="macro")
        metrics["pr_auc_macro"] = average_precision_score(y_true, y_pred, average="macro")
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
        metrics["pr_auc_macro"] = 0.0
    
    return metrics

"""Test metrics (plan §5) and the error used by the statistics."""
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

PRIMARY = {"classification": "balanced_accuracy", "regression": "rmse"}
SECONDARY = {"classification": ("log_loss", "macro_f1"), "regression": ("mae",)}


def metrics(y: np.ndarray, pred: np.ndarray, task: str) -> dict:
    """Balanced accuracy (primary), log-loss and macro-F1; or RMSE (primary) and MAE."""
    if task == "classification":
        p = pred / pred.sum(1, keepdims=True)
        label = p.argmax(1)
        with warnings.catch_warnings():  # classes absent from a small test fold
            warnings.simplefilter("ignore")
            return {"balanced_accuracy": float(balanced_accuracy_score(y, label)),
                    "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
                    "macro_f1": float(f1_score(y, label, average="macro", zero_division=0))}
    err = y - pred[:, 0]
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def error(m: dict, task: str) -> float:
    """Plan §5: one minus balanced accuracy, or RMSE (lower is better)."""
    return 1 - m["balanced_accuracy"] if task == "classification" else m["rmse"]

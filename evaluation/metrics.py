"""
evaluation/metrics.py

Evaluation utilities: accuracy, precision, recall, F1, confusion matrices,
and per-class breakdowns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils.config import EMOTIONS, IDX_TO_EMOTION


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def _resolve_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]]) -> Tuple[List[str], List[int]]:
    """Return (names, indices) restricted to classes present in the data."""
    present_indices = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if labels is not None:
        present_names = [labels[i] for i in present_indices if i < len(labels)]
    else:
        present_names = [IDX_TO_EMOTION.get(i, str(i)) for i in present_indices]
    return present_names, present_indices


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Compute a comprehensive metrics dictionary.

    Args:
        y_true: integer labels (0..num_classes-1)
        y_pred: integer predictions
        labels: list of class names (defaults to EMOTIONS)

    Returns:
        dict with accuracy, precision, recall, f1 (macro & weighted),
        per-class f1, and confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present_names, present_indices = _resolve_labels(y_true, y_pred, labels)

    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision_macro":    precision_score(y_true, y_pred, average="macro",    zero_division=0, labels=present_indices),
        "recall_macro":       recall_score(y_true, y_pred,    average="macro",    zero_division=0, labels=present_indices),
        "f1_macro":           f1_score(y_true, y_pred,        average="macro",    zero_division=0, labels=present_indices),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=present_indices),
        "recall_weighted":    recall_score(y_true, y_pred,    average="weighted", zero_division=0, labels=present_indices),
        "f1_weighted":        f1_score(y_true, y_pred,        average="weighted", zero_division=0, labels=present_indices),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=present_indices)
    for name, score in zip(present_names, per_class_f1):
        metrics[f"f1_{name}"] = float(score)

    # Confusion matrix (only present classes)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=present_indices)
    metrics["present_labels"] = present_names

    return metrics


def print_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Evaluation Report",
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present_names, present_indices = _resolve_labels(y_true, y_pred, labels)
    metrics = compute_metrics(y_true, y_pred, labels)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):          {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):       {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):      {metrics['recall_macro']:.4f}")
    print(f"\n  Per-class F1:")
    for lbl in present_names:
        v = metrics.get(f"f1_{lbl}", 0.0)
        bar = "█" * int(v * 20)
        print(f"    {lbl:12s}: {v:.3f}  {bar}")

    print(f"\n  Sklearn Classification Report:\n")
    print(classification_report(
        y_true, y_pred,
        labels=present_indices,
        target_names=present_names,
        zero_division=0,
    ))


def format_metrics_table(results: Dict[str, Dict]) -> str:
    """
    Format a model-comparison table for console output.
    results: {model_name: metrics_dict}
    """
    cols = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    header = f"{'Model':<22}" + "".join(f"{c:>18}" for c in cols)
    sep    = "-" * len(header)

    lines = [sep, header, sep]
    for name, m in results.items():
        row = f"{name:<22}" + "".join(f"{m.get(c, 0):>18.4f}" for c in cols)
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)

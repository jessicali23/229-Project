"""
evaluation/cross_validation.py

Speaker-independent (leave-one-speaker-out) cross-validation and
standard k-fold cross-validation for both baseline and deep models.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

from utils.config import EMOTIONS, EMOTION_TO_IDX, AudioConfig, FeatureConfig, AUDIO_CFG, FEATURE_CFG
from evaluation.metrics import compute_metrics, print_report


# ─── Speaker-Independent CV (RAVDESS) ────────────────────────────────────────

def leave_one_speaker_out_cv(
    samples: List[Tuple[str, str]],
    speaker_ids: List[int],
    train_fn: Callable,
    eval_fn:  Callable,
    verbose:  bool = True,
) -> Dict:
    """
    Leave-one-speaker-out cross-validation.

    Args:
        samples:     list of (path, label)
        speaker_ids: speaker id per sample (same length as samples)
        train_fn:    fn(train_samples) → model
        eval_fn:     fn(model, test_samples) → (y_true, y_pred)

    Returns:
        aggregated metrics dict
    """
    unique_speakers = sorted(set(speaker_ids))
    n_speakers = len(unique_speakers)

    if verbose:
        print(f"\nLeave-One-Speaker-Out CV  —  {n_speakers} speakers")

    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []

    for i, spk in enumerate(unique_speakers):
        train_idx = [j for j, s in enumerate(speaker_ids) if s != spk]
        test_idx  = [j for j, s in enumerate(speaker_ids) if s == spk]

        if not test_idx:
            continue

        train_samples = [samples[j] for j in train_idx]
        test_samples  = [samples[j] for j in test_idx]

        if verbose:
            print(f"  Fold {i+1:2d}/{n_speakers}  speaker={spk:02d}  "
                  f"train={len(train_samples):4d}  test={len(test_samples):3d}")

        model = train_fn(train_samples)
        y_true, y_pred = eval_fn(model, test_samples)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        m = compute_metrics(np.array(y_true), np.array(y_pred))
        for k, v in m.items():
            if isinstance(v, (int, float)):
                fold_metrics[k].append(v)

    # Aggregate
    agg = {k: float(np.mean(v)) for k, v in fold_metrics.items()}
    agg["std_accuracy"]   = float(np.std(fold_metrics.get("accuracy", [0])))
    agg["std_f1_weighted"] = float(np.std(fold_metrics.get("f1_weighted", [0])))

    if verbose:
        print(f"\n  LOSO Results:")
        print(f"    Accuracy   : {agg['accuracy']:.4f} ± {agg['std_accuracy']:.4f}")
        print(f"    F1 weighted: {agg['f1_weighted']:.4f} ± {agg['std_f1_weighted']:.4f}")

    # Also compute metrics on the full pooled predictions
    agg["pooled"] = compute_metrics(np.array(all_true), np.array(all_pred))

    return agg


# ─── Stratified K-Fold CV ─────────────────────────────────────────────────────

def stratified_kfold_cv(
    samples: List[Tuple[str, str]],
    train_fn: Callable,
    eval_fn:  Callable,
    k: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Standard stratified K-fold cross-validation.

    Args:
        samples:   list of (path, label)
        train_fn:  fn(train_samples) → model
        eval_fn:   fn(model, test_samples) → (y_true, y_pred)
        k:         number of folds
    """
    labels = [EMOTION_TO_IDX[lbl] for _, lbl in samples]
    skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    if verbose:
        print(f"\nStratified {k}-Fold Cross-Validation")

    fold_metrics = defaultdict(list)
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(samples, labels)):
        train_s = [samples[i] for i in train_idx]
        test_s  = [samples[i] for i in test_idx]

        if verbose:
            print(f"  Fold {fold+1}/{k}  train={len(train_s):4d}  test={len(test_s):4d}")

        model = train_fn(train_s)
        y_true, y_pred = eval_fn(model, test_s)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        m = compute_metrics(np.array(y_true), np.array(y_pred))
        for key, val in m.items():
            if isinstance(val, (int, float)):
                fold_metrics[key].append(val)

    agg = {k_: float(np.mean(v)) for k_, v in fold_metrics.items()}
    agg["std_accuracy"]    = float(np.std(fold_metrics.get("accuracy", [0])))
    agg["std_f1_weighted"] = float(np.std(fold_metrics.get("f1_weighted", [0])))
    agg["pooled"]          = compute_metrics(np.array(all_true), np.array(all_pred))

    if verbose:
        print(f"\n  K-Fold Results:")
        print(f"    Accuracy   : {agg['accuracy']:.4f} ± {agg['std_accuracy']:.4f}")
        print(f"    F1 weighted: {agg['f1_weighted']:.4f} ± {agg['std_f1_weighted']:.4f}")

    return agg

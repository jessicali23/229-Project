"""
models/baseline.py

Classical ML baselines: Logistic Regression and SVM with handcrafted features.
These also serve as sanity-check benchmarks for deep learning models.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils.config import EMOTIONS


# ─── Model factories ──────────────────────────────────────────────────────────

def build_logistic_regression(
    C: float = 1.0,
    max_iter: int = 2000,
    solver: str = "lbfgs",
    n_jobs: int = -1,
) -> Pipeline:
    """Standardized Logistic Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            multi_class="multinomial",
            n_jobs=n_jobs,
            random_state=42,
        )),
    ])


def build_svm(
    C: float = 10.0,
    gamma: str = "scale",
    kernel: str = "rbf",
) -> Pipeline:
    """Standardized SVM pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            decision_function_shape="ovr",
            probability=True,
            random_state=42,
        )),
    ])


# ─── Hyperparameter search ────────────────────────────────────────────────────

def tune_svm(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Pipeline:
    """Grid-search over SVM hyperparameters."""
    param_grid = {
        "clf__C":     [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.001, 0.01],
    }
    pipe = build_svm()
    gs = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring="f1_weighted",
        n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"Best SVM params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


def tune_lr(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Pipeline:
    """Grid-search over LR hyperparameters."""
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    pipe = build_logistic_regression()
    gs = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring="f1_weighted",
        n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"Best LR params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─── Train / Predict helpers ──────────────────────────────────────────────────

def train_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = False,
    cv: int = 5,
) -> Pipeline:
    if model_name == "svm":
        if tune:
            model = tune_svm(X_train, y_train, cv)
        else:
            model = build_svm()
            model.fit(X_train, y_train)
    elif model_name == "lr":
        if tune:
            model = tune_lr(X_train, y_train, cv)
        else:
            model = build_logistic_regression()
            model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return model


def save_baseline(model: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved baseline model → {path}")


def load_baseline(path: str) -> Pipeline:
    return joblib.load(path)


def predict_baseline(
    model: Pipeline,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_labels, probabilities)."""
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, probs

"""
utils/visualization.py
Plotting helpers for features, training curves, confusion matrices, and results.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from utils.config import EMOTIONS


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = EMOTIONS,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", cm.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    history keys: train_loss, val_loss, train_acc, val_acc
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train", color="steelblue")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"],  label="Val",   color="tomato", linestyle="--")
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    if "train_acc" in history:
        axes[1].plot([a * 100 for a in history["train_acc"]], label="Train", color="steelblue")
    if "val_acc" in history:
        axes[1].plot([a * 100 for a in history["val_acc"]],  label="Val",   color="tomato", linestyle="--")
    axes[1].set_title("Accuracy", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Feature Visualization ────────────────────────────────────────────────────

def plot_waveform_and_spectrogram(
    y: np.ndarray,
    S: np.ndarray,
    sr: int = 22050,
    emotion: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform
    times = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(times, y, color="steelblue", linewidth=0.5)
    axes[0].set_title(f"Waveform  —  emotion: {emotion}", fontsize=12)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # Mel-spectrogram
    img = axes[1].imshow(
        S, aspect="auto", origin="lower", cmap="magma",
        extent=[0, len(y) / sr, 0, S.shape[0]],
    )
    axes[1].set_title("Log-Mel Spectrogram", fontsize=12)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(img, ax=axes[1], label="dB")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_mfcc(
    mfcc: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    emotion: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        mfcc, aspect="auto", origin="lower", cmap="coolwarm",
        extent=[0, mfcc.shape[1] * hop_length / sr, 1, mfcc.shape[0] + 1],
    )
    ax.set_title(f"MFCCs  —  emotion: {emotion}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC coefficient")
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Experiment Comparison Bar Chart ─────────────────────────────────────────

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "f1_weighted",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    results: {model_name: {metric: value, ...}}
    """
    models  = list(results.keys())
    scores  = [results[m].get(metric, 0.0) * 100 for m in models]
    colors  = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    bars = ax.bar(models, scores, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel(f"{metric} (%)", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Per-Emotion F1 Heatmap ───────────────────────────────────────────────────

def plot_per_emotion_scores(
    results: Dict[str, Dict[str, float]],
    emotions: List[str] = EMOTIONS,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    results: {model_name: {emotion: f1, ...}}
    """
    models = list(results.keys())
    matrix = np.array([[results[m].get(e, 0.0) for e in emotions] for m in models])

    fig, ax = plt.subplots(figsize=(max(10, len(emotions) * 1.2), max(4, len(models) * 0.8)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=emotions,
        yticklabels=models,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Per-Emotion F1 Score by Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Model")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

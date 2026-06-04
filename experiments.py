"""
experiments.py

Automated experiment runner: trains all model configurations,
evaluates them, and produces a comparison report.

Usage:
    python experiments.py --data_dir ./data/raw --output_dir ./results
    python experiments.py --models cnn cnn_lstm transformer --epochs 50
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils.config import EMOTIONS
from utils.visualization import plot_model_comparison, plot_per_emotion_scores


# ─── Experiment configs ───────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = [
    {"model": "lr",          "feature_type": "handcrafted", "epochs": 1,   "label": "LogisticRegression"},
    {"model": "svm",         "feature_type": "handcrafted", "epochs": 1,   "label": "SVM-RBF"},
    {"model": "cnn",         "feature_type": "spectrogram", "epochs": 50,  "label": "CNN"},
    {"model": "lstm",        "feature_type": "sequence",    "epochs": 50,  "label": "Bi-LSTM"},
    {"model": "cnn_lstm",    "feature_type": "spectrogram", "epochs": 75,  "label": "CNN+LSTM"},
    {"model": "transformer", "feature_type": "spectrogram", "epochs": 75,  "label": "Transformer"},
]


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_experiment(exp: Dict, data_dir: str, output_dir: str, seed: int = 42) -> Dict:
    import subprocess
    import sys

    model   = exp["model"]
    label   = exp.get("label", model)
    epochs  = exp.get("epochs", 50)
    feature = exp.get("feature_type", "spectrogram")

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")

    metrics_path = os.path.join(output_dir, f"{model}_metrics.json")

    # Skip if already done
    if os.path.exists(metrics_path):
        print(f"  [SKIP] Already completed — loading {metrics_path}")
        with open(metrics_path) as f:
            return json.load(f)

    t0 = time.time()
    cmd = [
        sys.executable, "train.py",
        "--model",         model,
        "--feature_type",  feature,
        "--epochs",        str(epochs),
        "--data_dir",      data_dir,
        "--results_dir",   output_dir,
        "--seed",          str(seed),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] {label} failed (exit {result.returncode})")
        return {}

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["elapsed_seconds"] = elapsed
        print(f"  Completed in {elapsed:.0f}s  |  F1 weighted: {metrics.get('f1_weighted',0):.4f}")
        return metrics

    return {}


def run_all_experiments(
    models: List[str],
    data_dir: str,
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Dict]:
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, Dict] = {}

    configs = [e for e in EXPERIMENT_CONFIGS if e["model"] in models]

    for exp in configs:
        label   = exp.get("label", exp["model"])
        metrics = run_experiment(exp, data_dir, output_dir, seed)
        if metrics:
            all_results[label] = metrics

    return all_results


def generate_report(results: Dict[str, Dict], output_dir: str) -> None:
    if not results:
        print("[WARN] No results to report.")
        return

    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPARISON REPORT")
    print(f"{'='*70}")

    # Summary table
    cols = ["accuracy", "f1_macro", "f1_weighted"]
    header = f"{'Model':<22}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("f1_weighted", 0)):
        row = f"{name:<22}" + "".join(f"{m.get(c,0):.4f}":>18 for c in cols)
        print(row)

    # Best model
    best_name, best_m = max(results.items(), key=lambda x: x[1].get("f1_weighted", 0))
    print(f"\n  Best model: {best_name}")
    print(f"    Accuracy  : {best_m.get('accuracy',0):.4f}")
    print(f"    F1 (macro): {best_m.get('f1_macro',0):.4f}")
    print(f"    F1 (wt.)  : {best_m.get('f1_weighted',0):.4f}")

    # Save comparison plots
    plot_model_comparison(
        results,
        metric="f1_weighted",
        save_path=os.path.join(output_dir, "model_comparison_f1.png"),
    )
    plot_model_comparison(
        results,
        metric="accuracy",
        save_path=os.path.join(output_dir, "model_comparison_acc.png"),
    )

    # Per-emotion scores if available
    per_emo = {}
    for name, m in results.items():
        per_emo[name] = {e: m.get(f"f1_{e}", 0.0) for e in EMOTIONS}
    plot_per_emotion_scores(
        per_emo,
        save_path=os.path.join(output_dir, "per_emotion_f1.png"),
    )

    # Save full JSON
    report_path = os.path.join(output_dir, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report saved → {report_path}")
    print(f"Plots saved to: {output_dir}/")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run all speech emotion recognition experiments")
    p.add_argument("--models",     nargs="+",
                   default=["lr", "svm", "cnn", "lstm", "cnn_lstm", "transformer"],
                   choices=["lr", "svm", "cnn", "lstm", "cnn_lstm", "transformer"])
    p.add_argument("--data_dir",   type=str, default="./data/raw")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    results = run_all_experiments(args.models, args.data_dir, args.output_dir, args.seed)
    generate_report(results, args.output_dir)

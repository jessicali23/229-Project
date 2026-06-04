"""
evaluate.py

Load a trained checkpoint and evaluate it on a test set,
printing a full report and saving plots.

Usage:
    python evaluate.py --model_path ./checkpoints/cnn_lstm_best.pt --data_dir ./data/raw
    python evaluate.py --model_path ./checkpoints/svm_baseline.pkl --model_type svm
"""

import argparse
import json
import os
import numpy as np
import torch

from utils.config import EMOTIONS, AUDIO_CFG, FEATURE_CFG
from data.dataset_loader import load_all_samples, train_val_test_split, make_dataloaders
from evaluation.metrics import compute_metrics, print_report, format_metrics_table
from utils.visualization import plot_confusion_matrix, plot_per_emotion_scores


def evaluate_deep(model_path: str, data_dir: str, split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt.get("config")
    model_name = cfg.model if cfg else "unknown"

    # Rebuild model
    from train import build_model
    model = build_model(model_name, AUDIO_CFG).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_samples = load_all_samples(data_dir)
    _, _, test_s = train_val_test_split(all_samples)
    feature_type = "sequence" if model_name == "lstm" else "spectrogram"
    _, _, test_dl = make_dataloaders(
        test_s, test_s, test_s,
        feature_type=feature_type,
        batch_size=64,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    print_report(y_true, y_pred, EMOTIONS, title=f"Evaluation — {model_name}")

    metrics = compute_metrics(y_true, y_pred)
    os.makedirs("./results", exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title=f"Confusion Matrix — {model_name}",
        save_path=f"./results/{model_name}_eval_cm.png",
    )

    serializable = {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    with open(f"./results/{model_name}_eval_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return metrics


def evaluate_baseline(model_path: str, data_dir: str):
    from models.baseline import load_baseline
    from features.handcrafted import extract_features_from_samples

    model = load_baseline(model_path)
    all_samples = load_all_samples(data_dir)
    _, _, test_s = train_val_test_split(all_samples)

    X_test, y_test = extract_features_from_samples(test_s)
    y_pred = model.predict(X_test)

    print_report(y_test, y_pred, EMOTIONS, title=f"Evaluation — {os.path.basename(model_path)}")

    metrics = compute_metrics(y_test, y_pred)
    os.makedirs("./results", exist_ok=True)
    name = os.path.splitext(os.path.basename(model_path))[0]
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title=f"Confusion Matrix — {name}",
        save_path=f"./results/{name}_eval_cm.png",
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to .pt or .pkl checkpoint")
    p.add_argument("--data_dir",   default="./data/raw")
    p.add_argument("--model_type", default="auto",
                   choices=["auto", "deep", "baseline"])
    args = p.parse_args()

    is_baseline = (
        args.model_path.endswith(".pkl") or args.model_type == "baseline"
    )

    if is_baseline:
        evaluate_baseline(args.model_path, args.data_dir)
    else:
        evaluate_deep(args.model_path, args.data_dir)

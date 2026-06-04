"""
train.py

Main training entry point. Supports all model types:
    baseline (SVM, LR) — classical ML on handcrafted features
    cnn, lstm, cnn_lstm, transformer — deep learning on spectrograms/sequences

Usage:
    python train.py --model cnn_lstm --epochs 75 --batch_size 32
    python train.py --model svm --feature_type handcrafted
"""

import argparse
import os
import sys
import time
import random
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.config import (
    TrainConfig, AudioConfig, FeatureConfig,
    CNNConfig, LSTMConfig, CNNLSTMConfig, TransformerConfig,
    AUDIO_CFG, FEATURE_CFG, CNN_CFG, LSTM_CFG, CNN_LSTM_CFG, TRANSFORMER_CFG,
    EMOTIONS,
)
from data.dataset_loader import load_all_samples, train_val_test_split, make_dataloaders
from evaluation.metrics import compute_metrics, print_report
from utils.visualization import plot_training_curves, plot_confusion_matrix


# ─── Seeding ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(model_name: str, audio_cfg: AudioConfig) -> nn.Module:
    n_mels   = audio_cfg.n_mels
    n_frames = 1 + audio_cfg.samples // audio_cfg.hop_length

    if model_name == "cnn":
        from models.cnn import build_cnn
        return build_cnn(CNN_CFG)

    elif model_name == "lstm":
        from models.lstm import build_lstm
        input_size = AUDIO_CFG.n_mfcc * 3   # mfcc + delta + delta2
        return build_lstm(input_size, LSTM_CFG)

    elif model_name == "cnn_lstm":
        from models.cnn_lstm import build_cnn_lstm
        return build_cnn_lstm(CNN_LSTM_CFG, n_mels=n_mels)

    elif model_name == "transformer":
        from models.transformer import build_transformer
        return build_transformer(TRANSFORMER_CFG, n_mels=n_mels, n_frames=n_frames)

    else:
        raise ValueError(f"Unknown deep model: {model_name}")


# ─── Deep Learning Training Loop ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss    += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_samples += len(yb)

    return {
        "loss": total_loss / total_samples,
        "acc":  total_correct / total_samples,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits  = model(xb)
        loss    = criterion(logits, yb)

        total_loss    += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_samples += len(yb)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    return {
        "loss":   total_loss / total_samples,
        "acc":    total_correct / total_samples,
        "preds":  np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_deep_model(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading samples …")
    all_samples = load_all_samples(cfg.data_dir)
    train_s, val_s, test_s = train_val_test_split(
        all_samples, cfg.val_split, cfg.test_split, cfg.seed
    )

    feature_type = "sequence" if cfg.model == "lstm" else "spectrogram"
    train_dl, val_dl, test_dl = make_dataloaders(
        train_s, val_s, test_s,
        feature_type=feature_type,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg.model, AUDIO_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg.model}  ({n_params:,} parameters)")

    # Class weights to handle imbalance
    from collections import Counter
    label_counts = Counter(FEATURE_CFG.feature_type for _, _ in train_s)
    label_arr = np.array([1 for _ in train_s])   # placeholder
    # Simple approach: uniform weights unless dataset is very imbalanced
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    elif cfg.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.model))

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_f1   = 0.0
    best_ckpt     = os.path.join(cfg.checkpoint_dir, f"{cfg.model}_best.pt")
    patience_left = cfg.patience

    print(f"\nTraining for up to {cfg.epochs} epochs  (early stopping patience={cfg.patience})\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_m = train_epoch(model, train_dl, optimizer, criterion, device, cfg.grad_clip)
        val_m   = eval_epoch(model, val_dl, criterion, device)

        val_metrics = compute_metrics(val_m["labels"], val_m["preds"])
        val_f1      = val_metrics["f1_weighted"]

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_m["loss"])
        else:
            scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["acc"])
        history["val_acc"].append(val_m["acc"])

        writer.add_scalar("Loss/train",   train_m["loss"], epoch)
        writer.add_scalar("Loss/val",     val_m["loss"],   epoch)
        writer.add_scalar("Acc/train",    train_m["acc"],  epoch)
        writer.add_scalar("Acc/val",      val_m["acc"],    epoch)
        writer.add_scalar("F1/val",       val_f1,          epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['acc']:.3f}  "
            f"val_loss={val_m['loss']:.4f}  val_acc={val_m['acc']:.3f}  "
            f"val_f1={val_f1:.3f}  {elapsed:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_left = cfg.patience
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1,
                "config": cfg,
            }, best_ckpt)
            print(f"  ✓ New best val F1={val_f1:.4f} — checkpoint saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    writer.close()

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_m = eval_epoch(model, test_dl, criterion, device)
    print_report(test_m["labels"], test_m["preds"], EMOTIONS, title=f"Test Results — {cfg.model}")

    # Save plots
    plot_training_curves(
        history,
        title=f"Training — {cfg.model}",
        save_path=os.path.join(cfg.results_dir, f"{cfg.model}_training_curves.png"),
    )
    test_metrics = compute_metrics(test_m["labels"], test_m["preds"])
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title=f"Confusion Matrix — {cfg.model}",
        save_path=os.path.join(cfg.results_dir, f"{cfg.model}_confusion_matrix.png"),
    )

    # Save metrics JSON
    results_path = os.path.join(cfg.results_dir, f"{cfg.model}_metrics.json")
    serializable = {k: v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved → {results_path}")


# ─── Baseline Training ────────────────────────────────────────────────────────

def train_baseline_model(cfg: TrainConfig) -> None:
    from models.baseline import train_baseline, save_baseline
    from features.handcrafted import extract_features_from_samples

    print("\nLoading samples …")
    all_samples = load_all_samples(cfg.data_dir)
    train_s, val_s, test_s = train_val_test_split(
        all_samples, cfg.val_split, cfg.test_split, cfg.seed
    )

    print("\nExtracting handcrafted features …")
    X_train, y_train = extract_features_from_samples(train_s)
    X_val,   y_val   = extract_features_from_samples(val_s)
    X_test,  y_test  = extract_features_from_samples(test_s)

    print(f"\nFeature shape: {X_train.shape}")

    model = train_baseline(cfg.model, X_train, y_train, tune=True)

    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.model}_baseline.pkl")
    save_baseline(model, ckpt_path)

    preds_val  = model.predict(X_val)
    preds_test = model.predict(X_test)

    print_report(y_val,  preds_val,  EMOTIONS, title=f"Val Results — {cfg.model}")
    print_report(y_test, preds_test, EMOTIONS, title=f"Test Results — {cfg.model}")

    test_metrics = compute_metrics(y_test, preds_test)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title=f"Confusion Matrix — {cfg.model}",
        save_path=os.path.join(cfg.results_dir, f"{cfg.model}_confusion_matrix.png"),
    )

    results_path = os.path.join(cfg.results_dir, f"{cfg.model}_metrics.json")
    serializable = {k: v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved → {results_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a speech emotion recognition model")
    p.add_argument("--model",       type=str, default="cnn_lstm",
                   choices=["svm", "lr", "cnn", "lstm", "cnn_lstm", "transformer"])
    p.add_argument("--feature_type", type=str, default="spectrogram",
                   choices=["handcrafted", "spectrogram", "sequence"])
    p.add_argument("--data_dir",     type=str, default="./data/raw")
    p.add_argument("--epochs",       type=int, default=75)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--scheduler",    type=str, default="cosine",
                   choices=["cosine", "plateau", "step"])
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--results_dir",    type=str, default="./results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model=args.model,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        scheduler=args.scheduler,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
    )

    if args.model in ("svm", "lr"):
        train_baseline_model(cfg)
    else:
        train_deep_model(cfg)

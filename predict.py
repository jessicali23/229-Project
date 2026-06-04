"""
predict.py

Run inference on a single audio file using a trained checkpoint.

Usage:
    python predict.py --audio_path path/to/audio.wav --model_path ./checkpoints/cnn_lstm_best.pt
    python predict.py --audio_path path/to/audio.wav --model_path ./checkpoints/svm_baseline.pkl
"""

import argparse
import os
import sys
import numpy as np
import torch

from utils.config import EMOTIONS, IDX_TO_EMOTION, AUDIO_CFG, FEATURE_CFG
from utils.audio_utils import load_audio


def predict_deep(audio_path: str, model_path: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt.get("config")
    model_name = cfg.model if cfg else "cnn_lstm"

    from train import build_model
    model = build_model(model_name, AUDIO_CFG).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y = load_audio(audio_path, AUDIO_CFG)

    if model_name == "lstm":
        import librosa
        mfcc   = librosa.feature.mfcc(y=y, sr=AUDIO_CFG.sample_rate, n_mfcc=AUDIO_CFG.n_mfcc)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        seq    = np.concatenate([mfcc, delta, delta2], axis=0).T
        seq    = (seq - seq.mean(0)) / (seq.std(0) + 1e-8)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        from features.spectrogram import extract_spectrogram
        spec = extract_spectrogram(y, AUDIO_CFG)
        x    = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    predicted_idx   = int(probs.argmax())
    predicted_label = IDX_TO_EMOTION[predicted_idx]

    return {
        "predicted_emotion": predicted_label,
        "confidence":        float(probs[predicted_idx]),
        "all_probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(probs)},
    }


def predict_baseline(audio_path: str, model_path: str) -> dict:
    from models.baseline import load_baseline
    from features.handcrafted import extract_handcrafted_features

    model = load_baseline(model_path)
    y     = load_audio(audio_path, AUDIO_CFG)
    feat  = extract_handcrafted_features(y, AUDIO_CFG, FEATURE_CFG).reshape(1, -1)

    pred_idx = int(model.predict(feat)[0])
    probs    = model.predict_proba(feat).squeeze(0)

    predicted_label = IDX_TO_EMOTION.get(pred_idx, f"class_{pred_idx}")
    return {
        "predicted_emotion": predicted_label,
        "confidence":        float(probs[pred_idx]),
        "all_probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(probs)},
    }


def predict(audio_path: str, model_path: str) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith(".pkl"):
        return predict_baseline(audio_path, model_path)
    else:
        return predict_deep(audio_path, model_path)


def _print_result(result: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  Predicted emotion : {result['predicted_emotion'].upper()}")
    print(f"  Confidence        : {result['confidence']*100:.1f}%")
    print(f"\n  All probabilities:")
    sorted_probs = sorted(result["all_probabilities"].items(), key=lambda x: -x[1])
    for emo, p in sorted_probs:
        bar = "█" * int(p * 30)
        print(f"    {emo:12s}: {p*100:5.1f}%  {bar}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict emotion from a single audio file")
    p.add_argument("--audio_path",  required=True, help="Path to .wav audio file")
    p.add_argument("--model_path",  required=True, help="Path to trained model (.pt or .pkl)")
    args = p.parse_args()

    result = predict(args.audio_path, args.model_path)
    _print_result(result)

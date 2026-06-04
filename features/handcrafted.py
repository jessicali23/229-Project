"""
features/handcrafted.py

Extracts a rich, fixed-length feature vector from a raw audio signal.
Features include: MFCCs + deltas, chroma, ZCR, RMS energy,
spectral contrast, and tonnetz.
"""

import numpy as np
import librosa
from typing import Optional

from utils.config import AudioConfig, FeatureConfig, AUDIO_CFG, FEATURE_CFG


# ─── Per-feature extractors ───────────────────────────────────────────────────

def _stat(x: np.ndarray) -> np.ndarray:
    """Return [mean, std] across time axis (axis=-1)."""
    return np.concatenate([x.mean(axis=-1), x.std(axis=-1)], axis=-1)


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    include_delta: bool = True,
    include_delta2: bool = True,
) -> np.ndarray:
    """MFCCs and optional delta/delta-delta; returns (mean+std) vector."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    parts = [_stat(mfcc)]
    if include_delta:
        d = librosa.feature.delta(mfcc)
        parts.append(_stat(d))
    if include_delta2:
        d2 = librosa.feature.delta(mfcc, order=2)
        parts.append(_stat(d2))
    return np.concatenate(parts)


def extract_chroma(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    """Chroma STFT (12 bins) → mean + std → 24-dim."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return _stat(chroma)


def extract_zcr(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Zero-crossing rate → mean + std → 2-dim."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    return _stat(zcr)


def extract_rms(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Root Mean Square Energy → mean + std → 2-dim."""
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    return _stat(rms)


def extract_spectral_contrast(
    y: np.ndarray, sr: int, n_fft: int, hop_length: int
) -> np.ndarray:
    """Spectral contrast (7 bands) → mean + std → 14-dim."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return _stat(contrast)


def extract_tonnetz(y: np.ndarray, sr: int) -> np.ndarray:
    """Tonal centroid features (tonnetz) → mean + std → 12-dim."""
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    return _stat(tonnetz)


def extract_spectral_features(
    y: np.ndarray, sr: int, n_fft: int, hop_length: int
) -> np.ndarray:
    """Spectral centroid, bandwidth, rolloff → mean + std → 6-dim."""
    centroid  = _stat(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    bandwidth = _stat(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    rolloff   = _stat(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
    return np.concatenate([centroid, bandwidth, rolloff])


# ─── Unified extractor ────────────────────────────────────────────────────────

def extract_handcrafted_features(
    y: np.ndarray,
    audio_cfg: AudioConfig = AUDIO_CFG,
    feature_cfg: FeatureConfig = FEATURE_CFG,
) -> np.ndarray:
    """
    Extract all configured features from a waveform array.

    Returns:
        feat: float32 1-D numpy array (fixed length regardless of audio duration)
    """
    sr          = audio_cfg.sample_rate
    n_fft       = audio_cfg.n_fft
    hop_length  = audio_cfg.hop_length

    parts = []

    # MFCCs (always included)
    parts.append(
        extract_mfcc(
            y, sr,
            n_mfcc=feature_cfg.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            include_delta=feature_cfg.include_delta,
            include_delta2=feature_cfg.include_delta2,
        )
    )

    if feature_cfg.include_chroma:
        parts.append(extract_chroma(y, sr, n_fft, hop_length))

    if feature_cfg.include_zcr:
        parts.append(extract_zcr(y, hop_length))

    if feature_cfg.include_rms:
        parts.append(extract_rms(y, n_fft, hop_length))

    if feature_cfg.include_spectral_contrast:
        parts.append(extract_spectral_contrast(y, sr, n_fft, hop_length))

    if feature_cfg.include_tonnetz:
        parts.append(extract_tonnetz(y, sr))

    # Always add spectral shape features
    parts.append(extract_spectral_features(y, sr, n_fft, hop_length))

    feat = np.concatenate(parts).astype(np.float32)
    return feat


# ─── Batch extraction (for classical ML) ─────────────────────────────────────

def extract_features_from_samples(
    samples,  # list of (path, label)
    audio_cfg: AudioConfig = AUDIO_CFG,
    feature_cfg: FeatureConfig = FEATURE_CFG,
    verbose: bool = True,
):
    """
    Extract feature matrix X and label array y from a sample list.
    Returns:
        X: (N, D) float32 ndarray
        y: (N,)  int32 ndarray
    """
    from utils.audio_utils import load_audio
    from utils.config import EMOTION_TO_IDX
    from tqdm import tqdm

    X_list, y_list = [], []
    iterator = tqdm(samples, desc="Extracting features") if verbose else samples

    for path, label in iterator:
        try:
            wav = load_audio(path, audio_cfg)
            feat = extract_handcrafted_features(wav, audio_cfg, feature_cfg)
            X_list.append(feat)
            y_list.append(EMOTION_TO_IDX[label])
        except Exception as e:
            if verbose:
                print(f"  [WARN] Skipping {path}: {e}")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y

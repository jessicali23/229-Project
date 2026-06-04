"""
utils/audio_utils.py
Audio loading, preprocessing, and augmentation utilities.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import warnings

from utils.config import AudioConfig, AUDIO_CFG

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_audio(
    path: str,
    cfg: AudioConfig = AUDIO_CFG,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load an audio file, resample to cfg.sample_rate, trim silence,
    and pad/truncate to cfg.duration seconds.

    Returns:
        y: float32 array of shape (cfg.samples,)
    """
    try:
        y, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    except Exception:
        y, sr = sf.read(path, always_2d=False)
        if sr != cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=cfg.top_db)

    # Pad or truncate
    target = cfg.samples
    if len(y) < target:
        pad = target - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    else:
        y = y[:target]

    if normalize:
        y = normalize_audio(y)

    return y.astype(np.float32)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_audio(y: np.ndarray, cfg: AudioConfig = AUDIO_CFG, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply a random combination of augmentation transforms.
    Each transform is applied with probability cfg.augment_prob.
    """
    rng = np.random.default_rng(seed)

    def maybe(fn, *args, **kwargs):
        if rng.random() < cfg.augment_prob:
            return fn(*args, **kwargs)
        return y  # closure over current y – rebind below

    # Gaussian noise
    if rng.random() < cfg.augment_prob:
        noise_amp = rng.uniform(0.001, 0.015)
        y = y + noise_amp * rng.standard_normal(len(y)).astype(np.float32)

    # Time stretch (no pitch change)
    if rng.random() < cfg.augment_prob:
        rate = rng.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)

    # Pitch shift
    if rng.random() < cfg.augment_prob:
        steps = rng.uniform(-3.0, 3.0)
        y = librosa.effects.pitch_shift(y, sr=cfg.sample_rate, n_steps=steps)

    # Random gain
    if rng.random() < cfg.augment_prob:
        gain = rng.uniform(0.7, 1.3)
        y = y * gain

    # Shift in time
    if rng.random() < cfg.augment_prob:
        shift = int(rng.uniform(-0.1, 0.1) * len(y))
        y = np.roll(y, shift)

    # Re-pad / re-truncate after augmentations that change length
    target = cfg.samples
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]

    return normalize_audio(y).astype(np.float32)


# ─── Spectrogram Helpers ──────────────────────────────────────────────────────

def compute_melspectrogram(
    y: np.ndarray,
    cfg: AudioConfig = AUDIO_CFG,
    db_scale: bool = True,
) -> np.ndarray:
    """
    Compute (log) Mel-spectrogram.

    Returns:
        S: float32 array of shape (n_mels, time_frames)
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    if db_scale:
        S = librosa.power_to_db(S, ref=np.max)
    return S.astype(np.float32)


def normalize_spectrogram(S: np.ndarray) -> np.ndarray:
    """Normalize spectrogram to zero mean / unit std per channel."""
    mean = S.mean()
    std  = S.std() + 1e-8
    return ((S - mean) / std).astype(np.float32)

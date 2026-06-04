"""
features/spectrogram.py

Spectrogram-based feature extraction for CNN / CNN+LSTM / Transformer models.
Returns a (1, n_mels, T) float32 array suitable for PyTorch.
"""

import numpy as np
import librosa
from utils.config import AudioConfig, AUDIO_CFG


def extract_spectrogram(
    y: np.ndarray,
    cfg: AudioConfig = AUDIO_CFG,
    channels: int = 1,
) -> np.ndarray:
    """
    Compute a normalized log-Mel spectrogram.

    Args:
        y:        Waveform float32 array (already trimmed/padded to cfg.samples)
        cfg:      AudioConfig
        channels: 1 = mono, 3 = RGB-like (log-mel + delta + delta2)

    Returns:
        spec: float32 array of shape (channels, n_mels, T)
    """
    # Log-Mel spectrogram
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
    S_db = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)

    if channels == 1:
        spec = S_db[np.newaxis, :]              # (1, n_mels, T)
    else:
        # 3-channel: log-mel, delta, delta2 (like an RGB image)
        delta  = librosa.feature.delta(S_db)
        delta2 = librosa.feature.delta(S_db, order=2)
        spec = np.stack([S_db, delta, delta2], axis=0)  # (3, n_mels, T)

    # Normalize each channel independently
    for c in range(spec.shape[0]):
        mu  = spec[c].mean()
        std = spec[c].std() + 1e-8
        spec[c] = (spec[c] - mu) / std

    return spec.astype(np.float32)


def extract_stft_spectrogram(
    y: np.ndarray,
    cfg: AudioConfig = AUDIO_CFG,
) -> np.ndarray:
    """
    Plain STFT magnitude spectrogram.
    Returns (1, 1+n_fft//2, T) float32.
    """
    D = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    D_db = (D_db - D_db.mean()) / (D_db.std() + 1e-8)
    return D_db[np.newaxis].astype(np.float32)


def resize_spectrogram(
    spec: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """
    Pad or truncate the time axis to target_frames.
    spec shape: (C, n_mels, T)
    """
    T = spec.shape[-1]
    if T < target_frames:
        pad = target_frames - T
        spec = np.pad(spec, ((0,0),(0,0),(0,pad)), mode="constant")
    else:
        spec = spec[..., :target_frames]
    return spec

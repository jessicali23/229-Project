"""
utils/config.py
Central configuration for all hyperparameters, paths, and constants.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os

# ─── Emotion Labels ───────────────────────────────────────────────────────────

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

# RAVDESS emotion codes → unified labels
RAVDESS_MAP = {
    "01": "neutral",
    "02": "neutral",   # calm → neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised",
}

# TESS folder prefixes → unified labels
TESS_MAP = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "ps":       "surprised",
    "sad":      "sad",
}

# EmoDB codes → unified labels
EMODB_MAP = {
    "W": "angry",
    "L": "boredom",    # mapped to neutral
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}
EMODB_UNIFIED = {
    "W": "angry",
    "L": "neutral",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


# ─── Audio Parameters ─────────────────────────────────────────────────────────

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    duration: float = 3.0             # seconds – clips are trimmed/padded to this
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    fmin: float = 0.0
    fmax: float = 8000.0
    top_db: float = 60.0              # silence threshold
    augment: bool = True
    augment_prob: float = 0.5

    @property
    def samples(self) -> int:
        return int(self.sample_rate * self.duration)

    @property
    def spectrogram_shape(self) -> Tuple[int, int]:
        """(n_mels, time_frames)"""
        frames = 1 + self.samples // self.hop_length
        return (self.n_mels, frames)


# ─── Feature Config ───────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    feature_type: str = "handcrafted"   # "handcrafted" | "spectrogram" | "both"
    # Handcrafted dimensions
    n_mfcc: int = 40
    include_delta: bool = True
    include_delta2: bool = True
    include_chroma: bool = True
    include_zcr: bool = True
    include_rms: bool = True
    include_spectral_contrast: bool = True
    include_tonnetz: bool = True

    @property
    def handcrafted_dim(self) -> int:
        """Estimate of flattened handcrafted feature vector size."""
        dim = self.n_mfcc
        if self.include_delta:
            dim += self.n_mfcc
        if self.include_delta2:
            dim += self.n_mfcc
        if self.include_chroma:
            dim += 12
        if self.include_zcr:
            dim += 1
        if self.include_rms:
            dim += 1
        if self.include_spectral_contrast:
            dim += 7
        if self.include_tonnetz:
            dim += 6
        return dim  # mean+std so ×2
        return dim * 2


# ─── Model Configs ────────────────────────────────────────────────────────────

@dataclass
class CNNConfig:
    in_channels: int = 1
    base_filters: int = 32
    num_blocks: int = 4
    dropout: float = 0.3
    num_classes: int = len(EMOTIONS)


@dataclass
class LSTMConfig:
    input_size: int = 40          # will be set from feature dim
    hidden_size: int = 256
    num_layers: int = 3
    bidirectional: bool = True
    dropout: float = 0.3
    num_classes: int = len(EMOTIONS)


@dataclass
class CNNLSTMConfig:
    # CNN part
    cnn_base_filters: int = 32
    cnn_num_blocks: int = 3
    cnn_dropout: float = 0.2
    # LSTM part
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.3
    num_classes: int = len(EMOTIONS)


@dataclass
class TransformerConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    patch_size: int = 16          # for spectrogram patching
    num_classes: int = len(EMOTIONS)


# ─── Training Config ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    model: str = "cnn_lstm"        # svm | lr | cnn | lstm | cnn_lstm | transformer
    feature_type: str = "spectrogram"
    batch_size: int = 32
    epochs: int = 75
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"      # "cosine" | "step" | "plateau"
    patience: int = 15             # early stopping
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    # Paths
    data_dir: str = "./data/raw"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# ─── Singletons ───────────────────────────────────────────────────────────────

AUDIO_CFG   = AudioConfig()
FEATURE_CFG = FeatureConfig()
CNN_CFG     = CNNConfig()
LSTM_CFG    = LSTMConfig()
CNN_LSTM_CFG = CNNLSTMConfig()
TRANSFORMER_CFG = TransformerConfig()

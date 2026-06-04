"""
data/dataset_loader.py

Unified loading for RAVDESS, TESS, and EmoDB.
Returns lists of (file_path, emotion_label) tuples and
provides PyTorch Dataset wrappers for both handcrafted features
and spectrogram inputs.
"""

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from utils.config import (
    RAVDESS_MAP, TESS_MAP, EMODB_UNIFIED,
    EMOTIONS, EMOTION_TO_IDX, AudioConfig, FeatureConfig,
    AUDIO_CFG, FEATURE_CFG,
)
from utils.audio_utils import load_audio, augment_audio, compute_melspectrogram, normalize_spectrogram
from features.handcrafted import extract_handcrafted_features
from features.spectrogram import extract_spectrogram


# ─── Raw File Scanners ────────────────────────────────────────────────────────

def scan_ravdess(data_dir: str) -> List[Tuple[str, str]]:
    """
    RAVDESS filename format:
        03-01-{emotion:02d}-{intensity}-{statement}-{rep}-{actor:02d}.wav
    """
    samples = []
    root = Path(data_dir)
    for wav in root.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 7:
            continue
        if parts[0] != "03":          # audio-only (not video)
            continue
        emo_code = parts[2]
        label = RAVDESS_MAP.get(emo_code)
        if label and label in EMOTION_TO_IDX:
            samples.append((str(wav), label))
    return samples


def scan_tess(data_dir: str) -> List[Tuple[str, str]]:
    """
    TESS folder names contain the emotion word, e.g. YAF_angry, OAF_happy.
    """
    samples = []
    root = Path(data_dir)
    for wav in root.rglob("*.wav"):
        folder = wav.parent.name.lower()
        # Extract emotion suffix after underscore, or match full folder name
        parts = folder.split("_")
        emo_key = parts[-1] if len(parts) > 1 else folder
        label = TESS_MAP.get(emo_key)
        if label and label in EMOTION_TO_IDX:
            samples.append((str(wav), label))
    return samples


def scan_emodb(data_dir: str) -> List[Tuple[str, str]]:
    """
    EmoDB filename: {speaker:02d}{text:02d}{emotion}{version:02d}.wav
    Emotion is the 6th character (index 5).
    """
    samples = []
    root = Path(data_dir)
    for wav in root.rglob("*.wav"):
        name = wav.stem
        if len(name) < 6:
            continue
        emo_code = name[5].upper()
        label = EMODB_UNIFIED.get(emo_code)
        if label and label in EMOTION_TO_IDX:
            samples.append((str(wav), label))
    return samples


# ─── Unified Loader ───────────────────────────────────────────────────────────

_SCANNERS = {
    "ravdess": scan_ravdess,
    "tess":    scan_tess,
    "emodb":   scan_emodb,
}


def load_all_samples(
    data_dir: str,
    datasets: Optional[List[str]] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Recursively finds all datasets in data_dir by subfolder name.
    Returns list of (path, label) pairs.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    if datasets is None:
        datasets = list(_SCANNERS.keys())

    all_samples: List[Tuple[str, str]] = []
    for name in datasets:
        sub = root / name
        if not sub.exists():
            print(f"[WARN] Dataset folder not found: {sub}")
            continue
        scanner = _SCANNERS[name]
        found = scanner(str(sub))
        print(f"  {name}: {len(found)} samples")
        all_samples.extend(found)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    # Print class distribution
    from collections import Counter
    dist = Counter(label for _, label in all_samples)
    print(f"\nTotal: {len(all_samples)} samples across {len(dist)} emotions")
    for emo in EMOTIONS:
        print(f"    {emo:12s}: {dist.get(emo, 0)}")

    return all_samples


def train_val_test_split(
    samples: List[Tuple[str, str]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    data = list(samples)
    rng.shuffle(data)
    n = len(data)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    test  = data[:n_test]
    val   = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]
    print(f"Split → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def speaker_ids_ravdess(samples: List[Tuple[str, str]]) -> List[int]:
    """Extract speaker IDs from RAVDESS file paths."""
    ids = []
    for path, _ in samples:
        m = re.search(r"-(\d{2})\.wav$", path)
        ids.append(int(m.group(1)) if m else -1)
    return ids


# ─── PyTorch Datasets ─────────────────────────────────────────────────────────

class HandcraftedDataset(Dataset):
    """Dataset returning (feature_vector, label) for classical ML / LSTM."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        audio_cfg: AudioConfig = AUDIO_CFG,
        feature_cfg: FeatureConfig = FEATURE_CFG,
        augment: bool = False,
    ):
        self.samples = samples
        self.audio_cfg = audio_cfg
        self.feature_cfg = feature_cfg
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y = load_audio(path, self.audio_cfg)
        if self.augment and self.audio_cfg.augment:
            y = augment_audio(y, self.audio_cfg)

        feat = extract_handcrafted_features(y, self.audio_cfg, self.feature_cfg)
        label_idx = EMOTION_TO_IDX[label]

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


class SpectrogramDataset(Dataset):
    """Dataset returning (spectrogram_tensor, label) for CNN / CNN+LSTM / Transformer."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        audio_cfg: AudioConfig = AUDIO_CFG,
        augment: bool = False,
    ):
        self.samples = samples
        self.audio_cfg = audio_cfg
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y = load_audio(path, self.audio_cfg)
        if self.augment and self.audio_cfg.augment:
            y = augment_audio(y, self.audio_cfg)

        spec = extract_spectrogram(y, self.audio_cfg)   # (1, n_mels, T)
        label_idx = EMOTION_TO_IDX[label]

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


class SequenceDataset(Dataset):
    """Dataset returning (MFCC time series, label) for LSTM with sequential modeling."""

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        audio_cfg: AudioConfig = AUDIO_CFG,
        augment: bool = False,
    ):
        self.samples = samples
        self.audio_cfg = audio_cfg
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import librosa
        path, label = self.samples[idx]
        y = load_audio(path, self.audio_cfg)
        if self.augment and self.audio_cfg.augment:
            y = augment_audio(y, self.audio_cfg)

        cfg = self.audio_cfg
        mfcc = librosa.feature.mfcc(
            y=y, sr=cfg.sample_rate, n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft, hop_length=cfg.hop_length
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        # (T, 3*n_mfcc)  — time-major for LSTM
        seq = np.concatenate([mfcc, delta, delta2], axis=0).T
        seq = (seq - seq.mean(0)) / (seq.std(0) + 1e-8)

        label_idx = EMOTION_TO_IDX[label]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


# ─── DataLoader Factories ─────────────────────────────────────────────────────

def make_dataloaders(
    train_samples, val_samples, test_samples,
    feature_type: str = "spectrogram",
    batch_size: int = 32,
    num_workers: int = 4,
    audio_cfg: AudioConfig = AUDIO_CFG,
    feature_cfg: FeatureConfig = FEATURE_CFG,
):
    """Create train / val / test DataLoaders for the requested feature type."""

    def _ds(samples, augment):
        if feature_type == "spectrogram":
            return SpectrogramDataset(samples, audio_cfg, augment)
        elif feature_type == "sequence":
            return SequenceDataset(samples, audio_cfg, augment)
        else:
            return HandcraftedDataset(samples, audio_cfg, feature_cfg, augment)

    train_ds = _ds(train_samples, augment=True)
    val_ds   = _ds(val_samples,   augment=False)
    test_ds  = _ds(test_samples,  augment=False)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )

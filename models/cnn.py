"""
models/cnn.py

Convolutional Neural Network for emotion classification from Log-Mel spectrograms.
Architecture: stacked Conv blocks with BatchNorm + residual connections,
followed by global average pooling and a classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils.config import CNNConfig, CNN_CFG, EMOTIONS


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d → BN → ReLU → Conv2d → BN + residual shortcut → ReLU → MaxPool
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout2d(dropout)

        # 1×1 projection shortcut if channels differ
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + F.adaptive_avg_pool2d(residual, x.shape[2:]))
        x = self.drop(x)
        x = self.pool(x)
        return x


class AttentionPool(nn.Module):
    """Soft attention-weighted global pooling over time axis."""
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → collapse H then attend over W
        x = x.mean(dim=2)                          # (B, C, W)
        weights = torch.softmax(self.attn(x), dim=-1)  # (B, 1, W)
        out = (x * weights).sum(dim=-1)            # (B, C)
        return out


# ─── CNN model ────────────────────────────────────────────────────────────────

class EmotionCNN(nn.Module):
    """
    CNN for speech emotion recognition.
    Input: (B, 1, n_mels, T)  — single-channel log-Mel spectrogram
    """

    def __init__(self, cfg: CNNConfig = CNN_CFG):
        super().__init__()
        self.cfg = cfg

        # Progressive filter sizes: base, base*2, base*4, base*8
        filters = [cfg.in_channels] + [
            cfg.base_filters * (2 ** i) for i in range(cfg.num_blocks)
        ]

        self.blocks = nn.ModuleList([
            ConvBlock(filters[i], filters[i + 1], dropout=cfg.dropout)
            for i in range(cfg.num_blocks)
        ])

        self.attn_pool = AttentionPool(filters[-1])

        head_in = filters[-1]
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_in // 2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            logits: (B, num_classes)
        """
        for block in self.blocks:
            x = block(x)
        x = self.attn_pool(x)    # (B, C)
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_cnn(cfg: CNNConfig = CNN_CFG) -> EmotionCNN:
    return EmotionCNN(cfg)


# ─── Parameter count utility ──────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = CNN_CFG
    model = build_cnn(cfg)
    print(f"EmotionCNN  —  {count_parameters(model):,} trainable parameters")
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")

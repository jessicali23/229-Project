"""
models/lstm.py

Bidirectional LSTM for emotion recognition from MFCC time series.
Input: (B, T, input_size)  — time-major feature sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import LSTMConfig, LSTM_CFG


# ─── Attention over time ──────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over the LSTM output sequence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, H) — LSTM hidden states
        Returns:
            context: (B, H) — weighted sum
        """
        scores = self.attn(h).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return context


# ─── Bi-LSTM model ────────────────────────────────────────────────────────────

class EmotionLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM with temporal attention.

    Input:  (B, T, input_size)
    Output: (B, num_classes)
    """

    def __init__(self, cfg: LSTMConfig = LSTM_CFG):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        directions = 2 if cfg.bidirectional else 1
        self.out_size = cfg.hidden_size * directions

        self.attention = TemporalAttention(self.out_size)
        self.norm      = nn.LayerNorm(self.out_size)
        self.drop      = nn.Dropout(cfg.dropout)

        self.head = nn.Sequential(
            nn.Linear(self.out_size, self.out_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.out_size // 2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)
        Returns:
            logits: (B, num_classes)
        """
        h, _ = self.lstm(x)          # (B, T, H*dirs)
        h    = self.norm(h)
        ctx  = self.attention(h)     # (B, H*dirs)
        ctx  = self.drop(ctx)
        return self.head(ctx)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_lstm(input_size: int, cfg: LSTMConfig = LSTM_CFG) -> EmotionLSTM:
    cfg = LSTMConfig(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
    )
    return EmotionLSTM(cfg)


if __name__ == "__main__":
    from utils.config import LSTM_CFG
    n_mfcc = 40
    input_size = n_mfcc * 3  # MFCC + delta + delta2
    model = build_lstm(input_size)
    print(f"EmotionLSTM  —  {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    dummy = torch.randn(4, 130, input_size)
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")

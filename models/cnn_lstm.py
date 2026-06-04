"""
models/cnn_lstm.py

Hybrid CNN + Bidirectional LSTM architecture.

The CNN acts as a local feature extractor over spectrogram frames,
producing a sequence of frame-level embeddings that are then fed
to the LSTM to model temporal dynamics.

Input:  (B, 1, n_mels, T)
Output: (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import CNNLSTMConfig, CNN_LSTM_CFG
from models.lstm import TemporalAttention


# ─── CNN encoder ─────────────────────────────────────────────────────────────

class SpectrogramCNNEncoder(nn.Module):
    """
    Extract per-frame feature vectors from a spectrogram.
    Returns a (B, T', cnn_out_channels) sequence for the LSTM.
    """

    def __init__(self, base_filters: int, num_blocks: int, dropout: float):
        super().__init__()
        layers = []
        in_ch  = 1
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),   # halve frequency, keep time
                nn.Dropout2d(dropout),
            ]
            in_ch = out_ch
        self.cnn   = nn.Sequential(*layers)
        self.out_ch = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            seq: (B, T, cnn_out_channels*compressed_freq)
        """
        x = self.cnn(x)          # (B, C, H', T)
        B, C, H, T = x.shape
        # Collapse the frequency dimension → treat (C*H) as feature per time step
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, T, C, H)
        x = x.view(B, T, C * H)                  # (B, T, C*H)
        return x


# ─── CNN + LSTM model ─────────────────────────────────────────────────────────

class EmotionCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for speech emotion recognition.

    Input:  (B, 1, n_mels, T)
    Output: (B, num_classes)
    """

    def __init__(self, cfg: CNNLSTMConfig = CNN_LSTM_CFG, n_mels: int = 128):
        super().__init__()
        self.cfg = cfg

        self.cnn_enc = SpectrogramCNNEncoder(
            base_filters=cfg.cnn_base_filters,
            num_blocks=cfg.cnn_num_blocks,
            dropout=cfg.cnn_dropout,
        )

        # Compute CNN output feature size given n_mels
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, 10)
            cnn_out = self.cnn_enc(dummy)         # (1, 10, F)
            cnn_feat_size = cnn_out.shape[-1]

        # Project to a manageable size before LSTM
        self.proj = nn.Linear(cnn_feat_size, cfg.lstm_hidden)

        self.lstm = nn.LSTM(
            input_size=cfg.lstm_hidden,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.lstm_bidirectional,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )

        directions = 2 if cfg.lstm_bidirectional else 1
        lstm_out_size = cfg.lstm_hidden * directions

        self.attention = TemporalAttention(lstm_out_size)
        self.norm      = nn.LayerNorm(lstm_out_size)
        self.drop      = nn.Dropout(cfg.lstm_dropout)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.lstm_dropout),
            nn.Linear(lstm_out_size // 2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            logits: (B, num_classes)
        """
        seq = self.cnn_enc(x)           # (B, T, cnn_features)
        seq = F.relu(self.proj(seq))     # (B, T, lstm_hidden)
        h, _ = self.lstm(seq)           # (B, T, lstm_out)
        h    = self.norm(h)
        ctx  = self.attention(h)        # (B, lstm_out)
        ctx  = self.drop(ctx)
        return self.head(ctx)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_cnn_lstm(cfg: CNNLSTMConfig = CNN_LSTM_CFG, n_mels: int = 128) -> EmotionCNNLSTM:
    return EmotionCNNLSTM(cfg, n_mels=n_mels)


if __name__ == "__main__":
    model = build_cnn_lstm(n_mels=128)
    print(f"EmotionCNNLSTM  —  {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    dummy = torch.randn(4, 1, 128, 130)
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")

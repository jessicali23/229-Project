
"""
models/paper_cnn_bilstm.py

Exact replication of Table 2 from:
  "Speech emotion recognition with light weight deep neural ensemble model
   using hand crafted features" (Nature, 2025)

Architecture:
  Block 1: Conv1D(128, k=5) -> BN -> ReLU -> MaxPool
  Block 2: Conv1D(128, k=5) -> BN -> ReLU -> MaxPool -> Dropout(0.2)
  Block 3: Conv1D(64,  k=5) -> BN -> ReLU -> MaxPool
  Block 4: Conv1D(64,  k=3) -> BN -> ReLU -> MaxPool -> Dropout(0.2)
  Bi-LSTM: hidden=64, bidirectional -> output 128
  Block 5: Conv1D(32,  k=3) -> BN -> ReLU -> MaxPool -> Dropout(0.2)
  Flatten -> Dense(128) -> BN -> ReLU -> Dense(num_classes)

Input:  (batch, feature_dim)  flat 1D feature vector
Output: (batch, num_classes)  logits
"""

import torch
import torch.nn as nn


class PaperCNNBiLSTM(nn.Module):

    def __init__(self, input_features: int, num_classes: int = 7):
        super().__init__()

        # Block 1 — kernel 5, no dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            #nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Block 2 — kernel 5, dropout
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            #nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        # Block 3 — kernel 5, no dropout
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Block 4 — kernel 3, dropout
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding="same"),
            nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # Bi-LSTM inserted between block 4 and block 5
        # input_size=64 (from block4 channels), hidden=64, bidirectional -> 128 out
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Block 5 — kernel 3, dropout
        self.block5 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=3, padding="same"),
            nn.BatchNorm1d(32),
            #nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # Compute flattened size dynamically so any input_features works
        with torch.no_grad():
            d = torch.zeros(1, 1, input_features)
            d = self.block1(d)
            d = self.block2(d)
            d = self.block3(d)
            d = self.block4(d)
            d, _ = self.bilstm(d.permute(0, 2, 1))
            d = self.block5(d.permute(0, 2, 1))
            flat_size = d.flatten(1).shape[1]

        print(f"  PaperCNNBiLSTM: input={input_features}  flat={flat_size}  classes={num_classes}")

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, feature_dim)
        Returns:
            logits: (B, num_classes)
        """
        x = x.unsqueeze(1)                      # (B, feat) -> (B, 1, feat)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.permute(0, 2, 1)                  # -> (B, seq, 64)
        x, _ = self.bilstm(x)                   # -> (B, seq, 128)
        x = x.permute(0, 2, 1)                  # -> (B, 128, seq)
        x = self.block5(x)
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_paper_cnn_bilstm(input_features: int, num_classes: int = 7) -> PaperCNNBiLSTM:
    return PaperCNNBiLSTM(input_features=input_features, num_classes=num_classes)


if __name__ == "__main__":
    model = build_paper_cnn_bilstm(input_features=X_train.shape[1], num_classes=7)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(4, 3672)

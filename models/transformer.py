"""
models/transformer.py

Transformer-based encoder for speech emotion recognition.
Treats the spectrogram as a sequence of patch embeddings (ViT-style)
or as frame-level embeddings (AST-style).

Input:  (B, 1, n_mels, T)
Output: (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import TransformerConfig, TRANSFORMER_CFG


# ─── Patch Embedding ─────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Divide (1, n_mels, T) spectrogram into non-overlapping 2D patches
    and project each to d_model.
    """
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        n_mels: int,
        n_frames: int,
        in_channels: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )
        # Number of patches
        self.n_freq_patches  = n_mels   // patch_size
        self.n_time_patches  = n_frames // patch_size
        self.num_patches     = self.n_freq_patches * self.n_time_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            patches: (B, num_patches, d_model)
        """
        x = self.proj(x)                        # (B, d_model, H_p, W_p)
        B, D, Hf, Wt = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H_p*W_p, d_model)
        return x


# ─── Positional Encoding ─────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─── Transformer Emotion Classifier ──────────────────────────────────────────

class EmotionTransformer(nn.Module):
    """
    Vision-Transformer–style encoder for spectrograms.

    Architecture:
        PatchEmbedding → [CLS] token + Positional Encoding
        → Transformer Encoder (N layers)
        → CLS token → classification head
    """

    def __init__(
        self,
        cfg: TransformerConfig = TRANSFORMER_CFG,
        n_mels: int = 128,
        n_frames: int = 130,
    ):
        super().__init__()
        self.cfg = cfg

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            n_mels=n_mels,
            n_frames=n_frames,
        )
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learned positional embedding (cls + patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(cfg.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_encoder_layers
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]

        # Patch embedding
        patches = self.patch_embed(x)                # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        tokens = torch.cat([cls, patches], dim=1)    # (B, 1+N, D)

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        tokens = self.pos_drop(tokens)

        # Transformer
        tokens = self.transformer(tokens)            # (B, 1+N, D)
        tokens = self.norm(tokens)

        # Use CLS token for classification
        cls_out = tokens[:, 0]                       # (B, D)
        return self.head(cls_out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_transformer(
    cfg: TransformerConfig = TRANSFORMER_CFG,
    n_mels: int = 128,
    n_frames: int = 130,
) -> EmotionTransformer:
    return EmotionTransformer(cfg, n_mels=n_mels, n_frames=n_frames)


if __name__ == "__main__":
    model = build_transformer(n_mels=128, n_frames=128)
    print(f"EmotionTransformer  —  {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    dummy = torch.randn(4, 1, 128, 128)
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")

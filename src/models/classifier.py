"""Mudra classifiers on 63-dim hand landmarks.

Two architectures for comparison: a flat MLP baseline, and an attention-based
model that treats the 21 keypoints as a sequence of 3D tokens. Attention lets
the model learn explicit keypoint-to-keypoint relationships (thumb tip near
index PIP, etc.) that distinguish similar mudras.
"""
import torch
import torch.nn as nn

HAND_LANDMARKS = 21
COORDS = 3
HAND_FEATURE_DIM = HAND_LANDMARKS * COORDS


class MLPClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(COORDS, d_model)
        self.pos = nn.Parameter(torch.randn(1, HAND_LANDMARKS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = x.view(b, HAND_LANDMARKS, COORDS)
        tokens = self.embed(tokens) + self.pos
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


TWO_HAND_FEATURE_DIM = 2 * HAND_LANDMARKS * COORDS  # 126


class TwoHandMLPClassifier(nn.Module):
    """Flat MLP baseline on 126-dim concatenated two-hand features."""

    def __init__(self, n_classes: int, hidden: int = 384, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TWO_HAND_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SamyuktaClassifier(nn.Module):
    """Attention over 42 keypoints (21 per hand x 2 hands).

    Each keypoint gets a positional embedding (shared across hands, since
    keypoint 0 is the wrist on both) and a hand-identity embedding (so the
    model can distinguish left from right while still attending across them).
    The encoder learns cross-hand relationships — 'left index tip near right
    index tip' is exactly what defines Anjali, Kapota, etc.
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(COORDS, d_model)
        self.pos = nn.Parameter(torch.randn(1, HAND_LANDMARKS, d_model) * 0.02)
        self.hand_id = nn.Parameter(torch.randn(1, 2, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = x.view(b, 2, HAND_LANDMARKS, COORDS)
        tokens = self.embed(x)
        tokens = tokens + self.pos.unsqueeze(0)
        tokens = tokens + self.hand_id
        tokens = tokens.reshape(b, 2 * HAND_LANDMARKS, -1)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)

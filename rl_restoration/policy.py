"""Small reward-regression policy for the offline contextual bandit."""

from __future__ import annotations

import torch
from torch import nn


class RewardRouter(nn.Module):
    def __init__(self, input_dim: int, action_count: int, hidden_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_count = int(action_count)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def standardize_features(train_features, other_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_features - mean) / std, (other_features - mean) / std, mean, std


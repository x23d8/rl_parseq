"""Actor-critic network for the sequential restoration environment."""

from __future__ import annotations

import torch
from torch import nn


class RestorationActorCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_count: int,
        hidden_dim: int = 256,
        dropout: float = 0.05,
        prior_offset: int | None = None,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_count = int(action_count)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.prior_offset = prior_offset
        self.prior_scale = float(prior_scale)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_count)
        self.critic = nn.Linear(hidden_dim, 1)
        # With a teacher prior, PPO starts exactly from the locked bandit and
        # learns only a residual. This is the policy analogue of SFT -> PPO.
        if prior_offset is not None:
            nn.init.zeros_(self.actor.weight)
            nn.init.zeros_(self.actor.bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observations)
        logits = self.actor(hidden)
        if self.prior_offset is not None:
            prior = observations[:, self.prior_offset : self.prior_offset + self.action_count]
            logits = logits + self.prior_scale * prior
        return logits, self.critic(hidden).squeeze(-1)

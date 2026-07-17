"""Candidate-set actor, critic and reward teacher used by Phase 6."""

from __future__ import annotations

import torch
from torch import nn


class RewardTeacher(nn.Module):
    def __init__(self, input_dim: int, action_count: int, hidden_dim: int = 192, dropout: float = 0.08):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CandidateSetActorCritic(nn.Module):
    """Score each candidate after self-attention over all candidate observations."""

    def __init__(
        self,
        candidate_dim: int,
        action_count: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        prior_scale: float = 2.0,
    ):
        super().__init__()
        self.candidate_dim = int(candidate_dim)
        self.action_count = int(action_count)
        self.hidden_dim = int(hidden_dim)
        self.prior_scale = float(prior_scale)
        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_embedding = nn.Embedding(action_count, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.actor = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        # PPO must start exactly from the leakage-safe teacher. Candidate-set
        # attention then learns only a residual instead of destroying a useful
        # prior before the first rollout.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        teacher_prior: torch.Tensor,
        current_action: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, actions, _ = candidate_features.shape
        if actions != self.action_count:
            raise ValueError("Unexpected action dimension")
        step_per_candidate = step.float().view(batch, 1, 1).expand(batch, actions, 1)
        tokens = torch.cat((candidate_features, teacher_prior.unsqueeze(-1), step_per_candidate), dim=-1)
        action_ids = torch.arange(actions, device=candidate_features.device)
        encoded = self.candidate_projection(tokens) + self.action_embedding(action_ids)[None]
        encoded = self.candidate_encoder(encoded)
        context = encoded.mean(dim=1)
        current = encoded[torch.arange(batch, device=encoded.device), current_action]
        global_state = torch.cat((context, current), dim=-1)
        expanded = global_state[:, None, :].expand(batch, actions, global_state.shape[-1])
        action_states = torch.cat((encoded, expanded), dim=-1)
        logits = self.actor(action_states).squeeze(-1) + self.prior_scale * teacher_prior
        value = self.critic(global_state).squeeze(-1)
        return logits, value

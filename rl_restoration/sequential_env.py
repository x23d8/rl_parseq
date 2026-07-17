"""Offline two-step restoration environment used by PPO.

Step 0 selects a complete restoration view.  STOP keeps the baseline.  When a
non-STOP action is selected, step 1 observes that view's OCR output and selects
the final action. Repeating the first action accepts it; action 0 rolls back to
baseline. Views
are always generated from the original crop; preprocessing is never chained.
"""

from __future__ import annotations

import string

import numpy as np
import torch


PLATE_ALPHABET = "0123456789" + string.ascii_uppercase
MAX_PLATE_LENGTH = 12


def encode_predictions(predictions: np.ndarray) -> np.ndarray:
    """Encode OCR strings without using labels or edit distances."""

    alphabet = {character: index for index, character in enumerate(PLATE_ALPHABET)}
    result = np.zeros(
        (predictions.shape[0], predictions.shape[1], MAX_PLATE_LENGTH * len(PLATE_ALPHABET)),
        dtype=np.float32,
    )
    for row in range(predictions.shape[0]):
        for action in range(predictions.shape[1]):
            for position, character in enumerate(str(predictions[row, action])[:MAX_PLATE_LENGTH]):
                index = alphabet.get(character)
                if index is not None:
                    result[row, action, position * len(PLATE_ALPHABET) + index] = 1.0
    return result


def build_observable_cache(cache: dict) -> dict:
    """Build deployable observations from cached predictions and confidence."""

    predictions = np.asarray(cache["predictions"], dtype=str)
    confidence = np.asarray(cache["normalized_confidence"], dtype=np.float32)
    lengths = np.vectorize(len)(predictions).astype(np.float32) / MAX_PLATE_LENGTH
    baseline_predictions = predictions[:, [0]]
    changed = (predictions != baseline_predictions).astype(np.float32)
    one_hot = encode_predictions(predictions)
    observable = np.concatenate(
        (
            confidence[..., None],
            lengths[..., None],
            changed[..., None],
            one_hot,
        ),
        axis=2,
    ).astype(np.float32)
    action_count = predictions.shape[1]
    vote_fraction = np.zeros((len(predictions), action_count), dtype=np.float32)
    group_confidence = np.zeros_like(vote_fraction)
    baseline_distance = np.zeros_like(vote_fraction)

    def distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for row, left_character in enumerate(left, start=1):
            current = [row]
            for column, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    for row in range(len(predictions)):
        counts = {value: int((predictions[row] == value).sum()) for value in set(predictions[row])}
        max_confidence = {
            value: float(confidence[row][predictions[row] == value].max()) for value in counts
        }
        baseline = str(predictions[row, 0])
        for action in range(action_count):
            value = str(predictions[row, action])
            vote_fraction[row, action] = counts[value] / action_count
            group_confidence[row, action] = max_confidence[value]
            baseline_distance[row, action] = distance(value, baseline) / max(len(value), len(baseline), 1)
    candidate_summary = np.concatenate(
        (confidence, lengths, changed, vote_fraction, group_confidence, baseline_distance), axis=1
    ).astype(np.float32)
    return {"action_observations": observable, "candidate_summary": candidate_summary}


class OfflineSequentialRestorationEnv:
    """Vectorized deterministic MDP backed by train/validation trajectories."""

    def __init__(
        self,
        cache: dict,
        base_features: np.ndarray,
        device: torch.device,
        revisit_cost: float = 0.002,
        expose_candidate_summary: bool = False,
    ):
        self.cache = cache
        self.device = device
        self.base_features = torch.as_tensor(base_features, dtype=torch.float32, device=device)
        observable_cache = build_observable_cache(cache)
        observable = observable_cache["action_observations"]
        self.action_observations = torch.as_tensor(observable, dtype=torch.float32, device=device)
        self.candidate_summary = torch.as_tensor(
            observable_cache["candidate_summary"], dtype=torch.float32, device=device
        )
        self.expose_candidate_summary = bool(expose_candidate_summary)
        self.rewards = torch.as_tensor(cache["reward"], dtype=torch.float32, device=device)
        self.action_count = self.rewards.shape[1]
        self.revisit_cost = float(revisit_cost)
        summary_dimension = self.candidate_summary.shape[1] if self.expose_candidate_summary else 0
        self.observation_dim = self.base_features.shape[1] + self.action_observations.shape[2] + summary_dimension + self.action_count + 1

    def observations(self, indices: torch.Tensor, current_actions: torch.Tensor, step: int) -> torch.Tensor:
        current = self.action_observations[indices, current_actions]
        parts = [self.base_features[indices], current]
        if self.expose_candidate_summary:
            summary = self.candidate_summary[indices] if step == 1 else torch.zeros_like(self.candidate_summary[indices])
            parts.append(summary)
        action_one_hot = torch.nn.functional.one_hot(current_actions, self.action_count).float()
        step_feature = torch.full((len(indices), 1), float(step), device=self.device)
        parts.extend((action_one_hot, step_feature))
        return torch.cat(parts, dim=1)

    def state_zero(self, indices: torch.Tensor) -> torch.Tensor:
        baseline = torch.zeros_like(indices)
        return self.observations(indices, baseline, step=0)

    def state_one(self, indices: torch.Tensor, first_actions: torch.Tensor) -> torch.Tensor:
        return self.observations(indices, first_actions, step=1)

    def terminal_actions(self, first_actions: torch.Tensor, second_actions: torch.Tensor) -> torch.Tensor:
        return second_actions

    def terminal_rewards(
        self,
        indices: torch.Tensor,
        first_actions: torch.Tensor,
        second_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terminal = self.terminal_actions(first_actions, second_actions)
        reward = self.rewards[indices, terminal]
        revised = second_actions != first_actions
        reward = reward - revised.float() * self.revisit_cost
        return reward, terminal

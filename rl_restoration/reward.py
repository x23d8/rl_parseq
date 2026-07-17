"""Ground-truth OCR reward used by the offline restoration bandit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    edit_gain_weight: float = 1.0
    exact_gain_weight: float = 0.5
    harm_penalty: float = 0.5
    action_cost_weight: float = 1.0


def normalized_edit_accuracy(edit_distance: int, prediction: str, target: str) -> float:
    return 1.0 - float(edit_distance) / max(len(prediction), len(target), 1)


def ocr_reward(
    baseline_prediction: str,
    baseline_edit_distance: int,
    action_prediction: str,
    action_edit_distance: int,
    target: str,
    action_cost: float,
    config: RewardConfig = RewardConfig(),
) -> float:
    baseline_accuracy = normalized_edit_accuracy(baseline_edit_distance, baseline_prediction, target)
    action_accuracy = normalized_edit_accuracy(action_edit_distance, action_prediction, target)
    baseline_exact = baseline_prediction == target
    action_exact = action_prediction == target
    reward = config.edit_gain_weight * (action_accuracy - baseline_accuracy)
    reward += config.exact_gain_weight * (float(action_exact) - float(baseline_exact))
    reward -= config.action_cost_weight * float(action_cost)
    if baseline_exact and not action_exact:
        reward -= config.harm_penalty
    return float(reward)


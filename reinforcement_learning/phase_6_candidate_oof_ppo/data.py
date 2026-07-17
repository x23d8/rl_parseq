"""Cache loading and deterministic group-holdout utilities for Phase 6."""

from __future__ import annotations

import hashlib
import string
from pathlib import Path

import numpy as np
import pandas as pd


PLATE_ALPHABET = "0123456789" + string.ascii_uppercase
MAX_PLATE_LENGTH = 12


def candidate_ocr_features(cache: dict) -> np.ndarray:
    """Return label-free OCR string/consensus features for every candidate."""

    predictions = np.asarray(cache["predictions"], dtype=str)
    confidence = np.asarray(cache["normalized_confidence"], dtype=np.float32)
    samples, actions = predictions.shape
    alphabet = {character: index for index, character in enumerate(PLATE_ALPHABET)}
    one_hot = np.zeros(
        (samples, actions, MAX_PLATE_LENGTH * len(PLATE_ALPHABET)), dtype=np.float32
    )
    for row in range(samples):
        for action in range(actions):
            for position, character in enumerate(str(predictions[row, action])[:MAX_PLATE_LENGTH]):
                index = alphabet.get(character)
                if index is not None:
                    one_hot[row, action, position * len(PLATE_ALPHABET) + index] = 1.0
    lengths = np.vectorize(len)(predictions).astype(np.float32) / MAX_PLATE_LENGTH
    changed = (predictions != predictions[:, [0]]).astype(np.float32)
    vote_fraction = np.zeros((samples, actions), dtype=np.float32)
    group_confidence = np.zeros_like(vote_fraction)
    baseline_distance = np.zeros_like(vote_fraction)

    def distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for row_index, left_character in enumerate(left, start=1):
            current = [row_index]
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

    for row in range(samples):
        values = predictions[row]
        counts = {value: int((values == value).sum()) for value in set(values)}
        max_confidence = {value: float(confidence[row][values == value].max()) for value in counts}
        baseline = str(values[0])
        for action, value in enumerate(values):
            value = str(value)
            vote_fraction[row, action] = counts[value] / actions
            group_confidence[row, action] = max_confidence[value]
            baseline_distance[row, action] = distance(value, baseline) / max(len(value), len(baseline), 1)
    compact = np.stack(
        (confidence, lengths, changed, vote_fraction, group_confidence, baseline_distance), axis=2
    )
    return np.concatenate((compact, one_hot), axis=2).astype(np.float32)


def load_trajectory_cache(cache_dir: Path, split: str, action_names: list[str]) -> dict:
    """Load an existing trajectory cache without importing the legacy RL package."""

    if split.lower() == "test":
        raise ValueError("Phase 6 development must not load the audited test split")
    payload = np.load(cache_dir / f"{split}_state_features.npz", allow_pickle=True)
    image_paths = payload["image_paths"].astype(str)
    trajectories = pd.read_csv(cache_dir / f"{split}_action_trajectories.csv")
    if set(trajectories.image_path.astype(str)) - set(image_paths):
        raise ValueError(f"{split} trajectory paths do not match state features")

    matrices: dict[str, np.ndarray] = {}
    for column in ("reward", "exact", "edit_distance", "action_cost", "normalized_confidence"):
        pivot = trajectories.pivot(index="image_path", columns="action", values=column)
        pivot = pivot.reindex(index=image_paths, columns=action_names)
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete {column} matrix in {split} cache")
        matrices[column] = pivot.to_numpy()
    predictions = (
        trajectories.pivot(index="image_path", columns="action", values="prediction")
        .reindex(index=image_paths, columns=action_names)
        .fillna("")
        .to_numpy(dtype=str)
    )
    targets = (
        trajectories[trajectories.action == action_names[0]]
        .set_index("image_path")
        .reindex(image_paths)["target"]
        .astype(str)
        .to_numpy()
    )
    return {
        "features": payload["features"].astype(np.float32),
        "image_paths": image_paths,
        "targets": targets,
        "predictions": predictions,
        **matrices,
    }


def stable_group_holdout(groups: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split whole target groups using a stable hash, then approach the requested size."""

    groups = np.asarray(groups, dtype=str)
    unique, counts = np.unique(groups, return_counts=True)
    keyed = []
    for group, count in zip(unique, counts):
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        keyed.append((int.from_bytes(digest[:8], "big"), group, int(count)))
    keyed.sort()
    target = max(1, round(len(groups) * fraction))
    selected: set[str] = set()
    size = 0
    for _, group, count in keyed:
        if size >= target:
            break
        selected.add(group)
        size += count
    holdout = np.asarray([group in selected for group in groups], dtype=bool)
    development = ~holdout
    if not development.any() or not holdout.any():
        raise ValueError("Group holdout produced an empty partition")
    return development, holdout


def subset(cache: dict, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    return {key: value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value for key, value in cache.items()}


def load_candidate_features(path: Path, expected_paths: np.ndarray, action_names: list[str]) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    cached_paths = payload["image_paths"].astype(str)
    cached_actions = payload["action_names"].astype(str).tolist()
    if cached_actions != action_names:
        raise ValueError("Candidate feature action space does not match trajectories")
    if not np.array_equal(cached_paths, np.asarray(expected_paths, dtype=str)):
        raise ValueError("Candidate feature rows do not match trajectory rows")
    features = payload["candidate_features"].astype(np.float32)
    if features.ndim != 3 or features.shape[:2] != (len(expected_paths), len(action_names)):
        raise ValueError("candidate_features must have shape [samples, actions, features]")
    return features

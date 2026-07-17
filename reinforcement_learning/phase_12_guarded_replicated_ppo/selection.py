"""Label-free metadata safety guard for the Phase 12 PPO action."""

from __future__ import annotations

import numpy as np


def guarded_selection(
    ppo_selected: np.ndarray,
    input_transform: np.ndarray,
    crop_width: np.ndarray,
    crop_height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(ppo_selected, dtype=np.int64)
    transforms = np.asarray(input_transform, dtype=str)
    width = np.asarray(crop_width, dtype=np.int64)
    height = np.asarray(crop_height, dtype=np.int64)
    if not (len(selected) == len(transforms) == len(width) == len(height)):
        raise ValueError("Phase 12 guard inputs have different row counts")
    if (width <= 0).any() or (height <= 0).any():
        raise ValueError("Phase 12 guard requires positive crop dimensions")
    allowed = (transforms == "existing_plate_crop") & (np.minimum(width, height) < 128)
    final = np.where(allowed, selected, 0).astype(np.int64)
    return final, allowed


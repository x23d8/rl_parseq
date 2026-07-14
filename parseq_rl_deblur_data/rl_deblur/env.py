"""PixelRL-style environment: per-pixel discrete action set + dense reward.

Following "Fully Convolutional Network with Multi-Step Reinforcement Learning
for Image Processing" (Furuta et al.), every pixel is treated as an agent that
picks one operation from a shared toolbox each step. Because the policy
network is fully convolutional, all pixels share the same weights; we simply
compute every candidate operation over the *whole* image and then, per pixel,
select the value produced by whichever action that pixel chose.
"""

from __future__ import annotations

import cv2
import numpy as np

ACTIONS = [
    "keep",
    "pixel+1",
    "pixel-1",
    "pixel+3",
    "pixel-3",
    "gaussian_smooth",
    "bilateral_filter",
    "unsharp_mild",
    "unsharp_strong",
]
NUM_ACTIONS = len(ACTIONS)
PIXEL_DELTA = {1: 1.0, 2: -1.0, 3: 3.0, 4: -3.0}


def _unsharp(img_u8: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img_u8, (0, 0), sigma)
    sharp = img_u8.astype(np.float32) + amount * (img_u8.astype(np.float32) - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def compute_candidates(state: np.ndarray) -> np.ndarray:
    """state: (B, H, W) float32 in [0, 255]. Returns (B, NUM_ACTIONS, H, W) float32."""
    batch = state.shape[0]
    h, w = state.shape[1], state.shape[2]
    out = np.empty((batch, NUM_ACTIONS, h, w), dtype=np.float32)
    for b in range(batch):
        img_f = state[b]
        img_u8 = np.clip(img_f, 0, 255).astype(np.uint8)
        out[b, 0] = img_f
        for action_id, delta in PIXEL_DELTA.items():
            out[b, action_id] = np.clip(img_f + delta, 0, 255)
        out[b, 5] = cv2.GaussianBlur(img_u8, (3, 3), 0.5).astype(np.float32)
        out[b, 6] = cv2.bilateralFilter(img_u8, d=5, sigmaColor=25, sigmaSpace=5).astype(np.float32)
        out[b, 7] = _unsharp(img_u8, amount=0.5).astype(np.float32)
        out[b, 8] = _unsharp(img_u8, amount=1.5).astype(np.float32)
    return out


def step(state: np.ndarray, action_map: np.ndarray, clean: np.ndarray):
    """Apply per-pixel actions and compute the dense restoration reward.

    state: (B, H, W) float32 in [0, 255], current (possibly partially restored) image.
    action_map: (B, H, W) int64, action index chosen per pixel.
    clean: (B, H, W) float32 in [0, 255], ground-truth sharp image.

    Returns (next_state, reward) both (B, H, W) float32.
    """
    candidates = compute_candidates(state)
    next_state = np.take_along_axis(candidates, action_map[:, None, :, :], axis=1).squeeze(1)
    se_before = (state - clean) ** 2
    se_after = (next_state - clean) ** 2
    reward = (se_before - se_after) / (255.0 ** 2)
    return next_state, reward

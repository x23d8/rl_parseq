"""Fully convolutional actor-critic shared by every pixel (PixelRL-style).

7 conv layers with a dilation schedule (1,2,3,4,3,2,1) to grow the receptive
field without losing spatial resolution, then two 1-conv heads: a policy head
(per-pixel action logits) and a value head (per-pixel state value).

Also implements Reward Map Convolution (RMC) from the original paper -- the
technique the paper reports as its main performance driver. A learnable
spatial kernel lets each pixel's return bootstrap off its *neighbors'*
predicted values, not just its own, which propagates information faster
across flat/uniform regions of the plate (e.g. background) where a single
pixel's own reward signal is weak or noisy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rl_deblur.env import NUM_ACTIONS

DILATIONS = [1, 2, 3, 4, 3, 2, 1]


class RewardMapConv(nn.Module):
    """Learnable local-averaging kernel w used to convolve future value/reward maps.

    Weights are softmax-normalized so they always sum to 1 (a proper spatial
    weighted average, matching the w in Eq. 12-13 of the paper). Initialized
    near-identity (all weight on the center tap) so early training behaves
    like plain per-pixel A2C and gradually learns to spread credit spatially.
    """

    def __init__(self, kernel_size: int = 9):
        super().__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        init = torch.zeros(kernel_size * kernel_size)
        center = (kernel_size * kernel_size) // 2
        init[center] = 8.0  # softmax(logits) -> ~near one-hot at center initially
        self.logits = nn.Parameter(init)

    def kernel(self) -> torch.Tensor:
        w = torch.softmax(self.logits, dim=0).view(1, 1, self.kernel_size, self.kernel_size)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W) -> (B, H, W), same-size convolution per image."""
        w = self.kernel().to(dtype=x.dtype, device=x.device)
        out = F.conv2d(x.unsqueeze(1), w, padding=self.padding)
        return out.squeeze(1)


def _make_trunk(channels: int) -> nn.Sequential:
    layers = []
    in_ch = 1
    for dilation in DILATIONS:
        layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=dilation, dilation=dilation))
        layers.append(nn.ReLU(inplace=True))
        in_ch = channels
    return nn.Sequential(*layers)


class FCNActorCritic(nn.Module):
    """Separate policy and value trunks (rather than one shared trunk).

    A shared-trunk design was tried first (as in most actor-critic setups)
    but proved unstable here: gradients from the value regression flowed
    back through the shared features and measurably dragged the policy's
    output distribution away from its initial "do nothing" bias within a
    handful of epochs, causing a sudden collapse in restoration quality that
    was reproducible regardless of entropy coefficient or bootstrap target.
    Fully separate trunks remove that interference pathway at a modest
    (roughly 2x trunk) compute cost, which is negligible at this image size.
    """

    def __init__(self, channels: int = 64, num_actions: int = NUM_ACTIONS, rmc_kernel_size: int = 9):
        super().__init__()
        self.policy_trunk = _make_trunk(channels)
        self.value_trunk = _make_trunk(channels)
        self.policy_head = nn.Conv2d(channels, num_actions, kernel_size=3, padding=1)
        self.value_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.rmc = RewardMapConv(kernel_size=rmc_kernel_size)

        # Plain default (PyTorch Kaiming) init for both heads. Two "safety"
        # inits were tried and both backfired: (1) zeroing policy_head and
        # biasing it toward action 0 ("keep") was meant to avoid a bad
        # untrained policy, but Adam normalizes gradients by their running
        # RMS, so every weight element -- all starting from *exactly* the
        # same value -- drifts at a similar ~lr-sized step regardless of the
        # (tiny) raw loss, and once that shared drift crossed the fixed bias
        # margin, many pixels flipped their action simultaneously, causing a
        # sudden collapse reproducible across many unrelated hyperparameter
        # changes; (2) shrinking value_head's init didn't fix it either.
        # Plain small-random default init avoids the knife-edge/lockstep
        # dynamics of a special init entirely and is what the first,
        # empirically-stable version of this pipeline used.
        nn.init.uniform_(self.value_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor):
        """x: (B, 1, H, W) in [0, 1]. Returns (policy_logits (B,A,H,W), value (B,H,W))."""
        logits = self.policy_head(self.policy_trunk(x))
        value = self.value_head(self.value_trunk(x)).squeeze(1)
        return logits, value

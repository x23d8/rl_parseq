"""Auditable one-step action space for the restoration contextual bandit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = ROOT / "preprocessing_best_config"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from benchmark_multiscale_tta import apply_center_zoom, unwrap_plate_lines, upscale_small_image  # noqa: E402
from preprocessing import get_preprocessing_config, preprocess_plate_image  # noqa: E402


@dataclass(frozen=True)
class RestorationAction:
    name: str
    preprocessing: str
    cost: float
    upscale: float = 1.0
    zoom: float = 1.0
    unwrap_two_line: bool = False
    description: str = ""

    def apply(self, image: Image.Image) -> Image.Image:
        result = apply_center_zoom(image.convert("RGB"), self.zoom)
        result = upscale_small_image(result, self.upscale)
        if self.unwrap_two_line:
            result = unwrap_plate_lines(result)
        return preprocess_plate_image(result, get_preprocessing_config(self.preprocessing)).convert("RGB")


# Every action produces a complete view from the original crop. Pipelines are
# not chained, so CLAHE/deblur/denoise cannot accidentally be applied twice.
DEFAULT_ACTIONS: tuple[RestorationAction, ...] = (
    RestorationAction(
        "stop_baseline", "train_baseline", 0.000, description="Keep the training-time baseline view."
    ),
    RestorationAction("raw_rgb", "raw_rgb", 0.002, description="Keep colour and skip enhancement."),
    RestorationAction(
        "clahe_gentle", "clahe_clip1_tile4", 0.005, description="Gentle local contrast enhancement."
    ),
    RestorationAction(
        "homomorphic", "homomorphic_filter", 0.007, description="Correct uneven illumination."
    ),
    RestorationAction(
        "rl_bilateral", "rl_deblur_bilateral_lowpass", 0.010, description="Mild RL deblur and denoise."
    ),
    RestorationAction(
        "clahe_rl_bilateral",
        "clahe_rl_deblur_bilateral",
        0.012,
        description="Gentle CLAHE, RL deblur and bilateral denoise.",
    ),
    RestorationAction(
        "adaptive_noise", "adaptive_noise_3way", 0.008, description="Validation-locked noise router."
    ),
    RestorationAction(
        "up2_baseline",
        "train_baseline",
        0.006,
        upscale=2.0,
        description="Upscale small crops before baseline restoration.",
    ),
    RestorationAction(
        "up2_clahe",
        "clahe_clip1_tile4",
        0.009,
        upscale=2.0,
        description="Upscale small crops before gentle CLAHE.",
    ),
    RestorationAction(
        "unwrap_up2_adaptive",
        "adaptive_noise_3way",
        0.015,
        upscale=2.0,
        unwrap_two_line=True,
        description="Unwrap likely two-line plates, upscale, then route by noise.",
    ),
)


# Geometry-preserving profile for paired restoration benchmarks.  The first
# action is deliberately the untouched RGB input, so every learned selector
# can abstain and every reward/image-quality delta has the same raw reference.
FAIR_RESTORATION_ACTIONS: tuple[RestorationAction, ...] = (
    RestorationAction(
        "stop_baseline", "raw_rgb", 0.000, description="Keep the degraded RGB input unchanged."
    ),
    RestorationAction(
        "unsharp_mild", "unsharp_mild", 0.004, description="Mild global unsharp masking."
    ),
    RestorationAction(
        "clahe_gentle", "clahe_clip1_tile4", 0.005, description="Gentle local contrast enhancement."
    ),
    RestorationAction(
        "homomorphic", "homomorphic_filter", 0.007, description="Correct uneven illumination."
    ),
    RestorationAction(
        "wiener_deconv", "wiener_deconv", 0.010, description="Wiener deconvolution."
    ),
    RestorationAction(
        "rl_bilateral",
        "rl_deblur_bilateral_lowpass",
        0.010,
        description="Richardson-Lucy deconvolution with bilateral denoising.",
    ),
    RestorationAction(
        "clahe_rl_bilateral",
        "clahe_rl_deblur_bilateral",
        0.012,
        description="CLAHE followed by Richardson-Lucy deconvolution and denoising.",
    ),
    RestorationAction(
        "adaptive_noise", "adaptive_noise_3way", 0.008, description="Fixed quality-based denoising router."
    ),
)


ACTION_PROFILES = {
    "default": DEFAULT_ACTIONS,
    "fair_restoration": FAIR_RESTORATION_ACTIONS,
}


def get_action_profile(name: str = "default") -> tuple[RestorationAction, ...]:
    try:
        return ACTION_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown action profile {name!r}; choose from {sorted(ACTION_PROFILES)}") from exc


def action_by_name(name: str) -> RestorationAction:
    for action in DEFAULT_ACTIONS:
        if action.name == name:
            return action
    raise KeyError(f"Unknown restoration action: {name}")


def validate_action_space(actions: tuple[RestorationAction, ...] = DEFAULT_ACTIONS) -> None:
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise ValueError("Restoration action names must be unique")
    if not actions or actions[0].name != "stop_baseline" or actions[0].cost != 0:
        raise ValueError("The first action must be the zero-cost stop_baseline reference")
    for action in actions:
        get_preprocessing_config(action.preprocessing)

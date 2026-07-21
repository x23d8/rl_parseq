"""Controlled, resolution-aware augmentation for Phase 3 PARSeq fine-tuning.

The policy is deliberately conservative for OCR: it never flips an image,
never rotates it aggressively, and limits the number of simultaneous image
degradations.  Every call returns an audit trace so the training script can
write the augmentation distribution actually seen by the model.
"""

from __future__ import annotations

import io
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from preprocessing_best_config.benchmark_multiscale_tta import (
        apply_center_zoom,
        unwrap_plate_lines,
        upscale_small_image,
    )
    from preprocessing_best_config.preprocessing import (
        get_preprocessing_config,
        preprocess_plate_image,
    )
except ImportError:  # Direct execution with repository paths inserted.
    from benchmark_multiscale_tta import apply_center_zoom, unwrap_plate_lines, upscale_small_image
    from preprocessing import get_preprocessing_config, preprocess_plate_image


@dataclass(frozen=True)
class Phase3AugmentationConfig:
    """Probability limits for the full controlled policy."""

    profile: str = "full"
    near_identity_probability: float = 0.20
    zoom_probability: float = 0.42
    zoom_min: float = 0.84
    zoom_max: float = 1.16
    perspective_probability: float = 0.12
    affine_probability: float = 0.18
    unwrap_probability: float = 0.38
    resolution_probability: float = 0.42
    blur_probability: float = 0.20
    jpeg_probability: float = 0.20
    noise_probability: float = 0.16
    photometric_probability: float = 0.30
    max_degradations: int = 2
    min_downsample_scale: float = 0.38
    max_downsample_scale: float = 0.72
    jpeg_quality_min: int = 38
    jpeg_quality_max: int = 82
    preprocess_weights: tuple[tuple[str, float], ...] = (
        ("train_baseline", 0.34),
        ("clahe_clip1_tile4", 0.20),
        ("clahe_rl_deblur_bilateral", 0.16),
        ("adaptive_noise_3way", 0.20),
        ("raw_rgb", 0.10),
    )

    def validate(self) -> None:
        if self.profile not in {"full", "resolution_only", "restoration_only", "light"}:
            raise ValueError(f"Unknown augmentation profile: {self.profile}")
        probability_names = (
            "near_identity_probability",
            "zoom_probability",
            "perspective_probability",
            "affine_probability",
            "unwrap_probability",
            "resolution_probability",
            "blur_probability",
            "jpeg_probability",
            "noise_probability",
            "photometric_probability",
        )
        for name in probability_names:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.max_degradations < 0:
            raise ValueError("max_degradations must be non-negative")
        if not self.preprocess_weights or sum(weight for _, weight in self.preprocess_weights) <= 0:
            raise ValueError("preprocess_weights must have a positive total")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["preprocess_weights"] = [list(item) for item in self.preprocess_weights]
        return payload


def _weighted_choice(weighted_items: tuple[tuple[str, float], ...], rng: random.Random) -> str:
    names, weights = zip(*weighted_items)
    return rng.choices(names, weights=weights, k=1)[0]


def _quality_features(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0)
    return {
        "width": float(image.width),
        "height": float(image.height),
        "aspect": float(image.width / max(image.height, 1)),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(laplacian.var()),
        "noise": float(np.median(np.abs(residual))),
    }


def _motion_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    size = rng.choice((3, 5))
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)
    kernel /= kernel.sum()
    return Image.fromarray(cv2.filter2D(arr, -1, kernel))


def _gaussian_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    radius = rng.uniform(0.35, 1.15)
    return TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[radius, radius])


def _add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    sigma = rng.uniform(2.0, 9.0)
    generator = np.random.default_rng(rng.randrange(0, 2**32 - 1))
    noisy = arr + generator.normal(0.0, sigma, size=arr.shape)
    if rng.random() < 0.20:
        amount = rng.uniform(0.0005, 0.002)
        mask = generator.random(arr.shape[:2])
        noisy[mask < amount] = 0
        noisy[mask > 1.0 - amount] = 255
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def _jpeg_roundtrip(image: Image.Image, rng: random.Random, quality_min: int, quality_max: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=rng.randint(quality_min, quality_max))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _downsample_reconstruct(
    image: Image.Image,
    rng: random.Random,
    scale_min: float,
    scale_max: float,
) -> Image.Image:
    width, height = image.size
    scale = rng.uniform(scale_min, scale_max)
    tiny = image.resize(
        (max(8, round(width * scale)), max(6, round(height * scale))),
        rng.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)),
    )
    return tiny.resize((width, height), rng.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)))


def _photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.28))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.72, 1.30))
    if rng.random() < 0.55:
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.65, 1.25))
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    gamma = rng.uniform(0.78, 1.28)
    return Image.fromarray(np.clip((arr**gamma) * 255.0, 0, 255).astype(np.uint8))


def _mild_affine(image: Image.Image, rng: random.Random) -> Image.Image:
    fill = tuple(int(v) for v in np.median(np.asarray(image.convert("RGB")), axis=(0, 1)))
    return TF.affine(
        image,
        angle=rng.uniform(-2.2, 2.2),
        translate=[round(rng.uniform(-0.025, 0.025) * image.width), round(rng.uniform(-0.035, 0.035) * image.height)],
        scale=rng.uniform(0.96, 1.04),
        shear=[rng.uniform(-2.0, 2.0), 0.0],
        interpolation=InterpolationMode.BICUBIC,
        fill=fill,
    )


def _mild_perspective(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    dx = max(1, round(width * rng.uniform(0.015, 0.045)))
    dy = max(1, round(height * rng.uniform(0.02, 0.07)))
    start = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    end = [
        [rng.randint(0, dx), rng.randint(0, dy)],
        [width - 1 - rng.randint(0, dx), rng.randint(0, dy)],
        [width - 1 - rng.randint(0, dx), height - 1 - rng.randint(0, dy)],
        [rng.randint(0, dx), height - 1 - rng.randint(0, dy)],
    ]
    fill = tuple(int(v) for v in np.median(np.asarray(image.convert("RGB")), axis=(0, 1)))
    return TF.perspective(image, start, end, InterpolationMode.BICUBIC, fill=fill)


class ControlledPlateAugmenter:
    """Apply a bounded stochastic policy and return a compact audit trace."""

    def __init__(self, config: Phase3AugmentationConfig, seed: int | None = None):
        config.validate()
        self.config = config
        self.rng = random.Random(seed) if seed is not None else random

    def __call__(self, image: Image.Image) -> tuple[Image.Image, tuple[str, ...]]:
        cfg = self.config
        rng = self.rng
        image = image.convert("RGB")
        features = _quality_features(image)
        trace: list[str] = []

        near_identity = rng.random() < cfg.near_identity_probability
        if near_identity:
            preprocess_name = "train_baseline" if rng.random() < 0.75 else "raw_rgb"
            trace.append("near_identity")
        else:
            preprocess_name = _weighted_choice(cfg.preprocess_weights, rng)

            if cfg.profile in {"full", "resolution_only", "light"}:
                if rng.random() < cfg.zoom_probability:
                    image = apply_center_zoom(image, rng.uniform(cfg.zoom_min, cfg.zoom_max))
                    trace.append("zoom")
                if features["aspect"] < 1.9 and rng.random() < cfg.unwrap_probability:
                    image = unwrap_plate_lines(image)
                    trace.append("unwrap_two_line")
                if cfg.profile != "light" and rng.random() < cfg.perspective_probability:
                    image = _mild_perspective(image, rng)
                    trace.append("perspective")
                if rng.random() < cfg.affine_probability:
                    image = _mild_affine(image, rng)
                    trace.append("affine")

            degradation_candidates: list[str] = []
            small_crop = features["width"] < 96 or features["height"] < 40
            resolution_p = min(0.72, cfg.resolution_probability * (1.35 if small_crop else 1.0))
            if cfg.profile in {"full", "resolution_only", "light"} and rng.random() < resolution_p:
                degradation_candidates.append("low_resolution")
            if cfg.profile == "full":
                if rng.random() < cfg.blur_probability:
                    degradation_candidates.append("blur")
                if rng.random() < cfg.jpeg_probability:
                    degradation_candidates.append("jpeg")
                if rng.random() < cfg.noise_probability:
                    degradation_candidates.append("noise")
                if rng.random() < cfg.photometric_probability:
                    degradation_candidates.append("photometric")
            rng.shuffle(degradation_candidates)
            for degradation in degradation_candidates[: cfg.max_degradations]:
                if degradation == "low_resolution":
                    image = _downsample_reconstruct(image, rng, cfg.min_downsample_scale, cfg.max_downsample_scale)
                elif degradation == "blur":
                    image = _motion_blur(image, rng) if rng.random() < 0.45 else _gaussian_blur(image, rng)
                elif degradation == "jpeg":
                    image = _jpeg_roundtrip(image, rng, cfg.jpeg_quality_min, cfg.jpeg_quality_max)
                elif degradation == "noise":
                    image = _add_noise(image, rng)
                elif degradation == "photometric":
                    image = _photometric(image, rng)
                trace.append(degradation)

        # Upscale genuinely small crops before nonlinear restoration. This is
        # the ordering that performed best in the Phase 1 benchmark.
        if image.width < 128 or image.height < 48:
            factor = 3.0 if min(image.width, image.height) < 28 and rng.random() < 0.45 else 2.0
            previous_size = image.size
            image = upscale_small_image(image, factor)
            if image.size != previous_size:
                trace.append(f"upscale_{int(factor)}x")

        if preprocess_name != "raw_rgb":
            image = preprocess_plate_image(image, get_preprocessing_config(preprocess_name))
        trace.append(f"preprocess:{preprocess_name}")
        return image.convert("RGB"), tuple(trace)


def count_augmentation_trace(trace: tuple[str, ...], counter: Counter | None = None) -> Counter:
    counter = counter if counter is not None else Counter()
    counter.update(trace)
    counter["samples"] += 1
    return counter


def trace_rates(counter: Mapping[str, int]) -> list[dict[str, float | int | str]]:
    samples = max(int(counter.get("samples", 0)), 1)
    return [
        {"operation": name, "count": int(count), "sample_rate": float(count / samples)}
        for name, count in sorted(counter.items())
        if name != "samples"
    ]


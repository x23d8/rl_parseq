"""Reusable plate image preprocessing variants for official PARSeq ANPR runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class PreprocessingConfig:
    name: str
    grayscale: bool = True
    autocontrast: bool = False
    clahe_clip_limit: float | None = 2.0
    clahe_tile_size: int = 8
    sharpen_alpha: float = 1.5
    sharpen_sigma: float = 1.0
    adaptive_threshold: bool = False
    adaptive_block_size: int = 25
    adaptive_c: int = 7

    def to_dict(self) -> dict:
        return asdict(self)


RAW_CONFIG = PreprocessingConfig(
    name="raw",
    grayscale=False,
    clahe_clip_limit=None,
    sharpen_alpha=0.0,
)


DEFAULT_CONFIG = PreprocessingConfig(name="clahe_sharpen")


SWEEP_CONFIGS = [
    RAW_CONFIG,
    PreprocessingConfig(name="autocontrast", autocontrast=True, clahe_clip_limit=None, sharpen_alpha=0.0),
    PreprocessingConfig(name="clahe_1_5", clahe_clip_limit=1.5, sharpen_alpha=0.0),
    PreprocessingConfig(name="clahe_2_sharp_1_2", clahe_clip_limit=2.0, sharpen_alpha=1.2),
    DEFAULT_CONFIG,
    PreprocessingConfig(name="clahe_3_sharp_1_5", clahe_clip_limit=3.0, sharpen_alpha=1.5),
    PreprocessingConfig(name="adaptive_thresh", clahe_clip_limit=2.0, sharpen_alpha=1.0, adaptive_threshold=True),
]


def get_preprocessing_config(name: str) -> PreprocessingConfig:
    for cfg in SWEEP_CONFIGS:
        if cfg.name == name:
            return cfg
    raise KeyError(f"Unknown preprocessing config: {name}")


def list_preprocessing_configs() -> list[str]:
    return [cfg.name for cfg in SWEEP_CONFIGS]


def _opencv_preprocess(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    import cv2

    arr = np.asarray(image.convert("RGB"))
    if cfg.grayscale:
        work = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        work = arr

    if cfg.autocontrast:
        if work.ndim == 2:
            work = np.asarray(ImageOps.autocontrast(Image.fromarray(work)))
        else:
            work = np.asarray(ImageOps.autocontrast(Image.fromarray(work)))

    if cfg.clahe_clip_limit is not None:
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip_limit),
            tileGridSize=(int(cfg.clahe_tile_size), int(cfg.clahe_tile_size)),
        )
        if work.ndim == 2:
            work = clahe.apply(work)
        else:
            lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if cfg.sharpen_alpha > 0:
        blur = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma))
        work = cv2.addWeighted(work, 1.0 + float(cfg.sharpen_alpha), blur, -float(cfg.sharpen_alpha), 0)

    if cfg.adaptive_threshold:
        gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        block_size = int(cfg.adaptive_block_size)
        if block_size % 2 == 0:
            block_size += 1
        work = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(block_size, 3),
            int(cfg.adaptive_c),
        )

    if work.ndim == 2:
        work = np.stack([work, work, work], axis=-1)
    return Image.fromarray(work.astype(np.uint8))


def preprocess_plate_image(image: Image.Image, cfg: PreprocessingConfig | str | None = None) -> Image.Image:
    cfg = DEFAULT_CONFIG if cfg is None else get_preprocessing_config(cfg) if isinstance(cfg, str) else cfg
    if cfg.name == "raw":
        return image.convert("RGB")
    try:
        return _opencv_preprocess(image, cfg)
    except Exception:
        gray = ImageOps.grayscale(image)
        if cfg.autocontrast or cfg.clahe_clip_limit is not None:
            gray = ImageOps.autocontrast(gray)
        if cfg.sharpen_alpha > 0:
            gray = ImageEnhance.Sharpness(gray).enhance(1.0 + cfg.sharpen_alpha)
        return Image.merge("RGB", (gray, gray, gray))


def iter_named_configs(names: Iterable[str] | None = None) -> list[PreprocessingConfig]:
    if names is None:
        return list(SWEEP_CONFIGS)
    return [get_preprocessing_config(name) for name in names]

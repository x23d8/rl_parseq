"""Frozen PARSeq, OCR uncertainty and low-level image features."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


QUALITY_FEATURE_NAMES = (
    "log_width",
    "log_height",
    "aspect",
    "brightness",
    "contrast",
    "sharpness_log",
    "noise",
    "saturation",
    "dark_fraction",
)

OCR_FEATURE_NAMES = (
    "prediction_length",
    "mean_top1_probability",
    "minimum_top1_probability",
    "mean_entropy",
    "mean_top1_top2_margin",
    "normalized_log_confidence",
)


def image_quality_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    values = (
        np.log1p(image.width),
        np.log1p(image.height),
        image.width / max(image.height, 1),
        gray.mean() / 255.0,
        gray.std() / 128.0,
        np.log1p(sharpness) / 12.0,
        np.median(np.abs(residual)) / 32.0,
        hsv[..., 1].mean() / 255.0,
        float((gray < 50).mean()),
    )
    return np.asarray(values, dtype=np.float32)


@torch.inference_mode()
def parseq_state_features(
    model,
    images: torch.Tensor,
    predictions: list[str],
    logits: torch.Tensor,
):
    """Return encoder pooling plus compact token uncertainty features."""

    memory = model.model.encode(images)
    pooled = torch.cat((memory.mean(dim=1), memory.amax(dim=1)), dim=1)
    probabilities = logits.softmax(dim=-1)
    top2 = probabilities.topk(2, dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    rows = []
    for index, prediction in enumerate(predictions):
        token_count = min(len(prediction) + 1, probabilities.shape[1])
        token_top1 = top2[index, :token_count, 0]
        token_margin = token_top1 - top2[index, :token_count, 1]
        normalized_log_confidence = token_top1.clamp_min(1e-8).log().mean()
        rows.append(
            torch.stack(
                (
                    torch.tensor(float(len(prediction)) / 12.0, device=images.device),
                    token_top1.mean(),
                    token_top1.min(),
                    entropy[index, :token_count].mean() / np.log(probabilities.shape[-1]),
                    token_margin.mean(),
                    normalized_log_confidence,
                )
            )
        )
    return torch.cat((pooled, torch.stack(rows)), dim=1).float()

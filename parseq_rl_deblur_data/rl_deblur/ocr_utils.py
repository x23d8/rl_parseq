"""Shared helpers to load the fine-tuned PARSeq checkpoint and run OCR.

Used both as an *evaluation* metric (evaluate.py) and, optionally, as part of
the *training* reward (train.py) -- to reward-shape the RL policy toward
plates that PARSeq actually reads correctly, not just toward low pixel error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "parseq") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "parseq"))
if str(REPO_ROOT / "preprocessing_best_config") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "preprocessing_best_config"))

from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.utils import create_model  # noqa: E402

DEFAULT_OCR_CKPT = REPO_ROOT / "outputs" / "refinement_finetune" / "best_official_parseq_anpr.pt"
ANPR_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize_plate_text(text: object) -> str:
    return "".join(ch for ch in str(text).upper() if ch in ANPR_CHARSET)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_ocr_model(device, checkpoint_path: str | Path = DEFAULT_OCR_CKPT):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = create_model(
        cfg["experiment"], pretrained=False, decode_ar=cfg["decode_ar"],
        refine_iters=cfg["refine_iters"], charset_test=cfg["charset_test"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model._device = device
    model.to(device).eval()
    img_size = tuple(cfg["img_size"])
    transform = SceneTextDataModule.get_transform(img_size, augment=False)
    return model, transform


@torch.no_grad()
def ocr_predict(model, transform, gray_u8_batch, device) -> list[str]:
    preds, _ = ocr_predict_with_confidence(model, transform, gray_u8_batch, device)
    return preds


@torch.no_grad()
def ocr_predict_with_confidence(model, transform, gray_u8_batch, device) -> tuple[list[str], list[float]]:
    """Returns (normalized predictions, log-confidence = sum of log token-probs)."""
    tensors = []
    for gray in gray_u8_batch:
        rgb = Image.fromarray(gray).convert("RGB")
        tensors.append(transform(rgb))
    batch = torch.stack(tensors).to(device)
    logits = model(batch)
    probs = logits.softmax(-1)
    preds, token_probs = model.tokenizer.decode(probs)
    preds = [normalize_plate_text(p) for p in preds]
    log_confidences = [float(torch.log(tp.clamp_min(1e-8)).sum().item()) for tp in token_probs]
    return preds, log_confidences

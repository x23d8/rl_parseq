"""Evaluate the trained RL deblurring agent against baselines.

Compares, on the held-out test split:
  1. blurred (no restoration)          -- lower bound
  2. classical static unsharp mask     -- non-RL baseline
  3. RL agent (greedy policy)          -- our method
  4. clean ground truth                -- upper bound

Metrics: PSNR / SSIM (image quality) and PARSeq OCR exact-match / CER
(downstream task quality, using the fine-tuned checkpoint already in this repo).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rl_deblur import env
from rl_deblur.model import FCNActorCritic
from rl_deblur.ocr_utils import DEFAULT_OCR_CKPT, edit_distance, load_ocr_model, normalize_plate_text, ocr_predict
from rl_deblur.train import BlurPairDataset, psnr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "outputs" / "rl_deblur" / "dataset"
DEFAULT_AGENT_CKPT = REPO_ROOT / "outputs" / "rl_deblur" / "checkpoints" / "best_deblur_agent.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rl_deblur"


def classical_unsharp_baseline(img_u8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img_u8, (0, 0), 1.0)
    sharp = img_u8.astype(np.float32) + 1.5 * (img_u8.astype(np.float32) - blur.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


@torch.no_grad()
def rl_restore_batch(model: FCNActorCritic, blurred: np.ndarray, num_steps: int, device: torch.device) -> np.ndarray:
    state = blurred.copy()
    for _ in range(num_steps):
        state_t = torch.from_numpy(state / 255.0).unsqueeze(1).float().to(device)
        logits, _ = model(state_t)
        action_map = logits.argmax(dim=1).cpu().numpy()
        state, _ = env.step(state, action_map, state)  # clean unused when just applying action
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--agent-checkpoint", default=str(DEFAULT_AGENT_CKPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    agent_ckpt = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
    agent_cfg = agent_ckpt["config"]
    agent = FCNActorCritic(channels=agent_cfg["channels"], rmc_kernel_size=agent_cfg.get("rmc_kernel_size", 9)).to(device)
    agent.load_state_dict(agent_ckpt["model_state_dict"])
    agent.eval()
    num_steps = agent_cfg["num_steps"]

    ocr_model, ocr_transform = load_ocr_model(device, DEFAULT_OCR_CKPT)

    test_ds = BlurPairDataset(dataset_dir, "test", limit=args.limit)
    manifest = test_ds.frame
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    idx = 0
    for blurred, clean, _labels in tqdm(loader, desc="evaluate"):
        blurred_np = blurred.numpy()
        clean_np = clean.numpy()
        batch_labels = manifest["label"].iloc[idx: idx + len(blurred_np)].tolist()
        blur_kinds = manifest["blur_kind"].iloc[idx: idx + len(blurred_np)].tolist()
        idx += len(blurred_np)

        classical_np = np.stack([classical_unsharp_baseline(b.astype(np.uint8)) for b in blurred_np])
        rl_np = rl_restore_batch(agent, blurred_np, num_steps, device)

        blurred_u8 = np.clip(blurred_np, 0, 255).astype(np.uint8)
        clean_u8 = np.clip(clean_np, 0, 255).astype(np.uint8)
        rl_u8 = np.clip(rl_np, 0, 255).astype(np.uint8)

        preds_blurred = ocr_predict(ocr_model, ocr_transform, blurred_u8, device)
        preds_classical = ocr_predict(ocr_model, ocr_transform, classical_np, device)
        preds_rl = ocr_predict(ocr_model, ocr_transform, rl_u8, device)
        preds_clean = ocr_predict(ocr_model, ocr_transform, clean_u8, device)

        for i in range(len(blurred_np)):
            label = normalize_plate_text(batch_labels[i])
            rows.append({
                "blur_kind": blur_kinds[i],
                "label": label,
                "psnr_blurred": psnr(blurred_np[i], clean_np[i]),
                "psnr_classical": psnr(classical_np[i].astype(np.float32), clean_np[i]),
                "psnr_rl": psnr(rl_np[i], clean_np[i]),
                "ssim_blurred": ssim_fn(clean_u8[i], blurred_u8[i], data_range=255),
                "ssim_classical": ssim_fn(clean_u8[i], classical_np[i], data_range=255),
                "ssim_rl": ssim_fn(clean_u8[i], rl_u8[i], data_range=255),
                "pred_blurred": preds_blurred[i],
                "pred_classical": preds_classical[i],
                "pred_rl": preds_rl[i],
                "pred_clean": preds_clean[i],
                "exact_blurred": preds_blurred[i] == label,
                "exact_classical": preds_classical[i] == label,
                "exact_rl": preds_rl[i] == label,
                "exact_clean": preds_clean[i] == label,
                "cer_blurred": edit_distance(preds_blurred[i], label) / max(len(label), 1),
                "cer_classical": edit_distance(preds_classical[i], label) / max(len(label), 1),
                "cer_rl": edit_distance(preds_rl[i], label) / max(len(label), 1),
                "cer_clean": edit_distance(preds_clean[i], label) / max(len(label), 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "eval_test_predictions.csv", index=False)

    summary = {
        "num_samples": len(df),
        "psnr": {k: float(df[f"psnr_{k}"].mean()) for k in ["blurred", "classical", "rl"]},
        "ssim": {k: float(df[f"ssim_{k}"].mean()) for k in ["blurred", "classical", "rl"]},
        "exact_acc": {k: float(df[f"exact_{k}"].mean()) for k in ["blurred", "classical", "rl", "clean"]},
        "cer": {k: float(df[f"cer_{k}"].mean()) for k in ["blurred", "classical", "rl", "clean"]},
        "by_blur_kind": {
            kind: {
                "psnr_blurred": float(g["psnr_blurred"].mean()),
                "psnr_rl": float(g["psnr_rl"].mean()),
                "exact_blurred": float(g["exact_blurred"].mean()),
                "exact_rl": float(g["exact_rl"].mean()),
            }
            for kind, g in df.groupby("blur_kind")
        },
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Cache PARSeq encoder/OCR observations for every Phase 6 candidate view.

The script refuses the audited test split. Its default output is colocated with
the Phase 6 architecture under ``reinforcement_learning``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from rl_restoration.actions import DEFAULT_ACTIONS, RestorationAction  # noqa: E402
from rl_restoration.features import image_quality_features, parseq_state_features  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import normalize_plate_text  # noqa: E402


class CandidateDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, action: RestorationAction, image_size: tuple[int, int]):
        self.frame = frame.reset_index(drop=True)
        self.action = action
        self.image_size = tuple(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(str(row.image_path)) as opened:
            restored = self.action.apply(opened.convert("RGB"))
        quality = image_quality_features(restored)
        resized = TF.resize(restored, list(self.image_size), interpolation=InterpolationMode.BICUBIC)
        tensor = TF.normalize(TF.to_tensor(resized), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor, quality


def collate(batch):
    images, quality = zip(*batch)
    return torch.stack(images), np.stack(quality).astype(np.float32)


@torch.inference_mode()
def encode_action(model, model_cfg, frame: pd.DataFrame, action: RestorationAction, args) -> np.ndarray:
    loader = DataLoader(
        CandidateDataset(frame, action, tuple(model_cfg.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    batches = []
    for images, quality in tqdm(loader, desc=f"{args.split}: {action.name}", leave=False):
        images = images.to(args.device_obj, non_blocking=True)
        logits = model(images, max_length=model_cfg.max_label_length)
        predictions, _ = model.tokenizer.decode(logits.softmax(-1))
        predictions = [normalize_plate_text(value) for value in predictions]
        deep = parseq_state_features(model, images, predictions, logits).cpu().numpy()
        batches.append(np.concatenate((deep, quality), axis=1).astype(np.float32))
    return np.concatenate(batches, axis=0)


def run(args) -> dict:
    if args.split not in {"train", "val"}:
        raise ValueError("Candidate cache may only be built for train/val; audited test is forbidden")
    manifest = Path(args.manifest).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents and output_dir != HERE:
        raise ValueError("Phase 6 cache must remain inside reinforcement_learning/phase_6_candidate_oof_ppo")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    frame = frame[frame.split.astype(str).str.lower() == args.split].copy()
    frame = frame.drop_duplicates("image_path").reset_index(drop=True)
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_cfg, _ = load_notebook_checkpoint(checkpoint, args.device_obj, args.refine_iters)
    per_action = [encode_action(model, model_cfg, frame, action, args) for action in DEFAULT_ACTIONS]
    candidate_features = np.stack(per_action, axis=1)
    target = output_dir / f"{args.split}_candidate_features.npz"
    np.savez_compressed(
        target,
        candidate_features=candidate_features,
        image_paths=np.asarray(frame.image_path.astype(str), dtype=np.str_),
        action_names=np.asarray([action.name for action in DEFAULT_ACTIONS], dtype=np.str_),
    )
    summary = {
        "split": args.split,
        "samples": int(len(frame)),
        "actions": int(len(DEFAULT_ACTIONS)),
        "candidate_feature_dimension": int(candidate_features.shape[2]),
        "shape": list(candidate_features.shape),
        "checkpoint": str(checkpoint),
        "test_loaded": False,
        "output": str(target),
    }
    (output_dir / f"{args.split}_candidate_cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parent = ROOT / "outputs/phase3_controlled_aug_full_frozen_eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--checkpoint", default=str(parent / "best_phase3_parseq_anpr.pt"))
    parser.add_argument("--manifest", default=str(parent / "dataset_manifest.csv"))
    parser.add_argument("--output-dir", default=str(HERE / "results/candidate_cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


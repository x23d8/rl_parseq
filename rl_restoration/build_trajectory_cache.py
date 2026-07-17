"""Build an offline one-step trajectory cache for the restoration bandit.

PARSeq and all restoration tools are frozen.  Ground-truth labels are used only
to calculate rewards for train/validation.  Test is intentionally unsupported
by the default CLI so it cannot influence policy training or selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import edit_distance, normalize_plate_text  # noqa: E402
from rl_restoration.actions import DEFAULT_ACTIONS, RestorationAction, validate_action_space  # noqa: E402
from rl_restoration.features import image_quality_features, parseq_state_features  # noqa: E402
from rl_restoration.reward import RewardConfig, ocr_reward  # noqa: E402


class ActionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, action: RestorationAction, img_size: tuple[int, int]):
        self.frame = frame.reset_index(drop=True)
        self.action = action
        self.img_size = tuple(img_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        quality = image_quality_features(original)
        restored = self.action.apply(original)
        restored = TF.resize(restored, list(self.img_size), interpolation=InterpolationMode.BICUBIC)
        tensor = TF.normalize(TF.to_tensor(restored), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor, str(row["label"]), path, quality


def collate_batch(batch):
    images, labels, paths, quality = zip(*batch)
    return torch.stack(images), list(labels), list(paths), np.stack(quality)


@torch.inference_mode()
def predict_action(model, model_cfg, frame, action, args, collect_state=False):
    loader = DataLoader(
        ActionDataset(frame, action, tuple(model_cfg.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    rows = []
    features = []
    started = time.perf_counter()
    for images, labels, paths, quality in tqdm(loader, desc=f"cache {args.split}: {action.name}", leave=False):
        images = images.to(args.device_obj, non_blocking=True)
        logits = model(images, max_length=model_cfg.max_label_length)
        probabilities = logits.softmax(-1)
        predictions, token_probabilities = model.tokenizer.decode(probabilities)
        predictions = [normalize_plate_text(value) for value in predictions]
        confidences = [float(values.prod().item()) for values in token_probabilities]
        if collect_state:
            deep = parseq_state_features(model, images, predictions, logits).cpu().numpy()
            features.append(np.concatenate((deep, quality.astype(np.float32)), axis=1))
        for path, target, prediction, confidence in zip(paths, labels, predictions, confidences):
            target = normalize_plate_text(target)
            distance = edit_distance(prediction, target)
            rows.append(
                {
                    "image_path": path,
                    "target": target,
                    "action": action.name,
                    "action_cost": action.cost,
                    "prediction": prediction,
                    "confidence": confidence,
                    "normalized_confidence": math.exp(
                        math.log(max(confidence, 1e-12)) / max(len(prediction) + 1, 1)
                    ),
                    "edit_distance": distance,
                    "exact": prediction == target,
                    "target_length": max(len(target), 1),
                }
            )
    result = pd.DataFrame(rows)
    result["elapsed_action_seconds"] = time.perf_counter() - started
    return result, np.concatenate(features, axis=0) if features else None


def attach_rewards(predictions: pd.DataFrame, reward_cfg: RewardConfig):
    baseline = predictions[predictions["action"] == "stop_baseline"][
        ["image_path", "prediction", "edit_distance", "exact"]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "edit_distance": "baseline_edit_distance",
            "exact": "baseline_exact",
        }
    )
    result = predictions.merge(baseline, on="image_path", validate="many_to_one")
    result["reward"] = result.apply(
        lambda row: ocr_reward(
            row["baseline_prediction"],
            int(row["baseline_edit_distance"]),
            row["prediction"],
            int(row["edit_distance"]),
            row["target"],
            float(row["action_cost"]),
            reward_cfg,
        ),
        axis=1,
    )
    return result


def summarize_cache(predictions: pd.DataFrame, actions=DEFAULT_ACTIONS):
    rows = []
    for action in actions:
        subset = predictions[predictions["action"] == action.name]
        rows.append(
            {
                "action": action.name,
                "samples": len(subset),
                "exact_acc": float(subset["exact"].mean()),
                "char_acc": float(1.0 - subset["edit_distance"].sum() / subset["target_length"].sum()),
                "mean_reward": float(subset["reward"].mean()),
                "positive_reward_rate": float((subset["reward"] > 0).mean()),
                "cost": action.cost,
            }
        )
    return pd.DataFrame(rows).sort_values(["exact_acc", "char_acc"], ascending=False)


def oracle_predictions(predictions: pd.DataFrame):
    ranked = predictions.sort_values(
        ["image_path", "reward", "action_cost", "exact", "edit_distance"],
        ascending=[True, False, True, False, True],
    )
    return ranked.groupby("image_path", as_index=False).head(1).reset_index(drop=True)


def run(args):
    validate_action_space()
    if args.split == "test":
        locked_router = Path(args.locked_router_checkpoint).resolve() if args.locked_router_checkpoint else None
        if locked_router is None or not locked_router.exists():
            raise ValueError("Test cache requires --locked-router-checkpoint to prove the policy is already locked")
    elif args.split not in {"train", "val"}:
        raise ValueError("Unsupported cache split")
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    label_column = "label" if "label" in frame else "target"
    if not {"image_path", label_column}.issubset(frame.columns):
        raise ValueError(f"{manifest} requires image_path and label/target")
    if "split" in frame:
        frame = frame[frame["split"].astype(str).str.lower() == args.split].copy()
    frame = frame.rename(columns={label_column: "label"})
    if args.limit:
        frame = frame.head(args.limit).copy()
    frame["label"] = frame["label"].map(normalize_plate_text)
    frame = frame.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    model, model_cfg, _ = load_notebook_checkpoint(checkpoint, args.device_obj, args.refine_iters)
    all_predictions = []
    state_features = None
    for action in DEFAULT_ACTIONS:
        action_predictions, action_features = predict_action(
            model, model_cfg, frame, action, args, collect_state=action.name == "stop_baseline"
        )
        all_predictions.append(action_predictions)
        if action_features is not None:
            state_features = action_features
    predictions = attach_rewards(pd.concat(all_predictions, ignore_index=True), RewardConfig())
    predictions.to_csv(output_dir / f"{args.split}_action_trajectories.csv", index=False)
    np.savez_compressed(
        output_dir / f"{args.split}_state_features.npz",
        features=state_features,
        image_paths=np.asarray(frame["image_path"].astype(str).tolist(), dtype=np.str_),
        targets=np.asarray(frame["label"].astype(str).tolist(), dtype=np.str_),
    )
    action_summary = summarize_cache(predictions)
    action_summary.to_csv(output_dir / f"{args.split}_action_summary.csv", index=False)
    oracle = oracle_predictions(predictions)
    oracle.to_csv(output_dir / f"{args.split}_oracle_predictions.csv", index=False)
    summary = {
        "split": args.split,
        "samples": len(frame),
        "actions": [action.name for action in DEFAULT_ACTIONS],
        "feature_dimension": int(state_features.shape[1]),
        "baseline_exact": float(predictions[predictions.action == "stop_baseline"]["exact"].mean()),
        "oracle_exact": float(oracle["exact"].mean()),
        "oracle_char_acc": float(1.0 - oracle.edit_distance.sum() / oracle.target_length.sum()),
        "oracle_action_distribution": oracle["action"].value_counts().to_dict(),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
    }
    (output_dir / f"{args.split}_cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    default_run = ROOT / "outputs/phase3_controlled_aug_full_frozen_eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(default_run / "best_phase3_parseq_anpr.pt"))
    parser.add_argument("--manifest", default=str(default_run / "dataset_manifest.csv"))
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument(
        "--locked-router-checkpoint",
        default="",
        help="Required for test; no router hyperparameter may be changed after this checkpoint is supplied.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rl_restoration/trajectory_cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

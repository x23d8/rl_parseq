"""Fine-tune PARSeq weights with views selected by a locked RL router.

The router and restoration tools remain frozen.  PARSeq sees a controlled
mixture of baseline, raw and policy-selected train views.  Checkpoint selection
uses clean plus policy-view validation, with a hard clean-accuracy guard.  Test
is evaluated only after the checkpoint is locked.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import (  # noqa: E402
    edit_distance,
    greedy_decode,
    normalize_plate_text,
    parseq_plm_loss,
    set_decode_mode,
)
from rl_restoration.actions import action_by_name  # noqa: E402


class PolicyMixtureDataset(Dataset):
    def __init__(
        self,
        frame,
        img_size,
        policy_actions,
        hard_paths,
        baseline_probability=0.50,
        raw_probability=0.25,
        hard_policy_probability=0.70,
        hard_raw_probability=0.10,
    ):
        self.frame = frame.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.policy_actions = policy_actions
        self.hard_paths = set(hard_paths)
        self.baseline_probability = float(baseline_probability)
        self.raw_probability = float(raw_probability)
        self.hard_policy_probability = float(hard_policy_probability)
        self.hard_raw_probability = float(hard_raw_probability)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        draw = random.random()
        if path in self.hard_paths:
            baseline_probability = 1.0 - self.hard_policy_probability - self.hard_raw_probability
            raw_probability = self.hard_raw_probability
        else:
            baseline_probability = self.baseline_probability
            raw_probability = self.raw_probability
        if draw < baseline_probability:
            action_name, route = "stop_baseline", "baseline"
        elif draw < baseline_probability + raw_probability:
            action_name, route = "raw_rgb", "raw"
        else:
            action_name, route = self.policy_actions.get(path, "stop_baseline"), "policy"
        with Image.open(path) as opened:
            image = action_by_name(action_name).apply(opened.convert("RGB"))
        image = TF.resize(image, list(self.img_size), interpolation=InterpolationMode.BICUBIC)
        tensor = TF.normalize(TF.to_tensor(image), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor, str(row["label"]), path, {"route": route, "action": action_name}


class FixedActionDataset(Dataset):
    def __init__(self, frame, img_size, actions=None):
        self.frame = frame.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.actions = actions or {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        action_name = self.actions.get(path, "stop_baseline")
        with Image.open(path) as opened:
            image = action_by_name(action_name).apply(opened.convert("RGB"))
        image = TF.resize(image, list(self.img_size), interpolation=InterpolationMode.BICUBIC)
        tensor = TF.normalize(TF.to_tensor(image), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor, str(row["label"]), path, {"route": "eval", "action": action_name}


def collate_batch(batch):
    images, labels, paths, metadata = zip(*batch)
    return torch.stack(images), list(labels), list(paths), list(metadata)


def make_loader(dataset, batch_size, num_workers, sampler=None, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def tempered_sampler(frame, seed, hard_paths=(), hard_sample_weight=1.0, alpha=0.35, max_weight=4.0):
    groups = frame["plate_type"].fillna("unknown").astype(str)
    counts = groups.value_counts().to_dict()
    largest = max(counts.values())
    class_weights = {name: min(max_weight, (largest / count) ** alpha) for name, count in counts.items()}
    hard_paths = set(hard_paths)
    weights = torch.as_tensor(
        [
            class_weights[group] * (hard_sample_weight if str(path) in hard_paths else 1.0)
            for group, path in zip(groups, frame["image_path"])
        ],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=torch.Generator().manual_seed(seed)
    )


@torch.inference_mode()
def evaluate(model, loader, device, split, max_length):
    model.eval()
    rows = []
    for images, labels, paths, metadata in tqdm(loader, desc=f"eval {split}", leave=False):
        images = images.to(device, non_blocking=True)
        predictions, confidences = greedy_decode(model, images, max_length=max_length)
        for path, target, prediction, confidence, meta in zip(
            paths, labels, predictions, confidences.cpu().tolist(), metadata
        ):
            distance = edit_distance(prediction, target)
            rows.append(
                {
                    "image_path": path,
                    "target": target,
                    "prediction": prediction,
                    "exact": prediction == target,
                    "edit_distance": distance,
                    "confidence": confidence,
                    "action": meta["action"],
                }
            )
    frame = pd.DataFrame(rows)
    chars = frame.target.str.len().clip(lower=1).sum()
    metrics = {
        "split": split,
        "samples": len(frame),
        "exact_acc": float(frame.exact.mean()),
        "char_acc": float(1.0 - frame.edit_distance.sum() / chars),
        "cer": float(frame.edit_distance.sum() / chars),
    }
    return metrics, frame


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_encoder_trainable(model, trainable):
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = trainable


def train_epoch(model, loader, optimizer, scaler, device, grad_clip):
    model.train()
    loss_sum = 0.0
    samples = 0
    route_counts = Counter()
    action_counts = Counter()
    for images, labels, _paths, metadata in tqdm(loader, desc="train policy mixture", leave=False):
        for meta in metadata:
            route_counts[meta["route"]] += 1
            action_counts[meta["action"]] += 1
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = parseq_plm_loss(model, images, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().item()) * len(labels)
        samples += len(labels)
    return loss_sum / max(samples, 1), route_counts, action_counts


def save_checkpoint(path, model, model_cfg, args, epoch, clean_metrics, policy_metrics, score):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model_cfg),
            "rl_weight_finetune_config": vars(args),
            "epoch": epoch,
            "metrics": {"clean": clean_metrics, "policy": policy_metrics, "composite_score": score},
            "architecture": "official_strhub_parseq_rl_policy_mixture",
            "parent_checkpoint": args.checkpoint,
            "router_checkpoint": args.router_checkpoint,
        },
        path,
    )


def run(args):
    set_seed(args.seed)
    if args.baseline_probability < 0 or args.raw_probability < 0 or args.baseline_probability + args.raw_probability > 1:
        raise ValueError("Normal baseline/raw probabilities must be non-negative and sum to at most 1")
    if args.hard_policy_probability < 0 or args.hard_raw_probability < 0 or args.hard_policy_probability + args.hard_raw_probability > 1:
        raise ValueError("Hard policy/raw probabilities must be non-negative and sum to at most 1")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest["label"] = manifest["label"].map(normalize_plate_text)
    train_frame = manifest[manifest.split.astype(str).str.lower() == "train"].copy()
    val_frame = manifest[manifest.split.astype(str).str.lower() == "val"].copy()
    test_frame = manifest[manifest.split.astype(str).str.lower() == "test"].copy()

    train_selection = pd.read_csv(args.train_policy_selections)
    val_selection = pd.read_csv(args.val_policy_predictions)
    train_action_column = "selected_action" if "selected_action" in train_selection else "final_action"
    val_action_column = "selected_action" if "selected_action" in val_selection else "final_action"
    if train_action_column not in train_selection or val_action_column not in val_selection:
        raise ValueError("Policy selections require selected_action or final_action")
    train_actions = dict(zip(train_selection.image_path.astype(str), train_selection[train_action_column].astype(str)))
    val_actions = dict(zip(val_selection.image_path.astype(str), val_selection[val_action_column].astype(str)))
    fixed_mask = train_selection["fixed"]
    if fixed_mask.dtype != bool:
        fixed_mask = fixed_mask.astype(str).str.lower().isin(("true", "1", "yes"))
    hard_paths = set(train_selection.loc[fixed_mask, "image_path"].astype(str))
    if set(train_frame.image_path.astype(str)) - set(train_actions):
        raise ValueError("Router train selections do not cover the train manifest")
    if set(val_frame.image_path.astype(str)) - set(val_actions):
        raise ValueError("Router validation predictions do not cover the val manifest")

    model, model_cfg, _ = load_notebook_checkpoint(Path(args.checkpoint), device, args.refine_iters)
    model_cfg.pretrained = False
    train_ds = PolicyMixtureDataset(
        train_frame,
        model_cfg.img_size,
        train_actions,
        hard_paths,
        baseline_probability=args.baseline_probability,
        raw_probability=args.raw_probability,
        hard_policy_probability=args.hard_policy_probability,
        hard_raw_probability=args.hard_raw_probability,
    )
    clean_val_ds = FixedActionDataset(val_frame, model_cfg.img_size)
    policy_val_ds = FixedActionDataset(val_frame, model_cfg.img_size, val_actions)
    clean_test_ds = FixedActionDataset(test_frame, model_cfg.img_size)
    train_loader = make_loader(
        train_ds,
        args.batch_size,
        args.num_workers,
        sampler=tempered_sampler(
            train_frame,
            args.seed,
            hard_paths=hard_paths,
            hard_sample_weight=args.hard_sample_weight,
        ),
    )
    clean_val_loader = make_loader(clean_val_ds, args.batch_size, args.num_workers)
    policy_val_loader = make_loader(policy_val_ds, args.batch_size, args.num_workers)
    clean_test_loader = make_loader(clean_test_ds, args.batch_size, args.num_workers)

    set_decode_mode(model, args.refine_iters, True)
    parent_clean, parent_clean_predictions = evaluate(
        model, clean_val_loader, device, "val_clean_parent", model_cfg.max_label_length
    )
    parent_policy, parent_policy_predictions = evaluate(
        model, policy_val_loader, device, "val_policy_parent", model_cfg.max_label_length
    )
    parent_clean_predictions.to_csv(output_dir / "val_clean_parent_predictions.csv", index=False)
    parent_policy_predictions.to_csv(output_dir / "val_policy_parent_predictions.csv", index=False)
    parent_score = args.clean_score_weight * parent_clean["exact_acc"] + (1 - args.clean_score_weight) * parent_policy["exact_acc"]
    clean_floor = parent_clean["exact_acc"] - args.max_clean_drop_images / len(val_frame)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.freeze_encoder_epochs, 1), eta_min=args.learning_rate * 0.10
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_path = output_dir / "best_parseq_rl_policy_mixture.pt"
    save_checkpoint(best_path, model, model_cfg, args, 0, parent_clean, parent_policy, parent_score)
    best_key = (parent_score, parent_clean["exact_acc"], parent_clean["char_acc"], parent_policy["char_acc"])
    best_epoch = 0
    stale = 0
    history = []
    audit_rows = []

    for epoch in range(1, args.epochs + 1):
        encoder_trainable = epoch > args.freeze_encoder_epochs
        set_encoder_trainable(model, encoder_trainable)
        started = time.perf_counter()
        loss, route_counts, action_counts = train_epoch(
            model, train_loader, optimizer, scaler, device, args.grad_clip
        )
        set_decode_mode(model, args.refine_iters, True)
        clean_metrics, _ = evaluate(model, clean_val_loader, device, "val_clean", model_cfg.max_label_length)
        policy_metrics, _ = evaluate(model, policy_val_loader, device, "val_policy", model_cfg.max_label_length)
        score = args.clean_score_weight * clean_metrics["exact_acc"] + (1 - args.clean_score_weight) * policy_metrics["exact_acc"]
        eligible = clean_metrics["exact_acc"] >= clean_floor
        key = (score, clean_metrics["exact_acc"], clean_metrics["char_acc"], policy_metrics["char_acc"])
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "encoder_trainable": encoder_trainable,
            "clean_floor_passed": eligible,
            "composite_score": score,
            "seconds": time.perf_counter() - started,
            **{f"clean_{name}": value for name, value in clean_metrics.items()},
            **{f"policy_{name}": value for name, value in policy_metrics.items()},
        }
        history.append(row)
        total = sum(route_counts.values())
        for name, count in route_counts.items():
            audit_rows.append({"epoch": epoch, "kind": "route", "name": name, "count": count, "rate": count / total})
        for name, count in action_counts.items():
            audit_rows.append({"epoch": epoch, "kind": "action", "name": name, "count": count, "rate": count / total})
        print(json.dumps(row, ensure_ascii=False))
        if eligible and key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            save_checkpoint(best_path, model, model_cfg, args, epoch, clean_metrics, policy_metrics, score)
        else:
            stale += 1
        if encoder_trainable:
            scheduler.step()
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(output_dir / "mixture_audit.csv", index=False)
        if stale >= args.early_stopping_patience:
            break

    best_model, best_cfg, best_payload = load_notebook_checkpoint(best_path, device, args.refine_iters)
    final_clean_val, final_clean_val_predictions = evaluate(
        best_model, clean_val_loader, device, "val_clean_locked", best_cfg.max_label_length
    )
    final_policy_val, final_policy_val_predictions = evaluate(
        best_model, policy_val_loader, device, "val_policy_locked", best_cfg.max_label_length
    )
    final_test, final_test_predictions = evaluate(
        best_model, clean_test_loader, device, "test_clean_locked", best_cfg.max_label_length
    )
    final_clean_val_predictions.to_csv(output_dir / "val_clean_locked_predictions.csv", index=False)
    final_policy_val_predictions.to_csv(output_dir / "val_policy_locked_predictions.csv", index=False)
    final_test_predictions.to_csv(output_dir / "test_clean_locked_predictions.csv", index=False)
    summary = {
        "algorithm": "locked_rl_policy_then_parseq_hard_aware_mixture",
        "best_epoch": best_epoch,
        "parent_clean_validation": parent_clean,
        "parent_policy_validation": parent_policy,
        "locked_clean_validation": final_clean_val,
        "locked_policy_validation": final_policy_val,
        "locked_clean_test": final_test,
        "clean_floor": clean_floor,
        "best_checkpoint": str(best_path),
        "policy_checkpoint": args.router_checkpoint,
        "hard_train_samples": len(hard_paths),
        "test_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parent = ROOT / "outputs/phase3_controlled_aug_full_frozen_eval"
    router = ROOT / "outputs/rl_restoration/router_seed_123"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(parent / "best_phase3_parseq_anpr.pt"))
    parser.add_argument("--manifest", default=str(parent / "dataset_manifest.csv"))
    parser.add_argument("--router-checkpoint", default=str(router / "best_reward_router.pt"))
    parser.add_argument("--train-policy-selections", default=str(router / "train_policy_selections.csv"))
    parser.add_argument("--val-policy-predictions", default=str(router / "val_policy_predictions.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rl_restoration/parseq_policy_mixture"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--baseline-probability", type=float, default=0.50)
    parser.add_argument("--raw-probability", type=float, default=0.25)
    parser.add_argument("--hard-policy-probability", type=float, default=0.70)
    parser.add_argument("--hard-raw-probability", type=float, default=0.10)
    parser.add_argument("--hard-sample-weight", type=float, default=4.0)
    parser.add_argument("--clean-score-weight", type=float, default=0.60)
    parser.add_argument("--max-clean-drop-images", type=int, default=1)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

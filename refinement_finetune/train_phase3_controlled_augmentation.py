"""Phase 3: continue fine-tuning PARSeq with controlled OCR augmentations.

This script follows the training/evaluation contract of
``PARSeq_Official_ANPR_Refinement_Finetune.ipynb`` while keeping validation and
test images deterministic.  It starts from the best existing fine-tuned
checkpoint and writes enough manifests/statistics to audit the experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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
TRAIN_DIR = ROOT / "train_no_refinement"
PARSEQ_DIR = ROOT / "parseq"
PREPROCESSING_DIR = ROOT / "preprocessing_best_config"
for import_path in (ROOT, TRAIN_DIR, PARSEQ_DIR, PREPROCESSING_DIR, Path(__file__).resolve().parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from phase3_controlled_augmentation import (  # noqa: E402
    ControlledPlateAugmenter,
    Phase3AugmentationConfig,
    count_augmentation_trace,
    trace_rates,
)
from find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from preprocessing import get_preprocessing_config, preprocess_plate_image  # noqa: E402
from parseq_official_anpr_pipeline import (  # noqa: E402
    edit_distance,
    greedy_decode,
    normalize_plate_text,
    parseq_plm_loss,
    set_decode_mode,
)


DEFAULT_CHECKPOINT = ROOT / "outputs/refinement_finetune/best_official_parseq_anpr.pt"


@dataclass
class Phase3TrainingConfig:
    checkpoint: str
    output_dir: str
    data_root: str = str(ROOT / "dataset")
    val_manifest: str = ""
    test_manifest: str = ""
    policy_profile: str = "full"
    epochs: int = 12
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 3e-6
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    refine_iters: int = 1
    max_label_length: int = 12
    seed: int = 20260715
    amp: bool = True
    balance_alpha: float = 0.5
    max_sample_weight: float = 8.0
    freeze_encoder_epochs: int = 1
    early_stopping_patience: int = 4
    max_train_samples: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _is_nonempty(value: object) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _pick_reviewed_label(row: pd.Series) -> str:
    statuses = [str(row.get(name, "")).strip().lower() for name in ("review_status", "label_status")]
    if "rejected" in statuses:
        return ""
    for name in ("corrected_label", "extracted_character", "prediction", "raw_prediction", "original_label", "text", "label"):
        if name in row and _is_nonempty(row[name]):
            label = normalize_plate_text(row[name])
            if label:
                return label
    return ""


def _split_unsplit_frame(frame: pd.DataFrame, seed: int, val_ratio: float = 0.1, test_ratio: float = 0.1) -> pd.DataFrame:
    """Reproduce the notebook's deterministic, plate-type-stratified split."""

    frame = frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    splits = np.full(len(frame), "train", dtype=object)
    groups = frame.groupby("plate_type", sort=True).indices.values() if "plate_type" in frame else [np.arange(len(frame))]
    for positions in groups:
        indices = np.asarray(list(positions), dtype=int)
        rng.shuffle(indices)
        count = len(indices)
        n_test = max(1, int(round(count * test_ratio))) if count >= 3 else 0
        n_val = max(1, int(round(count * val_ratio))) if count >= 3 else 0
        if n_test + n_val >= count:
            n_test = 1 if count >= 3 else 0
            n_val = 1 if count - n_test >= 2 else 0
        splits[indices[:n_test]] = "test"
        splits[indices[n_test : n_test + n_val]] = "val"
    frame["split"] = splits
    return frame


def _finalize_frame(frame: pd.DataFrame, source_name: str, max_label_length: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["label"] = frame["label"].map(normalize_plate_text)
    frame = frame[frame["label"].str.len().between(1, max_label_length)]
    frame["image_path"] = frame["image_path"].map(lambda value: str(Path(str(value))))
    exists = frame["image_path"].map(lambda value: Path(value).exists())
    if not bool(exists.all()):
        print(f"warning: skip {int((~exists).sum())} missing images from {source_name}")
        frame = frame[exists]
    return frame.reset_index(drop=True)


def _read_prepared_source(source: dict, max_label_length: int) -> pd.DataFrame:
    root = _resolve_path(ROOT, source["path"])
    frames = []
    for split in ("train", "val", "test"):
        csv_path = root / f"{split}.csv"
        if not csv_path.exists():
            csv_path = root / "splits" / f"{split}.csv"
        if not csv_path.exists():
            continue
        raw = pd.read_csv(csv_path)
        image_col = "image_path" if "image_path" in raw else "image"
        if image_col not in raw or "label" not in raw:
            raise ValueError(f"{csv_path} requires image_path/image and label columns")
        frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "image_path": raw[image_col].map(lambda value: str(_resolve_path(root, value))),
                    "label": raw["label"],
                    "source_name": source.get("name", root.name),
                    "plate_type": raw.get("plate_type", source.get("plate_type", source.get("name", root.name))),
                    "label_status": raw.get("label_status", "prepared"),
                    "review_status": raw.get("review_status", "prepared"),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No train/val/test CSV found in {root}")
    return _finalize_frame(pd.concat(frames, ignore_index=True), source.get("name", root.name), max_label_length)


def _read_review_source(source: dict, max_label_length: int, seed: int) -> pd.DataFrame:
    root = _resolve_path(ROOT, source["path"])
    csv_path = root / "labels.csv"
    raw = pd.read_csv(csv_path)
    image_col = "image" if "image" in raw else "image_path"
    frame = pd.DataFrame(
        {
            "image_path": raw[image_col].map(lambda value: str(_resolve_path(root, value))),
            "label": raw.apply(_pick_reviewed_label, axis=1),
            "source_name": source.get("name", root.name),
            "plate_type": source.get("plate_type", source.get("name", root.name)),
            "label_status": raw.get("label_status", "reviewed"),
            "review_status": raw.get("review_status", "reviewed"),
        }
    )
    if "split" in raw:
        frame["split"] = raw["split"].astype(str).str.lower()
    else:
        frame = _split_unsplit_frame(frame, seed)
    return _finalize_frame(frame, source.get("name", root.name), max_label_length)


def _iter_label_directories(root: Path):
    if (root / "labels.txt").exists():
        yield root
    for child in sorted(root.iterdir() if root.exists() else []):
        if child.is_dir() and (child / "labels.txt").exists():
            yield child


def _read_labels_txt_source(source: dict, max_label_length: int, seed: int) -> pd.DataFrame:
    root = _resolve_path(ROOT, source["path"])
    rows = []
    for directory in _iter_label_directories(root):
        plate_type = source.get("plate_type") if directory == root else directory.name
        for line in (directory / "labels.txt").read_text(encoding="utf-8").splitlines():
            parts = line.strip().replace("\t", " ").split()
            if len(parts) < 2:
                continue
            rows.append(
                {
                    "image_path": str(directory / parts[0]),
                    "label": parts[1],
                    "source_name": source.get("name", root.name),
                    "plate_type": plate_type or directory.name,
                    "label_status": "labels_txt",
                    "review_status": "labels_txt",
                }
            )
    if not rows:
        raise FileNotFoundError(f"No labels.txt found under {root}")
    frame = _split_unsplit_frame(pd.DataFrame(rows), seed)
    return _finalize_frame(frame, source.get("name", root.name), max_label_length)


def build_dataset_frame(saved_config: dict, cfg: Phase3TrainingConfig) -> pd.DataFrame:
    sources = saved_config.get("dataset_sources") or [
        {"name": "data_root", "path": cfg.data_root, "format": "prepared_csv", "plate_type": "normal"}
    ]
    readers = {
        "prepared_csv": lambda source: _read_prepared_source(source, cfg.max_label_length),
        "collected_review_csv": lambda source: _read_review_source(source, cfg.max_label_length, saved_config.get("seed", 42)),
        "labels_txt": lambda source: _read_labels_txt_source(source, cfg.max_label_length, saved_config.get("seed", 42)),
        "color_filtered_labels_txt": lambda source: _read_labels_txt_source(source, cfg.max_label_length, saved_config.get("seed", 42)),
    }
    frames = []
    for source in sources:
        source = dict(source)
        if source.get("name") == "data_root":
            source["path"] = cfg.data_root
        fmt = source.get("format", "prepared_csv")
        if fmt not in readers:
            raise ValueError(f"Unsupported dataset format: {fmt}")
        frame = readers[fmt](source)
        print(f"loaded {len(frame):5d} rows from {source.get('name')} ({fmt})")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset=["image_path", "label"]).reset_index(drop=True)


def freeze_evaluation_splits(
    frame: pd.DataFrame,
    checkpoint_path: Path,
    val_manifest: str = "",
    test_manifest: str = "",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace mutable val/test labels with the parent run's locked manifests."""

    explicit = {"val": val_manifest, "test": test_manifest}
    filenames = {
        "val": "eval_val_predictions_best_refine.csv",
        "test": "eval_test_predictions_best_refine.csv",
    }
    resolved: dict[str, str] = {}
    frozen_parts = [frame[frame["split"].astype(str).str.lower() == "train"].copy()]
    for split in ("val", "test"):
        manifest = Path(explicit[split]).resolve() if explicit[split] else checkpoint_path.parent / filenames[split]
        if not manifest.exists():
            print(f"warning: no locked {split} manifest at {manifest}; using current dataset split")
            frozen_parts.append(frame[frame["split"].astype(str).str.lower() == split].copy())
            continue
        source = pd.read_csv(manifest)
        if not {"image_path", "target"}.issubset(source.columns):
            raise ValueError(f"{manifest} requires image_path and target columns")
        frozen = pd.DataFrame(
            {
                "split": split,
                "image_path": source["image_path"].astype(str),
                "label": source["target"].map(normalize_plate_text),
                "source_name": source.get("source_name", "locked_manifest"),
                "plate_type": source.get("plate_type", "unknown"),
                "label_status": source.get("label_status", "locked_manifest"),
                "review_status": source.get("review_status", "locked_manifest"),
            }
        )
        exists = frozen["image_path"].map(lambda value: Path(value).exists())
        if not bool(exists.all()):
            raise FileNotFoundError(f"Locked {split} manifest has {int((~exists).sum())} missing images")
        frozen_parts.append(frozen)
        resolved[split] = str(manifest)
    return pd.concat(frozen_parts, ignore_index=True), resolved


class Phase3PlateDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        img_size: Sequence[int],
        augmenter: ControlledPlateAugmenter | None,
        limit: int | None = None,
    ):
        subset = frame[frame["split"].astype(str).str.lower() == split].copy()
        if limit is not None:
            subset = subset.head(int(limit))
        self.frame = subset.reset_index(drop=True)
        self.split = split
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.augmenter = augmenter
        self.eval_config = get_preprocessing_config("train_baseline")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if self.augmenter is not None:
            image, trace = self.augmenter(image)
        else:
            image = preprocess_plate_image(image, self.eval_config)
            trace = ("eval_deterministic", "preprocess:train_baseline")
        image = TF.resize(image, list(self.img_size), interpolation=InterpolationMode.BICUBIC)
        tensor = TF.normalize(TF.to_tensor(image), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        metadata = {
            "split": self.split,
            "source_name": str(row.get("source_name", "")),
            "plate_type": str(row.get("plate_type", "")),
            "augmentation_trace": trace,
        }
        return tensor, str(row["label"]), path, metadata


def collate_batch(batch):
    images, labels, paths, metadata = zip(*batch)
    return torch.stack(list(images)), list(labels), list(paths), list(metadata)


def _balanced_sampler(frame: pd.DataFrame, alpha: float, max_weight: float, seed: int):
    """Tempered class balancing without repeating a one-image class hundreds of times."""

    groups = frame["plate_type"].fillna("unknown").astype(str)
    counts = groups.value_counts().to_dict()
    largest = max(counts.values())
    class_weights = {
        name: min(float(max_weight), (largest / max(count, 1)) ** float(alpha))
        for name, count in counts.items()
    }
    weights = torch.as_tensor([class_weights[name] for name in groups], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator), class_weights


def make_loaders(frame: pd.DataFrame, model_cfg, cfg: Phase3TrainingConfig, augmentation_cfg):
    augmenter = ControlledPlateAugmenter(augmentation_cfg)
    train_ds = Phase3PlateDataset(frame, "train", model_cfg.img_size, augmenter, cfg.max_train_samples)
    val_ds = Phase3PlateDataset(frame, "val", model_cfg.img_size, None)
    test_ds = Phase3PlateDataset(frame, "test", model_cfg.img_size, None)
    sampler, class_weights = _balanced_sampler(
        train_ds.frame, cfg.balance_alpha, cfg.max_sample_weight, cfg.seed
    )
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    return (
        DataLoader(train_ds, sampler=sampler, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
        {"train": train_ds, "val": val_ds, "test": test_ds},
        class_weights,
    )


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, split: str, max_length: int):
    model.eval()
    rows = []
    for images, labels, paths, metadata in tqdm(loader, desc=f"eval {split}", leave=False):
        images = images.to(device, non_blocking=True)
        predictions, confidences = greedy_decode(model, images, max_length=max_length)
        for path, target, prediction, confidence, meta in zip(
            paths, labels, predictions, confidences.detach().cpu().tolist(), metadata
        ):
            distance = edit_distance(prediction, target)
            rows.append(
                {
                    "image_path": path,
                    "target": target,
                    "prediction": prediction,
                    "exact": prediction == target,
                    "edit_distance": distance,
                    "confidence": float(confidence),
                    "source_name": meta["source_name"],
                    "plate_type": meta["plate_type"],
                }
            )
    predictions = pd.DataFrame(rows)
    edits = int(predictions["edit_distance"].sum()) if len(predictions) else 0
    chars = int(predictions["target"].str.len().clip(lower=1).sum()) if len(predictions) else 0
    metrics = {
        "split": split,
        "samples": int(len(predictions)),
        "exact_acc": float(predictions["exact"].mean()) if len(predictions) else 0.0,
        "cer": float(edits / max(chars, 1)),
        "char_acc": float(1.0 - edits / max(chars, 1)),
        "refine_iters": int(model.model.refine_iters),
    }
    return metrics, predictions


def train_one_epoch(model, loader, optimizer, scaler, cfg, device, epoch):
    model.train()
    totals = {"loss": 0.0, "samples": 0}
    audit = Counter()
    for images, labels, _paths, metadata in tqdm(loader, desc=f"train phase3 epoch {epoch}", leave=False):
        for meta in metadata:
            count_augmentation_trace(tuple(meta["augmentation_trace"]), audit)
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=bool(cfg.amp and device.type == "cuda")):
            loss = parseq_plm_loss(model, images, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(images.shape[0])
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["samples"] += batch_size
    return totals["loss"] / max(totals["samples"], 1), audit


def _set_encoder_trainable(model, trainable: bool) -> None:
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = trainable


def _save_checkpoint(path: Path, model, model_cfg, phase3_cfg, augmentation_cfg, epoch, metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model_cfg),
            "phase3_config": asdict(phase3_cfg),
            "augmentation_config": augmentation_cfg.to_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "architecture": "official_strhub_parseq_phase3_controlled_augmentation",
            "parent_checkpoint": phase3_cfg.checkpoint,
        },
        path,
    )


def _group_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (plate_type, source_name), group in predictions.groupby(["plate_type", "source_name"], dropna=False):
        edits = int(group["edit_distance"].sum())
        chars = int(group["target"].str.len().clip(lower=1).sum())
        rows.append(
            {
                "plate_type": plate_type,
                "source_name": source_name,
                "samples": len(group),
                "exact_acc": float(group["exact"].mean()),
                "char_acc": float(1.0 - edits / max(chars, 1)),
            }
        )
    return pd.DataFrame(rows)


def _validation_error_profile(predictions: pd.DataFrame) -> dict:
    """Summarize only validation errors; test labels never influence the policy."""

    frame = predictions.copy()
    dimensions = []
    for path in frame["image_path"]:
        with Image.open(path) as image:
            dimensions.append((image.width, image.height))
    frame[["width", "height"]] = pd.DataFrame(dimensions, index=frame.index)
    frame["aspect"] = frame["width"] / frame["height"].clip(lower=1)
    frame["resolution_group"] = pd.cut(
        frame["width"] * frame["height"],
        bins=[-1, 2500, 6000, float("inf")],
        labels=["tiny", "small", "regular"],
    ).astype(str)
    frame["layout_group"] = np.where(frame["aspect"] < 1.9, "likely_two_line", "wide")
    wrong = frame[~frame["exact"].astype(bool)].copy()
    wrong["length_error"] = wrong["prediction"].str.len() - wrong["target"].str.len()

    def grouped(column: str) -> list[dict]:
        rows = []
        for key, group in frame.groupby(column, dropna=False):
            rows.append(
                {
                    column: str(key),
                    "samples": int(len(group)),
                    "errors": int((~group["exact"].astype(bool)).sum()),
                    "exact_acc": float(group["exact"].mean()),
                }
            )
        return rows

    substitutions = Counter()
    for target, prediction in zip(wrong["target"], wrong["prediction"]):
        if len(target) == len(prediction):
            substitutions.update(
                f"{left}>{right}" for left, right in zip(target, prediction) if left != right
            )
    return {
        "source": "validation_only",
        "samples": int(len(frame)),
        "errors": int(len(wrong)),
        "resolution_groups": grouped("resolution_group"),
        "layout_groups": grouped("layout_group"),
        "plate_types": grouped("plate_type"),
        "error_length_delta": {str(key): int(value) for key, value in wrong["length_error"].value_counts().sort_index().items()},
        "top_equal_length_substitutions": [
            {"pair": pair, "count": count} for pair, count in substitutions.most_common(15)
        ],
    }


def _run_refinement_sweep(model, loader, device, max_length, split):
    rows = []
    predictions = {}
    for refine_iters in (0, 1, 2, 3):
        set_decode_mode(model, refine_iters=refine_iters, decode_ar=True)
        metrics, frame = evaluate(model, loader, device, f"{split}_refine_{refine_iters}", max_length)
        rows.append(metrics)
        predictions[refine_iters] = frame
    return pd.DataFrame(rows), predictions


def _write_preview(dataset: Phase3PlateDataset, output_dir: Path, count: int, seed: int) -> Counter:
    preview_dir = output_dir / "augmentation_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    audit = Counter()
    indices = [rng.randrange(len(dataset)) for _ in range(min(count, max(len(dataset), 1)))]
    for order, index in enumerate(indices):
        tensor, label, _path, meta = dataset[index]
        count_augmentation_trace(tuple(meta["augmentation_trace"]), audit)
        image = TF.to_pil_image(tensor * 0.5 + 0.5)
        operations = "__".join(item.replace(":", "-") for item in meta["augmentation_trace"])
        image.save(preview_dir / f"{order:03d}_{label}_{operations[:120]}.jpg", quality=92)
    pd.DataFrame(trace_rates(audit)).to_csv(output_dir / "augmentation_preview_stats.csv", index=False)
    return audit


def fit(cfg: Phase3TrainingConfig, dry_run: bool = False, preview_count: int = 32, device_name: str = "") -> dict:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(cfg.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model, model_cfg, parent_checkpoint = load_notebook_checkpoint(
        checkpoint_path, device, refine_iters=cfg.refine_iters
    )
    model_cfg.pretrained = False
    saved_config = parent_checkpoint.get("config", {})
    frame = build_dataset_frame(saved_config, cfg)
    frame, frozen_manifests = freeze_evaluation_splits(
        frame, checkpoint_path, cfg.val_manifest, cfg.test_manifest
    )
    frame.to_csv(output_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    dataset_summary = (
        frame.groupby(["split", "plate_type", "source_name"], dropna=False)
        .size()
        .reset_index(name="samples")
    )
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    augmentation_cfg = Phase3AugmentationConfig(profile=cfg.policy_profile)
    train_loader, val_loader, test_loader, datasets, class_weights = make_loaders(
        frame, model_cfg, cfg, augmentation_cfg
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "training": asdict(cfg),
                "augmentation": augmentation_cfg.to_dict(),
                "class_weights": class_weights,
                "frozen_eval_manifests": frozen_manifests,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    preview_audit = _write_preview(datasets["train"], output_dir, preview_count, cfg.seed)
    if dry_run:
        images, labels, _paths, _metadata = next(iter(train_loader))
        images = images[:2].to(device)
        labels = labels[:2]
        model.train()
        with torch.no_grad():
            smoke_loss = float(parseq_plm_loss(model, images, labels).item())
        summary = {
            "status": "dry_run_complete",
            "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
            "augmentation_preview_samples": int(preview_audit["samples"]),
            "two_sample_plm_loss": smoke_loss,
            "output_dir": str(output_dir),
        }
        (output_dir / "dry_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    baseline_val, baseline_val_predictions = evaluate(
        model, val_loader, device, "val_parent_checkpoint", cfg.max_label_length
    )
    baseline_val_predictions.to_csv(output_dir / "val_parent_checkpoint_predictions.csv", index=False)
    (output_dir / "parent_validation_error_profile.json").write_text(
        json.dumps(_validation_error_profile(baseline_val_predictions), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs - cfg.freeze_encoder_epochs, 1), eta_min=cfg.learning_rate * 0.10
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    history = []
    audit_rows = []
    best_key = (baseline_val["exact_acc"], baseline_val["char_acc"])
    best_epoch = 0
    stale_epochs = 0
    best_path = output_dir / "best_phase3_parseq_anpr.pt"
    # Epoch 0 is the parent model. Phase 3 is never allowed to replace it with
    # a checkpoint that scores worse on validation.
    _save_checkpoint(best_path, model, model_cfg, cfg, augmentation_cfg, 0, baseline_val)

    for epoch in range(1, cfg.epochs + 1):
        encoder_trainable = epoch > cfg.freeze_encoder_epochs
        _set_encoder_trainable(model, encoder_trainable)
        start = time.perf_counter()
        loss, audit = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch)
        set_decode_mode(model, refine_iters=cfg.refine_iters, decode_ar=True)
        val_metrics, _ = evaluate(model, val_loader, device, "val", cfg.max_label_length)
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_trainable": encoder_trainable,
            "seconds": elapsed,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        for operation in trace_rates(audit):
            audit_rows.append({"epoch": epoch, **operation})
        print(json.dumps(row, ensure_ascii=False))

        current_key = (val_metrics["exact_acc"], val_metrics["char_acc"])
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(best_path, model, model_cfg, cfg, augmentation_cfg, epoch, val_metrics)
        else:
            stale_epochs += 1
        if encoder_trainable:
            scheduler.step()
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(output_dir / "augmentation_stats_by_epoch.csv", index=False)
        if stale_epochs >= cfg.early_stopping_patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    model, model_cfg, best_checkpoint = load_notebook_checkpoint(best_path, device, refine_iters=cfg.refine_iters)
    val_sweep, val_predictions = _run_refinement_sweep(
        model, val_loader, device, cfg.max_label_length, "val"
    )
    val_sweep.to_csv(output_dir / "refinement_sweep_val.csv", index=False)
    best_refine = int(
        val_sweep.sort_values(["exact_acc", "char_acc"], ascending=False).iloc[0]["refine_iters"]
    )
    val_predictions[best_refine].to_csv(output_dir / "eval_val_predictions_best_refine.csv", index=False)

    set_decode_mode(model, refine_iters=best_refine, decode_ar=True)
    test_metrics, test_predictions = evaluate(
        model, test_loader, device, "test_locked", cfg.max_label_length
    )
    test_predictions.to_csv(output_dir / "eval_test_predictions_locked.csv", index=False)
    _group_metrics(val_predictions[best_refine]).to_csv(output_dir / "eval_val_by_plate_type.csv", index=False)
    _group_metrics(test_predictions).to_csv(output_dir / "eval_test_by_plate_type.csv", index=False)

    summary = {
        "status": "complete",
        "parent_checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_refine_iters_selected_on_validation": best_refine,
        "parent_validation_metrics": baseline_val,
        "phase3_validation_metrics": val_sweep[val_sweep["refine_iters"] == best_refine].iloc[0].to_dict(),
        "phase3_test_metrics_locked": test_metrics,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "policy_profile": cfg.policy_profile,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Phase 3 controlled augmentation fine-tuning for official PARSeq ANPR")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-root", default=str(ROOT / "dataset"))
    parser.add_argument("--val-manifest", default="", help="Locked validation manifest; defaults to the checkpoint directory")
    parser.add_argument("--test-manifest", default="", help="Locked test manifest; defaults to the checkpoint directory")
    parser.add_argument("--output-dir", default=str(ROOT / f"outputs/phase3_controlled_aug_{timestamp}"))
    parser.add_argument("--policy-profile", choices=["full", "resolution_only", "restoration_only", "light"], default="full")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--balance-alpha", type=float, default=0.5)
    parser.add_argument("--max-sample-weight", type=float, default=8.0)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-count", type=int, default=32)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    cfg = Phase3TrainingConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        policy_profile=args.policy_profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        refine_iters=args.refine_iters,
        seed=args.seed,
        amp=not args.no_amp,
        balance_alpha=args.balance_alpha,
        max_sample_weight=args.max_sample_weight,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        early_stopping_patience=args.early_stopping_patience,
        max_train_samples=args.max_train_samples,
    )
    summary = fit(cfg, dry_run=args.dry_run, preview_count=args.preview_count, device_name=args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()

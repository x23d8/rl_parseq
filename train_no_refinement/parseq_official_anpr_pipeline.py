"""Official PARSeq ANPR fine-tuning pipeline without iterative refinement.

This module intentionally wraps the vendored official PARSeq implementation under
``parseq/strhub`` instead of using the compact custom PARSeqOCR model from
``parseq_anpr_pipeline.py``. It keeps the original PARSeq architecture, tokenizer,
and permuted-language-modeling loss, with ``refine_iters`` defaulting to 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_DIR.parent
PARSEQ_DIR = PIPELINE_DIR / "parseq"
PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing_best_config"
if str(PARSEQ_DIR) not in sys.path:
    sys.path.insert(0, str(PARSEQ_DIR))
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from preprocessing import DEFAULT_CONFIG, get_preprocessing_config, preprocess_plate_image  # noqa: E402
from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.utils import create_model  # noqa: E402

ANPR_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "ocr_dataset_rescued_bbox_new"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "outputs" / "train_no_refinement"


@dataclass
class OfficialPARSeqANPRConfig:
    data_root: str = str(DEFAULT_DATA_ROOT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    experiment: str = "parseq"
    pretrained: bool = True
    decode_ar: bool = True
    refine_iters: int = 0
    charset_test: str = ANPR_CHARSET
    img_size: tuple[int, int] = (32, 128)
    max_label_length: int = 12
    preprocess: bool = False
    preprocess_config: str = DEFAULT_CONFIG.name
    augment: bool = True
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 20.0
    seed: int = 42
    amp: bool = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_plate_text(text: object) -> str:
    return "".join(ch for ch in str(text).upper() if ch in ANPR_CHARSET)


class PlateCropDataset(Dataset):
    def __init__(self, root: str | Path, split: str, transform=None, max_label_length: int = 12, limit: int | None = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        frame = pd.read_csv(self.root / f"{split}.csv")
        frame["label"] = frame["label"].map(normalize_plate_text)
        frame = frame[frame["label"].str.len().between(1, int(max_label_length))].copy()
        if limit is not None:
            frame = frame.head(int(limit)).copy()
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["label"], str(image_path)


def collate_batch(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(list(images), dim=0), list(labels), list(paths)


def build_official_transform(
    img_size: Sequence[int] = (32, 128),
    augment: bool = False,
    preprocess: bool = False,
    preprocess_config: str = DEFAULT_CONFIG.name,
):
    base_transform = SceneTextDataModule.get_transform(tuple(img_size), augment=augment)
    if not preprocess:
        return base_transform

    preprocess_cfg = get_preprocessing_config(preprocess_config)

    return T.Compose([T.Lambda(lambda image: preprocess_plate_image(image, preprocess_cfg)), base_transform])


def make_loaders(cfg: OfficialPARSeqANPRConfig):
    train_transform = build_official_transform(
        cfg.img_size,
        augment=cfg.augment,
        preprocess=cfg.preprocess,
        preprocess_config=cfg.preprocess_config,
    )
    eval_transform = build_official_transform(
        cfg.img_size,
        augment=False,
        preprocess=cfg.preprocess,
        preprocess_config=cfg.preprocess_config,
    )
    train_ds = PlateCropDataset(cfg.data_root, "train", train_transform, cfg.max_label_length)
    val_ds = PlateCropDataset(cfg.data_root, "val", eval_transform, cfg.max_label_length)
    test_ds = PlateCropDataset(cfg.data_root, "test", eval_transform, cfg.max_label_length)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
        {"train": train_ds, "val": val_ds, "test": test_ds},
    )


def create_official_parseq_model(cfg: OfficialPARSeqANPRConfig, device: str | torch.device = "cuda"):
    model = create_model(
        cfg.experiment,
        pretrained=cfg.pretrained,
        decode_ar=cfg.decode_ar,
        refine_iters=cfg.refine_iters,
        charset_test=cfg.charset_test,
    )
    device = torch.device(device)
    model.to(device)
    # The original Lightning Trainer sets this field internally. Manual loops need it for gen_tgt_perms().
    model._device = device
    return model


def set_decode_mode(model, refine_iters: int, decode_ar: Optional[bool] = None) -> None:
    model.model.refine_iters = int(refine_iters)
    if decode_ar is not None:
        model.model.decode_ar = bool(decode_ar)


def edit_distance(left: str, right: str) -> int:
    left = normalize_plate_text(left)
    right = normalize_plate_text(right)
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i]
        for j, rc in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (lc != rc)))
        previous = current
    return previous[-1]


@torch.no_grad()
def greedy_decode(model, images: torch.Tensor, max_length: Optional[int] = None):
    logits = model(images, max_length=max_length)
    probs = logits.softmax(-1)
    preds, token_probs = model.tokenizer.decode(probs)
    preds = [normalize_plate_text(pred) for pred in preds]
    confidences = torch.tensor([float(prob.prod().item()) for prob in token_probs], device=images.device)
    return preds, confidences


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str | torch.device = "cuda", split_name: str = "val", max_length: int | None = None):
    model.eval()
    device = torch.device(device)
    rows = []
    exact = 0
    edits = 0
    chars = 0
    total = 0
    for images, labels, paths in tqdm(loader, desc=f"eval {split_name}", leave=False):
        images = images.to(device, non_blocking=True)
        preds, confs = greedy_decode(model, images, max_length=max_length)
        for path, pred, target, conf in zip(paths, preds, labels, confs.detach().cpu().tolist()):
            target = normalize_plate_text(target)
            dist = edit_distance(pred, target)
            ok = pred == target
            exact += int(ok)
            edits += dist
            chars += max(len(target), 1)
            total += 1
            rows.append(
                {
                    "image_path": path,
                    "target": target,
                    "prediction": pred,
                    "exact": ok,
                    "edit_distance": dist,
                    "confidence": conf,
                }
            )
    metrics = {
        "split": split_name,
        "samples": total,
        "exact_acc": exact / max(total, 1),
        "cer": edits / max(chars, 1),
        "char_acc": 1.0 - edits / max(chars, 1),
        "refine_iters": int(model.model.refine_iters),
        "decode_ar": bool(model.model.decode_ar),
    }
    return metrics, pd.DataFrame(rows)


def parseq_plm_loss(system, images: torch.Tensor, labels: list[str]) -> torch.Tensor:
    system._device = images.device
    tgt = system.tokenizer.encode(labels, images.device)
    memory = system.model.encode(images)
    tgt_perms = system.gen_tgt_perms(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_padding_mask = (tgt_in == system.pad_id) | (tgt_in == system.eos_id)

    loss = torch.zeros((), dtype=memory.dtype, device=images.device)
    loss_numel = 0
    n = int((tgt_out != system.pad_id).sum().item())
    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = system.generate_attn_masks(perm)
        out = system.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = system.model.head(out).flatten(end_dim=1)
        loss = loss + n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=system.pad_id)
        loss_numel += n
        if i == 1:
            tgt_out = torch.where(tgt_out == system.eos_id, system.pad_id, tgt_out)
            n = int((tgt_out != system.pad_id).sum().item())
    return loss / max(loss_numel, 1)


def train_one_epoch(model, loader: DataLoader, optimizer, scaler, cfg: OfficialPARSeqANPRConfig, device: str | torch.device, epoch: int):
    model.train()
    device = torch.device(device)
    totals = {"loss": 0.0, "samples": 0}
    for images, labels, _paths in tqdm(loader, desc=f"train epoch {epoch}", leave=False):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
            loss = parseq_plm_loss(model, images, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["samples"] += batch_size
    return {"train_loss": totals["loss"] / max(totals["samples"], 1), "train_samples": totals["samples"]}


def save_checkpoint(path: str | Path, model, cfg: OfficialPARSeqANPRConfig, epoch: int, metrics: dict) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "epoch": int(epoch),
        "metrics": metrics,
        "architecture": "official_strhub_parseq",
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cuda"):
    checkpoint = torch.load(path, map_location=device)
    cfg = OfficialPARSeqANPRConfig(**checkpoint["config"])
    cfg.pretrained = False
    model = create_official_parseq_model(cfg, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model._device = torch.device(device)
    return model, cfg, checkpoint


def fit(cfg: OfficialPARSeqANPRConfig, device: str | torch.device | None = None):
    set_seed(cfg.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, datasets = make_loaders(cfg)
    model = create_official_parseq_model(cfg, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    base_metrics, _ = evaluate(model, val_loader, device=device, split_name="val_before_finetune", max_length=cfg.max_label_length)
    history = []
    best_val_exact = -1.0
    best_path = output_dir / "best_official_parseq_anpr.pt"
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch)
        val_metrics, _ = evaluate(model, val_loader, device=device, split_name="val", max_length=cfg.max_label_length)
        row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "seconds": time.time() - start}
        history.append(row)
        if val_metrics["exact_acc"] > best_val_exact:
            best_val_exact = val_metrics["exact_acc"]
            save_checkpoint(best_path, model, cfg, epoch, val_metrics)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    best_model, _best_cfg, best_ckpt = load_checkpoint(best_path, device=device)
    test_metrics, test_rows = evaluate(best_model, test_loader, device=device, split_name="test", max_length=cfg.max_label_length)
    test_rows.to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "config": asdict(cfg),
        "base_val_metrics": base_metrics,
        "best_val_exact": best_val_exact,
        "best_checkpoint": str(best_path),
        "best_epoch": best_ckpt.get("epoch"),
        "test_metrics": test_metrics,
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, history_df, summary


def evaluate_refinement_sweep(model, loader: DataLoader, refine_iters_list: Iterable[int] = (0, 1, 2, 3), device: str | torch.device = "cuda", split_name: str = "val", max_length: int | None = None):
    original_refine_iters = int(model.model.refine_iters)
    original_decode_ar = bool(model.model.decode_ar)
    rows = []
    predictions = {}
    try:
        for refine_iters in refine_iters_list:
            set_decode_mode(model, refine_iters=int(refine_iters), decode_ar=original_decode_ar)
            metrics, preds = evaluate(model, loader, device=device, split_name=f"{split_name}_refine_{refine_iters}", max_length=max_length)
            rows.append(metrics)
            predictions[int(refine_iters)] = preds
    finally:
        set_decode_mode(model, refine_iters=original_refine_iters, decode_ar=original_decode_ar)
    return pd.DataFrame(rows), predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune official PARSeq for ANPR with refinement disabled by default.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refine-iters", type=int, default=0, help="Keep this at 0 for the no-refinement training run.")
    parser.add_argument("--decode-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--preprocess-config", default=DEFAULT_CONFIG.name)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    cfg = OfficialPARSeqANPRConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        decode_ar=args.decode_ar,
        refine_iters=args.refine_iters,
        preprocess=args.preprocess,
        preprocess_config=args.preprocess_config,
        augment=not args.no_augment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    _model, _history, summary = fit(cfg, device=args.device or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()

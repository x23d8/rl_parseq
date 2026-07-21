"""Benchmark course-inspired image preprocessing before a fine-tuned PARSeq.

Selection is performed on validation data only.  The untouched test manifest is
then evaluated with the validation-selected top configurations, preventing test
set leakage while still producing a final generalization check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

PIPELINE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = PIPELINE_DIR / "train_no_refinement"
PARSEQ_DIR = PIPELINE_DIR / "parseq"
for path in (TRAIN_DIR, PARSEQ_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:  # Package import for tests/notebooks; direct import for CLI execution.
    from .preprocessing import DEFAULT_CONFIG, PreprocessingConfig, iter_named_configs, preprocess_plate_image
except ImportError:
    from preprocessing import DEFAULT_CONFIG, PreprocessingConfig, iter_named_configs, preprocess_plate_image  # type: ignore
from parseq_official_anpr_pipeline import (  # noqa: E402
    OfficialPARSeqANPRConfig,
    create_official_parseq_model,
    edit_distance,
    greedy_decode,
    normalize_plate_text,
    set_decode_mode,
)


class ManifestPlateDataset(Dataset):
    """Read the exact paths/labels from a prior evaluation manifest."""

    def __init__(self, manifest: str | Path, transform):
        self.manifest = Path(manifest)
        frame = pd.read_csv(self.manifest)
        required = {"image_path", "target"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{self.manifest} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["target"] = frame["target"].map(normalize_plate_text)
        exists = frame["image_path"].map(lambda value: Path(str(value)).exists())
        if not bool(exists.all()):
            missing_paths = frame.loc[~exists, "image_path"].head(5).tolist()
            raise FileNotFoundError(f"Missing {int((~exists).sum())} images, examples: {missing_paths}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        metadata = {
            key: row.get(key, "")
            for key in ("split", "source_name", "plate_type", "label_status", "review_status")
        }
        return image, row["target"], str(row["image_path"]), metadata


def collate_batch(batch):
    images, labels, paths, metadata = zip(*batch)
    return torch.stack(list(images)), list(labels), list(paths), list(metadata)


def _interpolation_mode(name: str):
    modes = {
        "bilinear": T.InterpolationMode.BILINEAR,
        "bicubic": T.InterpolationMode.BICUBIC,
        "lanczos": T.InterpolationMode.LANCZOS,
    }
    try:
        return modes[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported resize interpolation: {name}") from exc


class LetterboxResize:
    def __init__(self, size: Sequence[int], interpolation):
        self.height, self.width = (int(size[0]), int(size[1]))
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = min(self.width / max(image.width, 1), self.height / max(image.height, 1))
        resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = T.functional.resize(
            image,
            [resized_size[1], resized_size[0]],
            interpolation=self.interpolation,
        )
        canvas = Image.new("RGB", (self.width, self.height), (127, 127, 127))
        canvas.paste(resized, ((self.width - resized.width) // 2, (self.height - resized.height) // 2))
        return canvas


def build_transform(img_size: Sequence[int], cfg: PreprocessingConfig):
    # This is identical to the notebook evaluation transform except that the
    # resize kernel is an explicit experimental variable.
    interpolation = _interpolation_mode(cfg.resize_interpolation)
    if cfg.resize_mode == "stretch":
        resize = T.Resize(tuple(img_size), interpolation=interpolation)
    elif cfg.resize_mode == "letterbox":
        resize = LetterboxResize(img_size, interpolation)
    else:
        raise ValueError(f"Unsupported resize mode: {cfg.resize_mode}")
    return T.Compose(
        [
            T.Lambda(lambda image: preprocess_plate_image(image, cfg)),
            resize,
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_loader(manifest: Path, model_cfg: OfficialPARSeqANPRConfig, preprocess_cfg, args):
    dataset = ManifestPlateDataset(manifest, build_transform(model_cfg.img_size, preprocess_cfg))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def load_notebook_checkpoint(path: Path, device: torch.device, refine_iters: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    allowed = {item.name for item in fields(OfficialPARSeqANPRConfig)}
    cfg_values = {key: value for key, value in saved.items() if key in allowed}
    if "img_size" in cfg_values:
        cfg_values["img_size"] = tuple(cfg_values["img_size"])
    cfg_values.update(pretrained=False, refine_iters=int(refine_iters), augment=False)
    model_cfg = OfficialPARSeqANPRConfig(**cfg_values)
    model = create_official_parseq_model(model_cfg, device=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    set_decode_mode(model, refine_iters=refine_iters, decode_ar=True)
    model.eval()
    return model, model_cfg, checkpoint


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, config_name: str, split: str, max_length: int):
    start = time.perf_counter()
    rows = []
    for images, labels, paths, metadata in tqdm(loader, desc=f"{split}: {config_name}", leave=False):
        images = images.to(device, non_blocking=True)
        preds, confidences = greedy_decode(model, images, max_length=max_length)
        for path, target, pred, confidence, meta in zip(
            paths, labels, preds, confidences.cpu().tolist(), metadata
        ):
            distance = edit_distance(pred, target)
            rows.append(
                {
                    "config": config_name,
                    "image_path": path,
                    "target": target,
                    "prediction": pred,
                    "exact": pred == target,
                    "edit_distance": distance,
                    "target_length": max(len(target), 1),
                    "confidence": confidence,
                    **meta,
                }
            )
    elapsed = time.perf_counter() - start
    predictions = pd.DataFrame(rows)
    metrics = metrics_from_predictions(predictions)
    metrics.update(config=config_name, split=split, seconds=elapsed, images_per_second=len(predictions) / max(elapsed, 1e-9))
    return metrics, predictions


def metrics_from_predictions(predictions: pd.DataFrame) -> dict:
    samples = len(predictions)
    edits = int(predictions["edit_distance"].sum())
    chars = int(predictions["target_length"].sum())
    return {
        "samples": samples,
        "exact_acc": float(predictions["exact"].mean()) if samples else 0.0,
        "char_acc": 1.0 - edits / max(chars, 1),
        "cer": edits / max(chars, 1),
        "errors": int((~predictions["exact"]).sum()),
        "edit_errors": edits,
    }


def json_safe(value):
    """Convert pandas/numpy scalars and NaN values to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def paired_deltas(candidate: pd.DataFrame, baseline: pd.DataFrame, bootstrap_samples: int, seed: int) -> dict:
    cols = ["image_path", "exact", "edit_distance", "target_length"]
    pair = baseline[cols].merge(candidate[cols], on="image_path", suffixes=("_base", "_candidate"), validate="one_to_one")
    base_exact = pair["exact_base"].to_numpy(dtype=float)
    cand_exact = pair["exact_candidate"].to_numpy(dtype=float)
    base_edits = pair["edit_distance_base"].to_numpy(dtype=float)
    cand_edits = pair["edit_distance_candidate"].to_numpy(dtype=float)
    lengths = pair["target_length_base"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    exact_boot = np.empty(bootstrap_samples, dtype=float)
    char_boot = np.empty(bootstrap_samples, dtype=float)
    for idx in range(bootstrap_samples):
        sample = rng.integers(0, len(pair), len(pair))
        exact_boot[idx] = np.mean(cand_exact[sample] - base_exact[sample])
        char_boot[idx] = (base_edits[sample].sum() - cand_edits[sample].sum()) / lengths[sample].sum()
    return {
        "delta_exact": float(np.mean(cand_exact - base_exact)),
        "delta_char_acc": float((base_edits.sum() - cand_edits.sum()) / lengths.sum()),
        "exact_ci95_low": float(np.quantile(exact_boot, 0.025)),
        "exact_ci95_high": float(np.quantile(exact_boot, 0.975)),
        "char_ci95_low": float(np.quantile(char_boot, 0.025)),
        "char_ci95_high": float(np.quantile(char_boot, 0.975)),
        "fixed_images": int((~pair["exact_base"] & pair["exact_candidate"]).sum()),
        "broken_images": int((pair["exact_base"] & ~pair["exact_candidate"]).sum()),
    }


def _discover_run() -> Path:
    candidates = []
    for path in (PIPELINE_DIR / "outputs").glob("refinement_finetune*"):
        if (path / "best_official_parseq_anpr.pt").exists() and (path / "eval_val_predictions_best_refine.csv").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No refinement run with checkpoint and validation manifest was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _write_report(output_dir: Path, val_results: pd.DataFrame, test_results: pd.DataFrame, args, paths):
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(column) for column in frame.columns]
        body = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value).replace("|", "\\|"))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join(body)

    best = val_results.iloc[0]
    baseline = val_results[val_results["config"] == DEFAULT_CONFIG.name].iloc[0]
    lines = [
        "# PARSeq preprocessing benchmark",
        "",
        "Selection was performed on validation only; test was used only for the validation-selected finalists.",
        "",
        f"- Checkpoint: `{paths['checkpoint']}`",
        f"- Validation manifest: `{paths['val_manifest']}`",
        f"- Test manifest: `{paths['test_manifest']}`",
        f"- Refinement iterations: {args.refine_iters}",
        f"- Training-time preprocessing baseline: `{DEFAULT_CONFIG.name}`",
        "",
        "## Validation ranking",
        "",
        markdown_table(val_results[["config", "exact_acc", "char_acc", "cer", "delta_exact", "delta_char_acc", "fixed_images", "broken_images"]]),
        "",
        "## Locked test confirmation",
        "",
        markdown_table(test_results[["config", "exact_acc", "char_acc", "cer", "delta_exact", "delta_char_acc", "fixed_images", "broken_images"]]),
        "",
        "## Decision",
        "",
        f"Validation winner: `{best['config']}` ({best['exact_acc']:.4%} exact, {best['char_acc']:.4%} character accuracy).",
        f"Training baseline: `{DEFAULT_CONFIG.name}` ({baseline['exact_acc']:.4%} exact, {baseline['char_acc']:.4%} character accuracy).",
        "A preprocessing method should replace the training baseline only when its validation gain also holds on the locked test set; bootstrap intervals crossing zero indicate uncertainty at this dataset size.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(args: argparse.Namespace):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = _discover_run() if not args.run_dir else Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "best_official_parseq_anpr.pt"
    val_manifest = Path(args.val_manifest).resolve() if args.val_manifest else run_dir / "eval_val_predictions_best_refine.csv"
    test_manifest = Path(args.test_manifest).resolve() if args.test_manifest else run_dir / "eval_test_predictions_best_refine.csv"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_cfg, checkpoint_payload = load_notebook_checkpoint(checkpoint, device, args.refine_iters)
    configs = iter_named_configs(args.configs)
    if DEFAULT_CONFIG.name not in {cfg.name for cfg in configs}:
        configs.insert(0, DEFAULT_CONFIG)

    val_rows, val_predictions = [], {}
    for cfg in configs:
        loader = build_loader(val_manifest, model_cfg, cfg, args)
        metrics, predictions = evaluate(model, loader, device, cfg.name, "val", model_cfg.max_label_length)
        val_rows.append({**cfg.to_dict(), **metrics})
        val_predictions[cfg.name] = predictions
        predictions.to_csv(output_dir / f"predictions_val_{cfg.name}.csv", index=False)

    baseline_predictions = val_predictions[DEFAULT_CONFIG.name]
    enriched = []
    for idx, row in enumerate(val_rows):
        delta = paired_deltas(
            val_predictions[row["config"]], baseline_predictions, args.bootstrap_samples, args.seed + idx
        )
        enriched.append({**row, **delta})
    val_results = pd.DataFrame(enriched).sort_values(
        ["exact_acc", "char_acc", "images_per_second"], ascending=[False, False, False]
    ).reset_index(drop=True)
    val_results.to_csv(output_dir / "validation_results.csv", index=False)
    (output_dir / "best_preprocessing_config.json").write_text(
        json.dumps(json_safe(val_results.iloc[0].to_dict()), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    finalists = val_results.head(args.top_k)["config"].tolist()
    if DEFAULT_CONFIG.name not in finalists:
        finalists.append(DEFAULT_CONFIG.name)
    config_by_name = {cfg.name: cfg for cfg in configs}
    test_rows, test_predictions = [], {}
    for name in finalists:
        cfg = config_by_name[name]
        loader = build_loader(test_manifest, model_cfg, cfg, args)
        metrics, predictions = evaluate(model, loader, device, cfg.name, "test", model_cfg.max_label_length)
        test_rows.append({**cfg.to_dict(), **metrics})
        test_predictions[name] = predictions
        predictions.to_csv(output_dir / f"predictions_test_{name}.csv", index=False)
    test_baseline = test_predictions[DEFAULT_CONFIG.name]
    test_enriched = []
    for idx, row in enumerate(test_rows):
        delta = paired_deltas(
            test_predictions[row["config"]], test_baseline, args.bootstrap_samples, args.seed + 1000 + idx
        )
        test_enriched.append({**row, **delta})
    test_results = pd.DataFrame(test_enriched).sort_values(
        ["exact_acc", "char_acc", "images_per_second"], ascending=[False, False, False]
    ).reset_index(drop=True)
    test_results.to_csv(output_dir / "test_finalists_results.csv", index=False)

    paths = {"checkpoint": str(checkpoint), "val_manifest": str(val_manifest), "test_manifest": str(test_manifest)}
    summary = {
        "paths": paths,
        "device": str(device),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "refine_iters": args.refine_iters,
        "selection_rule": "validation exact_acc, then validation char_acc",
        "validation_winner": val_results.iloc[0].to_dict(),
        "test_finalists": test_results.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_report(output_dir, val_results, test_results, args, paths)
    return val_results, test_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="", help="Refinement output directory; auto-discovers latest by default.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--val-manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument("--output-dir", default=str(PIPELINE_DIR / "outputs" / "preprocessing_course_benchmark"))
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    validation, test = run_benchmark(parse_args())
    print("\nValidation ranking")
    print(validation[["config", "exact_acc", "char_acc", "delta_exact", "delta_char_acc"]].to_string(index=False))
    print("\nLocked test finalists")
    print(test[["config", "exact_acc", "char_acc", "delta_exact", "delta_char_acc"]].to_string(index=False))

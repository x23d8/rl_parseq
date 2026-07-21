"""Benchmark resolution-aware multi-scale TTA for the fine-tuned PARSeq model.

The selector is fitted on validation predictions only. The locked selector is
then applied to test predictions, including the previously irrecoverable test
subset. No model weights are updated by this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train_no_refinement"
PARSEQ_DIR = ROOT / "parseq"
for import_path in (TRAIN_DIR, PARSEQ_DIR, Path(__file__).resolve().parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from preprocessing import get_preprocessing_config, preprocess_plate_image  # noqa: E402
from parseq_official_anpr_pipeline import edit_distance, greedy_decode, normalize_plate_text  # noqa: E402


@dataclass(frozen=True)
class ViewSpec:
    name: str
    zoom: float
    upscale: float
    preprocessing: str
    unwrap_two_line: bool = False


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    if arr.size == 0:
        return (127, 127, 127)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]), axis=0)
    return tuple(int(value) for value in np.median(border, axis=0))


def apply_center_zoom(image: Image.Image, factor: float) -> Image.Image:
    """Scale content around the centre while preserving the original canvas."""

    image = image.convert("RGB")
    if abs(float(factor) - 1.0) < 1e-6:
        return image
    width, height = image.size
    scaled_width = max(1, round(width * float(factor)))
    scaled_height = max(1, round(height * float(factor)))
    scaled = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    if factor < 1.0:
        canvas = Image.new("RGB", (width, height), _background_color(image))
        canvas.paste(scaled, ((width - scaled_width) // 2, (height - scaled_height) // 2))
        return canvas
    left = max(0, (scaled_width - width) // 2)
    top = max(0, (scaled_height - height) // 2)
    return scaled.crop((left, top, left + width, top + height))


def upscale_small_image(image: Image.Image, factor: float) -> Image.Image:
    """Upscale low-resolution crops before nonlinear image enhancement."""

    if factor <= 1.0 or (image.width >= 128 and image.height >= 64):
        return image
    return image.resize(
        (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
        Image.Resampling.LANCZOS,
    )


def unwrap_plate_lines(image: Image.Image, aspect_threshold: float = 1.9) -> Image.Image:
    """Convert a likely two-line plate into top-line then bottom-line layout."""

    image = image.convert("RGB")
    width, height = image.size
    if height < 16 or width / max(height, 1) >= aspect_threshold:
        return image

    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    vertical_strokes = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=1)
    smooth = cv2.GaussianBlur(vertical_strokes.reshape(-1, 1), (1, 5), 0).ravel()
    start = max(4, round(height * 0.35))
    stop = min(height - 4, round(height * 0.65))
    if stop <= start:
        return image
    split = int(start + np.argmin(smooth[start:stop]))
    top = image.crop((0, 0, width, split))
    bottom = image.crop((0, split, width, height))
    line_height = max(top.height, bottom.height)

    def resize_line(line: Image.Image) -> Image.Image:
        line_width = max(1, round(line.width * line_height / max(line.height, 1)))
        return line.resize((line_width, line_height), Image.Resampling.LANCZOS)

    top = resize_line(top)
    bottom = resize_line(bottom)
    gap = max(2, round(line_height * 0.08))
    canvas = Image.new(
        "RGB",
        (top.width + gap + bottom.width, line_height),
        _background_color(image),
    )
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (top.width + gap, 0))
    return canvas


def transform_view(image: Image.Image, spec: ViewSpec, img_size: tuple[int, int]) -> torch.Tensor:
    image = apply_center_zoom(image, spec.zoom)
    image = upscale_small_image(image, spec.upscale)
    if spec.unwrap_two_line:
        image = unwrap_plate_lines(image)
    image = preprocess_plate_image(image, get_preprocessing_config(spec.preprocessing))
    image = T.functional.resize(image, list(img_size), interpolation=T.InterpolationMode.BICUBIC)
    tensor = T.functional.to_tensor(image)
    return T.functional.normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


class ViewDataset(Dataset):
    def __init__(self, manifest: Path, spec: ViewSpec, img_size: tuple[int, int]):
        frame = pd.read_csv(manifest).copy()
        required = {"image_path", "target"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")
        frame["target"] = frame["target"].map(normalize_plate_text)
        self.frame = frame.reset_index(drop=True)
        self.spec = spec
        self.img_size = tuple(img_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        tensor = transform_view(image, self.spec, self.img_size)
        return tensor, str(row["target"]), path


def collate_batch(batch):
    images, targets, paths = zip(*batch)
    return torch.stack(images), list(targets), list(paths)


def build_specs() -> list[ViewSpec]:
    specs = [ViewSpec("baseline", 1.0, 1.0, "train_baseline", False)]
    preprocessors = (
        "train_baseline",
        "clahe_clip1_tile4",
        "clahe_rl_deblur_bilateral",
        "adaptive_noise_3way",
    )
    for upscale in (2.0, 3.0):
        for zoom in (0.85, 0.93, 1.0, 1.07, 1.15):
            for preprocessing in preprocessors:
                name = f"full_z{zoom:.2f}_up{upscale:.0f}_{preprocessing}"
                specs.append(ViewSpec(name, zoom, upscale, preprocessing, False))
        for zoom in (0.93, 1.0, 1.07):
            for preprocessing in preprocessors:
                name = f"unwrap_z{zoom:.2f}_up{upscale:.0f}_{preprocessing}"
                specs.append(ViewSpec(name, zoom, upscale, preprocessing, True))
    return specs


@torch.inference_mode()
def predict_view(model, model_cfg, manifest: Path, spec: ViewSpec, args, split: str) -> pd.DataFrame:
    loader = DataLoader(
        ViewDataset(manifest, spec, tuple(model_cfg.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    rows: list[dict] = []
    for images, targets, paths in tqdm(loader, desc=f"{split}: {spec.name}", leave=False):
        images = images.to(args.device_obj, non_blocking=True)
        predictions, confidences = greedy_decode(model, images, max_length=model_cfg.max_label_length)
        for path, target, prediction, confidence in zip(
            paths, targets, predictions, confidences.detach().cpu().tolist()
        ):
            distance = edit_distance(prediction, target)
            normalized_confidence = math.exp(
                math.log(max(float(confidence), 1e-12)) / max(len(prediction) + 1, 1)
            )
            rows.append(
                {
                    "split": split,
                    "view": spec.name,
                    "image_path": path,
                    "target": target,
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "normalized_confidence": normalized_confidence,
                    "exact": prediction == target,
                    "edit_distance": distance,
                    "target_length": max(len(target), 1),
                    "zoom": spec.zoom,
                    "upscale": spec.upscale,
                    "preprocessing": spec.preprocessing,
                    "unwrap_two_line": spec.unwrap_two_line,
                }
            )
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame) -> dict:
    edits = int(frame["edit_distance"].sum())
    chars = int(frame["target_length"].sum())
    return {
        "samples": len(frame),
        "exact_acc": float(frame["exact"].mean()),
        "char_acc": 1.0 - edits / max(chars, 1),
        "cer": edits / max(chars, 1),
        "errors": int((~frame["exact"]).sum()),
        "edit_errors": edits,
    }


def view_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view, frame in predictions.groupby("view", sort=False):
        row = frame.iloc[0]
        rows.append(
            {
                "view": view,
                "zoom": row["zoom"],
                "upscale": row["upscale"],
                "preprocessing": row["preprocessing"],
                "unwrap_two_line": row["unwrap_two_line"],
                **metrics(frame),
            }
        )
    return pd.DataFrame(rows).sort_values(["exact_acc", "char_acc"], ascending=False).reset_index(drop=True)


def consensus_predictions(
    predictions: pd.DataFrame,
    reliability: dict[str, float],
    eligible_views: set[str],
    confidence_weight: float,
) -> pd.DataFrame:
    subset = predictions[predictions["view"].isin(eligible_views)]
    output = []
    for image_path, frame in subset.groupby("image_path", sort=False):
        candidates = []
        for prediction, group in frame.groupby("prediction", sort=False):
            support = sum(max(reliability[view] - 0.5, 0.01) for view in group["view"])
            confidence = float(group["normalized_confidence"].max())
            candidates.append(
                (
                    support + confidence_weight * confidence,
                    len(group),
                    confidence,
                    prediction,
                    group,
                )
            )
        _score, votes, confidence, prediction, supporting = max(
            candidates, key=lambda item: (item[0], item[1], item[2], -len(item[3]))
        )
        first = frame.iloc[0]
        target = str(first["target"])
        output.append(
            {
                "image_path": image_path,
                "target": target,
                "prediction": prediction,
                "exact": prediction == target,
                "edit_distance": edit_distance(prediction, target),
                "target_length": max(len(target), 1),
                "normalized_confidence": confidence,
                "votes": votes,
                "supporting_views": ";".join(supporting["view"].tolist()),
            }
        )
    return pd.DataFrame(output)


def fit_selector(val_predictions: pd.DataFrame, val_views: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    baseline_acc = float(val_views.loc[val_views["view"] == "baseline", "exact_acc"].iloc[0])
    reliability = dict(zip(val_views["view"], val_views["exact_acc"]))
    trials = []
    best_key = None
    best_payload = None
    # The final value deliberately admits every view. A weak individual view
    # (notably two-line unwrap) can still add useful consensus on hard crops.
    for tolerance in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.5):
        eligible = set(
            val_views.loc[val_views["exact_acc"] >= baseline_acc - tolerance, "view"].tolist()
        )
        eligible.add("baseline")
        for confidence_weight in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75):
            selected = consensus_predictions(
                val_predictions, reliability, eligible, confidence_weight
            )
            result = metrics(selected)
            trial = {
                "tolerance": tolerance,
                "confidence_weight": confidence_weight,
                "eligible_views": len(eligible),
                **result,
            }
            trials.append(trial)
            key = (result["exact_acc"], result["char_acc"], -len(eligible), -confidence_weight)
            if best_key is None or key > best_key:
                best_key = key
                best_payload = {
                    "tolerance": tolerance,
                    "confidence_weight": confidence_weight,
                    "eligible_views": sorted(eligible),
                    "baseline_validation_accuracy": baseline_acc,
                    "validation_metrics": result,
                }
    assert best_payload is not None
    return best_payload, pd.DataFrame(trials).sort_values(
        ["exact_acc", "char_acc", "eligible_views"], ascending=[False, False, True]
    )


def oracle_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for image_path, frame in predictions.groupby("image_path", sort=False):
        best = frame.sort_values(
            ["edit_distance", "normalized_confidence"], ascending=[True, False]
        ).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def paired_summary(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    pair = baseline[["image_path", "exact"]].merge(
        candidate[["image_path", "exact"]], on="image_path", suffixes=("_baseline", "_candidate")
    )
    return {
        "fixed_images": int((~pair["exact_baseline"] & pair["exact_candidate"]).sum()),
        "broken_images": int((pair["exact_baseline"] & ~pair["exact_candidate"]).sum()),
    }


def paired_changes(candidate: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = baseline[["image_path", "target", "prediction", "exact"]].rename(
        columns={"prediction": "baseline_prediction", "exact": "baseline_exact"}
    )
    selected = candidate[["image_path", "prediction", "exact", "supporting_views"]].rename(
        columns={"prediction": "consensus_prediction", "exact": "consensus_exact"}
    )
    pair = base.merge(selected, on="image_path", validate="one_to_one")
    fixed = pair[~pair["baseline_exact"] & pair["consensus_exact"]].copy()
    broken = pair[pair["baseline_exact"] & ~pair["consensus_exact"]].copy()
    return fixed, broken


def build_irrecoverable_report(
    test_predictions: pd.DataFrame,
    test_consensus: pd.DataFrame,
    source_csv: Path,
) -> pd.DataFrame:
    source = pd.read_csv(source_csv).copy()
    source["path_key"] = source["image_path"].map(lambda value: str(Path(str(value)).resolve()).lower())
    predictions = test_predictions.copy()
    predictions["path_key"] = predictions["image_path"].map(
        lambda value: str(Path(str(value)).resolve()).lower()
    )
    consensus = test_consensus.copy()
    consensus["path_key"] = consensus["image_path"].map(
        lambda value: str(Path(str(value)).resolve()).lower()
    )
    rows = []
    for source_row in source.itertuples(index=False):
        frame = predictions[predictions["path_key"] == source_row.path_key]
        selected = consensus[consensus["path_key"] == source_row.path_key].iloc[0]
        correct = frame[frame["exact"]]
        best = frame.sort_values(
            ["edit_distance", "normalized_confidence"], ascending=[True, False]
        ).iloc[0]
        with Image.open(source_row.image_path) as opened:
            image_width, image_height = opened.size
        rows.append(
            {
                "file": source_row.file,
                "target": source_row.target,
                "previous_best_prediction": source_row.best_prediction,
                "consensus_prediction": selected["prediction"],
                "consensus_exact": bool(selected["exact"]),
                "consensus_edit_distance": int(selected["edit_distance"]),
                "any_tta_candidate_exact": not correct.empty,
                "correct_candidate_count": len(correct),
                "correct_candidate_views": ";".join(correct["view"].tolist()),
                "oracle_prediction": best["prediction"],
                "oracle_edit_distance": int(best["edit_distance"]),
                "best_candidate_view": best["view"],
                "image_width": image_width,
                "image_height": image_height,
                "aspect_ratio": image_width / max(image_height, 1),
                "image_path": source_row.image_path,
                "copied_image_path": source_row.copied_image_path,
            }
        )
    return pd.DataFrame(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_report(output_dir: Path, summary: dict, top_views: pd.DataFrame, irrecoverable: pd.DataFrame):
    val = summary["validation"]
    test = summary["test"]
    lines = [
        "# Kiểm thử multi-scale zoom TTA cho PARSeq",
        "",
        "Đây là thử nghiệm chỉ thay đổi luồng inference; checkpoint PARSeq không được fine-tune lại.",
        "Tham số consensus được chọn hoàn toàn trên validation và được khóa trước khi đánh giá test.",
        "",
        "## Kết quả",
        "",
        "| Tập dữ liệu | Phương pháp | Exact Match | Character Accuracy | Sửa đúng | Làm sai |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| Validation | Baseline | {val['baseline']['exact_acc']:.4%} | {val['baseline']['char_acc']:.4%} | - | - |",
        f"| Validation | Consensus đã khóa | {val['consensus']['exact_acc']:.4%} | {val['consensus']['char_acc']:.4%} | {val['consensus_delta']['fixed_images']} | {val['consensus_delta']['broken_images']} |",
        f"| Validation | Oracle | {val['oracle']['exact_acc']:.4%} | {val['oracle']['char_acc']:.4%} | {val['oracle_delta']['fixed_images']} | {val['oracle_delta']['broken_images']} |",
        f"| Test | Baseline | {test['baseline']['exact_acc']:.4%} | {test['baseline']['char_acc']:.4%} | - | - |",
        f"| Test | Consensus đã khóa | {test['consensus']['exact_acc']:.4%} | {test['consensus']['char_acc']:.4%} | {test['consensus_delta']['fixed_images']} | {test['consensus_delta']['broken_images']} |",
        f"| Test | Oracle | {test['oracle']['exact_acc']:.4%} | {test['oracle']['char_acc']:.4%} | {test['oracle_delta']['fixed_images']} | {test['oracle_delta']['broken_images']} |",
        "",
        "Oracle dùng target để kiểm tra liệu có bất kỳ nhánh nào đọc đúng hay không; đây không phải kết quả có thể triển khai.",
        "",
        "## Nhóm 21 ảnh trước đây không pipeline nào đọc đúng",
        "",
        f"- Có ít nhất một nhánh TTA đọc đúng: **{int(irrecoverable['any_tta_candidate_exact'].sum())}/21**.",
        f"- Consensus đã khóa tự chọn đúng: **{int(irrecoverable['consensus_exact'].sum())}/21**.",
        "",
        "## Nhận xét",
        "",
        f"- Consensus tăng Exact Match trên test thêm **{test['consensus']['exact_acc'] - test['baseline']['exact_acc']:.4%}** và Character Accuracy thêm **{test['consensus']['char_acc'] - test['baseline']['char_acc']:.4%}**.",
        "- Upscale 2× ở zoom 1.00 là view đơn tốt nhất trên validation; upscale trước enhancement có tác dụng rõ hơn thay đổi zoom đơn thuần.",
        "- Unwrap có accuracy độc lập thấp, nhưng tạo được một số dự đoán đúng cho ảnh rất khó. Vì vậy unwrap chỉ nên là nhánh bỏ phiếu, không dùng mặc định.",
        f"- Selector vẫn làm sai {test['consensus_delta']['broken_images']} ảnh test vốn được baseline đọc đúng; cần cải thiện bước chọn ứng viên trước khi dùng production.",
        "",
        "## Các view đơn tốt nhất trên validation",
        "",
        "| View | Exact Match | Character Accuracy |",
        "| --- | ---: | ---: |",
    ]
    for row in top_views.head(12).itertuples(index=False):
        lines.append(f"| `{row.view}` | {row.exact_acc:.4%} | {row.char_acc:.4%} |")
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint).resolve()
    val_manifest = Path(args.val_manifest).resolve()
    test_manifest = Path(args.test_manifest).resolve()
    model, model_cfg, checkpoint_payload = load_notebook_checkpoint(
        checkpoint, args.device_obj, args.refine_iters
    )
    specs = build_specs()

    all_predictions = {}
    all_view_metrics = {}
    started = time.perf_counter()
    for split, manifest in (("val", val_manifest), ("test", test_manifest)):
        prediction_path = output_dir / f"predictions_{split}_all_views.csv"
        view_result_path = output_dir / f"{split}_view_results.csv"
        if args.reuse_predictions and prediction_path.exists() and view_result_path.exists():
            combined = pd.read_csv(prediction_path)
            views = pd.read_csv(view_result_path)
        else:
            frames = [predict_view(model, model_cfg, manifest, spec, args, split) for spec in specs]
            combined = pd.concat(frames, ignore_index=True)
            views = view_metrics(combined)
            combined.to_csv(prediction_path, index=False)
            views.to_csv(view_result_path, index=False)
        all_predictions[split] = combined
        all_view_metrics[split] = views

    selector, selector_trials = fit_selector(all_predictions["val"], all_view_metrics["val"])
    selector_trials.to_csv(output_dir / "validation_selector_trials.csv", index=False)
    reliability = dict(zip(all_view_metrics["val"]["view"], all_view_metrics["val"]["exact_acc"]))
    eligible = set(selector["eligible_views"])
    val_consensus = consensus_predictions(
        all_predictions["val"], reliability, eligible, selector["confidence_weight"]
    )
    test_consensus = consensus_predictions(
        all_predictions["test"], reliability, eligible, selector["confidence_weight"]
    )
    val_consensus.to_csv(output_dir / "predictions_val_locked_consensus.csv", index=False)
    test_consensus.to_csv(output_dir / "predictions_test_locked_consensus.csv", index=False)

    val_baseline = all_predictions["val"][all_predictions["val"]["view"] == "baseline"]
    test_baseline = all_predictions["test"][all_predictions["test"]["view"] == "baseline"]
    val_oracle = oracle_predictions(all_predictions["val"])
    test_oracle = oracle_predictions(all_predictions["test"])
    val_fixed, val_broken = paired_changes(val_consensus, val_baseline)
    test_fixed, test_broken = paired_changes(test_consensus, test_baseline)
    val_fixed.to_csv(output_dir / "validation_fixed_by_consensus.csv", index=False, encoding="utf-8-sig")
    val_broken.to_csv(output_dir / "validation_broken_by_consensus.csv", index=False, encoding="utf-8-sig")
    test_fixed.to_csv(output_dir / "test_fixed_by_consensus.csv", index=False, encoding="utf-8-sig")
    test_broken.to_csv(output_dir / "test_broken_by_consensus.csv", index=False, encoding="utf-8-sig")
    irrecoverable = build_irrecoverable_report(
        all_predictions["test"], test_consensus, Path(args.irrecoverable_csv).resolve()
    )
    irrecoverable.to_csv(output_dir / "irrecoverable_21_multiscale_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "inference_only_multiscale_zoom_upscale_unwrap_consensus",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "device": str(args.device_obj),
        "candidate_views": len(specs),
        "elapsed_seconds": time.perf_counter() - started,
        "selector": selector,
        "validation": {
            "baseline": metrics(val_baseline),
            "consensus": metrics(val_consensus),
            "consensus_delta": paired_summary(val_consensus, val_baseline),
            "oracle": metrics(val_oracle),
            "oracle_delta": paired_summary(val_oracle, val_baseline),
        },
        "test": {
            "baseline": metrics(test_baseline),
            "consensus": metrics(test_consensus),
            "consensus_delta": paired_summary(test_consensus, test_baseline),
            "oracle": metrics(test_oracle),
            "oracle_delta": paired_summary(test_oracle, test_baseline),
        },
        "irrecoverable_21": {
            "recognized_by_any_candidate": int(irrecoverable["any_tta_candidate_exact"].sum()),
            "recognized_by_locked_consensus": int(irrecoverable["consensus_exact"].sum()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, summary, all_view_metrics["val"], irrecoverable)
    return summary


def parse_args():
    default_run = ROOT / "outputs/testing/refinement_finetune_20260710_142307"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(default_run / "best_official_parseq_anpr.pt"))
    parser.add_argument("--val-manifest", default=str(default_run / "eval_val_predictions_best_refine.csv"))
    parser.add_argument("--test-manifest", default=str(default_run / "eval_test_predictions_best_refine.csv"))
    parser.add_argument(
        "--irrecoverable-csv",
        default=str(
            ROOT
            / "outputs/testing/irrecoverable_wrong_images_8pipelines/irrecoverable_wrong_images_8pipelines.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs/testing/preprocessing_multiscale_tta_benchmark")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse existing all-view CSV files and rerun only selector/report generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))

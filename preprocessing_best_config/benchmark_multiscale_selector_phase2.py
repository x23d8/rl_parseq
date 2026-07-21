"""Phase 2 selector for the multi-scale PARSeq TTA experiment.

This script does not run PARSeq and does not update OCR weights. It reuses the
65-view prediction CSV files produced by ``benchmark_multiscale_tta.py``.
Selector parameters are learned only from validation predictions. Validation
quality is estimated with grouped out-of-fold predictions; the selector is
then fitted once on all validation data and locked before evaluating test.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = ROOT / "outputs/testing/preprocessing_multiscale_tta_benchmark"
EPSILON = 1e-9
RANDOM_SEED = 20260714


def edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance between two short plate strings."""

    left, right = str(left), str(right)
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def evaluate(frame: pd.DataFrame) -> dict:
    edits = int(frame["edit_distance"].sum())
    characters = int(frame["target_length"].sum())
    return {
        "samples": int(len(frame)),
        "exact_acc": float(frame["exact"].mean()),
        "char_acc": float(1.0 - edits / max(characters, 1)),
        "cer": float(edits / max(characters, 1)),
        "errors": int((~frame["exact"]).sum()),
        "edit_errors": edits,
    }


def compare_with(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    paired = reference[["image_path", "exact"]].merge(
        candidate[["image_path", "exact"]],
        on="image_path",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    return {
        "fixed_images": int((~paired["exact_reference"] & paired["exact_candidate"]).sum()),
        "broken_images": int((paired["exact_reference"] & ~paired["exact_candidate"]).sum()),
    }


def normalize_path(value: str) -> str:
    return str(Path(str(value)).resolve()).lower()


def plate_shape(value: str) -> str:
    return "".join("D" if char.isdigit() else "L" if char.isalpha() else "X" for char in str(value))


def load_image_features(paths: list[str]) -> pd.DataFrame:
    rows = []
    for image_path in paths:
        with Image.open(image_path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        aspect = width / max(height, 1)
        if width < 64 or height < 24:
            scale_class = "tiny"
        elif width < 100 or height < 40:
            scale_class = "small"
        else:
            scale_class = "regular"
        layout_class = "two_line" if aspect < 1.9 else "single_line"
        rows.append(
            {
                "image_path": str(image_path),
                "image_width": float(width),
                "image_height": float(height),
                "aspect_ratio": float(aspect),
                "log_image_area": float(math.log1p(width * height)),
                "gray_mean": float(gray.mean() / 255.0),
                "gray_std": float(gray.std() / 255.0),
                "saturation_mean": float(hsv[..., 1].mean() / 255.0),
                "laplacian_log_variance": float(math.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
                "is_tiny": float(scale_class == "tiny"),
                "is_small": float(scale_class != "regular"),
                "is_two_line": float(layout_class == "two_line"),
                "route_class": f"{scale_class}_{layout_class}",
            }
        )
    return pd.DataFrame(rows).set_index("image_path", drop=False)


@dataclass
class ReferenceStats:
    view_reliability: dict[str, float]
    contextual_reliability: dict[str, float]
    length_log_prior: dict[int, float]
    shape_log_prior: dict[str, float]
    default_length_log_prior: float
    default_shape_log_prior: float
    confidence_calibrator: IsotonicRegression


def serialize_reference_stats(stats: ReferenceStats) -> dict:
    """Use a plain dictionary so the joblib artifact is import-path independent."""

    return {
        "view_reliability": stats.view_reliability,
        "contextual_reliability": stats.contextual_reliability,
        "length_log_prior": stats.length_log_prior,
        "shape_log_prior": stats.shape_log_prior,
        "default_length_log_prior": stats.default_length_log_prior,
        "default_shape_log_prior": stats.default_shape_log_prior,
        "confidence_calibrator": stats.confidence_calibrator,
    }


def deserialize_reference_stats(payload: dict) -> ReferenceStats:
    return ReferenceStats(**payload)


def fit_reference_stats(predictions: pd.DataFrame, images: pd.DataFrame) -> ReferenceStats:
    image_routes = images[["image_path", "route_class"]].reset_index(drop=True)
    enriched = predictions.merge(
        image_routes, on="image_path", how="left", validate="many_to_one"
    )
    view_reliability = enriched.groupby("view")["exact"].mean().astype(float).to_dict()
    contextual_reliability = {}
    shrinkage = 20.0
    for (route, view), group in enriched.groupby(["route_class", "view"], sort=False):
        global_reliability = view_reliability[str(view)]
        value = (float(group["exact"].sum()) + shrinkage * global_reliability) / (
            len(group) + shrinkage
        )
        contextual_reliability[f"{route}|{view}"] = float(value)

    targets = enriched.drop_duplicates("image_path")["target"].astype(str)
    length_counts = Counter(targets.str.len().tolist())
    shape_counts = Counter(targets.map(plate_shape).tolist())
    length_denominator = len(targets) + len(length_counts) + 1
    shape_denominator = len(targets) + len(shape_counts) + 1
    length_log_prior = {
        int(key): float(math.log((count + 1) / length_denominator))
        for key, count in length_counts.items()
    }
    shape_log_prior = {
        str(key): float(math.log((count + 1) / shape_denominator))
        for key, count in shape_counts.items()
    }

    # One row per (image, candidate) prevents a prediction repeated by many
    # views from dominating confidence calibration.
    calibration = (
        enriched.groupby(["image_path", "prediction"], sort=False)
        .agg(normalized_confidence=("normalized_confidence", "max"), exact=("exact", "first"))
        .reset_index()
    )
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(
        calibration["normalized_confidence"].astype(float).to_numpy(),
        calibration["exact"].astype(float).to_numpy(),
    )
    return ReferenceStats(
        view_reliability=view_reliability,
        contextual_reliability=contextual_reliability,
        length_log_prior=length_log_prior,
        shape_log_prior=shape_log_prior,
        default_length_log_prior=float(math.log(1 / length_denominator)),
        default_shape_log_prior=float(math.log(1 / shape_denominator)),
        confidence_calibrator=calibrator,
    )


def _context_reliability(stats: ReferenceStats, route: str, view: str) -> float:
    return stats.contextual_reliability.get(
        f"{route}|{view}", stats.view_reliability.get(view, 0.5)
    )


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def build_candidate_features(
    predictions: pd.DataFrame, images: pd.DataFrame, stats: ReferenceStats
) -> pd.DataFrame:
    """Aggregate 65 view rows into one feature row per unique prediction."""

    rows: list[dict] = []
    for image_path, image_group in predictions.groupby("image_path", sort=False):
        image = images.loc[image_path]
        route = str(image["route_class"])
        baseline_rows = image_group[image_group["view"] == "baseline"]
        baseline_prediction = str(
            baseline_rows.iloc[0]["prediction"] if not baseline_rows.empty else image_group.iloc[0]["prediction"]
        )
        total_views = max(len(image_group), 1)
        contextual_by_view = {
            str(row.view): _context_reliability(stats, route, str(row.view))
            for row in image_group.itertuples(index=False)
        }
        total_context_support = sum(contextual_by_view.values())
        candidate_groups = list(image_group.groupby("prediction", sort=False))
        vote_counts = {str(prediction): len(group) for prediction, group in candidate_groups}
        phase1_supports = {
            str(prediction): sum(
                max(stats.view_reliability.get(str(view), 0.5) - 0.5, 0.01)
                for view in group["view"]
            )
            for prediction, group in candidate_groups
        }
        context_supports = {
            str(prediction): sum(contextual_by_view[str(view)] for view in group["view"])
            for prediction, group in candidate_groups
        }

        for prediction_value, group in candidate_groups:
            prediction = str(prediction_value)
            views = group["view"].astype(str).tolist()
            context_values = np.asarray([contextual_by_view[view] for view in views], dtype=float)
            normalized_confidences = group["normalized_confidence"].astype(float).to_numpy()
            raw_confidences = group["confidence"].astype(float).to_numpy()
            calibrated = stats.confidence_calibrator.predict(normalized_confidences)
            vote_count = len(group)
            other_votes = [count for key, count in vote_counts.items() if key != prediction]
            other_context = [value for key, value in context_supports.items() if key != prediction]
            fuzzy_similarities = []
            fuzzy_weighted = []
            for all_row in image_group.itertuples(index=False):
                other = str(all_row.prediction)
                similarity = 1.0 - edit_distance(prediction, other) / max(len(prediction), len(other), 1)
                fuzzy_similarities.append(similarity)
                fuzzy_weighted.append(similarity * contextual_by_view[str(all_row.view)])

            letters = sum(char.isalpha() for char in prediction)
            digits = sum(char.isdigit() for char in prediction)
            transitions = sum(
                plate_shape(prediction)[index] != plate_shape(prediction)[index - 1]
                for index in range(1, len(prediction))
            )
            standard_private = bool(
                re.fullmatch(r"\d{2}[A-Z][A-Z0-9]?\d{4,6}", prediction)
            )
            military_or_special = bool(re.fullmatch(r"[A-Z]{1,2}\d{4,7}", prediction))
            full_votes = int((~group["unwrap_two_line"].astype(bool)).sum())
            unwrap_votes = vote_count - full_votes
            up2_votes = int(np.isclose(group["upscale"].astype(float), 2.0).sum())
            up3_votes = int(np.isclose(group["upscale"].astype(float), 3.0).sum())
            zooms = group["zoom"].astype(float)

            row = {
                "image_path": image_path,
                "target": str(group.iloc[0]["target"]),
                "prediction": prediction,
                "exact": bool(group.iloc[0]["exact"]),
                "edit_distance": int(group.iloc[0]["edit_distance"]),
                "target_length": int(group.iloc[0]["target_length"]),
                "supporting_views": ";".join(views),
                "route_class": route,
                "baseline_prediction": baseline_prediction,
                "vote_count": float(vote_count),
                "vote_fraction": float(vote_count / total_views),
                "vote_margin": float((vote_count - max(other_votes, default=0)) / total_views),
                "phase1_support": float(phase1_supports[prediction]),
                "phase1_support_fraction": float(
                    phase1_supports[prediction] / max(sum(phase1_supports.values()), EPSILON)
                ),
                "context_support": float(context_supports[prediction]),
                "context_support_fraction": float(
                    context_supports[prediction] / max(total_context_support, EPSILON)
                ),
                "context_support_margin": float(
                    (context_supports[prediction] - max(other_context, default=0.0))
                    / max(total_context_support, EPSILON)
                ),
                "support_reliability_mean": _safe_float(context_values.mean()),
                "support_reliability_max": _safe_float(context_values.max()),
                "normalized_confidence_max": _safe_float(normalized_confidences.max()),
                "normalized_confidence_mean": _safe_float(normalized_confidences.mean()),
                "normalized_confidence_min": _safe_float(normalized_confidences.min()),
                "normalized_confidence_std": _safe_float(normalized_confidences.std()),
                "raw_confidence_max": _safe_float(raw_confidences.max()),
                "calibrated_confidence_max": _safe_float(calibrated.max()),
                "calibrated_confidence_mean": _safe_float(calibrated.mean()),
                "fuzzy_consensus": _safe_float(np.mean(fuzzy_similarities)),
                "fuzzy_context_consensus": float(
                    sum(fuzzy_weighted) / max(total_context_support, EPSILON)
                ),
                "baseline_match": float(prediction == baseline_prediction),
                "distance_to_baseline": float(edit_distance(prediction, baseline_prediction)),
                "prediction_length": float(len(prediction)),
                "letter_count": float(letters),
                "digit_count": float(digits),
                "type_transitions": float(transitions),
                "starts_two_digits": float(len(prediction) >= 2 and prediction[:2].isdigit()),
                "standard_private_pattern": float(standard_private),
                "military_special_pattern": float(military_or_special),
                "alphanumeric_only": float(prediction.isalnum()),
                "length_log_prior": stats.length_log_prior.get(
                    len(prediction), stats.default_length_log_prior
                ),
                "shape_log_prior": stats.shape_log_prior.get(
                    plate_shape(prediction), stats.default_shape_log_prior
                ),
                "full_vote_fraction": float(full_votes / total_views),
                "unwrap_vote_fraction": float(unwrap_votes / total_views),
                "up2_vote_fraction": float(up2_votes / total_views),
                "up3_vote_fraction": float(up3_votes / total_views),
                "zoom_out_vote_fraction": float((zooms < 0.999).sum() / total_views),
                "zoom_in_vote_fraction": float((zooms > 1.001).sum() / total_views),
                "preprocess_diversity": float(group["preprocessing"].nunique() / 4.0),
                "zoom_diversity": float(group["zoom"].nunique() / 5.0),
            }
            for preprocessing in (
                "train_baseline",
                "clahe_clip1_tile4",
                "clahe_rl_deblur_bilateral",
                "adaptive_noise_3way",
            ):
                row[f"votes_{preprocessing}"] = float(
                    (group["preprocessing"].astype(str) == preprocessing).sum() / total_views
                )
            for image_feature in (
                "image_width",
                "image_height",
                "aspect_ratio",
                "log_image_area",
                "gray_mean",
                "gray_std",
                "saturation_mean",
                "laplacian_log_variance",
                "is_tiny",
                "is_small",
                "is_two_line",
            ):
                row[image_feature] = float(image[image_feature])
            # Explicit routing interactions let the linear ranker prefer an
            # unwrap/upscale branch only for a matching image geometry.
            row["unwrap_two_line_interaction"] = row["unwrap_vote_fraction"] * row["is_two_line"]
            row["upscale_small_interaction"] = (
                row["up2_vote_fraction"] + row["up3_vote_fraction"]
            ) * row["is_small"]
            row["up3_tiny_interaction"] = row["up3_vote_fraction"] * row["is_tiny"]
            rows.append(row)
    return pd.DataFrame(rows)


NON_FEATURE_COLUMNS = {
    "image_path",
    "target",
    "prediction",
    "exact",
    "edit_distance",
    "target_length",
    "supporting_views",
    "route_class",
    "baseline_prediction",
}


def feature_columns(candidates: pd.DataFrame) -> list[str]:
    return [column for column in candidates.columns if column not in NON_FEATURE_COLUMNS]


def pairwise_training_data(candidates: pd.DataFrame, columns: list[str]):
    differences = []
    labels = []
    weights = []
    for _image_path, group in candidates.groupby("image_path", sort=False):
        positives = group[group["exact"]]
        negatives = group[~group["exact"]]
        if positives.empty or negatives.empty:
            continue
        positive = positives.iloc[0][columns].astype(float).to_numpy()
        per_pair_weight = 1.0 / (2.0 * len(negatives))
        for negative_row in negatives.itertuples(index=False):
            negative = np.asarray([getattr(negative_row, column) for column in columns], dtype=float)
            difference = positive - negative
            differences.extend((difference, -difference))
            labels.extend((1, 0))
            weights.extend((per_pair_weight, per_pair_weight))
    if not differences:
        raise RuntimeError("Validation data has no recoverable ambiguous images for selector training.")
    return np.asarray(differences), np.asarray(labels), np.asarray(weights)


def make_ranker(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ranker",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=5000,
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def fit_ranker(candidates: pd.DataFrame, columns: list[str], c_value: float) -> Pipeline:
    features, labels, weights = pairwise_training_data(candidates, columns)
    model = make_ranker(c_value)
    model.fit(features, labels, ranker__sample_weight=weights)
    return model


def score_candidates(model: Pipeline, candidates: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scored = candidates.copy()
    scored["rank_score"] = model.decision_function(scored[columns].astype(float).to_numpy())
    return scored


def _phase1_row(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        ["phase1_support", "vote_count", "normalized_confidence_max", "prediction_length"],
        ascending=[False, False, False, True],
    ).iloc[0]


def select_predictions(candidates: pd.DataFrame, switch_margin: float) -> pd.DataFrame:
    selected = []
    for _image_path, group in candidates.groupby("image_path", sort=False):
        phase1 = _phase1_row(group)
        model_best = group.sort_values(
            ["rank_score", "phase1_support", "vote_count", "normalized_confidence_max"],
            ascending=False,
        ).iloc[0]
        score_gain = float(model_best["rank_score"] - phase1["rank_score"])
        chosen = model_best if score_gain >= switch_margin else phase1
        selected.append(
            {
                "image_path": chosen["image_path"],
                "target": chosen["target"],
                "prediction": chosen["prediction"],
                "exact": bool(chosen["exact"]),
                "edit_distance": int(chosen["edit_distance"]),
                "target_length": int(chosen["target_length"]),
                "normalized_confidence": float(chosen["normalized_confidence_max"]),
                "votes": int(chosen["vote_count"]),
                "rank_score": float(chosen["rank_score"]),
                "phase1_prediction": str(phase1["prediction"]),
                "phase1_rank_score": float(phase1["rank_score"]),
                "score_gain_over_phase1": score_gain,
                "switched_from_phase1": bool(chosen["prediction"] != phase1["prediction"]),
                "route_class": chosen["route_class"],
                "supporting_views": chosen["supporting_views"],
            }
        )
    return pd.DataFrame(selected)


def phase1_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    placeholder = candidates.copy()
    placeholder["rank_score"] = 0.0
    return select_predictions(placeholder, switch_margin=float("inf"))


def baseline_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[predictions["view"] == "baseline"].copy()


def assign_validation_folds(predictions: pd.DataFrame, folds: int) -> dict[str, int]:
    image_rows = []
    for image_path, group in predictions.groupby("image_path", sort=False):
        baseline_exact = bool(group.loc[group["view"] == "baseline", "exact"].iloc[0])
        oracle_exact = bool(group["exact"].any())
        category = "baseline_correct" if baseline_exact else "recoverable" if oracle_exact else "unrecoverable"
        image_rows.append({"image_path": image_path, "category": category})
    images = pd.DataFrame(image_rows)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    assignments = {}
    dummy = np.zeros(len(images))
    for fold, (_train, holdout) in enumerate(splitter.split(dummy, images["category"])):
        for index in holdout:
            assignments[str(images.iloc[index]["image_path"])] = fold
    return assignments


def make_change_table(candidate: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = reference[["image_path", "target", "prediction", "exact"]].rename(
        columns={"prediction": "reference_prediction", "exact": "reference_exact"}
    )
    right = candidate[
        [
            "image_path",
            "prediction",
            "exact",
            "score_gain_over_phase1",
            "route_class",
            "supporting_views",
        ]
    ].rename(columns={"prediction": "phase2_prediction", "exact": "phase2_exact"})
    paired = left.merge(right, on="image_path", validate="one_to_one")
    return (
        paired[~paired["reference_exact"] & paired["phase2_exact"]].copy(),
        paired[paired["reference_exact"] & ~paired["phase2_exact"]].copy(),
    )


def irrecoverable_report(
    test_predictions: pd.DataFrame,
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
    source_csv: Path,
) -> pd.DataFrame:
    source = pd.read_csv(source_csv)
    source["path_key"] = source["image_path"].map(normalize_path)
    raw = test_predictions.copy()
    raw["path_key"] = raw["image_path"].map(normalize_path)
    first = phase1.copy()
    first["path_key"] = first["image_path"].map(normalize_path)
    second = phase2.copy()
    second["path_key"] = second["image_path"].map(normalize_path)
    rows = []
    for item in source.itertuples(index=False):
        raw_group = raw[raw["path_key"] == item.path_key]
        phase1_row = first[first["path_key"] == item.path_key].iloc[0]
        phase2_row = second[second["path_key"] == item.path_key].iloc[0]
        correct = raw_group[raw_group["exact"]]
        rows.append(
            {
                "file": item.file,
                "target": item.target,
                "previous_best_prediction": item.best_prediction,
                "phase1_prediction": phase1_row["prediction"],
                "phase1_exact": bool(phase1_row["exact"]),
                "phase2_prediction": phase2_row["prediction"],
                "phase2_exact": bool(phase2_row["exact"]),
                "phase2_edit_distance": int(phase2_row["edit_distance"]),
                "phase2_switched": bool(phase2_row["switched_from_phase1"]),
                "score_gain_over_phase1": float(phase2_row["score_gain_over_phase1"]),
                "route_class": phase2_row["route_class"],
                "any_candidate_exact": bool(raw_group["exact"].any()),
                "correct_candidate_count": int(len(correct)),
                "correct_candidate_views": ";".join(correct["view"].astype(str).tolist()),
                "phase2_supporting_views": phase2_row["supporting_views"],
                "image_path": item.image_path,
                "copied_image_path": item.copied_image_path,
            }
        )
    return pd.DataFrame(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_report(output_dir: Path, summary: dict, trials: pd.DataFrame):
    validation = summary["validation"]
    test = summary["test"]
    hard = summary["irrecoverable_21"]
    best = summary["selector"]
    lines = [
        "# Phase 2: calibrated candidate selector cho multi-scale PARSeq TTA",
        "",
        "Phase 2 chỉ thay đổi cách chọn kết quả từ 65 nhánh inference của Phase 1; không fine-tune và không cập nhật trọng số PARSeq.",
        "Selector được chọn bằng Group K-fold trên validation, sau đó fit lại trên toàn bộ validation và khóa trước khi chạy test.",
        "",
        "## Kết quả chính",
        "",
        "| Tập dữ liệu | Phương pháp | Exact Match | Character Accuracy |",
        "| --- | --- | ---: | ---: |",
        f"| Validation OOF | Baseline | {validation['baseline']['exact_acc']:.4%} | {validation['baseline']['char_acc']:.4%} |",
        f"| Validation OOF | Phase 1 consensus | {validation['phase1_oof']['exact_acc']:.4%} | {validation['phase1_oof']['char_acc']:.4%} |",
        f"| Validation OOF | Phase 2 selector | {validation['phase2_oof']['exact_acc']:.4%} | {validation['phase2_oof']['char_acc']:.4%} |",
        f"| Test | Baseline | {test['baseline']['exact_acc']:.4%} | {test['baseline']['char_acc']:.4%} |",
        f"| Test | Phase 1 consensus tái tạo | {test['phase1']['exact_acc']:.4%} | {test['phase1']['char_acc']:.4%} |",
        f"| Test | Phase 2 selector đã khóa | {test['phase2']['exact_acc']:.4%} | {test['phase2']['char_acc']:.4%} |",
        "",
        "## Selector đã khóa",
        "",
        f"- Pairwise logistic C: `{best['C']}`.",
        f"- Ngưỡng chuyển khỏi kết quả Phase 1: `{best['switch_margin']}`.",
        f"- Số feature: `{best['feature_count']}`.",
        f"- Phase 2 sửa đúng/làm sai so với baseline trên test: **{test['phase2_vs_baseline']['fixed_images']}/{test['phase2_vs_baseline']['broken_images']}**.",
        f"- Phase 2 sửa đúng/làm sai so với Phase 1 trên test: **{test['phase2_vs_phase1']['fixed_images']}/{test['phase2_vs_phase1']['broken_images']}**.",
        "",
        "## 21 ảnh khó",
        "",
        f"- Có ít nhất một ứng viên đúng: **{hard['recognized_by_any_candidate']}/21**.",
        f"- Phase 1 tự chọn đúng: **{hard['recognized_by_phase1']}/21**.",
        f"- Phase 2 tự chọn đúng: **{hard['recognized_by_phase2']}/21**.",
        "",
        "## Cách selector hoạt động",
        "",
        "Mỗi chuỗi dự đoán duy nhất được gộp phiếu từ các view. Ranker sử dụng số phiếu, độ tin cậy đã hiệu chỉnh, độ tin cậy lịch sử của view, đồng thuận gần đúng theo edit distance, cấu trúc biển số và tương tác giữa kích thước/tỉ lệ ảnh với upscale hoặc unwrap. Chỉ chuyển khỏi consensus Phase 1 khi chênh lệch điểm vượt ngưỡng đã chọn trên validation.",
        "",
        "## Các cấu hình validation tốt nhất",
        "",
        "| C | Switch margin | Exact Match OOF | Character Accuracy OOF | Sửa đúng | Làm sai |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trials.head(10).itertuples(index=False):
        lines.append(
            f"| {row.C:g} | {row.switch_margin:g} | {row.exact_acc:.4%} | {row.char_acc:.4%} | {row.fixed_vs_baseline} | {row.broken_vs_baseline} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    source_dir = Path(args.phase1_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions = pd.read_csv(source_dir / "predictions_val_all_views.csv")
    test_predictions = pd.read_csv(source_dir / "predictions_test_all_views.csv")
    required = {"image_path", "target", "prediction", "view", "exact", "normalized_confidence"}
    missing = required - set(val_predictions.columns)
    if missing:
        raise ValueError(f"Phase 1 prediction CSV is missing columns: {sorted(missing)}")
    for frame in (val_predictions, test_predictions):
        frame["image_path"] = frame["image_path"].astype(str)
        frame["prediction"] = frame["prediction"].fillna("").astype(str)
        frame["target"] = frame["target"].fillna("").astype(str)
        frame["exact"] = frame["exact"].astype(bool)

    all_paths = list(dict.fromkeys(pd.concat([val_predictions, test_predictions])["image_path"].tolist()))
    all_images = load_image_features(all_paths)
    fold_assignments = assign_validation_folds(val_predictions, args.folds)
    c_values = [float(value) for value in args.c_values.split(",")]
    margins = [float(value) for value in args.switch_margins.split(",")]
    oof_by_c: dict[float, list[pd.DataFrame]] = {value: [] for value in c_values}
    phase1_oof_parts = []

    for fold in range(args.folds):
        holdout_paths = {path for path, assigned in fold_assignments.items() if assigned == fold}
        train_predictions = val_predictions[~val_predictions["image_path"].isin(holdout_paths)]
        holdout_predictions = val_predictions[val_predictions["image_path"].isin(holdout_paths)]
        train_images = all_images.loc[train_predictions["image_path"].drop_duplicates()]
        stats = fit_reference_stats(train_predictions, train_images)
        train_candidates = build_candidate_features(train_predictions, all_images, stats)
        holdout_candidates = build_candidate_features(holdout_predictions, all_images, stats)
        columns = feature_columns(train_candidates)
        phase1_oof_parts.append(phase1_predictions(holdout_candidates))
        for c_value in c_values:
            model = fit_ranker(train_candidates, columns, c_value)
            scored = score_candidates(model, holdout_candidates, columns)
            scored["fold"] = fold
            oof_by_c[c_value].append(scored)

    validation_baseline = baseline_predictions(val_predictions)
    phase1_oof = pd.concat(phase1_oof_parts, ignore_index=True)
    phase1_oof.to_csv(output_dir / "predictions_val_phase1_oof.csv", index=False, encoding="utf-8-sig")
    trial_rows = []
    selected_by_trial = {}
    for c_value, parts in oof_by_c.items():
        scored_oof = pd.concat(parts, ignore_index=True)
        for margin in margins:
            selected = select_predictions(scored_oof, margin)
            result = evaluate(selected)
            delta_baseline = compare_with(selected, validation_baseline)
            delta_phase1 = compare_with(selected, phase1_oof)
            key = (c_value, margin)
            selected_by_trial[key] = selected
            trial_rows.append(
                {
                    "C": c_value,
                    "switch_margin": margin,
                    **result,
                    "fixed_vs_baseline": delta_baseline["fixed_images"],
                    "broken_vs_baseline": delta_baseline["broken_images"],
                    "fixed_vs_phase1": delta_phase1["fixed_images"],
                    "broken_vs_phase1": delta_phase1["broken_images"],
                    "switches": int(selected["switched_from_phase1"].sum()),
                }
            )
    trials = pd.DataFrame(trial_rows)
    trials = trials.sort_values(
        ["exact_acc", "char_acc", "broken_vs_baseline", "switch_margin", "C"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    trials.to_csv(output_dir / "validation_selector_trials_oof.csv", index=False)
    best_trial = trials.iloc[0]
    best_key = (float(best_trial["C"]), float(best_trial["switch_margin"]))
    phase2_oof = selected_by_trial[best_key]
    phase2_oof.to_csv(output_dir / "predictions_val_phase2_oof.csv", index=False, encoding="utf-8-sig")

    # Final fit uses validation only. Test targets are not consulted until all
    # parameters and the switch threshold have been locked.
    full_stats = fit_reference_stats(val_predictions, all_images.loc[val_predictions["image_path"].drop_duplicates()])
    val_candidates = build_candidate_features(val_predictions, all_images, full_stats)
    test_candidates = build_candidate_features(test_predictions, all_images, full_stats)
    columns = feature_columns(val_candidates)
    final_model = fit_ranker(val_candidates, columns, best_key[0])
    val_scored = score_candidates(final_model, val_candidates, columns)
    test_scored = score_candidates(final_model, test_candidates, columns)
    phase2_val_fullfit = select_predictions(val_scored, best_key[1])
    phase1_test = phase1_predictions(test_scored)
    phase2_test = select_predictions(test_scored, best_key[1])
    phase2_val_fullfit.to_csv(output_dir / "predictions_val_phase2_fullfit_diagnostic.csv", index=False, encoding="utf-8-sig")
    phase1_test.to_csv(output_dir / "predictions_test_phase1_reconstructed.csv", index=False, encoding="utf-8-sig")
    phase2_test.to_csv(output_dir / "predictions_test_phase2_locked.csv", index=False, encoding="utf-8-sig")

    changed_phase1 = phase2_test[
        phase2_test["prediction"] != phase2_test["phase1_prediction"]
    ].copy()
    changed_phase1["edit_delta_vs_phase1"] = changed_phase1.apply(
        lambda row: int(row["edit_distance"])
        - edit_distance(str(row["phase1_prediction"]), str(row["target"])),
        axis=1,
    )
    changed_phase1.to_csv(
        output_dir / "test_changed_vs_phase1.csv", index=False, encoding="utf-8-sig"
    )

    test_baseline = baseline_predictions(test_predictions)
    fixed_baseline, broken_baseline = make_change_table(phase2_test, test_baseline)
    fixed_phase1, broken_phase1 = make_change_table(phase2_test, phase1_test)
    fixed_baseline.to_csv(output_dir / "test_fixed_vs_baseline.csv", index=False, encoding="utf-8-sig")
    broken_baseline.to_csv(output_dir / "test_broken_vs_baseline.csv", index=False, encoding="utf-8-sig")
    fixed_phase1.to_csv(output_dir / "test_fixed_vs_phase1.csv", index=False, encoding="utf-8-sig")
    broken_phase1.to_csv(output_dir / "test_broken_vs_phase1.csv", index=False, encoding="utf-8-sig")

    hard = irrecoverable_report(
        test_predictions,
        phase1_test,
        phase2_test,
        Path(args.irrecoverable_csv).resolve(),
    )
    hard.to_csv(output_dir / "irrecoverable_21_phase2_results.csv", index=False, encoding="utf-8-sig")
    hard_keys = set(hard["image_path"].map(normalize_path))
    hard_candidate_diagnostics = test_scored[
        test_scored["image_path"].map(normalize_path).isin(hard_keys)
    ].sort_values(["image_path", "rank_score"], ascending=[True, False])
    hard_candidate_diagnostics.to_csv(
        output_dir / "irrecoverable_21_candidate_scores.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(
        {
            "model": final_model,
            "reference_stats": serialize_reference_stats(full_stats),
            "feature_columns": columns,
            "switch_margin": best_key[1],
            "C": best_key[0],
            "phase1_source_dir": str(source_dir),
        },
        output_dir / "phase2_selector.joblib",
    )

    summary = {
        "experiment": "phase2_calibrated_resolution_aware_candidate_selector",
        "data_policy": "fit_and_select_on_validation_only_then_lock_for_test",
        "phase1_source_dir": str(source_dir),
        "selector": {
            "model": "pairwise_logistic_regression",
            "C": best_key[0],
            "switch_margin": best_key[1],
            "feature_count": len(columns),
            "folds": args.folds,
            "features": columns,
        },
        "validation": {
            "baseline": evaluate(validation_baseline),
            "phase1_oof": evaluate(phase1_oof),
            "phase2_oof": evaluate(phase2_oof),
            "phase2_oof_vs_baseline": compare_with(phase2_oof, validation_baseline),
            "phase2_oof_vs_phase1": compare_with(phase2_oof, phase1_oof),
            "phase2_fullfit_diagnostic": evaluate(phase2_val_fullfit),
        },
        "test": {
            "baseline": evaluate(test_baseline),
            "phase1": evaluate(phase1_test),
            "phase2": evaluate(phase2_test),
            "phase2_vs_baseline": compare_with(phase2_test, test_baseline),
            "phase2_vs_phase1": compare_with(phase2_test, phase1_test),
        },
        "irrecoverable_21": {
            "recognized_by_any_candidate": int(hard["any_candidate_exact"].sum()),
            "recognized_by_phase1": int(hard["phase1_exact"].sum()),
            "recognized_by_phase2": int(hard["phase2_exact"].sum()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, summary, trials)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", default=str(PHASE1_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs/testing/preprocessing_multiscale_selector_phase2"),
    )
    parser.add_argument(
        "--irrecoverable-csv",
        default=str(
            ROOT
            / "outputs/testing/irrecoverable_wrong_images_8pipelines/irrecoverable_wrong_images_8pipelines.csv"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--switch-margins", default="0,0.05,0.1,0.2,0.4,0.75,1.25,2,1000000000")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))

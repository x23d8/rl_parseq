"""Label-free, guard-short-circuited runtime for promoted Phase 12."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.action_space import COMPACT_VIEWS  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.runtime import (  # noqa: E402
    runtime_view_features,
    summarize_runtime_latency,
    synchronize_device,
)
from reinforcement_learning.phase_8_consensus_ppo.evaluate import (  # noqa: E402
    checkpoint_selection,
    load_checkpoint,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.evaluate import guard_sha256  # noqa: E402
from reinforcement_learning.phase_12_guarded_replicated_ppo.prepare_fresh_holdout import validate_lock  # noqa: E402
from reinforcement_learning.phase_12_guarded_replicated_ppo.promote import (  # noqa: E402
    validate_promotion_summary,
    validate_receipt,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.selection import guarded_selection  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_runtime_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"image_path", "input_contract", "input_transform"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Phase 12 runtime manifest lacks columns: {sorted(missing)}")
    columns = ["image_path", "input_contract", "input_transform"]
    for optional in ("crop_width", "crop_height"):
        if optional in manifest.columns:
            columns.append(optional)
    frame = manifest[columns].dropna(subset=list(required)).copy()
    frame.image_path = frame.image_path.astype(str).str.strip()
    frame.input_contract = frame.input_contract.astype(str).str.strip().str.lower()
    frame.input_transform = frame.input_transform.astype(str).str.strip().str.lower()
    if frame.empty or (frame.image_path == "").any() or frame.image_path.duplicated().any():
        raise ValueError("Phase 12 runtime image paths must be unique and non-empty")
    if not (frame.input_contract == "plate_crop").all():
        raise ValueError("Phase 12 runtime accepts only input_contract=plate_crop")
    if not set(frame.input_transform).issubset(
        {"existing_plate_crop", "crop_source_bounding_box"}
    ):
        raise ValueError("Phase 12 runtime received an unsupported input_transform")
    actual_width, actual_height = [], []
    for value in frame.image_path:
        image_path = Path(value)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Runtime crop has invalid dimensions: {image_path}")
        actual_width.append(width)
        actual_height.append(height)
    if "crop_width" in frame:
        declared = pd.to_numeric(frame.crop_width, errors="raise").astype(int).to_numpy()
        if not np.array_equal(declared, np.asarray(actual_width)):
            raise ValueError("Declared runtime crop_width differs from image content")
    if "crop_height" in frame:
        declared = pd.to_numeric(frame.crop_height, errors="raise").astype(int).to_numpy()
        if not np.array_equal(declared, np.asarray(actual_height)):
            raise ValueError("Declared runtime crop_height differs from image content")
    frame["crop_width"] = actual_width
    frame["crop_height"] = actual_height
    return frame.reset_index(drop=True)


def load_active_registry(path: Path) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or registry.get("status") != "active_external_validated":
        raise ValueError("Phase 12 registry is not externally validated and active")
    evaluation_path = Path(registry.get("external_evaluation", "")).resolve()
    if not evaluation_path.is_file() or sha256_file(evaluation_path) != registry.get("external_evaluation_sha256"):
        raise ValueError("Phase 12 promotion evaluation is unavailable or changed")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    validate_receipt(summary, evaluation_path)
    lock_path = Path(registry.get("candidate_lock", "")).resolve()
    if not lock_path.is_file() or sha256_file(lock_path) != registry.get("candidate_lock_sha256"):
        raise ValueError("Phase 12 candidate lock changed after promotion")
    lock = validate_lock(lock_path)
    if lock["guard"] != registry.get("guard") or guard_sha256(lock["guard"]) != registry.get("guard_sha256"):
        raise ValueError("Phase 12 runtime guard differs from the promoted guard")
    policy = Path(registry.get("policy_checkpoint", "")).resolve()
    parseq = Path(registry.get("parseq_checkpoint", "")).resolve()
    if (
        not policy.is_file()
        or sha256_file(policy) != registry.get("policy_checkpoint_sha256")
        or policy != (ROOT / lock["policy_checkpoint"]["path"]).resolve()
    ):
        raise ValueError("Phase 12 PPO checkpoint changed after promotion")
    if not parseq.is_file() or sha256_file(parseq) != registry.get("parseq_checkpoint_sha256"):
        raise ValueError("Phase 12 PARSeq checkpoint changed after promotion")
    return registry


def expand_allowed_view(
    baseline_features: np.ndarray,
    baseline_predictions: np.ndarray,
    baseline_confidence: np.ndarray,
    allowed: np.ndarray,
    allowed_result: tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None,
    view_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    features = baseline_features.copy()
    predictions = baseline_predictions.copy()
    confidence = baseline_confidence.copy()
    amortized = np.zeros(len(allowed), dtype=np.float64)
    if allowed_result is None:
        timing = {
            "view": view_name,
            "batch_seconds": [],
            "batch_sizes": [],
            "amortized_ms_per_image": amortized.tolist(),
        }
    else:
        selected_features, selected_predictions, selected_confidence, source_timing = allowed_result
        features[allowed] = selected_features
        predictions[allowed] = selected_predictions
        confidence[allowed] = selected_confidence
        amortized[allowed] = np.asarray(source_timing["amortized_ms_per_image"], dtype=np.float64)
        timing = {**source_timing, "amortized_ms_per_image": amortized.tolist()}
    return features, predictions, confidence, timing


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents:
        raise ValueError("Phase 12 runtime outputs must remain inside Phase 12")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Phase 12 runtime output already exists")
    registry_path = Path(args.registry).resolve()
    registry = load_active_registry(registry_path)
    frame = read_runtime_manifest(Path(args.manifest).resolve())
    checkpoint = load_checkpoint(Path(registry["policy_checkpoint"]).resolve())
    action_names = list(checkpoint["action_names"])
    expected_actions = [view.name for view in COMPACT_VIEWS]
    if action_names != expected_actions or registry.get("action_names") != expected_actions:
        raise ValueError("Phase 12 policy actions differ from compact views")

    dummy = np.zeros(len(frame), dtype=np.int64)
    _, guard_allowed = guarded_selection(
        dummy,
        frame.input_transform.to_numpy(dtype=str),
        frame.crop_width.to_numpy(dtype=np.int64),
        frame.crop_height.to_numpy(dtype=np.int64),
    )
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    parseq, parseq_cfg, _ = load_notebook_checkpoint(
        Path(registry["parseq_checkpoint"]).resolve(), args.device_obj, int(registry["refine_iters"])
    )
    synchronize_device(args.device_obj)
    runtime_started = time.perf_counter()
    baseline_features, baseline_predictions, baseline_confidence, baseline_timing = runtime_view_features(
        parseq, parseq_cfg, frame, COMPACT_VIEWS[0], args
    )
    feature_blocks = [baseline_features]
    prediction_blocks = [baseline_predictions]
    confidence_blocks = [baseline_confidence]
    view_timings = [baseline_timing]
    allowed_frame = frame.loc[guard_allowed, ["image_path"]].reset_index(drop=True)
    for view in COMPACT_VIEWS[1:]:
        allowed_result = (
            runtime_view_features(parseq, parseq_cfg, allowed_frame, view, args)
            if len(allowed_frame)
            else None
        )
        features, predictions, confidence, timing = expand_allowed_view(
            baseline_features,
            baseline_predictions,
            baseline_confidence,
            guard_allowed,
            allowed_result,
            view.name,
        )
        feature_blocks.append(features)
        prediction_blocks.append(predictions)
        confidence_blocks.append(confidence)
        view_timings.append(timing)
    synchronize_device(args.device_obj)
    policy_started = time.perf_counter()
    raw = np.stack(feature_blocks, axis=1)
    predictions = np.stack(prediction_blocks, axis=1)
    confidence = np.stack(confidence_blocks, axis=1)
    ppo_first, ppo_selected, ppo_revised = checkpoint_selection(
        checkpoint,
        raw,
        {"predictions": predictions, "normalized_confidence": confidence},
        args.device_obj,
    )
    selected, checked_allowed = guarded_selection(
        ppo_selected,
        frame.input_transform.to_numpy(dtype=str),
        frame.crop_width.to_numpy(dtype=np.int64),
        frame.crop_height.to_numpy(dtype=np.int64),
    )
    if not np.array_equal(guard_allowed, checked_allowed):
        raise RuntimeError("Phase 12 guard changed during runtime")
    synchronize_device(args.device_obj)
    policy_seconds = time.perf_counter() - policy_started
    total_seconds = time.perf_counter() - runtime_started
    rows = np.arange(len(frame))
    costs = np.asarray([view.cost for view in COMPACT_VIEWS], dtype=np.float32)
    result = pd.DataFrame(
        {
            "image_path": frame.image_path,
            "input_transform": frame.input_transform,
            "crop_width": frame.crop_width,
            "crop_height": frame.crop_height,
            "guard_allowed": guard_allowed,
            "baseline_prediction": predictions[:, 0],
            "ppo_first_action": [action_names[index] for index in ppo_first],
            "ppo_final_action": [action_names[index] for index in ppo_selected],
            "final_action": [action_names[index] for index in selected],
            "prediction": predictions[rows, selected],
            "normalized_confidence": confidence[rows, selected],
            "action_cost": costs[selected],
            "guard_rollback": (~guard_allowed) & (ppo_selected != 0),
            "revised": np.asarray(ppo_revised, dtype=bool) & guard_allowed,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "runtime_selections.csv", index=False)
    latency = summarize_runtime_latency(view_timings, policy_seconds, total_seconds, len(result))
    latency.update(
        {
            "guard_short_circuit": True,
            "candidate_views_evaluated_per_allowed_image": len(COMPACT_VIEWS),
            "candidate_views_evaluated_per_disallowed_image": 1,
            "mean_candidate_views_evaluated_per_image": float(
                1 + (len(COMPACT_VIEWS) - 1) * guard_allowed.mean()
            ),
        }
    )
    summary = {
        "samples": int(len(result)),
        "algorithm": registry["algorithm"],
        "policy_registry": str(registry_path),
        "label_free_runtime": True,
        "plate_crop_contract": True,
        "guard": registry["guard"],
        "guard_allowed_rate": float(guard_allowed.mean()),
        "guard_rollback_rate": float(result.guard_rollback.mean()),
        "baseline_rate": float((selected == 0).mean()),
        "revise_rate": float(result.revised.mean()),
        "mean_action_cost": float(costs[selected].mean()),
        "latency": latency,
    }
    (output_dir / "runtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", default=str(HERE / "results" / "active_policy.json"))
    parser.add_argument("--output-dir", default=str(HERE / "results" / "runtime_inference"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


"""Label-free Phase 7 restoration inference, enabled only by an active promotion registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.data import candidate_ocr_features  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.model import CandidateSetActorCritic, RewardTeacher  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.train import policy_selection, teacher_predict  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.action_space import COMPACT_VIEWS, view_metadata  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.build_cache import ViewDataset  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.promote import validate_promotion_summary  # noqa: E402
from rl_restoration.features import parseq_state_features  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import normalize_plate_text  # noqa: E402


def load_active_registry(path: Path) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or registry.get("status") != "active_external_validated":
        raise ValueError("Policy registry is not an externally validated active Phase 7 policy")
    evaluation_path = Path(registry.get("external_evaluation", ""))
    if not evaluation_path.is_file():
        raise FileNotFoundError("The external evaluation supporting this policy is unavailable")
    if hashlib.sha256(evaluation_path.read_bytes()).hexdigest() != registry.get("external_evaluation_sha256"):
        raise ValueError("The external evaluation changed after policy promotion")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    if Path(summary["checkpoint"]).resolve() != Path(registry["policy_checkpoint"]).resolve():
        raise ValueError("Registry policy checkpoint differs from its external evaluation")
    return registry


def read_input_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    if "image_path" not in manifest:
        raise ValueError("Runtime manifest must contain image_path")
    frame = manifest[["image_path"]].dropna().copy()
    frame.image_path = frame.image_path.astype(str).str.strip()
    if frame.empty or (frame.image_path == "").any() or frame.image_path.duplicated().any():
        raise ValueError("Runtime manifest must contain unique, non-empty image_path values")
    missing = [value for value in frame.image_path if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"Runtime manifest references missing image: {missing[0]}")
    return frame.reset_index(drop=True)


@torch.inference_mode()
def runtime_view_features(model, model_cfg, frame: pd.DataFrame, view, args):
    loader = DataLoader(
        ViewDataset(frame, view, tuple(model_cfg.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    metadata = np.asarray(view_metadata(view), dtype=np.float32)
    feature_blocks, prediction_blocks, confidence_blocks = [], [], []
    batch_seconds, batch_sizes = [], []
    iterator = iter(loader)
    while True:
        synchronize_device(args.device_obj)
        started = time.perf_counter()
        try:
            images = next(iterator)
        except StopIteration:
            break
        images = images.to(args.device_obj, non_blocking=True)
        logits = model(images, max_length=model_cfg.max_label_length)
        probabilities = logits.softmax(-1)
        predictions, token_probabilities = model.tokenizer.decode(probabilities)
        predictions = [normalize_plate_text(value) for value in predictions]
        deep = parseq_state_features(model, images, predictions, logits).cpu().numpy()
        feature_blocks.append(np.concatenate((deep, np.repeat(metadata[None], len(images), axis=0)), axis=1))
        prediction_blocks.extend(predictions)
        confidence_blocks.extend(
            math.exp(math.log(max(float(value.prod().item()), 1e-12)) / max(len(prediction) + 1, 1))
            for prediction, value in zip(predictions, token_probabilities)
        )
        synchronize_device(args.device_obj)
        batch_seconds.append(time.perf_counter() - started)
        batch_sizes.append(len(images))
    amortized_ms = [
        1000.0 * seconds / size
        for seconds, size in zip(batch_seconds, batch_sizes)
        for _ in range(size)
    ]
    return (
        np.concatenate(feature_blocks).astype(np.float32),
        np.asarray(prediction_blocks, dtype=str),
        np.asarray(confidence_blocks, dtype=np.float32),
        {
            "view": view.name,
            "batch_seconds": batch_seconds,
            "batch_sizes": batch_sizes,
            "amortized_ms_per_image": amortized_ms,
        },
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_runtime_latency(
    view_timings: list[dict],
    policy_seconds: float,
    total_seconds: float,
    samples: int,
) -> dict:
    if samples <= 0 or not view_timings:
        raise ValueError("Latency summary requires samples and view timings")
    per_view_arrays = [np.asarray(item["amortized_ms_per_image"], dtype=np.float64) for item in view_timings]
    if any(len(values) != samples for values in per_view_arrays):
        raise ValueError("Latency timings do not cover every runtime sample")
    candidate_ms = np.sum(np.stack(per_view_arrays, axis=0), axis=0)
    policy_ms_per_image = 1000.0 * policy_seconds / samples
    end_to_end_batch_amortized = candidate_ms + policy_ms_per_image
    return {
        "contract": "full_compact_view_generation_plus_policy_selection_v1",
        "model_load_and_output_io_excluded": True,
        "candidate_views_evaluated_per_image": len(view_timings),
        "candidate_generation_seconds": float(sum(sum(item["batch_seconds"]) for item in view_timings)),
        "policy_selection_seconds": float(policy_seconds),
        "total_wall_seconds": float(total_seconds),
        "mean_wall_ms_per_image": float(1000.0 * total_seconds / samples),
        "p95_batch_amortized_ms_per_image": float(np.percentile(end_to_end_batch_amortized, 95)),
        "throughput_images_per_second": float(samples / total_seconds),
        "per_view": {
            item["view"]: {
                "seconds": float(sum(item["batch_seconds"])),
                "mean_amortized_ms_per_image": float(np.mean(values)),
                "p95_amortized_ms_per_image": float(np.percentile(values, 95)),
            }
            for item, values in zip(view_timings, per_view_arrays)
        },
    }


def run(args):
    registry_path = Path(args.registry).resolve()
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents:
        raise ValueError("Runtime outputs must remain inside reinforcement_learning/phase_7_compact_multiscale_ppo")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Runtime output directory already exists; choose a new directory")
    registry = load_active_registry(registry_path)
    frame = read_input_manifest(Path(args.manifest).resolve())
    checkpoint = torch.load(Path(registry["policy_checkpoint"]).resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("test_used", True):
        raise ValueError("Active policy checkpoint is not marked test-free")
    action_names = list(checkpoint["action_names"])
    expected_actions = [view.name for view in COMPACT_VIEWS]
    if action_names != expected_actions or registry.get("action_names") != expected_actions:
        raise ValueError("Active policy action registry does not match locked compact views")
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    parseq, parseq_cfg, _ = load_notebook_checkpoint(
        Path(registry["parseq_checkpoint"]).resolve(), args.device_obj, int(registry["refine_iters"])
    )
    feature_blocks, prediction_blocks, confidence_blocks, view_timings = [], [], [], []
    synchronize_device(args.device_obj)
    runtime_started = time.perf_counter()
    for view in COMPACT_VIEWS:
        features, predictions, confidence, timing = runtime_view_features(parseq, parseq_cfg, frame, view, args)
        feature_blocks.append(features)
        prediction_blocks.append(predictions)
        confidence_blocks.append(confidence)
        view_timings.append(timing)
    synchronize_device(args.device_obj)
    policy_started = time.perf_counter()
    raw = np.stack(feature_blocks, axis=1)
    predictions = np.stack(prediction_blocks, axis=1)
    confidence = np.stack(confidence_blocks, axis=1)
    if checkpoint.get("candidate_ocr_strings", False):
        raw = np.concatenate((raw, candidate_ocr_features({"predictions": predictions, "normalized_confidence": confidence})), axis=2)
    candidates = ((raw - checkpoint["candidate_mean"]) / checkpoint["candidate_std"]).astype(np.float32)
    teacher_x = ((raw[:, 0] - checkpoint["teacher_mean"]) / checkpoint["teacher_std"]).astype(np.float32)
    teacher_cfg = checkpoint["teacher_config"]
    teacher = RewardTeacher(teacher_cfg["input_dim"], len(action_names), teacher_cfg["hidden_dim"], teacher_cfg["dropout"]).to(args.device_obj)
    teacher.load_state_dict(checkpoint["teacher_state_dict"])
    prior = teacher_predict(teacher, teacher_x, args.device_obj)
    model_cfg = checkpoint["model_config"]
    policy = CandidateSetActorCritic(
        model_cfg["candidate_dim"], model_cfg["action_count"], model_cfg["hidden_dim"],
        model_cfg["heads"], model_cfg["layers"], model_cfg["dropout"], model_cfg["prior_scale"],
    ).to(args.device_obj)
    policy.load_state_dict(checkpoint["model_state_dict"])
    first, selected, revised = policy_selection(
        policy, torch.from_numpy(candidates).to(args.device_obj), torch.from_numpy(prior).to(args.device_obj),
        checkpoint["first_margin"], checkpoint["revise_margin"], args.device_obj,
        checkpoint["teacher_margin"], checkpoint.get("disagreement_margin"),
        checkpoint.get("final_teacher_gain_margin"),
    )
    synchronize_device(args.device_obj)
    policy_seconds = time.perf_counter() - policy_started
    total_seconds = time.perf_counter() - runtime_started
    rows = np.arange(len(frame))
    costs = np.asarray([view.cost for view in COMPACT_VIEWS], dtype=np.float32)
    result = pd.DataFrame({
        "image_path": frame.image_path,
        "baseline_prediction": predictions[:, 0],
        "first_action": [action_names[index] for index in first],
        "final_action": [action_names[index] for index in selected],
        "prediction": predictions[rows, selected],
        "normalized_confidence": confidence[rows, selected],
        "action_cost": costs[selected],
        "revised": revised,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "runtime_selections.csv", index=False)
    summary = {
        "samples": int(len(result)), "policy_registry": str(registry_path),
        "policy_checkpoint": registry["policy_checkpoint"], "label_free_runtime": True,
        "baseline_rate": float((selected == 0).mean()), "revise_rate": float(revised.mean()),
        "mean_action_cost": float(costs[selected].mean()),
        "latency": summarize_runtime_latency(
            view_timings, policy_seconds, total_seconds, len(result)
        ),
    }
    (output_dir / "runtime_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with image_path; labels are neither read nor required.")
    parser.add_argument("--registry", default=str(HERE / "results/active_policy.json"))
    parser.add_argument("--output-dir", default=str(HERE / "results/runtime_inference"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

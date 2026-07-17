"""Label-free runtime for an externally promoted Phase 8 consensus policy."""

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
    prediction_agreement_selection,
    validate_candidate_lock,
)
from reinforcement_learning.phase_8_consensus_ppo.promote import validate_promotion_summary  # noqa: E402


def load_active_registry(path: Path) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or registry.get("status") != "active_external_validated":
        raise ValueError("Phase 8 registry is not externally validated and active")
    evaluation_path = Path(registry.get("external_evaluation", ""))
    if not evaluation_path.is_file():
        raise FileNotFoundError("The external evaluation supporting Phase 8 is unavailable")
    if hashlib.sha256(evaluation_path.read_bytes()).hexdigest() != registry.get("external_evaluation_sha256"):
        raise ValueError("The Phase 8 promotion evaluation changed after registry creation")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    candidate_lock_path = Path(registry.get("candidate_lock", ""))
    if not candidate_lock_path.is_file() or hashlib.sha256(candidate_lock_path.read_bytes()).hexdigest() != registry.get(
        "candidate_lock_sha256"
    ):
        raise ValueError("The prospective Phase 8 candidate lock changed after promotion")
    if [str(Path(value).resolve()) for value in summary["checkpoints"]] != [
        str(Path(value).resolve()) for value in registry["policy_checkpoints"]
    ]:
        raise ValueError("Registry checkpoints differ from the promotion evaluation")
    checkpoint_paths = [Path(value).resolve() for value in registry["policy_checkpoints"]]
    expected_checkpoint_hashes = registry.get("policy_checkpoint_sha256", [])
    if len(checkpoint_paths) != 2 or len(expected_checkpoint_hashes) != 2:
        raise ValueError("Registry must lock exactly two PPO checkpoint hashes")
    if any(
        not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        for path, expected in zip(checkpoint_paths, expected_checkpoint_hashes)
    ):
        raise ValueError("A promoted PPO checkpoint changed after registry creation")
    validate_candidate_lock(candidate_lock_path, checkpoint_paths)
    parseq_checkpoint = Path(registry.get("parseq_checkpoint", "")).resolve()
    if (
        not parseq_checkpoint.is_file()
        or hashlib.sha256(parseq_checkpoint.read_bytes()).hexdigest()
        != registry.get("parseq_checkpoint_sha256")
    ):
        raise ValueError("The promoted PARSeq checkpoint changed after registry creation")
    return registry


def read_plate_crop_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"image_path", "input_contract"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Runtime manifest is missing required columns: {sorted(missing)}")
    frame = manifest[["image_path", "input_contract"]].dropna().copy()
    frame.image_path = frame.image_path.astype(str).str.strip()
    contracts = frame.input_contract.astype(str).str.strip().str.lower()
    if frame.empty or frame.image_path.duplicated().any() or (frame.image_path == "").any():
        raise ValueError("Runtime image_path values must be unique and non-empty")
    if not (contracts == "plate_crop").all():
        raise ValueError("Phase 8 runtime accepts only input_contract=plate_crop")
    missing_paths = [value for value in frame.image_path if not Path(value).is_file()]
    if missing_paths:
        raise FileNotFoundError(missing_paths[0])
    return frame.reset_index(drop=True)


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents:
        raise ValueError("Phase 8 runtime outputs must remain inside Phase 8")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Runtime output already exists; refusing to overwrite it")
    registry_path = Path(args.registry).resolve()
    registry = load_active_registry(registry_path)
    frame = read_plate_crop_manifest(Path(args.manifest).resolve())
    checkpoint_a, checkpoint_b = [load_checkpoint(Path(value).resolve()) for value in registry["policy_checkpoints"]]
    action_names = list(checkpoint_a["action_names"])
    expected_actions = [view.name for view in COMPACT_VIEWS]
    if action_names != expected_actions or list(checkpoint_b["action_names"]) != expected_actions:
        raise ValueError("Phase 8 policy checkpoints do not match compact views")
    if registry.get("action_names") != expected_actions:
        raise ValueError("Phase 8 registry action names do not match compact views")

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
    label_free_cache = {"predictions": predictions, "normalized_confidence": confidence}
    _, selected_a, _ = checkpoint_selection(checkpoint_a, raw, label_free_cache, args.device_obj)
    _, selected_b, _ = checkpoint_selection(checkpoint_b, raw, label_free_cache, args.device_obj)
    selected, agreed_change = prediction_agreement_selection(predictions, selected_a, selected_b)
    synchronize_device(args.device_obj)
    policy_seconds = time.perf_counter() - policy_started
    total_seconds = time.perf_counter() - runtime_started
    rows = np.arange(len(frame))
    costs = np.asarray([view.cost for view in COMPACT_VIEWS], dtype=np.float32)
    result = pd.DataFrame(
        {
            "image_path": frame.image_path,
            "baseline_prediction": predictions[:, 0],
            "policy_a_action": [action_names[index] for index in selected_a],
            "policy_b_action": [action_names[index] for index in selected_b],
            "final_action": [action_names[index] for index in selected],
            "prediction": predictions[rows, selected],
            "prediction_agreement_change": agreed_change,
            "normalized_confidence": confidence[rows, selected],
            "action_cost": costs[selected],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "runtime_selections.csv", index=False)
    summary = {
        "samples": int(len(result)),
        "algorithm": registry["algorithm"],
        "policy_registry": str(registry_path),
        "label_free_runtime": True,
        "plate_crop_contract": True,
        "baseline_rate": float((selected == 0).mean()),
        "consensus_change_rate": float(agreed_change.mean()),
        "mean_action_cost": float(costs[selected].mean()),
        "latency": summarize_runtime_latency(
            view_timings, policy_seconds, total_seconds, len(result)
        ),
    }
    (output_dir / "runtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", default=str(HERE / "results/active_policy.json"))
    parser.add_argument("--output-dir", default=str(HERE / "results/runtime_inference"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

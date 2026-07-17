"""Label-free runtime for an externally promoted Phase 9 primary PPO."""

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

from preprocessing_best_config.find_best_preprocessing_config import (  # noqa: E402
    load_notebook_checkpoint,
)
from reinforcement_learning.phase_7_compact_multiscale_ppo.action_space import (  # noqa: E402
    COMPACT_VIEWS,
)
from reinforcement_learning.phase_7_compact_multiscale_ppo.runtime import (  # noqa: E402
    runtime_view_features,
    summarize_runtime_latency,
    synchronize_device,
)
from reinforcement_learning.phase_8_consensus_ppo.evaluate import (  # noqa: E402
    checkpoint_selection,
    load_checkpoint,
)
from reinforcement_learning.phase_8_consensus_ppo.runtime import (  # noqa: E402
    read_plate_crop_manifest,
)
from reinforcement_learning.phase_9_primary_ppo.prepare_fresh_holdout import (  # noqa: E402
    validate_candidate_lock,
)
from reinforcement_learning.phase_9_primary_ppo.promote import (  # noqa: E402
    validate_promotion_summary,
    validate_receipt,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_active_registry(path: Path) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or registry.get("status") != "active_external_validated":
        raise ValueError("Phase 9 registry is not externally validated and active")
    evaluation_path = Path(registry.get("external_evaluation", "")).resolve()
    if not evaluation_path.is_file() or sha256_file(evaluation_path) != registry.get("external_evaluation_sha256"):
        raise ValueError("Phase 9 promotion evaluation is missing or changed")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    validate_receipt(summary, evaluation_path)
    lock_path = Path(registry.get("candidate_lock", "")).resolve()
    if not lock_path.is_file() or sha256_file(lock_path) != registry.get("candidate_lock_sha256"):
        raise ValueError("Phase 9 candidate lock changed after promotion")
    lock = validate_candidate_lock(lock_path)
    checkpoint_path = Path(registry.get("policy_checkpoint", "")).resolve()
    parseq_path = Path(registry.get("parseq_checkpoint", "")).resolve()
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != registry.get("policy_checkpoint_sha256")
        or checkpoint_path != (ROOT / lock["policy_checkpoint"]["path"]).resolve()
    ):
        raise ValueError("Phase 9 PPO checkpoint changed after promotion")
    if not parseq_path.is_file() or sha256_file(parseq_path) != registry.get("parseq_checkpoint_sha256"):
        raise ValueError("Phase 9 PARSeq checkpoint changed after promotion")
    return registry


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents:
        raise ValueError("Phase 9 runtime outputs must remain inside Phase 9")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Phase 9 runtime output already exists")
    registry_path = Path(args.registry).resolve()
    registry = load_active_registry(registry_path)
    frame = read_plate_crop_manifest(Path(args.manifest).resolve())
    checkpoint = load_checkpoint(Path(registry["policy_checkpoint"]).resolve())
    action_names = list(checkpoint["action_names"])
    expected_actions = [view.name for view in COMPACT_VIEWS]
    if action_names != expected_actions or registry.get("action_names") != expected_actions:
        raise ValueError("Phase 9 policy action registry differs from compact views")

    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    parseq, parseq_cfg, _ = load_notebook_checkpoint(
        Path(registry["parseq_checkpoint"]).resolve(),
        args.device_obj,
        int(registry["refine_iters"]),
    )
    feature_blocks, prediction_blocks, confidence_blocks, view_timings = [], [], [], []
    synchronize_device(args.device_obj)
    runtime_started = time.perf_counter()
    for view in COMPACT_VIEWS:
        features, predictions, confidence, timing = runtime_view_features(
            parseq, parseq_cfg, frame, view, args
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
    label_free_cache = {"predictions": predictions, "normalized_confidence": confidence}
    first, selected, revised = checkpoint_selection(
        checkpoint, raw, label_free_cache, args.device_obj
    )
    synchronize_device(args.device_obj)
    policy_seconds = time.perf_counter() - policy_started
    total_seconds = time.perf_counter() - runtime_started
    rows = np.arange(len(frame))
    costs = np.asarray([view.cost for view in COMPACT_VIEWS], dtype=np.float32)
    result = pd.DataFrame(
        {
            "image_path": frame.image_path,
            "baseline_prediction": predictions[:, 0],
            "first_action": [action_names[index] for index in first],
            "final_action": [action_names[index] for index in selected],
            "prediction": predictions[rows, selected],
            "normalized_confidence": confidence[rows, selected],
            "action_cost": costs[selected],
            "revised": revised,
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
        "revise_rate": float(np.asarray(revised, dtype=bool).mean()),
        "mean_action_cost": float(costs[selected].mean()),
        "latency": summarize_runtime_latency(
            view_timings, policy_seconds, total_seconds, len(result)
        ),
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

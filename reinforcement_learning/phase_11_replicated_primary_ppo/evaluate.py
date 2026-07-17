"""One-shot evaluation of the prospectively locked Phase 11 seed-728 PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE7 = ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo"
DEFAULT_CACHE = PHASE7 / "results" / "phase11_fresh_external_cache"
DEFAULT_OUTPUT = HERE / "results" / "external_locked_evaluation"
DEFAULT_RECEIPT = HERE / "results" / "fresh_locked_confirmatory_receipt.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.data import (  # noqa: E402
    load_candidate_features,
    load_trajectory_cache,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.paired_statistics import improvement_gate  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.train import evaluate, paired_stats  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.evaluate_external import (  # noqa: E402
    validate_cache_artifacts,
    validate_cache_checkpoint,
    validate_external_cache,
)
from reinforcement_learning.phase_8_consensus_ppo.evaluate import (  # noqa: E402
    attach_external_metadata,
    checkpoint_selection,
    load_checkpoint,
)
from reinforcement_learning.phase_9_primary_ppo.evaluate import descriptive_metrics  # noqa: E402
from reinforcement_learning.phase_9_primary_ppo.prepare_fresh_holdout import sha256_file  # noqa: E402
from reinforcement_learning.phase_11_replicated_primary_ppo.prepare_fresh_holdout import (  # noqa: E402
    validate_lock,
)


def validate_holdout(cache_summary: dict, lock_path: Path) -> dict:
    manifest = cache_summary.get("manifest", {})
    manifest_path = Path(manifest.get("path", "")).resolve()
    audit_path = manifest_path.parent / "finalization_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = audit.get("selection", {})
    audit_manifest = audit.get("manifest", {})
    if (
        audit.get("contract") != "phase11_fresh_locked_confirmatory_manifest_v1"
        or audit.get("inference_run") is not False
        or not audit.get("candidate_locked_before_manifest", False)
        or not audit.get("formal_sample_ready", False)
        or not audit.get("power_target_ready", False)
        or int(selection.get("selected_rows", -1)) != 1500
        or int(selection.get("unique_labels", -1)) != 1500
        or int(selection.get("unique_images", -1)) != 1500
        or selection.get("historical_or_opened_label_overlap") != 0
        or selection.get("phase8_queue_or_phase9_source_overlap") != 0
        or selection.get("historical_or_opened_exact_image_overlap") != 0
        or int(cache_summary.get("samples", -1)) != 1500
    ):
        raise ValueError("Phase 11 holdout audit does not satisfy its locked power/data contract")
    if (
        manifest_path != Path(audit_manifest.get("path", "")).resolve()
        or manifest.get("sha256") != audit_manifest.get("sha256")
        or sha256_file(manifest_path) != manifest.get("sha256")
    ):
        raise ValueError("Phase 11 cache manifest differs from finalization")
    lock_entry = audit.get("candidate_lock", {})
    if (
        Path(lock_entry.get("path", "")).resolve() != lock_path
        or lock_entry.get("sha256") != sha256_file(lock_path)
    ):
        raise ValueError("Phase 11 candidate lock differs from the pre-manifest lock")
    return audit


def claim(
    receipt_path: Path,
    output_dir: Path,
    cache: dict,
    lock_path: Path,
    checkpoint_path: Path,
) -> dict:
    if HERE not in receipt_path.parents:
        raise ValueError("Phase 11 receipt must remain inside Phase 11")
    contract = {
        "evaluation_role": "locked_confirmatory",
        "split": "external_holdout",
        "output_directory": str(output_dir),
        "manifest_path": str(Path(cache["manifest"]["path"]).resolve()),
        "manifest_sha256": cache["manifest"]["sha256"],
        "candidate_lock_path": str(lock_path),
        "candidate_lock_sha256": sha256_file(lock_path),
        "policy_checkpoint_path": str(checkpoint_path),
        "policy_checkpoint_sha256": sha256_file(checkpoint_path),
        "parseq_checkpoint_path": str(Path(cache["checkpoint"]).resolve()),
        "parseq_checkpoint_sha256": cache["checkpoint_sha256"],
        "cache_artifact_sha256": {name: entry["sha256"] for name, entry in cache["artifacts"].items()},
    }
    claim_id = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "started",
        "one_shot": True,
        "inference_may_have_started": True,
        "claim_id": claim_id,
        **contract,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_path.open("x", encoding="utf-8") as destination:
            json.dump(receipt, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise FileExistsError("Phase 11 confirmatory evaluation was already claimed") from error
    return receipt


def complete(receipt_path: Path, started: dict, summary_path: Path, eligible: bool) -> None:
    if json.loads(receipt_path.read_text(encoding="utf-8")) != started:
        raise ValueError("Phase 11 receipt changed during evaluation")
    completed = {
        **started,
        "status": "completed",
        "summary_path": str(summary_path),
        "promotion_eligible": bool(eligible),
    }
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(completed, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, receipt_path)


def run(args: argparse.Namespace) -> dict:
    lock_path = Path(args.candidate_lock).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    receipt_path = Path(args.receipt).resolve()
    if HERE not in output_dir.parents or HERE not in receipt_path.parents:
        raise ValueError("Phase 11 evaluation artifacts must remain inside Phase 11")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite Phase 11 evaluation")
    lock = validate_lock(lock_path)
    checkpoint_path = (ROOT / lock["policy_checkpoint"]["path"]).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    cache_summary = validate_external_cache(cache_dir)
    if not cache_summary.get("power_contract", {}).get("formal_ready", False):
        raise ValueError("Phase 11 cache is below the formal minimum")
    validate_cache_checkpoint(cache_summary, required=True)
    validate_cache_artifacts(cache_dir, cache_summary, required=True)
    validate_holdout(cache_summary, lock_path)
    receipt = claim(receipt_path, output_dir, cache_summary, lock_path, checkpoint_path)

    action_names = list(checkpoint["action_names"])
    cache = load_trajectory_cache(cache_dir, "external_holdout", action_names)
    raw = load_candidate_features(
        cache_dir / "external_holdout_candidate_features.npz", cache["image_paths"], action_names
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    first, selected, revised = checkpoint_selection(checkpoint, raw, cache, device)
    metrics, frame = evaluate(cache, selected, action_names, first, revised)
    baseline_metrics, baseline_frame = evaluate(
        cache, np.zeros(len(selected), dtype=np.int64), action_names
    )
    statistics = paired_stats(frame, baseline_frame, seed=int(checkpoint["seed"]))
    gate = improvement_gate(metrics, statistics)
    frame = attach_external_metadata(frame, Path(cache_summary["manifest"]["path"]).resolve())
    slices = {
        "role": "descriptive_only_not_a_promotion_gate",
        "source": descriptive_metrics(frame, "source"),
        "input_transform": descriptive_metrics(frame, "input_transform"),
        "crop_size_bucket": descriptive_metrics(frame, "crop_size_bucket"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "policy_selections.csv", index=False)
    eligible = bool(gate["passed"])
    summary = {
        "algorithm": lock["algorithm"],
        "status": "eligible" if eligible else "external_holdout_failed_gate",
        "evaluation_role": "locked_confirmatory",
        "split": "external_holdout",
        "samples": int(len(selected)),
        "selection_rule": lock["selection_rule"],
        "candidate_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path), "status": lock["status"]},
        "policy_checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "cache": {"directory": str(cache_dir), "summary": cache_summary},
        "baseline": baseline_metrics,
        "policy": metrics,
        "statistics_vs_baseline": statistics,
        "formal_improvement_gate_vs_baseline": gate,
        "descriptive_slices": slices,
        "promotion_eligible": eligible,
        "external_holdout_evaluated_once": True,
        "audited_legacy_test_loaded": False,
        "test_used": False,
        "confirmatory_receipt": {"path": str(receipt_path), "claim_id": receipt["claim_id"], "one_shot": True},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    complete(receipt_path, receipt, summary_path, eligible)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", default=str(HERE / "prospective_policy.json"))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--device", default="")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


"""Promote Phase 8 only after a fresh locked-confirmatory external gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def validate_promotion_summary(summary: dict) -> None:
    if summary.get("algorithm") != "dual_seed_ppo_prediction_agreement_consensus":
        raise ValueError("Evaluation is not a Phase 8 consensus policy")
    if summary.get("evaluation_role") != "locked_confirmatory":
        raise ValueError("Only a fresh locked-confirmatory evaluation can promote Phase 8")
    if summary.get("split") != "external_holdout":
        raise ValueError("Promotion summary must evaluate split=external_holdout")
    if summary.get("status") != "eligible" or not summary.get("promotion_eligible", False):
        raise ValueError("Phase 8 external evaluation is not promotion-eligible")
    if not summary.get("formal_improvement_gate_vs_baseline", {}).get("passed", False):
        raise ValueError("Phase 8 did not pass every paired improvement gate")
    if not summary.get("external_holdout_evaluated_once", False):
        raise ValueError("Promotion requires a one-shot external evaluation")
    if summary.get("audited_legacy_test_loaded", True) or summary.get("test_used", True):
        raise ValueError("Promotion evaluation must be test-free")
    candidate_lock = summary.get("candidate_lock", {})
    if candidate_lock.get("status") != "prospective_locked_for_fresh_external":
        raise ValueError("Promotion evaluation lacks prospective candidate-lock provenance")
    cache_summary = summary.get("cache", {}).get("summary", {})
    manifest = cache_summary.get("manifest", {})
    if not manifest.get("group_disjoint", False) or manifest.get("input_contract") != "plate_crop":
        raise ValueError("Promotion requires group-disjoint plate-crop inputs")
    if not cache_summary.get("power_contract", {}).get("formal_ready", False):
        raise ValueError("Promotion requires an external cache that passed the formal power contract")
    if not cache_summary.get("checkpoint_sha256"):
        raise ValueError("Promotion requires a cache with locked PARSeq checkpoint SHA-256")
    if set(cache_summary.get("artifacts", {})) != {
        "candidate_features",
        "state_features",
        "action_trajectories",
    }:
        raise ValueError("Promotion requires SHA-locked external cache artifacts")
    receipt = summary.get("confirmatory_receipt", {})
    if not receipt.get("one_shot", False) or not receipt.get("path") or not receipt.get("claim_id"):
        raise ValueError("Promotion requires a one-shot confirmatory evaluation receipt")


def validate_confirmatory_receipt(summary: dict, evaluation_path: Path) -> dict:
    receipt_entry = summary["confirmatory_receipt"]
    receipt_path = Path(receipt_entry["path"]).resolve()
    if HERE not in receipt_path.parents or not receipt_path.is_file():
        raise FileNotFoundError("The Phase 8 confirmatory receipt is unavailable")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or not receipt.get("one_shot", False)
        or receipt.get("claim_id") != receipt_entry["claim_id"]
        or Path(receipt.get("summary_path", "")).resolve() != evaluation_path
    ):
        raise ValueError("Confirmatory receipt does not prove a completed one-shot evaluation")
    manifest = summary["cache"]["summary"]["manifest"]
    if (
        receipt.get("manifest_sha256") != manifest.get("sha256")
        or receipt.get("candidate_lock_sha256") != summary["candidate_lock"]["sha256"]
        or receipt.get("parseq_checkpoint_sha256") != summary["cache"]["summary"].get("checkpoint_sha256")
        or receipt.get("cache_artifact_sha256")
        != {
            name: entry.get("sha256")
            for name, entry in summary["cache"]["summary"].get("artifacts", {}).items()
        }
        or receipt.get("promotion_eligible") != summary.get("promotion_eligible")
    ):
        raise ValueError("Confirmatory receipt provenance differs from the promotion summary")
    return receipt


def run(args: argparse.Namespace) -> dict:
    evaluation_path = Path(args.evaluation).resolve()
    registry_path = Path(args.registry).resolve()
    if HERE not in evaluation_path.parents or HERE not in registry_path.parents:
        raise ValueError("Phase 8 promotion artifacts must remain inside Phase 8")
    if registry_path.exists():
        raise FileExistsError("An active Phase 8 registry already exists; refusing to overwrite it")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    receipt = validate_confirmatory_receipt(summary, evaluation_path)
    candidate_lock_path = Path(summary["candidate_lock"]["path"]).resolve()
    if not candidate_lock_path.is_file() or hashlib.sha256(candidate_lock_path.read_bytes()).hexdigest() != summary[
        "candidate_lock"
    ]["sha256"]:
        raise ValueError("Prospective candidate lock changed after external evaluation")
    checkpoint_paths = [Path(value).resolve() for value in summary["checkpoints"]]
    parseq_checkpoint = Path(summary["cache"]["summary"]["checkpoint"]).resolve()
    if len(checkpoint_paths) != 2 or not all(path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("Both locked PPO checkpoints must be available")
    actual_policy_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint_paths]
    if receipt.get("checkpoint_sha256") != actual_policy_hashes:
        raise ValueError("A PPO checkpoint changed after the one-shot external evaluation")
    if not parseq_checkpoint.is_file():
        raise FileNotFoundError("The locked PARSeq checkpoint is unavailable")
    parseq_hash = hashlib.sha256(parseq_checkpoint.read_bytes()).hexdigest()
    if parseq_hash != summary["cache"]["summary"].get("checkpoint_sha256"):
        raise ValueError("The PARSeq checkpoint changed after external cache construction")
    registry = {
        "schema_version": 1,
        "status": "active_external_validated",
        "algorithm": summary["algorithm"],
        "policy_checkpoints": [str(path) for path in checkpoint_paths],
        "policy_checkpoint_sha256": actual_policy_hashes,
        "parseq_checkpoint": str(parseq_checkpoint),
        "parseq_checkpoint_sha256": parseq_hash,
        "refine_iters": int(summary["cache"]["summary"].get("refine_iters", 2)),
        "action_names": list(summary["cache"]["summary"]["actions"]),
        "selection_rule": summary["selection_rule"],
        "candidate_lock": str(candidate_lock_path),
        "candidate_lock_sha256": summary["candidate_lock"]["sha256"],
        "external_evaluation": str(evaluation_path),
        "external_evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "external_samples": int(summary["samples"]),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default=str(HERE / "results/external_locked_evaluation/summary.json"))
    parser.add_argument("--registry", default=str(HERE / "results/active_policy.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

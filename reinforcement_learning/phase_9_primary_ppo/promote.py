"""Promote Phase 9 only after its fresh one-shot external gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_promotion_summary(summary: dict) -> None:
    if summary.get("algorithm") != "single_primary_candidate_oof_ppo":
        raise ValueError("Evaluation is not the locked Phase 9 policy")
    if summary.get("evaluation_role") != "locked_confirmatory" or summary.get("split") != "external_holdout":
        raise ValueError("Phase 9 promotion requires a locked external confirmation")
    if summary.get("status") != "eligible" or not summary.get("promotion_eligible", False):
        raise ValueError("Phase 9 external evaluation is not promotion-eligible")
    if not summary.get("formal_improvement_gate_vs_baseline", {}).get("passed", False):
        raise ValueError("Phase 9 did not pass every paired baseline gate")
    if not summary.get("external_holdout_evaluated_once", False):
        raise ValueError("Phase 9 promotion requires a one-shot evaluation")
    if summary.get("test_used", True) or summary.get("audited_legacy_test_loaded", True):
        raise ValueError("Phase 9 promotion evidence must remain test-free")
    if summary.get("candidate_lock", {}).get("status") != "prospective_locked_requires_new_external":
        raise ValueError("Phase 9 promotion lacks its prospective lock")
    cache = summary.get("cache", {}).get("summary", {})
    manifest = cache.get("manifest", {})
    if not manifest.get("group_disjoint", False) or manifest.get("input_contract") != "plate_crop":
        raise ValueError("Phase 9 promotion requires group-disjoint plate crops")
    if not cache.get("power_contract", {}).get("formal_ready", False):
        raise ValueError("Phase 9 external cache is underpowered")
    if not cache.get("checkpoint_sha256") or set(cache.get("artifacts", {})) != {
        "candidate_features",
        "state_features",
        "action_trajectories",
    }:
        raise ValueError("Phase 9 cache provenance is incomplete")
    receipt = summary.get("confirmatory_receipt", {})
    if not receipt.get("one_shot", False) or not receipt.get("path") or not receipt.get("claim_id"):
        raise ValueError("Phase 9 promotion lacks a one-shot receipt")


def validate_receipt(summary: dict, evaluation_path: Path) -> dict:
    entry = summary["confirmatory_receipt"]
    receipt_path = Path(entry["path"]).resolve()
    if HERE not in receipt_path.parents or not receipt_path.is_file():
        raise FileNotFoundError("Phase 9 confirmatory receipt is unavailable")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cache = summary["cache"]["summary"]
    policy = summary["policy_checkpoint"]
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or not receipt.get("one_shot", False)
        or receipt.get("claim_id") != entry["claim_id"]
        or Path(receipt.get("summary_path", "")).resolve() != evaluation_path
        or receipt.get("promotion_eligible") != summary.get("promotion_eligible")
        or receipt.get("manifest_sha256") != cache["manifest"]["sha256"]
        or receipt.get("candidate_lock_sha256") != summary["candidate_lock"]["sha256"]
        or receipt.get("policy_checkpoint_sha256") != policy["sha256"]
        or receipt.get("parseq_checkpoint_sha256") != cache["checkpoint_sha256"]
        or receipt.get("cache_artifact_sha256")
        != {name: value["sha256"] for name, value in cache["artifacts"].items()}
    ):
        raise ValueError("Phase 9 receipt provenance differs from its evaluation")
    return receipt


def run(args: argparse.Namespace) -> dict:
    evaluation_path = Path(args.evaluation).resolve()
    registry_path = Path(args.registry).resolve()
    if HERE not in evaluation_path.parents or HERE not in registry_path.parents:
        raise ValueError("Phase 9 promotion artifacts must remain inside Phase 9")
    if registry_path.exists():
        raise FileExistsError("An active Phase 9 registry already exists")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    validate_receipt(summary, evaluation_path)

    candidate_lock = Path(summary["candidate_lock"]["path"]).resolve()
    policy_checkpoint = Path(summary["policy_checkpoint"]["path"]).resolve()
    parseq_checkpoint = Path(summary["cache"]["summary"]["checkpoint"]).resolve()
    for path, expected, name in (
        (candidate_lock, summary["candidate_lock"]["sha256"], "candidate lock"),
        (policy_checkpoint, summary["policy_checkpoint"]["sha256"], "policy checkpoint"),
        (parseq_checkpoint, summary["cache"]["summary"]["checkpoint_sha256"], "PARSeq checkpoint"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Phase 9 {name} changed after evaluation")

    registry = {
        "schema_version": 1,
        "status": "active_external_validated",
        "algorithm": summary["algorithm"],
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_sha256": summary["policy_checkpoint"]["sha256"],
        "parseq_checkpoint": str(parseq_checkpoint),
        "parseq_checkpoint_sha256": summary["cache"]["summary"]["checkpoint_sha256"],
        "refine_iters": int(summary["cache"]["summary"].get("refine_iters", 2)),
        "action_names": list(summary["cache"]["summary"]["actions"]),
        "selection_rule": summary["selection_rule"],
        "candidate_lock": str(candidate_lock),
        "candidate_lock_sha256": summary["candidate_lock"]["sha256"],
        "external_evaluation": str(evaluation_path),
        "external_evaluation_sha256": sha256_file(evaluation_path),
        "external_samples": int(summary["samples"]),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default=str(HERE / "results" / "external_locked_evaluation" / "summary.json"))
    parser.add_argument("--registry", default=str(HERE / "results" / "active_policy.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


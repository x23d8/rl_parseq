"""Promote a Phase 7 policy only after the locked external statistical gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def validate_promotion_summary(summary: dict) -> None:
    gate = summary.get("formal_improvement_gate", {})
    if summary.get("promotion_status") != "eligible" or not gate.get("passed", False):
        raise ValueError("External evaluation did not pass the formal promotion gate")
    if summary.get("evaluation_role", "locked_confirmatory") != "locked_confirmatory":
        raise ValueError("Protocol-repair diagnostics cannot promote a policy")
    if not summary.get("external_holdout_evaluated_once", False):
        raise ValueError("Promotion requires a one-shot external evaluation")
    if summary.get("audited_legacy_test_loaded", True):
        raise ValueError("External evaluation must be marked test-free")
    external_cache = summary.get("external_cache", {})
    if not external_cache.get("checkpoint") or not external_cache.get("actions"):
        raise ValueError("External evaluation lacks runtime reconstruction metadata")


def run(args):
    evaluation_path = Path(args.evaluation).resolve()
    registry_path = Path(args.registry).resolve()
    if HERE not in evaluation_path.parents:
        raise ValueError("External evaluation must be stored inside Phase 7")
    if HERE not in registry_path.parents:
        raise ValueError("Active policy registry must remain inside Phase 7")
    if registry_path.exists():
        raise FileExistsError("An active policy registry already exists; refusing to overwrite promotion")
    summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validate_promotion_summary(summary)
    policy_checkpoint = Path(summary["checkpoint"]).resolve()
    parseq_checkpoint = Path(summary["external_cache"]["checkpoint"]).resolve()
    if not policy_checkpoint.is_file() or not parseq_checkpoint.is_file():
        raise FileNotFoundError("Promotion checkpoint is no longer available")
    registry = {
        "schema_version": 1,
        "status": "active_external_validated",
        "policy_checkpoint": str(policy_checkpoint),
        "parseq_checkpoint": str(parseq_checkpoint),
        "refine_iters": int(summary["external_cache"]["refine_iters"]),
        "action_names": list(summary["external_cache"]["actions"]),
        "external_evaluation": str(evaluation_path),
        "external_evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "external_samples": int(summary["samples"]),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default=str(HERE / "results/external_locked_evaluation/summary.json"))
    parser.add_argument("--registry", default=str(HERE / "results/active_policy.json"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

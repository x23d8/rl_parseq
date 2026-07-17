"""Report the safe next step for the Phase 9 prospective pipeline."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOLDOUT_DIR = HERE / "fresh_external_holdout"
CACHE_DIR = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "results"
    / "phase9_fresh_external_cache"
)
EVALUATION = HERE / "results" / "external_locked_evaluation" / "summary.json"
RECEIPT = HERE / "results" / "fresh_locked_confirmatory_receipt.json"
REGISTRY = HERE / "results" / "active_policy.json"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run() -> dict:
    audit = load(HOLDOUT_DIR / "finalization_audit.json")
    cache = load(CACHE_DIR / "external_holdout_cache_summary.json")
    evaluation = load(EVALUATION)
    receipt = load(RECEIPT)
    registry = load(REGISTRY)
    if audit is None:
        stage = "prepare_fresh_holdout_without_manual_review"
        commands = ["python reinforcement_learning\\run_phase.py phase9-prepare"]
    elif cache is None:
        base = (
            "python reinforcement_learning\\run_phase.py phase7-cache --split external_holdout "
            "--manifest reinforcement_learning\\phase_9_primary_ppo\\fresh_external_holdout\\external_manifest.csv "
            "--output-dir reinforcement_learning\\phase_7_compact_multiscale_ppo\\results\\phase9_fresh_external_cache"
        )
        stage = "preflight_then_build_cache"
        commands = [f"{base} --preflight", base]
    elif receipt is not None and receipt.get("status") == "started":
        stage = "confirmatory_claim_started_requires_audit"
        commands = ["Inspect the one-shot receipt and failure; do not rerun automatically."]
    elif evaluation is None:
        stage = "run_locked_confirmatory_once"
        commands = ["python reinforcement_learning\\run_phase.py phase9-evaluate"]
    elif not evaluation.get("promotion_eligible", False):
        stage = "confirmatory_gate_failed_keep_baseline"
        commands = ["Do not promote or retune on this opened holdout; retain baseline."]
    elif registry is None:
        stage = "promote"
        commands = ["python reinforcement_learning\\run_phase.py phase9-promote"]
    else:
        stage = "active"
        commands = ["Phase 9 is externally validated; use phase9-runtime on plate crops."]
    return {
        "stage": stage,
        "next_commands": commands,
        "artifacts": {
            "candidate_lock": (HERE / "prospective_policy.json").is_file(),
            "fresh_holdout": audit is not None,
            "fresh_holdout_samples": audit.get("selection", {}).get("selected_rows") if audit else None,
            "external_cache": cache is not None,
            "receipt_status": receipt.get("status") if receipt else None,
            "locked_evaluation": evaluation is not None,
            "promotion_eligible": evaluation.get("promotion_eligible") if evaluation else None,
            "active_registry": registry is not None,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))


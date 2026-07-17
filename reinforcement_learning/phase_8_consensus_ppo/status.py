"""Report the safe next action for the Phase 8 fresh-confirmatory pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE7 = ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_8_consensus_ppo.finalize_fresh_holdout import (  # noqa: E402
    HISTORICAL_MANIFEST,
    MEDIA_AUDIT,
    OPENED_EXTERNAL_MANIFEST,
    ORIGINAL_QUEUE,
    QUEUE_AUDIT,
    REVIEWED_QUEUE,
    manifest_labels,
    read_csv,
    validate_media_audit,
)
from reinforcement_learning.phase_8_consensus_ppo.review_contract import (  # noqa: E402
    MINIMUM_FORMAL_SAMPLES,
    assess_review_rows,
)
FINAL_DIR = HERE / "fresh_external_holdout"
CACHE_DIR = PHASE7 / "results" / "phase8_fresh_external_cache"
EVALUATION_DIR = HERE / "results" / "external_locked_evaluation"
RECEIPT_PATH = HERE / "results" / "fresh_locked_confirmatory_receipt.json"
REGISTRY_PATH = HERE / "results" / "active_policy.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_if_present(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def validate_finalization_audit(audit: dict | None) -> dict | None:
    if audit is None:
        return None
    manifest = audit.get("manifest", {})
    manifest_path = Path(manifest.get("path", "")).resolve()
    selection = audit.get("selection", {})
    if (
        audit.get("contract") != "phase8_fresh_locked_confirmatory_manifest_v1"
        or not audit.get("formal_sample_ready", False)
        or int(selection.get("selected_rows", -1)) < MINIMUM_FORMAL_SAMPLES
        or not manifest_path.is_file()
        or manifest.get("sha256") != sha256(manifest_path)
    ):
        raise ValueError("Finalized Phase 8 holdout audit is invalid or stale")
    return audit


def run(verbose: bool = False) -> dict:
    audit = json.loads(QUEUE_AUDIT.read_text(encoding="utf-8"))
    if sha256(ORIGINAL_QUEUE) != audit.get("review_queue", {}).get("sha256"):
        raise ValueError("Phase 8 immutable review queue differs from its audit")
    original, original_fields = read_csv(ORIGINAL_QUEUE)
    reviewed, reviewed_fields = read_csv(REVIEWED_QUEUE)
    if original_fields != reviewed_fields:
        raise ValueError("Phase 8 reviewed queue schema differs from its immutable queue")
    excluded = manifest_labels(HISTORICAL_MANIFEST) | manifest_labels(OPENED_EXTERNAL_MANIFEST)
    _, review, _ = assess_review_rows(
        original, reviewed, excluded, MINIMUM_FORMAL_SAMPLES
    )

    media_preflight = None
    if MEDIA_AUDIT.is_file():
        media_preflight = validate_media_audit(
            MEDIA_AUDIT,
            ORIGINAL_QUEUE.resolve(),
            HISTORICAL_MANIFEST.resolve(),
            OPENED_EXTERNAL_MANIFEST.resolve(),
        )

    final_audit = validate_finalization_audit(
        load_json_if_present(FINAL_DIR / "finalization_audit.json")
    )
    cache_summary = None
    if (CACHE_DIR / "external_holdout_cache_summary.json").is_file():
        from reinforcement_learning.phase_7_compact_multiscale_ppo.evaluate_external import (
            validate_cache_artifacts,
            validate_cache_checkpoint,
            validate_external_cache,
        )

        cache_summary = validate_external_cache(CACHE_DIR)
        validate_cache_checkpoint(cache_summary, required=True)
        validate_cache_artifacts(CACHE_DIR, cache_summary, required=True)
    evaluation_path = EVALUATION_DIR / "summary.json"
    evaluation = load_json_if_present(evaluation_path)
    if evaluation is not None:
        from reinforcement_learning.phase_8_consensus_ppo.promote import validate_confirmatory_receipt

        validate_confirmatory_receipt(evaluation, evaluation_path.resolve())
    receipt = load_json_if_present(RECEIPT_PATH)
    registry = load_json_if_present(REGISTRY_PATH)
    if registry is not None:
        from reinforcement_learning.phase_8_consensus_ppo.runtime import load_active_registry

        registry = load_active_registry(REGISTRY_PATH)

    if media_preflight is None:
        stage = "media_preflight"
        commands = ["python reinforcement_learning\\run_phase.py phase8-preflight"]
    elif not review["formal_ready"]:
        stage = "review"
        commands = ["python reinforcement_learning\\run_phase.py phase8-review"]
    elif final_audit is None:
        stage = "finalize_holdout"
        commands = ["python reinforcement_learning\\run_phase.py phase8-finalize"]
    elif cache_summary is None:
        stage = "preflight_then_build_cache"
        build_command = (
            "python reinforcement_learning\\run_phase.py phase7-cache --split external_holdout "
            "--manifest reinforcement_learning\\phase_8_consensus_ppo\\fresh_external_holdout\\external_manifest.csv "
            "--output-dir reinforcement_learning\\phase_7_compact_multiscale_ppo\\results\\phase8_fresh_external_cache"
        )
        commands = [f"{build_command} --preflight", build_command]
    elif receipt is not None and receipt.get("status") == "started":
        stage = "confirmatory_claim_started_requires_audit"
        commands = ["Inspect the one-shot receipt and evaluation failure; do not rerun automatically."]
    elif evaluation is None:
        stage = "run_locked_confirmatory_once"
        commands = [
            "python reinforcement_learning\\run_phase.py phase8-consensus --split external_holdout "
            "--cache-dir reinforcement_learning\\phase_7_compact_multiscale_ppo\\results\\phase8_fresh_external_cache "
            "--output-dir reinforcement_learning\\phase_8_consensus_ppo\\results\\external_locked_evaluation "
            "--evaluation-role locked_confirmatory"
        ]
    elif not evaluation.get("promotion_eligible", False):
        stage = "confirmatory_gate_failed_keep_baseline"
        commands = ["Do not promote; retain the baseline policy and audit the locked result."]
    elif registry is None:
        stage = "promote"
        commands = ["python reinforcement_learning\\run_phase.py phase8-promote"]
    else:
        stage = "active"
        commands = ["Phase 8 is externally validated and active; use phase8-runtime."]

    review_output = dict(review)
    if not verbose:
        review_output.pop("issue_preview", None)
    return {
        "stage": stage,
        "next_commands": commands,
        "review": review_output,
        "artifacts": {
            "finalized_holdout": final_audit is not None,
            "media_preflight_passed": bool(media_preflight and media_preflight.get("passed")),
            "external_cache": cache_summary is not None,
            "confirmatory_receipt_status": receipt.get("status") if receipt else None,
            "locked_evaluation": evaluation is not None,
            "promotion_eligible": evaluation.get("promotion_eligible") if evaluation else None,
            "active_registry": registry is not None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Include a preview of excluded/invalid rows.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run(verbose=args.verbose), ensure_ascii=False, indent=2))

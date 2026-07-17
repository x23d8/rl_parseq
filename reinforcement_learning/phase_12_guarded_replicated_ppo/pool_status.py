"""Count still-fresh Phase 12 candidates without rendering or model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_9_primary_ppo.prepare_fresh_holdout import (  # noqa: E402
    DEFAULT_HISTORICAL,
    DEFAULT_LABELS_CSV,
    DEFAULT_PHASE7_OPENED,
    DEFAULT_PHASE8_OPENED,
    DEFAULT_PHASE8_QUEUE,
    DEFAULT_SOURCE_ROOT,
    candidate_rows,
    read_manifest_labels,
    read_source_ids,
)


PHASE9 = ROOT / "reinforcement_learning" / "phase_9_primary_ppo" / "fresh_external_holdout" / "external_manifest.csv"
PHASE11 = ROOT / "reinforcement_learning" / "phase_11_replicated_primary_ppo" / "fresh_external_holdout" / "external_manifest.csv"
STABLE_SEED = "phase12-guarded-seed728-fresh-external-v1"
SHEET_ROW_DIRECTION = "toward_larger_row_numbers"
HOLDOUT_DIR = HERE / "fresh_external_holdout"
CACHE_DIR = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "results"
    / "phase12_fresh_external_cache"
)
EVALUATION = HERE / "results" / "external_locked_evaluation" / "summary.json"
RECEIPT = HERE / "results" / "fresh_locked_confirmatory_receipt.json"
REGISTRY = HERE / "results" / "active_policy.json"


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    labels_csv = Path(args.labels_csv).resolve()
    manifests = [Path(value).resolve() for value in args.opened_manifest]
    source_manifests = [Path(value).resolve() for value in args.opened_source_manifest]
    phase8_queue = Path(args.phase8_queue).resolve()
    excluded_labels = set().union(*(read_manifest_labels(path) for path in manifests))
    excluded_sources = read_source_ids(phase8_queue)
    for path in source_manifests:
        excluded_sources |= read_source_ids(path)
    candidates, exclusions = candidate_rows(
        labels_csv,
        source_root,
        first_sheet_row=args.first_sheet_row,
        excluded_labels=excluded_labels,
        excluded_source_ids=excluded_sources,
        stable_selection_seed=STABLE_SEED,
    )
    available = len(candidates)
    source_rows_scanned = available + sum(exclusions.values())
    audit = load_json(HOLDOUT_DIR / "finalization_audit.json")
    cache = load_json(CACHE_DIR / "external_holdout_cache_summary.json")
    evaluation = load_json(EVALUATION)
    receipt = load_json(RECEIPT)
    registry = load_json(REGISTRY)
    if audit is None and available < args.target_samples:
        stage = "waiting_for_fresh_data"
        commands = [
            "Add fresh rows/images, then rerun: python reinforcement_learning\\run_phase.py phase12-status"
        ]
    elif audit is None:
        stage = "ready_to_build_holdout"
        commands = ["python reinforcement_learning\\run_phase.py phase12-prepare"]
    elif cache is None:
        stage = "preflight_then_build_cache"
        base = (
            "python reinforcement_learning\\run_phase.py phase7-cache --split external_holdout "
            "--manifest reinforcement_learning\\phase_12_guarded_replicated_ppo\\fresh_external_holdout\\external_manifest.csv "
            "--output-dir reinforcement_learning\\phase_7_compact_multiscale_ppo\\results\\phase12_fresh_external_cache"
        )
        commands = [f"{base} --preflight", base]
    elif receipt is not None and receipt.get("status") == "started":
        stage = "confirmatory_claim_started_requires_audit"
        commands = ["Inspect the one-shot receipt and failure; do not rerun automatically."]
    elif evaluation is None:
        stage = "run_locked_confirmatory_once"
        commands = ["python reinforcement_learning\\run_phase.py phase12-evaluate"]
    elif not evaluation.get("promotion_eligible", False):
        stage = "confirmatory_gate_failed_keep_baseline"
        commands = ["Do not promote or retune on this opened holdout; retain baseline."]
    elif registry is None:
        stage = "promote"
        commands = ["python reinforcement_learning\\run_phase.py phase12-promote"]
    else:
        stage = "active"
        commands = ["Phase 12 is active; use phase12-runtime with a metadata-complete plate-crop manifest."]
    return {
        "stage": stage,
        "next_commands": commands,
        "pool_scan_inference_run": False,
        "manual_review_required": False,
        "source_contract": {
            "labels_csv": str(labels_csv),
            "first_sheet_row_inclusive": args.first_sheet_row,
            "sheet_row_direction": SHEET_ROW_DIRECTION,
            "selection_expression": f"sheet_row >= {args.first_sheet_row}",
            "label_source": "extracted_character",
            "excluded_review_status": ["rejected"],
            "pending_review_status_allowed": True,
        },
        "source_rows_scanned_from_first_sheet_row": source_rows_scanned,
        "available_unique_label_candidates_upper_bound": available,
        "target_samples": args.target_samples,
        "minimum_formal_samples": args.minimum_samples,
        "shortage_to_target": max(0, args.target_samples - available),
        "shortage_to_formal_minimum": max(0, args.minimum_samples - available),
        "note": "Upper bound is before rendered-image SHA and crop validation.",
        "exclusion_counts": dict(sorted(exclusions.items())),
        "artifacts": {
            "candidate_lock": (HERE / "prospective_policy.json").is_file(),
            "opened_development_audit": (
                HERE / "results" / "opened_development_guard_audit.json"
            ).is_file(),
            "fresh_holdout": audit is not None,
            "external_cache": cache is not None,
            "receipt_status": receipt.get("status") if receipt else None,
            "locked_evaluation": evaluation is not None,
            "promotion_eligible": evaluation.get("promotion_eligible") if evaluation else None,
            "active_registry": registry is not None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--phase8-queue", default=str(DEFAULT_PHASE8_QUEUE))
    parser.add_argument(
        "--opened-manifest",
        action="append",
        default=[str(DEFAULT_HISTORICAL), str(DEFAULT_PHASE7_OPENED), str(DEFAULT_PHASE8_OPENED), str(PHASE9), str(PHASE11)],
    )
    parser.add_argument(
        "--opened-source-manifest",
        action="append",
        default=[str(PHASE9), str(PHASE11)],
    )
    parser.add_argument("--first-sheet-row", type=int, default=733)
    parser.add_argument("--minimum-samples", type=int, default=500)
    parser.add_argument("--target-samples", type=int, default=1500)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

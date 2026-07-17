"""One-shot evaluation of the prospectively locked Phase 9 primary PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE7 = ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo"
DEFAULT_LOCK = HERE / "prospective_policy.json"
DEFAULT_HOLDOUT = HERE / "fresh_external_holdout"
DEFAULT_CACHE = PHASE7 / "results" / "phase9_fresh_external_cache"
DEFAULT_OUTPUT = HERE / "results" / "external_locked_evaluation"
DEFAULT_RECEIPT = HERE / "results" / "fresh_locked_confirmatory_receipt.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.data import (  # noqa: E402
    load_candidate_features,
    load_trajectory_cache,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.paired_statistics import (  # noqa: E402
    improvement_gate,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.train import (  # noqa: E402
    evaluate,
    paired_stats,
)
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
from reinforcement_learning.phase_9_primary_ppo.prepare_fresh_holdout import (  # noqa: E402
    sha256_file,
    validate_candidate_lock,
)


def validate_fresh_holdout(cache_summary: dict, candidate_lock_path: Path) -> dict:
    audit_path = DEFAULT_HOLDOUT / "finalization_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Phase 9 fresh holdout finalization audit is unavailable")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = cache_summary.get("manifest", {})
    manifest_path = Path(manifest.get("path", "")).resolve()
    audit_manifest = audit.get("manifest", {})
    selection = audit.get("selection", {})
    if (
        audit.get("contract") != "phase9_fresh_locked_confirmatory_manifest_v1"
        or audit.get("inference_run") is not False
        or not audit.get("candidate_locked_before_manifest", False)
        or not audit.get("formal_sample_ready", False)
        or int(selection.get("selected_rows", -1)) < 500
        or int(selection.get("selected_rows", -1)) != int(cache_summary.get("samples", -2))
        or int(selection.get("unique_labels", -1)) != int(selection.get("selected_rows", -2))
        or int(selection.get("unique_images", -1)) != int(selection.get("selected_rows", -2))
        or selection.get("historical_or_opened_label_overlap") != 0
        or selection.get("phase8_queue_source_overlap") != 0
        or selection.get("historical_or_opened_exact_image_overlap") != 0
    ):
        raise ValueError("Phase 9 fresh holdout audit does not satisfy the locked contract")
    if (
        manifest_path != Path(audit_manifest.get("path", "")).resolve()
        or manifest.get("sha256") != audit_manifest.get("sha256")
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != manifest.get("sha256")
    ):
        raise ValueError("Phase 9 cache manifest differs from the finalized holdout")
    lock_entry = audit.get("candidate_lock", {})
    if (
        Path(lock_entry.get("path", "")).resolve() != candidate_lock_path
        or lock_entry.get("sha256") != sha256_file(candidate_lock_path)
    ):
        raise ValueError("Phase 9 candidate lock differs from the pre-manifest lock")
    return audit


def claim_evaluation(
    receipt_path: Path,
    output_dir: Path,
    cache_summary: dict,
    candidate_lock_path: Path,
    checkpoint_path: Path,
) -> dict:
    if HERE not in receipt_path.parents:
        raise ValueError("Phase 9 receipt must remain inside Phase 9")
    contract = {
        "evaluation_role": "locked_confirmatory",
        "split": "external_holdout",
        "output_directory": str(output_dir),
        "manifest_path": str(Path(cache_summary["manifest"]["path"]).resolve()),
        "manifest_sha256": cache_summary["manifest"]["sha256"],
        "candidate_lock_path": str(candidate_lock_path),
        "candidate_lock_sha256": sha256_file(candidate_lock_path),
        "policy_checkpoint_path": str(checkpoint_path),
        "policy_checkpoint_sha256": sha256_file(checkpoint_path),
        "parseq_checkpoint_path": str(Path(cache_summary["checkpoint"]).resolve()),
        "parseq_checkpoint_sha256": cache_summary["checkpoint_sha256"],
        "cache_artifact_sha256": {
            name: entry["sha256"] for name, entry in cache_summary["artifacts"].items()
        },
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
        raise FileExistsError(
            f"Phase 9 confirmatory evaluation was already claimed: {receipt_path}"
        ) from error
    return receipt


def complete_evaluation(
    receipt_path: Path,
    started: dict,
    summary_path: Path,
    promotion_eligible: bool,
) -> None:
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    if current != started or current.get("status") != "started":
        raise ValueError("Phase 9 receipt changed during evaluation")
    completed = {
        **started,
        "status": "completed",
        "summary_path": str(summary_path),
        "promotion_eligible": bool(promotion_eligible),
    }
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(completed, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, receipt_path)


def descriptive_metrics(frame: pd.DataFrame, group_column: str) -> dict:
    result = {}
    for value, group in frame.groupby(group_column, dropna=False, sort=True):
        lengths = group.target.astype(str).str.len().clip(lower=1).sum()
        result[str(value)] = {
            "samples": int(len(group)),
            "baseline_exact": float(group.baseline_exact.astype(bool).mean()),
            "policy_exact": float(group.exact.astype(bool).mean()),
            "policy_character_accuracy": float(
                1.0 - group.edit_distance.astype(float).sum() / lengths
            ),
            "fixed": int(group.fixed.astype(bool).sum()),
            "broken": int(group.broken.astype(bool).sum()),
            "net_fixes": int(group.fixed.astype(bool).sum() - group.broken.astype(bool).sum()),
        }
    return result


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    receipt_path = Path(args.receipt).resolve()
    candidate_lock_path = Path(args.candidate_lock).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    if HERE not in output_dir.parents or HERE not in receipt_path.parents:
        raise ValueError("All Phase 9 evaluation artifacts must remain inside Phase 9")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Phase 9 evaluation: {output_dir}")

    lock = validate_candidate_lock(candidate_lock_path)
    checkpoint_path = (ROOT / lock["policy_checkpoint"]["path"]).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    cache_summary = validate_external_cache(cache_dir)
    if not cache_summary.get("power_contract", {}).get("formal_ready", False):
        raise ValueError("Phase 9 confirmatory cache is below the formal sample requirement")
    validate_cache_checkpoint(cache_summary, required=True)
    validate_cache_artifacts(cache_dir, cache_summary, required=True)
    validate_fresh_holdout(cache_summary, candidate_lock_path)

    receipt = claim_evaluation(
        receipt_path,
        output_dir,
        cache_summary,
        candidate_lock_path,
        checkpoint_path,
    )
    action_names = list(checkpoint["action_names"])
    cache = load_trajectory_cache(cache_dir, "external_holdout", action_names)
    raw = load_candidate_features(
        cache_dir / "external_holdout_candidate_features.npz",
        cache["image_paths"],
        action_names,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    first, selected, revised = checkpoint_selection(checkpoint, raw, cache, device)
    policy_metrics, policy_frame = evaluate(cache, selected, action_names, first, revised)
    baseline_metrics, baseline_frame = evaluate(
        cache, np.zeros(len(selected), dtype=np.int64), action_names
    )
    statistics = paired_stats(policy_frame, baseline_frame, seed=int(checkpoint["seed"]))
    gate = improvement_gate(policy_metrics, statistics)

    manifest_path = Path(cache_summary["manifest"]["path"]).resolve()
    policy_frame = attach_external_metadata(policy_frame, manifest_path)
    slices = {
        "role": "descriptive_only_not_a_promotion_gate",
        "source": descriptive_metrics(policy_frame, "source"),
        "input_transform": descriptive_metrics(policy_frame, "input_transform"),
        "crop_size_bucket": descriptive_metrics(policy_frame, "crop_size_bucket"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selections_path = output_dir / "policy_selections.csv"
    policy_frame.to_csv(selections_path, index=False)
    promotion_eligible = bool(gate["passed"])
    summary = {
        "algorithm": lock["algorithm"],
        "status": "eligible" if promotion_eligible else "external_holdout_failed_gate",
        "evaluation_role": "locked_confirmatory",
        "split": "external_holdout",
        "samples": int(len(selected)),
        "selection_rule": lock["selection_rule"],
        "candidate_lock": {
            "path": str(candidate_lock_path),
            "sha256": sha256_file(candidate_lock_path),
            "status": lock["status"],
        },
        "policy_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "cache": {"directory": str(cache_dir), "summary": cache_summary},
        "baseline": baseline_metrics,
        "policy": policy_metrics,
        "statistics_vs_baseline": statistics,
        "formal_improvement_gate_vs_baseline": gate,
        "descriptive_slices": slices,
        "promotion_eligible": promotion_eligible,
        "external_holdout_evaluated_once": True,
        "audited_legacy_test_loaded": False,
        "test_used": False,
        "confirmatory_receipt": {
            "path": str(receipt_path),
            "claim_id": receipt["claim_id"],
            "one_shot": True,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_evaluation(receipt_path, receipt, summary_path, promotion_eligible)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--device", default="")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

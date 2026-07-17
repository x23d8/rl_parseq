"""Evaluate a conservative prediction-agreement ensemble of two Phase 7 PPO policies."""

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
DEFAULT_CANDIDATE_LOCK = HERE / "prospective_policy.json"
DEFAULT_CONFIRMATORY_RECEIPT = HERE / "results" / "fresh_locked_confirmatory_receipt.json"
LOCKED_SELECTION_RULE = (
    "baseline unless both PPO policies predict the same non-baseline string; then use policy B action"
)
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reinforcement_learning.phase_6_candidate_oof_ppo.data import (  # noqa: E402
    candidate_ocr_features,
    load_candidate_features,
    load_trajectory_cache,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.model import (  # noqa: E402
    CandidateSetActorCritic,
    RewardTeacher,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.paired_statistics import improvement_gate  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.train import (  # noqa: E402
    evaluate,
    paired_stats,
    policy_selection,
    teacher_predict,
)
from reinforcement_learning.phase_7_compact_multiscale_ppo.evaluate_external import (  # noqa: E402
    validate_cache_artifacts,
    validate_cache_checkpoint,
    validate_external_cache,
)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_used", True):
        raise ValueError(f"Checkpoint is not marked test-free: {path}")
    if not checkpoint.get("candidate_ocr_strings", False):
        raise ValueError("Phase 8 requires OCR-string candidate observations in both policies")
    return checkpoint


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_confirmatory_evaluation(
    receipt_path: Path,
    output_dir: Path,
    cache_summary: dict,
    candidate_lock_path: Path,
    checkpoint_paths: list[Path],
) -> dict:
    if HERE not in receipt_path.parents:
        raise ValueError("Confirmatory receipt must remain inside Phase 8")
    manifest = cache_summary.get("manifest", {})
    contract = {
        "evaluation_role": "locked_confirmatory",
        "split": "external_holdout",
        "output_directory": str(output_dir),
        "manifest_path": str(Path(manifest.get("path", "")).resolve()),
        "manifest_sha256": manifest.get("sha256"),
        "candidate_lock_path": str(candidate_lock_path),
        "candidate_lock_sha256": file_sha256(candidate_lock_path),
        "checkpoint_sha256": [file_sha256(path) for path in checkpoint_paths],
        "parseq_checkpoint_path": str(Path(cache_summary.get("checkpoint", "")).resolve()),
        "parseq_checkpoint_sha256": cache_summary.get("checkpoint_sha256"),
        "cache_artifact_sha256": {
            name: entry.get("sha256") for name, entry in cache_summary.get("artifacts", {}).items()
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
            f"Locked-confirmatory evaluation was already claimed; inspect receipt {receipt_path}"
        ) from error
    return receipt


def complete_confirmatory_evaluation(
    receipt_path: Path, receipt: dict, summary_path: Path, promotion_eligible: bool
) -> None:
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    if current != receipt or current.get("status") != "started":
        raise ValueError("Confirmatory receipt changed while evaluation was running")
    completed = {
        **receipt,
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


def validate_candidate_lock(lock_path: Path, checkpoint_paths: list[Path]) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("status") != "prospective_locked_for_fresh_external":
        raise ValueError("Phase 8 candidate lock is not prospective and immutable")
    if lock.get("algorithm") != "dual_seed_ppo_prediction_agreement_consensus":
        raise ValueError("Candidate lock algorithm does not match Phase 8")
    if lock.get("selection_rule") != LOCKED_SELECTION_RULE:
        raise ValueError("Candidate lock selection rule does not match Phase 8")
    entries = lock.get("policy_checkpoints", [])
    if len(entries) != 2 or len(checkpoint_paths) != 2:
        raise ValueError("Candidate lock must contain exactly two PPO checkpoints")
    for entry, actual in zip(entries, checkpoint_paths):
        expected = (ROOT / entry["path"]).resolve()
        if actual.resolve() != expected or file_sha256(actual) != entry.get("sha256"):
            raise ValueError("Phase 8 checkpoint path/hash differs from the prospective lock")
    action_entry = lock.get("action_registry", {})
    action_path = (ROOT / action_entry.get("path", "")).resolve()
    if not action_path.is_file() or file_sha256(action_path) != action_entry.get("sha256"):
        raise ValueError("Phase 8 action registry differs from the prospective lock")
    return lock


def checkpoint_selection(
    checkpoint: dict,
    raw_candidate_features: np.ndarray,
    cache: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_names = list(checkpoint["action_names"])
    raw = np.concatenate((raw_candidate_features, candidate_ocr_features(cache)), axis=2)
    candidates = ((raw - checkpoint["candidate_mean"]) / checkpoint["candidate_std"]).astype(np.float32)
    teacher_x = ((raw[:, 0] - checkpoint["teacher_mean"]) / checkpoint["teacher_std"]).astype(np.float32)

    teacher_cfg = checkpoint["teacher_config"]
    teacher = RewardTeacher(
        teacher_cfg["input_dim"], len(action_names), teacher_cfg["hidden_dim"], teacher_cfg["dropout"]
    ).to(device)
    teacher.load_state_dict(checkpoint["teacher_state_dict"])
    prior = teacher_predict(teacher, teacher_x, device)

    model_cfg = checkpoint["model_config"]
    model = CandidateSetActorCritic(
        model_cfg["candidate_dim"],
        model_cfg["action_count"],
        model_cfg["hidden_dim"],
        model_cfg["heads"],
        model_cfg["layers"],
        model_cfg["dropout"],
        model_cfg["prior_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return policy_selection(
        model,
        torch.from_numpy(candidates).to(device),
        torch.from_numpy(prior).to(device),
        checkpoint["first_margin"],
        checkpoint["revise_margin"],
        device,
        checkpoint["teacher_margin"],
        checkpoint.get("disagreement_margin"),
        checkpoint.get("final_teacher_gain_margin"),
    )


def prediction_agreement_selection(
    predictions: np.ndarray, selected_a: np.ndarray, selected_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use policy B only when both PPO policies predict the same non-baseline string."""

    rows = np.arange(len(predictions))
    prediction_a = predictions[rows, selected_a]
    prediction_b = predictions[rows, selected_b]
    baseline = predictions[:, 0]
    agreed_change = (prediction_a == prediction_b) & (prediction_b != baseline)
    selected = np.where(agreed_change, selected_b, 0).astype(np.int64)
    return selected, agreed_change


def attach_external_metadata(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "source", "input_transform", "crop_width", "crop_height"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Fresh external manifest lacks analysis metadata: {sorted(missing)}")
    metadata_columns = ["image_path", "source", "input_transform", "crop_width", "crop_height"]
    metadata = manifest[metadata_columns].copy()
    metadata.image_path = metadata.image_path.astype(str)
    if metadata.image_path.duplicated().any():
        raise ValueError("External analysis metadata contains duplicate image_path")
    metadata["crop_width"] = pd.to_numeric(metadata.crop_width, errors="raise").astype(int)
    metadata["crop_height"] = pd.to_numeric(metadata.crop_height, errors="raise").astype(int)
    minimum_side = metadata[["crop_width", "crop_height"]].min(axis=1)
    metadata["crop_size_bucket"] = pd.cut(
        minimum_side,
        bins=[-np.inf, 31, 63, 127, np.inf],
        labels=["min_side_lt32", "min_side_32_63", "min_side_64_127", "min_side_ge128"],
    ).astype(str)
    merged = frame.merge(metadata, on="image_path", how="left", validate="one_to_one")
    if merged[list(required - {"image_path"})].isna().any().any():
        raise ValueError("External selections could not be joined to every manifest row")
    return merged


def descriptive_slice_metrics(frame: pd.DataFrame, group_column: str) -> dict:
    result = {}
    for value, group in frame.groupby(group_column, dropna=False, sort=True):
        lengths = group.target.astype(str).str.len().clip(lower=1).sum()
        result[str(value)] = {
            "samples": int(len(group)),
            "baseline_exact": float(group.baseline_exact.astype(bool).mean()),
            "consensus_exact": float(group.exact.astype(bool).mean()),
            "consensus_character_accuracy": float(1.0 - group.edit_distance.astype(float).sum() / lengths),
            "fixed": int(group.fixed.astype(bool).sum()),
            "broken": int(group.broken.astype(bool).sum()),
            "net_fixes": int(group.fixed.astype(bool).sum() - group.broken.astype(bool).sum()),
        }
    return result


def validate_locked_external_metadata(cache_summary: dict) -> None:
    manifest_path = Path(cache_summary.get("manifest", {}).get("path", "")).resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "source", "input_transform", "crop_width", "crop_height"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Fresh locked-confirmatory manifest lacks metadata: {sorted(missing)}")
    if len(manifest) != int(cache_summary.get("samples", -1)) or manifest.image_path.astype(str).duplicated().any():
        raise ValueError("Fresh locked-confirmatory metadata row count/paths differ from cache summary")
    width = pd.to_numeric(manifest.crop_width, errors="raise")
    height = pd.to_numeric(manifest.crop_height, errors="raise")
    if (width <= 0).any() or (height <= 0).any():
        raise ValueError("Fresh locked-confirmatory manifest contains invalid crop dimensions")


def read_cache_summary(cache_dir: Path, split: str) -> dict:
    if split == "external_holdout":
        return validate_external_cache(cache_dir)
    summary_path = cache_dir / f"{split}_cache_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("split") != split or summary.get("test_loaded", True):
        raise ValueError(f"Invalid test-free {split} cache summary")
    return summary


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents and output_dir != HERE:
        raise ValueError("All Phase 8 artifacts must remain inside reinforcement_learning/phase_8_consensus_ppo")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite locked Phase 8 output: {output_dir}")

    cache_dir = Path(args.cache_dir).resolve()
    cache_summary = read_cache_summary(cache_dir, args.split)
    if (
        args.split == "external_holdout"
        and args.evaluation_role == "locked_confirmatory"
        and not cache_summary.get("power_contract", {}).get("formal_ready", False)
    ):
        raise ValueError("Locked-confirmatory Phase 8 evaluation requires a cache that passed the formal power contract")
    if args.split == "external_holdout" and args.evaluation_role == "locked_confirmatory":
        validate_cache_checkpoint(cache_summary, required=True)
        validate_cache_artifacts(cache_dir, cache_summary, required=True)
        validate_locked_external_metadata(cache_summary)
    checkpoint_a_path = Path(args.checkpoint_a).resolve()
    checkpoint_b_path = Path(args.checkpoint_b).resolve()
    candidate_lock_path = Path(args.candidate_lock).resolve()
    candidate_lock = validate_candidate_lock(candidate_lock_path, [checkpoint_a_path, checkpoint_b_path])
    checkpoint_a = load_checkpoint(checkpoint_a_path)
    checkpoint_b = load_checkpoint(checkpoint_b_path)
    action_names = list(checkpoint_a["action_names"])
    if list(checkpoint_b["action_names"]) != action_names:
        raise ValueError("Consensus checkpoints use different action registries")

    confirmatory_receipt = None
    confirmatory_receipt_path = None
    if args.split == "external_holdout" and args.evaluation_role == "locked_confirmatory":
        confirmatory_receipt_path = Path(args.confirmatory_receipt).resolve()
        confirmatory_receipt = claim_confirmatory_evaluation(
            confirmatory_receipt_path,
            output_dir,
            cache_summary,
            candidate_lock_path,
            [checkpoint_a_path, checkpoint_b_path],
        )

    cache = load_trajectory_cache(cache_dir, args.split, action_names)
    raw = load_candidate_features(
        cache_dir / f"{args.split}_candidate_features.npz", cache["image_paths"], action_names
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    first_a, selected_a, revised_a = checkpoint_selection(checkpoint_a, raw, cache, device)
    first_b, selected_b, revised_b = checkpoint_selection(checkpoint_b, raw, cache, device)
    consensus, agreed_change = prediction_agreement_selection(cache["predictions"], selected_a, selected_b)

    metrics_a, frame_a = evaluate(cache, selected_a, action_names, first_a, revised_a)
    metrics_b, frame_b = evaluate(cache, selected_b, action_names, first_b, revised_b)
    consensus_metrics, consensus_frame = evaluate(cache, consensus, action_names)
    baseline_metrics, baseline_frame = evaluate(cache, np.zeros(len(consensus), dtype=np.int64), action_names)
    statistics = paired_stats(consensus_frame, baseline_frame, seed=int(checkpoint_b["seed"]))
    gate = improvement_gate(consensus_metrics, statistics)

    output_dir.mkdir(parents=True, exist_ok=True)
    consensus_frame["policy_a_action"] = frame_a.final_action
    consensus_frame["policy_b_action"] = frame_b.final_action
    consensus_frame["policy_a_prediction"] = frame_a.prediction
    consensus_frame["policy_b_prediction"] = frame_b.prediction
    consensus_frame["prediction_agreement_change"] = agreed_change
    descriptive_slices = None
    if args.split == "external_holdout":
        manifest_path = Path(cache_summary["manifest"]["path"]).resolve()
        analysis_columns = {"image_path", "source", "input_transform", "crop_width", "crop_height"}
        manifest_columns = set(pd.read_csv(manifest_path, nrows=0).columns)
        if analysis_columns.issubset(manifest_columns):
            consensus_frame = attach_external_metadata(consensus_frame, manifest_path)
            descriptive_slices = {
                "role": "descriptive_only_not_a_promotion_gate",
                "source": descriptive_slice_metrics(consensus_frame, "source"),
                "input_transform": descriptive_slice_metrics(consensus_frame, "input_transform"),
                "crop_size_bucket": descriptive_slice_metrics(consensus_frame, "crop_size_bucket"),
            }
        elif args.evaluation_role == "locked_confirmatory":
            raise ValueError("Fresh locked-confirmatory manifest lacks required descriptive metadata")
    consensus_frame.to_csv(output_dir / "consensus_selections.csv", index=False)

    role = "development_validation" if args.split == "val" else args.evaluation_role
    promotion_eligible = bool(
        args.split == "external_holdout" and role == "locked_confirmatory" and gate["passed"]
    )
    if promotion_eligible:
        status = "eligible"
    elif args.split == "external_holdout" and role == "locked_confirmatory":
        status = "external_holdout_failed_gate"
    elif args.split == "external_holdout":
        status = "protocol_repair_diagnostic_not_promotable"
    else:
        status = "prospective_candidate_not_promoted"
    summary = {
        "algorithm": "dual_seed_ppo_prediction_agreement_consensus",
        "status": status,
        "evaluation_role": role,
        "split": args.split,
        "samples": int(len(consensus)),
        "cache": {
            "directory": str(cache_dir),
            "summary": cache_summary,
        },
        "checkpoints": [str(checkpoint_a_path), str(checkpoint_b_path)],
        "candidate_lock": {
            "path": str(candidate_lock_path),
            "sha256": file_sha256(candidate_lock_path),
            "status": candidate_lock["status"],
        },
        "selection_rule": LOCKED_SELECTION_RULE,
        "policy_a": metrics_a,
        "policy_b": metrics_b,
        "baseline": baseline_metrics,
        "consensus": consensus_metrics,
        "consensus_change_rate": float(agreed_change.mean()),
        "statistics_vs_baseline": statistics,
        "formal_improvement_gate_vs_baseline": gate,
        "descriptive_slices": descriptive_slices,
        "promotion_eligible": promotion_eligible,
        "external_holdout_evaluated_once": args.split == "external_holdout",
        "audited_legacy_test_loaded": False,
        "test_used": False,
    }
    if confirmatory_receipt is not None and confirmatory_receipt_path is not None:
        summary["confirmatory_receipt"] = {
            "path": str(confirmatory_receipt_path),
            "claim_id": confirmatory_receipt["claim_id"],
            "one_shot": True,
        }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if confirmatory_receipt is not None and confirmatory_receipt_path is not None:
        complete_confirmatory_evaluation(
            confirmatory_receipt_path,
            confirmatory_receipt,
            summary_path,
            promotion_eligible,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "external_holdout"), default="val")
    parser.add_argument("--cache-dir", default=str(PHASE7 / "results/cache"))
    parser.add_argument(
        "--checkpoint-a",
        default=str(PHASE7 / "results/run_ocr_guard_seed_727/best_candidate_oof_ppo.pt"),
    )
    parser.add_argument(
        "--checkpoint-b",
        default=str(PHASE7 / "results/confirmatory_seed_728/best_candidate_oof_ppo.pt"),
    )
    parser.add_argument("--output-dir", default=str(HERE / "results/validation"))
    parser.add_argument(
        "--evaluation-role",
        choices=("protocol_repair_diagnostic", "locked_confirmatory"),
        default="protocol_repair_diagnostic",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--confirmatory-receipt", default=str(DEFAULT_CONFIRMATORY_RECEIPT))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

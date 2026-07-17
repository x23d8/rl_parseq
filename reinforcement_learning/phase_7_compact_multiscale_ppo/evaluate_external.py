"""One-shot evaluation of a locked Phase 7 policy on a new external holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    select_teacher,
    teacher_predict,
)


def validate_external_cache(cache_dir: Path) -> dict:
    """Reject a cache unless it carries the one-shot external-holdout contract."""

    summary_path = cache_dir / "external_holdout_cache_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Missing external_holdout_cache_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("split") != "external_holdout":
        raise ValueError("Cache summary is not for external_holdout")
    if summary.get("test_loaded", True):
        raise ValueError("External cache is not marked test-free")
    manifest = summary.get("manifest", {})
    manifest_path = Path(manifest.get("path", ""))
    manifest_sha256 = str(manifest.get("sha256", ""))
    if not manifest.get("external_contract") or len(manifest_sha256) != 64:
        raise ValueError("External cache lacks locked-manifest provenance")
    if not manifest.get("group_disjoint", False):
        raise ValueError("External cache is not certified group-disjoint from historical data")
    if manifest.get("input_contract") != "plate_crop":
        raise ValueError("External cache did not enforce plate-crop inputs")
    group_audit = summary.get("group_audit", {})
    if group_audit.get("group_overlap") != 0 or not group_audit.get(
        "historical_labels_used_for_exclusion_only", False
    ):
        raise ValueError("External cache lacks a valid historical group-leakage audit")
    try:
        int(manifest_sha256, 16)
    except ValueError as error:
        raise ValueError("External cache has an invalid manifest digest") from error
    if not manifest_path.is_file():
        raise FileNotFoundError("Locked external manifest is no longer available")
    actual_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_sha256 != manifest_sha256:
        raise ValueError("External manifest changed after cache construction")
    if int(summary.get("samples", -1)) <= 0:
        raise ValueError("External cache has no samples")
    return summary


def validate_cache_checkpoint(summary: dict, required: bool = False) -> Path | None:
    checkpoint_value = str(summary.get("checkpoint", "")).strip()
    checkpoint_sha256 = str(summary.get("checkpoint_sha256", "")).strip()
    if not checkpoint_value or not checkpoint_sha256:
        if required:
            raise ValueError("External cache does not lock the PARSeq checkpoint SHA-256")
        return None
    try:
        int(checkpoint_sha256, 16)
    except ValueError as error:
        raise ValueError("External cache has an invalid PARSeq checkpoint digest") from error
    if len(checkpoint_sha256) != 64:
        raise ValueError("External cache has an invalid PARSeq checkpoint digest")
    checkpoint_path = Path(checkpoint_value).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("PARSeq checkpoint used to build the external cache is unavailable")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_sha256:
        raise ValueError("PARSeq checkpoint changed after external cache construction")
    return checkpoint_path


def validate_cache_artifacts(cache_dir: Path, summary: dict, required: bool = False) -> dict:
    expected_names = {
        "candidate_features": "external_holdout_candidate_features.npz",
        "state_features": "external_holdout_state_features.npz",
        "action_trajectories": "external_holdout_action_trajectories.csv",
    }
    artifacts = summary.get("artifacts", {})
    if not artifacts and not required:
        return {}
    if set(artifacts) != set(expected_names):
        raise ValueError("External cache does not lock every required cache artifact")
    cache_dir = cache_dir.resolve()
    for name, filename in expected_names.items():
        entry = artifacts[name]
        path = Path(entry.get("path", "")).resolve()
        digest = str(entry.get("sha256", ""))
        if path != cache_dir / filename or len(digest) != 64:
            raise ValueError(f"External cache artifact provenance is invalid for {name}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"External cache artifact digest is invalid for {name}") from error
        if not path.is_file():
            raise FileNotFoundError(f"External cache artifact is unavailable: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"External cache artifact changed after construction: {name}")
    return artifacts


def run(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents and output_dir != HERE:
        raise ValueError("External evaluation artifacts must remain inside reinforcement_learning/phase_7_compact_multiscale_ppo")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("External evaluation output already exists; refusing to overwrite the audit")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_used", True):
        raise ValueError("Checkpoint is not marked as locked without test use")
    cache_summary = validate_external_cache(cache_dir)
    power_contract = cache_summary.get("power_contract", {})
    if args.evaluation_role == "locked_confirmatory" and not power_contract.get("formal_ready", False):
        raise ValueError("Locked-confirmatory evaluation requires a cache that passed the formal power contract")
    if args.evaluation_role == "locked_confirmatory":
        validate_cache_checkpoint(cache_summary, required=True)
        validate_cache_artifacts(cache_dir, cache_summary, required=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    action_names = checkpoint["action_names"]
    cache = load_trajectory_cache(cache_dir, "external_holdout", action_names)
    if len(cache["image_paths"]) != int(cache_summary["samples"]):
        raise ValueError("External cache samples do not match its provenance summary")
    raw = load_candidate_features(
        cache_dir / "external_holdout_candidate_features.npz", cache["image_paths"], action_names
    )
    if checkpoint.get("candidate_ocr_strings", False):
        raw = np.concatenate((raw, candidate_ocr_features(cache)), axis=2)
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
    candidate_tensor = torch.from_numpy(candidates).to(device)
    prior_tensor = torch.from_numpy(prior).to(device)
    first, selected, revised = policy_selection(
        model,
        candidate_tensor,
        prior_tensor,
        checkpoint["first_margin"],
        checkpoint["revise_margin"],
        device,
        checkpoint["teacher_margin"],
        checkpoint.get("disagreement_margin"),
        checkpoint.get("final_teacher_gain_margin"),
    )
    policy_metrics, policy_frame = evaluate(cache, selected, action_names, first, revised)
    teacher_selected = select_teacher(cache, prior, checkpoint["teacher_margin"])
    teacher_metrics, teacher_frame = evaluate(cache, teacher_selected, action_names)
    statistics = paired_stats(policy_frame, teacher_frame, checkpoint["seed"])
    gate = improvement_gate(policy_metrics, statistics)
    evaluation_role = args.evaluation_role
    if evaluation_role == "locked_confirmatory":
        promotion_status = "eligible" if gate["passed"] else "external_holdout_failed_gate"
    else:
        promotion_status = "protocol_repair_diagnostic_not_promotable"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_frame.to_csv(output_dir / "external_policy_selections.csv", index=False)
    teacher_frame.to_csv(output_dir / "external_teacher_selections.csv", index=False)
    summary = {
        "algorithm": checkpoint["algorithm"],
        "checkpoint": str(checkpoint_path),
        "samples": len(cache["image_paths"]),
        "external_cache_manifest": cache_summary["manifest"],
        "external_cache": {
            "directory": str(cache_dir),
            "checkpoint": cache_summary["checkpoint"],
            "refine_iters": int(cache_summary.get("refine_iters", 2)),
            "actions": cache_summary["actions"],
        },
        "policy": policy_metrics,
        "teacher": teacher_metrics,
        "statistics_vs_teacher": statistics,
        "formal_improvement_gate": gate,
        "promotion_status": promotion_status,
        "evaluation_role": evaluation_role,
        "external_holdout_evaluated_once": True,
        "audited_legacy_test_loaded": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(HERE / "results/run_seed_725/best_candidate_oof_ppo.pt"),
        help="Locked primary Phase 7 checkpoint. Do not choose this after viewing external results.",
    )
    parser.add_argument("--cache-dir", default=str(HERE / "results/external_cache"))
    parser.add_argument("--output-dir", default=str(HERE / "results/external_locked_evaluation"))
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--evaluation-role",
        choices=("locked_confirmatory", "protocol_repair_diagnostic"),
        default="locked_confirmatory",
        help="Protocol-repair diagnostics can never become promotion-eligible.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

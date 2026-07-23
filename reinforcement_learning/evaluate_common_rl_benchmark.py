"""Compare frozen RL/non-RL methods on one shared blurred val/test dataset.

This is a diagnostic benchmark, not a promotion evaluation.  It deliberately
reuses already opened synthetic PixelRL validation/test splits and never tunes
policy thresholds.  All OCR predictions must come from one fixed PARSeq
checkpoint and the script verifies sample/target alignment before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_restoration.actions import DEFAULT_ACTIONS  # noqa: E402
from rl_restoration.policy import RewardRouter  # noqa: E402
from rl_restoration.ppo_policy import RestorationActorCritic  # noqa: E402
from rl_restoration.sequential_env import OfflineSequentialRestorationEnv  # noqa: E402
from rl_restoration.train_ppo import evaluate_policy  # noqa: E402
from rl_restoration.train_router import (  # noqa: E402
    evaluate_selection,
    load_cache,
    predict_rewards,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.data import (  # noqa: E402
    candidate_ocr_features,
    load_candidate_features,
    load_trajectory_cache,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.model import (  # noqa: E402
    CandidateSetActorCritic,
    RewardTeacher,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.train import (  # noqa: E402
    evaluate as evaluate_candidate_policy,
    policy_selection,
    teacher_predict,
)
from train_no_refinement.parseq_official_anpr_pipeline import (  # noqa: E402
    edit_distance,
    normalize_plate_text,
)


METHOD_ORDER = (
    "raw_parseq",
    "classical_unsharp",
    "train_baseline",
    "pixelrl_a2c",
    "contextual_bandit_phase4",
    "ppo_phase5",
    "candidate_ppo_phase6",
    "compact_ppo_phase7",
)

METHOD_LABELS = {
    "raw_parseq": "Raw blurred + fixed PARSeq",
    "classical_unsharp": "Classical unsharp + fixed PARSeq",
    "train_baseline": "Train preprocessing + fixed PARSeq",
    "pixelrl_a2c": "PixelRL (A2C)",
    "contextual_bandit_phase4": "Contextual Bandit (Phase 4)",
    "ppo_phase5": "PPO Phase 5 (MLP)",
    "candidate_ppo_phase6": "Candidate-aware PPO (Phase 6)",
    "compact_ppo_phase7": "Compact PPO (Phase 7)",
}

PARADIGMS = {
    "raw_parseq": "No RL",
    "classical_unsharp": "No RL",
    "train_baseline": "No RL",
    "pixelrl_a2c": "A2C / PixelRL",
    "contextual_bandit_phase4": "Contextual bandit",
    "ppo_phase5": "PPO / actor-critic",
    "candidate_ppo_phase6": "PPO / Transformer",
    "compact_ppo_phase7": "PPO / Transformer",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_manifests(dataset_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in ("val", "test"):
        source = pd.read_csv(dataset_dir / f"{split}.csv")
        frame = pd.DataFrame(
            {
                "image_path": source.blurred_path.map(
                    lambda value: str((dataset_dir / str(value)).resolve())
                ),
                "label": source.label.astype(str).map(normalize_plate_text),
                # Cache builders are run in validation/diagnostic mode for both
                # opened splits; the true semantic split remains in source_split.
                "split": "val",
                "source_split": split,
                "blur_kind": source.blur_kind.astype(str),
                "source_path": source.source_path.astype(str),
            }
        )
        if frame.image_path.duplicated().any() or frame.label.eq("").any():
            raise ValueError(f"Invalid {split} benchmark manifest")
        missing = [path for path in frame.image_path if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{split} has {len(missing)} missing blurred images")
        target = output_dir / f"{split}_manifest.csv"
        frame.to_csv(target, index=False)
        result[split] = {
            "path": str(target.resolve()),
            "sha256": sha256(target),
            "samples": int(len(frame)),
        }
    return result


def load_router_predictions(cache: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_names = [action.name for action in DEFAULT_ACTIONS]
    if checkpoint["action_names"] != action_names:
        raise ValueError("Phase 4 action registry mismatch")
    model = RewardRouter(
        checkpoint["input_dim"], len(action_names), checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    features = (cache["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    rewards = predict_rewards(model, features.astype(np.float32), device)
    metrics, frame, _ = evaluate_selection(
        cache, rewards, action_names, float(checkpoint["selection_margin"])
    )
    return metrics, frame


def load_ppo5_predictions(
    cache: dict, checkpoint_path: Path, router_path: Path, device: torch.device
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_names = [action.name for action in DEFAULT_ACTIONS]
    if checkpoint["action_names"] != action_names:
        raise ValueError("Phase 5 action registry mismatch")
    standardized = (cache["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]

    router_checkpoint = torch.load(router_path, map_location="cpu", weights_only=False)
    teacher = RewardRouter(
        router_checkpoint["input_dim"],
        len(action_names),
        router_checkpoint["hidden_dim"],
        router_checkpoint["dropout"],
    ).to(device)
    teacher.load_state_dict(router_checkpoint["model_state_dict"])
    teacher_x = (cache["features"] - router_checkpoint["feature_mean"]) / router_checkpoint["feature_std"]
    prior = predict_rewards(teacher, teacher_x.astype(np.float32), device)
    features = np.concatenate((standardized, prior), axis=1).astype(np.float32)
    env = OfflineSequentialRestorationEnv(
        cache,
        features,
        device,
        checkpoint["revisit_cost"],
        checkpoint.get("candidate_summary", False),
    )
    model = RestorationActorCritic(
        checkpoint["input_dim"],
        len(action_names),
        checkpoint["hidden_dim"],
        checkpoint["dropout"],
        checkpoint["prior_offset"],
        checkpoint["prior_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_policy(
        model,
        env,
        cache,
        action_names,
        checkpoint["first_margin"],
        checkpoint["revise_margin"],
    )


def load_candidate_predictions(
    cache_dir: Path, checkpoint_path: Path, device: torch.device
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_names = checkpoint["action_names"]
    cache = load_trajectory_cache(cache_dir, "val", action_names)
    raw = load_candidate_features(
        cache_dir / "val_candidate_features.npz", cache["image_paths"], action_names
    )
    if checkpoint.get("candidate_ocr_strings", False):
        raw = np.concatenate((raw, candidate_ocr_features(cache)), axis=2)
    if raw.shape[-1] != int(checkpoint["model_config"]["candidate_dim"]):
        raise ValueError(
            f"Candidate dimension mismatch: cache={raw.shape[-1]}, "
            f"checkpoint={checkpoint['model_config']['candidate_dim']}"
        )
    candidates = ((raw - checkpoint["candidate_mean"]) / checkpoint["candidate_std"]).astype(
        np.float32
    )
    teacher_x = ((raw[:, 0] - checkpoint["teacher_mean"]) / checkpoint["teacher_std"]).astype(
        np.float32
    )
    teacher_cfg = checkpoint["teacher_config"]
    teacher = RewardTeacher(
        teacher_cfg["input_dim"],
        len(action_names),
        teacher_cfg["hidden_dim"],
        teacher_cfg["dropout"],
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
    metrics, frame = evaluate_candidate_policy(
        cache, selected, action_names, first=first, revised=revised
    )
    return cache, metrics, frame


def score_predictions(targets: np.ndarray, predictions: np.ndarray, baseline: np.ndarray) -> dict:
    targets = np.asarray([normalize_plate_text(value) for value in targets], dtype=str)
    predictions = np.asarray([normalize_plate_text(value) for value in predictions], dtype=str)
    baseline = np.asarray([normalize_plate_text(value) for value in baseline], dtype=str)
    exact = predictions == targets
    baseline_exact = baseline == targets
    edits = np.asarray([edit_distance(pred, target) for pred, target in zip(predictions, targets)])
    total_chars = sum(max(len(value), 1) for value in targets)
    return {
        "samples": int(len(targets)),
        "exact_acc": float(exact.mean()),
        "char_acc": float(1.0 - edits.sum() / total_chars),
        "cer": float(edits.sum() / total_chars),
        "fixed_vs_raw": int(((~baseline_exact) & exact).sum()),
        "broken_vs_raw": int((baseline_exact & (~exact)).sum()),
        "net_fixes_vs_raw": int(((~baseline_exact) & exact).sum() - (baseline_exact & (~exact)).sum()),
    }


def assert_aligned(reference_paths, reference_targets, paths, targets, name: str) -> None:
    if not np.array_equal(np.asarray(reference_paths, dtype=str), np.asarray(paths, dtype=str)):
        raise ValueError(f"{name} image rows are not aligned with the common manifest")
    normalized = np.asarray([normalize_plate_text(value) for value in targets], dtype=str)
    if not np.array_equal(np.asarray(reference_targets, dtype=str), normalized):
        raise ValueError(f"{name} targets are not aligned with the common manifest")


def markdown_report(summary: dict, table: pd.DataFrame) -> str:
    lines = [
        "# Common val/test RL benchmark",
        "",
        "Diagnostic comparison on the same synthetic blurred validation and test splits. "
        "All OCR outputs use one fixed PARSeq checkpoint; no policy threshold is retuned.",
        "",
        f"- PARSeq SHA-256: `{summary['protocol']['parseq_sha256']}`",
        f"- Validation samples: `{summary['protocol']['splits']['val']['samples']}`",
        f"- Test samples: `{summary['protocol']['splits']['test']['samples']}`",
        "- Status: opened diagnostic benchmark, not eligible for promotion.",
        "",
        "| Method | Paradigm | Val exact | Test exact | Test delta vs unsharp | Test fixed/broken vs raw | Test CER |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    indexed = table.set_index("method_key")
    for key in METHOD_ORDER:
        row = indexed.loc[key]
        lines.append(
            f"| {row['method']} | {row['paradigm']} | {row['val_exact']:.4%} | "
            f"{row['test_exact']:.4%} | {row['test_delta_vs_unsharp_points']:+.4f} pt | "
            f"{int(row['test_fixed_vs_raw'])}/{int(row['test_broken_vs_raw'])} | "
            f"{row['test_cer']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- PixelRL was trained on the synthetic training split; other frozen policies were trained on historical OCR crops.",
            "- The underlying clean source plates may overlap historical policy-development data, so results are diagnostic.",
            "- Phase 4-7 policies were not retrained for synthetic blur; this measures frozen transfer, not each method's best achievable result.",
            "- Fixed/broken uses raw blurred PARSeq as the single shared reference for every method.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args) -> dict:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_dir = output_dir / "manifests"
    manifests = prepare_manifests(dataset_dir, manifest_dir)
    if args.prepare_manifests_only:
        payload = {"dataset_dir": str(dataset_dir), "manifests": manifests}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    parseq_checkpoint = Path(args.parseq_checkpoint).resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split_results = {}
    per_sample = {}
    for split in ("val", "test"):
        manifest = pd.read_csv(manifest_dir / f"{split}_manifest.csv")
        paths = manifest.image_path.astype(str).to_numpy()
        targets = manifest.label.astype(str).map(normalize_plate_text).to_numpy()
        pixel = pd.read_csv(Path(args.pixelrl_output) / f"eval_{split}_predictions.csv")
        assert_aligned(paths, targets, pixel.image_path.astype(str), pixel.label.astype(str), "PixelRL")

        phase46_dir = Path(getattr(args, f"phase46_{split}_cache")).resolve()
        cache46 = load_cache(phase46_dir, "val", [action.name for action in DEFAULT_ACTIONS])
        assert_aligned(paths, targets, cache46["image_paths"], cache46["targets"], "Phase 4/5")
        raw_index = [action.name for action in DEFAULT_ACTIONS].index("raw_rgb")
        train_baseline_index = 0
        raw_predictions = cache46["predictions"][:, raw_index].astype(str)
        if not np.array_equal(
            np.asarray([normalize_plate_text(v) for v in pixel.pred_blurred], dtype=str),
            np.asarray([normalize_plate_text(v) for v in raw_predictions], dtype=str),
        ):
            raise ValueError("Raw blurred predictions differ across evaluator paths; fixed-PARSeq contract failed")

        _, bandit_frame = load_router_predictions(cache46, Path(args.router_checkpoint), device)
        _, ppo5_frame = load_ppo5_predictions(
            cache46, Path(args.ppo5_checkpoint), Path(args.router_checkpoint), device
        )

        cache6, _, phase6_frame = load_candidate_predictions(
            Path(getattr(args, f"phase6_{split}_cache")), Path(args.phase6_checkpoint), device
        )
        assert_aligned(paths, targets, cache6["image_paths"], cache6["targets"], "Phase 6")
        cache7, _, phase7_frame = load_candidate_predictions(
            Path(getattr(args, f"phase7_{split}_cache")), Path(args.phase7_checkpoint), device
        )
        assert_aligned(paths, targets, cache7["image_paths"], cache7["targets"], "Phase 7")

        predictions = {
            "raw_parseq": raw_predictions,
            "classical_unsharp": pixel.pred_classical.astype(str).to_numpy(),
            "train_baseline": cache46["predictions"][:, train_baseline_index].astype(str),
            "pixelrl_a2c": pixel.pred_rl.astype(str).to_numpy(),
            "contextual_bandit_phase4": bandit_frame.prediction.astype(str).to_numpy(),
            "ppo_phase5": ppo5_frame.prediction.astype(str).to_numpy(),
            "candidate_ppo_phase6": phase6_frame.prediction.astype(str).to_numpy(),
            "compact_ppo_phase7": phase7_frame.prediction.astype(str).to_numpy(),
        }
        metrics = {
            key: score_predictions(targets, prediction, raw_predictions)
            for key, prediction in predictions.items()
        }
        for key in ("raw_parseq", "classical_unsharp", "pixelrl_a2c"):
            suffix = {"raw_parseq": "blurred", "classical_unsharp": "classical", "pixelrl_a2c": "rl"}[key]
            metrics[key]["psnr"] = float(pixel[f"psnr_{suffix}"].mean())
            metrics[key]["ssim"] = float(pixel[f"ssim_{suffix}"].mean())
        split_results[split] = metrics
        sample_frame = pd.DataFrame({"image_path": paths, "target": targets, "blur_kind": manifest.blur_kind})
        for key, values in predictions.items():
            sample_frame[f"prediction_{key}"] = values
            sample_frame[f"exact_{key}"] = np.asarray(values, dtype=str) == targets
        sample_frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)
        per_sample[split] = str((output_dir / f"{split}_predictions.csv").resolve())

    table_rows = []
    for key in METHOD_ORDER:
        table_rows.append(
            {
                "method_key": key,
                "method": METHOD_LABELS[key],
                "paradigm": PARADIGMS[key],
                "val_exact": split_results["val"][key]["exact_acc"],
                "val_cer": split_results["val"][key]["cer"],
                "val_fixed_vs_raw": split_results["val"][key]["fixed_vs_raw"],
                "val_broken_vs_raw": split_results["val"][key]["broken_vs_raw"],
                "test_exact": split_results["test"][key]["exact_acc"],
                "test_cer": split_results["test"][key]["cer"],
                "test_fixed_vs_raw": split_results["test"][key]["fixed_vs_raw"],
                "test_broken_vs_raw": split_results["test"][key]["broken_vs_raw"],
                "val_delta_vs_unsharp_points": 100.0
                * (
                    split_results["val"][key]["exact_acc"]
                    - split_results["val"]["classical_unsharp"]["exact_acc"]
                ),
                "test_delta_vs_unsharp_points": 100.0
                * (
                    split_results["test"][key]["exact_acc"]
                    - split_results["test"]["classical_unsharp"]["exact_acc"]
                ),
                "val_psnr": split_results["val"][key].get("psnr"),
                "test_psnr": split_results["test"][key].get("psnr"),
                "val_ssim": split_results["val"][key].get("ssim"),
                "test_ssim": split_results["test"][key].get("ssim"),
            }
        )
    table = pd.DataFrame(table_rows)
    table.to_csv(output_dir / "summary.csv", index=False)
    summary = {
        "protocol": {
            "role": "opened_common_dataset_diagnostic_not_promotable",
            "dataset_dir": str(dataset_dir),
            "splits": manifests,
            "parseq_checkpoint": str(parseq_checkpoint),
            "parseq_sha256": sha256(parseq_checkpoint),
            "refine_iters": int(args.refine_iters),
            "common_reference": "raw blurred image + fixed PARSeq",
            "no_rl_comparison_reference": "classical unsharp + fixed PARSeq",
            "threshold_retuning": False,
        },
        "checkpoints": {
            "pixelrl": str(Path(args.pixelrl_checkpoint).resolve()),
            "phase4": str(Path(args.router_checkpoint).resolve()),
            "phase5": str(Path(args.ppo5_checkpoint).resolve()),
            "phase6": str(Path(args.phase6_checkpoint).resolve()),
            "phase7": str(Path(args.phase7_checkpoint).resolve()),
        },
        "results": split_results,
        "per_sample": per_sample,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(markdown_report(summary, table), encoding="utf-8")
    print(table.to_string(index=False))
    return summary


def parse_args():
    benchmark = ROOT / "reinforcement_learning" / "common_rl_benchmark"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "parseq_rl_deblur_data" / "outputs" / "rl_deblur" / "dataset"),
    )
    parser.add_argument("--output-dir", default=str(benchmark))
    parser.add_argument("--prepare-manifests-only", action="store_true")
    parser.add_argument(
        "--parseq-checkpoint",
        default=str(ROOT / "outputs" / "phase3_controlled_aug_full_frozen_eval" / "best_phase3_parseq_anpr.pt"),
    )
    parser.add_argument(
        "--pixelrl-checkpoint",
        default=str(ROOT / "outputs" / "rl_deblur" / "checkpoints" / "best_deblur_agent.pt"),
    )
    parser.add_argument("--pixelrl-output", default=str(benchmark / "pixelrl"))
    parser.add_argument(
        "--router-checkpoint",
        default=str(ROOT / "outputs" / "rl_restoration" / "router_seed_123" / "best_reward_router.pt"),
    )
    parser.add_argument(
        "--ppo5-checkpoint",
        default=str(ROOT / "outputs" / "rl_restoration" / "ppo_prior_seed_123" / "best_ppo_restoration_policy.pt"),
    )
    parser.add_argument(
        "--phase6-checkpoint",
        default=str(
            ROOT
            / "reinforcement_learning"
            / "phase_6_candidate_oof_ppo"
            / "results"
            / "run_residual_seed_123"
            / "best_candidate_oof_ppo.pt"
        ),
    )
    parser.add_argument(
        "--phase7-checkpoint",
        default=str(
            ROOT
            / "reinforcement_learning"
            / "phase_7_compact_multiscale_ppo"
            / "results"
            / "run_seed_725"
            / "best_candidate_oof_ppo.pt"
        ),
    )
    for split in ("val", "test"):
        parser.add_argument(
            f"--phase46-{split}-cache",
            default=str(benchmark / "cache" / f"phase46_{split}"),
        )
        parser.add_argument(
            f"--phase6-{split}-cache",
            default=str(
                ROOT
                / "reinforcement_learning"
                / "phase_6_candidate_oof_ppo"
                / "results"
                / f"common_benchmark_{split}"
            ),
        )
        parser.add_argument(
            f"--phase7-{split}-cache",
            default=str(
                ROOT
                / "reinforcement_learning"
                / "phase_7_compact_multiscale_ppo"
                / "results"
                / f"common_benchmark_{split}"
            ),
        )
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""One-shot evaluation of a validation-locked PPO restoration policy on test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from rl_restoration.actions import DEFAULT_ACTIONS  # noqa: E402
from rl_restoration.finetune_with_policy import FixedActionDataset, evaluate, make_loader  # noqa: E402
from rl_restoration.policy import RewardRouter  # noqa: E402
from rl_restoration.ppo_policy import RestorationActorCritic  # noqa: E402
from rl_restoration.sequential_env import OfflineSequentialRestorationEnv  # noqa: E402
from rl_restoration.train_ppo import evaluate_policy  # noqa: E402
from rl_restoration.train_router import load_cache, predict_rewards  # noqa: E402


def model_metrics(checkpoint_path, frame, actions, model_cfg_reference, args, device, name):
    model, model_cfg, _ = load_notebook_checkpoint(Path(checkpoint_path), device, args.refine_iters)
    dataset = FixedActionDataset(frame, model_cfg_reference.img_size, actions)
    loader = make_loader(dataset, args.batch_size, args.num_workers)
    return evaluate(model, loader, device, name, model_cfg.max_label_length)


def teacher_predictions(path, features, action_names, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["action_names"] != action_names:
        raise ValueError("Teacher router action space differs from PPO action space")
    model = RewardRouter(
        checkpoint["input_dim"], len(action_names), checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    standardized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    return predict_rewards(model, standardized.astype(np.float32), device)


def run(args):
    policy_path = Path(args.ppo_checkpoint).resolve()
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_used", True):
        raise ValueError("PPO checkpoint is not marked as validation-locked")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    action_names = [action.name for action in DEFAULT_ACTIONS]
    if checkpoint["action_names"] != action_names:
        raise ValueError("PPO action space differs from current registry")
    test = load_cache(Path(args.cache_dir), "test", action_names)
    standardized = (test["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    teacher = teacher_predictions(Path(checkpoint["teacher_router"]), test["features"], action_names, device)
    features = np.concatenate((standardized, teacher), axis=1).astype(np.float32)
    env = OfflineSequentialRestorationEnv(
        test, features, device, checkpoint["revisit_cost"], checkpoint.get("candidate_summary", False)
    )
    policy = RestorationActorCritic(
        checkpoint["input_dim"],
        len(action_names),
        checkpoint["hidden_dim"],
        checkpoint["dropout"],
        checkpoint["prior_offset"],
        checkpoint["prior_scale"],
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    cached_metrics, selections = evaluate_policy(
        policy,
        env,
        test,
        action_names,
        checkpoint["first_margin"],
        checkpoint["revise_margin"],
    )
    selections.to_csv(output_dir / "test_locked_ppo_selections.csv", index=False)

    frame = pd.DataFrame(
        {"image_path": test["image_paths"], "label": test["targets"], "plate_type": "test", "split": "test"}
    )
    actions = dict(zip(selections.image_path.astype(str), selections.final_action.astype(str)))
    _, reference_cfg, _ = load_notebook_checkpoint(Path(args.parent_checkpoint), device, args.refine_iters)
    parent_metrics, parent_predictions = model_metrics(
        args.parent_checkpoint, frame, actions, reference_cfg, args, device, "test_parent_ppo_locked"
    )
    parent_predictions.to_csv(output_dir / "test_parent_ppo_predictions.csv", index=False)
    comparison_metrics = None
    if args.comparison_checkpoint:
        comparison_metrics, comparison_predictions = model_metrics(
            args.comparison_checkpoint, frame, actions, reference_cfg, args, device, "test_comparison_ppo_locked"
        )
        comparison_predictions.to_csv(output_dir / "test_comparison_ppo_predictions.csv", index=False)
    summary = {
        "policy_locked_before_test": True,
        "algorithm": checkpoint["algorithm"],
        "ppo_checkpoint": str(policy_path),
        "ppo_seed": checkpoint["seed"],
        "ppo_epoch": checkpoint["epoch"],
        "first_margin": checkpoint["first_margin"],
        "revise_margin": checkpoint["revise_margin"],
        "cached_parent_ocr_policy_metrics": cached_metrics,
        "parent_ocr_recomputed": parent_metrics,
        "comparison_ocr_metrics": comparison_metrics,
        "action_distribution": selections.final_action.value_counts().to_dict(),
        "first_action_distribution": selections.first_action.value_counts().to_dict(),
        "test_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/rl_restoration/trajectory_cache"))
    parser.add_argument(
        "--ppo-checkpoint",
        default=str(ROOT / "outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt"),
    )
    parser.add_argument(
        "--parent-checkpoint",
        default=str(ROOT / "outputs/phase3_controlled_aug_full_frozen_eval/best_phase3_parseq_anpr.pt"),
    )
    parser.add_argument(
        "--comparison-checkpoint",
        default=str(ROOT / "outputs/rl_restoration/parseq_policy_hard_curriculum/best_parseq_rl_policy_mixture.pt"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rl_restoration/test_locked_ppo"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

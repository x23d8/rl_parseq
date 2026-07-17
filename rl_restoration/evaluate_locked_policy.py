"""Evaluate a previously locked restoration policy without test-time tuning."""

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
from rl_restoration.train_router import evaluate_selection, load_cache, predict_rewards  # noqa: E402


def evaluate_checkpoint(checkpoint_path, frame, actions, model_cfg_reference, args, device, name):
    model, model_cfg, _ = load_notebook_checkpoint(Path(checkpoint_path), device, args.refine_iters)
    dataset = FixedActionDataset(frame, model_cfg_reference.img_size, actions)
    loader = make_loader(dataset, args.batch_size, args.num_workers)
    metrics, predictions = evaluate(model, loader, device, name, model_cfg.max_label_length)
    return metrics, predictions


def run(args):
    cache_dir = Path(args.cache_dir).resolve()
    router_path = Path(args.router_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(router_path, map_location="cpu", weights_only=False)
    action_names = [action.name for action in DEFAULT_ACTIONS]
    if checkpoint["action_names"] != action_names:
        raise ValueError("Router action space differs from the current action registry")
    test = load_cache(cache_dir, "test", action_names)
    standardized = (test["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    router = RewardRouter(
        checkpoint["input_dim"], len(action_names), checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    router.load_state_dict(checkpoint["model_state_dict"])
    predicted_rewards = predict_rewards(router, standardized.astype(np.float32), device)
    policy_metrics_cache, selections, _ = evaluate_selection(
        test, predicted_rewards, action_names, float(checkpoint["selection_margin"])
    )
    selections.to_csv(output_dir / "test_locked_policy_selections.csv", index=False)
    frame = pd.DataFrame(
        {
            "image_path": test["image_paths"],
            "label": test["targets"],
            "plate_type": "test",
            "split": "test",
        }
    )
    actions = dict(zip(selections.image_path.astype(str), selections.selected_action.astype(str)))
    _, model_cfg_reference, _ = load_notebook_checkpoint(Path(args.parent_checkpoint), device, args.refine_iters)
    parent_metrics, parent_predictions = evaluate_checkpoint(
        args.parent_checkpoint, frame, actions, model_cfg_reference, args, device, "test_parent_policy_locked"
    )
    parent_predictions.to_csv(output_dir / "test_parent_policy_predictions.csv", index=False)
    comparison_metrics = None
    if args.comparison_checkpoint:
        comparison_metrics, comparison_predictions = evaluate_checkpoint(
            args.comparison_checkpoint,
            frame,
            actions,
            model_cfg_reference,
            args,
            device,
            "test_comparison_policy_locked",
        )
        comparison_predictions.to_csv(output_dir / "test_comparison_policy_predictions.csv", index=False)
    baseline = test["exact"][:, action_names.index("stop_baseline")].astype(bool)
    policy_exact = selections.exact.astype(bool).to_numpy()
    summary = {
        "policy_locked_before_test": True,
        "router_checkpoint": str(router_path),
        "router_epoch": checkpoint["epoch"],
        "selection_margin": checkpoint["selection_margin"],
        "cached_parent_policy_metrics": policy_metrics_cache,
        "parent_policy_metrics_recomputed": parent_metrics,
        "comparison_policy_metrics": comparison_metrics,
        "fixed_vs_baseline": int(((~baseline) & policy_exact).sum()),
        "broken_vs_baseline": int((baseline & (~policy_exact)).sum()),
        "action_distribution": selections.selected_action.value_counts().to_dict(),
        "test_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parent = ROOT / "outputs/phase3_controlled_aug_full_frozen_eval/best_phase3_parseq_anpr.pt"
    router = ROOT / "outputs/rl_restoration/router_seed_123/best_reward_router.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/rl_restoration/trajectory_cache"))
    parser.add_argument("--router-checkpoint", default=str(router))
    parser.add_argument("--parent-checkpoint", default=str(parent))
    parser.add_argument(
        "--comparison-checkpoint",
        default=str(ROOT / "outputs/rl_restoration/parseq_policy_mixture/best_parseq_rl_policy_mixture.pt"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rl_restoration/test_locked_evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


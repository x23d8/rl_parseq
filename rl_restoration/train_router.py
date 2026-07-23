"""Train and lock a restoration contextual-bandit policy on offline rewards."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_restoration.actions import get_action_profile  # noqa: E402
from rl_restoration.policy import RewardRouter, standardize_features  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(cache_dir: Path, split: str, action_names: list[str]):
    # Older caches in this repository stored paths as a NumPy object array.
    # They contain only locally generated strings, so allow_pickle maintains
    # backward compatibility; new caches use fixed-width Unicode arrays.
    feature_payload = np.load(cache_dir / f"{split}_state_features.npz", allow_pickle=True)
    features = feature_payload["features"].astype(np.float32)
    image_paths = feature_payload["image_paths"].astype(str)
    trajectories = pd.read_csv(cache_dir / f"{split}_action_trajectories.csv")
    trajectory_paths = trajectories.image_path.astype(str)
    path_to_index = {path: index for index, path in enumerate(image_paths)}
    if set(trajectory_paths) - set(path_to_index):
        raise ValueError(f"{split} trajectories and feature paths do not match")

    matrices = {}
    for column in ("reward", "exact", "edit_distance", "action_cost", "normalized_confidence"):
        pivot = trajectories.pivot(index="image_path", columns="action", values=column)
        pivot = pivot.reindex(index=image_paths, columns=action_names)
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete {column} matrix in {split} cache")
        matrices[column] = pivot.to_numpy()
    targets = trajectories[trajectories.action == action_names[0]].set_index("image_path").reindex(image_paths)["target"].astype(str).to_numpy()
    predictions = trajectories.pivot(index="image_path", columns="action", values="prediction").reindex(index=image_paths, columns=action_names).to_numpy()
    return {
        "features": features,
        "image_paths": image_paths,
        "targets": targets,
        "predictions": predictions,
        **matrices,
    }


def evaluate_selection(cache, predicted_rewards, action_names, margin: float):
    stop_index = action_names.index("stop_baseline")
    best_indices = predicted_rewards.argmax(axis=1)
    gains = predicted_rewards[np.arange(len(best_indices)), best_indices] - predicted_rewards[:, stop_index]
    selected = np.where(gains >= margin, best_indices, stop_index)
    rows = np.arange(len(selected))
    exact = cache["exact"][rows, selected].astype(bool)
    baseline_exact = cache["exact"][:, stop_index].astype(bool)
    edits = cache["edit_distance"][rows, selected]
    target_lengths = np.asarray([max(len(value), 1) for value in cache["targets"]])
    fixed = (~baseline_exact) & exact
    broken = baseline_exact & (~exact)
    metrics = {
        "samples": int(len(selected)),
        "exact_acc": float(exact.mean()),
        "char_acc": float(1.0 - edits.sum() / target_lengths.sum()),
        "fixed": int(fixed.sum()),
        "broken": int(broken.sum()),
        "net_fixes": int(fixed.sum() - broken.sum()),
        "mean_cost": float(cache["action_cost"][rows, selected].mean()),
        "stop_rate": float((selected == stop_index).mean()),
        "margin": float(margin),
    }
    selected_frame = pd.DataFrame(
        {
            "image_path": cache["image_paths"],
            "target": cache["targets"],
            "selected_action": [action_names[index] for index in selected],
            "prediction": cache["predictions"][rows, selected],
            "exact": exact,
            "edit_distance": edits.astype(int),
            "baseline_exact": baseline_exact,
            "fixed": fixed,
            "broken": broken,
            "predicted_reward_gain": gains,
            "actual_reward": cache["reward"][rows, selected],
        }
    )
    return metrics, selected_frame, selected


def metric_key(metrics):
    return (
        metrics["exact_acc"],
        metrics["char_acc"],
        metrics["net_fixes"],
        -metrics["broken"],
        -metrics["mean_cost"],
    )


@torch.inference_mode()
def predict_rewards(model, features, device, batch_size=512):
    model.eval()
    result = []
    tensor = torch.from_numpy(features.astype(np.float32))
    for start in range(0, len(tensor), batch_size):
        result.append(model(tensor[start : start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(result)


def run(args):
    set_seed(args.seed)
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    actions = get_action_profile(args.action_profile)
    action_names = [action.name for action in actions]
    train = load_cache(cache_dir, "train", action_names)
    val = load_cache(cache_dir, "val", action_names)
    train_x, val_x, feature_mean, feature_std = standardize_features(train["features"], val["features"])
    train_y = train["reward"].astype(np.float32)

    oracle_indices = train_y.argmax(axis=1)
    stop_index = action_names.index("stop_baseline")
    sample_weights = np.ones(len(train_y), dtype=np.float32)
    sample_weights += 2.0 * (oracle_indices != stop_index)
    sample_weights += 1.0 * (~train["exact"][:, stop_index].astype(bool))
    dataset = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_y),
        torch.from_numpy(sample_weights),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = RewardRouter(train_x.shape[1], len(action_names), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    margins = [float(value) for value in args.margins.split(",")]
    history = []
    best_key = (-1.0, -1.0, -10**9, -10**9, -10**9)
    best_payload = None
    best_path = output_dir / "best_reward_router.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, rewards, weights in loader:
            features = features.to(device)
            rewards = rewards.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(features)
            per_action = F.smooth_l1_loss(predicted, rewards, reduction="none")
            regression = (per_action.mean(dim=1) * weights).mean()
            target_distribution = torch.softmax(rewards / args.reward_temperature, dim=1)
            ranking = -(target_distribution * torch.log_softmax(predicted / args.reward_temperature, dim=1)).sum(dim=1)
            loss = regression + args.ranking_weight * (ranking * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()

        val_predicted = predict_rewards(model, val_x, device)
        epoch_candidates = []
        for margin in margins:
            metrics, _, _ = evaluate_selection(val, val_predicted, action_names, margin)
            epoch_candidates.append(metrics)
        selected_metrics = max(epoch_candidates, key=metric_key)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **selected_metrics}
        history.append(row)
        if metric_key(selected_metrics) > best_key:
            best_key = metric_key(selected_metrics)
            best_payload = {"epoch": epoch, "margin": selected_metrics["margin"], "metrics": selected_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_x.shape[1],
                    "action_names": action_names,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "selection_margin": selected_metrics["margin"],
                    "epoch": epoch,
                    "validation_metrics": selected_metrics,
                    "seed": args.seed,
                    "action_profile": args.action_profile,
                },
                best_path,
            )
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row, ensure_ascii=False))

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "router_history.csv", index=False)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = RewardRouter(
        checkpoint["input_dim"], len(action_names), checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    train_predicted = predict_rewards(best_model, train_x, device)
    val_predicted = predict_rewards(best_model, val_x, device)
    train_metrics, train_selected, _ = evaluate_selection(
        train, train_predicted, action_names, checkpoint["selection_margin"]
    )
    val_metrics, val_selected, _ = evaluate_selection(
        val, val_predicted, action_names, checkpoint["selection_margin"]
    )
    train_selected.to_csv(output_dir / "train_policy_selections.csv", index=False)
    val_selected.to_csv(output_dir / "val_policy_predictions.csv", index=False)

    val_oracle = val["reward"].argmax(axis=1)
    rows = np.arange(len(val_oracle))
    oracle_metrics = {
        "exact_acc": float(val["exact"][rows, val_oracle].mean()),
        "char_acc": float(
            1.0
            - val["edit_distance"][rows, val_oracle].sum()
            / sum(max(len(value), 1) for value in val["targets"])
        ),
    }
    baseline_metrics, _, _ = evaluate_selection(
        val, np.zeros_like(val_predicted), action_names, margin=1.0
    )
    summary = {
        "algorithm": "offline_contextual_bandit_reward_regression",
        "best": best_payload,
        "train_policy": train_metrics,
        "validation_baseline": baseline_metrics,
        "validation_policy": val_metrics,
        "validation_oracle": oracle_metrics,
        "action_distribution_validation": val_selected.selected_action.value_counts().to_dict(),
        "checkpoint": str(best_path),
        "test_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/rl_restoration/trajectory_cache"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rl_restoration/router"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reward-temperature", type=float, default=0.08)
    parser.add_argument("--ranking-weight", type=float, default=0.10)
    parser.add_argument("--margins", default="0,0.005,0.01,0.02,0.03,0.05,0.08")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--action-profile",
        choices=["default", "fair_restoration"],
        default="default",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

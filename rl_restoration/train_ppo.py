"""Train a two-step restoration policy with BC initialization and PPO/GAE.

Only train rewards update the policy. Validation selects checkpoint and
conservative accept/revise margins. Test is never loaded by this script.
"""

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
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_restoration.actions import DEFAULT_ACTIONS  # noqa: E402
from rl_restoration.ppo_policy import RestorationActorCritic  # noqa: E402
from rl_restoration.policy import RewardRouter  # noqa: E402
from rl_restoration.sequential_env import OfflineSequentialRestorationEnv  # noqa: E402
from rl_restoration.train_router import load_cache, metric_key, predict_rewards  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (other - mean) / std, mean, std


def load_teacher_prior(path: Path, train_features: np.ndarray, val_features: np.ndarray, action_names, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["action_names"] != action_names:
        raise ValueError("Teacher router action space differs from PPO action space")
    model = RewardRouter(
        checkpoint["input_dim"], len(action_names), checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_x = (train_features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    val_x = (val_features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    return (
        predict_rewards(model, train_x.astype(np.float32), device),
        predict_rewards(model, val_x.astype(np.float32), device),
        checkpoint,
    )


def policy_actions(model, env, indices, first_margin: float, revise_margin: float):
    model.eval()
    with torch.inference_mode():
        logits0, _ = model(env.state_zero(indices))
        best0 = logits0.argmax(dim=1)
        advantage0 = logits0.gather(1, best0[:, None]).squeeze(1) - logits0[:, 0]
        first = torch.where((best0 != 0) & (advantage0 >= first_margin), best0, torch.zeros_like(best0))
        final = first.clone()
        revised = torch.zeros_like(first, dtype=torch.bool)
        active = first != 0
        if active.any():
            active_indices = indices[active]
            logits1, _ = model(env.state_one(active_indices, first[active]))
            best1 = logits1.argmax(dim=1)
            current_logits = logits1.gather(1, first[active][:, None]).squeeze(1)
            advantage1 = logits1.gather(1, best1[:, None]).squeeze(1) - current_logits
            second = torch.where(advantage1 >= revise_margin, best1, first[active])
            final[active] = env.terminal_actions(first[active], second)
            revised[active] = (second != 0) & (second != first[active])
    return first, final, revised


def evaluate_policy(model, env, cache, action_names, first_margin, revise_margin):
    indices = torch.arange(len(cache["features"]), device=env.device)
    first, final, revised = policy_actions(model, env, indices, first_margin, revise_margin)
    final_np = final.cpu().numpy()
    first_np = first.cpu().numpy()
    revised_np = revised.cpu().numpy()
    rows = np.arange(len(final_np))
    exact = cache["exact"][rows, final_np].astype(bool)
    baseline = cache["exact"][:, 0].astype(bool)
    edits = cache["edit_distance"][rows, final_np]
    total_chars = sum(max(len(value), 1) for value in cache["targets"])
    metrics = {
        "samples": int(len(final_np)),
        "exact_acc": float(exact.mean()),
        "char_acc": float(1.0 - edits.sum() / total_chars),
        "fixed": int(((~baseline) & exact).sum()),
        "broken": int((baseline & (~exact)).sum()),
        "net_fixes": int(((~baseline) & exact).sum() - (baseline & (~exact)).sum()),
        "stop_rate": float((final_np == 0).mean()),
        "revise_rate": float(revised_np.mean()),
        "mean_cost": float(cache["action_cost"][rows, final_np].mean()),
        "first_margin": float(first_margin),
        "revise_margin": float(revise_margin),
    }
    frame = pd.DataFrame(
        {
            "image_path": cache["image_paths"],
            "target": cache["targets"],
            "first_action": [action_names[index] for index in first_np],
            "final_action": [action_names[index] for index in final_np],
            "revised": revised_np,
            "prediction": cache["predictions"][rows, final_np],
            "exact": exact,
            "edit_distance": edits.astype(int),
            "baseline_exact": baseline,
            "fixed": (~baseline) & exact,
            "broken": baseline & (~exact),
        }
    )
    return metrics, frame


def behavior_clone(model, optimizer, env, cache, args):
    oracle = cache["reward"].argmax(axis=1).astype(np.int64)
    weights = np.ones(len(oracle), dtype=np.float32)
    weights += args.hard_sample_weight * (oracle != 0)
    weights += args.error_sample_weight * (~cache["exact"][:, 0].astype(bool))
    weight_tensor = torch.as_tensor(weights, device=env.device)
    history = []
    for epoch in range(1, args.bc_epochs + 1):
        sampled = torch.multinomial(weight_tensor, len(oracle), replacement=True)
        losses = []
        for start in range(0, len(sampled), args.batch_size):
            indices = sampled[start : start + args.batch_size]
            targets = torch.as_tensor(oracle[indices.cpu().numpy()], device=env.device)
            logits0, _ = model(env.state_zero(indices))
            # Random first views make accept, rollback and revise identifiable.
            first = torch.randint(1, env.action_count, (len(indices),), device=env.device)
            targets1 = targets
            logits1, _ = model(env.state_one(indices, first))
            loss = args.bc_first_weight * F.cross_entropy(logits0, targets) + args.bc_second_weight * F.cross_entropy(logits1, targets1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        history.append({"stage": "bc", "epoch": epoch, "loss": float(np.mean(losses))})
    return history


def collect_rollout(model, env, cache, args):
    oracle = cache["reward"].argmax(axis=1)
    weights = np.ones(len(oracle), dtype=np.float32)
    weights += args.hard_sample_weight * (oracle != 0)
    weights += args.error_sample_weight * (~cache["exact"][:, 0].astype(bool))
    sampled = torch.multinomial(torch.as_tensor(weights, device=env.device), args.rollout_size, replacement=True)
    observations, actions, old_log_probs, advantages, returns = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sampled), args.batch_size):
            indices = sampled[start : start + args.batch_size]
            obs0 = env.state_zero(indices)
            logits0, values0 = model(obs0)
            dist0 = Categorical(logits=logits0)
            action0 = dist0.sample()
            log0 = dist0.log_prob(action0)
            active = action0 != 0

            reward0 = torch.zeros_like(values0)
            advantage0 = reward0 - values0
            return0 = reward0
            if active.any():
                active_indices = indices[active]
                obs1 = env.state_one(active_indices, action0[active])
                logits1, values1 = model(obs1)
                dist1 = Categorical(logits=logits1)
                action1 = dist1.sample()
                log1 = dist1.log_prob(action1)
                terminal_reward, _ = env.terminal_rewards(active_indices, action0[active], action1)
                advantage1 = terminal_reward - values1
                delta0 = -args.step_cost + args.gamma * values1 - values0[active]
                advantage0[active] = delta0 + args.gamma * args.gae_lambda * advantage1
                return0[active] = advantage0[active] + values0[active]

                observations.append(obs1)
                actions.append(action1)
                old_log_probs.append(log1)
                advantages.append(advantage1)
                returns.append(terminal_reward)

            observations.append(obs0)
            actions.append(action0)
            old_log_probs.append(log0)
            advantages.append(advantage0)
            returns.append(return0)
    return tuple(torch.cat(items).detach() for items in (observations, actions, old_log_probs, advantages, returns))


def ppo_update(model, optimizer, rollout, args):
    observations, actions, old_log_probs, advantages, returns = rollout
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    losses = []
    clip_fractions = []
    for _ in range(args.update_epochs):
        order = torch.randperm(len(observations), device=observations.device)
        for start in range(0, len(order), args.minibatch_size):
            batch = order[start : start + args.minibatch_size]
            logits, values = model(observations[batch])
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(actions[batch])
            ratio = (log_probs - old_log_probs[batch]).exp()
            unclipped = ratio * advantages[batch]
            clipped = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantages[batch]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.smooth_l1_loss(values, returns[batch])
            entropy = distribution.entropy().mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            clip_fractions.append(float(((ratio - 1.0).abs() > args.clip_ratio).float().mean().item()))
    return float(np.mean(losses)), float(np.mean(clip_fractions))


def run(args):
    set_seed(args.seed)
    if "test" in args.cache_splits.lower():
        raise ValueError("PPO training must not load test")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    action_names = [action.name for action in DEFAULT_ACTIONS]
    train = load_cache(Path(args.cache_dir), "train", action_names)
    val = load_cache(Path(args.cache_dir), "val", action_names)
    train_x, val_x, feature_mean, feature_std = standardize(train["features"], val["features"])
    prior_offset = train_x.shape[1]
    teacher_train, teacher_val, _ = load_teacher_prior(
        Path(args.teacher_router), train["features"], val["features"], action_names, device
    )
    train_x = np.concatenate((train_x, teacher_train), axis=1)
    val_x = np.concatenate((val_x, teacher_val), axis=1)
    train_env = OfflineSequentialRestorationEnv(
        train, train_x.astype(np.float32), device, args.revisit_cost, args.candidate_summary
    )
    val_env = OfflineSequentialRestorationEnv(
        val, val_x.astype(np.float32), device, args.revisit_cost, args.candidate_summary
    )
    model = RestorationActorCritic(
        train_env.observation_dim,
        len(action_names),
        args.hidden_dim,
        args.dropout,
        prior_offset=prior_offset,
        prior_scale=args.prior_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = behavior_clone(model, optimizer, train_env, train, args)

    margins = [float(value) for value in args.margins.split(",")]
    best_key = (-1.0, -1.0, -10**9, -10**9, -10**9)
    best_payload = None
    checkpoint_path = output_dir / "best_ppo_restoration_policy.pt"
    for epoch in range(1, args.ppo_epochs + 1):
        rollout = collect_rollout(model, train_env, train, args)
        model.train()
        loss, clip_fraction = ppo_update(model, optimizer, rollout, args)
        candidates = []
        for first_margin in margins:
            for revise_margin in margins:
                metrics, _ = evaluate_policy(model, val_env, val, action_names, first_margin, revise_margin)
                candidates.append(metrics)
        selected = max(candidates, key=metric_key)
        row = {"stage": "ppo", "epoch": epoch, "loss": loss, "clip_fraction": clip_fraction, **selected}
        history.append(row)
        if metric_key(selected) > best_key:
            best_key = metric_key(selected)
            best_payload = row
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_env.observation_dim,
                    "action_names": action_names,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "prior_offset": prior_offset,
                    "prior_scale": args.prior_scale,
                    "teacher_router": str(Path(args.teacher_router).resolve()),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "first_margin": selected["first_margin"],
                    "revise_margin": selected["revise_margin"],
                    "revisit_cost": args.revisit_cost,
                    "candidate_summary": args.candidate_summary,
                    "epoch": epoch,
                    "seed": args.seed,
                    "validation_metrics": selected,
                    "algorithm": "bc_initialized_ppo_gae_two_step",
                    "test_used": False,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row, ensure_ascii=False))

    pd.DataFrame(history).to_csv(output_dir / "ppo_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    locked = RestorationActorCritic(
        checkpoint["input_dim"],
        len(action_names),
        checkpoint["hidden_dim"],
        checkpoint["dropout"],
        checkpoint["prior_offset"],
        checkpoint["prior_scale"],
    ).to(device)
    locked.load_state_dict(checkpoint["model_state_dict"])
    train_metrics, train_frame = evaluate_policy(
        locked, train_env, train, action_names, checkpoint["first_margin"], checkpoint["revise_margin"]
    )
    val_metrics, val_frame = evaluate_policy(
        locked, val_env, val, action_names, checkpoint["first_margin"], checkpoint["revise_margin"]
    )
    train_frame.to_csv(output_dir / "train_ppo_selections.csv", index=False)
    val_frame.to_csv(output_dir / "val_ppo_selections.csv", index=False)
    oracle = val["reward"].argmax(axis=1)
    rows = np.arange(len(oracle))
    summary = {
        "algorithm": "bc_initialized_ppo_gae_two_step",
        "best": best_payload,
        "train_policy": train_metrics,
        "validation_policy": val_metrics,
        "validation_oracle_exact": float(val["exact"][rows, oracle].mean()),
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "test_used": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/rl_restoration/trajectory_cache"))
    parser.add_argument(
        "--teacher-router",
        default=str(ROOT / "outputs/rl_restoration/router_seed_123/best_reward_router.pt"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rl_restoration/ppo_seed_20260715"))
    parser.add_argument("--cache-splits", default="train,val")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=50)
    parser.add_argument("--rollout-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.10)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--prior-scale", type=float, default=20.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--step-cost", type=float, default=0.001)
    parser.add_argument("--revisit-cost", type=float, default=0.002)
    parser.add_argument("--hard-sample-weight", type=float, default=6.0)
    parser.add_argument("--error-sample-weight", type=float, default=3.0)
    parser.add_argument("--bc-second-weight", type=float, default=0.5)
    parser.add_argument("--bc-first-weight", type=float, default=1.0)
    parser.add_argument("--candidate-summary", action="store_true")
    parser.add_argument("--margins", default="0,0.025,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

"""Train candidate-aware PPO with OOF teacher priors and a locked group holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RL_ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.data import (  # noqa: E402
    candidate_ocr_features,
    load_candidate_features,
    load_trajectory_cache,
    stable_group_holdout,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.model import (  # noqa: E402
    CandidateSetActorCritic,
    RewardTeacher,
)
from reinforcement_learning.phase_6_candidate_oof_ppo.paired_statistics import (  # noqa: E402
    improvement_gate,
    mcnemar_exact,
    paired_bootstrap,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    result = np.empty(len(groups), dtype=np.int64)
    for index, group in enumerate(np.asarray(groups, dtype=str)):
        digest = hashlib.sha256(f"fold:{seed}:{group}".encode("utf-8")).digest()
        result[index] = int.from_bytes(digest[:8], "big") % folds
    if len(np.unique(result)) != folds:
        raise ValueError("Not all requested OOF folds contain data")
    return result


def normalize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((train - mean) / std, *((value - mean) / std for value in others), mean, std)


def sample_weights(rewards: np.ndarray, exact: np.ndarray) -> np.ndarray:
    oracle = rewards.argmax(axis=1)
    weights = np.ones(len(rewards), dtype=np.float32)
    weights += 5.0 * (oracle != 0)
    weights += 2.0 * (~exact[:, 0].astype(bool))
    return weights


def fit_teacher(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    args,
    seed_offset: int = 0,
) -> RewardTeacher:
    torch.manual_seed(args.seed + seed_offset)
    model = RewardTeacher(x.shape[1], y.shape[1], args.teacher_hidden, args.teacher_dropout).to(args.device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=1e-4)
    x_tensor = torch.from_numpy(x).float().to(args.device_obj)
    y_tensor = torch.from_numpy(y).float().to(args.device_obj)
    weight_tensor = torch.from_numpy(weights).float().to(args.device_obj)
    for _ in range(args.teacher_epochs):
        order = torch.randperm(len(x_tensor), device=args.device_obj)
        model.train()
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            predicted = model(x_tensor[batch])
            regression = F.smooth_l1_loss(predicted, y_tensor[batch], reduction="none").mean(dim=1)
            target = torch.softmax(y_tensor[batch] / args.reward_temperature, dim=1)
            ranking = -(target * torch.log_softmax(predicted / args.reward_temperature, dim=1)).sum(dim=1)
            loss = ((regression + args.teacher_ranking_weight * ranking) * weight_tensor[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.inference_mode()
def teacher_predict(model: RewardTeacher, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    tensor = torch.from_numpy(features).float()
    for start in range(0, len(tensor), 512):
        result.append(model(tensor[start : start + 512].to(device)).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def make_oof_teacher(dev_x: np.ndarray, dev_cache: dict, args):
    folds = group_folds(dev_cache["targets"], args.oof_folds, args.seed)
    weights = sample_weights(dev_cache["reward"], dev_cache["exact"])
    predictions = np.empty_like(dev_cache["reward"], dtype=np.float32)
    for fold in range(args.oof_folds):
        train_mask = folds != fold
        valid_mask = ~train_mask
        model = fit_teacher(
            dev_x[train_mask],
            dev_cache["reward"][train_mask].astype(np.float32),
            weights[train_mask],
            args,
            seed_offset=fold + 1,
        )
        predictions[valid_mask] = teacher_predict(model, dev_x[valid_mask], args.device_obj)
    full_model = fit_teacher(
        dev_x,
        dev_cache["reward"].astype(np.float32),
        weights,
        args,
        seed_offset=100,
    )
    return predictions, full_model, folds


def cache_subset(cache: dict, mask: np.ndarray) -> dict:
    return {
        key: value[mask] if isinstance(value, np.ndarray) and value.shape[0] == len(mask) else value
        for key, value in cache.items()
    }


def model_forward(model, candidates, priors, indices, current, step):
    return model(candidates[indices], priors[indices], current, step)


def behavior_clone(model, optimizer, candidates, priors, cache, args):
    rewards = torch.from_numpy(cache["reward"].astype(np.float32)).to(args.device_obj)
    targets = torch.from_numpy(cache["reward"].argmax(axis=1).astype(np.int64)).to(args.device_obj)
    weights = torch.from_numpy(sample_weights(cache["reward"], cache["exact"])).to(args.device_obj)
    history = []
    for epoch in range(1, args.bc_epochs + 1):
        sampled = torch.multinomial(weights, len(weights), replacement=True)
        losses = []
        model.train()
        for start in range(0, len(sampled), args.batch_size):
            indices = sampled[start : start + args.batch_size]
            current0 = torch.zeros(len(indices), dtype=torch.long, device=args.device_obj)
            step0 = torch.zeros(len(indices), device=args.device_obj)
            logits0, value0 = model_forward(model, candidates, priors, indices, current0, step0)
            current1 = torch.randint(1, rewards.shape[1], (len(indices),), device=args.device_obj)
            step1 = torch.ones(len(indices), device=args.device_obj)
            logits1, value1 = model_forward(model, candidates, priors, indices, current1, step1)
            target = targets[indices]
            best_reward = rewards[indices].max(dim=1).values
            actor_loss = F.cross_entropy(logits0, target) + 0.5 * F.cross_entropy(logits1, target)
            critic_loss = F.smooth_l1_loss(value0, best_reward) + F.smooth_l1_loss(value1, best_reward)
            loss = actor_loss + args.value_coef * critic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        history.append({"stage": "bc", "epoch": epoch, "loss": float(np.mean(losses))})
    return history


def collect_rollout(model, candidates, priors, cache, args):
    weights = torch.from_numpy(sample_weights(cache["reward"], cache["exact"])).to(args.device_obj)
    sampled = torch.multinomial(weights, args.rollout_size, replacement=True)
    rewards = torch.from_numpy(cache["reward"].astype(np.float32)).to(args.device_obj)
    records = {key: [] for key in ("indices", "current", "step", "action", "log_prob", "advantage", "return")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sampled), args.batch_size):
            indices = sampled[start : start + args.batch_size]
            current0 = torch.zeros(len(indices), dtype=torch.long, device=args.device_obj)
            step0 = torch.zeros(len(indices), device=args.device_obj)
            logits0, values0 = model_forward(model, candidates, priors, indices, current0, step0)
            dist0 = Categorical(logits=logits0)
            action0 = dist0.sample()
            log0 = dist0.log_prob(action0)
            terminal0 = action0 == 0
            advantage0 = rewards[indices, action0] - values0
            return0 = rewards[indices, action0]
            active = ~terminal0
            if active.any():
                active_indices = indices[active]
                current1 = action0[active]
                step1 = torch.ones(len(active_indices), device=args.device_obj)
                logits1, values1 = model_forward(model, candidates, priors, active_indices, current1, step1)
                dist1 = Categorical(logits=logits1)
                action1 = dist1.sample()
                log1 = dist1.log_prob(action1)
                terminal_reward = rewards[active_indices, action1] - (action1 != current1).float() * args.revisit_cost
                advantage1 = terminal_reward - values1
                delta0 = -args.step_cost + args.gamma * values1 - values0[active]
                advantage0[active] = delta0 + args.gamma * args.gae_lambda * advantage1
                return0[active] = advantage0[active] + values0[active]
                for key, value in zip(
                    records,
                    (active_indices, current1, step1, action1, log1, advantage1, terminal_reward),
                ):
                    records[key].append(value)
            for key, value in zip(records, (indices, current0, step0, action0, log0, advantage0, return0)):
                records[key].append(value)
    return {key: torch.cat(value).detach() for key, value in records.items()}


def ppo_update(model, optimizer, candidates, priors, rollout, args):
    advantages = rollout["advantage"]
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    losses = []
    for _ in range(args.update_epochs):
        order = torch.randperm(len(advantages), device=args.device_obj)
        for start in range(0, len(order), args.minibatch_size):
            batch = order[start : start + args.minibatch_size]
            logits, values = model_forward(
                model,
                candidates,
                priors,
                rollout["indices"][batch],
                rollout["current"][batch],
                rollout["step"][batch],
            )
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(rollout["action"][batch])
            ratio = (log_prob - rollout["log_prob"][batch]).exp()
            raw = ratio * advantages[batch]
            clipped = ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio) * advantages[batch]
            policy_loss = -torch.minimum(raw, clipped).mean()
            value_loss = F.smooth_l1_loss(values, rollout["return"][batch])
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
    return float(np.mean(losses))


@torch.inference_mode()
def policy_selection(
    model,
    candidates,
    priors,
    first_margin: float,
    revise_margin: float,
    device,
    teacher_margin: float = 0.0,
    disagreement_margin: float | None = None,
    final_teacher_gain_margin: float | None = None,
):
    model.eval()
    final_parts, first_parts, revise_parts = [], [], []
    for start in range(0, len(candidates), 512):
        stop = min(start + 512, len(candidates))
        batch_candidates = candidates[start:stop]
        batch_priors = priors[start:stop]
        current0 = torch.zeros(len(batch_candidates), dtype=torch.long, device=device)
        step0 = torch.zeros(len(batch_candidates), device=device)
        logits0, _ = model(batch_candidates, batch_priors, current0, step0)
        teacher_best = batch_priors.argmax(dim=1)
        teacher_gain = batch_priors.gather(1, teacher_best[:, None]).squeeze(1) - batch_priors[:, 0]
        teacher_selected = torch.where(
            (teacher_best != 0) & (teacher_gain >= teacher_margin), teacher_best, current0
        )
        best0 = logits0.argmax(dim=1)
        gain0 = logits0.gather(1, best0[:, None]).squeeze(1) - logits0[:, 0]
        first = torch.where((best0 != 0) & (gain0 >= first_margin), best0, current0)
        final = first.clone()
        revised = torch.zeros_like(first, dtype=torch.bool)
        active = first != 0
        if active.any():
            logits1, _ = model(
                batch_candidates[active],
                batch_priors[active],
                first[active],
                torch.ones(int(active.sum()), device=device),
            )
            best1 = logits1.argmax(dim=1)
            current_logit = logits1.gather(1, first[active, None]).squeeze(1)
            gain1 = logits1.gather(1, best1[:, None]).squeeze(1) - current_logit
            second = torch.where(gain1 >= revise_margin, best1, first[active])
            final[active] = second
            revised[active] = second != first[active]
        if disagreement_margin is not None:
            # Only promote an action that differs from the OOF teacher when the
            # learned candidate-set residual itself supports that disagreement.
            # The teacher prior is subtracted so a large prior cannot masquerade
            # as evidence contributed by the new observation architecture.
            residual0 = logits0 - model.prior_scale * batch_priors
            final_residual = residual0.gather(1, final[:, None]).squeeze(1)
            teacher_residual = residual0.gather(1, teacher_selected[:, None]).squeeze(1)
            weak_disagreement = (final != teacher_selected) & (
                final_residual - teacher_residual < disagreement_margin
            )
            final[weak_disagreement] = teacher_selected[weak_disagreement]
            first[weak_disagreement] = teacher_selected[weak_disagreement]
            revised[weak_disagreement] = False
        if final_teacher_gain_margin is not None:
            # A final action is allowed only when the OOF reward teacher also
            # predicts enough gain over baseline. This is a label-free runtime
            # safety gate, distinct from teacher_margin (which only constructs
            # the teacher's own proposed action for disagreement handling).
            rows = torch.arange(len(final), device=device)
            final_gain = batch_priors[rows, final] - batch_priors[:, 0]
            unsafe_final = (final != 0) & (final_gain < final_teacher_gain_margin)
            final[unsafe_final] = 0
            first[unsafe_final] = 0
            revised[unsafe_final] = False
        first_parts.append(first.cpu().numpy())
        final_parts.append(final.cpu().numpy())
        revise_parts.append(revised.cpu().numpy())
    return np.concatenate(first_parts), np.concatenate(final_parts), np.concatenate(revise_parts)


def evaluate(cache, selected: np.ndarray, action_names: list[str], first=None, revised=None):
    rows = np.arange(len(selected))
    exact = cache["exact"][rows, selected].astype(bool)
    baseline = cache["exact"][:, 0].astype(bool)
    edits = cache["edit_distance"][rows, selected].astype(float)
    lengths = np.asarray([max(len(value), 1) for value in cache["targets"]], dtype=float)
    fixed = (~baseline) & exact
    broken = baseline & (~exact)
    metrics = {
        "samples": int(len(selected)),
        "exact_acc": float(exact.mean()),
        "char_acc": float(1.0 - edits.sum() / lengths.sum()),
        "fixed": int(fixed.sum()),
        "broken": int(broken.sum()),
        "net_fixes": int(fixed.sum() - broken.sum()),
        "mean_cost": float(cache["action_cost"][rows, selected].mean()),
        "stop_rate": float((selected == 0).mean()),
        "revise_rate": float(np.asarray(revised, dtype=bool).mean()) if revised is not None else 0.0,
    }
    frame = pd.DataFrame(
        {
            "image_path": cache["image_paths"],
            "target": cache["targets"],
            "first_action": [action_names[index] for index in (first if first is not None else selected)],
            "final_action": [action_names[index] for index in selected],
            "prediction": cache["predictions"][rows, selected],
            "exact": exact,
            "edit_distance": edits.astype(int),
            "baseline_exact": baseline,
            "fixed": fixed,
            "broken": broken,
        }
    )
    return metrics, frame


def select_teacher(cache, priors: np.ndarray, margin: float):
    best = priors.argmax(axis=1)
    gain = priors[np.arange(len(best)), best] - priors[:, 0]
    return np.where((best != 0) & (gain >= margin), best, 0)


def metric_key(metrics: dict):
    return metrics["exact_acc"], metrics["char_acc"], metrics["net_fixes"], -metrics["broken"], -metrics["mean_cost"]


def paired_stats(candidate_frame: pd.DataFrame, reference_frame: pd.DataFrame, seed: int):
    lengths = candidate_frame.target.astype(str).str.len().clip(lower=1).to_numpy(dtype=float)
    candidate_char = 1.0 - candidate_frame.edit_distance.to_numpy(dtype=float) / lengths
    reference_char = 1.0 - reference_frame.edit_distance.to_numpy(dtype=float) / lengths
    exact_candidate = candidate_frame.exact.to_numpy(dtype=bool)
    exact_reference = reference_frame.exact.to_numpy(dtype=bool)
    return {
        "paired_bootstrap": paired_bootstrap(exact_candidate, exact_reference, candidate_char, reference_char, seed),
        "mcnemar": mcnemar_exact(exact_candidate, exact_reference),
    }


def run(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    if RL_ROOT not in output_dir.parents and output_dir != RL_ROOT:
        raise ValueError("All RL artifacts must remain inside reinforcement_learning")
    if (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"Locked Phase 6 run already exists at {output_dir}; choose a new output directory and protocol seed"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dir = Path(args.trajectory_cache).resolve()
    candidate_dir = Path(args.candidate_cache).resolve()
    action_payload = np.load(candidate_dir / "train_candidate_features.npz", allow_pickle=False)
    action_names = action_payload["action_names"].astype(str).tolist()
    if not action_names or action_names[0] not in {"stop_baseline", "baseline"}:
        raise ValueError("Action 0 must be the baseline/STOP action")
    train_cache = load_trajectory_cache(cache_dir, "train", action_names)
    val_cache = load_trajectory_cache(cache_dir, "val", action_names)
    train_candidates = load_candidate_features(candidate_dir / "train_candidate_features.npz", train_cache["image_paths"], action_names)
    val_candidates = load_candidate_features(candidate_dir / "val_candidate_features.npz", val_cache["image_paths"], action_names)
    if args.candidate_ocr_strings:
        train_candidates = np.concatenate(
            (train_candidates, candidate_ocr_features(train_cache)), axis=2
        )
        val_candidates = np.concatenate((val_candidates, candidate_ocr_features(val_cache)), axis=2)

    development_mask, holdout_mask = stable_group_holdout(train_cache["targets"], args.holdout_fraction, args.seed)
    if set(train_cache["targets"][development_mask]) & set(train_cache["targets"][holdout_mask]):
        raise RuntimeError("Target leakage detected between development and holdout")
    dev_cache = cache_subset(train_cache, development_mask)
    holdout_cache = cache_subset(train_cache, holdout_mask)
    dev_candidates_raw = train_candidates[development_mask]
    holdout_candidates_raw = train_candidates[holdout_mask]

    normalized = normalize(
        dev_candidates_raw.reshape(-1, dev_candidates_raw.shape[-1]),
        val_candidates.reshape(-1, val_candidates.shape[-1]),
        holdout_candidates_raw.reshape(-1, holdout_candidates_raw.shape[-1]),
    )
    dev_flat, val_flat, holdout_flat, candidate_mean, candidate_std = normalized
    dev_candidates_np = dev_flat.reshape(dev_candidates_raw.shape).astype(np.float32)
    val_candidates_np = val_flat.reshape(val_candidates.shape).astype(np.float32)
    holdout_candidates_np = holdout_flat.reshape(holdout_candidates_raw.shape).astype(np.float32)

    teacher_normalized = normalize(
        dev_candidates_raw[:, 0], val_candidates[:, 0], holdout_candidates_raw[:, 0]
    )
    dev_teacher_x, val_teacher_x, holdout_teacher_x, teacher_mean, teacher_std = teacher_normalized
    oof_prior, full_teacher, fold_ids = make_oof_teacher(dev_teacher_x.astype(np.float32), dev_cache, args)
    val_prior = teacher_predict(full_teacher, val_teacher_x.astype(np.float32), args.device_obj)
    holdout_prior = teacher_predict(full_teacher, holdout_teacher_x.astype(np.float32), args.device_obj)

    teacher_margin_candidates = [float(value) for value in args.margins.split(",")]
    teacher_trials = []
    for margin in teacher_margin_candidates:
        selected = select_teacher(val_cache, val_prior, margin)
        metrics, _ = evaluate(val_cache, selected, action_names)
        teacher_trials.append((metrics, margin))
    teacher_val_metrics, teacher_margin = max(teacher_trials, key=lambda value: metric_key(value[0]))
    teacher_holdout_selected = select_teacher(holdout_cache, holdout_prior, teacher_margin)
    teacher_holdout_metrics, teacher_holdout_frame = evaluate(holdout_cache, teacher_holdout_selected, action_names)

    dev_candidates = torch.from_numpy(dev_candidates_np).to(args.device_obj)
    val_candidates_tensor = torch.from_numpy(val_candidates_np).to(args.device_obj)
    holdout_candidates = torch.from_numpy(holdout_candidates_np).to(args.device_obj)
    dev_prior = torch.from_numpy(oof_prior).to(args.device_obj)
    val_prior_tensor = torch.from_numpy(val_prior).to(args.device_obj)
    holdout_prior_tensor = torch.from_numpy(holdout_prior).to(args.device_obj)
    model = CandidateSetActorCritic(
        dev_candidates.shape[-1],
        len(action_names),
        args.hidden_dim,
        args.attention_heads,
        args.attention_layers,
        args.dropout,
        args.prior_scale,
    ).to(args.device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = behavior_clone(model, optimizer, dev_candidates, dev_prior, dev_cache, args)
    margins = teacher_margin_candidates
    disagreement_margins = [float(value) for value in args.disagreement_margins.split(",")]
    final_teacher_gain_margins = (
        [None]
        if not args.final_teacher_gain_margins.strip()
        else [float(value) for value in args.final_teacher_gain_margins.split(",")]
    )
    best_key = (-1.0, -1.0, -10**9, -10**9, -10**9)
    best_state = None
    best_row = None

    def consider(epoch: int, stage: str, loss: float):
        nonlocal best_key, best_state, best_row
        trials = []
        for first_margin in margins:
            for revise_margin in margins:
                for disagreement_margin in disagreement_margins:
                    for final_teacher_gain_margin in final_teacher_gain_margins:
                        first, selected, revised = policy_selection(
                            model,
                            val_candidates_tensor,
                            val_prior_tensor,
                            first_margin,
                            revise_margin,
                            args.device_obj,
                            teacher_margin,
                            disagreement_margin,
                            final_teacher_gain_margin,
                        )
                        metrics, _ = evaluate(val_cache, selected, action_names, first, revised)
                        trials.append(
                            (
                                metrics,
                                first_margin,
                                revise_margin,
                                disagreement_margin,
                                final_teacher_gain_margin,
                            )
                        )
        metrics, first_margin, revise_margin, disagreement_margin, final_teacher_gain_margin = max(
            trials, key=lambda value: metric_key(value[0])
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            "loss": loss,
            "first_margin": first_margin,
            "revise_margin": revise_margin,
            "disagreement_margin": disagreement_margin,
            "final_teacher_gain_margin": final_teacher_gain_margin,
            **metrics,
        }
        history.append(row)
        if metric_key(metrics) > best_key:
            best_key = metric_key(metrics)
            best_row = row
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    consider(0, "bc_locked", history[-1]["loss"] if history else 0.0)
    for epoch in range(1, args.ppo_epochs + 1):
        rollout = collect_rollout(model, dev_candidates, dev_prior, dev_cache, args)
        model.train()
        loss = ppo_update(model, optimizer, dev_candidates, dev_prior, rollout, args)
        consider(epoch, "ppo", loss)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(history[-1], ensure_ascii=False))
    model.load_state_dict(best_state)

    val_first, val_selected, val_revised = policy_selection(
        model,
        val_candidates_tensor,
        val_prior_tensor,
        best_row["first_margin"],
        best_row["revise_margin"],
        args.device_obj,
        teacher_margin,
        best_row["disagreement_margin"],
        best_row["final_teacher_gain_margin"],
    )
    val_metrics, val_frame = evaluate(val_cache, val_selected, action_names, val_first, val_revised)
    holdout_first, holdout_selected, holdout_revised = policy_selection(
        model,
        holdout_candidates,
        holdout_prior_tensor,
        best_row["first_margin"],
        best_row["revise_margin"],
        args.device_obj,
        teacher_margin,
        best_row["disagreement_margin"],
        best_row["final_teacher_gain_margin"],
    )
    holdout_metrics, holdout_frame = evaluate(
        holdout_cache, holdout_selected, action_names, holdout_first, holdout_revised
    )
    stats = paired_stats(holdout_frame, teacher_holdout_frame, args.seed)
    gate = improvement_gate(holdout_metrics, stats)
    checkpoint = output_dir / "best_candidate_oof_ppo.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "candidate_dim": int(dev_candidates.shape[-1]),
                "action_count": len(action_names),
                "hidden_dim": args.hidden_dim,
                "heads": args.attention_heads,
                "layers": args.attention_layers,
                "dropout": args.dropout,
                "prior_scale": args.prior_scale,
            },
            "teacher_state_dict": full_teacher.state_dict(),
            "teacher_config": {"input_dim": int(dev_teacher_x.shape[1]), "hidden_dim": args.teacher_hidden, "dropout": args.teacher_dropout},
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "teacher_mean": teacher_mean,
            "teacher_std": teacher_std,
            "action_names": action_names,
            "first_margin": best_row["first_margin"],
            "revise_margin": best_row["revise_margin"],
            "disagreement_margin": best_row["disagreement_margin"],
            "teacher_margin": teacher_margin,
            "final_teacher_gain_margin": best_row["final_teacher_gain_margin"],
            "seed": args.seed,
            "algorithm": "candidate_set_ppo_with_oof_teacher_residual",
            "candidate_ocr_strings": args.candidate_ocr_strings,
            "test_used": False,
            "holdout_used_for_selection": False,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {"image_path": dev_cache["image_paths"], "target": dev_cache["targets"], "oof_fold": fold_ids}
    ).to_csv(output_dir / "development_oof_assignments.csv", index=False)
    val_frame.to_csv(output_dir / "validation_selections.csv", index=False)
    holdout_frame.to_csv(output_dir / "holdout_selections.csv", index=False)
    teacher_holdout_frame.to_csv(output_dir / "holdout_teacher_selections.csv", index=False)
    protocol = {
        "seed": args.seed,
        "group_key": "normalized target/label",
        "development_samples": int(development_mask.sum()),
        "holdout_samples": int(holdout_mask.sum()),
        "development_groups": int(len(set(train_cache["targets"][development_mask]))),
        "holdout_groups": int(len(set(train_cache["targets"][holdout_mask]))),
        "group_overlap": 0,
        "oof_folds": args.oof_folds,
        "validation_role": "checkpoint_and_margin_selection; previously audited validation",
        "holdout_role": "single locked run evaluation; excluded from teacher, BC, PPO and selection",
        "audited_test_loaded": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "algorithm": "candidate_set_ppo_with_oof_teacher_residual",
        "best_validation_checkpoint": best_row,
        "validation_policy": val_metrics,
        "validation_teacher": teacher_val_metrics,
        "validation_delta_exact_vs_teacher": val_metrics["exact_acc"] - teacher_val_metrics["exact_acc"],
        "holdout_policy": holdout_metrics,
        "holdout_teacher": teacher_holdout_metrics,
        "holdout_statistics_vs_teacher": stats,
        "formal_improvement_gate": gate,
        "promotion_status": "eligible" if gate["passed"] else "experimental_not_promoted",
        "protocol": protocol,
        "checkpoint": str(checkpoint),
        "test_used": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-cache", default=str(ROOT / "outputs/rl_restoration/trajectory_cache"))
    parser.add_argument("--candidate-cache", default=str(HERE / "results/candidate_cache"))
    parser.add_argument("--output-dir", default=str(HERE / "results/run_residual_seed_123"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--teacher-hidden", type=int, default=256)
    parser.add_argument("--teacher-dropout", type=float, default=0.10)
    parser.add_argument("--teacher-lr", type=float, default=3e-4)
    parser.add_argument("--teacher-ranking-weight", type=float, default=0.1)
    parser.add_argument("--reward-temperature", type=float, default=0.08)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=50)
    parser.add_argument("--rollout-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-scale", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--step-cost", type=float, default=0.001)
    parser.add_argument("--revisit-cost", type=float, default=0.002)
    parser.add_argument("--margins", default="0,0.05,0.1,0.2")
    parser.add_argument("--disagreement-margins", default="-2,-0.5,0,0.1,0.2")
    parser.add_argument(
        "--final-teacher-gain-margins",
        default="",
        help=(
            "Optional comma-separated OOF teacher gain thresholds applied to the PPO final action. "
            "Empty preserves the historical architecture without this safety gate."
        ),
    )
    parser.add_argument(
        "--no-candidate-ocr-strings",
        action="store_false",
        dest="candidate_ocr_strings",
        help="Ablation: remove per-candidate OCR string and consensus tokens.",
    )
    parser.set_defaults(candidate_ocr_strings=True)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

"""A2C training loop for the PixelRL-style deblurring agent.

Synchronous, batch-parallel A2C variant of the asynchronous A3C used in the
original paper (Furuta et al.) -- same MDP (multi-step, per-pixel action,
dense reward), simpler single-process training loop which is enough at this
dataset scale and avoids multiprocessing complexity.

Two additions beyond a minimal PixelRL port:
  1. Reward Map Convolution (RMC, see model.RewardMapConv) -- the paper's own
     reported main performance driver, implemented via the n-step return
     recursion R_t = r_t + gamma * conv(R_{t+1}, w).
  2. An optional OCR-aware terminal reward: since pixel-fidelity reward alone
     was observed to *hurt* downstream OCR exact-match accuracy (agent learns
     to look closer to the clean image without necessarily helping PARSeq
     read it), we add a bonus at the last step equal to the drop in
     character-error-rate between the raw blurred input and the restored
     output, measured by the frozen fine-tuned PARSeq model.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rl_deblur import env
from rl_deblur.model import FCNActorCritic
from rl_deblur.ocr_utils import DEFAULT_OCR_CKPT, edit_distance, load_ocr_model, ocr_predict_with_confidence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "outputs" / "rl_deblur" / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rl_deblur"


@dataclass
class TrainConfig:
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    epochs: int = 15
    batch_size: int = 32
    num_steps: int = 5
    gamma: float = 0.95
    lr: float = 1e-4
    entropy_coef: float = 0.0002
    value_coef: float = 0.5
    channels: int = 64
    rmc_kernel_size: int = 9
    cer_reward_weight: float = 0.0
    logconf_reward_weight: float = 0.0
    ocr_checkpoint: str = str(DEFAULT_OCR_CKPT)
    resume_checkpoint: str | None = None
    seed: int = 42
    limit_train: int | None = None
    limit_val: int | None = None


class BlurPairDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, limit: int | None = None):
        self.dataset_dir = dataset_dir
        self.frame = pd.read_csv(dataset_dir / f"{split}.csv")
        if limit is not None:
            self.frame = self.frame.head(int(limit)).copy().reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        clean = np.asarray(Image.open(self.dataset_dir / row["clean_path"]), dtype=np.float32)
        blurred = np.asarray(Image.open(self.dataset_dir / row["blurred_path"]), dtype=np.float32)
        return blurred, clean, str(row["label"])


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-8:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def rollout(
    model: FCNActorCritic,
    blurred: torch.Tensor,
    clean: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    greedy: bool = False,
    labels: list[str] | None = None,
    ocr_model=None,
    ocr_transform=None,
):
    """Runs a fixed-length episode and returns everything needed for the A2C update.

    Per-step reward mixes pixel fidelity with downstream OCR quality:
        reward_t = (SE_t - SE_{t+1}) / 255^2                          [env.step, always on]
                 + cer_reward_weight   * (CER_t - CER_{t+1})           [beta]
                 + logconf_reward_weight * (logconf_{t+1} - logconf_t) [gamma]
    computed every step (not just at the episode's end), so restoring pixel
    fidelity without helping OCR -- or vice versa -- is penalized/rewarded
    immediately instead of only once at the end of the episode. This directly
    targets the reward-misalignment problem found during evaluation (RL
    improved PSNR/SSIM but *hurt* OCR exact-match accuracy versus the raw
    blurred input, because pixel fidelity and OCR correctness are correlated
    but not identical objectives).

    Returns: final_state, log_probs, values, entropies, rewards, bootstrap_value, ocr_reward_mean
    """
    state = blurred.numpy().copy()
    clean_np = clean.numpy()

    ocr_active = (cfg.cer_reward_weight > 0 or cfg.logconf_reward_weight > 0) and labels is not None and ocr_model is not None
    if ocr_active:
        preds, logconfs = ocr_predict_with_confidence(ocr_model, ocr_transform, np.clip(state, 0, 255).astype(np.uint8), device)
        cer_prev = np.array([edit_distance(p, l) / max(len(l), 1) for p, l in zip(preds, labels)], dtype=np.float32)
        logconf_prev = np.array(logconfs, dtype=np.float32)

    log_probs, values, entropies, rewards = [], [], [], []
    ocr_reward_sum = 0.0
    for _ in range(cfg.num_steps):
        state_t = torch.from_numpy(state / 255.0).unsqueeze(1).float().to(device)
        logits, value = model(state_t)
        probs = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)  # (B,H,W,A)
        dist = torch.distributions.Categorical(probs=probs)
        action_map = probs.argmax(dim=-1) if greedy else dist.sample()
        log_prob = dist.log_prob(action_map)
        entropy = dist.entropy()

        next_state, reward = env.step(state, action_map.detach().cpu().numpy(), clean_np)

        if ocr_active:
            preds, logconfs = ocr_predict_with_confidence(ocr_model, ocr_transform, np.clip(next_state, 0, 255).astype(np.uint8), device)
            cer_cur = np.array([edit_distance(p, l) / max(len(l), 1) for p, l in zip(preds, labels)], dtype=np.float32)
            logconf_cur = np.array(logconfs, dtype=np.float32)

            ocr_term = (
                cfg.cer_reward_weight * (cer_prev - cer_cur)
                + cfg.logconf_reward_weight * (logconf_cur - logconf_prev)
            )
            ocr_reward_sum += float(ocr_term.mean())
            reward = reward + ocr_term[:, None, None]
            cer_prev, logconf_prev = cer_cur, logconf_cur

        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)
        rewards.append(torch.from_numpy(reward).to(device))
        state = next_state

    # Each image gets a genuinely fixed T-step episode with no continuation --
    # there is no more reward to earn for this image once the episode ends,
    # so the correct n-step target is a true terminal return (bootstrap 0),
    # not V(s_T). Bootstrapping with the (early on, poorly calibrated) value
    # estimate was tried and empirically destabilized training: it injects an
    # arbitrary-signed extra term into every step's return via the backward
    # recursion, which measurably collapsed greedy validation PSNR gain from
    # ~0 to strongly negative within a few epochs in local smoke tests.
    bootstrap_value = torch.zeros_like(rewards[-1])

    return state, log_probs, values, entropies, rewards, bootstrap_value, ocr_reward_sum / cfg.num_steps


def compute_a2c_loss(model: FCNActorCritic, log_probs, values, entropies, rewards, bootstrap_value, gamma: float, value_coef: float, entropy_coef: float, rmc_coef: float = 0.1):
    """A2C + Reward Map Convolution loss.

    `ret` (the n-step return, Eq. 13 in the paper) is built by recursively
    convolving future returns with the *trainable* RMC kernel `model.rmc`, so
    it carries a gradient path to the kernel's parameters. Using that
    non-detached `ret` as a regression target for both the policy advantage
    and the value loss creates a circular objective -- the kernel can then
    "cheat" by reshaping itself to match whatever the value net currently
    predicts, rather than learning a meaningful spatial credit-assignment
    pattern (empirically this collapsed training: greedy validation PSNR
    gain went strongly negative within a few epochs). So the kernel is
    trained through its own decoupled term instead:
      - `value_loss` regresses the value net toward a stop-gradient `ret`
        (standard TD target, no gradient into the RMC kernel).
      - `rmc_loss` regresses the (non-detached) `ret` toward a stop-gradient
        `value`, which trains only the RMC kernel to keep the convolved
        bootstrap consistent with the value net's current (fixed) estimate.
    The policy advantage always uses a fully detached `ret`, so the RMC
    kernel cannot influence policy updates except by genuinely reshaping the
    future returns used to train the value net.
    """
    returns = []
    R = bootstrap_value.detach()
    for r in reversed(rewards):
        R = r + gamma * model.rmc(R)
        returns.insert(0, R)

    policy_loss = torch.zeros((), device=rewards[0].device)
    value_loss = torch.zeros((), device=rewards[0].device)
    rmc_loss = torch.zeros((), device=rewards[0].device)
    entropy_loss = torch.zeros((), device=rewards[0].device)
    for log_prob, value, entropy, ret in zip(log_probs, values, entropies, returns):
        advantage = (ret.detach() - value).detach()
        policy_loss = policy_loss - (log_prob * advantage).mean()
        value_loss = value_loss + (ret.detach() - value).pow(2).mean()
        rmc_loss = rmc_loss + (ret - value.detach()).pow(2).mean()
        entropy_loss = entropy_loss - entropy.mean()

    # Average over steps instead of summing, so entropy_coef/value_coef mean
    # the same thing regardless of num_steps (a raw T-step sum made the
    # entropy term ~1000x larger than the policy-gradient term at T=5 with
    # this task's tiny per-pixel reward scale, and completely dominated
    # training -- see the "Sudden collapse" note in fit()).
    n = len(rewards)
    policy_loss, value_loss, rmc_loss, entropy_loss = (t / n for t in (policy_loss, value_loss, rmc_loss, entropy_loss))

    total = policy_loss + value_coef * value_loss + rmc_coef * rmc_loss + entropy_coef * entropy_loss
    return total, policy_loss.item(), value_loss.item(), entropy_loss.item()


@torch.no_grad()
def evaluate(model: FCNActorCritic, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> dict:
    model.eval()
    psnr_before, psnr_after = [], []
    for blurred, clean, _labels in loader:
        final_state, *_ = rollout(model, blurred, clean, cfg, device, greedy=True)
        clean_np = clean.numpy()
        for b in range(blurred.shape[0]):
            psnr_before.append(psnr(blurred[b].numpy(), clean_np[b]))
            psnr_after.append(psnr(final_state[b], clean_np[b]))
    model.train()
    return {
        "psnr_before": float(np.mean(psnr_before)),
        "psnr_after": float(np.mean(psnr_after)),
        "psnr_gain": float(np.mean(psnr_after) - np.mean(psnr_before)),
    }


def fit(cfg: TrainConfig, device: str | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    dataset_dir = Path(cfg.dataset_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BlurPairDataset(dataset_dir, "train", limit=cfg.limit_train)
    val_ds = BlurPairDataset(dataset_dir, "val", limit=cfg.limit_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = FCNActorCritic(channels=cfg.channels, rmc_kernel_size=cfg.rmc_kernel_size).to(device)
    resume_info = None
    if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_checkpoint not found: {resume_path}")

        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        resume_cfg = resume_ckpt.get("config", {})
        resume_channels = resume_cfg.get("channels", cfg.channels)
        resume_rmc_kernel_size = resume_cfg.get("rmc_kernel_size", cfg.rmc_kernel_size)
        if resume_channels != cfg.channels or resume_rmc_kernel_size != cfg.rmc_kernel_size:
            raise ValueError(
                "Resume checkpoint architecture does not match current config: "
                f"checkpoint channels={resume_channels}, rmc_kernel_size={resume_rmc_kernel_size}; "
                f"current channels={cfg.channels}, rmc_kernel_size={cfg.rmc_kernel_size}"
            )

        model.load_state_dict(resume_ckpt["model_state_dict"])
        resume_info = {
            "checkpoint": str(resume_path),
            "epoch": resume_ckpt.get("epoch"),
            "config": resume_cfg,
            "val_metrics": resume_ckpt.get("val_metrics"),
        }
        print(f"Resumed agent weights from {resume_path} (epoch={resume_info['epoch']}).")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ocr_model, ocr_transform = (None, None)
    if cfg.cer_reward_weight > 0 or cfg.logconf_reward_weight > 0:
        ocr_model, ocr_transform = load_ocr_model(device, cfg.ocr_checkpoint)
        print(f"Loaded OCR reward model from {cfg.ocr_checkpoint} "
              f"(cer_weight={cfg.cer_reward_weight}, logconf_weight={cfg.logconf_reward_weight})")

    base_val = evaluate(model, val_loader, cfg, device)
    print(f"[epoch 0 / before training] val psnr_before={base_val['psnr_before']:.3f} "
          f"psnr_after={base_val['psnr_after']:.3f} gain={base_val['psnr_gain']:.3f}")

    history = []
    best_gain = -1e9
    best_path = output_dir / "checkpoints" / "best_deblur_agent.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        model.train()
        epoch_loss, epoch_reward, epoch_ocr_bonus, n_batches = 0.0, 0.0, 0.0, 0
        for blurred, clean, labels in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            _, log_probs, values, entropies, rewards, bootstrap_value, ocr_bonus = rollout(
                model, blurred, clean, cfg, device, greedy=False,
                labels=labels, ocr_model=ocr_model, ocr_transform=ocr_transform,
            )
            loss, p_loss, v_loss, e_loss = compute_a2c_loss(
                model, log_probs, values, entropies, rewards, bootstrap_value,
                cfg.gamma, cfg.value_coef, cfg.entropy_coef,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_reward += float(torch.stack(rewards).sum(dim=0).mean().item())
            epoch_ocr_bonus += ocr_bonus
            n_batches += 1

        val_metrics = evaluate(model, val_loader, cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_mean_episode_reward": epoch_reward / max(n_batches, 1),
            "train_mean_ocr_bonus": epoch_ocr_bonus / max(n_batches, 1),
            "seconds": time.time() - start,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(f"[epoch {epoch}] loss={row['train_loss']:.4f} "
              f"val_psnr_gain={val_metrics['psnr_gain']:.3f} "
              f"ocr_bonus={row['train_mean_ocr_bonus']:.4f} "
              f"({row['seconds']:.1f}s)")

        if val_metrics["psnr_gain"] > best_gain:
            best_gain = val_metrics["psnr_gain"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "resume_info": resume_info,
            }, best_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "resume_info": resume_info,
        "base_val_metrics": base_val,
        "best_val_psnr_gain": best_gain,
        "best_checkpoint": str(best_path),
        "final_val_metrics": history[-1] if history else None,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PixelRL-style deblurring agent with A2C + RMC.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--rmc-kernel-size", type=int, default=9)
    parser.add_argument("--cer-reward-weight", type=float, default=0.0, help="beta: weight on per-step CER reduction")
    parser.add_argument("--logconf-reward-weight", type=float, default=0.0, help="gamma: weight on per-step OCR log-confidence increase")
    parser.add_argument("--ocr-checkpoint", default=str(DEFAULT_OCR_CKPT))
    parser.add_argument("--resume-checkpoint", default=None, help="Warm-start agent weights from an existing RL checkpoint")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        gamma=args.gamma,
        lr=args.lr,
        channels=args.channels,
        rmc_kernel_size=args.rmc_kernel_size,
        cer_reward_weight=args.cer_reward_weight,
        logconf_reward_weight=args.logconf_reward_weight,
        ocr_checkpoint=args.ocr_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
    summary = fit(cfg, device=args.device or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

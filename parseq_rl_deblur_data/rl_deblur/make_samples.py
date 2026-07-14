"""Render a before/after visual grid (blurred | classical | RL-restored | clean) for the report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rl_deblur import env
from rl_deblur.evaluate import DEFAULT_AGENT_CKPT, DEFAULT_DATASET_DIR, classical_unsharp_baseline, rl_restore_batch
from rl_deblur.model import FCNActorCritic
from rl_deblur.train import BlurPairDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "rl_deblur" / "samples" / "before_after_grid.png"
CELL_W, CELL_H = 128, 32
PAD = 8
LABEL_H = 16


def build_grid(rows: list[list[np.ndarray]], col_titles: list[str]) -> Image.Image:
    n_rows = len(rows)
    n_cols = len(col_titles)
    width = n_cols * (CELL_W + PAD) + PAD
    height = LABEL_H + n_rows * (CELL_H + PAD) + PAD
    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for c, title in enumerate(col_titles):
        x = PAD + c * (CELL_W + PAD)
        draw.text((x, 0), title, fill=(255, 255, 255))
    for r, row in enumerate(rows):
        y = LABEL_H + PAD + r * (CELL_H + PAD)
        for c, img in enumerate(row):
            x = PAD + c * (CELL_W + PAD)
            canvas.paste(Image.fromarray(img).convert("RGB"), (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--agent-checkpoint", default=str(DEFAULT_AGENT_CKPT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    agent = FCNActorCritic(channels=cfg["channels"], rmc_kernel_size=cfg.get("rmc_kernel_size", 9)).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    ds = BlurPairDataset(Path(args.dataset_dir), "test")
    idxs = np.linspace(0, len(ds) - 1, args.num_samples).astype(int)
    blurred = np.stack([ds[i][0] for i in idxs])
    clean = np.stack([ds[i][1] for i in idxs])

    classical = np.stack([classical_unsharp_baseline(b.astype(np.uint8)) for b in blurred])
    rl_out = rl_restore_batch(agent, blurred, cfg["num_steps"], device)

    rows = []
    for i in range(len(idxs)):
        rows.append([
            np.clip(blurred[i], 0, 255).astype(np.uint8),
            classical[i],
            np.clip(rl_out[i], 0, 255).astype(np.uint8),
            np.clip(clean[i], 0, 255).astype(np.uint8),
        ])

    grid = build_grid(rows, ["blurred (input)", "classical unsharp", "RL agent (ours)", "clean (ground truth)"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved sample grid to {out_path}")


if __name__ == "__main__":
    main()

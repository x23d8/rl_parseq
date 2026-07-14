"""Synthesize a paired (blurred, clean) plate-crop dataset for RL-based deblurring.

The project has no real blurry plate images, so we start from the clean plate
crops under ``color_filtered/`` and synthetically degrade them with Gaussian /
motion / defocus blur. Ground truth (the original clean pixels) is kept so the
RL environment can compute a dense per-pixel reward.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["color_filtered/blue", "color_filtered/other", "color_filtered/yellow"]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rl_deblur" / "dataset"
CANVAS_SIZE = (128, 32)  # (width, height), matches PARSeq img_size=(32,128) convention


@dataclass
class BlurSpec:
    kind: str
    strength: float


def list_clean_samples() -> list[tuple[Path, str]]:
    samples = []
    for rel in SOURCE_DIRS:
        folder = REPO_ROOT / rel
        labels_path = folder / "labels.txt"
        if not labels_path.exists():
            continue
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            fname, label = line.split("\t")
            img_path = folder / fname
            if img_path.exists():
                samples.append((img_path, label))
    return samples


def random_blur_spec(rng: random.Random) -> BlurSpec:
    """Mild, recoverable degradation levels.

    The original ranges (gaussian sigma up to 2.5, defocus sigma up to 4.5,
    motion length up to 13) destroy enough high-frequency text detail on a
    32x128 crop that no restoration policy -- RL or otherwise -- can recover
    the character shapes. These lighter ranges keep stroke edges partially
    intact so the reward signal (pixel error reduction) has something to
    learn from.
    """
    kind = rng.choice(["gaussian", "motion", "defocus"])
    if kind == "gaussian":
        strength = rng.uniform(0.8, 1.8)
    elif kind == "motion":
        strength = rng.uniform(4, 9)
    else:  # defocus (larger gaussian to emulate out-of-focus blur)
        strength = rng.uniform(1.5, 3.0)
    return BlurSpec(kind=kind, strength=strength)


def apply_blur(gray: np.ndarray, spec: BlurSpec) -> np.ndarray:
    if spec.kind == "gaussian":
        ksize = int(spec.strength * 3) | 1  # force odd
        return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=spec.strength)
    if spec.kind == "defocus":
        ksize = int(spec.strength * 3) | 1
        return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=spec.strength)
    if spec.kind == "motion":
        length = max(int(spec.strength), 3)
        angle = random.uniform(0, 180)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        rot_mat = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
        kernel /= max(kernel.sum(), 1e-6)
        return cv2.filter2D(gray, -1, kernel)
    raise ValueError(spec.kind)


def to_canvas_gray(image: Image.Image) -> np.ndarray:
    resized = image.convert("L").resize(CANVAS_SIZE, Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def build_dataset(output_dir: Path, seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    rng = random.Random(seed)
    samples = list_clean_samples()
    rng.shuffle(samples)

    n = len(samples)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    split_for_index = (
        ["val"] * n_val + ["test"] * n_test + ["train"] * (n - n_val - n_test)
    )
    rng.shuffle(split_for_index)

    for split in ("train", "val", "test"):
        (output_dir / split / "clean").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "blurred").mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, ((img_path, label), split) in enumerate(tqdm(list(zip(samples, split_for_index)), desc="synthesizing blur")):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        clean = to_canvas_gray(image)
        spec = random_blur_spec(rng)
        blurred = apply_blur(clean, spec)

        stem = f"{idx:06d}_{img_path.stem}"
        clean_path = output_dir / split / "clean" / f"{stem}.png"
        blurred_path = output_dir / split / "blurred" / f"{stem}.png"
        Image.fromarray(clean).save(clean_path)
        Image.fromarray(blurred).save(blurred_path)

        rows.append(
            {
                "split": split,
                "source_path": str(img_path.relative_to(REPO_ROOT)),
                "clean_path": str(clean_path.relative_to(output_dir)),
                "blurred_path": str(blurred_path.relative_to(output_dir)),
                "label": label,
                "blur_kind": spec.kind,
                "blur_strength": round(spec.strength, 3),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    for split in ("train", "val", "test"):
        manifest[manifest["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)

    print(manifest["split"].value_counts())
    print(manifest["blur_kind"].value_counts())
    print(f"Saved dataset to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic blur dataset for RL deblurring.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(Path(args.output_dir), seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)


if __name__ == "__main__":
    main()

"""Build Phase-3-aligned paired degradation datasets for fair RL evaluation.

Every source keeps its original train/val/test assignment.  Each source yields
one blur sample (Gaussian, motion, or defocus) and one non-blur degradation
(noise, low light, or low contrast).  The blur subset is directly consumable
by the PixelRL trainer; the combined manifest is used by the Bandit/PPO cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs/phase3_controlled_aug_full_frozen_eval/dataset_manifest.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dataset"
CANVAS = (128, 32)
BLUR_KINDS = ("gaussian", "motion", "defocus")
AUX_KINDS = ("gaussian_noise", "low_light", "low_contrast")


def stable_seed(seed: int, split: str, index: int, family: str) -> int:
    payload = f"{seed}|{split}|{index}|{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(CANVAS, Image.Resampling.BICUBIC), dtype=np.uint8)


def motion_blur(gray: np.ndarray, length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(gray, -1, kernel)


def apply_blur(gray: np.ndarray, kind: str, rng: random.Random) -> tuple[np.ndarray, float]:
    if kind == "gaussian":
        strength = rng.uniform(0.8, 1.8)
        ksize = max(3, int(math.ceil(strength * 3)) | 1)
        return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=strength), strength
    if kind == "defocus":
        strength = rng.uniform(1.5, 3.0)
        ksize = max(5, int(math.ceil(strength * 3)) | 1)
        return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=strength), strength
    if kind == "motion":
        strength = rng.uniform(4.0, 9.0)
        length = max(3, int(round(strength)))
        return motion_blur(gray, length, rng.uniform(0.0, 180.0)), strength
    raise KeyError(kind)


def apply_aux(gray: np.ndarray, kind: str, rng: random.Random) -> tuple[np.ndarray, float]:
    work = gray.astype(np.float32)
    if kind == "gaussian_noise":
        sigma = rng.uniform(8.0, 24.0)
        generator = np.random.default_rng(rng.randrange(2**32))
        result = work + generator.normal(0.0, sigma, size=work.shape)
        return np.clip(result, 0, 255).astype(np.uint8), sigma
    if kind == "low_light":
        gamma = rng.uniform(1.5, 2.4)
        scale = rng.uniform(0.55, 0.80)
        result = 255.0 * np.power(work / 255.0, gamma) * scale
        return np.clip(result, 0, 255).astype(np.uint8), gamma
    if kind == "low_contrast":
        alpha = rng.uniform(0.35, 0.65)
        center = rng.uniform(105.0, 150.0)
        result = center + alpha * (work - center)
        return np.clip(result, 0, 255).astype(np.uint8), alpha
    raise KeyError(kind)


def build(manifest_path: Path, output_dir: Path, seed: int) -> dict:
    source = pd.read_csv(manifest_path)
    required = {"image_path", "label", "split"}
    if not required.issubset(source.columns):
        raise ValueError(f"{manifest_path} must contain {sorted(required)}")
    source["split"] = source["split"].astype(str).str.lower()
    if not set(source.split).issubset({"train", "val", "test"}):
        raise ValueError("Only train/val/test splits are supported")
    if source.image_path.duplicated().any():
        raise ValueError("Phase 3 manifest contains duplicate image paths")
    missing = [value for value in source.image_path if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Phase 3 source images are missing")

    blur_root = output_dir / "paired_blur"
    multi_root = output_dir / "multi"
    for split in ("train", "val", "test"):
        for folder in ("clean", "blurred"):
            (blur_root / split / folder).mkdir(parents=True, exist_ok=True)
        (multi_root / split / "auxiliary").mkdir(parents=True, exist_ok=True)

    blur_rows: list[dict] = []
    multi_rows: list[dict] = []
    split_counters = {"train": 0, "val": 0, "test": 0}
    for global_index, row in tqdm(source.reset_index(drop=True).iterrows(), total=len(source), desc="degrading"):
        split = str(row["split"])
        local_index = split_counters[split]
        split_counters[split] += 1
        clean = load_gray(Path(row["image_path"]))
        stem = f"{local_index:06d}_{Path(row['image_path']).stem}"
        clean_path = blur_root / split / "clean" / f"{stem}.png"
        Image.fromarray(clean).save(clean_path)

        blur_kind = BLUR_KINDS[local_index % len(BLUR_KINDS)]
        blur_rng = random.Random(stable_seed(seed, split, local_index, "blur"))
        blurred, blur_strength = apply_blur(clean, blur_kind, blur_rng)
        blurred_path = blur_root / split / "blurred" / f"{stem}.png"
        Image.fromarray(blurred).save(blurred_path)

        relative_clean = clean_path.relative_to(blur_root)
        relative_blurred = blurred_path.relative_to(blur_root)
        blur_row = {
            "split": split,
            "source_path": str(row["image_path"]),
            "clean_path": str(relative_clean),
            "blurred_path": str(relative_blurred),
            "label": str(row["label"]),
            "blur_kind": blur_kind,
            "blur_strength": round(float(blur_strength), 6),
        }
        blur_rows.append(blur_row)
        common = {
            "source_split": split,
            "source_path": str(row["image_path"]),
            "clean_path": str(clean_path.resolve()),
            "label": str(row["label"]),
        }
        multi_rows.append(
            {
                **common,
                "sample_id": f"{split}_{local_index:06d}_blur",
                "split": split,
                "image_path": str(blurred_path.resolve()),
                "degradation_family": "blur",
                "degradation_kind": blur_kind,
                "degradation_strength": round(float(blur_strength), 6),
            }
        )

        aux_kind = AUX_KINDS[local_index % len(AUX_KINDS)]
        aux_rng = random.Random(stable_seed(seed, split, local_index, "aux"))
        auxiliary, aux_strength = apply_aux(clean, aux_kind, aux_rng)
        auxiliary_path = multi_root / split / "auxiliary" / f"{stem}_{aux_kind}.png"
        Image.fromarray(auxiliary).save(auxiliary_path)
        multi_rows.append(
            {
                **common,
                "sample_id": f"{split}_{local_index:06d}_aux",
                "split": split,
                "image_path": str(auxiliary_path.resolve()),
                "degradation_family": "auxiliary",
                "degradation_kind": aux_kind,
                "degradation_strength": round(float(aux_strength), 6),
            }
        )

    blur_frame = pd.DataFrame(blur_rows)
    multi_frame = pd.DataFrame(multi_rows)
    blur_frame.to_csv(blur_root / "manifest.csv", index=False)
    multi_frame.to_csv(multi_root / "manifest.csv", index=False)
    for split in ("train", "val", "test"):
        blur_frame[blur_frame.split == split].to_csv(blur_root / f"{split}.csv", index=False)
        multi_frame[multi_frame.split == split].to_csv(multi_root / f"{split}.csv", index=False)

    summary = {
        "source_manifest": str(manifest_path.resolve()),
        "seed": seed,
        "canvas": list(CANVAS),
        "source_samples": int(len(source)),
        "blur_samples": int(len(blur_frame)),
        "multi_samples": int(len(multi_frame)),
        "split_source_counts": source.split.value_counts().sort_index().to_dict(),
        "multi_degradation_counts": multi_frame.degradation_kind.value_counts().sort_index().to_dict(),
        "split_preserved": True,
        "source_overlap_between_splits": False,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(Path(args.manifest), Path(args.output_dir), args.seed), indent=2))

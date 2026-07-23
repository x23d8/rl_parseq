"""Locked evaluation for the two fair restoration experiments.

Experiment 1 compares learned action selectors on all six degradations.
Experiment 2 restricts the same locked policies to blur and adds PixelRL.
All model/action selection is frozen from validation before test is scored.
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
from PIL import Image
from scipy.stats import binomtest
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_restoration.actions import get_action_profile  # noqa: E402
from rl_restoration.policy import RewardRouter  # noqa: E402
from rl_restoration.ppo_policy import RestorationActorCritic  # noqa: E402
from rl_restoration.sequential_env import OfflineSequentialRestorationEnv  # noqa: E402
from rl_restoration.train_ppo import evaluate_policy  # noqa: E402
from rl_restoration.train_router import evaluate_selection, load_cache, predict_rewards  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import (  # noqa: E402
    edit_distance,
    normalize_plate_text,
)


ACTION_PROFILE = "fair_restoration"
MULTI_METHODS = ("raw", "best_global", "bandit", "ppo2", "ocr_oracle")
BLUR_METHODS = (
    "raw",
    "classical_unsharp",
    "unsharp_mild",
    "wiener_deconv",
    "rl_bilateral",
    "best_blur_global",
    "bandit",
    "ppo2",
    "pixelrl",
    "ocr_oracle",
    "psnr_oracle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_psnr(cache_dir: Path, split: str, paths: np.ndarray, action_names: list[str]) -> np.ndarray:
    frame = pd.read_csv(cache_dir / f"{split}_action_trajectories.csv")
    matrix = frame.pivot(index="image_path", columns="action", values="psnr")
    matrix = matrix.reindex(index=paths, columns=action_names)
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete {split} PSNR matrix")
    return matrix.to_numpy(dtype=np.float64)


def load_bandit(cache: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_names = checkpoint["action_names"]
    model = RewardRouter(
        checkpoint["input_dim"], len(action_names), checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    features = (cache["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    rewards = predict_rewards(model, features.astype(np.float32), device)
    metrics, frame, selected = evaluate_selection(
        cache, rewards, action_names, float(checkpoint["selection_margin"])
    )
    return metrics, frame, selected, checkpoint


def load_ppo(
    cache: dict,
    checkpoint_path: Path,
    teacher_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_names = checkpoint["action_names"]
    standardized = (cache["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]

    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher = RewardRouter(
        teacher_checkpoint["input_dim"],
        len(action_names),
        teacher_checkpoint["hidden_dim"],
        teacher_checkpoint["dropout"],
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher_x = (
        (cache["features"] - teacher_checkpoint["feature_mean"])
        / teacher_checkpoint["feature_std"]
    )
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
    metrics, frame = evaluate_policy(
        model,
        env,
        cache,
        action_names,
        checkpoint["first_margin"],
        checkpoint["revise_margin"],
    )
    selected = np.asarray([action_names.index(value) for value in frame.final_action], dtype=np.int64)
    return metrics, frame, selected, checkpoint


def choose_global(cache: dict, psnr: np.ndarray, indices: np.ndarray) -> int:
    """Choose one static action using validation only."""
    target_lengths = np.asarray([max(len(value), 1) for value in cache["targets"]], dtype=np.int64)
    keys = []
    for action_index in range(cache["exact"].shape[1]):
        exact = float(cache["exact"][indices, action_index].mean())
        edits = cache["edit_distance"][indices, action_index].sum()
        char_acc = float(1.0 - edits / target_lengths[indices].sum())
        keys.append((exact, char_acc, float(psnr[indices, action_index].mean()), -action_index))
    return int(max(range(len(keys)), key=lambda index: keys[index]))


def ocr_oracle(cache: dict) -> np.ndarray:
    """Ground-truth OCR oracle; an upper bound, never a deployable method."""
    chosen = []
    for row in range(len(cache["targets"])):
        keys = [
            (
                int(cache["exact"][row, action]),
                -int(cache["edit_distance"][row, action]),
                float(cache["normalized_confidence"][row, action]),
                -float(cache["action_cost"][row, action]),
                -action,
            )
            for action in range(cache["exact"].shape[1])
        ]
        chosen.append(max(range(len(keys)), key=lambda index: keys[index]))
    return np.asarray(chosen, dtype=np.int64)


def method_frame(
    cache: dict,
    psnr: np.ndarray,
    selected: np.ndarray,
    action_names: list[str],
) -> pd.DataFrame:
    rows = np.arange(len(selected))
    return pd.DataFrame(
        {
            "image_path": cache["image_paths"],
            "target": [normalize_plate_text(value) for value in cache["targets"]],
            "selected_action": [action_names[index] for index in selected],
            "prediction": [
                normalize_plate_text(value) for value in cache["predictions"][rows, selected]
            ],
            "exact": cache["exact"][rows, selected].astype(bool),
            "edit_distance": cache["edit_distance"][rows, selected].astype(int),
            "psnr": psnr[rows, selected],
            "cost": cache["action_cost"][rows, selected],
        }
    )


def attach_manifest(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "image_path",
        "source_path",
        "clean_path",
        "degradation_family",
        "degradation_kind",
        "degradation_strength",
    ]
    merged = frame.merge(manifest[columns], on="image_path", how="left", validate="one_to_one")
    if merged.clean_path.isna().any():
        raise ValueError("Cache rows do not align with the benchmark manifest")
    return merged


def score(frame: pd.DataFrame, raw_exact: np.ndarray) -> dict:
    exact = frame.exact.to_numpy(dtype=bool)
    total_chars = sum(max(len(value), 1) for value in frame.target.astype(str))
    return {
        "samples": int(len(frame)),
        "exact_acc": float(exact.mean()),
        "cer": float(frame.edit_distance.sum() / total_chars),
        "char_acc": float(1.0 - frame.edit_distance.sum() / total_chars),
        "fixed_vs_raw": int(((~raw_exact) & exact).sum()),
        "broken_vs_raw": int((raw_exact & (~exact)).sum()),
        "net_fixes_vs_raw": int(((~raw_exact) & exact).sum() - (raw_exact & (~exact)).sum()),
        "mean_psnr": float(frame.psnr.mean()),
        "mean_ssim": float(frame.ssim.mean()) if "ssim" in frame and frame.ssim.notna().any() else None,
        "mean_cost": float(frame.cost.mean()),
    }


def paired_statistics(
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    raw_exact = raw.exact.to_numpy(dtype=bool)
    exact = frame.exact.to_numpy(dtype=bool)
    fixed = int(((~raw_exact) & exact).sum())
    broken = int((raw_exact & (~exact)).sum())
    p_value = 1.0 if fixed + broken == 0 else float(
        binomtest(min(fixed, broken), fixed + broken, p=0.5, alternative="two-sided").pvalue
    )

    groups = frame.source_path.astype(str).to_numpy()
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    target_lengths = np.asarray([max(len(value), 1) for value in frame.target.astype(str)])
    method_edits = frame.edit_distance.to_numpy(dtype=np.float64)
    raw_edits = raw.edit_distance.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    exact_delta = np.empty(bootstrap_samples, dtype=np.float64)
    cer_delta = np.empty(bootstrap_samples, dtype=np.float64)
    group_rows = [np.flatnonzero(group_inverse == index) for index in range(len(unique_groups))]
    for iteration in range(bootstrap_samples):
        sampled_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([group_rows[index] for index in sampled_groups])
        exact_delta[iteration] = (exact[indices].mean() - raw_exact[indices].mean()) * 100.0
        cer_delta[iteration] = (
            method_edits[indices].sum() - raw_edits[indices].sum()
        ) / target_lengths[indices].sum()
    return {
        "mcnemar_fixed": fixed,
        "mcnemar_broken": broken,
        "mcnemar_exact_p": p_value,
        "exact_delta_points_ci95": [float(value) for value in np.quantile(exact_delta, [0.025, 0.975])],
        "cer_delta_ci95": [float(value) for value in np.quantile(cer_delta, [0.025, 0.975])],
        "bootstrap_unit": "source_path",
        "bootstrap_samples": bootstrap_samples,
    }


def compute_ssim(frames: dict[str, pd.DataFrame], action_profile: str) -> None:
    """Compute SSIM once per (image, action), shared across method frames."""
    actions = {action.name: action for action in get_action_profile(action_profile)}
    cache: dict[tuple[str, str], float] = {}
    total = sum(len(frame) for frame in frames.values())
    done = 0
    for method, frame in frames.items():
        values = []
        for row in frame.itertuples(index=False):
            done += 1
            key = (str(row.image_path), str(row.selected_action))
            if key not in cache:
                with Image.open(row.clean_path) as clean_image:
                    clean = np.asarray(clean_image.convert("L"), dtype=np.uint8)
                with Image.open(row.image_path) as degraded_image:
                    restored_image = actions[row.selected_action].apply(degraded_image)
                restored = np.asarray(restored_image.convert("L"), dtype=np.uint8)
                if restored.shape != clean.shape:
                    restored = np.asarray(
                        Image.fromarray(restored).resize((clean.shape[1], clean.shape[0]), Image.Resampling.BICUBIC)
                    )
                cache[key] = float(structural_similarity(clean, restored, data_range=255))
            values.append(cache[key])
        frame["ssim"] = values
        print(f"SSIM {method}: {len(frame)} rows ({done}/{total})", flush=True)


def pixel_eval_frame(
    path: Path,
    reference: pd.DataFrame,
    output_key: str,
    selected_action: str,
) -> pd.DataFrame:
    pixel = pd.read_csv(path)
    pixel.image_path = pixel.image_path.astype(str)
    wanted = reference[[
        "image_path", "source_path", "clean_path", "degradation_family", "degradation_kind",
        "degradation_strength", "target",
    ]]
    merged = wanted.merge(pixel, on="image_path", how="left", validate="one_to_one")
    prediction_column = f"pred_{output_key}"
    if merged[prediction_column].isna().any():
        raise ValueError(f"Pixel evaluation output {output_key} does not align with common blur rows")
    common_raw = reference.prediction.astype(str).map(normalize_plate_text).to_numpy()
    pixel_raw = merged.pred_blurred.astype(str).map(normalize_plate_text).to_numpy()
    if not np.array_equal(common_raw, pixel_raw):
        mismatches = int((common_raw != pixel_raw).sum())
        raise ValueError(
            f"Fixed PARSeq contract mismatch: {mismatches} raw blur predictions differ"
        )
    prediction = merged[prediction_column].astype(str).map(normalize_plate_text)
    target = merged.target.astype(str).map(normalize_plate_text)
    return pd.DataFrame(
        {
            "image_path": merged.image_path,
            "source_path": merged.source_path,
            "clean_path": merged.clean_path,
            "degradation_family": merged.degradation_family,
            "degradation_kind": merged.degradation_kind,
            "degradation_strength": merged.degradation_strength,
            "target": target,
            "selected_action": selected_action,
            "prediction": prediction,
            "exact": prediction.to_numpy() == target.to_numpy(),
            "edit_distance": [edit_distance(p, t) for p, t in zip(prediction, target)],
            "psnr": merged[f"psnr_{output_key}"].astype(float),
            "ssim": merged[f"ssim_{output_key}"].astype(float),
            "cost": np.nan,
        }
    )


def summary_table(
    frames: dict[str, pd.DataFrame], raw_key: str, order: tuple[str, ...], seed: int, bootstrap: int
) -> tuple[pd.DataFrame, dict]:
    raw = frames[raw_key]
    raw_exact = raw.exact.to_numpy(dtype=bool)
    rows, paired = [], {}
    raw_acc = float(raw_exact.mean())
    for method in order:
        metrics = score(frames[method], raw_exact)
        metrics["method"] = method
        metrics["exact_delta_vs_raw_points"] = (metrics["exact_acc"] - raw_acc) * 100.0
        rows.append(metrics)
        paired[method] = paired_statistics(frames[method], raw, seed, bootstrap)
    return pd.DataFrame(rows), paired


def action_distribution(frame: pd.DataFrame) -> dict:
    return {str(key): int(value) for key, value in frame.selected_action.value_counts().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument("--cache-dir", default=str(base / "cache"))
    parser.add_argument("--manifest", default=str(base / "dataset" / "multi" / "manifest.csv"))
    parser.add_argument("--bandit", default=str(base / "models" / "bandit_seed_123" / "best_reward_router.pt"))
    parser.add_argument("--ppo", default=str(base / "models" / "ppo2_seed_123" / "best_ppo_restoration_policy.pt"))
    parser.add_argument("--pixelrl-val", default=str(base / "pixelrl_eval" / "eval_val_predictions.csv"))
    parser.add_argument("--pixelrl-test", default=str(base / "pixelrl_eval" / "eval_test_predictions.csv"))
    parser.add_argument("--parseq-checkpoint", required=True)
    parser.add_argument("--output-dir", default=str(base / "results"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest.image_path = manifest.image_path.astype(str)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    actions = get_action_profile(ACTION_PROFILE)
    action_names = [action.name for action in actions]

    split_data = {}
    for split in ("val", "test"):
        cache = load_cache(cache_dir, split, action_names)
        psnr = load_psnr(cache_dir, split, cache["image_paths"], action_names)
        bandit_metrics, _, bandit_selected, bandit_ckpt = load_bandit(cache, Path(args.bandit), device)
        ppo_metrics, _, ppo_selected, ppo_ckpt = load_ppo(
            cache, Path(args.ppo), Path(args.bandit), device
        )
        split_data[split] = {
            "cache": cache,
            "psnr": psnr,
            "bandit_selected": bandit_selected,
            "ppo_selected": ppo_selected,
            "bandit_metrics": bandit_metrics,
            "ppo_metrics": ppo_metrics,
        }

    val_manifest = manifest.loc[manifest.split == "val"]
    val_paths = split_data["val"]["cache"]["image_paths"]
    val_kind = val_manifest.set_index("image_path").reindex(val_paths)
    if val_kind.degradation_family.isna().any():
        raise ValueError("Validation manifest/cache mismatch")
    all_val_indices = np.arange(len(val_paths))
    blur_val_indices = np.flatnonzero(val_kind.degradation_family.to_numpy() == "blur")
    best_global = choose_global(split_data["val"]["cache"], split_data["val"]["psnr"], all_val_indices)
    best_blur = choose_global(split_data["val"]["cache"], split_data["val"]["psnr"], blur_val_indices)

    cache = split_data["test"]["cache"]
    psnr = split_data["test"]["psnr"]
    count = len(cache["image_paths"])
    selected = {
        "raw": np.zeros(count, dtype=np.int64),
        "best_global": np.full(count, best_global, dtype=np.int64),
        "bandit": split_data["test"]["bandit_selected"],
        "ppo2": split_data["test"]["ppo_selected"],
        "ocr_oracle": ocr_oracle(cache),
        "psnr_oracle": psnr.argmax(axis=1),
    }
    test_manifest = manifest.loc[manifest.split == "test"]
    frames = {
        method: attach_manifest(method_frame(cache, psnr, indices, action_names), test_manifest)
        for method, indices in selected.items()
    }
    compute_ssim({key: frames[key] for key in ("raw", "best_global", "bandit", "ppo2")}, ACTION_PROFILE)
    frames["ocr_oracle"]["ssim"] = np.nan
    frames["psnr_oracle"]["ssim"] = np.nan

    multi_frames = {method: frames[method].copy() for method in MULTI_METHODS}
    multi_table, multi_paired = summary_table(
        multi_frames, "raw", MULTI_METHODS, args.seed, args.bootstrap_samples
    )

    blur_mask = frames["raw"].degradation_family.eq("blur").to_numpy()
    blur_frames = {method: frame.loc[blur_mask].reset_index(drop=True) for method, frame in frames.items()}
    for action_name in ("unsharp_mild", "wiener_deconv", "rl_bilateral"):
        index = action_names.index(action_name)
        action_indices = np.full(count, index, dtype=np.int64)
        frame = attach_manifest(method_frame(cache, psnr, action_indices, action_names), test_manifest)
        blur_frames[action_name] = frame.loc[blur_mask].reset_index(drop=True)
    blur_indices = np.full(count, best_blur, dtype=np.int64)
    best_blur_frame = attach_manifest(method_frame(cache, psnr, blur_indices, action_names), test_manifest)
    blur_frames["best_blur_global"] = best_blur_frame.loc[blur_mask].reset_index(drop=True)
    compute_ssim(
        {
            key: blur_frames[key]
            for key in ("unsharp_mild", "wiener_deconv", "rl_bilateral", "best_blur_global")
        },
        ACTION_PROFILE,
    )
    blur_frames["classical_unsharp"] = pixel_eval_frame(
        Path(args.pixelrl_test), blur_frames["raw"], "classical", "classical_unsharp_cv"
    )
    blur_frames["pixelrl"] = pixel_eval_frame(
        Path(args.pixelrl_test), blur_frames["raw"], "rl", "pixelrl_a2c"
    )
    blur_table, blur_paired = summary_table(
        blur_frames, "raw", BLUR_METHODS, args.seed + 1, args.bootstrap_samples
    )

    multi_table.to_csv(output_dir / "multi_summary.csv", index=False)
    blur_table.to_csv(output_dir / "blur_summary.csv", index=False)
    multi_long = pd.concat(
        [frame.assign(method=method) for method, frame in multi_frames.items()], ignore_index=True
    )
    blur_long = pd.concat(
        [blur_frames[method].assign(method=method) for method in BLUR_METHODS], ignore_index=True
    )
    multi_long.to_csv(output_dir / "multi_test_predictions.csv", index=False)
    blur_long.to_csv(output_dir / "blur_test_predictions.csv", index=False)

    by_kind = []
    for method, frame in multi_frames.items():
        for kind, group in frame.groupby("degradation_kind"):
            raw_group = multi_frames["raw"].loc[multi_frames["raw"].degradation_kind == kind]
            metrics = score(group.reset_index(drop=True), raw_group.exact.to_numpy(dtype=bool))
            metrics.update({"method": method, "degradation_kind": kind})
            by_kind.append(metrics)
    pd.DataFrame(by_kind).to_csv(output_dir / "multi_by_degradation.csv", index=False)

    pixel_val = pd.read_csv(args.pixelrl_val)
    summary = {
        "protocol": {
            "action_profile": ACTION_PROFILE,
            "parseq_checkpoint": str(Path(args.parseq_checkpoint).resolve()),
            "parseq_sha256": sha256(Path(args.parseq_checkpoint)),
            "validation_samples": int(len(split_data["val"]["cache"]["image_paths"])),
            "test_samples": count,
            "blur_test_samples": int(blur_mask.sum()),
            "best_global_action_from_validation": action_names[best_global],
            "best_blur_action_from_validation": action_names[best_blur],
            "test_used_for_selection": False,
            "bandit_checkpoint_epoch": int(bandit_ckpt["epoch"]),
            "ppo_checkpoint_epoch": int(ppo_ckpt["epoch"]),
            "pixelrl_validation_exact": float(pixel_val.exact_rl.mean()),
            "pixelrl_validation_psnr": float(pixel_val.psnr_rl.mean()),
        },
        "multi_paired_statistics": multi_paired,
        "blur_paired_statistics": blur_paired,
        "action_distributions": {
            "bandit_test": action_distribution(frames["bandit"]),
            "ppo2_test": action_distribution(frames["ppo2"]),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(multi_table.to_string(index=False), flush=True)
    print(blur_table.to_string(index=False), flush=True)
    print(json.dumps(summary["protocol"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

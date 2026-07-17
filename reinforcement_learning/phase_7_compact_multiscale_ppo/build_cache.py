"""Build compact multiscale trajectories and candidate observations for Phase 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MIN_FORMAL_EXTERNAL_SAMPLES = 500
for path in (ROOT, ROOT / "train_no_refinement", ROOT / "parseq", ROOT / "preprocessing_best_config"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preprocessing_best_config.benchmark_multiscale_tta import ViewSpec, transform_view  # noqa: E402
from preprocessing_best_config.find_best_preprocessing_config import load_notebook_checkpoint  # noqa: E402
from reinforcement_learning.phase_7_compact_multiscale_ppo.action_space import (  # noqa: E402
    COMPACT_VIEWS,
    CompactView,
    view_metadata,
)
from rl_restoration.features import parseq_state_features  # noqa: E402
from rl_restoration.reward import RewardConfig, ocr_reward  # noqa: E402
from train_no_refinement.parseq_official_anpr_pipeline import edit_distance, normalize_plate_text  # noqa: E402


class ViewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, view: CompactView, image_size: tuple[int, int]):
        self.frame = frame.reset_index(drop=True)
        self.view = view
        self.spec = ViewSpec(view.name, view.zoom, view.upscale, view.preprocessing, view.unwrap_two_line)
        self.image_size = tuple(image_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(str(row.image_path)) as opened:
            image = opened.convert("RGB")
        return transform_view(image, self.spec, self.image_size)


@torch.inference_mode()
def encode_view(model, model_cfg, frame: pd.DataFrame, view: CompactView, args):
    loader = DataLoader(
        ViewDataset(frame, view, tuple(model_cfg.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    features, rows = [], []
    metadata = np.asarray(view_metadata(view), dtype=np.float32)
    for offset, images in enumerate(tqdm(loader, desc=f"{args.split}: {view.name}", leave=False)):
        images = images.to(args.device_obj, non_blocking=True)
        logits = model(images, max_length=model_cfg.max_label_length)
        probabilities = logits.softmax(-1)
        predictions, token_probabilities = model.tokenizer.decode(probabilities)
        predictions = [normalize_plate_text(value) for value in predictions]
        deep = parseq_state_features(model, images, predictions, logits).cpu().numpy()
        tiled_metadata = np.repeat(metadata[None], len(images), axis=0)
        features.append(np.concatenate((deep, tiled_metadata), axis=1).astype(np.float32))
        start = offset * args.batch_size
        batch_frame = frame.iloc[start : start + len(images)]
        for (_, record), prediction, token_values in zip(batch_frame.iterrows(), predictions, token_probabilities):
            target = normalize_plate_text(record.label)
            confidence = float(token_values.prod().item())
            distance = edit_distance(prediction, target)
            rows.append(
                {
                    "image_path": str(record.image_path),
                    "target": target,
                    "action": view.name,
                    "action_cost": view.cost,
                    "prediction": prediction,
                    "confidence": confidence,
                    "normalized_confidence": math.exp(math.log(max(confidence, 1e-12)) / max(len(prediction) + 1, 1)),
                    "edit_distance": distance,
                    "exact": prediction == target,
                    "target_length": max(len(target), 1),
                }
            )
    return np.concatenate(features), pd.DataFrame(rows)


def attach_rewards(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.action == COMPACT_VIEWS[0].name][
        ["image_path", "prediction", "edit_distance", "exact"]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "edit_distance": "baseline_edit_distance",
            "exact": "baseline_exact",
        }
    )
    result = frame.merge(baseline, on="image_path", validate="many_to_one")
    config = RewardConfig()
    result["reward"] = result.apply(
        lambda row: ocr_reward(
            row.baseline_prediction,
            int(row.baseline_edit_distance),
            row.prediction,
            int(row.edit_distance),
            row.target,
            float(row.action_cost),
            config,
        ),
        axis=1,
    )
    return result


def validate_external_group_disjoint(manifest: pd.DataFrame, historical_manifest_path: Path) -> dict:
    """Require external normalized-label groups to be absent from historical splits."""

    if not historical_manifest_path.is_file():
        raise FileNotFoundError(f"Historical manifest for leakage audit is missing: {historical_manifest_path}")
    historical = pd.read_csv(historical_manifest_path)
    if "label" not in historical and "target" not in historical:
        raise ValueError("Historical manifest must contain a label or target column")
    historical = historical.rename(columns={"target": "label"}) if "label" not in historical else historical
    historical_labels = set(historical.label.astype(str).map(normalize_plate_text))
    external_labels = manifest.label.astype(str).map(normalize_plate_text)
    overlap = external_labels[external_labels.isin(historical_labels)]
    if not overlap.empty:
        raise ValueError(
            "External manifest is not group-disjoint from historical train/val/test: "
            f"{len(overlap)} row(s), {overlap.nunique()} normalized label(s) overlap"
        )
    return {
        "group_key": "normalized target/label",
        "group_overlap": 0,
        "historical_manifest": {
            "path": str(historical_manifest_path),
            "sha256": hashlib.sha256(historical_manifest_path.read_bytes()).hexdigest(),
            "rows": int(len(historical)),
        },
        "historical_labels_used_for_exclusion_only": True,
    }


def validate_external_input_contract(manifest: pd.DataFrame) -> None:
    if "input_contract" not in manifest:
        raise ValueError("External manifest must explicitly declare input_contract=plate_crop")
    contracts = manifest.input_contract.astype(str).str.strip().str.lower()
    if not (contracts == "plate_crop").all():
        raise ValueError("Every external evaluation image must satisfy input_contract=plate_crop")


def external_power_contract(sample_count: int, allow_underpowered_diagnostic: bool) -> dict:
    contract = {
        "minimum_formal_samples": MIN_FORMAL_EXTERNAL_SAMPLES,
        "actual_samples": int(sample_count),
        "formal_ready": sample_count >= MIN_FORMAL_EXTERNAL_SAMPLES,
        "underpowered_diagnostic_override": bool(allow_underpowered_diagnostic),
    }
    if not contract["formal_ready"] and not allow_underpowered_diagnostic:
        raise ValueError(
            f"Formal external evaluation requires at least {MIN_FORMAL_EXTERNAL_SAMPLES} group-disjoint "
            f"plate crops; received {sample_count}. Use --allow-underpowered-diagnostic only for diagnostics."
        )
    return contract


def external_manifest_preflight(
    manifest: pd.DataFrame,
    manifest_path: Path,
    source_rows: int,
    group_audit: dict | None = None,
    power_contract: dict | None = None,
) -> dict:
    """Validate a locked external manifest without loading the OCR model or writing artifacts."""

    missing_paths = [path for path in manifest.image_path.astype(str) if not Path(path).is_file()]
    if missing_paths:
        preview = ", ".join(missing_paths[:3])
        suffix = " ..." if len(missing_paths) > 3 else ""
        raise FileNotFoundError(f"External manifest references {len(missing_paths)} missing image(s): {preview}{suffix}")
    return {
        "ready_for_external_cache": True,
        "split": "external_holdout",
        "source_rows": int(source_rows),
        "selected_rows": int(len(manifest)),
        "distinct_labels": int(manifest.label.nunique()),
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "external_contract": True,
            "group_disjoint": bool(group_audit and group_audit.get("group_overlap") == 0),
            "input_contract": "plate_crop",
        },
        "group_audit": group_audit,
        "power_contract": power_contract,
        "inference_run": False,
        "artifacts_written": False,
    }


def run(args):
    if args.split not in {"train", "val", "external_holdout"}:
        raise ValueError("Phase 7 cache only supports train, val, or a newly supplied external_holdout")
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents and output_dir != HERE:
        raise ValueError("Phase 7 artifacts must remain inside reinforcement_learning/phase_7_compact_multiscale_ppo")
    if args.split == "external_holdout" and output_dir == (HERE / "results/cache").resolve():
        raise ValueError("External holdout requires an explicit dedicated --output-dir inside Phase 7")
    manifest_path = Path(args.manifest).resolve()
    manifest = pd.read_csv(manifest_path)
    source_rows = len(manifest)
    if "image_path" not in manifest:
        raise ValueError("Manifest must contain an image_path column")
    if "label" not in manifest and "target" not in manifest:
        raise ValueError("Manifest must contain a label or target column")
    if args.split == "external_holdout" and "split" not in manifest:
        raise ValueError("External manifest must declare split=external_holdout for every evaluated row")
    if args.split == "external_holdout":
        validate_external_input_contract(manifest)
    if "split" in manifest:
        manifest = manifest[manifest.split.astype(str).str.lower() == args.split].copy()
    if manifest.empty:
        raise ValueError(f"Manifest contains no rows for split {args.split!r}")
    manifest = manifest.rename(columns={"target": "label"}) if "label" not in manifest else manifest
    if manifest.image_path.isna().any() or manifest.label.isna().any():
        raise ValueError("Manifest image_path and label values must be non-empty")
    manifest.image_path = manifest.image_path.astype(str).str.strip()
    manifest.label = manifest.label.astype(str).str.strip()
    if (manifest.image_path == "").any() or (manifest.label == "").any():
        raise ValueError("Manifest image_path and label values must be non-empty")
    if args.split == "external_holdout" and manifest.image_path.duplicated().any():
        raise ValueError("External manifest must not contain duplicate image_path values")
    manifest = manifest.drop_duplicates("image_path").reset_index(drop=True)
    group_audit = None
    if args.split == "external_holdout":
        historical_manifest_path = Path(args.internal_manifest).resolve()
        group_audit = validate_external_group_disjoint(manifest, historical_manifest_path)
    power_contract = None
    if args.split == "external_holdout":
        power_contract = external_power_contract(len(manifest), args.allow_underpowered_diagnostic)
    if args.preflight:
        if args.split != "external_holdout":
            raise ValueError("--preflight is reserved for a newly supplied external_holdout manifest")
        return external_manifest_preflight(manifest, manifest_path, source_rows, group_audit, power_contract)
    if args.split == "external_holdout":
        audit_artifacts = list(output_dir.glob("external_holdout_*"))
        if audit_artifacts:
            raise FileExistsError("External cache already exists; refusing to overwrite locked holdout artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.device_obj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_cfg, _ = load_notebook_checkpoint(Path(args.checkpoint).resolve(), args.device_obj, args.refine_iters)
    feature_blocks, trajectory_blocks = [], []
    for view in COMPACT_VIEWS:
        features, trajectories = encode_view(model, model_cfg, manifest, view, args)
        feature_blocks.append(features)
        trajectory_blocks.append(trajectories)
    candidates = np.stack(feature_blocks, axis=1)
    trajectories = attach_rewards(pd.concat(trajectory_blocks, ignore_index=True))
    action_names = np.asarray([view.name for view in COMPACT_VIEWS], dtype=np.str_)
    paths = np.asarray(manifest.image_path.astype(str), dtype=np.str_)
    targets = np.asarray(manifest.label.map(normalize_plate_text), dtype=np.str_)
    np.savez_compressed(
        output_dir / f"{args.split}_candidate_features.npz",
        candidate_features=candidates,
        image_paths=paths,
        action_names=action_names,
    )
    np.savez_compressed(
        output_dir / f"{args.split}_state_features.npz",
        features=candidates[:, 0],
        image_paths=paths,
        targets=targets,
    )
    trajectories.to_csv(output_dir / f"{args.split}_action_trajectories.csv", index=False)
    artifact_paths = {
        "candidate_features": output_dir / f"{args.split}_candidate_features.npz",
        "state_features": output_dir / f"{args.split}_state_features.npz",
        "action_trajectories": output_dir / f"{args.split}_action_trajectories.csv",
    }
    pivot = trajectories.pivot(index="image_path", columns="action", values="exact").reindex(index=paths, columns=action_names)
    baseline_exact = float(pivot.iloc[:, 0].mean())
    oracle_exact = float(pivot.any(axis=1).mean())
    summary = {
        "split": args.split,
        "samples": len(paths),
        "actions": action_names.tolist(),
        "candidate_shape": list(candidates.shape),
        "baseline_exact": baseline_exact,
        "oracle_exact": oracle_exact,
        "oracle_gain_points": 100.0 * (oracle_exact - baseline_exact),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).resolve().read_bytes()).hexdigest(),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in artifact_paths.items()
        },
        "refine_iters": int(args.refine_iters),
        "test_loaded": False,
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_rows": source_rows,
            "selected_rows": len(paths),
            "external_contract": args.split == "external_holdout",
            "group_disjoint": bool(group_audit and group_audit.get("group_overlap") == 0),
            "input_contract": "plate_crop" if args.split == "external_holdout" else None,
        },
        "group_audit": group_audit,
        "power_contract": power_contract,
    }
    (output_dir / f"{args.split}_cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "external_holdout"))
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "outputs/testing/refinement_finetune_20260710_142307/best_official_parseq_anpr.pt"),
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "outputs/phase3_controlled_aug_full_frozen_eval/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--internal-manifest",
        default=str(ROOT / "outputs/phase3_controlled_aug_full_frozen_eval/dataset_manifest.csv"),
        help="Historical train/val/test manifest used only to reject overlapping external label groups.",
    )
    parser.add_argument("--output-dir", default=str(HERE / "results/cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate an external manifest without model inference or artifact writes.",
    )
    parser.add_argument(
        "--allow-underpowered-diagnostic",
        action="store_true",
        help=f"Allow fewer than {MIN_FORMAL_EXTERNAL_SAMPLES} external samples; resulting cache is diagnostic only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

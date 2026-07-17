"""Merge historical development data with the opened Phase 9 cache.

Phase 9 is no longer a holdout after its one-shot gate failed. This script
turns it into explicitly marked development data, using a stable 80/20 group
split for adaptation/selection, and never loads the audited legacy test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.data import (  # noqa: E402
    load_candidate_features,
    load_trajectory_cache,
    stable_group_holdout,
)


PHASE7_CACHE = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "results"
    / "cache"
)
PHASE9_CACHE = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "results"
    / "phase9_fresh_external_cache"
)
PHASE9_EVALUATION = (
    ROOT
    / "reinforcement_learning"
    / "phase_9_primary_ppo"
    / "results"
    / "external_locked_evaluation"
    / "summary.json"
)
DEFAULT_OUTPUT = HERE / "results" / "domain_adaptive_cache"
DOMAIN_SPLIT_SEED = 1001


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_paths(cache_dir: Path, split: str) -> dict[str, Path]:
    return {
        "candidate_features": cache_dir / f"{split}_candidate_features.npz",
        "state_features": cache_dir / f"{split}_state_features.npz",
        "action_trajectories": cache_dir / f"{split}_action_trajectories.csv",
    }


def load_raw(cache_dir: Path, split: str, action_names: list[str]) -> dict:
    cache = load_trajectory_cache(cache_dir, split, action_names)
    candidate = load_candidate_features(
        cache_dir / f"{split}_candidate_features.npz", cache["image_paths"], action_names
    )
    state = np.load(cache_dir / f"{split}_state_features.npz", allow_pickle=True)
    trajectories = pd.read_csv(cache_dir / f"{split}_action_trajectories.csv")
    if not np.array_equal(state["image_paths"].astype(str), cache["image_paths"]):
        raise ValueError(f"{split} state/cache paths differ")
    return {
        "cache": cache,
        "candidate_features": candidate,
        "state_features": state["features"].astype(np.float32),
        "trajectories": trajectories,
    }


def subset(raw: dict, mask: np.ndarray) -> dict:
    paths = raw["cache"]["image_paths"][mask]
    path_set = set(paths.astype(str))
    trajectories = raw["trajectories"][
        raw["trajectories"].image_path.astype(str).isin(path_set)
    ].copy()
    return {
        "image_paths": paths,
        "targets": raw["cache"]["targets"][mask],
        "candidate_features": raw["candidate_features"][mask],
        "state_features": raw["state_features"][mask],
        "trajectories": trajectories,
    }


def all_rows(raw: dict) -> dict:
    return subset(raw, np.ones(len(raw["cache"]["image_paths"]), dtype=bool))


def write_split(output_dir: Path, split: str, parts: list[dict], action_names: list[str]) -> dict:
    image_paths = np.concatenate([part["image_paths"] for part in parts]).astype(str)
    targets = np.concatenate([part["targets"] for part in parts]).astype(str)
    candidate = np.concatenate([part["candidate_features"] for part in parts]).astype(np.float32)
    state = np.concatenate([part["state_features"] for part in parts]).astype(np.float32)
    if len(set(image_paths)) != len(image_paths):
        raise ValueError(f"Duplicate image paths in mixed {split} split")
    trajectories = pd.concat([part["trajectories"] for part in parts], ignore_index=True)
    if len(trajectories) != len(image_paths) * len(action_names):
        raise ValueError(f"Incomplete mixed {split} trajectories")
    np.savez_compressed(
        output_dir / f"{split}_candidate_features.npz",
        candidate_features=candidate,
        image_paths=image_paths,
        action_names=np.asarray(action_names),
    )
    np.savez_compressed(
        output_dir / f"{split}_state_features.npz",
        features=state,
        image_paths=image_paths,
        targets=targets,
    )
    trajectories.to_csv(output_dir / f"{split}_action_trajectories.csv", index=False)
    return {
        "samples": int(len(image_paths)),
        "groups": int(len(set(targets))),
        "candidate_shape": list(candidate.shape),
        "baseline_exact": float(
            trajectories[trajectories.action == action_names[0]].exact.astype(bool).mean()
        ),
        "image_paths": image_paths,
        "targets": targets,
    }


def run(args: argparse.Namespace) -> dict:
    phase7_dir = Path(args.phase7_cache).resolve()
    phase9_dir = Path(args.phase9_cache).resolve()
    phase9_evaluation = Path(args.phase9_evaluation).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir != HERE and HERE not in output_dir.parents:
        raise ValueError("Phase 10 development artifacts must remain inside Phase 10")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Phase 10 development cache: {output_dir}")
    evaluation = json.loads(phase9_evaluation.read_text(encoding="utf-8"))
    if (
        evaluation.get("evaluation_role") != "locked_confirmatory"
        or evaluation.get("status") != "external_holdout_failed_gate"
        or evaluation.get("promotion_eligible") is not False
        or not evaluation.get("external_holdout_evaluated_once", False)
    ):
        raise ValueError("Phase 9 is not a completed failed holdout available for development")

    phase7_summary = json.loads((phase7_dir / "train_cache_summary.json").read_text(encoding="utf-8"))
    phase9_summary = json.loads(
        (phase9_dir / "external_holdout_cache_summary.json").read_text(encoding="utf-8")
    )
    action_names = list(phase7_summary["actions"])
    if list(phase9_summary.get("actions", [])) != action_names:
        raise ValueError("Phase 7 and Phase 9 action spaces differ")
    if phase7_summary.get("checkpoint") != phase9_summary.get("checkpoint"):
        raise ValueError("Phase 7 and Phase 9 caches use different PARSeq checkpoints")

    base_train = load_raw(phase7_dir, "train", action_names)
    base_val = load_raw(phase7_dir, "val", action_names)
    phase9 = load_raw(phase9_dir, "external_holdout", action_names)
    base_validation_groups = set(base_val["cache"]["targets"].astype(str))
    base_train_keep = np.asarray(
        [target not in base_validation_groups for target in base_train["cache"]["targets"].astype(str)],
        dtype=bool,
    )
    excluded_phase7_train_overlap = int((~base_train_keep).sum())
    if not base_train_keep.any():
        raise ValueError("Removing Phase 7 validation groups emptied the training split")
    adapt_mask, selection_mask = stable_group_holdout(
        phase9["cache"]["targets"], args.selection_fraction, args.seed
    )
    if set(phase9["cache"]["targets"][adapt_mask]) & set(
        phase9["cache"]["targets"][selection_mask]
    ):
        raise RuntimeError("Phase 9 domain split leaked label groups")

    output_dir.mkdir(parents=True, exist_ok=False)
    train = write_split(
        output_dir,
        "train",
        [subset(base_train, base_train_keep), subset(phase9, adapt_mask)],
        action_names,
    )
    val = write_split(
        output_dir,
        "val",
        [all_rows(base_val), subset(phase9, selection_mask)],
        action_names,
    )
    overlap = set(train["targets"]) & set(val["targets"])
    if overlap:
        raise ValueError(f"Mixed train/validation label leakage: {len(overlap)} groups")

    audit = {
        "contract": "phase10_domain_adaptive_development_cache_v1",
        "test_used": False,
        "phase9_role": "opened_failed_holdout_reclassified_as_development_only",
        "phase9_evaluation": {
            "path": str(phase9_evaluation),
            "sha256": sha256_file(phase9_evaluation),
        },
        "domain_split": {
            "seed": args.seed,
            "selection_fraction": args.selection_fraction,
            "adaptation_samples": int(adapt_mask.sum()),
            "selection_samples": int(selection_mask.sum()),
            "group_overlap": 0,
        },
        "mixed": {
            "train_samples": train["samples"],
            "validation_samples": val["samples"],
            "train_validation_group_overlap": 0,
            "phase7_train_rows_excluded_for_validation_group_overlap": excluded_phase7_train_overlap,
            "train_baseline_exact": train["baseline_exact"],
            "validation_baseline_exact": val["baseline_exact"],
            "candidate_shape_train": train["candidate_shape"],
            "candidate_shape_validation": val["candidate_shape"],
        },
        "sources": {
            "phase7": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for split in ("train", "val")
                for name, path in {
                    f"{split}_{key}": value for key, value in artifact_paths(phase7_dir, split).items()
                }.items()
            },
            "phase9": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in artifact_paths(phase9_dir, "external_holdout").items()
            },
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for split in ("train", "val")
            for name, path in {
                f"{split}_{key}": value for key, value in artifact_paths(output_dir, split).items()
            }.items()
        },
    }
    (output_dir / "development_cache_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-cache", default=str(PHASE7_CACHE))
    parser.add_argument("--phase9-cache", default=str(PHASE9_CACHE))
    parser.add_argument("--phase9-evaluation", default=str(PHASE9_EVALUATION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=DOMAIN_SPLIT_SEED)
    parser.add_argument("--selection-fraction", type=float, default=0.20)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

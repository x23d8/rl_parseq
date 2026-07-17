"""Reproduce Phase 12 guard evidence on opened Phase 8/9/11 data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.data import load_trajectory_cache  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.paired_statistics import improvement_gate  # noqa: E402
from reinforcement_learning.phase_6_candidate_oof_ppo.train import evaluate, paired_stats  # noqa: E402
from reinforcement_learning.phase_12_guarded_replicated_ppo.prepare_fresh_holdout import (  # noqa: E402
    validate_lock,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.selection import guarded_selection  # noqa: E402


DATASETS = (
    (
        "phase8",
        ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "results" / "external_locked_evaluation",
        "consensus_selections.csv",
        "policy_b_action",
    ),
    (
        "phase9",
        ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "results" / "phase9_posthoc_consensus_diagnostic",
        "consensus_selections.csv",
        "policy_b_action",
    ),
    (
        "phase11",
        ROOT / "reinforcement_learning" / "phase_11_replicated_primary_ppo" / "results" / "external_locked_evaluation",
        "policy_selections.csv",
        "final_action",
    ),
)


def run(args: argparse.Namespace) -> dict:
    lock_path = Path(args.candidate_lock).resolve()
    output_path = Path(args.output).resolve()
    if HERE not in output_path.parents:
        raise ValueError("Phase 12 development audit must remain inside Phase 12")
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite the Phase 12 development audit")
    lock = validate_lock(lock_path)
    candidate_frames, baseline_frames = [], []
    dataset_results = {}
    for name, directory, selections_name, action_column in DATASETS:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        actions = list(summary["cache"]["summary"]["actions"])
        cache = load_trajectory_cache(
            Path(summary["cache"]["directory"]), "external_holdout", actions
        )
        selections = pd.read_csv(directory / selections_name)
        required = {action_column, "input_transform", "crop_width", "crop_height"}
        missing = required.difference(selections.columns)
        if missing:
            raise ValueError(f"{name} selections lack guard columns: {sorted(missing)}")
        ppo_selected = np.asarray(
            [actions.index(value) for value in selections[action_column]], dtype=np.int64
        )
        selected, allowed = guarded_selection(
            ppo_selected,
            selections.input_transform.to_numpy(dtype=str),
            selections.crop_width.to_numpy(dtype=np.int64),
            selections.crop_height.to_numpy(dtype=np.int64),
        )
        metrics, candidate = evaluate(cache, selected, actions)
        baseline_metrics, baseline = evaluate(
            cache, np.zeros(len(selected), dtype=np.int64), actions
        )
        statistics = paired_stats(candidate, baseline, seed=args.seed)
        dataset_results[name] = {
            "samples": int(len(selected)),
            "guard_allowed_rate": float(allowed.mean()),
            "baseline_exact": baseline_metrics["exact_acc"],
            "guarded_exact": metrics["exact_acc"],
            "fixed": metrics["fixed"],
            "broken": metrics["broken"],
            "statistics_vs_baseline": statistics,
        }
        candidate_frames.append(candidate)
        baseline_frames.append(baseline)
    combined_candidate = pd.concat(candidate_frames, ignore_index=True)
    combined_baseline = pd.concat(baseline_frames, ignore_index=True)
    combined_stats = paired_stats(combined_candidate, combined_baseline, seed=args.seed)
    fixed = int(combined_candidate.fixed.sum())
    broken = int(combined_candidate.broken.sum())
    combined_metrics = {"net_fixes": fixed - broken}
    combined = {
        "samples": int(len(combined_candidate)),
        "fixed": fixed,
        "broken": broken,
        "statistics_vs_baseline": combined_stats,
        "formal_gate_if_it_were_confirmatory": improvement_gate(combined_metrics, combined_stats),
    }
    expected = lock["development_evidence"]["combined"]
    if (
        combined["samples"] != int(expected["samples"])
        or fixed != int(expected["fixed"])
        or broken != int(expected["broken"])
        or not np.isclose(
            combined_stats["paired_bootstrap"]["delta_exact"], expected["exact_delta"]
        )
        or not np.isclose(combined_stats["mcnemar"]["p_value_exact"], expected["mcnemar_p"])
    ):
        raise ValueError("Reproduced Phase 12 evidence differs from the prospective lock")
    audit = {
        "contract": "phase12_opened_development_guard_audit_v1",
        "role": "opened_development_only_not_promotable",
        "inference_run": False,
        "test_used": False,
        "candidate_lock": str(lock_path),
        "guard": lock["guard"],
        "datasets": dataset_results,
        "combined": combined,
        "promotion_eligible": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", default=str(HERE / "prospective_policy.json"))
    parser.add_argument(
        "--output", default=str(HERE / "results" / "opened_development_guard_audit.json")
    )
    parser.add_argument("--seed", type=int, default=1201)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


"""Unified command dispatcher for the PARSeq improvement phases."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "phase1": ROOT / "preprocessing_best_config" / "benchmark_multiscale_tta.py",
    "phase2": ROOT / "preprocessing_best_config" / "benchmark_multiscale_selector_phase2.py",
    "phase3": ROOT / "refinement_finetune" / "train_phase3_controlled_augmentation.py",
    "phase6-cache": ROOT / "reinforcement_learning" / "phase_6_candidate_oof_ppo" / "build_candidate_cache.py",
    "phase6": ROOT / "reinforcement_learning" / "phase_6_candidate_oof_ppo" / "train.py",
    "phase7-cache": ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo" / "build_cache.py",
    "phase7": ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo" / "train.py",
    "phase7-external": ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo" / "evaluate_external.py",
    "phase7-promote": ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo" / "promote.py",
    "phase7-runtime": ROOT / "reinforcement_learning" / "phase_7_compact_multiscale_ppo" / "runtime.py",
    "phase8-consensus": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "evaluate.py",
    "phase8-promote": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "promote.py",
    "phase8-runtime": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "runtime.py",
    "phase8-review": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "review_server.py",
    "phase8-finalize": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "finalize_fresh_holdout.py",
    "phase8-status": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "status.py",
    "phase8-preflight": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "preflight_fresh_review_queue.py",
    "phase8-runtime-crops": ROOT / "reinforcement_learning" / "phase_8_consensus_ppo" / "prepare_runtime_crops.py",
    "phase9-prepare": ROOT / "reinforcement_learning" / "phase_9_primary_ppo" / "prepare_fresh_holdout.py",
    "phase9-status": ROOT / "reinforcement_learning" / "phase_9_primary_ppo" / "status.py",
    "phase9-evaluate": ROOT / "reinforcement_learning" / "phase_9_primary_ppo" / "evaluate.py",
    "phase9-promote": ROOT / "reinforcement_learning" / "phase_9_primary_ppo" / "promote.py",
    "phase9-runtime": ROOT / "reinforcement_learning" / "phase_9_primary_ppo" / "runtime.py",
    "phase10-prepare": ROOT / "reinforcement_learning" / "phase_10_domain_adaptive_ppo" / "prepare_domain_cache.py",
    "phase10-train": ROOT / "reinforcement_learning" / "phase_10_domain_adaptive_ppo" / "train.py",
    "phase11-prepare": ROOT / "reinforcement_learning" / "phase_11_replicated_primary_ppo" / "prepare_fresh_holdout.py",
    "phase11-evaluate": ROOT / "reinforcement_learning" / "phase_11_replicated_primary_ppo" / "evaluate.py",
    "phase12-status": ROOT / "reinforcement_learning" / "phase_12_guarded_replicated_ppo" / "pool_status.py",
    "phase12-prepare": ROOT / "reinforcement_learning" / "phase_12_guarded_replicated_ppo" / "prepare_fresh_holdout.py",
    "phase12-evaluate": ROOT / "reinforcement_learning" / "phase_12_guarded_replicated_ppo" / "evaluate.py",
    "phase12-promote": ROOT / "reinforcement_learning" / "phase_12_guarded_replicated_ppo" / "promote.py",
    "phase12-runtime": ROOT / "reinforcement_learning" / "phase_12_guarded_replicated_ppo" / "runtime.py",
    "phase12-development-audit": ROOT / "reinforcement_learning" / "phase_12_guarded_replicated_ppo" / "audit_opened_development.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("phase", choices=TARGETS)
    args, remaining = parser.parse_known_args()
    target = TARGETS[args.phase]
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

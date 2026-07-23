"""Unified command dispatcher for the PARSeq improvement phases.

Run ``python reinforcement_learning/run_phase.py --list`` to see the canonical
entrypoint and the auxiliary commands available for every phase.  Arguments
after the command name are forwarded unchanged to the underlying script.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "phase1": ROOT / "reinforcement_learning" / "phase_1_multiscale_tta" / "run.py",
    "phase2": ROOT / "reinforcement_learning" / "phase_2_calibrated_selector" / "run.py",
    "phase3": ROOT / "reinforcement_learning" / "phase_3_controlled_augmentation" / "train.py",
    "phase4": ROOT / "rl_restoration" / "train_router.py",
    "phase4-cache": ROOT / "rl_restoration" / "build_trajectory_cache.py",
    "phase4-evaluate": ROOT / "rl_restoration" / "evaluate_locked_policy.py",
    "phase4-finetune": ROOT / "rl_restoration" / "finetune_with_policy.py",
    "phase5": ROOT / "rl_restoration" / "train_ppo.py",
    "phase5-evaluate": ROOT / "rl_restoration" / "evaluate_locked_ppo.py",
    "phase5-runtime": ROOT / "rl_restoration" / "ppo_runtime.py",
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
    "auto-manifest": ROOT / "reinforcement_learning" / "auto" / "manifest.py",
    "auto-cache": ROOT / "reinforcement_learning" / "auto" / "build_cache.py",
    "auto-train": ROOT / "reinforcement_learning" / "auto" / "train.py",
    "auto-evaluate": ROOT / "reinforcement_learning" / "auto" / "evaluate.py",
    "auto-experiment": ROOT / "reinforcement_learning" / "auto" / "experiment.py",
    "auto-runtime": ROOT / "reinforcement_learning" / "auto" / "runtime.py",
}

# Short names point at the primary operation of phases whose implementation has
# several protocol-specific entrypoints.  The explicit aliases above remain the
# preferred interface in automation because they state the intended operation.
TARGETS.update(
    {
        "phase6-train": TARGETS["phase6"],
        "phase7-train": TARGETS["phase7"],
        "phase8": TARGETS["phase8-consensus"],
        "phase9": TARGETS["phase9-evaluate"],
        "phase10": TARGETS["phase10-train"],
        "phase11": TARGETS["phase11-evaluate"],
        "phase12": TARGETS["phase12-evaluate"],
    }
)


def _print_commands() -> None:
    print(__doc__.splitlines()[0])
    print("\nAvailable commands:")
    width = max(map(len, TARGETS))
    for name in TARGETS:
        relative = TARGETS[name].relative_to(ROOT)
        print(f"  {name:<{width}}  {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("phase", nargs="?", choices=TARGETS)
    parser.add_argument("--list", action="store_true", dest="list_commands")
    args, remaining = parser.parse_known_args()
    if args.list_commands or args.phase is None:
        _print_commands()
        return

    target = TARGETS[args.phase]
    if not target.is_file():
        raise SystemExit(
            f"Entrypoint for {args.phase!r} is missing: {target}. "
            "Restore the phase source files before running this command."
        )
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

"""Train domain-adaptive candidate-set PPO on the Phase 10 mixed cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.train import parse_args, run  # noqa: E402


def main() -> None:
    cache = HERE / "results" / "domain_adaptive_cache"
    defaults = {
        "--trajectory-cache": str(cache),
        "--candidate-cache": str(cache),
        "--output-dir": str(HERE / "results" / "run_seed_1001"),
        "--seed": "1001",
        "--holdout-fraction": "0.20",
    }
    supplied = set(sys.argv[1:])
    for flag, value in defaults.items():
        if flag not in supplied:
            sys.argv.extend((flag, value))
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


"""Train Phase 7 using the shared candidate-set PPO architecture."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_6_candidate_oof_ppo.train import parse_args, run  # noqa: E402


def main():
    defaults = {
        "--trajectory-cache": str(HERE / "results/cache"),
        "--candidate-cache": str(HERE / "results/cache"),
        "--output-dir": str(HERE / "results/run_ocr_guard_seed_727"),
        "--seed": "727",
        "--holdout-fraction": "0.20",
    }
    supplied = set(sys.argv[1:])
    additions = []
    for flag, value in defaults.items():
        if flag not in supplied:
            additions.extend((flag, value))
    sys.argv.extend(additions)
    summary = run(parse_args())
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Entrypoint for Phase 2 calibrated candidate selector."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "preprocessing_best_config" / "benchmark_multiscale_selector_phase2.py"

if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")


"""Entrypoint for Phase 3 controlled-augmentation fine-tuning."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "refinement_finetune" / "train_phase3_controlled_augmentation.py"

if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")


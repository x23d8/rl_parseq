"""Nine-view action space retaining the 65-view validation oracle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactView:
    name: str
    zoom: float
    upscale: float
    preprocessing: str
    unwrap_two_line: bool
    cost: float


# Greedy set cover was fitted only on the already-audited legacy validation
# predictions. These nine views retain all 388/397 exact cases of the 65-view
# oracle while reducing OCR calls by 86.2%.
COMPACT_VIEWS = (
    CompactView("baseline", 1.00, 1.0, "train_baseline", False, 0.000),
    CompactView("unwrap_z1.07_up2_clahe_rl_deblur_bilateral", 1.07, 2.0, "clahe_rl_deblur_bilateral", True, 0.012),
    CompactView("full_z0.93_up3_clahe_clip1_tile4", 0.93, 3.0, "clahe_clip1_tile4", False, 0.007),
    CompactView("unwrap_z1.00_up3_train_baseline", 1.00, 3.0, "train_baseline", True, 0.010),
    CompactView("full_z0.85_up2_adaptive_noise_3way", 0.85, 2.0, "adaptive_noise_3way", False, 0.008),
    CompactView("full_z1.00_up2_clahe_rl_deblur_bilateral", 1.00, 2.0, "clahe_rl_deblur_bilateral", False, 0.009),
    CompactView("unwrap_z0.93_up3_adaptive_noise_3way", 0.93, 3.0, "adaptive_noise_3way", True, 0.013),
    CompactView("full_z1.15_up3_train_baseline", 1.15, 3.0, "train_baseline", False, 0.006),
    CompactView("full_z1.15_up2_train_baseline", 1.15, 2.0, "train_baseline", False, 0.004),
)


def view_metadata(view: CompactView) -> tuple[float, ...]:
    preprocessors = (
        "train_baseline",
        "clahe_clip1_tile4",
        "clahe_rl_deblur_bilateral",
        "adaptive_noise_3way",
    )
    return (
        view.zoom,
        view.upscale / 3.0,
        float(view.unwrap_two_line),
        *(float(view.preprocessing == name) for name in preprocessors),
    )


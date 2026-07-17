"""Paired statistical tests required by the locked Phase 6 protocol.

The module deliberately avoids the top-level name ``statistics`` because a
directly executed script places this directory on ``sys.path`` and would then
shadow Python's standard-library module used by PyTorch Lightning.
"""

from __future__ import annotations

import math

import numpy as np


def paired_bootstrap(
    candidate_exact: np.ndarray,
    reference_exact: np.ndarray,
    candidate_char: np.ndarray,
    reference_char: np.ndarray,
    seed: int = 20260715,
    samples: int = 10000,
) -> dict:
    rng = np.random.default_rng(seed)
    exact_delta = np.asarray(candidate_exact, dtype=float) - np.asarray(reference_exact, dtype=float)
    char_delta = np.asarray(candidate_char, dtype=float) - np.asarray(reference_char, dtype=float)
    n = len(exact_delta)
    exact_means = np.empty(samples, dtype=np.float64)
    char_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        exact_means[start : start + count] = exact_delta[indices].mean(axis=1)
        char_means[start : start + count] = char_delta[indices].mean(axis=1)
    return {
        "delta_exact": float(exact_delta.mean()),
        "delta_exact_ci95": [float(value) for value in np.quantile(exact_means, [0.025, 0.975])],
        "delta_character": float(char_delta.mean()),
        "delta_character_ci95": [float(value) for value in np.quantile(char_means, [0.025, 0.975])],
        "bootstrap_samples": int(samples),
    }


def mcnemar_exact(candidate_exact: np.ndarray, reference_exact: np.ndarray) -> dict:
    candidate = np.asarray(candidate_exact, dtype=bool)
    reference = np.asarray(reference_exact, dtype=bool)
    fixed = int((candidate & ~reference).sum())
    broken = int((~candidate & reference).sum())
    discordant = fixed + broken
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(fixed, broken) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"fixed": fixed, "broken": broken, "discordant": discordant, "p_value_exact": float(p_value)}


def improvement_gate(metrics: dict, stats: dict) -> dict:
    exact_ci = stats["paired_bootstrap"]["delta_exact_ci95"]
    passed = {
        "delta_exact_positive": stats["paired_bootstrap"]["delta_exact"] > 0,
        "exact_ci_excludes_zero": exact_ci[0] > 0,
        "character_not_lower": stats["paired_bootstrap"]["delta_character"] >= 0,
        "net_fixes_positive": metrics["net_fixes"] > 0,
        "mcnemar_significant": stats["mcnemar"]["p_value_exact"] < 0.05,
    }
    return {"passed": bool(all(passed.values())), "checks": passed}

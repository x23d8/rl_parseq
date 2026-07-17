"""Focused regression tests for the leakage and model contracts of Phase 6."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from reinforcement_learning.phase_6_candidate_oof_ppo.data import candidate_ocr_features, stable_group_holdout
from reinforcement_learning.phase_6_candidate_oof_ppo.model import CandidateSetActorCritic
from reinforcement_learning.phase_6_candidate_oof_ppo.paired_statistics import mcnemar_exact, paired_bootstrap


class Phase6Tests(unittest.TestCase):
    def test_group_holdout_has_no_target_overlap(self):
        groups = np.asarray(["A", "A", "B", "C", "C", "D", "E", "F"])
        development, holdout = stable_group_holdout(groups, 0.25, 123)
        self.assertFalse(set(groups[development]) & set(groups[holdout]))
        self.assertTrue(development.any())
        self.assertTrue(holdout.any())

    def test_actor_starts_exactly_from_teacher_prior(self):
        model = CandidateSetActorCritic(candidate_dim=7, action_count=4, hidden_dim=16, heads=4, layers=1)
        candidates = torch.randn(3, 4, 7)
        prior = torch.randn(3, 4)
        current = torch.tensor([0, 1, 2])
        logits, values = model(candidates, prior, current, torch.tensor([0.0, 1.0, 1.0]))
        self.assertTrue(torch.allclose(logits, prior * model.prior_scale))
        self.assertEqual(values.shape, (3,))

    def test_paired_statistics_count_discordant_pairs(self):
        candidate = np.asarray([True, True, False, True])
        reference = np.asarray([True, False, True, False])
        result = mcnemar_exact(candidate, reference)
        self.assertEqual(result["fixed"], 2)
        self.assertEqual(result["broken"], 1)
        bootstrap = paired_bootstrap(candidate, reference, candidate.astype(float), reference.astype(float), samples=100)
        self.assertAlmostEqual(bootstrap["delta_exact"], 0.25)

    def test_candidate_ocr_features_are_label_free(self):
        cache = {
            "predictions": np.asarray([["12A", "12A", "12B"], ["34C", "34D", "34D"]]),
            "normalized_confidence": np.asarray([[0.9, 0.8, 0.7], [0.6, 0.7, 0.8]], dtype=np.float32),
            "targets": np.asarray(["SECRET1", "SECRET2"]),
        }
        first = candidate_ocr_features(cache)
        cache["targets"] = np.asarray(["CHANGED", "LABELS"])
        second = candidate_ocr_features(cache)
        self.assertEqual(first.shape, (2, 3, 438))
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()

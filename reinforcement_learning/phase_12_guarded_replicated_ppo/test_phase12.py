from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from reinforcement_learning.phase_12_guarded_replicated_ppo.evaluate import (
    HERE as PHASE12_DIR,
    aligned_guard_metadata,
    claim,
    complete,
    guard_sha256,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.prepare_fresh_holdout import (
    validate_lock,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.pool_status import (
    SHEET_ROW_DIRECTION,
    build_parser as build_status_parser,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.promote import validate_promotion_summary
from reinforcement_learning.phase_12_guarded_replicated_ppo.runtime import (
    expand_allowed_view,
    read_runtime_manifest,
)
from reinforcement_learning.phase_12_guarded_replicated_ppo.selection import guarded_selection


class Phase12Tests(unittest.TestCase):
    def test_current_lock_matches_policy_action_registry_and_guard(self):
        lock = validate_lock(PHASE12_DIR / "prospective_policy.json")
        self.assertEqual(lock["guard"]["required_input_transform"], "existing_plate_crop")
        self.assertEqual(lock["guard"]["minimum_side_exclusive_upper_bound"], 128)
        self.assertEqual(
            lock["external_contract"]["sheet_row_direction"],
            "toward_larger_row_numbers",
        )

    def test_source_contract_means_row_733_and_rows_below(self):
        args = build_status_parser().parse_args([])
        self.assertEqual(args.first_sheet_row, 733)
        self.assertEqual(SHEET_ROW_DIRECTION, "toward_larger_row_numbers")

    def test_guard_requires_existing_crop_below_128_min_side(self):
        selected, allowed = guarded_selection(
            np.asarray([2, 3, 4, 5]),
            np.asarray(["existing_plate_crop", "crop_source_bounding_box", "existing_plate_crop", "existing_plate_crop"]),
            np.asarray([80, 80, 200, 300]),
            np.asarray([40, 40, 127, 128]),
        )
        np.testing.assert_array_equal(allowed, np.asarray([True, False, True, False]))
        np.testing.assert_array_equal(selected, np.asarray([2, 0, 4, 0]))

    def test_guard_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            guarded_selection(
                np.asarray([1]),
                np.asarray(["existing_plate_crop"]),
                np.asarray([0]),
                np.asarray([20]),
            )

    def test_external_guard_metadata_aligns_to_cache_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            pd.DataFrame(
                {
                    "image_path": ["b.jpg", "a.jpg"],
                    "input_transform": ["crop_source_bounding_box", "existing_plate_crop"],
                    "crop_width": [200, 80],
                    "crop_height": [100, 40],
                }
            ).to_csv(manifest, index=False)
            aligned = aligned_guard_metadata(manifest, np.asarray(["a.jpg", "b.jpg"]))
            self.assertEqual(aligned.image_path.tolist(), ["a.jpg", "b.jpg"])
            self.assertEqual(aligned.crop_width.tolist(), [80, 200])

    def test_runtime_manifest_uses_actual_dimensions_and_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "plate.jpg"
            Image.new("RGB", (80, 40), "white").save(image)
            manifest = root / "runtime.csv"
            pd.DataFrame(
                {
                    "image_path": [str(image)],
                    "input_contract": ["plate_crop"],
                    "input_transform": ["existing_plate_crop"],
                    "crop_width": [80],
                    "crop_height": [40],
                }
            ).to_csv(manifest, index=False)
            frame = read_runtime_manifest(manifest)
            self.assertEqual(frame.crop_width.tolist(), [80])
            self.assertEqual(frame.crop_height.tolist(), [40])
            frame.loc[0, "crop_width"] = 81
            frame.to_csv(manifest, index=False)
            with self.assertRaises(ValueError):
                read_runtime_manifest(manifest)

    def test_guard_short_circuit_expands_only_allowed_rows(self):
        baseline_features = np.asarray([[1.0], [2.0]])
        baseline_predictions = np.asarray(["BASE1", "BASE2"])
        baseline_confidence = np.asarray([0.8, 0.9])
        allowed_result = (
            np.asarray([[9.0]]),
            np.asarray(["ALT2"]),
            np.asarray([0.95]),
            {
                "view": "alternate",
                "batch_seconds": [0.01],
                "batch_sizes": [1],
                "amortized_ms_per_image": [10.0],
            },
        )
        features, predictions, confidence, timing = expand_allowed_view(
            baseline_features,
            baseline_predictions,
            baseline_confidence,
            np.asarray([False, True]),
            allowed_result,
            "alternate",
        )
        np.testing.assert_array_equal(features, np.asarray([[1.0], [9.0]]))
        np.testing.assert_array_equal(predictions, np.asarray(["BASE1", "ALT2"]))
        np.testing.assert_array_equal(confidence, np.asarray([0.8, 0.95]))
        self.assertEqual(timing["amortized_ms_per_image"], [0.0, 10.0])

    def test_promotion_requires_locked_guard_and_full_power_target(self):
        guard = {
            "required_input_transform": "existing_plate_crop",
            "minimum_side_exclusive_upper_bound": 128,
            "fallback_action": "baseline",
            "label_free": True,
        }
        valid = {
            "algorithm": "guarded_replicated_candidate_oof_ppo_seed728",
            "evaluation_role": "locked_confirmatory",
            "split": "external_holdout",
            "status": "eligible",
            "promotion_eligible": True,
            "formal_improvement_gate_vs_baseline": {"passed": True},
            "external_holdout_evaluated_once": True,
            "test_used": False,
            "audited_legacy_test_loaded": False,
            "candidate_lock": {"status": "prospective_locked_waiting_for_fresh_data"},
            "guard": {**guard, "sha256": guard_sha256(guard)},
            "cache": {
                "summary": {
                    "samples": 1500,
                    "manifest": {"group_disjoint": True, "input_contract": "plate_crop"},
                    "power_contract": {"formal_ready": True},
                    "checkpoint_sha256": "a" * 64,
                    "artifacts": {
                        "candidate_features": {},
                        "state_features": {},
                        "action_trajectories": {},
                    },
                }
            },
            "confirmatory_receipt": {"one_shot": True, "path": "receipt", "claim_id": "claim"},
        }
        validate_promotion_summary(valid)
        valid["cache"]["summary"]["samples"] = 1499
        with self.assertRaises(ValueError):
            validate_promotion_summary(valid)

    def test_confirmatory_receipt_is_exclusive_and_locks_guard(self):
        with tempfile.TemporaryDirectory(dir=PHASE12_DIR) as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            lock = root / "lock.json"
            policy = root / "policy.pt"
            parseq = root / "parseq.pt"
            for path in (manifest, lock, policy, parseq):
                path.write_text(path.name, encoding="utf-8")
            digest = lambda path: __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            cache = {
                "manifest": {"path": str(manifest), "sha256": digest(manifest)},
                "checkpoint": str(parseq),
                "checkpoint_sha256": digest(parseq),
                "artifacts": {
                    "candidate_features": {"sha256": "1" * 64},
                    "state_features": {"sha256": "2" * 64},
                    "action_trajectories": {"sha256": "3" * 64},
                },
            }
            guard = {"label_free": True}
            receipt_path = root / "receipt.json"
            output = root / "evaluation"
            started = claim(receipt_path, output, cache, lock, policy, guard)
            self.assertEqual(started["guard_sha256"], guard_sha256(guard))
            with self.assertRaises(FileExistsError):
                claim(receipt_path, output, cache, lock, policy, guard)
            output.mkdir()
            summary = output / "summary.json"
            summary.write_text("{}", encoding="utf-8")
            complete(receipt_path, started, summary, False)
            completed = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertFalse(completed["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from reinforcement_learning.phase_9_primary_ppo.evaluate import (
    HERE as PHASE9_DIR,
    claim_evaluation,
    complete_evaluation,
)
from reinforcement_learning.phase_9_primary_ppo.prepare_fresh_holdout import (
    candidate_rows,
    validate_candidate_lock,
)
from reinforcement_learning.phase_9_primary_ppo.promote import validate_promotion_summary


class Phase9Tests(unittest.TestCase):
    def test_current_prospective_lock_matches_checkpoint_and_action_registry(self):
        lock = validate_candidate_lock(PHASE9_DIR / "prospective_policy.json")
        self.assertEqual(lock["algorithm"], "single_primary_candidate_oof_ppo")
        self.assertTrue(lock["selection_provenance"]["selected_after_opening_phase8_holdout"])
        self.assertFalse(lock["external_contract"]["manual_acceptance_required"])

    def test_candidate_rows_trust_extracted_character_without_manual_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = [
                "id",
                "image",
                "extracted_character",
                "review_status",
                "source",
                "cropped",
                "bounding_box",
            ]
            rows = [
                ["before", "before.jpg", "11A11111", "accepted", "test", "true", "[0,0,1,1]"],
                ["accepted", "accepted.jpg", "22B22222", "accepted", "test", "true", "[0,0,1,1]"],
                ["blank", "blank.jpg", "33C33333", "", "test", "true", "[0,0,1,1]"],
                ["corrected", "corrected.jpg", "44D44444", "corrected", "test", "true", "[0,0,1,1]"],
                ["rejected", "rejected.jpg", "55E55555", "rejected", "test", "true", "[0,0,1,1]"],
                ["duplicate", "duplicate.jpg", "33-C33333", "", "test", "true", "[0,0,1,1]"],
                ["opened", "opened.jpg", "66F66666", "", "test", "true", "[0,0,1,1]"],
                ["phase8", "phase8.jpg", "77G77777", "", "test", "true", "[0,0,1,1]"],
            ]
            for row in rows:
                (root / row[1]).touch()
            labels = root / "labels.csv"
            with labels.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(fields)
                writer.writerows(rows)
            selected, counts = candidate_rows(
                labels,
                root,
                first_sheet_row=3,
                excluded_labels={"66F66666"},
                excluded_source_ids={"phase8"},
            )
            self.assertEqual(
                {row["source_id"] for row in selected},
                {"accepted", "blank", "corrected"},
            )
            self.assertEqual(counts["review_status=rejected"], 1)
            self.assertEqual(counts["duplicate_candidate_label"], 1)
            self.assertEqual(counts["label_historical_or_opened"], 1)
            self.assertEqual(counts["source_already_in_phase8_queue"], 1)

    def test_promotion_summary_requires_every_formal_gate(self):
        valid = {
            "algorithm": "single_primary_candidate_oof_ppo",
            "evaluation_role": "locked_confirmatory",
            "split": "external_holdout",
            "status": "eligible",
            "promotion_eligible": True,
            "formal_improvement_gate_vs_baseline": {"passed": True},
            "external_holdout_evaluated_once": True,
            "test_used": False,
            "audited_legacy_test_loaded": False,
            "candidate_lock": {"status": "prospective_locked_requires_new_external"},
            "cache": {
                "summary": {
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
        valid["formal_improvement_gate_vs_baseline"]["passed"] = False
        with self.assertRaises(ValueError):
            validate_promotion_summary(valid)

    def test_confirmatory_receipt_is_exclusive(self):
        with tempfile.TemporaryDirectory(dir=PHASE9_DIR) as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            lock = root / "lock.json"
            policy = root / "policy.pt"
            parseq = root / "parseq.pt"
            for path in (manifest, lock, policy, parseq):
                path.write_text(path.name, encoding="utf-8")
            import hashlib

            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
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
            receipt_path = root / "receipt.json"
            output = root / "evaluation"
            started = claim_evaluation(receipt_path, output, cache, lock, policy)
            with self.assertRaises(FileExistsError):
                claim_evaluation(receipt_path, output, cache, lock, policy)
            output.mkdir()
            summary = output / "summary.json"
            summary.write_text("{}", encoding="utf-8")
            complete_evaluation(receipt_path, started, summary, False)
            import json

            completed = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertFalse(completed["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()

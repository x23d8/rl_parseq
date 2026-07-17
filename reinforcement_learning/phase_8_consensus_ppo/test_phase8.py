from __future__ import annotations

import unittest
import copy
import tempfile
from pathlib import Path

import numpy as np

from reinforcement_learning.phase_8_consensus_ppo.evaluate import (
    HERE as PHASE8_DIR,
    attach_external_metadata,
    claim_confirmatory_evaluation,
    complete_confirmatory_evaluation,
    descriptive_slice_metrics,
    prediction_agreement_selection,
    validate_locked_external_metadata,
)
from reinforcement_learning.phase_8_consensus_ppo.promote import (
    validate_confirmatory_receipt,
    validate_promotion_summary,
)
from reinforcement_learning.phase_8_consensus_ppo.preflight_fresh_review_queue import audit_media_rows
from reinforcement_learning.phase_8_consensus_ppo.prepare_runtime_crops import run as prepare_runtime_crops
from reinforcement_learning.phase_8_consensus_ppo.runtime import read_plate_crop_manifest
from reinforcement_learning.phase_7_compact_multiscale_ppo.runtime import summarize_runtime_latency
from reinforcement_learning.phase_8_consensus_ppo.finalize_fresh_holdout import validate_review_rows
from reinforcement_learning.phase_8_consensus_ppo.review_contract import assess_review_rows
from reinforcement_learning.phase_8_consensus_ppo.review_server import ReviewStore
from reinforcement_learning.phase_6_candidate_oof_ppo.train import policy_selection


class FixedPolicyForSafetyGate:
    prior_scale = 0.0

    def eval(self):
        return self

    def __call__(self, candidates, priors, current, step):
        import torch

        logits = torch.zeros((len(candidates), 2), device=candidates.device)
        logits[:, 1] = 1.0
        return logits, torch.zeros(len(candidates), device=candidates.device)


class Phase8Tests(unittest.TestCase):
    def test_final_teacher_gain_gate_rolls_unsafe_action_back_to_baseline(self):
        import torch

        candidates = torch.zeros((2, 2, 1))
        priors = torch.tensor([[0.0, -0.1], [0.0, 0.2]])
        first, selected, revised = policy_selection(
            FixedPolicyForSafetyGate(),
            candidates,
            priors,
            first_margin=0.0,
            revise_margin=10.0,
            device=torch.device("cpu"),
            teacher_margin=0.0,
            disagreement_margin=None,
            final_teacher_gain_margin=0.05,
        )
        np.testing.assert_array_equal(first, np.asarray([0, 1]))
        np.testing.assert_array_equal(selected, np.asarray([0, 1]))
        np.testing.assert_array_equal(revised, np.asarray([False, False]))

    def test_prediction_agreement_requires_same_nonbaseline_prediction(self):
        predictions = np.asarray(
            [
                ["BASE0", "FIX0", "ALT0"],
                ["BASE1", "SAME1", "SAME1"],
                ["BASE2", "ALT2", "BASE2"],
                ["BASE3", "BAD3", "BAD3"],
            ]
        )
        selected_a = np.asarray([1, 1, 1, 1])
        selected_b = np.asarray([2, 2, 2, 1])
        selected, changed = prediction_agreement_selection(predictions, selected_a, selected_b)
        np.testing.assert_array_equal(selected, np.asarray([0, 2, 0, 1]))
        np.testing.assert_array_equal(changed, np.asarray([False, True, False, True]))

    def test_diagnostic_evaluation_cannot_promote(self):
        valid = {
            "algorithm": "dual_seed_ppo_prediction_agreement_consensus",
            "evaluation_role": "locked_confirmatory",
            "split": "external_holdout",
            "status": "eligible",
            "promotion_eligible": True,
            "formal_improvement_gate_vs_baseline": {"passed": True},
            "external_holdout_evaluated_once": True,
            "audited_legacy_test_loaded": False,
            "test_used": False,
            "confirmatory_receipt": {
                "path": "receipt.json",
                "claim_id": "claim",
                "one_shot": True,
            },
            "candidate_lock": {"status": "prospective_locked_for_fresh_external"},
            "cache": {
                "summary": {
                    "manifest": {"group_disjoint": True, "input_contract": "plate_crop"},
                    "power_contract": {"formal_ready": True},
                    "checkpoint_sha256": "a" * 64,
                    "artifacts": {
                        "candidate_features": {"sha256": "b" * 64},
                        "state_features": {"sha256": "c" * 64},
                        "action_trajectories": {"sha256": "d" * 64},
                    },
                }
            },
        }
        validate_promotion_summary(valid)
        diagnostic = copy.deepcopy(valid)
        diagnostic["evaluation_role"] = "protocol_repair_diagnostic"
        with self.assertRaises(ValueError):
            validate_promotion_summary(diagnostic)
        missing_receipt = copy.deepcopy(valid)
        missing_receipt.pop("confirmatory_receipt")
        with self.assertRaises(ValueError):
            validate_promotion_summary(missing_receipt)

    def test_runtime_requires_explicit_plate_crop_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "plate.jpg"
            image.touch()
            manifest = root / "runtime.csv"
            manifest.write_text(
                f"image_path,input_contract\n{image},plate_crop\n", encoding="utf-8"
            )
            self.assertEqual(len(read_plate_crop_manifest(manifest)), 1)
            manifest.write_text(
                f"image_path,input_contract\n{image},full_image\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                read_plate_crop_manifest(manifest)

    def test_runtime_latency_reports_full_candidate_cost(self):
        timing = summarize_runtime_latency(
            [
                {
                    "view": "baseline",
                    "batch_seconds": [0.003],
                    "batch_sizes": [2],
                    "amortized_ms_per_image": [1.0, 2.0],
                },
                {
                    "view": "alternate",
                    "batch_seconds": [0.007],
                    "batch_sizes": [2],
                    "amortized_ms_per_image": [3.0, 4.0],
                },
            ],
            policy_seconds=0.002,
            total_seconds=0.012,
            samples=2,
        )
        self.assertEqual(timing["candidate_views_evaluated_per_image"], 2)
        self.assertAlmostEqual(timing["mean_wall_ms_per_image"], 6.0)
        self.assertAlmostEqual(timing["p95_batch_amortized_ms_per_image"], 6.9)
        self.assertAlmostEqual(timing["throughput_images_per_second"], 2 / 0.012)

    def test_detector_adapter_creates_label_free_plate_crop_manifest(self):
        with tempfile.TemporaryDirectory(dir=PHASE8_DIR) as temporary:
            root = Path(temporary)
            source_image = root / "source.jpg"
            from PIL import Image

            Image.new("RGB", (100, 50), "red").save(source_image)
            detector_manifest = root / "detector.csv"
            detector_manifest.write_text(
                f'image_path,bounding_box,source_id\n{source_image},"[10, 5, 60, 35]",plate_1\n',
                encoding="utf-8",
            )
            import argparse

            output_dir = root / "crops"
            audit = prepare_runtime_crops(
                argparse.Namespace(manifest=str(detector_manifest), output_dir=str(output_dir))
            )
            self.assertTrue(audit["label_free"])
            runtime_manifest = Path(audit["runtime_manifest"]["path"])
            frame = read_plate_crop_manifest(runtime_manifest)
            self.assertEqual(len(frame), 1)
            import pandas as pd

            detector_frame = pd.read_csv(runtime_manifest)
            self.assertEqual(detector_frame.input_transform.tolist(), ["crop_source_bounding_box"])
            self.assertEqual(detector_frame.crop_width.tolist(), [50])
            self.assertEqual(detector_frame.crop_height.tolist(), [30])
            with Image.open(frame.iloc[0].image_path) as crop:
                self.assertEqual(crop.size, (50, 30))

            invalid_manifest = root / "invalid.csv"
            invalid_manifest.write_text(
                f'image_path,bounding_box,source_id\n{source_image},"[0, 0, 20, 20]",../escape\n',
                encoding="utf-8",
            )
            invalid_output = root / "invalid_output"
            with self.assertRaises(ValueError):
                prepare_runtime_crops(
                    argparse.Namespace(manifest=str(invalid_manifest), output_dir=str(invalid_output))
                )
            self.assertFalse(invalid_output.exists())

    def test_external_metadata_supports_descriptive_error_slices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "image_path,source,input_transform,crop_width,crop_height\n"
                "a.jpg,camera_a,existing_plate_crop,80,30\n"
                "b.jpg,camera_b,crop_source_bounding_box,160,80\n",
                encoding="utf-8",
            )
            validate_locked_external_metadata(
                {"manifest": {"path": str(manifest)}, "samples": 2}
            )
            import pandas as pd

            frame = pd.DataFrame(
                {
                    "image_path": ["a.jpg", "b.jpg"],
                    "target": ["ABC123", "XYZ789"],
                    "exact": [True, False],
                    "baseline_exact": [False, False],
                    "edit_distance": [0, 1],
                    "fixed": [True, False],
                    "broken": [False, False],
                }
            )
            enriched = attach_external_metadata(frame, manifest)
            self.assertEqual(
                enriched.crop_size_bucket.tolist(),
                ["min_side_lt32", "min_side_64_127"],
            )
            slices = descriptive_slice_metrics(enriched, "source")
            self.assertEqual(slices["camera_a"]["fixed"], 1)
            self.assertEqual(slices["camera_b"]["consensus_exact"], 0.0)

    def test_review_finalizer_preserves_source_rows_and_requires_final_labels(self):
        base = {
            "source_id": "fresh_1",
            "source_image_path": "plate.jpg",
            "current_extracted_character": "51-A12345",
            "review_decision": "",
            "corrected_label": "",
        }
        original = [base, {**base, "source_id": "fresh_2", "current_extracted_character": "59B99999"}]
        reviewed = [
            {**original[0], "review_decision": "accepted"},
            {**original[1], "review_decision": "corrected", "corrected_label": "59B88888"},
        ]
        selected = validate_review_rows(original, reviewed, set(), minimum_samples=2)
        self.assertEqual([row["final_label"] for row in selected], ["51A12345", "59B88888"])
        changed = copy.deepcopy(reviewed)
        changed[0]["source_image_path"] = "other.jpg"
        with self.assertRaises(ValueError):
            validate_review_rows(original, changed, set(), minimum_samples=2)
        overlap = copy.deepcopy(reviewed)
        overlap_selected, assessment, statuses = assess_review_rows(
            original, overlap, {"51A12345"}, minimum_samples=1
        )
        self.assertEqual([row["final_label"] for row in overlap_selected], ["59B88888"])
        self.assertEqual(assessment["excluded_label_overlap"], 1)
        self.assertEqual(statuses[0]["status"], "excluded_label_overlap")
        blank = copy.deepcopy(reviewed)
        blank[1]["corrected_label"] = ""
        with self.assertRaises(ValueError):
            validate_review_rows(original, blank, set(), minimum_samples=2)

    def test_review_store_updates_only_review_fields_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = ["source_id", "review_decision", "corrected_label"]
            original = root / "original.csv"
            reviewed = root / "reviewed.csv"
            content = "source_id,review_decision,corrected_label\nfresh_1,,\n"
            original.write_text(content, encoding="utf-8")
            reviewed.write_text(content, encoding="utf-8")
            store = ReviewStore(original, reviewed)
            result = store.update(0, "corrected", "51-a12345")
            self.assertTrue(result["saved"])
            self.assertEqual(result["progress"]["eligible_unique"], 1)
            saved = read_csv_for_test(reviewed)
            backup = reviewed.with_suffix(reviewed.suffix + ".bak")
            self.assertTrue(backup.is_file())
            self.assertEqual(read_csv_for_test(backup)[0]["review_decision"], "")
            self.assertEqual(saved[0]["source_id"], "fresh_1")
            self.assertEqual(saved[0]["review_decision"], "corrected")
            self.assertEqual(saved[0]["corrected_label"], "51A12345")
            with self.assertRaises(ValueError):
                store.update(0, "corrected", "")
            with self.assertRaises(ValueError):
                store.update(0, "corrected", "51Đ12345")
            with self.assertRaises(ValueError):
                store.update(0, "corrected", "1234567890123")

    def test_review_contract_keeps_one_unique_nonoverlapping_group(self):
        base = {
            "source_id": "fresh_1",
            "current_extracted_character": "51A11111",
            "review_decision": "",
            "corrected_label": "",
        }
        original = [
            base,
            {**base, "source_id": "fresh_2", "current_extracted_character": "59B22222"},
            {**base, "source_id": "fresh_3", "current_extracted_character": "30C33333"},
        ]
        reviewed = [
            {**original[0], "review_decision": "accepted"},
            {**original[1], "review_decision": "corrected", "corrected_label": "51A11111"},
            {**original[2], "review_decision": "corrected", "corrected_label": "88D88888"},
        ]
        selected, assessment, statuses = assess_review_rows(
            original, reviewed, {"88D88888"}, minimum_samples=1
        )
        self.assertEqual([row["final_label"] for row in selected], ["51A11111"])
        self.assertEqual(assessment["eligible_unique"], 1)
        self.assertEqual(assessment["excluded_duplicate_label"], 1)
        self.assertEqual(assessment["excluded_label_overlap"], 1)
        self.assertEqual(statuses[1]["first_queue_index"], 1)

        with_blank_tail = [
            {**original[0], "review_decision": "accepted"},
            {**original[1], "review_decision": "accepted"},
            original[2],
        ]
        _, prefix_assessment, _ = assess_review_rows(
            original, with_blank_tail, set(), minimum_samples=2
        )
        self.assertTrue(prefix_assessment["formal_ready"])
        self.assertFalse(prefix_assessment["review_complete"])
        self.assertEqual(prefix_assessment["reviewed_prefix"], 2)

    def test_review_context_prefers_visual_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_image = root / "source.jpg"
            visual_image = root / "visual.jpg"
            source_image.touch()
            visual_image.touch()
            fields = [
                "source_id",
                "source_image_path",
                "visual_image_path",
                "review_decision",
                "corrected_label",
            ]
            original = root / "original.csv"
            reviewed = root / "reviewed.csv"
            import csv

            for path in (original, reviewed):
                with path.open("w", encoding="utf-8", newline="") as destination:
                    writer = csv.DictWriter(destination, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "source_id": "fresh_1",
                            "source_image_path": str(source_image),
                            "visual_image_path": str(visual_image),
                            "review_decision": "",
                            "corrected_label": "",
                        }
                    )
            store = ReviewStore(original, reviewed)
            self.assertEqual(store.context_path(0), visual_image)

    def test_confirmatory_receipt_is_exclusive_and_completable(self):
        with tempfile.TemporaryDirectory(dir=PHASE8_DIR) as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            lock = root / "lock.json"
            checkpoint_a = root / "a.pt"
            checkpoint_b = root / "b.pt"
            for path, value in (
                (manifest, "manifest"),
                (lock, "lock"),
                (checkpoint_a, "a"),
                (checkpoint_b, "b"),
            ):
                path.write_text(value, encoding="utf-8")
            import hashlib

            cache_summary = {
                "manifest": {
                    "path": str(manifest),
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "checkpoint": str(checkpoint_a),
                "checkpoint_sha256": hashlib.sha256(checkpoint_a.read_bytes()).hexdigest(),
                "artifacts": {
                    "candidate_features": {"sha256": "1" * 64},
                    "state_features": {"sha256": "2" * 64},
                    "action_trajectories": {"sha256": "3" * 64},
                },
            }
            receipt_path = root / "receipt.json"
            output_dir = root / "evaluation"
            receipt = claim_confirmatory_evaluation(
                receipt_path, output_dir, cache_summary, lock, [checkpoint_a, checkpoint_b]
            )
            with self.assertRaises(FileExistsError):
                claim_confirmatory_evaluation(
                    receipt_path, output_dir, cache_summary, lock, [checkpoint_a, checkpoint_b]
                )
            summary_path = output_dir / "summary.json"
            output_dir.mkdir()
            summary = {
                "candidate_lock": {"sha256": hashlib.sha256(lock.read_bytes()).hexdigest()},
                "cache": {"summary": cache_summary},
                "promotion_eligible": True,
                "confirmatory_receipt": {
                    "path": str(receipt_path),
                    "claim_id": receipt["claim_id"],
                    "one_shot": True,
                },
            }
            import json

            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            complete_confirmatory_evaluation(receipt_path, receipt, summary_path, True)
            completed = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertTrue(completed["promotion_eligible"])
            self.assertEqual(
                validate_confirmatory_receipt(summary, summary_path.resolve())["claim_id"],
                receipt["claim_id"],
            )
            tampered = copy.deepcopy(summary)
            tampered["cache"]["summary"]["artifacts"]["state_features"]["sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                validate_confirmatory_receipt(tampered, summary_path.resolve())

    def test_media_preflight_detects_exact_overlap_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            image_a.write_bytes(b"same-plate-bytes")
            image_b.write_bytes(b"same-plate-bytes")
            rows = [
                {
                    "source_id": "a",
                    "source_image_path": str(image_a),
                    "required_input_transform": "existing_plate_crop",
                },
                {
                    "source_id": "b",
                    "source_image_path": str(image_b),
                    "required_input_transform": "existing_plate_crop",
                },
            ]
            import hashlib

            digest = hashlib.sha256(image_a.read_bytes()).hexdigest()
            audit = audit_media_rows(rows, {digest})
            self.assertFalse(audit["passed"])
            self.assertEqual(audit["prior_exact_overlap_count"], 2)
            self.assertEqual(audit["queue_exact_duplicate_count"], 1)
            self.assertEqual(audit["render_error_count"], 0)


def read_csv_for_test(path: Path):
    import csv

    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


if __name__ == "__main__":
    unittest.main()

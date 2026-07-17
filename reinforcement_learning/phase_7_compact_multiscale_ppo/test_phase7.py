"""Integration contracts for the locked compact multiscale action space."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reinforcement_learning.phase_7_compact_multiscale_ppo.action_space import COMPACT_VIEWS


HERE = Path(__file__).resolve().parent


class Phase7Tests(unittest.TestCase):
    def test_compact_action_registry_is_unique_and_baseline_first(self):
        names = [view.name for view in COMPACT_VIEWS]
        self.assertEqual(len(names), 9)
        self.assertEqual(len(set(names)), 9)
        self.assertEqual(names[0], "baseline")
        self.assertEqual(COMPACT_VIEWS[0].cost, 0.0)

    def test_cached_validation_retains_full_65_view_oracle(self):
        summary = json.loads((HERE / "results/cache/val_cache_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["test_loaded"])
        self.assertAlmostEqual(summary["baseline_exact"], 0.9269521410579346)
        self.assertAlmostEqual(summary["oracle_exact"], 0.9773299748110831)
        payload = np.load(HERE / "results/cache/val_candidate_features.npz", allow_pickle=False)
        self.assertEqual(payload["candidate_features"].shape, (397, 9, 781))
        self.assertTrue(np.isfinite(payload["candidate_features"]).all())

    def test_confirmatory_run_did_not_load_test_or_overlap_groups(self):
        summary = json.loads(
            (HERE / "results/confirmatory_seed_728/summary.json").read_text(encoding="utf-8")
        )
        self.assertFalse(summary["test_used"])
        self.assertFalse(summary["protocol"]["audited_test_loaded"])
        self.assertEqual(summary["protocol"]["group_overlap"], 0)
        self.assertGreater(summary["holdout_statistics_vs_teacher"]["paired_bootstrap"]["delta_exact"], 0)

    def test_external_evaluator_requires_locked_manifest_provenance(self):
        from reinforcement_learning.phase_7_compact_multiscale_ppo.evaluate_external import (
            validate_cache_artifacts,
            validate_cache_checkpoint,
            validate_external_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                validate_external_cache(cache_dir)
            (cache_dir / "external_holdout_cache_summary.json").write_text(
                json.dumps({"split": "external_holdout", "samples": 2, "test_loaded": False}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_external_cache(cache_dir)
            expected = {
                "split": "external_holdout",
                "samples": 2,
                "test_loaded": False,
                "group_audit": {
                    "group_overlap": 0,
                    "historical_labels_used_for_exclusion_only": True,
                },
                "manifest": {
                    "external_contract": True,
                    "group_disjoint": True,
                    "input_contract": "plate_crop",
                    "path": str(cache_dir / "external_manifest.csv"),
                    "sha256": "",
                },
            }
            manifest_path = Path(expected["manifest"]["path"])
            manifest_path.write_text("image_path,label,split\nimage.png,ABC123,external_holdout\n", encoding="utf-8")
            expected["manifest"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (cache_dir / "external_holdout_cache_summary.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            self.assertEqual(validate_external_cache(cache_dir), expected)
            manifest_path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_external_cache(cache_dir)

            checkpoint = cache_dir / "parseq.pt"
            checkpoint.write_bytes(b"locked-parseq")
            expected["checkpoint"] = str(checkpoint)
            expected["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(validate_cache_checkpoint(expected, required=True), checkpoint.resolve())
            checkpoint.write_bytes(b"changed-parseq")
            with self.assertRaises(ValueError):
                validate_cache_checkpoint(expected, required=True)

            artifacts = {}
            for name, filename in {
                "candidate_features": "external_holdout_candidate_features.npz",
                "state_features": "external_holdout_state_features.npz",
                "action_trajectories": "external_holdout_action_trajectories.csv",
            }.items():
                path = cache_dir / filename
                path.write_bytes(name.encode("ascii"))
                artifacts[name] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            expected["artifacts"] = artifacts
            self.assertEqual(validate_cache_artifacts(cache_dir, expected, required=True), artifacts)
            (cache_dir / "external_holdout_state_features.npz").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                validate_cache_artifacts(cache_dir, expected, required=True)

    def test_external_manifest_preflight_never_runs_inference(self):
        import pandas as pd

        from reinforcement_learning.phase_7_compact_multiscale_ppo.build_cache import external_manifest_preflight

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "plate.png"
            image_path.touch()
            manifest_path = root / "external_manifest.csv"
            manifest_path.write_text(
                "image_path,label,split\n" + f"{image_path},ABC123,external_holdout\n", encoding="utf-8"
            )
            frame = pd.DataFrame({"image_path": [str(image_path)], "label": ["ABC123"]})
            audit = external_manifest_preflight(frame, manifest_path, source_rows=1)
            self.assertTrue(audit["ready_for_external_cache"])
            self.assertFalse(audit["inference_run"])
            self.assertFalse(audit["artifacts_written"])
            image_path.unlink()
            with self.assertRaises(FileNotFoundError):
                external_manifest_preflight(frame, manifest_path, source_rows=1)

    def test_external_manifest_must_be_group_disjoint(self):
        import pandas as pd

        from reinforcement_learning.phase_7_compact_multiscale_ppo.build_cache import (
            validate_external_group_disjoint,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical_path = root / "historical.csv"
            historical_path.write_text(
                "image_path,label,split\ntrain.png,ABC-123,train\ntest.png,XYZ999,test\n",
                encoding="utf-8",
            )
            clean = pd.DataFrame({"label": ["NEW456"]})
            audit = validate_external_group_disjoint(clean, historical_path)
            self.assertEqual(audit["group_overlap"], 0)
            self.assertTrue(audit["historical_labels_used_for_exclusion_only"])
            overlapping = pd.DataFrame({"label": ["abc123"]})
            with self.assertRaises(ValueError):
                validate_external_group_disjoint(overlapping, historical_path)

    def test_external_input_and_power_contracts(self):
        import pandas as pd

        from reinforcement_learning.phase_7_compact_multiscale_ppo.build_cache import (
            MIN_FORMAL_EXTERNAL_SAMPLES,
            external_power_contract,
            validate_external_input_contract,
        )

        validate_external_input_contract(pd.DataFrame({"input_contract": ["plate_crop", "PLATE_CROP"]}))
        with self.assertRaises(ValueError):
            validate_external_input_contract(pd.DataFrame({"input_contract": ["full_image"]}))
        with self.assertRaises(ValueError):
            external_power_contract(MIN_FORMAL_EXTERNAL_SAMPLES - 1, False)
        diagnostic = external_power_contract(154, True)
        self.assertFalse(diagnostic["formal_ready"])
        self.assertTrue(external_power_contract(MIN_FORMAL_EXTERNAL_SAMPLES, False)["formal_ready"])

    def test_promotion_requires_external_gate_and_runtime_metadata(self):
        from reinforcement_learning.phase_7_compact_multiscale_ppo.promote import validate_promotion_summary

        with self.assertRaises(ValueError):
            validate_promotion_summary({"promotion_status": "external_holdout_failed_gate"})
        valid = {
            "promotion_status": "eligible",
            "formal_improvement_gate": {"passed": True},
            "external_holdout_evaluated_once": True,
            "audited_legacy_test_loaded": False,
            "evaluation_role": "locked_confirmatory",
            "external_cache": {"checkpoint": "parseq.pt", "actions": ["baseline"]},
        }
        validate_promotion_summary(valid)
        valid["external_cache"] = {"checkpoint": "parseq.pt"}
        with self.assertRaises(ValueError):
            validate_promotion_summary(valid)
        valid["external_cache"] = {"checkpoint": "parseq.pt", "actions": ["baseline"]}
        valid["evaluation_role"] = "protocol_repair_diagnostic"
        with self.assertRaises(ValueError):
            validate_promotion_summary(valid)


if __name__ == "__main__":
    unittest.main()

"""Build the no-review, fresh 1,500-sample Phase 11 external holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_8_consensus_ppo.finalize_fresh_holdout import (  # noqa: E402
    manifest_image_hashes,
    render_plate_crop,
)
from reinforcement_learning.phase_9_primary_ppo.prepare_fresh_holdout import (  # noqa: E402
    DEFAULT_HISTORICAL,
    DEFAULT_LABELS_CSV,
    DEFAULT_PHASE7_OPENED,
    DEFAULT_PHASE8_OPENED,
    DEFAULT_PHASE8_QUEUE,
    DEFAULT_SOURCE_ROOT,
    candidate_rows,
    read_manifest_labels,
    read_source_ids,
    safe_filename,
    sha256_file,
)


PHASE9_MANIFEST = (
    ROOT
    / "reinforcement_learning"
    / "phase_9_primary_ppo"
    / "fresh_external_holdout"
    / "external_manifest.csv"
)
DEFAULT_LOCK = HERE / "prospective_policy.json"
DEFAULT_OUTPUT = HERE / "fresh_external_holdout"
STABLE_SEED = "phase11-replicated-seed728-fresh-external-v1"


def validate_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("status") != "prospective_locked_requires_new_external"
        or lock.get("algorithm") != "replicated_primary_candidate_oof_ppo_seed728"
    ):
        raise ValueError("Phase 11 candidate lock is invalid")
    for field in ("policy_checkpoint", "action_registry"):
        entry = lock.get(field, {})
        artifact = (ROOT / entry.get("path", "")).resolve()
        if not artifact.is_file() or sha256_file(artifact) != entry.get("sha256"):
            raise ValueError(f"Locked Phase 11 {field} is unavailable or changed")
    return lock


def write_manifest(
    output_dir: Path,
    rendered: list[tuple[dict[str, str | int], bytes, str, str, int, int]],
) -> tuple[Path, list[dict[str, str | int]]]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, str | int]] = []
    for index, (source, payload, suffix, digest, width, height) in enumerate(rendered, start=1):
        destination = images_dir / safe_filename(index, str(source["source_id"]), suffix)
        destination.write_bytes(payload)
        rows.append(
            {
                "image_path": str(destination.resolve()),
                "label": source["normalized_label"],
                "split": "external_holdout",
                "input_contract": "plate_crop",
                "source_id": source["source_id"],
                "source_sheet_row": source["source_sheet_row"],
                "source": source["source"],
                "input_transform": source["required_input_transform"],
                "crop_width": width,
                "crop_height": height,
                "crop_sha256": digest,
                "label_source": "extracted_character",
                "source_review_status": source["review_status"],
            }
        )
    manifest = output_dir / "external_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


def run(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    labels_csv = Path(args.labels_csv).resolve()
    historical = Path(args.historical_manifest).resolve()
    phase7 = Path(args.phase7_opened_manifest).resolve()
    phase8 = Path(args.phase8_opened_manifest).resolve()
    phase9 = Path(args.phase9_opened_manifest).resolve()
    phase8_queue = Path(args.phase8_queue).resolve()
    lock_path = Path(args.candidate_lock).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir != HERE and HERE not in output_dir.parents:
        raise ValueError("Phase 11 holdout artifacts must remain inside Phase 11")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite locked Phase 11 holdout: {output_dir}")
    lock = validate_lock(lock_path)
    contract = lock["external_contract"]
    if (
        args.first_sheet_row != int(contract["first_sheet_row_inclusive"])
        or args.minimum_samples != int(contract["minimum_samples"])
        or args.target_samples != int(contract["target_samples"])
        or contract.get("manual_acceptance_required") is not False
    ):
        raise ValueError("Phase 11 builder arguments differ from the prospective lock")

    opened = [historical, phase7, phase8, phase9]
    excluded_labels = set().union(*(read_manifest_labels(path) for path in opened))
    excluded_source_ids = read_source_ids(phase8_queue) | read_source_ids(phase9)
    prior_hashes = set().union(*(manifest_image_hashes(path) for path in opened))
    candidates, exclusions = candidate_rows(
        labels_csv,
        source_root,
        args.first_sheet_row,
        excluded_labels,
        excluded_source_ids,
        stable_selection_seed=STABLE_SEED,
    )
    selected_hashes: set[str] = set()
    rendered: list[tuple[dict[str, str | int], bytes, str, str, int, int]] = []
    for row in candidates:
        try:
            payload, suffix = render_plate_crop(row)  # type: ignore[arg-type]
            digest = hashlib.sha256(payload).hexdigest()
            if digest in prior_hashes:
                exclusions["exact_image_historical_or_opened"] += 1
                continue
            if digest in selected_hashes:
                exclusions["duplicate_rendered_image"] += 1
                continue
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("empty crop")
        except Exception:
            exclusions["render_or_media_error"] += 1
            continue
        selected_hashes.add(digest)
        rendered.append((row, payload, suffix, digest, width, height))
        if len(rendered) == args.target_samples:
            break
    if len(rendered) < args.minimum_samples:
        raise ValueError(
            f"Only {len(rendered)} Phase 11 samples remain; need {args.minimum_samples}"
        )
    if len(rendered) < args.target_samples:
        raise ValueError(
            f"Power-locked Phase 11 target is {args.target_samples}, but only {len(rendered)} remain"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, rows = write_manifest(output_dir, rendered)
    audit = {
        "contract": "phase11_fresh_locked_confirmatory_manifest_v1",
        "inference_run": False,
        "candidate_locked_before_manifest": True,
        "candidate_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "label_contract": {
            "source": "extracted_character",
            "first_sheet_row_inclusive": args.first_sheet_row,
            "excluded_review_status": ["rejected"],
            "manual_acceptance_required": False,
            "corrected_status_uses_extracted_character": True,
        },
        "inputs": {
            "labels_csv": {"path": str(labels_csv), "sha256": sha256_file(labels_csv)},
            "phase8_queue": {"path": str(phase8_queue), "sha256": sha256_file(phase8_queue)},
            "opened_manifests": [
                {"path": str(path), "sha256": sha256_file(path)} for path in opened
            ],
        },
        "selection": {
            "stable_selection_seed": STABLE_SEED,
            "target_samples": args.target_samples,
            "minimum_samples": args.minimum_samples,
            "candidate_rows_before_media_checks": len(candidates),
            "selected_rows": len(rows),
            "unique_labels": len({str(row["label"]) for row in rows}),
            "unique_images": len({str(row["crop_sha256"]) for row in rows}),
            "historical_or_opened_label_overlap": 0,
            "phase8_queue_or_phase9_source_overlap": 0,
            "historical_or_opened_exact_image_overlap": 0,
            "exclusion_counts": dict(sorted(Counter(exclusions).items())),
        },
        "power_contract": lock["power_contract"],
        "formal_sample_ready": len(rows) >= args.minimum_samples,
        "power_target_ready": len(rows) == args.target_samples,
        "manifest": {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)},
    }
    (output_dir / "finalization_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--historical-manifest", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--phase7-opened-manifest", default=str(DEFAULT_PHASE7_OPENED))
    parser.add_argument("--phase8-opened-manifest", default=str(DEFAULT_PHASE8_OPENED))
    parser.add_argument("--phase9-opened-manifest", default=str(PHASE9_MANIFEST))
    parser.add_argument("--phase8-queue", default=str(DEFAULT_PHASE8_QUEUE))
    parser.add_argument("--candidate-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--first-sheet-row", type=int, default=733)
    parser.add_argument("--minimum-samples", type=int, default=500)
    parser.add_argument("--target-samples", type=int, default=1500)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


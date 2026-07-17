"""Create a label-review queue for a genuinely fresh Phase 8 confirmatory holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_8_consensus_ppo.review_contract import (  # noqa: E402
    normalize_label,
    validate_final_label,
)


DEFAULT_HISTORICAL = ROOT / "outputs" / "phase3_controlled_aug_full_frozen_eval" / "dataset_manifest.csv"
DEFAULT_OPENED_EXTERNAL = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "external_holdout_cropped"
    / "external_manifest.csv"
)
REVIEW_FIELDS = {"review_decision", "corrected_label"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_labels(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "label" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest lacks label column: {path}")
        return {normalize_label(row["label"]) for row in reader if normalize_label(row["label"])}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader), list(reader.fieldnames or [])


def migrate_review_state(
    selected: list[dict[str, str | int]],
    previous_original_path: Path,
    previous_reviewed_path: Path,
) -> tuple[list[dict[str, str | int]], dict]:
    previous_original, original_fields = read_csv(previous_original_path)
    previous_reviewed, reviewed_fields = read_csv(previous_reviewed_path)
    if original_fields != reviewed_fields or len(previous_original) != len(previous_reviewed):
        raise ValueError("Previous immutable/reviewed queues differ in schema or row count")
    if len(selected) < len(previous_original):
        raise ValueError("Expanded queue cannot be shorter than the previous queue")
    selected_fields = list(selected[0])
    if selected_fields != original_fields:
        raise ValueError("Expanded queue schema differs from the previous queue")
    for index, (new, old, reviewed) in enumerate(
        zip(selected, previous_original, previous_reviewed), start=2
    ):
        for field in selected_fields:
            old_value = str(old[field])
            if field not in REVIEW_FIELDS and (str(new[field]) != old_value or reviewed[field] != old_value):
                raise ValueError(f"Queue expansion changed prior immutable field {field!r} at row {index}")
    migrated = [dict(row) for row in selected]
    for index, reviewed in enumerate(previous_reviewed):
        migrated[index]["review_decision"] = reviewed["review_decision"]
        migrated[index]["corrected_label"] = reviewed["corrected_label"]
    carried = sum(bool(row["review_decision"].strip()) for row in previous_reviewed)
    return migrated, {
        "previous_original_queue": {
            "path": str(previous_original_path),
            "sha256": sha256(previous_original_path),
            "rows": len(previous_original),
        },
        "previous_reviewed_queue": {
            "path": str(previous_reviewed_path),
            "sha256_at_migration": sha256(previous_reviewed_path),
            "reviewed_decisions_carried": carried,
        },
        "prefix_preserved": True,
    }


def run(args: argparse.Namespace) -> dict:
    labels_csv = Path(args.labels_csv).resolve()
    source_root = Path(args.source_root).resolve()
    historical_path = Path(args.historical_manifest).resolve()
    opened_external_path = Path(args.opened_external_manifest).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()
    reviewed_output_path = Path(args.reviewed_output).resolve() if args.reviewed_output else None
    outputs = [output_path, audit_path]
    if reviewed_output_path is not None:
        outputs.append(reviewed_output_path)
    for output in outputs:
        if HERE not in output.parents:
            raise ValueError(f"Review queue artifacts must remain inside {HERE}")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite locked review queue artifact: {output}")

    excluded_labels = read_labels(historical_path) | read_labels(opened_external_path)
    candidates: list[dict[str, str | int]] = []
    exclusion_counts: Counter[str] = Counter()
    seen_candidate_labels: set[str] = set()
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "id",
            "image",
            "visual_image",
            "extracted_character",
            "review_status",
            "label_status",
            "source",
            "cropped",
            "bounding_box",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"labels.csv lacks required columns: {sorted(missing)}")
        for sheet_row, row in enumerate(reader, start=2):
            if sheet_row < args.first_sheet_row:
                continue
            review_status = (row.get("review_status") or "").strip().lower()
            label = (row.get("extracted_character") or "").strip()
            if not label:
                exclusion_counts["empty extracted_character"] += 1
                continue
            try:
                normalized = validate_final_label(label)
            except ValueError:
                exclusion_counts["invalid model label"] += 1
                continue
            if review_status == "rejected":
                exclusion_counts["review_status=rejected"] += 1
                continue
            if normalized in excluded_labels:
                exclusion_counts["label already opened or historical"] += 1
                continue
            if normalized in seen_candidate_labels:
                exclusion_counts["duplicate candidate label"] += 1
                continue
            image_path = (source_root / (row.get("image") or "")).resolve()
            if not image_path.is_file():
                exclusion_counts["missing source image"] += 1
                continue
            visual_value = (row.get("visual_image") or "").strip()
            visual_path = (source_root / visual_value).resolve() if visual_value else image_path
            seen_candidate_labels.add(normalized)
            candidates.append(
                {
                    "source_sheet_row": sheet_row,
                    "source_id": (row.get("id") or "").strip(),
                    "source_image_path": str(image_path),
                    "visual_image_path": str(visual_path),
                    "current_extracted_character": label,
                    "normalized_label": normalized,
                    "current_review_status": review_status,
                    "current_label_status": (row.get("label_status") or "").strip().lower(),
                    "source": (row.get("source") or "").strip(),
                    "source_cropped": (row.get("cropped") or "").strip().lower(),
                    "source_bounding_box": (row.get("bounding_box") or "").strip(),
                    "required_input_transform": (
                        "existing_plate_crop"
                        if (row.get("cropped") or "").strip().lower() == "true"
                        else "crop_source_bounding_box"
                    ),
                    "review_decision": "",
                    "corrected_label": "",
                }
            )

    # A stable hash avoids source-file ordering becoming an implicit selection heuristic.
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"phase8-fresh-review-v1:{row['source_id']}".encode("utf-8")
        ).digest()
    )
    selected = candidates[: args.queue_size]
    if len(selected) < args.minimum_required:
        raise ValueError(
            f"Only {len(selected)} fresh unique-label candidates remain; need at least {args.minimum_required}"
        )

    migration = None
    migrated_review = None
    migration_args = [args.previous_original_queue, args.previous_reviewed_queue, args.reviewed_output]
    if any(migration_args) and not all(migration_args):
        raise ValueError(
            "Queue migration requires --previous-original-queue, --previous-reviewed-queue, and --reviewed-output"
        )
    if all(migration_args):
        migrated_review, migration = migrate_review_state(
            selected,
            Path(args.previous_original_queue).resolve(),
            Path(args.previous_reviewed_queue).resolve(),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0])
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    if reviewed_output_path is not None and migrated_review is not None:
        with reviewed_output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(migrated_review)
    audit = {
        "contract": "phase8_fresh_holdout_review_queue_v2" if migration else "phase8_fresh_holdout_review_queue_v1",
        "not_an_evaluation_manifest": True,
        "inference_run": False,
        "labels_csv": {"path": str(labels_csv), "sha256": sha256(labels_csv)},
        "historical_manifest": {"path": str(historical_path), "sha256": sha256(historical_path)},
        "opened_external_manifest": {
            "path": str(opened_external_path),
            "sha256": sha256(opened_external_path),
        },
        "selection": {
            "first_sheet_row": args.first_sheet_row,
            "stable_selection_seed": "phase8-fresh-review-v1",
            "queue_size": len(selected),
            "minimum_required_after_review": args.minimum_required,
            "unique_normalized_labels": len({row["normalized_label"] for row in selected}),
            "excluded_label_groups": len(excluded_labels),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "review_queue": {"path": str(output_path), "sha256": sha256(output_path)},
        "review_instruction": (
            "Set review_decision to accepted, corrected, or rejected. For corrected rows fill corrected_label. "
            "Do not run OCR/PPO until a contiguous hash-ordered review prefix contains at least 500 "
            "leakage-free unique final-label groups."
        ),
    }
    if migration is not None and reviewed_output_path is not None:
        audit["migration"] = migration
        audit["reviewed_working_copy"] = {
            "path": str(reviewed_output_path),
            "sha256_at_migration": sha256(reviewed_output_path),
        }
    with audit_path.open("w", encoding="utf-8") as destination:
        json.dump(audit, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--historical-manifest", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--opened-external-manifest", default=str(DEFAULT_OPENED_EXTERNAL))
    parser.add_argument("--first-sheet-row", type=int, default=734)
    parser.add_argument("--queue-size", type=int, default=650)
    parser.add_argument("--minimum-required", type=int, default=500)
    parser.add_argument("--output", default=str(HERE / "fresh_holdout_review_queue.csv"))
    parser.add_argument("--audit", default=str(HERE / "fresh_holdout_review_queue_audit.json"))
    parser.add_argument("--previous-original-queue", default="")
    parser.add_argument("--previous-reviewed-queue", default="")
    parser.add_argument("--reviewed-output", default="")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

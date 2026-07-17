"""Build a no-review, leakage-free Phase 9 external holdout from labels.csv.

The user's ``extracted_character`` field is treated as ground truth. Rows at or
after the configured sheet row are candidates unless ``review_status`` is
``rejected``. Manual acceptance is deliberately not required. Formal protocol
filters (opened labels, source IDs, duplicate labels/images, invalid media) are
still enforced automatically before a stable, prospectively sized sample is
locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
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
from reinforcement_learning.phase_8_consensus_ppo.review_contract import (  # noqa: E402
    MINIMUM_FORMAL_SAMPLES,
    normalize_label,
    validate_final_label,
)


DEFAULT_SOURCE_ROOT = Path(r"D:\NEO\image_processing\dataset_general")
DEFAULT_LABELS_CSV = DEFAULT_SOURCE_ROOT / "labels.csv"
DEFAULT_HISTORICAL = ROOT / "outputs" / "phase3_controlled_aug_full_frozen_eval" / "dataset_manifest.csv"
DEFAULT_PHASE7_OPENED = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "external_holdout_cropped"
    / "external_manifest.csv"
)
DEFAULT_PHASE8_OPENED = (
    ROOT
    / "reinforcement_learning"
    / "phase_8_consensus_ppo"
    / "fresh_external_holdout"
    / "external_manifest.csv"
)
DEFAULT_PHASE8_QUEUE = (
    ROOT
    / "reinforcement_learning"
    / "phase_8_consensus_ppo"
    / "fresh_holdout_review_queue_v2.csv"
)
DEFAULT_CANDIDATE_LOCK = HERE / "prospective_policy.json"
DEFAULT_OUTPUT_DIR = HERE / "fresh_external_holdout"
STABLE_SELECTION_SEED = "phase9-primary-seed727-fresh-external-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader), list(reader.fieldnames or [])


def read_manifest_labels(path: Path) -> set[str]:
    rows, fields = read_csv(path)
    if "label" not in fields:
        raise ValueError(f"Manifest lacks label column: {path}")
    return {normalize_label(row["label"]) for row in rows if normalize_label(row["label"])}


def read_source_ids(path: Path) -> set[str]:
    rows, fields = read_csv(path)
    if "source_id" not in fields:
        raise ValueError(f"Queue lacks source_id column: {path}")
    return {(row.get("source_id") or "").strip() for row in rows if (row.get("source_id") or "").strip()}


def safe_source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Source image escapes dataset root: {relative}")
    return path


def validate_candidate_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("status") != "prospective_locked_requires_new_external"
        or lock.get("algorithm") != "single_primary_candidate_oof_ppo"
    ):
        raise ValueError("Phase 9 candidate lock is invalid")
    checkpoint = (ROOT / lock.get("policy_checkpoint", {}).get("path", "")).resolve()
    action_registry = (ROOT / lock.get("action_registry", {}).get("path", "")).resolve()
    for artifact, entry, name in (
        (checkpoint, lock.get("policy_checkpoint", {}), "policy checkpoint"),
        (action_registry, lock.get("action_registry", {}), "action registry"),
    ):
        if not artifact.is_file() or sha256_file(artifact) != entry.get("sha256"):
            raise ValueError(f"Locked Phase 9 {name} is missing or changed")
    return lock


def candidate_rows(
    labels_csv: Path,
    source_root: Path,
    first_sheet_row: int,
    excluded_labels: set[str],
    excluded_source_ids: set[str],
    stable_selection_seed: str = STABLE_SELECTION_SEED,
) -> tuple[list[dict[str, str | int]], Counter[str]]:
    candidates: list[dict[str, str | int]] = []
    counts: Counter[str] = Counter()
    seen_labels: set[str] = set()
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "id",
            "image",
            "extracted_character",
            "review_status",
            "source",
            "cropped",
            "bounding_box",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"labels.csv lacks required columns: {sorted(missing)}")
        for sheet_row, row in enumerate(reader, start=2):
            if sheet_row < first_sheet_row:
                continue
            source_id = (row.get("id") or "").strip()
            if not source_id:
                counts["empty_source_id"] += 1
                continue
            if source_id in excluded_source_ids:
                counts["source_already_in_phase8_queue"] += 1
                continue
            if (row.get("review_status") or "").strip().lower() == "rejected":
                counts["review_status=rejected"] += 1
                continue
            raw_label = (row.get("extracted_character") or "").strip()
            if not raw_label:
                counts["empty_extracted_character"] += 1
                continue
            try:
                label = validate_final_label(raw_label)
            except ValueError:
                counts["invalid_model_label"] += 1
                continue
            if label in excluded_labels:
                counts["label_historical_or_opened"] += 1
                continue
            if label in seen_labels:
                counts["duplicate_candidate_label"] += 1
                continue
            image_value = (row.get("image") or "").strip()
            if not image_value:
                counts["empty_image_path"] += 1
                continue
            image_path = safe_source_path(source_root, image_value)
            if not image_path.is_file():
                counts["missing_source_image"] += 1
                continue
            cropped = (row.get("cropped") or "").strip().lower() == "true"
            seen_labels.add(label)
            candidates.append(
                {
                    "source_sheet_row": sheet_row,
                    "source_id": source_id,
                    "source_image_path": str(image_path),
                    "normalized_label": label,
                    "raw_extracted_character": raw_label,
                    "review_status": (row.get("review_status") or "").strip().lower(),
                    "source": (row.get("source") or "").strip(),
                    "required_input_transform": (
                        "existing_plate_crop" if cropped else "crop_source_bounding_box"
                    ),
                    "source_bounding_box": (row.get("bounding_box") or "").strip(),
                }
            )
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{stable_selection_seed}:{row['source_id']}".encode("utf-8")
        ).digest()
    )
    return candidates, counts


def safe_filename(index: int, source_id: str, suffix: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_-]+", "_", source_id).strip("_") or "sample"
    return f"phase9_{index:06d}_{clean}{suffix}"


def run(args: argparse.Namespace) -> dict:
    if args.minimum_samples < MINIMUM_FORMAL_SAMPLES:
        raise ValueError(f"Phase 9 cannot lower minimum below {MINIMUM_FORMAL_SAMPLES}")
    if args.target_samples < args.minimum_samples:
        raise ValueError("target_samples must be at least minimum_samples")
    source_root = Path(args.source_root).resolve()
    labels_csv = Path(args.labels_csv).resolve()
    historical = Path(args.historical_manifest).resolve()
    phase7_opened = Path(args.phase7_opened_manifest).resolve()
    phase8_opened = Path(args.phase8_opened_manifest).resolve()
    phase8_queue = Path(args.phase8_queue).resolve()
    candidate_lock_path = Path(args.candidate_lock).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir != HERE and HERE not in output_dir.parents:
        raise ValueError(f"All Phase 9 holdout artifacts must remain inside {HERE}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite locked Phase 9 holdout: {output_dir}")
    if not labels_csv.is_file() or not source_root.is_dir():
        raise FileNotFoundError("Phase 9 source root or labels.csv is unavailable")

    lock = validate_candidate_lock(candidate_lock_path)
    locked_contract = lock["external_contract"]
    if (
        int(locked_contract.get("first_sheet_row_inclusive", -1)) != args.first_sheet_row
        or int(locked_contract.get("minimum_samples", -1)) != args.minimum_samples
        or int(locked_contract.get("target_samples", -1)) != args.target_samples
        or locked_contract.get("manual_acceptance_required") is not False
    ):
        raise ValueError("Holdout arguments differ from the prospective Phase 9 data contract")

    opened_manifests = [historical, phase7_opened, phase8_opened]
    excluded_labels = set().union(*(read_manifest_labels(path) for path in opened_manifests))
    excluded_source_ids = read_source_ids(phase8_queue)
    prior_image_hashes = set().union(*(manifest_image_hashes(path) for path in opened_manifests))
    candidates, exclusions = candidate_rows(
        labels_csv,
        source_root,
        args.first_sheet_row,
        excluded_labels,
        excluded_source_ids,
    )

    selected: list[dict[str, str | int]] = []
    rendered: list[tuple[dict[str, str | int], bytes, str, str, int, int]] = []
    selected_hashes: set[str] = set()
    for row in candidates:
        try:
            payload, suffix = render_plate_crop(row)  # type: ignore[arg-type]
            digest = hashlib.sha256(payload).hexdigest()
            if digest in prior_image_hashes:
                exclusions["exact_image_historical_or_opened"] += 1
                continue
            if digest in selected_hashes:
                exclusions["duplicate_rendered_image"] += 1
                continue
            from io import BytesIO

            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("empty rendered image")
        except Exception:
            exclusions["render_or_media_error"] += 1
            continue
        selected_hashes.add(digest)
        selected.append(row)
        rendered.append((row, payload, suffix, digest, width, height))
        if len(selected) == args.target_samples:
            break
    if len(selected) < args.minimum_samples:
        raise ValueError(
            f"Only {len(selected)} formal Phase 9 rows remain after automated checks; "
            f"need {args.minimum_samples}"
        )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "external_manifest.csv"
    fields = [
        "image_path",
        "label",
        "split",
        "input_contract",
        "source_id",
        "source_sheet_row",
        "source",
        "input_transform",
        "crop_width",
        "crop_height",
        "crop_sha256",
        "label_source",
        "source_review_status",
    ]
    manifest_rows = []
    for index, (row, payload, suffix, digest, width, height) in enumerate(rendered, start=1):
        destination = images_dir / safe_filename(index, str(row["source_id"]), suffix)
        destination.write_bytes(payload)
        manifest_rows.append(
            {
                "image_path": str(destination.resolve()),
                "label": row["normalized_label"],
                "split": "external_holdout",
                "input_contract": "plate_crop",
                "source_id": row["source_id"],
                "source_sheet_row": row["source_sheet_row"],
                "source": row["source"],
                "input_transform": row["required_input_transform"],
                "crop_width": width,
                "crop_height": height,
                "crop_sha256": digest,
                "label_source": "extracted_character",
                "source_review_status": row["review_status"],
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "contract": "phase9_fresh_locked_confirmatory_manifest_v1",
        "inference_run": False,
        "candidate_locked_before_manifest": True,
        "candidate_lock": {
            "path": str(candidate_lock_path),
            "sha256": sha256_file(candidate_lock_path),
        },
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
                {"path": str(path), "sha256": sha256_file(path)} for path in opened_manifests
            ],
        },
        "selection": {
            "stable_selection_seed": STABLE_SELECTION_SEED,
            "target_samples": args.target_samples,
            "minimum_samples": args.minimum_samples,
            "candidate_rows_before_media_checks": len(candidates),
            "selected_rows": len(manifest_rows),
            "unique_labels": len({str(row["label"]) for row in manifest_rows}),
            "unique_images": len({str(row["crop_sha256"]) for row in manifest_rows}),
            "historical_or_opened_label_overlap": 0,
            "phase8_queue_source_overlap": 0,
            "historical_or_opened_exact_image_overlap": 0,
            "exclusion_counts": dict(sorted(exclusions.items())),
        },
        "formal_sample_ready": len(manifest_rows) >= args.minimum_samples,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
    }
    audit_path = output_dir / "finalization_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--historical-manifest", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--phase7-opened-manifest", default=str(DEFAULT_PHASE7_OPENED))
    parser.add_argument("--phase8-opened-manifest", default=str(DEFAULT_PHASE8_OPENED))
    parser.add_argument("--phase8-queue", default=str(DEFAULT_PHASE8_QUEUE))
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--first-sheet-row", type=int, default=733)
    parser.add_argument("--minimum-samples", type=int, default=500)
    parser.add_argument("--target-samples", type=int, default=650)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

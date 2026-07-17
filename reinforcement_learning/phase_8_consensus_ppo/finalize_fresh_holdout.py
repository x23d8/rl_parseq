"""Finalize a reviewed queue into a locked, plate-crop confirmatory manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_8_consensus_ppo.review_contract import (  # noqa: E402
    MINIMUM_FORMAL_SAMPLES,
    assess_review_rows,
    normalize_label,
)


ORIGINAL_QUEUE = HERE / "fresh_holdout_review_queue_v2.csv"
REVIEWED_QUEUE = HERE / "fresh_holdout_review_queue_reviewed_v2.csv"
QUEUE_AUDIT = HERE / "fresh_holdout_review_queue_v2_audit.json"
MEDIA_AUDIT = HERE / "fresh_holdout_review_queue_v2_media_audit.json"
HISTORICAL_MANIFEST = ROOT / "outputs" / "phase3_controlled_aug_full_frozen_eval" / "dataset_manifest.csv"
OPENED_EXTERNAL_MANIFEST = (
    ROOT
    / "reinforcement_learning"
    / "phase_7_compact_multiscale_ppo"
    / "external_holdout_cropped"
    / "external_manifest.csv"
)
OUTPUT_DIR = HERE / "fresh_external_holdout"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        return list(reader), fields


def manifest_labels(path: Path) -> set[str]:
    rows, fields = read_csv(path)
    if "label" not in fields:
        raise ValueError(f"Manifest lacks label column: {path}")
    return {normalize_label(row["label"]) for row in rows if normalize_label(row["label"])}


def validate_review_rows(
    original: list[dict[str, str]],
    reviewed: list[dict[str, str]],
    excluded_labels: set[str],
    minimum_samples: int = MINIMUM_FORMAL_SAMPLES,
) -> list[dict[str, str]]:
    selected, assessment, _ = assess_review_rows(
        original, reviewed, excluded_labels, minimum_samples
    )
    if not assessment["formal_ready"]:
        raise ValueError(
            "Reviewed prefix is not formal-ready: "
            f"eligible_prefix={assessment['eligible_prefix']}/{minimum_samples}, "
            f"reviewed_prefix={assessment['reviewed_prefix']}, blank={assessment['blank']}, "
            f"invalid_final_label={assessment['invalid_final_label']}"
        )
    return selected[: assessment["eligible_prefix"]]


def render_plate_crop(row: dict[str, str]) -> tuple[bytes, str]:
    source_path = Path(row["source_image_path"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    suffix = source_path.suffix.lower() or ".jpg"
    transform = row["required_input_transform"].strip().lower()
    if transform == "existing_plate_crop":
        return source_path.read_bytes(), suffix
    if transform != "crop_source_bounding_box":
        raise ValueError(f"Unsupported input transform: {transform}")
    try:
        box = [int(round(float(value))) for value in json.loads(row["source_bounding_box"])]
        if len(box) != 4:
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid bounding box for {row['source_id']}") from error
    with Image.open(source_path) as opened:
        width, height = opened.size
        left, top = max(0, box[0]), max(0, box[1])
        right, bottom = min(width, box[2]), min(height, box[3])
        if right <= left or bottom <= top:
            raise ValueError(f"Empty clipped bounding box for {row['source_id']}")
        crop = opened.convert("RGB").crop((left, top, right, bottom))
        buffer = io.BytesIO()
        if suffix in {".jpg", ".jpeg"}:
            crop.save(buffer, format="JPEG", quality=95, subsampling=0)
        elif suffix == ".png":
            crop.save(buffer, format="PNG")
        else:
            suffix = ".png"
            crop.save(buffer, format="PNG")
    return buffer.getvalue(), suffix


def manifest_image_hashes(path: Path) -> set[str]:
    rows, fields = read_csv(path)
    if "image_path" not in fields:
        raise ValueError(f"Manifest lacks image_path column: {path}")
    result: set[str] = set()
    for row in rows:
        image = Path(row["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        result.add(sha256_file(image))
    return result


def validate_media_audit(
    audit_path: Path,
    queue_path: Path,
    historical_path: Path,
    opened_path: Path,
) -> dict:
    if HERE not in audit_path.parents or not audit_path.is_file():
        raise FileNotFoundError("Phase 8 queue media preflight audit is unavailable")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    media = audit.get("media", {})
    if (
        audit.get("contract") != "phase8_fresh_review_queue_media_preflight_v1"
        or audit.get("inference_run") is not False
        or not audit.get("passed", False)
        or not media.get("passed", False)
        or int(media.get("render_error_count", -1)) != 0
        or int(media.get("prior_exact_overlap_count", -1)) != 0
        or int(media.get("queue_exact_duplicate_count", -1)) != 0
    ):
        raise ValueError("Phase 8 queue did not pass the no-inference media preflight")
    expected = (
        ("queue", queue_path),
        ("historical_manifest", historical_path),
        ("opened_external_manifest", opened_path),
    )
    for field, path in expected:
        entry = audit.get(field, {})
        if Path(entry.get("path", "")).resolve() != path or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"Media preflight provenance changed for {field}")
    return audit


def run(args: argparse.Namespace) -> dict:
    if args.minimum_samples < MINIMUM_FORMAL_SAMPLES:
        raise ValueError(f"Formal Phase 8 finalization cannot lower minimum below {MINIMUM_FORMAL_SAMPLES}")
    original_path = Path(args.original_queue).resolve()
    reviewed_path = Path(args.reviewed_queue).resolve()
    queue_audit_path = Path(args.queue_audit).resolve()
    media_audit_path = Path(args.media_audit).resolve()
    historical_path = Path(args.historical_manifest).resolve()
    opened_path = Path(args.opened_external_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if HERE != output_dir and HERE not in output_dir.parents:
        raise ValueError(f"Final holdout must remain inside {HERE}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite locked output: {output_dir}")

    queue_audit = json.loads(queue_audit_path.read_text(encoding="utf-8"))
    if sha256_file(original_path) != queue_audit.get("review_queue", {}).get("sha256"):
        raise ValueError("Immutable review queue changed after candidate selection")
    media_audit = validate_media_audit(
        media_audit_path, original_path, historical_path, opened_path
    )
    original, original_fields = read_csv(original_path)
    reviewed, reviewed_fields = read_csv(reviewed_path)
    if reviewed_fields != original_fields:
        raise ValueError("Reviewed queue schema/order differs from the immutable queue")
    excluded_labels = manifest_labels(historical_path) | manifest_labels(opened_path)
    selected, review_assessment, _ = assess_review_rows(
        original, reviewed, excluded_labels, args.minimum_samples
    )
    if not review_assessment["formal_ready"]:
        raise ValueError(
            "Reviewed prefix is not formal-ready: "
            f"eligible_prefix={review_assessment['eligible_prefix']}/{args.minimum_samples}, "
            f"reviewed_prefix={review_assessment['reviewed_prefix']}, "
            f"remaining={review_assessment['remaining_to_minimum']}"
        )
    selected = selected[: review_assessment["eligible_prefix"]]

    rendered: list[dict[str, object]] = []
    selected_hashes: set[str] = set()
    destination_names: set[str] = set()
    for row in selected:
        value, suffix = render_plate_crop(row)
        digest = sha256_bytes(value)
        if digest in selected_hashes:
            raise ValueError("Reviewed holdout contains duplicate rendered image content")
        selected_hashes.add(digest)
        destination_name = f"phase8_{row['source_id']}{suffix}"
        if destination_name in destination_names:
            raise ValueError(f"Reviewed holdout contains duplicate source_id destination: {destination_name}")
        destination_names.add(destination_name)
        with Image.open(io.BytesIO(value)) as rendered_image:
            crop_width, crop_height = rendered_image.size
        rendered.append(
            {
                "row": row,
                "bytes": value,
                "sha256": digest,
                "suffix": suffix,
                "crop_width": crop_width,
                "crop_height": crop_height,
            }
        )
    prior_hashes = manifest_image_hashes(historical_path) | manifest_image_hashes(opened_path)
    overlap = selected_hashes & prior_hashes
    if overlap:
        raise ValueError(f"Fresh holdout contains {len(overlap)} exact image overlap(s) with opened data")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "external_manifest.csv"
    manifest_fields = [
        "image_path",
        "label",
        "split",
        "input_contract",
        "source_sheet_row",
        "source_id",
        "review_decision",
        "input_transform",
        "source",
        "source_cropped",
        "crop_width",
        "crop_height",
        "image_sha256",
    ]
    manifest_rows = []
    for item in rendered:
        row = item["row"]
        destination = images_dir / f"phase8_{row['source_id']}{item['suffix']}"
        destination.write_bytes(item["bytes"])
        manifest_rows.append(
            {
                "image_path": str(destination),
                "label": row["final_label"],
                "split": "external_holdout",
                "input_contract": "plate_crop",
                "source_sheet_row": row["source_sheet_row"],
                "source_id": row["source_id"],
                "review_decision": row["review_decision"],
                "input_transform": row["required_input_transform"],
                "source": row["source"],
                "source_cropped": row["source_cropped"],
                "crop_width": item["crop_width"],
                "crop_height": item["crop_height"],
                "image_sha256": item["sha256"],
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "contract": "phase8_fresh_locked_confirmatory_manifest_v1",
        "inference_run": False,
        "formal_sample_ready": len(manifest_rows) >= MINIMUM_FORMAL_SAMPLES,
        "immutable_queue": {"path": str(original_path), "sha256": sha256_file(original_path)},
        "media_preflight": {"path": str(media_audit_path), "sha256": sha256_file(media_audit_path)},
        "reviewed_queue": {"path": str(reviewed_path), "sha256": sha256_file(reviewed_path)},
        "historical_manifest": {"path": str(historical_path), "sha256": sha256_file(historical_path)},
        "opened_external_manifest": {"path": str(opened_path), "sha256": sha256_file(opened_path)},
        "selection": {
            "reviewed_rows": len(reviewed),
            "accepted": sum(row["review_decision"] == "accepted" for row in selected),
            "corrected": sum(row["review_decision"] == "corrected" for row in selected),
            "rejected": sum(row["review_decision"].strip().lower() == "rejected" for row in reviewed),
            "selected_rows": len(selected),
            "unique_normalized_labels": len({row["final_label"] for row in selected}),
            "historical_or_opened_label_overlap": 0,
            "historical_or_opened_exact_image_overlap": 0,
            "review_assessment": review_assessment,
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "images_directory": str(images_dir),
    }
    audit_path = output_dir / "finalization_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-queue", default=str(ORIGINAL_QUEUE))
    parser.add_argument("--reviewed-queue", default=str(REVIEWED_QUEUE))
    parser.add_argument("--queue-audit", default=str(QUEUE_AUDIT))
    parser.add_argument("--media-audit", default=str(MEDIA_AUDIT))
    parser.add_argument("--historical-manifest", default=str(HISTORICAL_MANIFEST))
    parser.add_argument("--opened-external-manifest", default=str(OPENED_EXTERNAL_MANIFEST))
    parser.add_argument("--minimum-samples", type=int, default=MINIMUM_FORMAL_SAMPLES)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

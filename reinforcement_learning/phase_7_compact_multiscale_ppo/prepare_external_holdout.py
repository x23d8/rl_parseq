"""Prepare a locked, auditable external holdout from the general dataset labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "external_holdout"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_status(value: str | None) -> str:
    return (value or "").strip().lower()


def prepare(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).resolve()
    labels_csv = Path(args.labels_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    manifest_path = output_dir / "external_manifest.csv"
    audit_path = output_dir / "preparation_audit.json"

    if HERE != output_dir and HERE not in output_dir.parents:
        raise ValueError(f"Output must remain inside {HERE}")
    if not labels_csv.is_file():
        raise FileNotFoundError(labels_csv)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    selected: list[dict[str, str | int]] = []
    excluded: list[dict[str, str | int]] = []
    review_status_counts: Counter[str] = Counter()
    label_status_counts: Counter[str] = Counter()

    with labels_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "id",
            "image",
            "extracted_character",
            "review_status",
            "label_status",
            "cropped",
            "bounding_box",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"labels.csv is missing required columns: {missing}")

        for sheet_row, row in enumerate(reader, start=2):
            if sheet_row > args.last_sheet_row:
                break

            review_status = normalized_status(row.get("review_status"))
            label_status = normalized_status(row.get("label_status"))
            review_status_counts[review_status] += 1
            label_status_counts[label_status] += 1

            record_id = (row.get("id") or "").strip()
            relative_image = (row.get("image") or "").strip()
            label = (row.get("extracted_character") or "").strip()
            audit_base = {
                "sheet_row": sheet_row,
                "id": record_id,
                "review_status": review_status,
                "label_status": label_status,
            }

            if review_status == "rejected":
                excluded.append({**audit_base, "reason": "review_status=rejected"})
                continue
            if not label:
                excluded.append({**audit_base, "reason": "empty extracted_character"})
                continue
            if not relative_image:
                raise ValueError(f"Sheet row {sheet_row} has an empty image path")

            source_image = (source_root / relative_image).resolve()
            if not source_image.is_file():
                raise FileNotFoundError(f"Sheet row {sheet_row}: {source_image}")

            destination_image = images_dir / source_image.name
            source_is_cropped = normalized_status(row.get("cropped")) == "true"
            input_transform = "existing_plate_crop"
            crop_box = ""
            if not source_is_cropped:
                if not args.crop_uncropped:
                    input_transform = "uncropped_source_image"
                else:
                    try:
                        coordinates = json.loads((row.get("bounding_box") or "").strip())
                        if len(coordinates) != 4:
                            raise ValueError
                        crop_box = json.dumps([int(round(float(value))) for value in coordinates])
                    except (TypeError, ValueError, json.JSONDecodeError) as error:
                        raise ValueError(f"Sheet row {sheet_row} has an invalid bounding_box") from error
                    input_transform = "crop_source_bounding_box"
            selected.append(
                {
                    **audit_base,
                    "source_image": str(source_image),
                    "destination_image": str(destination_image),
                    "label": label,
                    "source_is_cropped": source_is_cropped,
                    "input_transform": input_transform,
                    "crop_box": crop_box,
                }
            )

    destination_names = [Path(str(row["destination_image"])).name for row in selected]
    duplicate_names = sorted(name for name, count in Counter(destination_names).items() if count > 1)
    if duplicate_names:
        raise ValueError(f"Duplicate destination image names: {duplicate_names[:5]}")
    if not selected:
        raise ValueError("No valid external holdout samples were selected")

    images_dir.mkdir(parents=True, exist_ok=False)
    for row in selected:
        if row["input_transform"] != "crop_source_bounding_box":
            shutil.copy2(str(row["source_image"]), str(row["destination_image"]))
            continue
        left, top, right, bottom = json.loads(str(row["crop_box"]))
        with Image.open(str(row["source_image"])) as opened:
            width, height = opened.size
            left, top = max(0, left), max(0, top)
            right, bottom = min(width, right), min(height, bottom)
            if right <= left or bottom <= top:
                raise ValueError(f"Invalid clipped crop for sheet row {row['sheet_row']}")
            cropped = opened.convert("RGB").crop((left, top, right, bottom))
            if Path(str(row["destination_image"])).suffix.lower() in {".jpg", ".jpeg"}:
                cropped.save(str(row["destination_image"]), quality=95, subsampling=0)
            else:
                cropped.save(str(row["destination_image"]))

    manifest_fields = [
        "image_path",
        "label",
        "split",
        "source_sheet_row",
        "source_id",
        "review_status",
        "label_status",
        "input_contract",
        "input_transform",
        "source_cropped",
        "source_bounding_box",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=manifest_fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "image_path": row["destination_image"],
                    "label": row["label"],
                    "split": "external_holdout",
                    "source_sheet_row": row["sheet_row"],
                    "source_id": row["id"],
                    "review_status": row["review_status"],
                    "label_status": row["label_status"],
                    "input_contract": "plate_crop" if row["input_transform"] != "uncropped_source_image" else "mixed",
                    "input_transform": row["input_transform"],
                    "source_cropped": str(row["source_is_cropped"]).lower(),
                    "source_bounding_box": row["crop_box"],
                }
            )

    audit = {
        "contract": "phase7_external_holdout_preparation_v2",
        "source_root": str(source_root),
        "labels_csv": {
            "path": str(labels_csv),
            "sha256": sha256(labels_csv),
        },
        "selection": {
            "first_sheet_row": 2,
            "last_sheet_row_inclusive": args.last_sheet_row,
            "source_rows_considered": args.last_sheet_row - 1,
            "rule": "exclude review_status=rejected; label=extracted_character",
            "input_contract": "plate_crop" if args.crop_uncropped else "mixed",
            "selected_rows": len(selected),
            "excluded_rows": len(excluded),
            "review_status_counts": dict(sorted(review_status_counts.items())),
            "label_status_counts": dict(sorted(label_status_counts.items())),
        },
        "excluded": excluded,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "images_directory": str(images_dir),
        "copied_existing_crops": sum(row["input_transform"] == "existing_plate_crop" for row in selected),
        "cropped_from_source_bounding_box": sum(
            row["input_transform"] == "crop_source_bounding_box" for row in selected
        ),
        "uncropped_source_images": sum(row["input_transform"] == "uncropped_source_image" for row in selected),
    }
    with audit_path.open("w", encoding="utf-8") as destination:
        json.dump(audit, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--last-sheet-row", type=int, default=733)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--crop-uncropped",
        action="store_true",
        help="Crop rows marked cropped=false by their bounding_box so every manifest input is a plate crop.",
    )
    return parser


if __name__ == "__main__":
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

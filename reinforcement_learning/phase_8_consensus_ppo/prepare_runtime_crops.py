"""Convert detector bounding boxes into a label-free Phase 8 plate-crop runtime manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "results" / "runtime_detector_crops"


def parse_box(value: str, source_id: str) -> list[int]:
    try:
        box = [int(round(float(item))) for item in json.loads(value)]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid bounding_box for {source_id}") from error
    if len(box) != 4:
        raise ValueError(f"bounding_box must contain four coordinates for {source_id}")
    return box


def render_crop(source_path: Path, box: list[int], source_id: str) -> bytes:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        left, top = max(0, box[0]), max(0, box[1])
        right, bottom = min(image.width, box[2]), min(image.height, box[3])
        if right <= left or bottom <= top:
            raise ValueError(f"bounding_box clips to an empty crop for {source_id}")
        crop = image.crop((left, top, right, bottom))
        output = io.BytesIO()
        crop.save(output, format="JPEG", quality=95, subsampling=0)
        return output.getvalue()


def run(args: argparse.Namespace) -> dict:
    input_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if HERE not in output_dir.parents:
        raise ValueError("Detector-crop runtime artifacts must remain inside Phase 8")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite detector-crop output: {output_dir}")
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        required = {"image_path", "bounding_box"}
        if not required.issubset(fields):
            raise ValueError(f"Detector manifest is missing columns: {sorted(required - fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Detector manifest is empty")

    rendered = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        source_id = (row.get("source_id") or f"detector_{index:06d}").strip()
        source_key = source_id.casefold()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", source_id) or source_key in seen_ids:
            raise ValueError(f"Detector source_id must be unique and non-empty: {source_id!r}")
        seen_ids.add(source_key)
        source_path = Path((row.get("image_path") or "").strip()).resolve()
        box = parse_box(row.get("bounding_box") or "", source_id)
        value = render_crop(source_path, box, source_id)
        rendered.append((source_id, source_path, box, value))

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir()
    manifest_rows = []
    for source_id, source_path, box, value in rendered:
        destination = images_dir / f"{source_id}.jpg"
        destination.write_bytes(value)
        with Image.open(io.BytesIO(value)) as crop:
            crop_width, crop_height = crop.size
        manifest_rows.append(
            {
                "image_path": str(destination),
                "input_contract": "plate_crop",
                "input_transform": "crop_source_bounding_box",
                "crop_width": crop_width,
                "crop_height": crop_height,
                "source_image_path": str(source_path),
                "source_id": source_id,
                "bounding_box": json.dumps(box),
                "image_sha256": hashlib.sha256(value).hexdigest(),
            }
        )

    runtime_manifest = output_dir / "runtime_plate_crops.csv"
    manifest_fields = list(manifest_rows[0])
    with runtime_manifest.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    audit = {
        "contract": "phase8_detector_to_plate_crop_runtime_v1",
        "label_free": True,
        "inference_run": False,
        "source_manifest": {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "samples": len(manifest_rows),
        "runtime_manifest": {
            "path": str(runtime_manifest),
            "sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
        },
        "images_directory": str(images_dir),
    }
    (output_dir / "crop_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with image_path,bounding_box and optional source_id.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

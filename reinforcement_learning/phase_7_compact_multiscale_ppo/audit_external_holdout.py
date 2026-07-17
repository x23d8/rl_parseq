"""Audit an external manifest against all historical train/val/test samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_EXTERNAL = HERE / "external_holdout" / "external_manifest.csv"
DEFAULT_INTERNAL = HERE.parents[1] / "outputs" / "phase3_controlled_aug_full_frozen_eval" / "dataset_manifest.csv"
DEFAULT_DISJOINT = HERE / "external_holdout" / "external_manifest_group_disjoint.csv"
DEFAULT_AUDIT = HERE / "external_holdout" / "external_group_audit.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: str | None) -> str:
    return "".join(character for character in (value or "").upper() if character.isalnum())


def read_manifest(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows, fields


def run(args: argparse.Namespace) -> dict:
    internal_path = Path(args.internal_manifest).resolve()
    external_path = Path(args.external_manifest).resolve()
    disjoint_path = Path(args.disjoint_manifest).resolve()
    audit_path = Path(args.audit).resolve()
    for output in (disjoint_path, audit_path):
        if HERE not in output.parents:
            raise ValueError(f"Audit outputs must remain inside {HERE}")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite locked audit output: {output}")

    internal, _ = read_manifest(internal_path, {"image_path", "label", "split"})
    external, external_fields = read_manifest(
        external_path, {"image_path", "label", "split", "source_sheet_row", "source_id"}
    )
    if any(row["split"].strip().lower() != "external_holdout" for row in external):
        raise ValueError("Every external row must declare split=external_holdout")

    internal_labels: dict[str, set[str]] = defaultdict(set)
    internal_label_rows: Counter[str] = Counter()
    for row in internal:
        label = normalize_label(row["label"])
        if not label:
            raise ValueError("Internal manifest contains an empty normalized label")
        internal_labels[label].add(row["split"].strip().lower())
        internal_label_rows[label] += 1

    external_label_counts: Counter[str] = Counter()
    overlap_by_internal_split: Counter[str] = Counter()
    overlapping: list[dict[str, object]] = []
    disjoint: list[dict[str, str]] = []
    for row in external:
        label = normalize_label(row["label"])
        if not label:
            raise ValueError(f"External row {row.get('source_sheet_row')} has an empty normalized label")
        external_label_counts[label] += 1
        overlap_splits = sorted(internal_labels.get(label, set()))
        if overlap_splits:
            for split in overlap_splits:
                overlap_by_internal_split[split] += 1
            overlapping.append(
                {
                    "source_sheet_row": int(row["source_sheet_row"]),
                    "source_id": row["source_id"],
                    "label": row["label"],
                    "internal_splits": overlap_splits,
                    "internal_rows_with_label": internal_label_rows[label],
                }
            )
        else:
            disjoint.append(row)

    internal_hashes: dict[str, str] = {}
    for row in internal:
        image = Path(row["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        internal_hashes.setdefault(file_sha256(image), str(image.resolve()))

    external_hashes: dict[str, list[str]] = defaultdict(list)
    exact_internal_overlaps: list[dict[str, str | int]] = []
    for row in external:
        image = Path(row["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        digest = file_sha256(image)
        external_hashes[digest].append(str(image.resolve()))
        if digest in internal_hashes:
            exact_internal_overlaps.append(
                {
                    "source_sheet_row": int(row["source_sheet_row"]),
                    "source_id": row["source_id"],
                    "label": row["label"],
                    "internal_image_path": internal_hashes[digest],
                }
            )

    duplicate_external_groups = [paths for paths in external_hashes.values() if len(paths) > 1]
    if exact_internal_overlaps:
        raise ValueError(f"External manifest contains {len(exact_internal_overlaps)} exact internal image copies")
    if duplicate_external_groups:
        raise ValueError(f"External manifest contains {len(duplicate_external_groups)} duplicate image groups")
    if not disjoint:
        raise ValueError("No group-disjoint external rows remain")

    disjoint_path.parent.mkdir(parents=True, exist_ok=True)
    with disjoint_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=external_fields)
        writer.writeheader()
        writer.writerows(disjoint)

    audit = {
        "contract": "phase7_external_group_disjoint_audit_v1",
        "group_key": "normalized target/label",
        "internal_manifest": {
            "path": str(internal_path),
            "sha256": file_sha256(internal_path),
            "rows": len(internal),
            "splits": dict(sorted(Counter(row["split"].strip().lower() for row in internal).items())),
        },
        "supplied_external_manifest": {
            "path": str(external_path),
            "sha256": file_sha256(external_path),
            "rows": len(external),
            "distinct_normalized_labels": len(external_label_counts),
        },
        "leakage_audit": {
            "exact_internal_image_overlaps": len(exact_internal_overlaps),
            "duplicate_external_image_groups": len(duplicate_external_groups),
            "rows_with_historical_label_overlap": len(overlapping),
            "distinct_overlapping_labels": len({normalize_label(str(row["label"])) for row in overlapping}),
            "overlap_rows_by_internal_split": dict(sorted(overlap_by_internal_split.items())),
            "overlapping_rows": overlapping,
        },
        "locked_evaluation_manifest": {
            "path": str(disjoint_path),
            "sha256": file_sha256(disjoint_path),
            "rows": len(disjoint),
            "distinct_normalized_labels": len({normalize_label(row["label"]) for row in disjoint}),
            "selection_rule": "normalized label absent from historical train, val, and test manifests",
        },
    }
    with audit_path.open("w", encoding="utf-8") as destination:
        json.dump(audit, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-manifest", default=str(DEFAULT_INTERNAL))
    parser.add_argument("--external-manifest", default=str(DEFAULT_EXTERNAL))
    parser.add_argument("--disjoint-manifest", default=str(DEFAULT_DISJOINT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

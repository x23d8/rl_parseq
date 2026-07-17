"""Preflight every Phase 8 queue crop and exact-image overlap without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforcement_learning.phase_8_consensus_ppo.finalize_fresh_holdout import (  # noqa: E402
    HISTORICAL_MANIFEST,
    OPENED_EXTERNAL_MANIFEST,
    ORIGINAL_QUEUE,
    manifest_image_hashes,
    read_csv,
    render_plate_crop,
    sha256_bytes,
    sha256_file,
)


DEFAULT_OUTPUT = HERE / "fresh_holdout_review_queue_v2_media_audit.json"


def audit_media_rows(rows: list[dict[str, str]], prior_hashes: set[str]) -> dict:
    ordered_hashes: list[str] = []
    first_by_hash: dict[str, int] = {}
    overlaps: list[dict] = []
    duplicates: list[dict] = []
    errors: list[dict] = []
    for queue_index, row in enumerate(rows, start=1):
        try:
            value, _ = render_plate_crop(row)
            digest = sha256_bytes(value)
        except Exception as error:  # Keep a complete audit instead of stopping at the first bad row.
            errors.append(
                {
                    "queue_index": queue_index,
                    "source_id": row.get("source_id"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        ordered_hashes.append(digest)
        if digest in prior_hashes:
            overlaps.append(
                {"queue_index": queue_index, "source_id": row.get("source_id"), "sha256": digest}
            )
        if digest in first_by_hash:
            duplicates.append(
                {
                    "queue_index": queue_index,
                    "first_queue_index": first_by_hash[digest],
                    "source_id": row.get("source_id"),
                    "sha256": digest,
                }
            )
        else:
            first_by_hash[digest] = queue_index
    return {
        "samples": len(rows),
        "rendered": len(ordered_hashes),
        "ordered_rendered_sha256": hashlib.sha256("\n".join(ordered_hashes).encode("ascii")).hexdigest(),
        "render_errors": errors,
        "prior_exact_overlaps": overlaps,
        "queue_exact_duplicates": duplicates,
        "render_error_count": len(errors),
        "prior_exact_overlap_count": len(overlaps),
        "queue_exact_duplicate_count": len(duplicates),
        "passed": not errors and not overlaps and not duplicates and len(ordered_hashes) == len(rows),
    }


def run(args: argparse.Namespace) -> dict:
    queue_path = Path(args.queue).resolve()
    historical_path = Path(args.historical_manifest).resolve()
    opened_path = Path(args.opened_external_manifest).resolve()
    output_path = Path(args.output).resolve()
    if HERE not in output_path.parents:
        raise ValueError("Phase 8 media audit must remain inside Phase 8")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite media audit: {output_path}")
    rows, _ = read_csv(queue_path)
    prior_hashes = manifest_image_hashes(historical_path) | manifest_image_hashes(opened_path)
    media = audit_media_rows(rows, prior_hashes)
    audit = {
        "contract": "phase8_fresh_review_queue_media_preflight_v1",
        "inference_run": False,
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "historical_manifest": {"path": str(historical_path), "sha256": sha256_file(historical_path)},
        "opened_external_manifest": {"path": str(opened_path), "sha256": sha256_file(opened_path)},
        "media": media,
        "passed": media["passed"],
    }
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(ORIGINAL_QUEUE))
    parser.add_argument("--historical-manifest", default=str(HISTORICAL_MANIFEST))
    parser.add_argument("--opened-external-manifest", default=str(OPENED_EXTERNAL_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))

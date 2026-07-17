"""Shared review-integrity contract for the fresh Phase 8 holdout."""

from __future__ import annotations

from collections import Counter
import math


MINIMUM_FORMAL_SAMPLES = 500
MAX_LABEL_LENGTH = 12
PLATE_ALPHABET = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REVIEW_FIELDS = {"review_decision", "corrected_label"}
VALID_DECISIONS = {"accepted", "corrected", "rejected"}


def normalize_label(value: str | None) -> str:
    return "".join(character for character in (value or "").upper() if character in PLATE_ALPHABET)


def validate_final_label(value: str | None) -> str:
    raw = (value or "").upper()
    unsupported = sorted({character for character in raw if character.isalnum() and character not in PLATE_ALPHABET})
    if unsupported:
        raise ValueError(f"Label contains unsupported alphanumeric character(s): {unsupported}")
    normalized = normalize_label(raw)
    if not normalized:
        raise ValueError("Final label is empty after normalization")
    if len(normalized) > MAX_LABEL_LENGTH:
        raise ValueError(f"Final label exceeds model max_label_length={MAX_LABEL_LENGTH}")
    return normalized


def validate_immutable_rows(
    original: list[dict[str, str]], reviewed: list[dict[str, str]]
) -> None:
    if len(reviewed) != len(original):
        raise ValueError("Reviewed queue must preserve every row from the immutable queue")
    for index, (source, review) in enumerate(zip(original, reviewed), start=2):
        if list(source) != list(review):
            raise ValueError("Reviewed queue columns/order differ from the immutable queue")
        for field in source:
            if field not in REVIEW_FIELDS and source[field] != review[field]:
                raise ValueError(f"Reviewed queue changed immutable field {field!r} at row {index}")


def assess_review_rows(
    original: list[dict[str, str]],
    reviewed: list[dict[str, str]],
    excluded_labels: set[str] | None = None,
    minimum_samples: int = MINIMUM_FORMAL_SAMPLES,
) -> tuple[list[dict[str, str]], dict, list[dict]]:
    """Validate review state and select only leakage-free unique label groups.

    A reviewer decides whether the visible label is correct. Eligibility is a
    separate data-contract decision: labels already opened by historical runs and
    repeated final-label groups are excluded deterministically without asking the
    reviewer to relabel them as rejected.
    """

    validate_immutable_rows(original, reviewed)
    excluded_labels = {normalize_label(value) for value in (excluded_labels or set()) if normalize_label(value)}
    decision_counts: Counter[str] = Counter()
    selected: list[dict[str, str]] = []
    selected_labels: dict[str, int] = {}
    issues: list[dict] = []
    statuses: list[dict] = []
    prefix_open = True
    reviewed_prefix = 0
    eligible_prefix = 0

    for queue_index, review in enumerate(reviewed, start=1):
        csv_row = queue_index + 1
        decision = (review.get("review_decision") or "").strip().lower()
        if not decision:
            decision_counts["blank"] += 1
            prefix_open = False
            statuses.append({"status": "blank"})
            continue
        if decision not in VALID_DECISIONS:
            decision_counts["invalid"] += 1
            prefix_open = False
            issue = {
                "queue_index": queue_index,
                "csv_row": csv_row,
                "status": "invalid_decision",
                "decision": decision,
            }
            issues.append(issue)
            statuses.append(issue)
            continue

        decision_counts[decision] += 1
        if decision == "rejected":
            if prefix_open:
                reviewed_prefix = queue_index
            statuses.append({"status": "rejected"})
            continue

        raw_label = (
            review.get("current_extracted_character", "")
            if decision == "accepted"
            else review.get("corrected_label", "")
        )
        try:
            label = validate_final_label(raw_label)
        except ValueError as error:
            decision_counts["invalid_final_label"] += 1
            prefix_open = False
            issue = {
                "queue_index": queue_index,
                "csv_row": csv_row,
                "status": "invalid_final_label",
                "decision": decision,
                "error": str(error),
            }
            issues.append(issue)
            statuses.append(issue)
            continue
        if label in excluded_labels:
            decision_counts["excluded_label_overlap"] += 1
            if prefix_open:
                reviewed_prefix = queue_index
            issue = {
                "queue_index": queue_index,
                "csv_row": csv_row,
                "status": "excluded_label_overlap",
                "decision": decision,
                "label": label,
            }
            issues.append(issue)
            statuses.append(issue)
            continue
        if label in selected_labels:
            decision_counts["excluded_duplicate_label"] += 1
            if prefix_open:
                reviewed_prefix = queue_index
            issue = {
                "queue_index": queue_index,
                "csv_row": csv_row,
                "status": "excluded_duplicate_label",
                "decision": decision,
                "label": label,
                "first_queue_index": selected_labels[label],
            }
            issues.append(issue)
            statuses.append(issue)
            continue

        selected_labels[label] = queue_index
        selected.append({**review, "final_label": label, "review_decision": decision})
        if prefix_open:
            reviewed_prefix = queue_index
            eligible_prefix += 1
        statuses.append({"status": "eligible", "label": label})

    total = len(reviewed)
    blank = decision_counts["blank"]
    reviewed_count = total - blank
    eligible_count = len(selected)
    eligible_rate = eligible_prefix / reviewed_prefix if reviewed_prefix else 0.0
    projected_eligible = min(total, round(eligible_prefix + (total - reviewed_prefix) * eligible_rate))
    projected_prefix_for_minimum = (
        min(total, math.ceil(minimum_samples / eligible_rate)) if eligible_rate else total
    )
    review_complete = (
        blank == 0
        and decision_counts["invalid"] == 0
        and decision_counts["invalid_final_label"] == 0
    )
    summary = {
        "total": total,
        "reviewed": reviewed_count,
        "reviewed_prefix": reviewed_prefix,
        "accepted": decision_counts["accepted"],
        "corrected": decision_counts["corrected"],
        "rejected": decision_counts["rejected"],
        "blank": blank,
        "invalid": decision_counts["invalid"],
        "invalid_final_label": decision_counts["invalid_final_label"],
        "reviewed_usable": decision_counts["accepted"] + decision_counts["corrected"],
        "eligible_unique": eligible_count,
        "eligible_prefix": eligible_prefix,
        "excluded_label_overlap": decision_counts["excluded_label_overlap"],
        "excluded_duplicate_label": decision_counts["excluded_duplicate_label"],
        "minimum_formal_samples": int(minimum_samples),
        "remaining_to_minimum": max(0, int(minimum_samples) - eligible_prefix),
        "projected_final_eligible": projected_eligible,
        "projected_prefix_for_minimum": projected_prefix_for_minimum,
        "estimated_additional_reviews": max(0, projected_prefix_for_minimum - reviewed_prefix),
        "review_complete": review_complete,
        "formal_ready": eligible_prefix >= minimum_samples,
        "issue_count": len(issues),
        "issue_preview": issues[:10],
    }
    return selected, summary, statuses

from typing import Iterable, List

from .score_utils import ensure_open_unit_interval


def grade_track_classification(predicted_track: str, gold_track: str) -> float:
    if predicted_track.strip().lower() == gold_track.strip().lower():
        return ensure_open_unit_interval(1.0)
    return ensure_open_unit_interval(0.0)


def grade_information_collection(
    requested_fields: Iterable[str],
    required_fields: Iterable[str],
) -> float:
    requested = {field.strip().lower() for field in requested_fields if field.strip()}
    required = {field.strip().lower() for field in required_fields if field.strip()}
    if not requested or not required:
        return ensure_open_unit_interval(0.0)

    overlap = requested & required
    return ensure_open_unit_interval(len(overlap) / len(required))


def grade_batch_classification(predictions: List[str], gold_labels: List[str]) -> float:
    if len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must have the same length")
    if not predictions:
        return ensure_open_unit_interval(0.0)

    total = sum(
        grade_track_classification(predicted, gold)
        for predicted, gold in zip(predictions, gold_labels)
    )
    return ensure_open_unit_interval(total / len(predictions))


# Backward-compatible alias while the environment transitions from category to track naming.
def grade_classification(predicted_category: str, gold_category: str) -> float:
    return grade_track_classification(predicted_category, gold_category)

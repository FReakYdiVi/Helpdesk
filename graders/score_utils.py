import math
from typing import Any


MIN_SCORE = 0.001
MAX_SCORE = 0.999


def ensure_open_unit_interval(value: Any) -> float:
    """Return a native Python float strictly inside the open unit interval."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return MIN_SCORE

    if not math.isfinite(score):
        return MIN_SCORE

    score = max(0.0, min(1.0, score))
    if score <= 0.0:
        return MIN_SCORE
    if score >= 1.0:
        return MAX_SCORE
    return float(score)

import os


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _read_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


SIMILAR_EXAMPLES_LIMIT = _read_int("SIMILAR_EXAMPLES_LIMIT", 3, minimum=1)
SIMILAR_EXAMPLE_TEXT_MAX_CHARS = _read_int("SIMILAR_EXAMPLE_TEXT_MAX_CHARS", 80, minimum=16)
OLLAMA_EXTRACT_NUM_PREDICT = _read_int("OLLAMA_EXTRACT_NUM_PREDICT", 64, minimum=8)

NEIGHBOR_PRIOR_TOP_K = _read_int("NEIGHBOR_PRIOR_TOP_K", SIMILAR_EXAMPLES_LIMIT, minimum=1)
NEIGHBOR_PRIOR_MIN_CONSIDERED_UNCLEAR = _read_int(
    "NEIGHBOR_PRIOR_MIN_CONSIDERED_UNCLEAR",
    2,
    minimum=1,
)
NEIGHBOR_PRIOR_RATIO_UNCLEAR = _read_float("NEIGHBOR_PRIOR_RATIO_UNCLEAR", 0.60)
NEIGHBOR_PRIOR_MIN_CONSIDERED_OVERRIDE = _read_int(
    "NEIGHBOR_PRIOR_MIN_CONSIDERED_OVERRIDE",
    3,
    minimum=1,
)
NEIGHBOR_PRIOR_RATIO_OVERRIDE = _read_float("NEIGHBOR_PRIOR_RATIO_OVERRIDE", 0.80)

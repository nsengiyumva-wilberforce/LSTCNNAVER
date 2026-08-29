"""Dataset emotion taxonomies used by Ding et al. (IEEE TAFFC 2025)."""

from __future__ import annotations

# RAVDESS filename field 3 (1-indexed).
RAVDESS_ID_TO_EMOTION = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

# SAVEE prefix codes.
SAVEE_CODE_TO_EMOTION = {
    "a": "angry",
    "d": "disgust",
    "f": "fearful",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprised",
}

# MEAD folder names (front-view talking-face corpus).
MEAD_EMOTIONS = [
    "angry",
    "contempt",
    "disgusted",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

DATASET_EMOTIONS = {
    "ravdess": list(RAVDESS_ID_TO_EMOTION.values()),
    "savee": [
        "angry",
        "disgust",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ],
    "mead": list(MEAD_EMOTIONS),
    "synthetic": list(RAVDESS_ID_TO_EMOTION.values()),
}

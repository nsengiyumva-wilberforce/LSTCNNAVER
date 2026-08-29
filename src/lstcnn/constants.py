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

# Fig. 3 caption: dropout 0.3 / 0.4 / 0.5; dense 16 (RAVDESS/MEAD) or 14 (SAVEE).
PAPER_DROPOUT = {"mead": 0.3, "ravdess": 0.4, "savee": 0.5, "synthetic": 0.4}
PAPER_FUSION_HIDDEN = {"savee": 14, "ravdess": 16, "mead": 16, "synthetic": 16}


def paper_dropout(dataset: str) -> float:
    return PAPER_DROPOUT.get(dataset.lower(), 0.4)


def fusion_hidden_for(dataset: str, num_classes: int) -> int:
    if dataset.lower() == "savee" or num_classes == 7:
        return 14
    return PAPER_FUSION_HIDDEN.get(dataset.lower(), 16)

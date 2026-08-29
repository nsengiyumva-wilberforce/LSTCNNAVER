#!/usr/bin/env python3
"""Live webcam + mic SER using the QAT TFLite export.

Same preview as realtime.py (fused / face-only / voice-only). Face-only and
voice-only zero the unused branch — the TFLite graph has a single fused head.

    python realtime_tflite.py --tflite outputs/ravdess/tflite/lstcnn.tflite
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _default_tflite() -> Path | None:
    preferred = ROOT / "outputs" / "ravdess" / "tflite" / "lstcnn.tflite"
    if preferred.is_file():
        return preferred
    hits = sorted(ROOT.glob("outputs/**/*.tflite"))
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    argv = sys.argv[1:]
    has_model = any(a == "--tflite" or a.startswith("--tflite=") for a in argv)
    if not has_model:
        found = _default_tflite()
        if found is None:
            raise SystemExit(
                "Pass --tflite path/to/lstcnn.tflite "
                "(or export one to outputs/ravdess/tflite/lstcnn.tflite)."
            )
        argv = ["--tflite", str(found), *argv]
    sys.argv = [sys.argv[0], *argv]
    from realtime import main as realtime_main

    realtime_main()


if __name__ == "__main__":
    main()

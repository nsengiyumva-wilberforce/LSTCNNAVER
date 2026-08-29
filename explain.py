#!/usr/bin/env python3
"""Integrated Gradients explanations for one audio-visual clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.engine import load_checkpoint
from lstcnn.explain import explain_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", default=None, help="Optional separate audio file")
    parser.add_argument("--out-dir", default="outputs/explain")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    result = explain_file(model, cfg, args.video, args.audio, args.out_dir, device)
    print(f"Predicted class id: {result['pred']}")
    print(f"Wrote IG plots under {args.out_dir}")


if __name__ == "__main__":
    main()

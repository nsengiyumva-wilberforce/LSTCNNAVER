#!/usr/bin/env python3
"""Run the model on a single clip and print class probabilities.

A clip is six aligned (face, MFCC) windows; logits are averaged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.functional import softmax

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.constants import DATASET_EMOTIONS
from lstcnn.engine import load_checkpoint
from lstcnn.preprocess import extract_mfcc_vectors, sample_video_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    data = cfg["data"]
    faces = sample_video_frames(
        args.video,
        num_frames=data["num_frames"],
        image_size=data["image_size"],
        detect_face=data["detect_face"],
    )
    mfcc = extract_mfcc_vectors(
        args.audio or args.video,
        num_segments=data["num_audio_segments"],
        sr=data.get("sample_rate"),
        n_mfcc=data["n_mfcc"],
        n_fft=data["n_fft"],
        hop_length=data["hop_length"],
    )
    with torch.no_grad():
        logits = model(
            torch.from_numpy(faces).to(device),
            torch.from_numpy(mfcc).to(device),
        )
        probs = softmax(logits.mean(dim=0, keepdim=True), dim=1)[0].cpu()
    names = DATASET_EMOTIONS.get(data["dataset"].lower(), [str(i) for i in range(len(probs))])
    ranking = sorted(enumerate(probs.tolist()), key=lambda kv: kv[1], reverse=True)
    for idx, p in ranking:
        label = names[idx] if idx < len(names) else str(idx)
        print(f"{label:12s}  {p:.4f}")


if __name__ == "__main__":
    main()

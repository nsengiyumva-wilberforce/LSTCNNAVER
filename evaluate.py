#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.constants import DATASET_EMOTIONS
from lstcnn.engine import build_dataloaders, evaluate_ensemble, evaluate_loader, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    cfg = None
    for path in args.checkpoint:
        model, loaded = load_checkpoint(path, device)
        models.append(model)
        cfg = loaded
    assert cfg is not None
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    loaders = build_dataloaders(cfg)
    names = DATASET_EMOTIONS.get(cfg["data"]["dataset"].lower(), [])
    tta = bool(cfg.get("train", {}).get("tta", False))
    if len(models) == 1:
        metrics = evaluate_loader(models[0], loaders[args.split], device, names, tta=tta)
    else:
        print(f"Ensemble of {len(models)} checkpoints")
        metrics = evaluate_ensemble(models, loaders[args.split], device, names, tta=tta)
    print(metrics["report"])
    print(
        f"{args.split} window={metrics['accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}  "
        f"clip-majority={metrics.get('clip_accuracy')}"
    )
    out_dir = Path(args.checkpoint[0]).parent
    out = out_dir / f"eval_{args.split}.json"
    out.write_text(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in {"preds", "labels"}},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

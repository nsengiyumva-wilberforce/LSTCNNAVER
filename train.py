#!/usr/bin/env python3
"""Train the lightweight spatio-temporal AVER model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.config import load_config
from lstcnn.engine import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LST-CNN (Ding et al., IEEE TAFFC 2025)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", default=None, help="ravdess | savee | mead | synthetic")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-dir", default=None, help="Preprocessed face/MFCC cache (default data/cache)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.out_dir:
        cfg["train"]["out_dir"] = args.out_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_classes is not None:
        cfg["model"]["num_classes"] = args.num_classes
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.cache_dir is not None:
        cfg["data"]["cache_dir"] = args.cache_dir
    train_model(cfg)


if __name__ == "__main__":
    main()

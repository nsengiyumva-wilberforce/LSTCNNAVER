#!/usr/bin/env python3
"""Quantization-aware training and TFLite export (Ding et al. deployment path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.config import load_config
from lstcnn.engine import build_dataloaders
from lstcnn.tf_model import build_keras_model, faces_to_keras, mfcc_to_keras


def _batches(loader):
    faces_all, mfcc_all, labels_all = [], [], []
    for batch in loader:
        faces_all.append(faces_to_keras(batch["faces"].numpy()))
        mfcc_all.append(mfcc_to_keras(batch["mfcc"].numpy()))
        labels_all.append(batch["label"].numpy())
    return (
        np.concatenate(faces_all, axis=0),
        np.concatenate(mfcc_all, axis=0),
        np.concatenate(labels_all, axis=0),
    )


def _try_qat(model: keras.Model, train_data, val_data, epochs: int, batch_size: int, lr: float):
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("tensorflow-model-optimization not installed; skipping QAT.")
        return model
    try:
        print("Applying quantization-aware training...", flush=True)
        q_model = tfmot.quantization.keras.quantize_model(model)
    except Exception as exc:
        print(f"QAT annotate failed ({exc}); using float weights + TFLite PTQ.", flush=True)
        return model
    q_model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    q_model.fit(
        [train_data[0], train_data[1]],
        train_data[2],
        validation_data=([val_data[0], val_data[1]], val_data[2]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    return q_model


def main() -> None:
    parser = argparse.ArgumentParser(description="QAT + TFLite export")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--out-dir", default="outputs/tflite")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["dataset"] = args.dataset
    if args.dataset == "synthetic":
        cfg["train"]["num_workers"] = 0
    qat_cfg = cfg.get("qat", {})
    epochs = args.epochs if args.epochs is not None else qat_cfg.get("epochs", 5)
    batch_size = qat_cfg.get("batch_size", 16)
    lr = qat_cfg.get("lr", 1e-4)

    loaders = build_dataloaders(cfg)
    data = cfg["data"]
    model = build_keras_model(
        num_classes=cfg["model"]["num_classes"],
        image_size=data["image_size"],
        n_mfcc=data["n_mfcc"],
        dropout=cfg["model"]["dropout"],
        fusion_hidden=cfg["model"].get("fusion_hidden", 16),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    x_train_f, x_train_a, y_train = _batches(loaders["train"])
    x_val_f, x_val_a, y_val = _batches(loaders["val"])
    train_pack = (x_train_f, x_train_a, y_train)
    val_pack = (x_val_f, x_val_a, y_val)

    print("Float training before QAT...")
    model.fit(
        [x_train_f, x_train_a],
        y_train,
        validation_data=([x_val_f, x_val_a], y_val),
        epochs=max(epochs, 1),
        batch_size=batch_size,
        verbose=2,
    )

    export_model = _try_qat(model, train_pack, val_pack, epochs, batch_size, lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    path = out_dir / "lstcnn.tflite"
    path.write_bytes(tflite_model)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()

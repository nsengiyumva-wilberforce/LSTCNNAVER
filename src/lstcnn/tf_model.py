"""Keras twin of the PyTorch LST-CNN (Fig. 3) for QAT and TFLite export."""

from __future__ import annotations

from typing import Any

import numpy as np
import tf_keras as keras
from tf_keras import layers


def build_keras_model(
    num_classes: int = 8,
    image_size: int = 64,
    n_mfcc: int = 40,
    dropout: float = 0.4,
    fusion_hidden: int = 16,
    **_unused,
) -> keras.Model:
    faces = keras.Input(shape=(image_size, image_size, 1), name="faces")
    x = faces
    for channels in (16, 32, 64):
        x = layers.Conv2D(channels, 3, padding="valid", use_bias=True)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(dropout)(x)
    visual = layers.Flatten()(x)

    mfcc = keras.Input(shape=(n_mfcc,), name="mfcc")
    a = layers.Reshape((n_mfcc, 1))(mfcc)
    for i, channels in enumerate((16, 32)):
        a = layers.Conv1D(channels, 5, padding="valid", use_bias=True)(a)
        a = layers.ReLU()(a)
        a = layers.MaxPool1D(2)(a)
        if i == 1:
            a = layers.Dropout(dropout)(a)
    audio = layers.Flatten()(a)

    fused = layers.Concatenate()([visual, audio])
    fused = layers.Dense(fusion_hidden, activation="relu")(fused)
    fused = layers.Dropout(dropout)(fused)
    logits = layers.Dense(num_classes, name="logits")(fused)
    return keras.Model(inputs=[faces, mfcc], outputs=logits, name="lstcnn")


def faces_to_keras(faces: Any) -> np.ndarray:
    """(B, 1, H, W) or (B, H, W) → (B, H, W, 1)."""
    arr = np.asarray(faces, dtype=np.float32)
    if arr.ndim == 4:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 3:
        arr = arr[:, :, :, None]
    return arr


def mfcc_to_keras(mfcc: Any) -> np.ndarray:
    """(B, 40) mean-MFCC vectors."""
    return np.asarray(mfcc, dtype=np.float32)

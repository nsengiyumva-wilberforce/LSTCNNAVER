"""Keras twin of the PyTorch LST-CNN for QAT and TFLite export.

Uses tf_keras (Keras 2) so TFMOT QAT and TFLiteConverter work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers


def build_keras_model(
    num_classes: int = 8,
    image_size: int = 64,
    num_frames: int = 6,
    n_mfcc: int = 40,
    mfcc_frames_per_segment: int = 32,
    dropout: float = 0.3,
) -> keras.Model:
    """Spatial 16-32-64 (6-channel faces) + audio 16-32 (6 segments)."""
    faces = keras.Input(shape=(image_size, image_size, num_frames), name="faces")
    x = faces
    for channels in (16, 32, 64):
        x = layers.Conv2D(channels, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(2)(x)
    visual = layers.Flatten()(x)

    mfcc = keras.Input(shape=(num_frames, mfcc_frames_per_segment, n_mfcc), name="mfcc")
    a = mfcc
    time_len = mfcc_frames_per_segment
    for channels in (16, 32):
        a = layers.Conv2D(channels, (1, 3), padding="same", use_bias=False)(a)
        a = layers.BatchNormalization()(a)
        a = layers.ReLU()(a)
        a = layers.MaxPool2D((1, 2))(a)
        time_len //= 2
    a = layers.AveragePooling2D((1, time_len))(a)
    audio = layers.Flatten()(a)

    fused = layers.Concatenate()([visual, audio])
    fused = layers.Dropout(dropout)(fused)
    logits = layers.Dense(num_classes, name="logits")(fused)
    return keras.Model(inputs=[faces, mfcc], outputs=logits, name="lstcnn")


def faces_to_keras(faces: Any) -> np.ndarray:
    """(B, T, 1, H, W) → (B, H, W, T)."""
    arr = np.asarray(faces)
    if arr.ndim == 5:
        arr = np.squeeze(arr, axis=2)
        arr = np.transpose(arr, (0, 2, 3, 1))
    return arr.astype(np.float32)


def mfcc_to_keras(mfcc: Any) -> np.ndarray:
    """(B, S, n_mfcc, T) → (B, S, T, n_mfcc)."""
    arr = np.asarray(mfcc)
    if arr.ndim == 4:
        arr = np.transpose(arr, (0, 1, 3, 2))
    return arr.astype(np.float32)

"""TFLite inference for the QAT Keras LST-CNN export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from lstcnn.metrics import classification_metrics


def faces_to_nhwc(faces: Any) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 3:
        arr = arr[:, :, :, None]
    return arr


def mfcc_to_vec(mfcc: Any) -> np.ndarray:
    return np.asarray(mfcc, dtype=np.float32)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_tflite_meta(path: str | Path) -> dict:
    sidecar = Path(path).with_suffix(".json")
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {}


class TFLiteLSTCNN:
    """Runs `lstcnn.tflite` (fused logits; face/voice-only via a zeroed branch)."""

    def __init__(self, path: str | Path) -> None:
        import tensorflow as tf

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.path}")
        self.interpreter = tf.lite.Interpreter(model_path=str(self.path))
        self.interpreter.allocate_tensors()
        self._refresh_io()
        self.meta = load_tflite_meta(self.path)
        self.num_classes = int(self.meta.get("num_classes") or self._out["shape"][-1] or 8)

    def _refresh_io(self) -> None:
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        self._face_in, self._mfcc_in = _split_inputs(inputs)
        self._out = outputs[0]

    def _quantize(self, detail: dict, array: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(array)
        scale, zero = detail.get("quantization") or (0.0, 0)
        if detail["dtype"] == np.int8 and scale:
            return np.round(array / scale + zero).astype(np.int8)
        if detail["dtype"] == np.uint8 and scale:
            return np.round(array / scale + zero).astype(np.uint8)
        return array.astype(detail["dtype"], copy=False)

    def _dequantize(self, detail: dict, array: np.ndarray) -> np.ndarray:
        scale, zero = detail.get("quantization") or (0.0, 0)
        if scale and array.dtype in (np.int8, np.uint8):
            return (array.astype(np.float32) - zero) * scale
        return np.asarray(array, dtype=np.float32)

    def _resize_if_needed(self, faces: np.ndarray, mfcc: np.ndarray) -> None:
        face_shape = (faces.shape[0],) + tuple(self._face_in["shape"][1:])
        mfcc_shape = (mfcc.shape[0],) + tuple(self._mfcc_in["shape"][1:])
        if tuple(self._face_in["shape"]) == face_shape and tuple(self._mfcc_in["shape"]) == mfcc_shape:
            return
        self.interpreter.resize_tensor_input(self._face_in["index"], face_shape)
        self.interpreter.resize_tensor_input(self._mfcc_in["index"], mfcc_shape)
        self.interpreter.allocate_tensors()
        self._refresh_io()

    def _invoke_batch(self, faces: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
        self._resize_if_needed(faces, mfcc)
        self.interpreter.set_tensor(self._face_in["index"], self._quantize(self._face_in, faces))
        self.interpreter.set_tensor(self._mfcc_in["index"], self._quantize(self._mfcc_in, mfcc))
        self.interpreter.invoke()
        return self._dequantize(self._out, self.interpreter.get_tensor(self._out["index"]))

    def logits(self, faces: Any, mfcc: Any) -> np.ndarray:
        faces_n = faces_to_nhwc(_as_numpy(faces))
        mfcc_n = mfcc_to_vec(_as_numpy(mfcc))
        if faces_n.shape[0] != mfcc_n.shape[0]:
            raise ValueError(f"Batch mismatch: faces {faces_n.shape} vs mfcc {mfcc_n.shape}")
        fixed = int(self._face_in["shape"][0] or 0)
        if fixed in (0, -1) or fixed == faces_n.shape[0]:
            try:
                return self._invoke_batch(faces_n, mfcc_n)
            except (ValueError, RuntimeError):
                pass
        rows = [
            self._invoke_batch(faces_n[i : i + 1], mfcc_n[i : i + 1])
            for i in range(faces_n.shape[0])
        ]
        return np.concatenate(rows, axis=0)

    def predict_windows(
        self, faces: Any, mfcc: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Mean softmax over the six windows; face/voice-only zero the other input."""
        faces_n = faces_to_nhwc(_as_numpy(faces))
        mfcc_n = mfcc_to_vec(_as_numpy(mfcc))
        fused_logits = self.logits(faces_n, mfcc_n)
        vis_logits = self.logits(faces_n, np.zeros_like(mfcc_n))
        aud_logits = self.logits(np.zeros_like(faces_n), mfcc_n)
        fused = _softmax(fused_logits.mean(axis=0, keepdims=True), axis=1)[0]
        visual = _softmax(vis_logits.mean(axis=0, keepdims=True), axis=1)[0]
        audio = _softmax(aud_logits.mean(axis=0, keepdims=True), axis=1)[0]
        pred = int(fused.argmax())
        agreement = float((fused_logits.argmax(axis=1) == pred).mean())
        return fused, visual, audio, agreement


def _split_inputs(details: list[dict]) -> tuple[dict, dict]:
    face = next((d for d in details if "face" in d["name"].lower()), None)
    mfcc = next((d for d in details if "mfcc" in d["name"].lower()), None)
    if face is None or mfcc is None:
        by_rank = sorted(details, key=lambda d: len(d["shape"]), reverse=True)
        if len(by_rank) < 2:
            raise ValueError(f"Expected faces+mfcc inputs, got { [d['name'] for d in details] }")
        face = face or by_rank[0]
        mfcc = mfcc or by_rank[1]
    return face, mfcc


def evaluate_tflite_loader(model: TFLiteLSTCNN, loader, class_names: list[str]) -> dict:
    preds: list[int] = []
    labels: list[int] = []
    for batch in loader:
        logits = model.logits(batch["faces"], batch["mfcc"])
        preds.extend(logits.argmax(axis=1).tolist())
        labels.extend(int(x) for x in batch["label"].tolist())
    return classification_metrics(preds, labels, class_names)

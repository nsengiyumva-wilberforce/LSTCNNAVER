#!/usr/bin/env python3
"""Live webcam + mic SER with a trained PyTorch checkpoint.

Same six-window protocol as offline infer: Haar 64×64 faces and 40-d mean
MFCCs, then fused / face-only / voice-only heads (unused branch zeroed).

    python realtime.py --checkpoint outputs/ravdess/best.pt
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import softmax

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.constants import DATASET_EMOTIONS
from lstcnn.data import apply_mfcc_scale
from lstcnn.engine import load_checkpoint
from lstcnn.preprocess import frame_indices, mfcc_vectors_from_waveform, prepare_face


def _default_checkpoint() -> Path | None:
    preferred = [
        ROOT / "outputs" / "ravdess" / "best.pt",
        ROOT / "outputs" / "savee" / "best.pt",
        ROOT / "outputs" / "mead" / "best.pt",
        ROOT / "outputs" / "synthetic" / "best.pt",
    ]
    for path in preferred:
        if path.is_file():
            return path
    hits = sorted(ROOT.glob("outputs/**/best.pt"))
    return hits[0] if len(hits) == 1 else None


class MicBuffer:
    """Circular float32 mono buffer filled from a sounddevice callback."""

    def __init__(self, sr: int, seconds: float) -> None:
        self.sr = int(sr)
        self.capacity = max(int(self.sr * seconds), 1)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()

    def callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        chunk = np.asarray(indata[:, 0], dtype=np.float32)
        with self._lock:
            n = chunk.size
            if n >= self.capacity:
                self._data[:] = chunk[-self.capacity :]
                self._write = 0
                self._filled = self.capacity
                return
            end = self._write + n
            if end <= self.capacity:
                self._data[self._write : end] = chunk
            else:
                first = self.capacity - self._write
                self._data[self._write :] = chunk[:first]
                self._data[: n - first] = chunk[first:]
            self._write = (self._write + n) % self.capacity
            self._filled = min(self.capacity, self._filled + n)

    @property
    def samples(self) -> int:
        with self._lock:
            return self._filled

    def snapshot(self) -> np.ndarray:
        with self._lock:
            if self._filled < self.capacity:
                return self._data[: self._filled].copy()
            return np.concatenate([self._data[self._write :], self._data[: self._write]])


def _sample_faces(faces: list[np.ndarray], num_frames: int) -> np.ndarray:
    if not faces:
        raise RuntimeError("No webcam frames yet.")
    idx = frame_indices(len(faces), num_frames)
    stacked = np.stack([faces[int(i)] for i in idx], axis=0)
    return stacked[:, None, ...].astype(np.float32)


def _draw_row(frame: np.ndarray, y: int, title: str, name: str, prob: float, color) -> None:
    cv2.putText(frame, f"{title:<5} {name:<10} {prob:.2f}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    x1 = 280 + int(220 * max(0.0, min(1.0, prob)))
    cv2.rectangle(frame, (280, y - 14), (500, y + 4), (40, 40, 40), -1)
    cv2.rectangle(frame, (280, y - 14), (x1, y + 4), color, -1)


def _overlay(
    bgr: np.ndarray,
    names: list[str],
    fused: np.ndarray,
    visual: np.ndarray,
    audio: np.ndarray,
    agreement: float,
) -> np.ndarray:
    out = bgr.copy()
    fi, vi, ai = int(fused.argmax()), int(visual.argmax()), int(audio.argmax())
    _draw_row(out, 28, "FUSED", names[fi] if fi < len(names) else str(fi), float(fused[fi]), (0, 220, 0))
    _draw_row(out, 56, "FACE", names[vi] if vi < len(names) else str(vi), float(visual[vi]), (80, 180, 255))
    _draw_row(out, 84, "VOICE", names[ai] if ai < len(names) else str(ai), float(audio[ai]), (255, 180, 80))
    cv2.putText(
        out,
        f"windows agree {agreement:.0%}   q quit",
        (16, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
    )
    return out


def _predict(model, faces: np.ndarray, mfcc: np.ndarray, device: torch.device):
    face_t = torch.from_numpy(faces).to(device)
    mfcc_t = torch.from_numpy(mfcc).to(device)
    with torch.no_grad():
        fused_logits = model(face_t, mfcc_t)
        vis_logits = model.forward_visual(face_t)
        aud_logits = model.forward_audio(mfcc_t)
        fused = softmax(fused_logits.mean(dim=0, keepdim=True), dim=1)[0].cpu().numpy()
        visual = softmax(vis_logits.mean(dim=0, keepdim=True), dim=1)[0].cpu().numpy()
        audio = softmax(aud_logits.mean(dim=0, keepdim=True), dim=1)[0].cpu().numpy()
        pred = int(fused.argmax())
        agreement = float((fused_logits.argmax(dim=1) == pred).float().mean().item())
    return fused, visual, audio, agreement


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam + mic SER (PyTorch)")
    parser.add_argument("--checkpoint", default=None, help="Path to best.pt from train.py")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window-sec", type=float, default=3.0, help="Rolling clip length (six windows)")
    parser.add_argument("--hop-sec", type=float, default=0.25, help="Seconds between inferences")
    parser.add_argument("--mic", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--no-display", action="store_true", help="Print predictions only (no OpenCV window)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else _default_checkpoint()
    if ckpt is None or not ckpt.is_file():
        raise SystemExit(
            "Pass --checkpoint path/to/best.pt "
            "(or train first so outputs/ravdess/best.pt exists)."
        )

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install sounddevice for the microphone: pip install sounddevice") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(ckpt, device)
    data = cfg["data"]
    dataset = str(data.get("dataset", "")).lower()
    names = DATASET_EMOTIONS.get(dataset, [str(i) for i in range(cfg["model"]["num_classes"])])
    num_frames = int(data.get("num_frames", 6))
    n_mfcc = int(data["n_mfcc"])
    image_size = int(data["image_size"])
    detect_face = bool(data.get("detect_face", True))
    face_margin = float(data.get("face_margin", 0.0))
    align_face = bool(data.get("align_face", False))
    sr = int(data["sample_rate"] or 48000)
    stretch = None  # paper stretch is train-only; live audio is already speech

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}.")

    mic = MicBuffer(sr, args.window_sec)
    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        device=args.mic,
        callback=mic.callback,
        blocksize=1024,
    )
    stream.start()

    face_buf: deque[np.ndarray] = deque(maxlen=max(int(30 * args.window_sec), num_frames))
    last_infer = 0.0
    last_overlay = None
    print(f"Realtime SER  checkpoint={ckpt}  device={device}  mic={sr} Hz  q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Lost camera frame.", file=sys.stderr)
                break
            face = prepare_face(
                frame,
                image_size=image_size,
                detect_face=detect_face,
                face_margin=face_margin,
                align_face=align_face,
            )
            face_buf.append(face)

            now = time.monotonic()
            if now - last_infer >= args.hop_sec and len(face_buf) >= num_frames and mic.samples > sr // 5:
                faces = _sample_faces(list(face_buf), num_frames)
                mfcc = mfcc_vectors_from_waveform(
                    mic.snapshot(),
                    sr,
                    num_segments=int(data.get("num_audio_segments", num_frames)),
                    n_mfcc=n_mfcc,
                    n_fft=int(data["n_fft"]),
                    hop_length=int(data["hop_length"]),
                    time_stretch=stretch,
                )
                mfcc = apply_mfcc_scale(mfcc, data.get("mfcc_mean"), data.get("mfcc_std"))
                fused, visual, audio, agree = _predict(model, faces, mfcc, device)
                last_overlay = (fused, visual, audio, agree)
                last_infer = now
                fi = int(fused.argmax())
                label = names[fi] if fi < len(names) else str(fi)
                print(
                    f"fused={label:10s} {fused[fi]:.2f}  "
                    f"face={names[int(visual.argmax())]:10s} {visual.max():.2f}  "
                    f"voice={names[int(audio.argmax())]:10s} {audio.max():.2f}  "
                    f"agree={agree:.0%}",
                    flush=True,
                )

            shown = frame
            if last_overlay is not None:
                shown = _overlay(frame, names, *last_overlay)
            if args.no_display:
                time.sleep(0.01)
                continue
            try:
                cv2.imshow("LST-CNN realtime", shown)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as exc:
                raise SystemExit(
                    "OpenCV has no GUI (headless build). "
                    "Re-run with --no-display or: pip install opencv-python"
                ) from exc
            if key in (ord("q"), 27):
                break
    finally:
        stream.stop()
        stream.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

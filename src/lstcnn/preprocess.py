"""Face and MFCC preprocessing described in Ding et al. (IEEE TAFFC 2025).

Paper:
  * Faces converted to grayscale and resized to 64x64.
  * Audio represented as 40 Mel-Frequency Cepstral Coefficients.
  * 1D convolution is applied along the MFCC time axis.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import librosa
import numpy as np

_HAAR_NAME = "haarcascade_frontalface_default.xml"
_FACE_CASCADE: cv2.CascadeClassifier | None = None
_CASCADE_UNAVAILABLE = False


def _haar_candidates() -> list[str]:
    data_dir = getattr(cv2, "data", None)
    base = getattr(data_dir, "haarcascades", "") if data_dir is not None else ""
    return [
        str(Path(base) / _HAAR_NAME) if base else "",
        str(Path(cv2.__file__).resolve().parent / "data" / _HAAR_NAME),
        str(Path(__file__).resolve().parents[2] / "assets" / _HAAR_NAME),
    ]


def _cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE, _CASCADE_UNAVAILABLE
    if _CASCADE_UNAVAILABLE:
        return None
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    for path in _haar_candidates():
        if not path or not Path(path).is_file():
            continue
        try:
            classifier = cv2.CascadeClassifier(path)
        except AttributeError:
            _CASCADE_UNAVAILABLE = True
            return None
        if not classifier.empty():
            _FACE_CASCADE = classifier
            return classifier
    _CASCADE_UNAVAILABLE = True
    return None


def detect_and_crop_face(gray: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """Crop the largest frontal face; fall back to a centered square."""
    cascade = _cascade()
    if cascade is None:
        return _center_square(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
    if len(faces) == 0:
        return _center_square(gray)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_x, pad_y = int(w * margin), int(h * margin)
    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + w + pad_x, gray.shape[1])
    y1 = min(y + h + pad_y, gray.shape[0])
    return gray[y0:y1, x0:x1]


def _center_square(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return gray[y0 : y0 + side, x0 : x0 + side]


def prepare_face(bgr_or_gray: np.ndarray, image_size: int = 64, detect_face: bool = True) -> np.ndarray:
    if bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray
    crop = detect_and_crop_face(gray) if detect_face else _center_square(gray)
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0 - 0.5) / 0.5


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 6,
    image_size: int = 64,
    detect_face: bool = True,
) -> np.ndarray:
    """Uniformly sample `num_frames` grayscale faces from a video.

    Returns an array of shape (T, 1, H, W).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty video: {video_path}")
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        faces = prepare_face(frame, image_size=image_size, detect_face=detect_face)
        frames.append(faces[None, ...])
    cap.release()
    if not frames:
        raise RuntimeError(f"No readable frames in {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames], axis=0)


def _mfcc_from_wave(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    max_frames: int,
) -> np.ndarray:
    if waveform.size == 0:
        waveform = np.zeros(max(sr // 10, n_fft), dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-6
    mfcc = (mfcc - mean) / std
    time = mfcc.shape[1]
    if time >= max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        pad = np.zeros((n_mfcc, max_frames - time), dtype=mfcc.dtype)
        mfcc = np.concatenate([mfcc, pad], axis=1)
    return mfcc.astype(np.float32)


def extract_mfcc(
    audio_path: str | Path,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_frames: int = 128,
) -> np.ndarray:
    """Return CMVN-normalized MFCCs of shape (n_mfcc, max_frames)."""
    waveform, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return _mfcc_from_wave(waveform, sr, n_mfcc, n_fft, hop_length, max_frames)


def extract_mfcc_segments(
    audio_path: str | Path,
    num_segments: int = 6,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    frames_per_segment: int = 32,
) -> np.ndarray:
    """Split audio into `num_segments` equal chunks, aligned with video frames.

    Returns (num_segments, n_mfcc, frames_per_segment).
    """
    waveform, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    if waveform.size == 0:
        waveform = np.zeros(sr, dtype=np.float32)
    length = waveform.shape[0]
    edges = np.linspace(0, length, num_segments + 1, dtype=int)
    segments = []
    for i in range(num_segments):
        chunk = waveform[edges[i] : edges[i + 1]]
        segments.append(
            _mfcc_from_wave(chunk, sr, n_mfcc, n_fft, hop_length, frames_per_segment)
        )
    return np.stack(segments, axis=0)

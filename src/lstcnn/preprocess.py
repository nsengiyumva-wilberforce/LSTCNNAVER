"""Face and MFCC preprocessing described in Ding et al. (IEEE TAFFC 2025).

Paper:
  * Faces: Haar crop, grayscale, 64x64; six frames at one-sixth intervals.
  * Audio: 40 MFCCs; 1D conv along cepstral bands with a 5×1 kernel.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import librosa
import numpy as np

_VIDEO_AUDIO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpeg", ".mpg"}


def require_ffmpeg() -> str:
    """RAVDESS Video_Speech packs are mp4; libsndfile cannot decode them."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to read audio from .mp4 files. "
            "Install it with: sudo apt install ffmpeg"
        )
    return ffmpeg

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


def frame_indices(total: int, num_frames: int = 6) -> np.ndarray:
    """Frame indexes at one-sixth intervals: 0, T/6, 2T/6, …, 5T/6."""
    if total <= 0:
        raise ValueError(f"Need a positive frame count, got {total}")
    idx = (np.arange(num_frames, dtype=np.float64) * total / num_frames).astype(int)
    return np.clip(idx, 0, total - 1)


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 6,
    image_size: int = 64,
    detect_face: bool = True,
) -> np.ndarray:
    """Sample `num_frames` grayscale faces at one-sixth intervals.

    Returns an array of shape (T, 1, H, W).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty video: {video_path}")
    indices = frame_indices(total, num_frames)
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


def load_waveform(
    path: str | Path,
    sr: int | None = None,
    return_sr: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Load mono audio from wav/mp3 or from a video file's soundtrack.

    `sr=None` keeps the source rate (paper: librosa defaults, native sampling rate).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio/video not found: {path}")

    wav_sidecar = _companion_wav(path)
    if wav_sidecar is not None:
        try:
            y, native = librosa.load(str(wav_sidecar), sr=sr, mono=True)
            if y.size > 0:
                out = y.astype(np.float32)
                return (out, int(native)) if return_sr else out
        except Exception:
            pass

    if path.suffix.lower() not in _VIDEO_AUDIO_EXTS:
        try:
            y, native = librosa.load(str(path), sr=sr, mono=True)
            if y.size > 0:
                out = y.astype(np.float32)
                return (out, int(native)) if return_sr else out
        except Exception as exc:
            raise RuntimeError(f"Could not read audio {path}: {exc}") from exc

    y, native = _ffmpeg_load(path, sr)
    return (y, native) if return_sr else y


def _companion_wav(path: Path) -> Path | None:
    """Prefer a RAVDESS audio-only wav (modality 03) next to the video."""
    same = path.with_suffix(".wav")
    if same.is_file():
        return same
    parts = path.stem.split("-")
    if len(parts) >= 7:
        audio_stem = "-".join(["03", *parts[1:]])
        sibling = path.parent / f"{audio_stem}.wav"
        if sibling.is_file():
            return sibling
    return None


def _probe_sample_rate(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 48000
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
        return int(proc.stdout.strip().splitlines()[0])
    except Exception:
        return 48000


def _ffmpeg_load(path: Path, sr: int | None) -> tuple[np.ndarray, int]:
    used_sr = sr or _probe_sample_rate(path)
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(used_sr),
        "-v",
        "error",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed on {path}: {err or exc.returncode}") from exc
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError(f"ffmpeg produced empty audio for {path}")
    return y, used_sr


def _mean_mfcc(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """librosa MFCC (n_mfcc, time), then average over time → (n_mfcc,)."""
    if waveform.size == 0:
        waveform = np.zeros(max(sr // 10, n_fft), dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc.mean(axis=1).astype(np.float32)


def extract_mfcc_vectors(
    audio_path: str | Path,
    num_segments: int = 6,
    sr: int | None = None,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    time_stretch: float | None = None,
) -> np.ndarray:
    """Six non-overlapping windows → 40 mean MFCCs each (Fig. 2).

    Returns (num_segments, n_mfcc).
    """
    waveform, native_sr = load_waveform(audio_path, sr=sr, return_sr=True)
    if time_stretch is not None and abs(time_stretch - 1.0) > 1e-6 and waveform.size > 1:
        waveform = librosa.effects.time_stretch(waveform, rate=float(time_stretch))
    if waveform.size == 0:
        waveform = np.zeros(max(native_sr // 10, n_fft), dtype=np.float32)
    length = waveform.shape[0]
    edges = np.linspace(0, length, num_segments + 1, dtype=int)
    vectors = [
        _mean_mfcc(waveform[edges[i] : edges[i + 1]], native_sr, n_mfcc, n_fft, hop_length)
        for i in range(num_segments)
    ]
    return np.stack(vectors, axis=0)


def extract_mfcc_segments(
    audio_path: str | Path,
    num_segments: int = 6,
    sr: int | None = None,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    frames_per_segment: int = 32,
    time_stretch: float | None = None,
) -> np.ndarray:
    """Alias of extract_mfcc_vectors (paper averages each window to 40-d)."""
    del frames_per_segment
    return extract_mfcc_vectors(
        audio_path,
        num_segments=num_segments,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        time_stretch=time_stretch,
    )

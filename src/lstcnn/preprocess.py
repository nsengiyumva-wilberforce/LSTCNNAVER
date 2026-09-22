"""Face and MFCC preprocessing described in Ding et al. (IEEE TAFFC 2025).

Paper / supplementary:
  * Faces: Haar crop, grayscale 64×64. The 68-point dlib model (v19.24.2) is
    used in the supplementary *occlusion / IG* tests, not as a training warp.
  * Six windows: face at the center of each one-sixth; 40 mean MFCCs per sixth.
  * RAVDESS: drop the ~1 s head/tail hold so every sixth is speech; time-stretch
    0.8 is train-only augmentation on the full (trimmed) utterance, then split.
  * Literal paper §III.B (“whole duration”, Pydub interval starts) scores ~86%
    here; trimmed speech-centered windows score ~91%.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import cv2
import librosa
import numpy as np

_VIDEO_AUDIO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpeg", ".mpg"}
_HAAR_NAME = "haarcascade_frontalface_default.xml"
_DLIB_PREDICTOR_NAME = "shape_predictor_68_face_landmarks.dat"
_FACE_CASCADE: cv2.CascadeClassifier | None = None
_CASCADE_UNAVAILABLE = False
_DLIB_PREDICTOR = None
_DLIB_DETECTOR = None
_DLIB_FAILED = False

# 68-point indices (iBUG / dlib).
_LEFT_EYE = slice(36, 42)
_RIGHT_EYE = slice(42, 48)


def require_ffmpeg() -> str:
    """RAVDESS Video_Speech packs are mp4; libsndfile cannot decode them."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to read audio from .mp4 files. "
            "Install it with: sudo apt install ffmpeg"
        )
    return ffmpeg


def _assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets"


def _haar_candidates() -> list[str]:
    data_dir = getattr(cv2, "data", None)
    base = getattr(data_dir, "haarcascades", "") if data_dir is not None else ""
    return [
        str(Path(base) / _HAAR_NAME) if base else "",
        str(Path(cv2.__file__).resolve().parent / "data" / _HAAR_NAME),
        str(_assets_dir() / _HAAR_NAME),
    ]


def _cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE, _CASCADE_UNAVAILABLE
    if _CASCADE_UNAVAILABLE:
        return None
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    if not hasattr(cv2, "CascadeClassifier"):
        _CASCADE_UNAVAILABLE = True
        return None
    for path in _haar_candidates():
        if not path or not Path(path).is_file():
            continue
        classifier = cv2.CascadeClassifier(path)
        if not classifier.empty():
            _FACE_CASCADE = classifier
            return classifier
    _CASCADE_UNAVAILABLE = True
    return None


def _dlib_predictor():
    """Load the supplementary's 68-point model (dlib v19.24.2)."""
    global _DLIB_PREDICTOR, _DLIB_FAILED
    if _DLIB_FAILED:
        return None
    if _DLIB_PREDICTOR is not None:
        return _DLIB_PREDICTOR
    try:
        import dlib
    except ImportError:
        _DLIB_FAILED = True
        return None
    path = _assets_dir() / _DLIB_PREDICTOR_NAME
    if not path.is_file():
        _DLIB_FAILED = True
        return None
    _DLIB_PREDICTOR = dlib.shape_predictor(str(path))
    return _DLIB_PREDICTOR


def _dlib_detector():
    global _DLIB_DETECTOR
    try:
        import dlib
    except ImportError:
        return None
    if _DLIB_DETECTOR is None:
        _DLIB_DETECTOR = dlib.get_frontal_face_detector()
    return _DLIB_DETECTOR


def trim_top_db_from_cfg(data_cfg: dict | None) -> float | None:
    """RAVDESS head/tail trim; other corpora leave the file untouched."""
    if not data_cfg:
        return None
    if str(data_cfg.get("dataset", "")).lower() != "ravdess":
        return None
    if not bool(data_cfg.get("trim_silence", True)):
        return None
    return float(data_cfg.get("trim_top_db", 30))


def face_tool_status() -> dict[str, bool]:
    """Check tools without loading models (safe to call before a process pool)."""
    haar = hasattr(cv2, "CascadeClassifier") and any(
        Path(path).is_file() for path in _haar_candidates() if path
    )
    try:
        import dlib  # noqa: F401
    except ImportError:
        dlib68 = False
    else:
        dlib68 = (_assets_dir() / _DLIB_PREDICTOR_NAME).is_file()
    return {"haar": haar, "dlib68": dlib68}


def _largest_face_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Haar box (paper); dlib HOG if Haar misses."""
    boxes: list[tuple[int, int, int, int]] = []
    cascade = _cascade()
    if cascade is not None:
        detected = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32)
        )
        boxes.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in detected)
    if not boxes:
        detector = _dlib_detector()
        if detector is not None:
            for rect in detector(gray, 1):
                boxes.append((int(rect.left()), int(rect.top()), int(rect.width()), int(rect.height())))
    if not boxes:
        return None
    return max(boxes, key=lambda box: box[2] * box[3])


def landmarks_68(
    gray: np.ndarray,
    box: tuple[int, int, int, int] | None,
    expand: float = 0.2,
) -> np.ndarray | None:
    """68 landmarks in image coordinates, or None if the predictor cannot run."""
    predictor = _dlib_predictor()
    if predictor is None or box is None:
        return None
    import dlib

    x, y, w, h = box
    pad_x, pad_y = int(w * expand), int(h * expand)
    x0 = int(max(x - pad_x, 0))
    y0 = int(max(y - pad_y, 0))
    x1 = int(min(x + w + pad_x, gray.shape[1] - 1))
    y1 = int(min(y + h + pad_y, gray.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    gray_u8 = np.ascontiguousarray(gray)
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
    shape = predictor(gray_u8, dlib.rectangle(x0, y0, x1, y1))
    pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    if pts.shape != (68, 2):
        return None
    return pts


def align_face_landmarks(
    gray: np.ndarray,
    landmarks: np.ndarray,
    image_size: int = 64,
    desired_left_eye: tuple[float, float] = (0.32, 0.38),
) -> np.ndarray:
    """Similarity-warp so both eyes sit on a canonical 64×64 template.

    This is what makes the supplementary's median-68 IG overlay well-defined:
    every training face lives in the same landmark coordinate system.
    """
    left_eye = landmarks[_LEFT_EYE].mean(axis=0)
    right_eye = landmarks[_RIGHT_EYE].mean(axis=0)
    delta_y = float(right_eye[1] - left_eye[1])
    delta_x = float(right_eye[0] - left_eye[0])
    angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
    dist = float(np.hypot(delta_x, delta_y))
    desired_dist = (1.0 - 2.0 * desired_left_eye[0]) * image_size
    scale = desired_dist / max(dist, 1e-6)
    eyes_center = (
        float((left_eye[0] + right_eye[0]) * 0.5),
        float((left_eye[1] + right_eye[1]) * 0.5),
    )
    matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    matrix[0, 2] += image_size * 0.5 - eyes_center[0]
    matrix[1, 2] += image_size * desired_left_eye[1] - eyes_center[1]
    return cv2.warpAffine(
        gray,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def detect_and_crop_face(gray: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Crop the largest frontal Haar box; fall back to a centered square."""
    box = _largest_face_box(gray)
    if box is None:
        return _center_square(gray)
    x, y, w, h = box
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


def prepare_face(
    bgr_or_gray: np.ndarray,
    image_size: int = 64,
    detect_face: bool = True,
    face_margin: float = 0.0,
    align_face: bool = False,
) -> np.ndarray:
    """Return a 64×64 Haar crop in [0, 1]. Optional dlib-68 warp (supp. occlusion/IG)."""
    if bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray
    if gray.dtype != np.uint8:
        peak = float(np.max(gray)) if gray.size else 0.0
        if peak <= 1.0:
            gray = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = np.ascontiguousarray(gray)
    if not detect_face:
        crop = _center_square(gray)
        resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    box = _largest_face_box(gray)
    if align_face:
        points = landmarks_68(gray, box)
        if points is not None:
            aligned = align_face_landmarks(gray, points, image_size=image_size)
            return aligned.astype(np.float32) / 255.0

    crop = detect_and_crop_face(gray, margin=face_margin) if box is not None else _center_square(gray)
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def apply_gaussian_noise(
    image: np.ndarray,
    stddev: float,
    seed_key: str | None = None,
    clip_low: float = 0.0,
    clip_high: float = 1.0,
) -> np.ndarray:
    """Additive Gaussian noise (Keras GaussianNoise: stddev=0.01 on RAVDESS).

    Resampled every call unless `seed_key` is set. `stddev` is σ, not variance.
    """
    if stddev <= 0.0:
        return image
    if seed_key is None:
        rng = np.random.default_rng()
    else:
        seed = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, stddev, size=image.shape).astype(np.float32)
    return np.clip(image + noise, clip_low, clip_high).astype(np.float32)


def frame_indices(total: int, num_frames: int = 6) -> np.ndarray:
    """Frame index at the *center* of each of `num_frames` equal windows.

    Aligns the sampled face with the matching non-overlapping MFCC sixth.
    For T=60: 5, 15, 25, 35, 45, 55 — not the silent frame 0.
    """
    if total <= 0:
        raise ValueError(f"Need a positive frame count, got {total}")
    idx = ((np.arange(num_frames, dtype=np.float64) + 0.5) * total / num_frames).astype(int)
    return np.clip(idx, 0, total - 1)


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 6,
    image_size: int = 64,
    detect_face: bool = True,
    face_margin: float = 0.0,
    align_face: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    """Sample `num_frames` grayscale faces at one-sixth window centers.

    Optional `start_time` / `end_time` (seconds) restrict sampling to the
    speech span so intro/outro holds are not used as labeled windows.

    Returns an array of shape (T, 1, H, W) with pixels in [0, 1].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    if start_time is not None and end_time is not None and end_time > start_time:
        f0 = int(round(float(start_time) * fps))
        f1 = int(round(float(end_time) * fps))
        f0 = max(0, min(f0, max(total - 1, 0)))
        f1 = max(f0 + 1, min(f1, total))
        indices = f0 + frame_indices(f1 - f0, num_frames)
        indices = np.clip(indices, 0, total - 1)
    else:
        indices = frame_indices(total, num_frames)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        faces = prepare_face(
            frame,
            image_size=image_size,
            detect_face=detect_face,
            face_margin=face_margin,
            align_face=align_face,
        )
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


def trim_speech_interval(
    waveform: np.ndarray,
    sr: int,
    top_db: float,
    min_seconds: float = 0.4,
) -> tuple[np.ndarray, float, float]:
    """Drop leading/trailing silence. Returns (trimmed, t0, t1) in seconds of the original."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration = float(y.size) / float(sr) if sr else 0.0
    if y.size == 0 or sr <= 0:
        return y, 0.0, duration
    trimmed, (start, end) = librosa.effects.trim(y, top_db=float(top_db))
    min_keep = max(int(float(sr) * min_seconds), int(sr) // 4)
    if end - start < min_keep:
        return y, 0.0, duration
    return trimmed.astype(np.float32), start / float(sr), end / float(sr)


def load_speech_span(
    path: str | Path,
    sr: int | None = None,
    top_db: float | None = None,
    time_stretch: float | None = None,
) -> tuple[np.ndarray, int, float, float]:
    """Load mono audio, optionally trim holds, optionally stretch the *whole* utterance.

    `t0`/`t1` are the speech span in the *original* file (before stretch) so
    video frames stay aligned with the unstretched clip.
    """
    y, native = load_waveform(path, sr=sr, return_sr=True)
    duration = float(y.size) / float(native) if native else 0.0
    t0, t1 = 0.0, duration
    if top_db is not None:
        y, t0, t1 = trim_speech_interval(y, int(native), float(top_db))
    if time_stretch is not None and abs(float(time_stretch) - 1.0) > 1e-6 and y.size > 1:
        y = librosa.effects.time_stretch(y, rate=float(time_stretch))
    return y.astype(np.float32), int(native), float(t0), float(t1)


def mfcc_vectors_from_waveform(
    waveform: np.ndarray,
    sr: int,
    num_segments: int = 6,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    time_stretch: float | None = None,
) -> np.ndarray:
    """Six non-overlapping windows → 40 mean MFCCs each (Fig. 2).

    Prefer stretching the full utterance in `load_speech_span` / `clip_av_windows`.
    Per-segment stretch remains for live buffers that are already one window long.
    Returns (num_segments, n_mfcc).
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if time_stretch is not None and abs(float(time_stretch) - 1.0) > 1e-6 and waveform.size > 1:
        waveform = librosa.effects.time_stretch(waveform, rate=float(time_stretch))
    if waveform.size == 0:
        waveform = np.zeros(max(int(sr) // 10, n_fft), dtype=np.float32)
    length = waveform.shape[0]
    edges = np.linspace(0, length, num_segments + 1, dtype=int)
    vectors = [
        _mean_mfcc(waveform[edges[i] : edges[i + 1]], int(sr), n_mfcc, n_fft, hop_length)
        for i in range(num_segments)
    ]
    return np.stack(vectors, axis=0)


def extract_mfcc_vectors(
    audio_path: str | Path,
    num_segments: int = 6,
    sr: int | None = None,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    time_stretch: float | None = None,
    top_db: float | None = None,
) -> np.ndarray:
    """Six non-overlapping windows → 40 mean MFCCs each (Fig. 2).

    Returns (num_segments, n_mfcc).
    """
    waveform, native_sr, _, _ = load_speech_span(
        audio_path, sr=sr, top_db=top_db, time_stretch=time_stretch
    )
    return mfcc_vectors_from_waveform(
        waveform,
        native_sr,
        num_segments=num_segments,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        time_stretch=None,
    )


def clip_av_windows(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    num_frames: int = 6,
    image_size: int = 64,
    detect_face: bool = True,
    face_margin: float = 0.0,
    align_face: bool = False,
    sr: int | None = None,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    time_stretch: float | None = None,
    top_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned (T, 1, H, W) faces and (T, 40) mean MFCCs from the same span."""
    waveform, native_sr, t0, t1 = load_speech_span(
        audio_path, sr=sr, top_db=top_db, time_stretch=time_stretch
    )
    faces = sample_video_frames(
        video_path,
        num_frames=num_frames,
        image_size=image_size,
        detect_face=detect_face,
        face_margin=face_margin,
        align_face=align_face,
        start_time=t0,
        end_time=t1,
    )
    mfcc = mfcc_vectors_from_waveform(
        waveform,
        native_sr,
        num_segments=num_frames,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        time_stretch=None,
    )
    return faces, mfcc


def extract_mfcc_segments(
    audio_path: str | Path,
    num_segments: int = 6,
    sr: int | None = None,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    frames_per_segment: int = 32,
    time_stretch: float | None = None,
    top_db: float | None = None,
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
        top_db=top_db,
    )

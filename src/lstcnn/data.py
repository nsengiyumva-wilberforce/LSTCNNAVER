"""Audio-visual dataset scanners for RAVDESS, SAVEE, MEAD, and synthetic data."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lstcnn.cache import load_cached_clip
from lstcnn.constants import DATASET_EMOTIONS, RAVDESS_ID_TO_EMOTION, SAVEE_CODE_TO_EMOTION
from lstcnn.preprocess import apply_gaussian_noise


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac"}


@dataclass
class Sample:
    video_path: str
    audio_path: str
    label: int
    speaker: str
    emotion: str


@dataclass
class WindowItem:
    """One of the six aligned (face, MFCC) windows from a clip (paper §III)."""

    clip: Sample
    part: int

    @property
    def label(self) -> int:
        return self.clip.label

    @property
    def speaker(self) -> str:
        return self.clip.speaker

    @property
    def emotion(self) -> str:
        return self.clip.emotion

    @property
    def video_path(self) -> str:
        return self.clip.video_path


def paper_time_stretch(dataset: str, data_cfg: dict) -> float | None:
    """RAVDESS time-stretch 0.8 as train-only augmentation of the full clip."""
    if str(dataset).lower() != "ravdess":
        return None
    value = data_cfg.get("time_stretch")
    if value is None:
        return None
    rate = float(value)
    if abs(rate - 1.0) < 1e-6:
        return None
    return rate


def paper_face_noise_var(dataset: str, data_cfg: dict) -> float:
    if str(dataset).lower() != "ravdess":
        return 0.0
    return float(data_cfg.get("face_noise_var") or 0.0)


def mfcc_train_stats(
    items: list[WindowItem],
    data_cfg: dict,
    cache_dir: Path | None,
    time_stretch: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-coefficient mean/std on train windows. Preserves energy across clips."""
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for item in items:
        key = item.clip.video_path
        if key not in cache:
            _, mfcc = load_cached_clip(item.clip, data_cfg, cache_dir, time_stretch)
            cache[key] = mfcc
        rows.append(cache[key][item.part])
    stacked = np.stack(rows, axis=0) if rows else np.zeros((1, int(data_cfg.get("n_mfcc", 40))), dtype=np.float32)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def apply_mfcc_scale(
    mfcc: np.ndarray,
    mean: np.ndarray | list[float] | None,
    std: np.ndarray | list[float] | None,
) -> np.ndarray:
    if mean is None or std is None:
        return mfcc
    mean_a = np.asarray(mean, dtype=np.float32)
    std_a = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
    return ((mfcc - mean_a) / std_a).astype(np.float32)


def expand_windows(samples: list[Sample], num_parts: int) -> list[WindowItem]:
    return [WindowItem(clip=sample, part=part) for sample in samples for part in range(num_parts)]


def _label_maps(dataset: str) -> tuple[list[str], dict[str, int]]:
    names = DATASET_EMOTIONS[dataset]
    return names, {name: i for i, name in enumerate(names)}


def _ravdess_id_rest(stem: str) -> str | None:
    """Fields 2–7 of 01-01-01-01-01-01-01, used to pair video (01) with wav (03)."""
    parts = stem.split("-")
    if len(parts) < 7:
        return None
    return "-".join(parts[1:])


def _index_ravdess_wavs(root: Path) -> dict[str, Path]:
    """Map '01-01-01-01-01-01' → Audio_Speech 03-*.wav under root."""
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() != ".wav":
            continue
        rest = _ravdess_id_rest(path.stem)
        if rest:
            index[rest] = path
    return index


def scan_ravdess(root: Path, speech_only: bool = True) -> list[Sample]:
    """Parse official RAVDESS filenames: 01-01-04-01-01-01-12.mp4.

    Faces come from Video_Speech (modality 01). Audio prefers Audio_Speech
    wavs (modality 03) when they exist under the same root; otherwise the
    mp4 soundtrack is used.
    """
    _, to_id = _label_maps("ravdess")
    wavs = _index_ravdess_wavs(root)
    samples: list[Sample] = []
    skipped_video_only = 0
    skipped_song = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        parts = path.stem.split("-")
        if len(parts) < 7:
            continue
        modality, channel, emotion_id = (int(p) for p in parts[:3])
        if speech_only and channel != 1:
            skipped_song += 1
            continue
        if modality == 2:
            skipped_video_only += 1
            continue
        if modality != 1:
            continue
        emotion = RAVDESS_ID_TO_EMOTION.get(emotion_id)
        if emotion is None:
            continue
        actor = parts[6]
        rest = _ravdess_id_rest(path.stem)
        audio = wavs.get(rest, path) if rest else path
        samples.append(
            Sample(
                video_path=str(path),
                audio_path=str(audio),
                label=to_id[emotion],
                speaker=f"actor_{int(actor):02d}",
                emotion=emotion,
            )
        )
    n_wav = sum(Path(s.audio_path).suffix.lower() == ".wav" for s in samples)
    print(
        f"RAVDESS: kept {len(samples)} speech Audio-Video clips (paper 1440). "
        f"Skipped {skipped_video_only} video-only and {skipped_song} song. "
        f"Audio: {n_wav} wav / {len(samples) - n_wav} from mp4."
    )
    return samples


_SAVEE_RE = re.compile(r"^([A-Z]{2})_([a-z]+)(\d+)$", re.IGNORECASE)


def scan_savee(root: Path) -> list[Sample]:
    """Match SAVEE stems such as DC_a01 / DC_sa01 / DC_su12."""
    _, to_id = _label_maps("savee")
    videos: dict[str, Path] = {}
    audios: dict[str, Path] = {}
    for path in root.rglob("*"):
        key = path.stem.lower()
        if path.suffix.lower() in VIDEO_EXTS:
            videos[key] = path
        elif path.suffix.lower() in AUDIO_EXTS:
            audios[key] = path

    samples: list[Sample] = []
    keys = sorted(set(videos) | set(audios))
    for key in keys:
        match = _SAVEE_RE.match(key)
        if not match:
            continue
        speaker, code, _ = match.groups()
        code = code.lower()
        emotion = SAVEE_CODE_TO_EMOTION.get(code)
        if emotion is None:
            continue
        video = videos.get(key) or audios.get(key)
        audio = audios.get(key) or videos.get(key)
        if video is None or audio is None:
            continue
        samples.append(
            Sample(
                video_path=str(video),
                audio_path=str(audio),
                label=to_id[emotion],
                speaker=speaker.upper(),
                emotion=emotion,
            )
        )
    return samples


def scan_mead(root: Path, view: str = "front") -> list[Sample]:
    """MEAD layout: {actor}/video/{view}/{emotion}/{intensity}/*.mp4."""
    _, to_id = _label_maps("mead")
    samples: list[Sample] = []
    for video in root.rglob("*"):
        if video.suffix.lower() not in VIDEO_EXTS:
            continue
        parts = video.parts
        try:
            v_idx = next(i for i, p in enumerate(parts) if p.lower() == "video")
        except StopIteration:
            continue
        if v_idx + 3 >= len(parts):
            continue
        actor = parts[v_idx - 1] if v_idx > 0 else "unknown"
        video_view = parts[v_idx + 1]
        emotion_folder = parts[v_idx + 2].lower()
        if video_view.lower() != view.lower():
            continue
        emotion = _normalize_mead_emotion(emotion_folder)
        if emotion not in to_id:
            continue
        audio = _find_mead_audio(root, actor, emotion_folder, video)
        samples.append(
            Sample(
                video_path=str(video),
                audio_path=str(audio or video),
                label=to_id[emotion],
                speaker=str(actor),
                emotion=emotion,
            )
        )
    return samples


def _normalize_mead_emotion(name: str) -> str:
    aliases = {
        "anger": "angry",
        "angry": "angry",
        "contempt": "contempt",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "fear": "fear",
        "happy": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprised",
        "surprised": "surprised",
        "neutral": "neutral",
    }
    return aliases.get(name.lower(), name.lower())


def _find_mead_audio(root: Path, actor: str, emotion_folder: str, video: Path) -> Path | None:
    stem = video.stem
    candidates = [
        root / actor / "audio" / emotion_folder,
        video.parent,
        video.parent.parent,
    ]
    for folder in candidates:
        if not folder.exists():
            continue
        for ext in AUDIO_EXTS:
            match = folder / f"{stem}{ext}"
            if match.exists():
                return match
            nested = list(folder.rglob(f"{stem}{ext}"))
            if nested:
                return nested[0]
    return None


def scan_dataset(name: str, root: Path, speech_only: bool = True) -> list[Sample]:
    name = name.lower()
    if name == "ravdess":
        return scan_ravdess(root, speech_only=speech_only)
    if name == "savee":
        return scan_savee(root)
    if name == "mead":
        return scan_mead(root)
    raise ValueError(f"Unknown dataset '{name}'.")


def _nested_holdout(
    samples: list,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool,
) -> dict[str, list]:
    """80/20 train–test, then 80/20 train–val on the remainder (paper protocol).

    `val_ratio` is the fraction of the *trainval* set used for validation.
    """
    from sklearn.model_selection import train_test_split

    if not samples:
        return {"train": [], "val": [], "test": []}

    def _split(items: list, test_size: float, rng_seed: int) -> tuple[list, list]:
        if len(items) < 2 or test_size <= 0:
            return items, []
        labels = [s.label for s in items] if stratify else None
        try:
            left, right = train_test_split(
                items,
                test_size=test_size,
                random_state=rng_seed,
                stratify=labels,
            )
        except ValueError:
            left, right = train_test_split(items, test_size=test_size, random_state=rng_seed)
        return list(left), list(right)

    trainval, test = _split(samples, test_ratio, seed)
    train, val = _split(trainval, val_ratio, seed + 1)
    return {"train": train, "val": val, "test": test}


def _speaker_holdout(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    speakers = sorted({s.speaker for s in samples})
    rng.shuffle(speakers)
    n = len(speakers)
    n_test = max(1, int(round(n * test_ratio))) if n else 0
    n_val = max(1, int(round(n * val_ratio))) if n else 0
    test_spk = set(speakers[:n_test])
    val_spk = set(speakers[n_test : n_test + n_val])
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        if sample.speaker in test_spk:
            splits["test"].append(sample)
        elif sample.speaker in val_spk:
            splits["val"].append(sample)
        else:
            splits["train"].append(sample)
    return splits


def split_samples(
    samples: list[Sample],
    mode: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    num_parts: int = 6,
) -> dict[str, list[WindowItem]]:
    """Return window items. Default `stratified` matches the paper (8640 segments).

    `stratified` / `random`: expand each clip to `num_parts` windows, then nested
    80/20. Windows from the same video can land in different splits (paper protocol).
    `clip` / `stratified_clip`: hold out whole videos, then expand.
    `speaker`: hold out whole speakers, then expand.
    """
    mode = mode.lower()
    if mode in {"clip", "stratified_clip", "random_clip"}:
        clip_splits = _nested_holdout(
            samples,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify=mode != "random_clip",
        )
        return {name: expand_windows(subset, num_parts) for name, subset in clip_splits.items()}

    if mode == "speaker":
        clip_splits = _speaker_holdout(samples, val_ratio, test_ratio, seed)
        return {name: expand_windows(subset, num_parts) for name, subset in clip_splits.items()}

    windows = expand_windows(samples, num_parts)
    if mode in {"stratified", "random"}:
        return _nested_holdout(
            windows,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify=mode == "stratified",
        )
    raise ValueError(f"Unknown split mode '{mode}'. Use stratified, clip, or speaker.")


class AudioVisualDataset(Dataset):
    """One Haar face + 40-d mean MFCC per index (already window-expanded)."""

    def __init__(
        self,
        items: list[WindowItem],
        cfg: dict,
        mfcc_mean: np.ndarray | None = None,
        mfcc_std: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.items = items
        self.cfg = cfg["data"]
        self.dataset_name = cfg["data"]["dataset"].lower()
        cache = self.cfg.get("cache_dir")
        self.cache_dir = Path(cache) / self.dataset_name if cache else None
        self.time_stretch = paper_time_stretch(self.dataset_name, self.cfg)
        self.face_noise_var = paper_face_noise_var(self.dataset_name, self.cfg)
        self.augment = bool(augment)
        n_mfcc = int(self.cfg.get("n_mfcc", 40))
        self.mfcc_mean = np.zeros(n_mfcc, dtype=np.float32) if mfcc_mean is None else np.asarray(mfcc_mean, dtype=np.float32)
        self.mfcc_std = np.ones(n_mfcc, dtype=np.float32) if mfcc_std is None else np.asarray(mfcc_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        item = self.items[index]
        sample, part = item.clip, item.part
        try:
            stretch = self.time_stretch if self.augment else None
            faces, mfcc = load_cached_clip(
                sample, self.cfg, self.cache_dir, stretch
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load {sample.video_path}") from exc
        face = np.array(faces[part], copy=True)
        vector = np.array(mfcc[part], copy=True)
        vector = ((vector - self.mfcc_mean) / self.mfcc_std).astype(np.float32)
        if self.augment and self.face_noise_var > 0.0:
            # Keras GaussianNoise(0.01): stddev 0.01, a new draw each epoch.
            face = apply_gaussian_noise(face, self.face_noise_var)
        return {
            "faces": torch.from_numpy(face),
            "mfcc": torch.from_numpy(vector),
            "label": sample.label,
            "speaker": sample.speaker,
            "emotion": sample.emotion,
            "video_path": sample.video_path,
        }


class SyntheticAVDataset(Dataset):
    """Deterministic dummy clips so the pipeline can run without corpora."""

    def __init__(self, size: int, cfg: dict, seed: int = 0) -> None:
        self.size = size
        self.cfg = cfg["data"]
        self.num_classes = cfg["model"]["num_classes"]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        data = self.cfg
        label = index % self.num_classes
        rng = np.random.default_rng(index)
        faces = rng.normal(0.0, 0.3, size=(1, data["image_size"], data["image_size"]))
        faces = np.clip(faces + 0.15 * label, -1.0, 1.0).astype(np.float32)
        mfcc = rng.normal(0.0, 0.3, size=(data["n_mfcc"],)).astype(np.float32)
        mfcc = (mfcc + 0.2 * label).astype(np.float32)
        emotion = DATASET_EMOTIONS["synthetic"][label]
        return {
            "faces": torch.from_numpy(faces),
            "mfcc": torch.from_numpy(mfcc),
            "label": label,
            "speaker": f"synth_{index % 8}",
            "emotion": emotion,
            "video_path": f"synthetic/{index}.mp4",
        }

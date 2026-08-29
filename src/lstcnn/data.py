"""Audio-visual dataset scanners for RAVDESS, SAVEE, MEAD, and synthetic data."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lstcnn.constants import DATASET_EMOTIONS, RAVDESS_ID_TO_EMOTION, SAVEE_CODE_TO_EMOTION
from lstcnn.preprocess import extract_mfcc_segments, sample_video_frames


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac"}


@dataclass
class Sample:
    video_path: str
    audio_path: str
    label: int
    speaker: str
    emotion: str


def _label_maps(dataset: str) -> tuple[list[str], dict[str, int]]:
    names = DATASET_EMOTIONS[dataset]
    return names, {name: i for i, name in enumerate(names)}


def scan_ravdess(root: Path, speech_only: bool = True) -> list[Sample]:
    """Parse official RAVDESS filenames: 03-01-04-01-01-01-12.mp4."""
    _, to_id = _label_maps("ravdess")
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        parts = path.stem.split("-")
        if len(parts) < 7:
            continue
        modality, channel, emotion_id = (int(p) for p in parts[:3])
        if speech_only and channel != 1:
            continue
        if modality not in (1, 2):
            continue
        emotion = RAVDESS_ID_TO_EMOTION.get(emotion_id)
        if emotion is None:
            continue
        actor = parts[6]
        samples.append(
            Sample(
                video_path=str(path),
                audio_path=str(path),
                label=to_id[emotion],
                speaker=f"actor_{int(actor):02d}",
                emotion=emotion,
            )
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


def split_samples(
    samples: list[Sample],
    mode: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    if mode == "random":
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_ratio)) if n else 0
        n_val = max(1, int(n * val_ratio)) if n else 0
        test = shuffled[:n_test]
        val = shuffled[n_test : n_test + n_val]
        train = shuffled[n_test + n_val :]
        return {"train": train, "val": val, "test": test}

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


class AudioVisualDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: dict) -> None:
        self.samples = samples
        self.cfg = cfg["data"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.samples[index]
        faces = sample_video_frames(
            sample.video_path,
            num_frames=self.cfg["num_frames"],
            image_size=self.cfg["image_size"],
            detect_face=self.cfg["detect_face"],
        )
        mfcc = extract_mfcc_segments(
            sample.audio_path,
            num_segments=self.cfg["num_audio_segments"],
            sr=self.cfg["sample_rate"],
            n_mfcc=self.cfg["n_mfcc"],
            n_fft=self.cfg["n_fft"],
            hop_length=self.cfg["hop_length"],
            frames_per_segment=self.cfg["mfcc_frames_per_segment"],
        )
        return {
            "faces": torch.from_numpy(faces),
            "mfcc": torch.from_numpy(mfcc),
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
        faces = rng.normal(0.0, 0.3, size=(data["num_frames"], 1, data["image_size"], data["image_size"]))
        faces = np.clip(faces + 0.15 * label, -1.0, 1.0).astype(np.float32)
        mfcc = rng.normal(
            0.0,
            0.3,
            size=(
                data["num_audio_segments"],
                data["n_mfcc"],
                data["mfcc_frames_per_segment"],
            ),
        )
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

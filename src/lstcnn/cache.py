"""Disk cache of Haar faces + mean MFCCs so training does not re-decode every epoch."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from lstcnn.preprocess import clip_av_windows, trim_top_db_from_cfg


def cache_key(sample: Any, data_cfg: dict, time_stretch: float | None) -> str:
    payload = "|".join(
        [
            sample.video_path,
            sample.audio_path,
            str(data_cfg.get("image_size", 64)),
            str(data_cfg.get("num_frames", 6)),
            str(data_cfg.get("n_mfcc", 40)),
            str(data_cfg.get("n_fft", 2048)),
            str(data_cfg.get("hop_length", 512)),
            str(data_cfg.get("sample_rate")),
            str(bool(data_cfg.get("detect_face", True))),
            str(data_cfg.get("face_margin", 0.0)),
            str(bool(data_cfg.get("align_face", False))),
            str(time_stretch),
            str(bool(data_cfg.get("trim_silence", True))),
            str(data_cfg.get("trim_top_db", 30)),
            "face_unit=0-1",
            "mfcc_raw=1",
            "align=haar",
            "frame=center",
            "stretch=full_clip",
            "trim=librosa",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    stem = Path(sample.video_path).stem
    return f"{stem}_{digest}.npz"


def cache_path(cache_dir: Path, sample: Any, data_cfg: dict, time_stretch: float | None) -> Path:
    return Path(cache_dir) / cache_key(sample, data_cfg, time_stretch)


def compute_clip_features(
    sample: Any,
    data_cfg: dict,
    time_stretch: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(data_cfg.get("num_frames", 6))
    faces, mfcc = clip_av_windows(
        sample.video_path,
        sample.audio_path,
        num_frames=n,
        image_size=int(data_cfg.get("image_size", 64)),
        detect_face=bool(data_cfg.get("detect_face", True)),
        face_margin=float(data_cfg.get("face_margin", 0.0)),
        align_face=bool(data_cfg.get("align_face", False)),
        sr=data_cfg.get("sample_rate"),
        n_mfcc=int(data_cfg.get("n_mfcc", 40)),
        n_fft=int(data_cfg.get("n_fft", 2048)),
        hop_length=int(data_cfg.get("hop_length", 512)),
        time_stretch=time_stretch,
        top_db=trim_top_db_from_cfg(data_cfg),
    )
    return faces.astype(np.float32), mfcc.astype(np.float32)


def load_cached_clip(
    sample: Any,
    data_cfg: dict,
    cache_dir: Path | None,
    time_stretch: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_dir is None:
        return compute_clip_features(sample, data_cfg, time_stretch)
    path = cache_path(cache_dir, sample, data_cfg, time_stretch)
    if path.is_file():
        with np.load(path) as blob:
            return blob["faces"], blob["mfcc"]
    faces, mfcc = compute_clip_features(sample, data_cfg, time_stretch)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, faces=faces, mfcc=mfcc)
    return faces, mfcc


def _warm_one(payload: tuple) -> str:
    sample, data_cfg, cache_dir, time_stretch = payload
    path = cache_path(Path(cache_dir), sample, data_cfg, time_stretch)
    if path.is_file():
        return str(path)
    load_cached_clip(sample, data_cfg, Path(cache_dir), time_stretch)
    return str(path)


def warm_feature_cache(
    samples: list[Any],
    data_cfg: dict,
    cache_dir: Path,
    time_stretch: float | None,
    workers: int = 4,
) -> None:
    """Decode each unique clip once and write npz files."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        s
        for s in samples
        if not cache_path(cache_dir, s, data_cfg, time_stretch).is_file()
    ]
    if not pending:
        print(f"Feature cache ready ({len(samples)} clips) at {cache_dir}")
        return
    print(f"Caching {len(pending)} / {len(samples)} clips → {cache_dir}")
    jobs = [(s, data_cfg, str(cache_dir), time_stretch) for s in pending]
    workers = max(1, int(workers))
    if workers == 1:
        for job in tqdm(jobs, desc="cache"):
            _warm_one(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_warm_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="cache"):
            fut.result()

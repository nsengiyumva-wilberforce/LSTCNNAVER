from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.config import load_config
from lstcnn.explain import face_heatmap, integrated_gradients, mfcc_coefficient_importance
from lstcnn.flops import PAPER_GFLOPS, PAPER_PARAMS_M, count_macs, gflops_from_macs
from lstcnn.model import apply_dataset_hparams, build_model, conv_layer_counts, count_parameters


def _cfg():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["data"]["dataset"] = "synthetic"
    cfg["model"]["num_classes"] = 8
    cfg["model"]["boost"] = False
    apply_dataset_hparams(cfg)
    return cfg


def test_paper_layer_counts_and_widths():
    cfg = _cfg()
    model = build_model(cfg)
    n2d, n1d = conv_layer_counts(model)
    assert n2d == 3, n2d
    assert n1d == 2, n1d
    vis = [b.conv.out_channels for b in model.visual.backbone]
    aud = [b.conv.out_channels for b in model.audio.backbone]
    assert vis == [16, 32, 64], vis
    assert aud == [16, 32], aud
    assert model.visual.backbone[0].conv.in_channels == 1
    assert model.audio.backbone[0].conv.in_channels == 1
    assert model.visual.backbone[0].conv.kernel_size == (3, 3)
    assert model.audio.backbone[0].conv.kernel_size == (5,)
    assert model.visual.feature_dim == 2304
    assert model.audio.feature_dim == 224
    assert model.fusion_hidden == 16
    assert isinstance(model.visual.drop, torch.nn.Dropout)
    assert isinstance(model.audio.drop, torch.nn.Dropout)


def test_forward_shapes():
    cfg = _cfg()
    model = build_model(cfg)
    model.eval()
    batch = 4
    size = cfg["data"]["image_size"]
    faces = torch.randn(batch, 1, size, size)
    mfcc = torch.randn(batch, cfg["data"]["n_mfcc"])
    logits = model(faces, mfcc)
    assert logits.shape == (batch, 8)


def test_paper_parameter_and_flop_budget():
    cfg = _cfg()
    model = build_model(cfg)
    n = count_parameters(model)
    gflops = gflops_from_macs(count_macs(model, n_mfcc=cfg["data"]["n_mfcc"]))
    assert abs(n / 1e6 - PAPER_PARAMS_M) < 0.01, n
    # Fig. 3 valid padding is ~0.009 G with eq. (3); Table V rounds to 0.014.
    assert gflops < PAPER_GFLOPS + 0.001, gflops
    assert gflops > 0.008, gflops


def test_savee_fusion_width():
    cfg = _cfg()
    cfg["data"]["dataset"] = "savee"
    cfg["model"]["num_classes"] = 7
    apply_dataset_hparams(cfg)
    model = build_model(cfg)
    assert cfg["model"]["dropout"] == 0.5
    assert model.fusion_hidden == 14
    assert model.num_classes == 7


def test_integrated_gradients():
    cfg = _cfg()
    model = build_model(cfg)
    model.eval()
    faces = torch.randn(1, 1, 64, 64)
    mfcc = torch.randn(1, 40)
    ig = integrated_gradients(model, faces, mfcc, steps=8, branch="both")
    assert ig["face_attr"].shape == faces.shape
    assert ig["mfcc_attr"].shape == mfcc.shape
    heat = face_heatmap(ig["face_attr"])
    assert heat.shape == (64, 64)
    coeffs = mfcc_coefficient_importance(ig["mfcc_attr"])
    assert coeffs.shape == (40,)


def test_one_sixth_frame_indices():
    from lstcnn.preprocess import frame_indices

    idx = frame_indices(60, 6)
    assert list(idx) == [5, 15, 25, 35, 45, 55]


def test_stratified_nested_split():
    from lstcnn.data import Sample, split_samples

    samples = [
        Sample("v", "a", label=i % 5, speaker=str(i), emotion="e")
        for i in range(100)
    ]
    splits = split_samples(samples, "stratified", val_ratio=0.2, test_ratio=0.2, seed=0, num_parts=1)
    n = len(samples)
    assert abs(len(splits["test"]) / n - 0.2) < 0.05
    rest = n - len(splits["test"])
    assert abs(len(splits["val"]) / rest - 0.2) < 0.05
    ids = [{id(s) for s in splits[k]} for k in ("train", "val", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert sum(len(s) for s in splits.values()) == n


def test_paper_window_split_expands_and_can_leak_clip():
    from lstcnn.data import Sample, split_samples

    samples = [
        Sample(f"v{i}", "a", label=i % 4, speaker="s", emotion="e")
        for i in range(40)
    ]
    splits = split_samples(samples, "stratified", val_ratio=0.2, test_ratio=0.2, seed=0, num_parts=6)
    assert sum(len(part) for part in splits.values()) == 240
    train_clips = {item.clip.video_path for item in splits["train"]}
    test_clips = {item.clip.video_path for item in splits["test"]}
    assert train_clips & test_clips, "paper protocol splits windows, so clips may overlap"


def test_clip_split_holds_out_videos():
    from lstcnn.data import Sample, split_samples

    samples = [
        Sample(f"v{i}", "a", label=i % 4, speaker="s", emotion="e")
        for i in range(40)
    ]
    splits = split_samples(samples, "clip", val_ratio=0.2, test_ratio=0.2, seed=0, num_parts=6)
    assert sum(len(part) for part in splits.values()) == 240
    sets = [{item.clip.video_path for item in splits[k]} for k in ("train", "val", "test")]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])


def test_cache_key_includes_stretch():
    from lstcnn.cache import cache_key
    from lstcnn.data import Sample

    sample = Sample("a.mp4", "a.mp4", 0, "x", "happy")
    cfg = {
        "image_size": 64,
        "num_frames": 6,
        "n_mfcc": 40,
        "n_fft": 2048,
        "hop_length": 512,
        "sample_rate": None,
        "detect_face": True,
    }
    assert cache_key(sample, cfg, None) == cache_key(sample, cfg, None)
    assert cache_key(sample, cfg, 0.8) != cache_key(sample, cfg, None)


def test_trim_speech_interval_drops_holds():
    from lstcnn.preprocess import trim_speech_interval

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    speech = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = np.concatenate([np.zeros(sr, dtype=np.float32), speech, np.zeros(sr, dtype=np.float32)])
    trimmed, t0, t1 = trim_speech_interval(y, sr, top_db=30)
    assert 0.7 < t0 < 1.3
    assert 1.7 < t1 < 2.3
    assert trimmed.shape[0] < y.shape[0]


def test_full_clip_stretch_then_split():
    from lstcnn.preprocess import mfcc_vectors_from_waveform

    sr = 16000
    y = (0.2 * np.sin(2 * np.pi * 180 * np.linspace(0, 1.2, int(sr * 1.2), endpoint=False))).astype(np.float32)
    raw = mfcc_vectors_from_waveform(y, sr, num_segments=6, n_mfcc=40, n_fft=512, hop_length=256)
    stretched = mfcc_vectors_from_waveform(
        y, sr, num_segments=6, n_mfcc=40, n_fft=512, hop_length=256, time_stretch=0.8
    )
    assert raw.shape == stretched.shape == (6, 40)
    assert not np.allclose(raw, stretched)


def test_eye_alignment_output_size():
    from lstcnn.preprocess import align_face_landmarks

    gray = np.zeros((200, 200), dtype=np.uint8)
    landmarks = np.full((68, 2), 100.0, dtype=np.float32)
    landmarks[36:42] = (70.0, 80.0)
    landmarks[42:48] = (130.0, 80.0)
    aligned = align_face_landmarks(gray, landmarks, image_size=64)
    assert aligned.shape == (64, 64)


def test_ravdess_pairs_speech_wav(tmp_path):
    from lstcnn.data import scan_ravdess

    vdir = tmp_path / "Video_Speech_Actor_01" / "Actor_01"
    adir = tmp_path / "Audio_Speech_Actors" / "Actor_01"
    vdir.mkdir(parents=True)
    adir.mkdir(parents=True)
    (vdir / "01-01-01-01-01-01-01.mp4").write_bytes(b"x")
    (adir / "03-01-01-01-01-01-01.wav").write_bytes(b"x")
    samples = scan_ravdess(tmp_path)
    assert len(samples) == 1
    assert samples[0].video_path.endswith(".mp4")
    assert samples[0].audio_path.endswith("03-01-01-01-01-01-01.wav")


def test_mean_mfcc_is_raw():
    from lstcnn.preprocess import _mean_mfcc

    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    vec = _mean_mfcc(wave, sr, n_mfcc=40, n_fft=2048, hop_length=512)
    assert vec.shape == (40,)
    zscored = abs(float(vec.mean())) < 1e-5 and abs(float(vec.std()) - 1.0) < 1e-4
    assert not zscored


def test_prepare_face_unit_range():
    from lstcnn.preprocess import prepare_face

    gray = np.full((80, 80), 128, dtype=np.uint8)
    face = prepare_face(gray, image_size=64, detect_face=False)
    assert face.shape == (64, 64)
    assert 0.0 <= float(face.min()) <= float(face.max()) <= 1.0
    assert abs(float(face.mean()) - 128 / 255) < 1e-5


def test_gaussian_noise_is_stddev_and_resamples():
    from lstcnn.preprocess import apply_gaussian_noise

    image = np.full((8, 8), 0.5, dtype=np.float32)
    silent = apply_gaussian_noise(image, 0.0)
    assert np.array_equal(silent, image)
    a = apply_gaussian_noise(image, 0.01, seed_key=None)
    b = apply_gaussian_noise(image, 0.01, seed_key=None)
    assert a.shape == image.shape
    assert not np.allclose(a, b)
    # σ=0.01, not √0.01≈0.1: rms should be near 0.01 not 0.1.
    rms = float(np.sqrt(np.mean((a - image) ** 2)))
    assert 0.002 < rms < 0.03


def test_augment_face_keeps_shape():
    from lstcnn.data import augment_face, augment_mfcc

    face = np.random.rand(1, 64, 64).astype(np.float32)
    assert augment_face(face).shape == face.shape
    assert augment_mfcc(np.random.randn(40).astype(np.float32)).shape == (40,)


def test_boosted_complexity_is_nonzero():
    from lstcnn.flops import model_complexity

    cfg = _cfg()
    cfg["model"]["boost"] = True
    apply_dataset_hparams(cfg)
    stats = model_complexity(build_model(cfg))
    assert stats["n_params"] > 100_000
    assert stats["gflops"] > 0.01
    assert stats["visual_params"] > stats["audio_params"]


def test_boosted_forward_shapes():
    cfg = _cfg()
    cfg["model"]["boost"] = True
    apply_dataset_hparams(cfg)
    model = build_model(cfg)
    model.eval()
    faces = torch.randn(3, 1, 64, 64)
    mfcc = torch.randn(3, 40)
    logits = model(faces, mfcc)
    assert logits.shape == (3, 8)
    assert model.forward_visual(faces).shape == (3, 8)
    assert model.forward_audio(mfcc).shape == (3, 8)
    assert getattr(model, "is_boosted", False)


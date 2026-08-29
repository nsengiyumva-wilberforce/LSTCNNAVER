from __future__ import annotations

import sys
from pathlib import Path

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
    assert list(idx) == [0, 10, 20, 30, 40, 50]


def test_stratified_nested_split():
    from lstcnn.data import Sample, split_samples

    samples = [
        Sample("v", "a", label=i % 5, speaker=str(i), emotion="e")
        for i in range(100)
    ]
    splits = split_samples(samples, "stratified", val_ratio=0.2, test_ratio=0.2, seed=0)
    n = len(samples)
    assert abs(len(splits["test"]) / n - 0.2) < 0.05
    rest = n - len(splits["test"])
    assert abs(len(splits["val"]) / rest - 0.2) < 0.05
    ids = [{id(s) for s in splits[k]} for k in ("train", "val", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert sum(len(s) for s in splits.values()) == n

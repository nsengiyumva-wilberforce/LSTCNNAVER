from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lstcnn.config import load_config
from lstcnn.explain import face_heatmap, integrated_gradients, mfcc_coefficient_importance
from lstcnn.flops import PAPER_GFLOPS, PAPER_PARAMS_M, count_macs, gflops_from_macs
from lstcnn.model import build_model, conv_layer_counts, count_parameters


def _cfg():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["data"]["dataset"] = "synthetic"
    cfg["model"]["num_classes"] = 8
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
    assert model.visual.backbone[0].conv.in_channels == 6


def test_forward_shapes():
    cfg = _cfg()
    model = build_model(cfg)
    model.eval()
    batch = 4
    frames = cfg["data"]["num_frames"]
    size = cfg["data"]["image_size"]
    faces = torch.randn(batch, frames, 1, size, size)
    mfcc = torch.randn(
        batch,
        cfg["data"]["num_audio_segments"],
        cfg["data"]["n_mfcc"],
        cfg["data"]["mfcc_frames_per_segment"],
    )
    logits = model(faces, mfcc)
    assert logits.shape == (batch, 8)
    assert frames == 6
    assert cfg["data"]["num_audio_segments"] == 6


def test_paper_parameter_and_flop_budget():
    cfg = _cfg()
    model = build_model(cfg)
    n = count_parameters(model)
    gflops = gflops_from_macs(
        count_macs(
            model,
            image_size=cfg["data"]["image_size"],
            mfcc_frames_per_segment=cfg["data"]["mfcc_frames_per_segment"],
            num_frames=cfg["data"]["num_frames"],
            n_mfcc=cfg["data"]["n_mfcc"],
        )
    )
    assert abs(n / 1e6 - PAPER_PARAMS_M) < 0.01, n
    assert abs(gflops - PAPER_GFLOPS) < 0.005, gflops


def test_integrated_gradients():
    cfg = _cfg()
    model = build_model(cfg)
    model.eval()
    faces = torch.randn(1, 6, 1, 64, 64)
    mfcc = torch.randn(1, 6, 40, 32)
    ig = integrated_gradients(model, faces, mfcc, steps=8, branch="both")
    assert ig["face_attr"].shape == faces.shape
    assert ig["mfcc_attr"].shape == mfcc.shape
    heat = face_heatmap(ig["face_attr"])
    assert heat.shape == (64, 64)
    coeffs = mfcc_coefficient_importance(ig["mfcc_attr"])
    assert coeffs.shape == (40,)

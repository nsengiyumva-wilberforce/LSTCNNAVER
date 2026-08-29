"""MAC/FLOP counter using Ding et al. eq. (3): (2×Yh Yw Ci Co Kh Kw) + Yh Yw Co."""

from __future__ import annotations

from lstcnn.model import LightweightSTCNN

PAPER_PARAMS_M = 0.06
PAPER_GFLOPS = 0.014


def conv_flops(yh: int, yw: int, ci: int, co: int, kh: int, kw: int) -> int:
    return (2 * yh * yw * ci * co * kh * kw) + yh * yw * co


def dense_flops(width: int, neurons: int) -> int:
    return (2 * width * neurons) + neurons


def count_macs(
    model: LightweightSTCNN,
    image_size: int = 64,
    n_mfcc: int = 40,
    **_unused,
) -> int:
    """Paper FLOPs for one face + one 40-d MFCC vector (Fig. 3)."""
    del image_size
    flops = 0
    # Spatial: 64 → 62/31 → 29/14 → 12/6  (valid 3×3, pool 2)
    sizes = [(62, 62, 1, 16, 3, 3), (29, 29, 16, 32, 3, 3), (12, 12, 32, 64, 3, 3)]
    for yh, yw, ci, co, kh, kw in sizes:
        flops += conv_flops(yh, yw, ci, co, kh, kw)
    # Temporal: 40 → 36/18 → 14/7  (valid 5×1, pool 2)
    flops += conv_flops(36, 1, 1, 16, 5, 1)
    flops += conv_flops(14, 1, 16, 32, 5, 1)
    hidden = model.fusion_hidden
    flops += dense_flops(model.fused_dim, hidden)
    flops += dense_flops(hidden, model.num_classes)
    return flops


def gflops_from_macs(macs: int) -> float:
    return macs / 1e9

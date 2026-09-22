"""MAC/FLOP counter using Ding et al. eq. (3): (2×Yh Yw Ci Co Kh Kw) + Yh Yw Co."""

from __future__ import annotations

import torch
from torch import nn

PAPER_PARAMS_M = 0.06
PAPER_GFLOPS = 0.014


def conv_flops(yh: int, yw: int, ci: int, co: int, kh: int, kw: int) -> int:
    return (2 * yh * yw * ci * co * kh * kw) + yh * yw * co


def dense_flops(width: int, neurons: int) -> int:
    return (2 * width * neurons) + neurons


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs_hooks(model: nn.Module, image_size: int = 64, n_mfcc: int = 40) -> int:
    """Eq. (3) over a dummy forward: Conv2d, Conv1d, and Linear only."""
    device = next(model.parameters()).device
    total = 0

    def conv2d_hook(mod: nn.Conv2d, _inp, out) -> None:
        nonlocal total
        yh, yw = int(out.shape[-2]), int(out.shape[-1])
        kh, kw = mod.kernel_size
        total += conv_flops(yh, yw, mod.in_channels, mod.out_channels, kh, kw)

    def conv1d_hook(mod: nn.Conv1d, _inp, out) -> None:
        nonlocal total
        length = int(out.shape[-1])
        kernel = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else int(mod.kernel_size)
        total += conv_flops(length, 1, mod.in_channels, mod.out_channels, kernel, 1)

    def linear_hook(mod: nn.Linear, _inp, _out) -> None:
        nonlocal total
        total += dense_flops(mod.in_features, mod.out_features)

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(
            torch.zeros(1, 1, image_size, image_size, device=device),
            torch.zeros(1, n_mfcc, device=device),
        )
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return total


def count_macs(
    model: nn.Module,
    image_size: int = 64,
    n_mfcc: int = 40,
    **_unused,
) -> int:
    """Paper FLOPs for one face + one 40-d MFCC vector (Fig. 3)."""
    if getattr(model, "is_boosted", False):
        return count_macs_hooks(model, image_size=image_size, n_mfcc=n_mfcc)
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


def model_complexity(model: nn.Module, image_size: int = 64, n_mfcc: int = 40) -> dict[str, float | int]:
    n_params = count_parameters(model)
    visual = count_parameters(model.visual) if hasattr(model, "visual") else 0
    audio = count_parameters(model.audio) if hasattr(model, "audio") else 0
    macs = count_macs(model, image_size=image_size, n_mfcc=n_mfcc)
    return {
        "n_params": n_params,
        "params_m": n_params / 1e6,
        "visual_params": visual,
        "audio_params": audio,
        "fusion_params": max(n_params - visual - audio, 0),
        "macs": macs,
        "gflops": gflops_from_macs(macs),
        "paper_params_m": PAPER_PARAMS_M,
        "paper_gflops": PAPER_GFLOPS,
    }


def print_model_complexity(model: nn.Module, image_size: int = 64, n_mfcc: int = 40) -> dict[str, float | int]:
    stats = model_complexity(model, image_size=image_size, n_mfcc=n_mfcc)
    name = "Boosted ST-CNN" if getattr(model, "is_boosted", False) else "LST-CNN (Fig. 3)"
    print("Model complexity")
    print(f"  architecture   {name}")
    print(
        f"  parameters     {stats['n_params']:,}  ({stats['params_m']:.3f}M, "
        f"paper {PAPER_PARAMS_M}M)"
    )
    print(
        f"  FLOPs          {stats['gflops']:.4f} G  "
        f"(paper {PAPER_GFLOPS} G, eq. 3, one 64×64 face + 40-d MFCC)"
    )
    print(
        f"  by branch      visual {stats['visual_params']:,}  "
        f"audio {stats['audio_params']:,}  "
        f"fusion {stats['fusion_params']:,}"
    )
    return stats

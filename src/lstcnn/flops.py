"""MAC/FLOP counter. The paper reports 0.06M params and 0.014 GFLOPs."""

from __future__ import annotations

from lstcnn.model import LightweightSTCNN

PAPER_PARAMS_M = 0.06
PAPER_GFLOPS = 0.014


def conv2d_macs(in_ch: int, out_ch: int, kernel: int, height: int, width: int) -> int:
    return height * width * kernel * kernel * in_ch * out_ch


def conv1d_macs(in_ch: int, out_ch: int, kernel: int, length: int) -> int:
    return length * kernel * in_ch * out_ch


def linear_macs(in_features: int, out_features: int) -> int:
    return in_features * out_features


def count_macs(
    model: LightweightSTCNN,
    image_size: int = 64,
    mfcc_frames_per_segment: int = 32,
    num_frames: int = 6,
    n_mfcc: int = 40,
) -> int:
    """Multiply-accumulates for one clip (6 stacked frames + 6 audio segments)."""
    macs = 0
    vis_in = num_frames if model.stack_frames_as_channels else 1
    repeats = 1 if model.stack_frames_as_channels else num_frames
    h = w = image_size
    prev = vis_in
    for block in model.visual.backbone:
        out_ch = block.conv.out_channels
        k = block.conv.kernel_size[0]
        macs += repeats * conv2d_macs(prev, out_ch, k, h, w)
        prev = out_ch
        h //= 2
        w //= 2

    prev = n_mfcc
    length = mfcc_frames_per_segment
    for block in model.audio.backbone:
        out_ch = block.conv.out_channels
        k = block.conv.kernel_size[0]
        macs += num_frames * conv1d_macs(prev, out_ch, k, length)
        prev = out_ch
        length //= 2

    from torch import nn

    for lin in model.classifier.modules():
        if isinstance(lin, nn.Linear):
            macs += linear_macs(lin.in_features, lin.out_features)
    return macs


def gflops_from_macs(macs: int) -> float:
    return macs / 1e9

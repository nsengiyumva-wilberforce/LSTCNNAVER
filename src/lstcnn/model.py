"""Lightweight Spatio-Temporal CNN (Ding, Tang, Lu, IEEE TAFFC 2025).

Paper architecture (author-confirmed):
  * Spatial 2D CNN: 16 → 32 → 64 on 6 grayscale 64x64 frames.
  * Audio 1D CNN: 16 → 32 on 6 time-aligned MFCC segments (40 coeffs).
  * ~0.06M parameters, ~0.014 GFLOPs.
  * Deployed with quantization-aware training and TFLite conversion.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SpatialCNN2D(nn.Module):
    """3-layer 2D CNN. Six frames are stacked as input channels."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64),
        kernel: int = 3,
        pool: str = "flatten",
        in_ch: int = 6,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Paper specifies a 3-layer 2D CNN.")
        layers: list[nn.Module] = []
        prev = in_ch
        for ch in channels:
            layers.append(ConvBNAct2d(prev, ch, kernel))
            prev = ch
        self.backbone = nn.Sequential(*layers)
        spatial = image_size // (2 ** len(channels))
        if pool == "flatten":
            self.head = nn.Flatten()
            self.feature_dim = channels[-1] * spatial * spatial
        elif pool == "gap":
            self.head = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = channels[-1]
        else:
            raise ValueError(f"Unknown visual pool '{pool}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1, H, W) or (B, T, H, W) or (B, T, H, W) stacked channels."""
        x = self._as_stacked_channels(x)
        return self.head(self.backbone(x)).flatten(1)

    @staticmethod
    def _as_stacked_channels(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            # (B, T, C, H, W) → (B, T*C, H, W); C is 1 for grayscale
            batch, time, channels, height, width = x.shape
            return x.reshape(batch, time * channels, height, width)
        if x.ndim == 4:
            return x
        raise ValueError(f"Expected 4D or 5D visual tensor, got {tuple(x.shape)}")


class AudioCNN1D(nn.Module):
    """2-layer 1D CNN, shared across 6 time-aligned MFCC segments."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32),
        kernel: int = 3,
        in_ch: int = 40,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError("Paper specifies a 2-layer 1D CNN.")
        layers: list[nn.Module] = []
        prev = in_ch
        for ch in channels:
            layers.append(ConvBNAct1d(prev, ch, kernel))
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.AdaptiveAvgPool1d(1)
        self.per_segment_dim = channels[-1]
        self.feature_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, n_mfcc, T) or (B, n_mfcc, T). Returns (B, S*C) or (B, C)."""
        if x.ndim == 4:
            batch, segments, n_mfcc, time = x.shape
            h = self.backbone(x.reshape(batch * segments, n_mfcc, time))
            feat = self.head(h).flatten(1).view(batch, segments * self.per_segment_dim)
            return feat
        if x.ndim == 3:
            return self.head(self.backbone(x)).flatten(1)
        raise ValueError(f"Expected 3D or 4D audio tensor, got {tuple(x.shape)}")


class LightweightSTCNN(nn.Module):
    """Aligned 6-frame / 6-segment audio-visual fusion."""

    def __init__(
        self,
        num_classes: int = 8,
        visual_channels: Sequence[int] = (16, 32, 64),
        visual_kernel: int = 3,
        visual_pool: str = "flatten",
        audio_channels: Sequence[int] = (16, 32),
        audio_kernel: int = 3,
        n_mfcc: int = 40,
        num_frames: int = 6,
        dropout: float = 0.3,
        fusion_hidden: int = 0,
        image_size: int = 64,
        stack_frames_as_channels: bool = True,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.stack_frames_as_channels = stack_frames_as_channels
        visual_in = num_frames if stack_frames_as_channels else 1
        self.visual = SpatialCNN2D(
            channels=visual_channels,
            kernel=visual_kernel,
            pool=visual_pool,
            in_ch=visual_in,
            image_size=image_size,
        )
        self.audio = AudioCNN1D(
            channels=audio_channels,
            kernel=audio_kernel,
            in_ch=n_mfcc,
        )
        audio_out = audio_channels[-1] * num_frames
        fused_dim = self.visual.feature_dim + audio_out
        self.audio_out_dim = audio_out
        head: list[nn.Module] = [nn.Dropout(dropout)]
        if fusion_hidden and fusion_hidden > 0:
            head.extend(
                [
                    nn.Linear(fused_dim, fusion_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_hidden, num_classes),
                ]
            )
        else:
            head.append(nn.Linear(fused_dim, num_classes))
        self.classifier = nn.Sequential(*head)
        self.num_classes = num_classes
        self.fused_dim = fused_dim

    def forward(
        self,
        faces: torch.Tensor,
        mfcc: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_feat = self.visual(faces)
        audio_feat = self.audio(mfcc)
        fused = torch.cat([visual_feat, audio_feat], dim=1)
        logits = self.classifier(fused)
        if return_features:
            return logits, visual_feat, audio_feat
        return logits

    def forward_visual(self, faces: torch.Tensor) -> torch.Tensor:
        visual_feat = self.visual(faces)
        zeros = visual_feat.new_zeros(faces.shape[0], self.audio_out_dim)
        return self.classifier(torch.cat([visual_feat, zeros], dim=1))

    def forward_audio(self, mfcc: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio(mfcc)
        zeros = audio_feat.new_zeros(mfcc.shape[0], self.visual.feature_dim)
        return self.classifier(torch.cat([zeros, audio_feat], dim=1))


def build_model(cfg: dict) -> LightweightSTCNN:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return LightweightSTCNN(
        num_classes=model_cfg["num_classes"],
        visual_channels=tuple(model_cfg["visual_channels"]),
        visual_kernel=model_cfg["visual_kernel"],
        visual_pool=model_cfg["visual_pool"],
        audio_channels=tuple(model_cfg["audio_channels"]),
        audio_kernel=model_cfg["audio_kernel"],
        n_mfcc=data_cfg["n_mfcc"],
        num_frames=data_cfg["num_frames"],
        dropout=model_cfg["dropout"],
        fusion_hidden=model_cfg.get("fusion_hidden", 0),
        image_size=data_cfg["image_size"],
        stack_frames_as_channels=model_cfg.get("stack_frames_as_channels", True),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv_layer_counts(model: LightweightSTCNN) -> tuple[int, int]:
    n2d = sum(isinstance(m, nn.Conv2d) for m in model.visual.modules())
    n1d = sum(isinstance(m, nn.Conv1d) for m in model.audio.modules())
    return n2d, n1d

"""Lightweight Spatio-Temporal CNN (Ding, Tang, Lu, IEEE TAFFC 2025).

Fig. 3:
  * Spatial 2D CNN on one 64×64 grayscale face (valid 3×3, 16→32→64, max-pool 2).
    Flatten 6×6×64 = 2304. Dropout before flatten.
  * Temporal 1D CNN on one 40-d mean-MFCC vector (valid 5×1, 16→32, max-pool 2).
    Flatten 7×32 = 224. Dropout before flatten.
  * Concat → dense 16 (RAVDESS/MEAD) or 14 (SAVEE) → dropout → dense C.
  * Dropout rates: MEAD 0.3, RAVDESS 0.4, SAVEE 0.5.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from lstcnn.constants import fusion_hidden_for, paper_dropout


def _keras_xavier_init(module: nn.Module) -> None:
    """Match Keras glorot_uniform / zero bias (Conv2D, Conv1D, Dense defaults)."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvActPool2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.conv(x)))


class ConvActPool1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.conv(x)))


class SpatialCNN2D(nn.Module):
    """3-layer 2D CNN, Fig. 3 layers 1–8. Input (B, 1, 64, 64) → 2304-d flatten."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64),
        kernel: int = 3,
        in_ch: int = 1,
        dropout: float = 0.4,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Paper specifies a 3-layer 2D CNN.")
        layers: list[nn.Module] = []
        prev = in_ch
        for ch in channels:
            layers.append(ConvActPool2d(prev, ch, kernel))
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, image_size, image_size)
            self.feature_dim = int(self.head(self.backbone(dummy)).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_image(x)
        return self.head(self.drop(self.backbone(x)))

    @staticmethod
    def _as_image(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.unsqueeze(1)
        if x.ndim == 4:
            return x
        if x.ndim == 5:
            batch, time, channels, height, width = x.shape
            if time != 1:
                raise ValueError("Fig. 3 takes one face per forward; got a frame stack.")
            return x.reshape(batch, time * channels, height, width)
        raise ValueError(f"Expected 3D/4D/5D visual tensor, got {tuple(x.shape)}")


class AudioCNN1D(nn.Module):
    """2-layer 1D CNN, Fig. 3 layers 9–14. Input (B, 40) → 224-d flatten."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32),
        kernel: int = 5,
        n_mfcc: int = 40,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError("Paper specifies a 2-layer 1D CNN.")
        layers: list[nn.Module] = []
        prev = 1
        for ch in channels:
            layers.append(ConvActPool1d(prev, ch, kernel))
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mfcc)
            self.feature_dim = int(self.head(self.backbone(dummy)).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_cepstra(x)
        return self.head(self.drop(self.backbone(x)))

    @staticmethod
    def _as_cepstra(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3 and x.shape[1] == 1:
            return x
        raise ValueError(f"Expected (B, 40) or (B, 1, 40) MFCC vector, got {tuple(x.shape)}")


class LightweightSTCNN(nn.Module):
    """Fig. 3: flatten both branches, concat, dense-16/14, dropout, dense-C."""

    def __init__(
        self,
        num_classes: int = 8,
        visual_channels: Sequence[int] = (16, 32, 64),
        visual_kernel: int = 3,
        audio_channels: Sequence[int] = (16, 32),
        audio_kernel: int = 5,
        n_mfcc: int = 40,
        dropout: float = 0.4,
        fusion_hidden: int = 16,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.n_mfcc = n_mfcc
        self.visual = SpatialCNN2D(
            channels=visual_channels,
            kernel=visual_kernel,
            in_ch=1,
            dropout=dropout,
            image_size=image_size,
        )
        self.audio = AudioCNN1D(
            channels=audio_channels,
            kernel=audio_kernel,
            n_mfcc=n_mfcc,
            dropout=dropout,
        )
        fused_dim = self.visual.feature_dim + self.audio.feature_dim
        self.audio_out_dim = self.audio.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )
        self.apply(_keras_xavier_init)
        nn.init.zeros_(self.classifier[-1].bias)
        self.num_classes = num_classes
        self.fused_dim = fused_dim
        self.fusion_hidden = fusion_hidden

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


def _kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.net(x).view(x.size(0), x.size(1), 1, 1)
        return x * weight


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.se(self.bn2(self.conv2(y)))
        return self.act(x + y)


class BoostedSpatialCNN(nn.Module):
    """Same-pad residual+SE visual tower. 64×64 → 4×4×128 = 2048-d."""

    def __init__(self, dropout: float = 0.3, image_size: int = 64) -> None:
        super().__init__()

        def stem(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stage1 = nn.Sequential(stem(1, 32), ResidualBlock(32), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(stem(32, 64), ResidualBlock(64), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(stem(64, 128), ResidualBlock(128), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(stem(128, 128), ResidualBlock(128), nn.MaxPool2d(2))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            feat = self.stage4(self.stage3(self.stage2(self.stage1(dummy))))
            self.feature_dim = int(self.head(feat).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = SpatialCNN2D._as_image(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.head(self.drop(x))


class GatedFusion(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, hidden: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, hidden)
        self.a_proj = nn.Linear(audio_dim, hidden)
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([visual, audio], dim=1))
        fused = self.norm(gate * self.v_proj(visual) + (1.0 - gate) * self.a_proj(audio))
        return self.head(fused)


class BoostedSTCNN(nn.Module):
    """Stronger visual branch + gated fusion. Paper audio CNN is unchanged."""

    is_boosted = True

    def __init__(
        self,
        num_classes: int = 8,
        audio_channels: Sequence[int] = (16, 32),
        audio_kernel: int = 5,
        n_mfcc: int = 40,
        dropout: float = 0.3,
        fusion_hidden: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.n_mfcc = n_mfcc
        self.num_classes = num_classes
        self.fusion_hidden = fusion_hidden
        self.visual = BoostedSpatialCNN(dropout=dropout, image_size=image_size)
        self.audio = AudioCNN1D(
            channels=audio_channels,
            kernel=audio_kernel,
            n_mfcc=n_mfcc,
            dropout=dropout,
        )
        self.audio_out_dim = self.audio.feature_dim
        self.fused_dim = self.visual.feature_dim + self.audio.feature_dim
        self.fusion = GatedFusion(
            self.visual.feature_dim,
            self.audio.feature_dim,
            fusion_hidden,
            num_classes,
            dropout,
        )
        self.apply(_kaiming_init)

    def forward(
        self,
        faces: torch.Tensor,
        mfcc: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_feat = self.visual(faces)
        audio_feat = self.audio(mfcc)
        logits = self.fusion(visual_feat, audio_feat)
        if return_features:
            return logits, visual_feat, audio_feat
        return logits

    def forward_visual(self, faces: torch.Tensor) -> torch.Tensor:
        visual_feat = self.visual(faces)
        zeros = visual_feat.new_zeros(faces.shape[0], self.audio_out_dim)
        return self.fusion(visual_feat, zeros)

    def forward_audio(self, mfcc: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio(mfcc)
        zeros = audio_feat.new_zeros(mfcc.shape[0], self.visual.feature_dim)
        return self.fusion(zeros, audio_feat)


def apply_dataset_hparams(cfg: dict) -> dict:
    """Fill dropout / fusion width from the paper’s per-dataset settings."""
    name = cfg["data"]["dataset"].lower()
    n_cls = cfg["model"]["num_classes"]
    if bool(cfg["model"].get("boost", False)):
        cfg["model"]["dropout"] = float(cfg["model"].get("boost_dropout", 0.3))
        cfg["model"]["fusion_hidden"] = int(cfg["model"].get("boost_fusion", 64))
        return cfg
    cfg["model"]["dropout"] = paper_dropout(name)
    cfg["model"]["fusion_hidden"] = fusion_hidden_for(name, n_cls)
    return cfg


def build_model(cfg: dict) -> LightweightSTCNN | BoostedSTCNN:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    if bool(model_cfg.get("boost", False)):
        return BoostedSTCNN(
            num_classes=model_cfg["num_classes"],
            audio_channels=tuple(model_cfg["audio_channels"]),
            audio_kernel=model_cfg["audio_kernel"],
            n_mfcc=data_cfg["n_mfcc"],
            dropout=model_cfg["dropout"],
            fusion_hidden=int(model_cfg.get("fusion_hidden") or 64),
            image_size=data_cfg["image_size"],
        )
    return LightweightSTCNN(
        num_classes=model_cfg["num_classes"],
        visual_channels=tuple(model_cfg["visual_channels"]),
        visual_kernel=model_cfg["visual_kernel"],
        audio_channels=tuple(model_cfg["audio_channels"]),
        audio_kernel=model_cfg["audio_kernel"],
        n_mfcc=data_cfg["n_mfcc"],
        dropout=model_cfg["dropout"],
        fusion_hidden=model_cfg.get("fusion_hidden") or fusion_hidden_for(data_cfg["dataset"], model_cfg["num_classes"]),
        image_size=data_cfg["image_size"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv_layer_counts(model: LightweightSTCNN) -> tuple[int, int]:
    n2d = sum(isinstance(m, nn.Conv2d) for m in model.visual.modules())
    n1d = sum(isinstance(m, nn.Conv1d) for m in model.audio.modules())
    return n2d, n1d

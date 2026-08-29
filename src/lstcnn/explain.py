"""Integrated Gradients for the visual and audio branches.

Ding et al. use IG to show that facial predictions depend on the eyebrows,
eyes, and mouth, and that audio predictions use all 40 MFCC coefficients.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from lstcnn.preprocess import extract_mfcc_segments, sample_video_frames


def _interpolate(baseline: torch.Tensor, input_tensor: torch.Tensor, steps: int) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, steps, device=input_tensor.device, dtype=input_tensor.dtype)
    shape = [steps] + [1] * (input_tensor.ndim - 1)
    alphas = alphas.view(*shape)
    return baseline + alphas * (input_tensor - baseline)


def integrated_gradients(
    model: nn.Module,
    faces: torch.Tensor,
    mfcc: torch.Tensor,
    target: int | None = None,
    steps: int = 50,
    branch: str = "both",
) -> dict[str, torch.Tensor]:
    """IG attributions for one example (batch size 1).

    `branch`:
      * both   — fused audio-visual model
      * visual — visual path with audio zeroed
      * audio  — audio path with visual zeroed
    """
    model.eval()
    faces = faces.detach()
    mfcc = mfcc.detach()
    face_base = torch.zeros_like(faces)
    mfcc_base = torch.zeros_like(mfcc)

    face_path = _interpolate(face_base, faces, steps).requires_grad_(True)
    mfcc_path = _interpolate(mfcc_base, mfcc, steps).requires_grad_(True)

    if branch == "visual":
        logits = model.forward_visual(face_path)
        inputs = [face_path]
    elif branch == "audio":
        logits = model.forward_audio(mfcc_path)
        inputs = [mfcc_path]
    else:
        logits = model(face_path, mfcc_path)
        inputs = [face_path, mfcc_path]

    if target is None:
        target = int(logits[-1].argmax().item())
    score = logits[:, target].sum()
    grads = torch.autograd.grad(score, inputs)

    out: dict[str, torch.Tensor] = {"target": torch.tensor(target)}
    if branch in {"both", "visual"}:
        avg_grad = grads[0].mean(dim=0, keepdim=True)
        out["face_attr"] = (faces - face_base) * avg_grad
    if branch in {"both", "audio"}:
        idx = 0 if branch == "audio" else 1
        avg_grad = grads[idx].mean(dim=0, keepdim=True)
        out["mfcc_attr"] = (mfcc - mfcc_base) * avg_grad
    return out


def mfcc_coefficient_importance(attr: torch.Tensor) -> np.ndarray:
    """Mean absolute attribution per MFCC coefficient, shape (40,)."""
    squeezed = attr.detach().cpu().numpy()
    arr = np.abs(squeezed)
    feat_axes = [i for i, size in enumerate(arr.shape) if size == 40]
    if not feat_axes:
        raise ValueError(f"No MFCC axis of size 40 in {arr.shape}")
    feat_ax = feat_axes[-1] if arr.ndim >= 3 and arr.shape[-2] == 40 else feat_axes[0]
    reduce = tuple(i for i in range(arr.ndim) if i != feat_ax)
    return arr.mean(axis=reduce)


def face_heatmap(attr: torch.Tensor) -> np.ndarray:
    """Collapse IG over frames/channels into a 2D map in [0, 1]."""
    arr = attr.detach().cpu().numpy()
    while arr.ndim > 2:
        arr = np.abs(arr).mean(axis=0)
    peak = arr.max() + 1e-8
    return (arr / peak).astype(np.float32)


def explain_file(
    model: nn.Module,
    cfg: dict,
    video_path: str | Path,
    audio_path: str | Path | None,
    out_dir: str | Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    data = cfg["data"]
    faces = sample_video_frames(
        video_path,
        num_frames=data["num_frames"],
        image_size=data["image_size"],
        detect_face=data["detect_face"],
    )
    mfcc = extract_mfcc_segments(
        audio_path or video_path,
        num_segments=data["num_audio_segments"],
        sr=data["sample_rate"],
        n_mfcc=data["n_mfcc"],
        n_fft=data["n_fft"],
        hop_length=data["hop_length"],
        frames_per_segment=data["mfcc_frames_per_segment"],
    )
    face_t = torch.from_numpy(faces).unsqueeze(0).to(device)
    mfcc_t = torch.from_numpy(mfcc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(face_t, mfcc_t)
        pred = int(logits.argmax(dim=1).item())

    ig = integrated_gradients(
        model,
        face_t,
        mfcc_t,
        target=pred,
        steps=cfg["explain"]["ig_steps"],
        branch="both",
    )
    heatmap = face_heatmap(ig["face_attr"])
    coeff_imp = mfcc_coefficient_importance(ig["mfcc_attr"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_face_overlay(faces, heatmap, out / "ig_face.png")
    _save_mfcc_bar(coeff_imp, out / "ig_mfcc.png")
    np.save(out / "ig_mfcc_importance.npy", coeff_imp)
    np.save(out / "ig_face_heatmap.npy", heatmap)
    return {"heatmap": heatmap, "mfcc_importance": coeff_imp, "pred": pred}


def _save_face_overlay(faces: np.ndarray, heatmap: np.ndarray, path: Path) -> None:
    frame = faces[len(faces) // 2, 0]
    vis = (frame * 0.5 + 0.5)
    vis = np.clip(vis, 0.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(vis, cmap="gray")
    axes[0].set_title("Input face")
    axes[0].axis("off")
    axes[1].imshow(vis, cmap="gray")
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Integrated Gradients")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_mfcc_bar(importance: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(importance)), importance, color="#3b6d9a")
    ax.set_xlabel("MFCC coefficient")
    ax.set_ylabel("mean |IG|")
    ax.set_title("Audio Integrated Gradients (40 MFCCs)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

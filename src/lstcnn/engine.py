"""Training and evaluation loops."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lstcnn.model import apply_dataset_hparams, build_model

from lstcnn.cache import warm_feature_cache
from lstcnn.constants import DATASET_EMOTIONS
from lstcnn.data import (
    AudioVisualDataset,
    SyntheticAVDataset,
    mfcc_train_stats,
    paper_face_noise_var,
    paper_time_stretch,
    scan_dataset,
    split_samples,
)
from lstcnn.flops import PAPER_GFLOPS, PAPER_PARAMS_M, print_model_complexity
from lstcnn.preprocess import face_tool_status, require_ffmpeg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    return {
        "faces": torch.stack([b["faces"] for b in batch]),
        "mfcc": torch.stack([b["mfcc"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "speaker": [b["speaker"] for b in batch],
        "emotion": [b["emotion"] for b in batch],
        "video_path": [b["video_path"] for b in batch],
    }


def _loader_kwargs(train_cfg: dict, shuffle: bool) -> dict:
    workers = int(train_cfg.get("num_workers", 0))
    kwargs: dict = {
        "batch_size": train_cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return kwargs


def build_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    name = data_cfg["dataset"].lower()
    if name == "synthetic":
        apply_dataset_hparams(cfg)
        loaders = {}
        split_sizes = (("train", 256, 0), ("val", 64, 1), ("test", 64, 2))
        for split, size, offset in split_sizes:
            dataset = SyntheticAVDataset(size, cfg, seed=cfg["seed"] + offset)
            loaders[split] = DataLoader(
                dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=split == "train",
                num_workers=0,
                collate_fn=collate,
            )
        return loaders

    samples = scan_dataset(name, Path(data_cfg["root"]), speech_only=data_cfg.get("speech_only", True))
    if not samples:
        raise FileNotFoundError(
            f"No {name} samples under {data_cfg['root']}. "
            "Download the corpus and point data.root at it, or use dataset=synthetic."
        )
    n_wav = sum(Path(s.audio_path).suffix.lower() == ".wav" for s in samples)
    print(
        f"Found {len(samples)} {name} clips "
        f"({n_wav} with .wav audio, {len(samples) - n_wav} from video soundtrack)"
    )
    emotions = DATASET_EMOTIONS[name]
    cfg["model"]["num_classes"] = len(emotions)
    apply_dataset_hparams(cfg)
    num_parts = int(data_cfg.get("num_frames", 6))
    stretch = paper_time_stretch(name, data_cfg)
    noise_var = paper_face_noise_var(name, data_cfg)
    splits = split_samples(
        samples,
        mode=train_cfg["split"],
        val_ratio=train_cfg["val_ratio"],
        test_ratio=train_cfg["test_ratio"],
        seed=cfg["seed"],
        num_parts=num_parts,
    )
    n_windows = sum(len(part) for part in splits.values())
    print(
        f"Windows train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}  (clips={len(samples)}, parts={num_parts}, "
        f"split={train_cfg['split']}, total={n_windows})"
    )
    tools = face_tool_status()
    print(
        f"Face tools: Haar={'yes' if tools['haar'] else 'NO'}  "
        f"dlib68={'yes' if tools['dlib68'] else 'NO (pip install dlib; 68-point .dat in assets/)'}"
    )
    if stretch is not None or noise_var > 0.0 or bool(data_cfg.get("trim_silence", True)):
        print(
            f"{name.upper()} paper transforms: "
            f"time_stretch={stretch} (train only, full clip)  "
            f"trim_silence={bool(data_cfg.get('trim_silence', True))} "
            f"top_db={data_cfg.get('trim_top_db', 30)}  "
            f"face_noise_std={noise_var} (Keras GaussianNoise, train only, resampled)  "
            f"align_face={bool(data_cfg.get('align_face', False))}"
        )
    cache_dir = data_cfg.get("cache_dir")
    cache_path = Path(cache_dir) / name if cache_dir else None
    if cache_path:
        workers = int(train_cfg.get("num_workers", 4))
        warm_feature_cache(samples, data_cfg, cache_path, None, workers=workers)
        if stretch is not None:
            warm_feature_cache(samples, data_cfg, cache_path, stretch, workers=workers)

    mfcc_mean, mfcc_std = mfcc_train_stats(splits["train"], data_cfg, cache_path, None)
    cfg["data"]["mfcc_mean"] = mfcc_mean.tolist()
    cfg["data"]["mfcc_std"] = mfcc_std.tolist()
    print(
        f"MFCC train-set scale: coeff0 mean={float(mfcc_mean[0]):.1f} std={float(mfcc_std[0]):.1f} "
        f"(applied to all splits; not per-window z-score)"
    )

    loaders: dict[str, DataLoader] = {}
    for split, subset in splits.items():
        dataset = AudioVisualDataset(
            subset,
            cfg,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            augment=split == "train",
        )
        loaders[split] = DataLoader(dataset, **_loader_kwargs(train_cfg, split == "train"))
    return loaders


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        for ema_param, param in zip(self.ema.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buf, buf in zip(self.ema.buffers(), model.buffers()):
            ema_buf.copy_(buf)


def _predict_logits(model: nn.Module, faces: torch.Tensor, mfcc: torch.Tensor, tta: bool) -> torch.Tensor:
    logits = model(faces, mfcc)
    if not tta:
        return logits
    flipped = torch.flip(faces, dims=[-1])
    return 0.5 * (logits + model(flipped, mfcc))


def _mixup_batch(
    faces: torch.Tensor,
    mfcc: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(faces.size(0), device=faces.device)
    mixed_faces = lam * faces + (1.0 - lam) * faces[index]
    mixed_mfcc = lam * mfcc + (1.0 - lam) * mfcc[index]
    return mixed_faces, mixed_mfcc, target, target[index], lam


def _class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor | None:
    items = getattr(dataset, "items", None)
    if not items:
        return None
    counts = np.bincount([item.label for item in items], minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
    mixup: float = 0.0,
    ema: ModelEma | None = None,
    tta: bool = False,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            faces = batch["faces"].to(device)
            mfcc = batch["mfcc"].to(device)
            target = batch["label"].to(device)
            if train and mixup > 0.0:
                faces, mfcc, y_a, y_b, lam = _mixup_batch(faces, mfcc, target, mixup)
                logits = model(faces, mfcc)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = _predict_logits(model, faces, mfcc, tta=tta and not train)
                loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            losses.append(float(loss.item()))
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels.extend(target.detach().cpu().tolist())
    acc = accuracy_score(labels, preds) if labels else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc)


def _clip_majority_accuracy(preds: list[int], labels: list[int], videos: list[str]) -> float | None:
    if not videos or len(videos) != len(preds):
        return None
    buckets: dict[str, list[tuple[int, int]]] = {}
    for pred, label, video in zip(preds, labels, videos):
        buckets.setdefault(video, []).append((pred, label))
    correct = 0
    for items in buckets.values():
        votes = [pred for pred, _ in items]
        truth = items[0][1]
        majority = max(set(votes), key=votes.count)
        correct += int(majority == truth)
    return correct / len(buckets) if buckets else 0.0


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    tta: bool = False,
) -> dict:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    videos: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = _predict_logits(
                model,
                batch["faces"].to(device),
                batch["mfcc"].to(device),
                tta=tta,
            )
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["label"].tolist())
            videos.extend(batch.get("video_path", []))
    used = sorted(set(labels) | set(preds))
    names = [class_names[i] for i in used if i < len(class_names)]
    report = classification_report(labels, preds, labels=used, target_names=names, zero_division=0)
    clip_acc = _clip_majority_accuracy(preds, labels, videos)
    return {
        "accuracy": float(accuracy_score(labels, preds)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0,
        "clip_accuracy": clip_acc,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
        "report": report,
        "preds": preds,
        "labels": labels,
    }


def evaluate_ensemble(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    tta: bool = False,
) -> dict:
    for model in models:
        model.eval()
    preds: list[int] = []
    labels: list[int] = []
    videos: list[str] = []
    with torch.no_grad():
        for batch in loader:
            faces = batch["faces"].to(device)
            mfcc = batch["mfcc"].to(device)
            logits = None
            for model in models:
                part = _predict_logits(model, faces, mfcc, tta=tta)
                logits = part if logits is None else logits + part
            assert logits is not None
            logits = logits / len(models)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["label"].tolist())
            videos.extend(batch.get("video_path", []))
    used = sorted(set(labels) | set(preds))
    names = [class_names[i] for i in used if i < len(class_names)]
    report = classification_report(labels, preds, labels=used, target_names=names, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0,
        "clip_accuracy": _clip_majority_accuracy(preds, labels, videos),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
        "report": report,
        "preds": preds,
        "labels": labels,
    }


def train_model(cfg: dict, device: torch.device | None = None) -> Path:
    set_seed(cfg["seed"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = cfg["data"]["dataset"].lower()
    if name in DATASET_EMOTIONS:
        cfg["model"]["num_classes"] = len(DATASET_EMOTIONS[name])
    apply_dataset_hparams(cfg)
    train_cfg = cfg["train"]
    model_seed = int(train_cfg.get("model_seed", cfg["seed"]))
    set_seed(model_seed)
    model = build_model(cfg).to(device)
    complexity = print_model_complexity(
        model,
        image_size=int(cfg["data"]["image_size"]),
        n_mfcc=int(cfg["data"]["n_mfcc"]),
    )
    n_params = int(complexity["n_params"])
    gflops = float(complexity["gflops"])
    boost = bool(cfg["model"].get("boost", False))
    mixup = float(train_cfg.get("mixup", 0.0))
    tta = bool(train_cfg.get("tta", False))
    extras = ["boost"] if boost else []
    if mixup:
        extras.append(f"mixup={mixup}")
    if float(train_cfg.get("ema", 0.0) or 0.0) > 0.0:
        extras.append(f"ema={train_cfg.get('ema')}")
    if tta:
        extras.append("tta")
    extras.append(f"model_seed={model_seed}")
    print(f"Device: {device}  |  {', '.join(extras)}")
    if name != "synthetic":
        require_ffmpeg()
    loaders = build_dataloaders(cfg)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    weight_tensor = None
    if bool(train_cfg.get("class_weight", False)):
        weight_tensor = _class_weights(loaders["train"].dataset, cfg["model"]["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    opt_kwargs = dict(
        lr=train_cfg["lr"],
        betas=(0.9, 0.999),
        eps=float(train_cfg.get("adam_eps", 1e-7)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    optimizer = (
        torch.optim.AdamW(model.parameters(), **opt_kwargs)
        if opt_name == "adamw"
        else torch.optim.Adam(model.parameters(), **opt_kwargs)
    )
    scheduler = None
    if str(train_cfg.get("lr_schedule", "")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("cosine_tmax", 160)),
            eta_min=1e-6,
        )
    ema = None
    ema_decay = float(train_cfg.get("ema", 0.0) or 0.0)
    if ema_decay > 0.0:
        ema = ModelEma(model, decay=ema_decay)

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_loss = float("inf")
    best_acc = -1.0
    stale = 0
    history: list[dict] = []
    min_delta = float(train_cfg.get("min_delta", 0.001))
    patience = int(train_cfg.get("patience", 30))
    monitor = str(train_cfg.get("monitor", "val_loss")).lower()

    for epoch in range(1, train_cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            f"train {epoch}",
            mixup=mixup,
            ema=ema,
        )
        eval_model = ema.ema if ema is not None else model
        va_loss, va_acc = _run_epoch(
            eval_model,
            loaders["val"],
            criterion,
            None,
            device,
            f"val {epoch}",
            tta=tta,
        )
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        print(
            f"epoch {epoch:03d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )
        if monitor == "val_acc":
            improved = va_acc > best_acc + min_delta
        else:
            improved = best_loss - va_loss >= min_delta
        if improved:
            best_loss = min(best_loss, va_loss)
            best_acc = max(best_acc, va_acc)
            stale = 0
            torch.save(
                {
                    "model": eval_model.state_dict(),
                    "config": cfg,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "epoch": epoch,
                    "n_params": n_params,
                    "gflops": gflops,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"Early stop at epoch {epoch} "
                    f"(no {monitor} gain ≥ {min_delta} for {patience} epochs; "
                    f"best val_acc {best_acc:.4f}, val_loss {best_loss:.4f})"
                )
                break

    ckpt = torch_load(best_path, device)
    model.load_state_dict(ckpt["model"])
    class_names = DATASET_EMOTIONS.get(cfg["data"]["dataset"].lower(), [str(i) for i in range(cfg["model"]["num_classes"])])
    test_metrics = evaluate_loader(model, loaders["test"], device, class_names, tta=tta)
    clip_acc = test_metrics.get("clip_accuracy")
    extra_clip = f"  clip-majority={clip_acc:.4f}" if clip_acc is not None else ""
    print(f"test window={test_metrics['accuracy']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}{extra_clip}")
    vis_acc = aud_acc = None
    if hasattr(model, "forward_visual"):
        vis_ok = aud_ok = n = 0
        model.eval()
        with torch.no_grad():
            for batch in loaders["test"]:
                faces = batch["faces"].to(device)
                mfcc = batch["mfcc"].to(device)
                y = batch["label"].to(device)
                vis_ok += int((model.forward_visual(faces).argmax(1) == y).sum().item())
                aud_ok += int((model.forward_audio(mfcc).argmax(1) == y).sum().item())
                n += int(y.numel())
        vis_acc = vis_ok / n if n else 0.0
        aud_acc = aud_ok / n if n else 0.0
        print(
            f"test unimodal  visual*={vis_acc:.4f} (paper spatial 0.9203)  "
            f"audio*={aud_acc:.4f} (paper temporal 0.4585)  n={n}"
        )
    summary = {
        "n_params": n_params,
        "params_m": n_params / 1e6,
        "gflops": gflops,
        "paper_complexity": {"params_m": PAPER_PARAMS_M, "gflops": PAPER_GFLOPS},
        "best_val_loss": best_loss,
        "best_val_acc": ckpt.get("val_acc"),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_clip_accuracy": test_metrics.get("clip_accuracy"),
        "test_visual_zero_audio": vis_acc,
        "test_audio_zero_visual": aud_acc,
        "paper_reference": {
            "SAVEE": 0.9757,
            "RAVDESS": 0.9589,
            "MEAD": 0.9857,
        },
        "split": train_cfg["split"],
        "history": history,
        "classification_report": test_metrics["report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "class_names": class_names,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(test_metrics["report"], encoding="utf-8")
    print(test_metrics["report"])
    print(f"Saved checkpoint to {best_path}")
    return best_path


def torch_load(path: str | Path, map_location) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch_load(path, device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

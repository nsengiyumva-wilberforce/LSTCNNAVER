"""Training and evaluation loops."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lstcnn.cache import warm_feature_cache
from lstcnn.constants import DATASET_EMOTIONS
from lstcnn.data import AudioVisualDataset, SyntheticAVDataset, scan_dataset, split_samples
from lstcnn.flops import PAPER_GFLOPS, PAPER_PARAMS_M, count_macs, gflops_from_macs
from lstcnn.preprocess import require_ffmpeg


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
    splits = split_samples(
        samples,
        mode=train_cfg["split"],
        val_ratio=train_cfg["val_ratio"],
        test_ratio=train_cfg["test_ratio"],
        seed=cfg["seed"],
    )
    print(
        f"Clips  train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}  (×{data_cfg.get('num_frames', 6)} windows)"
    )
    cache_dir = data_cfg.get("cache_dir")
    if cache_dir:
        cache_path = Path(cache_dir) / name
        workers = int(train_cfg.get("num_workers", 4))
        clips = splits["train"] + splits["val"] + splits["test"]
        warm_feature_cache(clips, data_cfg, cache_path, None, workers=workers)

    loaders: dict[str, DataLoader] = {}
    for split, subset in splits.items():
        dataset = AudioVisualDataset(subset, cfg, augment=split == "train")
        loaders[split] = DataLoader(dataset, **_loader_kwargs(train_cfg, split == "train"))
    return loaders


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
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
            logits = model(faces, mfcc)
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels.extend(target.detach().cpu().tolist())
    acc = accuracy_score(labels, preds) if labels else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["faces"].to(device), batch["mfcc"].to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["label"].tolist())
    used = sorted(set(labels) | set(preds))
    names = [class_names[i] for i in used if i < len(class_names)]
    report = classification_report(labels, preds, labels=used, target_names=names, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
        "report": report,
        "preds": preds,
        "labels": labels,
    }


def train_model(cfg: dict, device: torch.device | None = None) -> Path:
    set_seed(cfg["seed"])
    if cfg["data"]["dataset"].lower() != "synthetic":
        require_ffmpeg()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apply_dataset_hparams(cfg)
    loaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    train_cfg = cfg["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(0.9, 0.999),
        eps=float(train_cfg.get("adam_eps", 1e-7)),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_loss = float("inf")
    stale = 0
    history: list[dict] = []
    min_delta = float(train_cfg.get("min_delta", 0.001))
    patience = int(train_cfg.get("patience", 30))

    n_params = count_parameters(model)
    macs = count_macs(
        model,
        image_size=cfg["data"]["image_size"],
        n_mfcc=cfg["data"]["n_mfcc"],
    )
    gflops = gflops_from_macs(macs)
    print(
        f"Device: {device}  |  parameters: {n_params:,} ({n_params / 1e6:.3f}M, paper {PAPER_PARAMS_M}M)  "
        f"|  {gflops:.4f} GFLOPs (paper {PAPER_GFLOPS})"
    )

    for epoch in range(1, train_cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, f"train {epoch}")
        va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, None, device, f"val {epoch}")
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        print(
            f"epoch {epoch:03d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )
        if best_loss - va_loss >= min_delta:
            best_loss = va_loss
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "val_loss": best_loss,
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
                    f"(Δval_loss < {min_delta} for {patience} epochs; "
                    f"restored epoch {epoch - patience}, val_loss {best_loss:.4f})"
                )
                break

    ckpt = torch_load(best_path, device)
    model.load_state_dict(ckpt["model"])
    class_names = DATASET_EMOTIONS.get(cfg["data"]["dataset"].lower(), [str(i) for i in range(cfg["model"]["num_classes"])])
    test_metrics = evaluate_loader(model, loaders["test"], device, class_names)
    summary = {
        "n_params": n_params,
        "params_m": n_params / 1e6,
        "gflops": gflops,
        "paper_complexity": {"params_m": PAPER_PARAMS_M, "gflops": PAPER_GFLOPS},
        "best_val_loss": best_loss,
        "best_val_acc": ckpt.get("val_acc"),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "paper_reference": {
            "SAVEE": 0.9757,
            "RAVDESS": 0.9589,
            "MEAD": 0.9857,
        },
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

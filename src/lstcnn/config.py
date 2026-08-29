"""YAML config helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    return cfg


def merge_cli(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge dotted CLI overrides such as train.epochs=10."""
    merged = dict(cfg)
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        cursor = merged
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return merged

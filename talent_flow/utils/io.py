#!/usr/bin/env python3
"""Disk I/O helpers (numpy/json/pickle, with directory auto-creation)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_npz(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_project_root_to_path() -> None:
    """Добавляет корень проекта в sys.path"""
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_project_path(path: Path | str | None) -> Path | None:
    """Делает путь абсолютным относительно проекта"""
    if path is None:
        return None

    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def to_project_relative_path(path: Path | str | None) -> str | None:
    """Возвращает путь относительно проекта"""
    if path is None:
        return None

    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def save_json(data: Any, path: Path) -> None:
    """Сохраняет JSON в UTF-8"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    """Читает JSON или возвращает default"""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Фиксирует seed для повторяемых запусков"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_checkpoint(
    *,
    model: torch.nn.Module,
    model_name: str,
    epoch: int,
    best_metric: float,
    metric_name: str = "macro_f1",
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint_path: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает checkpoint в общем формате"""
    checkpoint: dict[str, Any] = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": float(best_metric),
        "metric_name": metric_name,
    }
    if checkpoint_path is not None:
        checkpoint["checkpoint_path"] = to_project_relative_path(checkpoint_path)
    if metric_name == "macro_f1":
        checkpoint["macro_f1"] = float(best_metric)
        checkpoint["best_macro_f1"] = float(best_metric)
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        checkpoint.update(extra)
    return checkpoint

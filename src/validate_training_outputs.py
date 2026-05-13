from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import torch

from src.training_helpers import PROJECT_ROOT, resolve_project_path


CLASSIFIER_METRIC_KEYS = {"best_macro_f1", "checkpoint"}
COMMON_METRIC_KEYS = {"best_metric", "metric_name", "checkpoint"}
CHECKPOINT_KEYS = {
    "model_name",
    "model_state_dict",
    "epoch",
    "best_metric",
    "metric_name",
}


def parse_args() -> argparse.Namespace:
    """Читает настройки проверки из командной строки"""
    parser = argparse.ArgumentParser(description="Check model checkpoints and metric files")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--allow-empty-checkpoints", action="store_true")
    return parser.parse_args()


def is_absolute_path(value: str) -> bool:
    """Проверяет unix и windows пути"""
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    """Читает json-файл с метриками"""
    return json.loads(path.read_text(encoding="utf-8"))


def find_path_errors(data: Any, source: Path, errors: list[str], name: str = "") -> None:
    """Ищет абсолютные пути внутри json"""
    if isinstance(data, dict):
        for key, value in data.items():
            child_name = f"{name}.{key}" if name else key
            find_path_errors(value, source, errors, child_name)
        return

    if isinstance(data, list):
        for index, value in enumerate(data):
            find_path_errors(value, source, errors, f"{name}[{index}]")
        return

    if isinstance(data, str) and is_absolute_path(data):
        # В метриках должны лежать только относительные пути
        rel_source = source.relative_to(PROJECT_ROOT)
        errors.append(f"{rel_source}: absolute path in {name or '<root>'}: {data}")


def validate_metrics(path: Path, errors: list[str]) -> None:
    """Проверяет один файл с метриками"""
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        # Битый json
        errors.append(f"{path.relative_to(PROJECT_ROOT)}: bad json: {exc}")
        return

    find_path_errors(data, path, errors)

    if path.name.endswith("_metrics.json") and isinstance(data, dict):
        # Новые metrics-файлы могут хранить любую главную метрику
        if "metric_name" in data:
            expected_keys = COMMON_METRIC_KEYS
        else:
            expected_keys = CLASSIFIER_METRIC_KEYS

        missing = sorted(expected_keys - data.keys())
        if missing:
            errors.append(f"{path.relative_to(PROJECT_ROOT)}: missing keys {missing}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Загружает checkpoint с учетом разных версий torch"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(path: Path, errors: list[str]) -> None:
    """Проверяет один checkpoint модели"""
    rel_path = path.relative_to(PROJECT_ROOT)
    try:
        checkpoint = load_checkpoint(path)
    except Exception as exc:
        # Если checkpoint не читается, остальные проверки не имеют смысла
        errors.append(f"{rel_path}: cannot load checkpoint: {exc}")
        return

    if not isinstance(checkpoint, dict):
        # Общий формат checkpoint у нас всегда dict
        errors.append(f"{rel_path}: checkpoint is not a dict")
        return

    # Базовые ключи нужны всем моделям
    missing = sorted(CHECKPOINT_KEYS - checkpoint.keys())
    if missing:
        errors.append(f"{rel_path}: missing keys {missing}")

    # Для macro_f1 храним и общий best_metric, и старое понятное имя
    metric_name = checkpoint.get("metric_name")
    if metric_name == "macro_f1":
        for key in ("macro_f1", "best_macro_f1"):
            if key not in checkpoint:
                errors.append(f"{rel_path}: missing key {key}")

    checkpoint_path = checkpoint.get("checkpoint_path")
    if isinstance(checkpoint_path, str) and is_absolute_path(checkpoint_path):
        # Абсолютный путь в checkpoint ломает перенос проекта на другом устройстве
        errors.append(f"{rel_path}: checkpoint_path is absolute: {checkpoint_path}")

    if "model_state_dict" in checkpoint and not isinstance(checkpoint["model_state_dict"], dict):
        # Веса модели должны лежать отдельным словарем
        errors.append(f"{rel_path}: model_state_dict is not a dict")


def collect_checkpoints(checkpoints_dir: Path, extra_paths: list[Path]) -> list[Path]:
    """Собирает checkpoint-файлы из папки и аргументов"""
    paths: list[Path] = []
    if checkpoints_dir.exists():
        paths.extend(sorted(checkpoints_dir.rglob("*.pt")))
        paths.extend(sorted(checkpoints_dir.rglob("*.pth")))

    for path in extra_paths:
        full_path = resolve_project_path(path)
        if full_path is not None:
            paths.append(full_path)

    return sorted(set(paths))


def main() -> int:
    """Запускает все проверки и печатает результат"""
    args = parse_args()
    metrics_dir = resolve_project_path(args.metrics_dir)
    checkpoints_dir = resolve_project_path(args.checkpoints_dir)
    errors: list[str] = []

    # проверяем json-метрики
    if metrics_dir is None or not metrics_dir.exists():
        errors.append(f"metrics dir not found: {args.metrics_dir}")
    else:
        for path in sorted(metrics_dir.rglob("*.json")):
            validate_metrics(path, errors)

    # проверяем сохраненные веса моделей
    checkpoints = collect_checkpoints(checkpoints_dir or PROJECT_ROOT / "outputs/models", args.checkpoint)
    if not checkpoints and not args.allow_empty_checkpoints:
        errors.append("no checkpoints found, use --allow-empty-checkpoints for a dry check")

    for path in checkpoints:
        validate_checkpoint(path, errors)

    if errors:
        print("Found problems:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Training outputs look ok")
    print(f"metrics_dir={metrics_dir.relative_to(PROJECT_ROOT) if metrics_dir else args.metrics_dir}")
    print(f"checkpoints={len(checkpoints)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

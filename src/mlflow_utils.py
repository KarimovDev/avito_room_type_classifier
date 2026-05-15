from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from src.training_helpers import PROJECT_ROOT, to_project_relative_path


EXPERIMENT_NAME = "room_type_classifier"
REPO_OWNER = "YashinSergey"
REPO_NAME = "room_type_classifier"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"


def _load_mlflow():
    """Импортирует MLflow только во время обучения"""
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _prepare_param(value: Any) -> str | int | float | bool | None:
    """Приводит параметры к простым значениям для MLflow"""
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return to_project_relative_path(value)
    return str(value)


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str | int | float | bool | None]:
    """Разворачивает вложенные параметры в плоский словарь"""
    flat: dict[str, str | int | float | bool | None] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, name))
        elif isinstance(value, list | tuple):
            flat[name] = str(value)
        else:
            flat[name] = _prepare_param(value)
    return flat


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Оставляет только числовые метрики"""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def log_mlflow_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Логирует числовые метрики текущей эпохи или финального результата"""
    mlflow = _load_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return

    values = numeric_metrics(metrics)
    if values:
        mlflow.log_metrics(values, step=step)


def log_mlflow_params(params: dict[str, Any]) -> None:
    """Логирует дополнительные параметры после обучения"""
    mlflow = _load_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return

    mlflow.log_params(flatten_params(params))


def log_mlflow_artifacts(paths: list[Path | str | None]) -> None:
    """Логирует файлы с метриками и checkpoint как artifacts"""
    mlflow = _load_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return

    for path in paths:
        if path is None:
            continue
        file_path = Path(path)
        if file_path.is_file():
            mlflow.log_artifact(str(file_path))


def setup_mlflow_tracking(mlflow: Any) -> None:
    """Настраивает DagsHub или локальный MLflow"""
    if os.getenv("RTC_MLFLOW_LOCAL") == "1":
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
        client = mlflow.tracking.MlflowClient()
        if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
            client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACTS_DIR.as_uri())
        mlflow.set_experiment(EXPERIMENT_NAME)
        return

    try:
        import dagshub
    except ImportError as exc:
        raise RuntimeError("DagsHub не установлен. Выполните `just install-tracking`") from exc

    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_mlflow_run(model_name: str, run_name: str, params: dict[str, Any]) -> Any:
    """Стартует MLflow run для train-скрипта"""
    mlflow = _load_mlflow()
    if mlflow is None:
        print("MLflow не установлен, логирование пропущено")
        return None

    setup_mlflow_tracking(mlflow)

    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag("model", model_name)
    mlflow.log_params(flatten_params(params))
    return run


def end_mlflow_run() -> None:
    """Завершает активный MLflow run"""
    mlflow = _load_mlflow()
    if mlflow is not None and mlflow.active_run() is not None:
        mlflow.end_run()

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.mlflow_utils import EXPERIMENT_NAME, setup_mlflow_tracking
from src.training_helpers import PROJECT_ROOT, resolve_project_path, to_project_relative_path


KNOWN_MODELS = {
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet",
    "convnext_nano",
    "convnext_tiny",
    "yolo",
}

FIELDNAMES = [
    "model",
    "variant",
    "run_name",
    "metric_name",
    "best_metric",
    "best_macro_f1",
    "best_accuracy",
    "best_train_loss",
    "best_val_loss",
    "best_epoch",
    "checkpoint",
    "mlflow_run_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model comparison table from MLflow")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "model_comparison.csv",
        help="Куда сохранить CSV с таблицей сравнения",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Сохранить все запуски, а не только лучший запуск каждой модели",
    )
    return parser.parse_args()


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow не установлен. Выполните `just install-tracking`") from exc
    return mlflow


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_epoch(value: float | None) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return str(value)


def _metric(run: Any, *names: str) -> float | None:
    for name in names:
        value = _as_float(run.data.metrics.get(name))
        if value is not None:
            return value
    return None


def _param(run: Any, *names: str) -> str:
    for name in names:
        value = run.data.params.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _model_name(run: Any) -> str:
    return _param(run, "model", "model_name") or run.data.tags.get("model", "")


def _row_from_run(run: Any) -> dict[str, str]:
    model = _model_name(run)
    return {
        "model": model,
        "variant": _param(run, "variant"),
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "metric_name": _param(run, "metric_name") or "macro_f1",
        "best_metric": _format_metric(_metric(run, "best_metric", "best_macro_f1", "macro_f1")),
        "best_macro_f1": _format_metric(_metric(run, "best_macro_f1", "macro_f1")),
        "best_accuracy": _format_metric(_metric(run, "best_accuracy", "accuracy")),
        "best_train_loss": _format_metric(_metric(run, "best_train_loss")),
        "best_val_loss": _format_metric(_metric(run, "best_val_loss", "val_loss")),
        "best_epoch": _format_epoch(_metric(run, "best_epoch")) or _param(run, "best_epoch"),
        "checkpoint": _param(run, "checkpoint"),
        "mlflow_run_id": run.info.run_id,
    }


def _sort_key(row: dict[str, str]) -> tuple[float, str]:
    value = _as_float(row["best_macro_f1"]) or _as_float(row["best_metric"])
    return (value if value is not None else -1.0, row["model"])


def _group_key(row: dict[str, str]) -> str:
    model = row["model"]
    variant = row["variant"]
    if variant:
        return f"{model}:{variant}"
    return model


def _best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _group_key(row)
        if key not in best or _sort_key(row) > _sort_key(best[key]):
            best[key] = row
    return list(best.values())


def load_best_rows() -> list[dict[str, str]]:
    return load_rows(all_runs=False)


def load_rows(all_runs: bool) -> list[dict[str, str]]:
    mlflow = _load_mlflow()
    setup_mlflow_tracking(mlflow)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )
    rows = []
    for run in runs:
        model = _model_name(run)
        if model not in KNOWN_MODELS:
            continue
        rows.append(_row_from_run(run))

    if not all_runs:
        rows = _best_rows(rows)
    return sorted(rows, key=_sort_key, reverse=True)


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_path = resolve_project_path(args.output)
    if output_path is None:
        raise ValueError("--output не должен быть пустым")

    rows = load_rows(all_runs=args.all_runs)
    save_csv(rows, output_path)

    print(f"rows={len(rows)}")
    print(f"output={to_project_relative_path(output_path)}")
    if not rows:
        print("В MLflow пока нет запусков обучающих моделей")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

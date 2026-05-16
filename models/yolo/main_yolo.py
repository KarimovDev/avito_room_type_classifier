from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from ultralyticsplus import YOLO, postprocess_classify_output

from src.mlflow_utils import end_mlflow_run, log_mlflow_artifacts, log_mlflow_metrics, log_mlflow_params, start_mlflow_run
from src.training_helpers import resolve_project_path, save_json, to_project_relative_path


YOLO_REPO_ID = "keremberke/yolov8m-scene-classification"
YOLO_FILENAME = "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO scene classifier")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/val_images"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/yolo/downloads/keremberke/yolov8m-scene-classification/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/yolo/artifacts"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics/yolo"))
    parser.add_argument("--project-checkpoint", type=Path, default=Path("outputs/models/yolo/yolo_best.pt"))
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def download_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        print(f"YOLO checkpoint найден: {checkpoint_path}")
        return checkpoint_path

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    downloaded_path = hf_hub_download(
        repo_id=YOLO_REPO_ID,
        filename=YOLO_FILENAME,
        local_dir=checkpoint_path.parent,
        local_dir_use_symlinks=False,
        token=hf_token,
    )
    print(f"YOLO checkpoint скачан: {downloaded_path}")
    return Path(downloaded_path)


def get_image_paths(images_dir: Path, max_images: int) -> list[Path]:
    image_paths = [
        path
        for path in sorted(images_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    return image_paths[:max_images]


def run_predictions(model: YOLO, image_paths: list[Path]) -> tuple[list[dict], float]:
    """Top-2 predictions for a small folder."""
    start = time.perf_counter()
    rows: list[dict] = []

    for image_path in image_paths:
        results = model.predict(image_path)
        processed_result = postprocess_classify_output(model, result=results[0])
        top_items = sorted(processed_result.items(), key=lambda item: item[1], reverse=True)[:2]

        rows.append(
            {
                "image": to_project_relative_path(image_path),
                "top_predictions": [
                    {"class_name": class_name, "confidence": float(confidence)}
                    for class_name, confidence in top_items
                ],
            }
        )

    return rows, time.perf_counter() - start


def write_log(predictions: list[dict], log_path: Path) -> None:
    lines: list[str] = []
    for prediction in predictions:
        lines.append(f"\nФайл: {prediction['image']}")
        for item in prediction["top_predictions"]:
            lines.append(f"\t{item['class_name']}: {item['confidence']:.3f}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metrics(
    *,
    predictions: list[dict],
    metrics_path: Path,
    checkpoint_path: Path,
    load_time: float,
    predict_time: float,
    confidence: float,
) -> None:
    top1_confidences = [
        item["top_predictions"][0]["confidence"]
        for item in predictions
        if item["top_predictions"]
    ]
    avg_confidence = sum(top1_confidences) / len(top1_confidences) if top1_confidences else 0.0

    metrics = {
        "model": "yolo",
        "model_name": "yolov8m-scene-classification",
        "metric_name": "avg_top1_confidence",
        "best_metric": avg_confidence,
        "checkpoint": to_project_relative_path(checkpoint_path),
        "num_images": len(predictions),
        "confidence_threshold": confidence,
        "load_time_sec": round(load_time, 4),
        "predict_time_sec": round(predict_time, 4),
        "avg_time_per_image_sec": round(predict_time / len(predictions), 4) if predictions else None,
        "predictions": predictions,
    }
    save_json(metrics, metrics_path)


def save_project_checkpoint(
    *,
    model: YOLO,
    checkpoint_path: Path,
    source_checkpoint: Path,
    best_metric: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "yolov8m-scene-classification",
            "model_state_dict": model.model.state_dict(),
            "epoch": 0,
            "best_metric": float(best_metric),
            "metric_name": "avg_top1_confidence",
            "checkpoint_path": to_project_relative_path(checkpoint_path),
            "source_checkpoint": to_project_relative_path(source_checkpoint),
        },
        checkpoint_path,
    )


def main() -> int:
    args = parse_args()
    images_dir = resolve_project_path(args.images_dir)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir)
    metrics_dir = resolve_project_path(args.metrics_dir)
    project_checkpoint = resolve_project_path(args.project_checkpoint)

    if images_dir is None or not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {args.images_dir}")
    if checkpoint_path is None or output_dir is None or metrics_dir is None or project_checkpoint is None:
        raise ValueError("paths are not configured")

    # inference run, not training
    run_name = args.run_name or f"yolo_inference_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    start_mlflow_run(
        "yolo",
        run_name,
        {
            "model": "yolo",
            "model_name": "yolov8m-scene-classification",
            "metric_name": "avg_top1_confidence",
            "images_dir": to_project_relative_path(images_dir),
            "max_images": args.max_images,
            "confidence": args.confidence,
        },
    )

    start = time.perf_counter()
    checkpoint_path = download_checkpoint(checkpoint_path)
    model = YOLO(checkpoint_path)
    model.overrides["conf"] = args.confidence
    load_time = time.perf_counter() - start

    image_paths = get_image_paths(images_dir, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"images not found: {args.images_dir}")

    predictions, predict_time = run_predictions(model, image_paths)
    top1_confidences = [
        item["top_predictions"][0]["confidence"]
        for item in predictions
        if item["top_predictions"]
    ]
    avg_confidence = sum(top1_confidences) / len(top1_confidences) if top1_confidences else 0.0

    log_path = output_dir / "log.txt"
    metrics_path = metrics_dir / "yolo_metrics.json"
    write_log(predictions, log_path)
    save_project_checkpoint(
        model=model,
        checkpoint_path=project_checkpoint,
        source_checkpoint=checkpoint_path,
        best_metric=avg_confidence,
    )
    save_metrics(
        predictions=predictions,
        metrics_path=metrics_path,
        checkpoint_path=project_checkpoint,
        load_time=load_time,
        predict_time=predict_time,
        confidence=args.confidence,
    )
    log_mlflow_metrics(
        {
            "best_metric": avg_confidence,
            "avg_top1_confidence": avg_confidence,
            "num_images": len(predictions),
            "load_time_sec": load_time,
            "predict_time_sec": predict_time,
        }
    )
    log_mlflow_params(
        {
            "best_epoch": 0,
            "checkpoint": to_project_relative_path(project_checkpoint),
            "metrics_json": to_project_relative_path(metrics_path),
        }
    )
    log_mlflow_artifacts([metrics_path, log_path, project_checkpoint])
    end_mlflow_run()

    print(f"YOLO checkpoint: {to_project_relative_path(checkpoint_path)}")
    print(f"Изображений: {len(image_paths)}")
    print(f"Загрузка модели: {load_time:.2f}с")
    print(f"Предсказания: {predict_time:.2f}с")
    print(f"Лог: {to_project_relative_path(log_path)}")
    print(f"Метрики: {to_project_relative_path(metrics_path)}")
    print(f"Project checkpoint: {to_project_relative_path(project_checkpoint)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

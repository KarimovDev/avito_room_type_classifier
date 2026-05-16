from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from src.dataloaders import create_test_dataloader
from src.device import get_default_device
from src.evaluate_ensemble import build_model, get_model_name, load_checkpoint, normalize_weights
from src.training_helpers import to_project_relative_path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = [
    ROOT_DIR / "outputs" / "models" / "convnext_nano" / "convnext_nano_best.pt",
    ROOT_DIR / "outputs" / "models" / "resnet50" / "resnet50_best.pt",
    ROOT_DIR / "outputs" / "models" / "resnet18" / "resnet18_best.pt",
]
DEFAULT_WEIGHTS = [0.34349677767909004, 0.3383500954431325, 0.3181531268777774]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сделать submission ансамблем моделей")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[path for path in DEFAULT_CHECKPOINTS if path.exists()],
        help="Пути к checkpoint-файлам моделей.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="Веса моделей в том же порядке, что и --checkpoints.",
    )
    parser.add_argument("--test-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "test_df.csv")
    parser.add_argument("--test-images", type=Path, default=ROOT_DIR / "data" / "raw" / "test_images")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data" / "submissions" / "submission_ensemble.csv")
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="Опциональный CSV с confidence по каждой строке. По умолчанию не сохраняется.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT_DIR / "reports" / "metrics" / "ensemble" / "submission_summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download-timeout", type=int, default=30)
    return parser.parse_args()


def take_batch_item(batch_field, index: int):
    if isinstance(batch_field, torch.Tensor):
        return batch_field[index].item()
    if isinstance(batch_field, (list, tuple)):
        return batch_field[index]
    return batch_field[index]


def get_expected_image_path(test_images_dir: Path, image_id: str) -> Path:
    return test_images_dir / f"{image_id}.jpg"


def get_missing_image_ids(test_df: pd.DataFrame, test_images_dir: Path) -> list[str]:
    missing_image_ids = []

    for image_id in test_df["image_id_ext"].astype(str):
        if not get_expected_image_path(test_images_dir, image_id).exists():
            missing_image_ids.append(image_id)

    return missing_image_ids


def download_missing_images(test_df: pd.DataFrame, test_images_dir: Path, timeout: int) -> int:
    if "image" not in test_df.columns:
        raise ValueError("В test_csv нет колонки 'image' для скачивания недостающих картинок")

    test_images_dir.mkdir(parents=True, exist_ok=True)
    downloaded_count = 0

    for _, row in test_df.iterrows():
        image_id = str(row["image_id_ext"])
        output_path = get_expected_image_path(test_images_dir, image_id)

        if output_path.exists():
            continue

        image_url = row.get("image")
        if pd.isna(image_url) or not str(image_url).strip():
            raise ValueError(f"Нет локальной картинки и нет url в колонке image для image_id_ext={image_id}")

        response = requests.get(str(image_url).strip(), timeout=timeout)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        downloaded_count += 1

    return downloaded_count


def prepare_test_images(test_df: pd.DataFrame, test_images_dir: Path, timeout: int) -> int:
    missing_before = get_missing_image_ids(test_df, test_images_dir)
    if not missing_before:
        return 0

    downloaded_count = download_missing_images(test_df, test_images_dir, timeout)
    missing_after = get_missing_image_ids(test_df, test_images_dir)

    if missing_after:
        sample = ", ".join(missing_after[:20])
        more_count = len(missing_after) - 20
        suffix = f" и еще {more_count}" if more_count > 0 else ""
        raise FileNotFoundError("После скачивания все еще не найдены картинки: " + sample + suffix)

    return downloaded_count


@torch.inference_mode()
def predict_model(checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> tuple[dict, np.ndarray, list[str]]:
    checkpoint = load_checkpoint(checkpoint_path)
    model_name = get_model_name(checkpoint_path, checkpoint)
    num_classes = int(checkpoint["num_classes"])
    image_size = int(checkpoint.get("image_size", 224))

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader = create_test_dataloader(
        test_csv_path=str(args.test_csv),
        test_image_root=str(args.test_images),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        filter_can_predict=False,
    )

    probs = []
    image_ids = []
    for images, batch_image_ids, _item_ids in loader:
        images = images.to(device)
        batch_probs = model(images).softmax(dim=1).cpu().numpy()
        probs.append(batch_probs)
        for index in range(batch_probs.shape[0]):
            image_ids.append(str(take_batch_item(batch_image_ids, index)))

    model_info = {
        "name": model_name,
        "checkpoint": to_project_relative_path(checkpoint_path),
        "image_size": image_size,
        "num_classes": num_classes,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_info, np.concatenate(probs), image_ids


def write_submission(path: Path, image_ids: list[str], preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id_ext", "Predicted"])
        writer.writeheader()
        for image_id, pred in zip(image_ids, preds, strict=True):
            writer.writerow({"image_id_ext": image_id, "Predicted": int(pred)})


def write_details(path: Path, image_ids: list[str], preds: np.ndarray, confidence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id_ext", "Predicted", "confidence"])
        writer.writeheader()
        for image_id, pred, conf in zip(image_ids, preds, confidence, strict=True):
            writer.writerow(
                {
                    "image_id_ext": image_id,
                    "Predicted": int(pred),
                    "confidence": float(conf),
                }
            )


def main() -> int:
    args = parse_args()
    if not args.checkpoints:
        raise ValueError("Не найдено ни одного checkpoint-файла")
    if len(args.weights) != len(args.checkpoints):
        raise ValueError("Количество --weights должно совпадать с количеством --checkpoints")

    missing_checkpoints = [str(path) for path in args.checkpoints if not path.exists()]
    if missing_checkpoints:
        raise FileNotFoundError("Не найдены checkpoint-файлы: " + ", ".join(missing_checkpoints))

    test_df = pd.read_csv(args.test_csv)
    full_image_ids = test_df["image_id_ext"].astype(str).tolist()
    downloaded_images = prepare_test_images(test_df, args.test_images, args.download_timeout)

    weights = normalize_weights(args.weights)
    device = get_default_device()
    model_infos = []
    all_probs = []
    base_image_ids = None

    for checkpoint_path in args.checkpoints:
        model_info, probs, image_ids = predict_model(checkpoint_path, args, device)
        model_infos.append(model_info)
        all_probs.append(probs)

        if base_image_ids is None:
            base_image_ids = image_ids
        elif base_image_ids != image_ids:
            raise ValueError("Порядок test-объектов отличается между моделями")

    ensemble_probs = np.zeros_like(all_probs[0], dtype=float)
    for weight, probs in zip(weights, all_probs, strict=True):
        ensemble_probs += weight * probs

    preds = ensemble_probs.argmax(axis=1)
    confidence = ensemble_probs.max(axis=1)
    image_ids = base_image_ids or []

    if image_ids != full_image_ids:
        missing_predictions = sorted(set(full_image_ids) - set(image_ids))
        if missing_predictions:
            raise ValueError("Не получены предсказания для test-строк: " + ", ".join(missing_predictions[:20]))
        raise ValueError("Порядок test-строк в submission не совпал с test_csv")

    write_submission(args.output, full_image_ids, preds)
    if args.details_output is not None:
        write_details(args.details_output, full_image_ids, preds, confidence)

    summary = {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "submission": to_project_relative_path(args.output),
        "test_csv": to_project_relative_path(args.test_csv),
        "test_images": to_project_relative_path(args.test_images),
        "num_rows": len(full_image_ids),
        "model_predicted_rows": len(image_ids),
        "downloaded_images": downloaded_images,
        "weights": weights,
        "models": model_infos,
        "confidence_mean": float(confidence.mean()),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
    }
    if args.details_output is not None:
        summary["details"] = to_project_relative_path(args.details_output)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Submission: {to_project_relative_path(args.output)}")
    if args.details_output is not None:
        print(f"Details: {to_project_relative_path(args.details_output)}")
    print(f"Summary: {to_project_relative_path(args.summary_output)}")
    print(f"Rows: {len(full_image_ids)}")
    print(f"Downloaded images: {downloaded_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

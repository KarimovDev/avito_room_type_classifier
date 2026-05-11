"""Инференс ConvNeXt-Tiny по чекпоинту train_convnext_tiny (processed test CSV + raw test_images)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.convnext_tiny.model import build_convnext_tiny
from src.dataloaders import create_test_dataloader
from src.device import get_default_device
from src.labels import load_label_mapping


def _take_batch_field(batch, j):
    """Достаёт j-й элемент из батча (tensor, list или tuple) после collate DataLoader."""
    if isinstance(batch, torch.Tensor):
        return batch[j].item()
    if isinstance(batch, (list, tuple)):
        return batch[j]
    return batch[j]


def _load_ckpt(path: Path, map_location: str | torch.device) -> dict:
    """Загружает dict чекпоинта; учитывает разные версии torch.load (weights_only)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args() -> argparse.Namespace:
    """Парсит аргументы CLI для инференса (чекпоинт, test CSV, выходной CSV, top-k)."""
    p = argparse.ArgumentParser(description="Predict with ConvNeXt-Tiny checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True, help="convnext_tiny_best.pt")
    p.add_argument(
        "--model-name",
        type=str,
        default="convnext_tiny",
        help="Имя модели: отчёт инференса в reports/metrics/<model-name>/",
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="По умолчанию: reports/metrics/<model-name>/",
    )
    p.add_argument("--test-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "test_df.csv")
    p.add_argument("--test-images", type=Path, default=ROOT_DIR / "data" / "raw" / "test_images")
    p.add_argument("--output", type=Path, default=ROOT_DIR / "outputs" / "convnext_tiny_test_predictions.csv")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    """Загружает модель, прогоняет test_dataloader, сохраняет предсказания в CSV и JSON-отчёт в reports/metrics."""
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model_name = args.model_name
    metrics_dir = (args.metrics_dir or (ROOT_DIR / "reports" / "metrics" / model_name)).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    infer_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = get_default_device()
    ckpt = _load_ckpt(args.checkpoint, map_location="cpu")
    num_classes = int(ckpt["num_classes"])
    image_size = int(ckpt.get("image_size", 224))

    model = build_convnext_tiny(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader = create_test_dataloader(
        test_csv_path=str(args.test_csv),
        test_image_root=str(args.test_images),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    label_map = load_label_mapping()
    rows: list[dict] = []
    k = min(max(1, args.top_k), num_classes)

    for images, image_ids, item_ids in loader:
        images = images.to(device)
        logits = model(images)
        prob = logits.softmax(dim=1)
        vals, idx = prob.topk(k, dim=1)
        for j in range(prob.shape[0]):
            iid = _take_batch_field(image_ids, j)
            it = _take_batch_field(item_ids, j)
            pred0 = int(idx[j, 0].item())
            row = {
                "image_id_ext": iid,
                "item_id": it,
                "pred": pred0,
                "confidence": float(vals[j, 0].item()),
                "label": label_map.get(pred0, str(pred0)),
            }
            for t in range(1, k):
                pid = int(idx[j, t].item())
                row[f"pred_{t + 1}"] = pid
                row[f"prob_{t + 1}"] = float(vals[j, t].item())
                row[f"label_{t + 1}"] = label_map.get(pid, str(pid))
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    summary = {
        "run_id": infer_run_id,
        "model_name": model_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_csv": str(args.test_csv.resolve()),
        "test_images": str(args.test_images.resolve()),
        "predictions_csv": str(args.output.resolve()),
        "num_rows": len(rows),
        "top_k": args.top_k,
        "num_classes": num_classes,
        "image_size": image_size,
    }
    summary_path = metrics_dir / f"{model_name}_inference_{infer_run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"Inference report: {summary_path}")


if __name__ == "__main__":
    main()

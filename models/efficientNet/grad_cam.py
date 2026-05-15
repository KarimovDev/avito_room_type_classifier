from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.models import efficientnet_b0, efficientnet_b1


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.labels import load_label_mapping
from src.device import get_default_device
from src.transforms import get_val_transforms


MODEL_BUILDERS = {
    "b0": efficientnet_b0,
    "b1": efficientnet_b1,
}


def parse_args() -> argparse.Namespace:
    """Читаем параметры запуска GradCAM из командной строки."""
    parser = argparse.ArgumentParser(description="Build GradCAM visualization for EfficientNet")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "models" / "efficientNet" / "artifacts" / "efficientnet_b1_best.pt",
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "models" / "efficientNet" / "artifacts" / "grad_cam",
    )
    return parser.parse_args()

def get_default_image_size(variant: str) -> int:
    """Размер картинки по умолчанию для варианта EfficientNet."""
    return 240 if variant == "b1" else 224


def build_model(variant: str, num_classes: int) -> torch.nn.Module:
    """Собираем EfficientNet и подменяем последний слой под число классов."""
    model = MODEL_BUILDERS.get(variant, efficientnet_b0)(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def get_sample_from_split(split: str, sample_index: int) -> tuple[Path, int | None]:
    """Берём пример из train/val по индексу.

    В CSV берём `image_id_ext` и строим путь: `{split}_images/{image_id_ext}.jpg`.
    Если файла нет на диске - выкидываем строку из выборки
    """
    csv_path = ROOT_DIR / "data" / "raw" / f"{split}_df.csv"
    image_root = ROOT_DIR / "data" / "raw" / f"{split}_images"
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_id_ext"].astype(str).map(lambda image_id: image_root / f"{image_id}.jpg")
    df = df[df["image_path"].map(lambda path: path.exists())].reset_index(drop=True)
    if sample_index >= len(df):
        raise IndexError(f"--sample-index={sample_index} is out of range for {split} split with {len(df)} images")

    row = df.iloc[sample_index]
    return Path(row["image_path"]), int(row["result"])


def load_rgb_image(image_path: Path) -> Image.Image:
    """Открываем изображение и приводим к RGB (с учётом EXIF-ориентации)."""
    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert("RGB")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем чекпоинт и восстанавливаем модель
    device = get_default_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    variant = checkpoint.get("variant", "b0")
    num_classes = int(checkpoint.get("num_classes", 20))
    image_size = int(checkpoint.get("image_size", get_default_image_size(variant)))

    model = build_model(variant, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Выбираем изображение
    image_path = args.image
    true_class = None
    if image_path is None:
        image_path, true_class = get_sample_from_split(args.split, args.sample_index)

    # Препроцессинг как на валидации, чтобы картинка была в правильном формате для модели
    image = load_rgb_image(image_path)
    transform = get_val_transforms(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Считаем предсказание модели
    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_probability, predicted_class = torch.max(probabilities, dim=0)

    # Выбираем на какой класс строим GradCAM
    target_class = int(args.target_class if args.target_class is not None else predicted_class.item())

    # Подключаем библиотеку GradCAM и выбираем слой, по которому строим карту
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Обычно берут последний сверточный блок (он показывает высокоуровневые признаки)
    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(target_class)]

    # Строим GradCAM, получаем heatmap
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Накладываем heatmap на исходное изображение (в виде цветной подсветки)
    rgb_image = np.asarray(image.resize((image_size, image_size))).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    # Загружаем названия классов
    labels = load_label_mapping()
    stem = f"{image_path.stem}_{variant}_target_{target_class}"
    output_path = args.output_dir / f"{stem}_grad_cam.jpg"
    metadata_path = args.output_dir / f"{stem}_metadata.json"

    # Сохраняем изображение с наложенным GradCAM
    Image.fromarray(visualization).save(output_path)
    # Сохраняем метаданные предсказания
    metadata = {
        "image": str(image_path),
        "checkpoint": str(args.checkpoint),
        "variant": variant,
        "image_size": image_size,
        "true_class": true_class,
        "true_label": labels.get(true_class) if true_class is not None else None,
        "predicted_class": int(predicted_class.item()),
        "predicted_label": labels.get(int(predicted_class.item())),
        "predicted_probability": float(predicted_probability.item()),
        "target_class": target_class,
        "target_label": labels.get(target_class),
        "output": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"GradCAM saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(
        f"predicted={metadata['predicted_label'] or metadata['predicted_class']} "
        f"probability={metadata['predicted_probability']:.4f}"
    )


if __name__ == "__main__":
    main()

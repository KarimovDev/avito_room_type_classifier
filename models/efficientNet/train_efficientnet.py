from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    efficientnet_b0,
    efficientnet_b1,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.metrics import calculate_macro_f1, calculate_per_class_f1
from src.training_helpers import build_checkpoint, to_project_relative_path


MODEL_BUILDERS = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
}


def parse_args() -> argparse.Namespace:
    """Читает параметры запуска из командной строки"""
    parser = argparse.ArgumentParser(description="Train EfficientNet baseline")
    parser.add_argument("--variant", choices=MODEL_BUILDERS.keys(), default="b0")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "train_df.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "val_df.csv")
    parser.add_argument("--train-images", type=Path, default=ROOT_DIR / "data" / "raw" / "train_images")
    parser.add_argument("--val-images", type=Path, default=ROOT_DIR / "data" / "raw" / "val_images")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "models" / "efficientnet")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT_DIR / "reports" / "metrics" / "efficientnet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-col", default="result")
    parser.add_argument("--class-balance", choices=["loss", "none"], default="loss")
    parser.add_argument(
        "--use-weighted-sampling",
        action="store_true",
        help="Использовать WeightedRandomSampler для балансировки train DataLoader",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Печатать прогресс обучения каждые N батчей (0 = выключено).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Сколько эпох без улучшения macro-F1 на val до остановки (0 = выключено).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Минимальный прирост macro-F1, чтобы считать эпоху улучшением",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="none",
        help="После эпохи: none или ReduceLROnPlateau по val_loss",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=3,
        help="Параметр patience у ReduceLROnPlateau (эпох без снижения val_loss).",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.1,
        help="Множитель lr при срабатывании ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--plateau-min-lr",
        type=float,
        default=1e-7,
        help="Нижняя граница lr для ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Не сохранять веса модели, оставить только JSON с метриками.",
    )
    return parser.parse_args()


def build_model(variant: str, num_classes: int) -> nn.Module:
    """Собирает EfficientNet под нужное число классов"""
    builder, weights = MODEL_BUILDERS[variant]
    # Берем предобученные веса ImageNet
    model = builder(weights=weights)
    # Меняем последний Linear слой
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_class_weights(csv_path: Path, target_col: str, num_classes: int, device: torch.device) -> torch.Tensor:
    """Считает веса классов для CrossEntropyLoss"""
    targets = pd.read_csv(csv_path)[target_col].astype(int)
    max_target = int(targets.max())
    if max_target >= num_classes:
        raise ValueError(
            f"Found target={max_target}, but num_classes={num_classes}. "
            "Increase --num-classes or check class indexing."
        )

    counts = torch.bincount(torch.tensor(targets.to_list()), minlength=num_classes).float()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    existing_classes = counts > 0
    weights[existing_classes] = counts.sum() / (existing_classes.sum() * counts[existing_classes])
    return weights.to(device)


def validate_paths(args: argparse.Namespace) -> None:
    """Проверяем, что входные файлы/папки действительно существуют."""
    paths = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--train-images": args.train_images,
        "--val-images": args.val_images,
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        joined_paths = "\n".join(missing)
        raise FileNotFoundError(f"Missing input paths:\n{joined_paths}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    log_every: int = 0,
) -> float:
    """Обучает модель одну эпоху"""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        # Переносим батч на нужное устройство (CPU/CUDA/MPS).
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Усредняем loss по всем картинкам
        total_loss += loss.item() * images.size(0)

        if log_every and batch_idx % log_every == 0:
            print(f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[dict[str, object]]]:
    """Считает loss и macro-F1 на validation"""
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        # Самый вероятный класс.
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        # Собираем ответы для метрик
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    # Для macro-F1 используем общую функцию из src
    macro_f1 = calculate_macro_f1(y_true, y_pred)
    per_class_f1 = calculate_per_class_f1(y_true, y_pred, num_classes)
    return total_loss / len(loader.dataset), macro_f1, per_class_f1


def add_label_names(
    per_class_f1: list[dict[str, object]],
    label_mapping: dict[int, str],
) -> list[dict[str, object]]:
    """Добавляет название класса к per-class метрикам"""
    return [
        {
            **item,
            "label": label_mapping.get(int(item["class_id"]), str(item["class_id"])),
        }
        for item in per_class_f1
    ]


def print_per_class_f1(per_class_f1: list[dict[str, object]]) -> None:
    """Печатает классы от худшего F1 к лучшему"""
    print("per_class_f1:")
    for item in sorted(per_class_f1, key=lambda row: (float(row["f1"]), int(row["class_id"]))):
        print(
            f"  class={item['class_id']:>2} "
            f"f1={float(item['f1']):.4f} "
            f"support={item['support']:>4} "
            f"label={item['label']}"
        )


def save_comparison_row(metrics_path: Path, row: dict[str, object]) -> None:
    """Добавляет строку в CSV со сравнениями"""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metrics_path.exists()
    fieldnames = ["model", "variant", "num_classes", "best_epoch", "best_macro_f1", "checkpoint"]

    with metrics_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    # Проверяем входные файлы и папки
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Выбираем устройство через общий helper
    device = get_default_device()
    print(f"Using device: {device}")
    label_mapping = load_label_mapping()
    idx_to_class = {str(class_id): label for class_id, label in label_mapping.items()}
    
    # Готовим общий даталоадер (читает CSV и берёт картинки из папок).
    train_loader, val_loader = create_dataloaders(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        train_image_root=args.train_images,
        val_image_root=args.val_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_weighted_sampling=args.use_weighted_sampling,
    )

    # Собираем модель и переносим её на нужное устройство.
    model = build_model(args.variant, args.num_classes).to(device)
    class_weights = None
    if args.class_balance == "loss":
        # Веса классов учитываются прямо в функции потерь.
        class_weights = get_class_weights(args.train_csv, args.target_col, args.num_classes, device)
        print(f"class_weights={class_weights.cpu().tolist()}")

    # Loss для многоклассовой классификации
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # AdamW с weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    # Храним лучший результат по macro-F1
    best_macro_f1 = -1.0
    best_epoch = 0
    best_per_class_f1: list[dict[str, object]] = []
    checkpoint_path = args.output_dir / f"efficientnet_{args.variant}_best.pt"
    history = []
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        # Обучение и затем проверка на валидации.
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            log_every=args.log_every,
        )
        val_loss, macro_f1, per_class_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
        )
        per_class_f1 = add_label_names(per_class_f1, label_mapping)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "macro_f1": macro_f1,
                "per_class_f1": per_class_f1,
                "lr": current_lr,
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"lr={current_lr:.2e}"
        )

        improved = macro_f1 > best_macro_f1 + args.early_stopping_min_delta
        if improved:
            # Если стало лучше, обновляем best
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_per_class_f1 = per_class_f1
            epochs_without_improvement = 0
            if not args.no_save_checkpoint:
                torch.save(
                    build_checkpoint(
                        model=model,
                        model_name="efficientnet",
                        epoch=best_epoch,
                        best_metric=best_macro_f1,
                        optimizer=optimizer,
                        checkpoint_path=checkpoint_path,
                        extra={
                            "variant": args.variant,
                            "num_classes": args.num_classes,
                            "image_size": args.image_size,
                            "use_weighted_sampling": args.use_weighted_sampling,
                            "per_class_f1": best_per_class_f1,
                            "idx_to_class": idx_to_class,
                        },
                    ),
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"Ранняя остановка, т.к. нет улучшения macro-F1 > {best_macro_f1:.4f} "
                f"на {args.early_stopping_patience} эпох (min_delta={args.early_stopping_min_delta})"
            )
            break

    # Сохраняем метрики и историю обучения
    metrics = {
        "model": "efficientnet",
        "variant": args.variant,
        "num_classes": args.num_classes,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_per_class_f1": best_per_class_f1,
        "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(checkpoint_path),
        "history": history,
        "stop_reason": stop_reason,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "class_balance": args.class_balance,
            "use_weighted_sampling": args.use_weighted_sampling,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "lr_scheduler": args.lr_scheduler,
            "plateau_patience": args.plateau_patience,
            "plateau_factor": args.plateau_factor,
            "plateau_min_lr": args.plateau_min_lr,
            "save_checkpoint": not args.no_save_checkpoint,
        },
    }
    metrics_path = args.metrics_dir / f"efficientnet_{args.variant}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Добавляем короткую строку в CSV для сравнения запусков
    comparison_path = args.metrics_dir / "model_comparison.csv"
    save_comparison_row(
        comparison_path,
        {
            "model": "efficientnet",
            "variant": args.variant,
            "num_classes": args.num_classes,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(checkpoint_path),
        },
    )

    print(f"best_macro_f1={best_macro_f1:.4f}")
    print_per_class_f1(best_per_class_f1)
    if args.no_save_checkpoint:
        print("checkpoint=не сохранялся")
    else:
        print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.resnet18.resnet18 import build_resnet18
from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1


def parse_args() -> argparse.Namespace:
    """Читает параметры запуска обучения из командной строки

    Returns:
        Namespace с путями, гиперпараметрами и настройками модели
    """
    parser = argparse.ArgumentParser(description="Train ResNet18 on room type dataset")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимого обучения.")
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "train_df.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "val_df.csv")
    parser.add_argument("--train-images", type=Path, default=ROOT_DIR / "data" / "raw" / "train_images")
    parser.add_argument("--val-images", type=Path, default=ROOT_DIR / "data" / "raw" / "val_images")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "models" / "resnet18")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT_DIR / "reports" / "metrics" / "resnet18")
    parser.add_argument("--no-pretrained", action="store_true", help="Не использовать веса ImageNet.")
    parser.add_argument("--no-class-weights", action="store_true", help="Отключить веса классов в loss.")
    parser.add_argument(
        "--no-weighted-sampling",
        action="store_true",
        help="Отключить WeightedRandomSampler для train DataLoader.",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Не сохранять веса модели, оставить только JSON с F1-метриками.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Сколько эпох ждать улучшения macro-F1 перед остановкой. 0 = не останавливать.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Минимальный прирост macro-F1, который считается улучшением.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Фиксирует основные источники случайности для повторяемых экспериментов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_paths(args: argparse.Namespace) -> None:
    """Проверяет, что входные CSV и папки с изображениями существуют

    Args:
        args: Параметры запуска из parse_args()
    """
    paths = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--train-images": args.train_images,
        "--val-images": args.val_images,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены входные файлы/папки:\n" + "\n".join(missing))


def get_class_weights(csv_path: Path, num_classes: int, device: torch.device) -> torch.Tensor:
    """Считает веса классов для CrossEntropyLoss

    Args:
        csv_path: Путь к processed train CSV
        num_classes: Количество классов
        device: Устройство, на котором будет считаться loss

    Returns:
        Tensor весов классов на выбранном устройстве
    """
    targets = pd.read_csv(csv_path)["result"].astype(int)
    counts = torch.bincount(torch.tensor(targets.to_list()), minlength=num_classes).float()

    # Редкие классы получают больший вес, а отсутствующие классы не участвуют в loss
    weights = torch.zeros(num_classes, dtype=torch.float32)
    existing_classes = counts > 0
    weights[existing_classes] = counts.sum() / (existing_classes.sum() * counts[existing_classes])
    return weights.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Обучает модель одну эпоху

    Args:
        model: ResNet18
        loader: Train DataLoader
        criterion: Функция потерь
        optimizer: Оптимизатор
        device: CPU/CUDA/MPS

    Returns:
        Средний train loss за эпоху
    """
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        # Переносим данные на тот же device, где находится модель
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float, list[dict[str, object]]]:
    """Проверяет модель на validation и считает F1-метрики

    Args:
        model: ResNet18
        loader: Validation DataLoader
        criterion: Функция потерь
        device: CPU/CUDA/MPS
        num_classes: Количество классов

    Returns:
        Средний validation loss, accuracy, macro-F1 и F1 по каждому классу
    """
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    accuracy = float(calculate_accuracy(y_true, y_pred))
    macro_f1 = float(calculate_macro_f1(y_true, y_pred))
    per_class_f1 = calculate_per_class_f1(y_true, y_pred, num_classes)
    return total_loss / len(loader.dataset), accuracy, macro_f1, per_class_f1


def add_label_names(per_class_f1: list[dict[str, object]]) -> list[dict[str, object]]:
    """Добавляет текстовые названия классов к per-class F1

    Args:
        per_class_f1: Список метрик по классам

    Returns:
        Такой же список, но с полем label
    """
    label_mapping = load_label_mapping()
    return [
        {
            **item,
            "label": label_mapping.get(int(item["class_id"]), str(item["class_id"])),
        }
        for item in per_class_f1
    ]


def save_metrics_report(metrics: dict[str, object], metrics_dir: Path) -> tuple[Path, Path]:
    """Сохраняет полный отчет последнего запуска и общий список экспериментов

    Args:
        metrics: Полный отчет по обучению
        metrics_dir: Папка для метрик конкретной модели

    Returns:
        Пути к JSON последнего запуска и JSON со списком экспериментов
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(metrics.get("run_id") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    metrics_with_run = {
        "run_id": run_id,
        **metrics,
    }

    metrics_path = metrics_dir / "resnet18_metrics.json"
    experiments_path = metrics_dir / "resnet18_experiments.json"
    metrics_path.write_text(json.dumps(metrics_with_run, indent=2, ensure_ascii=False), encoding="utf-8")

    hyperparameters = metrics["hyperparameters"]
    best_metrics = metrics["best_metrics"]
    experiment = {
        "run_id": run_id,
        "model": metrics["model"],
        "best_epoch": metrics["best_epoch"],
        "best_macro_f1": metrics["best_macro_f1"],
        "best_accuracy": best_metrics.get("accuracy"),
        "best_train_loss": metrics["best_train_loss"],
        "best_val_loss": metrics["best_val_loss"],
        "stop_reason": metrics["stop_reason"],
        "checkpoint": metrics["checkpoint"],
        "epochs": hyperparameters["epochs"],
        "batch_size": hyperparameters["batch_size"],
        "image_size": hyperparameters["image_size"],
        "learning_rate": hyperparameters["learning_rate"],
        "weight_decay": hyperparameters["weight_decay"],
        "seed": hyperparameters["seed"],
        "pretrained": hyperparameters["pretrained"],
        "class_weights": hyperparameters["class_weights"],
        "weighted_sampling": hyperparameters["weighted_sampling"],
        "early_stopping_patience": hyperparameters["early_stopping_patience"],
        "early_stopping_min_delta": hyperparameters["early_stopping_min_delta"],
        "metrics_json": str(metrics_path),
    }

    if experiments_path.exists():
        experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    else:
        experiments = []
    experiments.append(experiment)
    experiments_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics_path, experiments_path


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs должен быть >= 1")

    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    set_seed(args.seed)

    # Используем общий выбор устройства проекта: CUDA -> MPS -> CPU
    device = get_default_device()
    print(f"Using device: {device}")

    # DataLoader берется из общего data-pipeline, чтобы ResNet18 училась на processed CSV
    train_loader, val_loader = create_dataloaders(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        train_image_root=args.train_images,
        val_image_root=args.val_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_weighted_sampling=not args.no_weighted_sampling,
        seed=args.seed,
    )

    model = build_resnet18(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = get_class_weights(args.train_csv, args.num_classes, device)

    # CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    best_train_loss = None
    best_val_loss = None
    history: list[dict[str, object]] = []
    checkpoint_path = args.output_dir / f"resnet18_{run_id}_best.pt"
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, macro_f1, per_class_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
        )
        per_class_f1 = add_label_names(per_class_f1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class_f1": per_class_f1,
            }
        )

        improved = macro_f1 > best_macro_f1 + args.early_stopping_min_delta
        if improved:
            # Сохраняем только лучший чекпоинт, чтобы не плодить большие файлы
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_metrics = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class_f1": per_class_f1,
            }
            if not args.no_save_checkpoint:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": args.num_classes,
                        "image_size": args.image_size,
                        "epoch": best_epoch,
                        "macro_f1": best_macro_f1,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"accuracy={accuracy:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"best_macro_f1={best_macro_f1:.4f} "
            f"best_epoch={best_epoch} "
            f"no_improve={epochs_without_improvement}"
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"early stopping: macro_f1 не улучшался {args.early_stopping_patience} эпох, "
                f"best_macro_f1={best_macro_f1:.4f}"
            )
            break

    metrics = {
        "run_id": run_id,
        "model": "resnet18",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "checkpoint": None if args.no_save_checkpoint else str(checkpoint_path),
        "stop_reason": stop_reason,
        "history": history,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
            "class_weights": not args.no_class_weights,
            "weighted_sampling": not args.no_weighted_sampling,
            "save_checkpoint": not args.no_save_checkpoint,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
    }
    metrics_path, experiments_path = save_metrics_report(metrics, args.metrics_dir)

    print(f"best_macro_f1={best_macro_f1:.4f}")
    if args.no_save_checkpoint:
        print("checkpoint=не сохранялся")
    else:
        print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    print(f"experiments={experiments_path}")


if __name__ == "__main__":
    main()

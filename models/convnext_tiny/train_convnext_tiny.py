"""Обучение ConvNeXt-Tiny: те же метрики и отчёты, что у models/resnet18/train_resnet18.py."""

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
from torch import amp, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.convnext_tiny.model import build_convnext_tiny
from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1


def parse_args() -> argparse.Namespace:
    """Парсит аргументы CLI для обучения (пути к данным, гиперпараметры, early stopping)."""
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny on room type dataset")
    parser.add_argument("--num-classes", type=int, default=20, help="Число классов (метки 0..N-1 в CSV result).")
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
    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext_tiny",
        help="Имя модели: каталог metrics = reports/metrics/<model-name>/ (как resnet18).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="По умолчанию: outputs/models/<model-name>/",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="По умолчанию: reports/metrics/<model-name>/",
    )
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
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Mixed precision на CUDA (обычно быстрее и меньше VRAM).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Не показывать tqdm по батчам (как раньше: лог только после эпохи).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Фиксирует seed для random, numpy и torch (включая CUDA при наличии)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_paths(args: argparse.Namespace) -> None:
    """Проверяет существование train/val CSV и каталогов с изображениями."""
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
    """Считает веса классов для CrossEntropyLoss по частотам в train CSV; проверяет согласованность с num_classes."""
    targets = pd.read_csv(csv_path)["result"].astype(int)
    t = torch.tensor(targets.to_list(), dtype=torch.int64)
    max_label = int(t.max().item())
    min_label = int(t.min().item())
    if min_label < 0:
        raise ValueError(f"Метки в CSV не могут быть отрицательными: min={min_label}")
    need_classes = max_label + 1
    if need_classes > num_classes:
        raise ValueError(
            f"В train CSV встречаются метки до {max_label} включительно ({need_classes} классов), "
            f"а --num-classes={num_classes}. Укажите --num-classes не меньше {need_classes}."
        )
    counts = torch.bincount(t, minlength=num_classes).float()
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
    *,
    use_amp: bool = False,
    show_progress: bool = True,
    epoch: int = 1,
) -> float:
    """Одна эпоха обучения; tqdm по батчам и опционально AMP на CUDA."""
    model.train()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = amp.GradScaler("cuda", enabled=amp_enabled)
    iterator = tqdm(loader, desc=f"train ep{epoch}", leave=True) if show_progress else loader
    total_loss = 0.0
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    *,
    use_amp: bool = False,
    show_progress: bool = True,
    epoch: int = 1,
) -> tuple[float, float, float, list[dict[str, object]]]:
    """Валидация: loss, accuracy, macro-F1 и метрики по каждому классу."""
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")
    total_loss = 0.0
    per_class_loss_sum = torch.zeros(num_classes, dtype=torch.float64)
    per_class_loss_count = torch.zeros(num_classes, dtype=torch.float64)
    y_true: list[int] = []
    y_pred: list[int] = []

    iterator = tqdm(loader, desc=f"val ep{epoch}", leave=True) if show_progress else loader
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)
            per_sample_loss = F.cross_entropy(
                outputs, targets, weight=criterion.weight, reduction="none"
            )
        predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if class_mask.any():
                per_class_loss_sum[class_id] += per_sample_loss[class_mask].sum().cpu()
                per_class_loss_count[class_id] += class_mask.sum().cpu()
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    accuracy = float(calculate_accuracy(y_true, y_pred))
    macro_f1 = float(calculate_macro_f1(y_true, y_pred))
    per_class_f1 = calculate_per_class_f1(y_true, y_pred, num_classes)
    per_class_loss = per_class_loss_sum / per_class_loss_count.clamp_min(1)
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    for item in per_class_f1:
        class_id = int(item["class_id"])
        class_mask = y_true_array == class_id
        if class_mask.any():
            item["accuracy"] = float((y_pred_array[class_mask] == class_id).mean())
        else:
            item["accuracy"] = 0.0
        item["loss"] = float(per_class_loss[class_id])
    return total_loss / len(loader.dataset), accuracy, macro_f1, per_class_f1


def add_label_names(per_class_f1: list[dict[str, object]]) -> list[dict[str, object]]:
    """Добавляет текстовое поле label к каждому элементу per-class F1."""
    label_mapping = load_label_mapping()
    return [
        {
            **item,
            "label": label_mapping.get(int(item["class_id"]), str(item["class_id"])),
        }
        for item in per_class_f1
    ]


def save_metrics_report(
    metrics: dict[str, object],
    metrics_dir: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Пишет {model_name}_metrics.json и дополняет {model_name}_experiments.json (как ResNet18)."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(metrics.get("run_id") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    metrics_with_run = {"run_id": run_id, **metrics}

    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    experiments_path = metrics_dir / f"{model_name}_experiments.json"
    metrics_path.write_text(json.dumps(metrics_with_run, indent=2, ensure_ascii=False), encoding="utf-8")

    hyperparameters = metrics["hyperparameters"]
    best_epoch_metrics = metrics["best_epoch_metrics"]
    experiment = {
        "run_id": run_id,
        "model": metrics["model"],
        "model_name": model_name,
        "metrics_dir": str(metrics_dir.resolve()),
        "best_epoch": metrics["best_epoch"],
        "best_macro_f1": metrics["best_macro_f1"],
        "best_accuracy": best_epoch_metrics.get("accuracy"),
        "best_train_loss": best_epoch_metrics.get("train_loss"),
        "best_val_loss": best_epoch_metrics.get("val_loss"),
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


def append_epoch_to_history(
    metrics_dir: Path,
    model_name: str,
    run_id: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    accuracy: float,
    macro_f1: float,
    best_macro_f1: float,
    best_epoch: int,
) -> None:
    """После каждой эпохи (train+val) дописывает строку в {model_name}_epoch_history.jsonl."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / f"{model_name}_epoch_history.jsonl"
    line = json.dumps(
        {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "best_macro_f1_so_far": best_macro_f1,
            "best_epoch_so_far": best_epoch,
        },
        ensure_ascii=False,
    )
    with history_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    """Полный цикл обучения, сохранение лучшего чекпоинта и JSON-метрик."""
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs должен быть >= 1")

    validate_paths(args)
    model_name = args.model_name
    output_dir = (args.output_dir or (ROOT_DIR / "outputs" / "models" / model_name)).resolve()
    metrics_dir = (args.metrics_dir or (ROOT_DIR / "reports" / "metrics" / model_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    set_seed(args.seed)

    device = get_default_device()
    print(f"Using device: {device}", flush=True)
    print(f"metrics_dir={metrics_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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
        pin_memory=device.type == "cuda",
        persistent_workers=device.type == "cuda" and args.num_workers > 0,
    )
    print(
        f"Даталоадеры: train={len(train_loader.dataset)} объектов, {len(train_loader)} батчей/эпоха; "
        f"val={len(val_loader.dataset)}",
        flush=True,
    )
    print("Загрузка модели (ImageNet веса при первом запуске могут кешироваться долго)...", flush=True)

    model = build_convnext_tiny(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = get_class_weights(args.train_csv, args.num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    show_progress = not args.no_progress
    if use_amp:
        print("AMP (mixed precision): включён", flush=True)
    if show_progress:
        print("Прогресс по батчам: tqdm (отключить: --no-progress)", flush=True)
    print("Первый батч на GPU часто долгий (JIT/kernels); дальше обычно быстрее.", flush=True)

    best_macro_f1 = -1.0
    best_epoch = 0
    best_epoch_metrics: dict[str, object] = {}
    checkpoint_path = output_dir / f"{model_name}_best.pt"
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        print(f"--- Эпоха {epoch}/{args.epochs} ---", flush=True)
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            show_progress=show_progress,
            epoch=epoch,
        )
        val_loss, accuracy, macro_f1, per_class_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            use_amp=use_amp,
            show_progress=show_progress,
            epoch=epoch,
        )
        per_class_f1 = add_label_names(per_class_f1)

        improved = macro_f1 > best_macro_f1 + args.early_stopping_min_delta
        if improved:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class_metrics": [
                    {
                        "class_id": item["class_id"],
                        "label": item["label"],
                        "f1": item["f1"],
                        "accuracy": item["accuracy"],
                        "loss": item["loss"],
                        "support": item["support"],
                    }
                    for item in per_class_f1
                ],
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
            f"no_improve={epochs_without_improvement}",
            flush=True,
        )

        append_epoch_to_history(
            metrics_dir,
            model_name,
            run_id,
            epoch,
            train_loss,
            val_loss,
            accuracy,
            macro_f1,
            best_macro_f1,
            best_epoch,
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"early stopping: macro_f1 не улучшался {args.early_stopping_patience} эпох, "
                f"best_macro_f1={best_macro_f1:.4f}",
                flush=True,
            )
            break

    metrics = {
        "run_id": run_id,
        "model": model_name,
        "hyperparameters": {
            "model_name": model_name,
            "metrics_dir": str(metrics_dir),
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
            "amp": use_amp,
            "progress_bars": show_progress,
        },
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_epoch_metrics": best_epoch_metrics,
        "checkpoint": None if args.no_save_checkpoint else str(checkpoint_path),
        "stop_reason": stop_reason,
    }
    metrics_path, experiments_path = save_metrics_report(metrics, metrics_dir, model_name)

    print(f"best_macro_f1={best_macro_f1:.4f}", flush=True)
    if args.no_save_checkpoint:
        print("checkpoint=не сохранялся", flush=True)
    else:
        print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"experiments={experiments_path}", flush=True)


if __name__ == "__main__":
    main()

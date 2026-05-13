import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import create_dataloaders
from src.metrics import calculate_macro_f1
from src.device import get_default_device
from src.training_helpers import build_checkpoint, save_json, set_seed, to_project_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet50 on room type dataset")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train_df.csv")
    parser.add_argument("--val-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "val_df.csv")
    parser.add_argument("--train-images", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train_images")
    parser.add_argument("--val-images", type=Path, default=PROJECT_ROOT / "data" / "raw" / "val_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "models" / "resnet50")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "reports" / "metrics" / "resnet50")
    parser.add_argument("--no-weighted-sampling", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    paths = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--train-images": args.train_images,
        "--val-images": args.val_images,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены входные файлы/папки:\n" + "\n".join(missing))


# загрузка и разбиение
def load_dataset(args: argparse.Namespace):
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

    classes_df = pd.read_csv(args.train_csv, usecols=["result", "label"]).dropna()
    classes_df["result"] = classes_df["result"].astype(int)
    classes = (
        classes_df.sort_values("result")
        .drop_duplicates(subset=["result"], keep="first")["label"]
        .tolist()
    )
    return train_loader, val_loader, classes

# создание модели
# Мы берём pretrained ResNet50 и меняем последний слой под наше число классов.
def build_model(num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# обучение одной эпохи
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += targets.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc

# валидация
# - val_loss
# - val_acc
# - val_f1_macro
# - сохраняем все y_true и y_pred для отчёта
@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += targets.size(0)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / running_total
    val_acc = running_correct / running_total

    # macro F1 полезен, когда классы несбалансированы:
    # каждый класс в среднем имеет одинаковый вес
    val_f1_macro = calculate_macro_f1(all_targets, all_preds)

    return val_loss, val_acc, val_f1_macro, all_targets, all_preds


# отчет по валидации
def evaluate_and_print_report(model, loader, criterion, device, classes):
    val_loss, val_acc, val_f1_macro, all_targets, all_preds = validate(
        model, loader, criterion, device
    )

    print('\nValidation metrics:')
    print(f'val_loss: {val_loss:.4f}')
    print(f'val_acc: {val_acc:.4f}')
    print(f'val_f1_macro: {val_f1_macro:.4f}')

    print('\nClassification report:')
    report = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        digits=4,
        zero_division=0
    )
    print(report)

    # Можно выводить confusion_matrix
    # print('Confusion matrix:')
    # cm = confusion_matrix(all_targets, all_preds)
    # print(cm)

    return val_loss, val_acc, val_f1_macro, report

# Лучшую модель сохраняем по val_f1_macro, а не по accuracy.
# Это часто лучше для многоклассовой задачи.
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args: argparse.Namespace,
    classes,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_metrics = {}
    checkpoint_path = args.output_dir / "resnet50_best.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1_macro, all_targets, all_preds = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f'Epoch [{epoch + 1}/{args.epochs}] -- '
            f'train_loss: {train_loss:.4f} -- train_acc: {train_acc:.4f} -- '
            f'val_loss: {val_loss:.4f} -- val_acc: {val_acc:.4f} -- '
            f'val_f1_macro: {val_f1_macro:.4f}'
        )

        # Сохраняем лучшую модель по macro F1
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model_wts = copy.deepcopy(model.state_dict())
            if not args.no_save_checkpoint:
                torch.save(
                    build_checkpoint(
                        model=model,
                        model_name="resnet50",
                        epoch=epoch + 1,
                        best_metric=best_val_f1,
                        optimizer=optimizer,
                        checkpoint_path=checkpoint_path,
                        extra={
                            "num_classes": len(classes),
                            "image_size": args.image_size,
                            "idx_to_class": {str(i): class_name for i, class_name in enumerate(classes)},
                        },
                    ),
                    checkpoint_path,
                )
                print(f'Best model saved to: {checkpoint_path}')
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "accuracy": val_acc,
                "macro_f1": best_val_f1,
            }

    model.load_state_dict(best_model_wts)
    return model, best_val_f1, best_metrics

def main():
    args = parse_args()
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = get_default_device()
    print(args.train_csv)
    print(args.train_images)
    print(f'DEVICE: {device}')

    train_loader, val_loader, classes = load_dataset(args)

    print('Classes:')
    for i, cls_name in enumerate(classes):
        print(f'  {i}: {cls_name}')

    # Количество классов определяется автоматически
    model = build_model(num_classes=len(classes))
    model = model.to(device)

    # Если классы сильно несбалансированы, можно добавить class weights.
    criterion = nn.CrossEntropyLoss()

    # Adam — хороший старт.
    # Потом можно попробовать AdamW.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Уменьшает learning rate, если val_loss не улучшается.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )

    model, best_val_f1, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        classes=classes,
    )

    metrics = {
        "model": "resnet50",
        "best_macro_f1": best_val_f1,
        "best_epoch_metrics": best_metrics,
        "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(args.output_dir / "resnet50_best.pt"),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "weighted_sampling": not args.no_weighted_sampling,
            "save_checkpoint": not args.no_save_checkpoint,
        },
    }
    save_json(metrics, args.metrics_dir / "resnet50_metrics.json")

    print('Training finished.')

    print('\nBest model evaluation on validation set:')
    evaluate_and_print_report(model, val_loader, criterion, device, classes)

if __name__ == '__main__':
    main()

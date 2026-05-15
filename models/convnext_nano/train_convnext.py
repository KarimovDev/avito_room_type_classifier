import torch
import torch.nn as nn
import torch.optim as optim
import timm
from pathlib import Path
import sys
import argparse
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.device import get_default_device 
from src.dataloaders import create_dataloaders
from src.labels import load_label_mapping
from src.mlflow_utils import end_mlflow_run, log_mlflow_artifacts, log_mlflow_metrics, log_mlflow_params, start_mlflow_run
from src.training_helpers import build_checkpoint, save_json, set_seed, to_project_relative_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt Nano on room type dataset")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train_df.csv")
    parser.add_argument("--val-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "val_df.csv")
    parser.add_argument("--train-images", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train_images")
    parser.add_argument("--val-images", type=Path, default=PROJECT_ROOT / "data" / "raw" / "val_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "models" / "convnext_nano")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "reports" / "metrics" / "convnext_nano")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampling", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def validate_paths(args):
    paths = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--train-images": args.train_images,
        "--val-images": args.val_images,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены входные файлы/папки:\n" + "\n".join(missing))

def main():
    args = parse_args()
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    DEVICE = get_default_device()
    print(f"Используемое устройство: {DEVICE}")

    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        train_image_root=args.train_images,
        val_image_root=args.val_images,
        use_weighted_sampling=not args.no_weighted_sampling,
        seed=args.seed,
    )

    print(f"ИТОГО: Объектов в Train: {len(train_loader.dataset)}")

    print(f"Инициализация ConvNeXt Nano...")
    model = timm.create_model(
        'convnext_nano', 
        pretrained=not args.no_pretrained, 
        num_classes=args.num_classes, 
        drop_rate=0.5, 
        drop_path_rate=0.3
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

    print(f"Начало обучения на {args.epochs} эпох...")
    
    best_macro_f1 = 0.0
    best_epoch = 0
    best_epoch_metrics = {}
    checkpoint_path = args.output_dir / "convnext_nano_best.pt"
    metrics_path = args.metrics_dir / "convnext_nano_metrics.json"
    idx_to_class = {str(class_id): label for class_id, label in load_label_mapping().items()}

    start_mlflow_run(
        "convnext_nano",
        "convnext_nano",
        {
            "model": "convnext_nano",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
            "weighted_sampling": not args.no_weighted_sampling,
        },
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
        val_acc = 100 * correct / len(all_targets) if len(all_targets) > 0 else 0

        scheduler.step(macro_f1)

        print(f"Эпоха {epoch+1}/{args.epochs} | Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Acc: {val_acc:.2f}% | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            best_epoch_metrics = {
                "epoch": best_epoch,
                "train_loss": float(epoch_train_loss),
                "val_loss": float(epoch_val_loss),
                "accuracy": float(val_acc / 100),
                "macro_f1": float(macro_f1),
            }
            if not args.no_save_checkpoint:
                torch.save(
                    build_checkpoint(
                        model=model,
                        model_name="convnext_nano",
                        epoch=epoch + 1,
                        best_metric=best_macro_f1,
                        optimizer=optimizer,
                        checkpoint_path=checkpoint_path,
                        extra={
                            "num_classes": args.num_classes,
                            "image_size": args.image_size,
                            "idx_to_class": idx_to_class,
                        },
                    ),
                    checkpoint_path,
                )
            
            metrics = {
                "epoch": int(epoch + 1),
                "model": "convnext_nano",
                "best_epoch": best_epoch,
                "best_macro_f1": float(best_macro_f1),
                "best_accuracy": best_epoch_metrics.get("accuracy"),
                "best_train_loss": best_epoch_metrics.get("train_loss"),
                "best_val_loss": best_epoch_metrics.get("val_loss"),
                "best_epoch_metrics": best_epoch_metrics,
                "train_loss": float(epoch_train_loss),
                "val_loss": float(epoch_val_loss),
                "accuracy": float(val_acc / 100),
                "macro_f1": float(macro_f1),
                "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(checkpoint_path),
                "hyperparameters": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "image_size": args.image_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "pretrained": not args.no_pretrained,
                    "weighted_sampling": not args.no_weighted_sampling,
                    "save_checkpoint": not args.no_save_checkpoint,
                },
            }
            save_json(metrics, metrics_path)
            print(f"Найдена лучшая модель (F1: {best_macro_f1:.4f})")

        log_mlflow_metrics(
            {
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "accuracy": val_acc / 100,
                "macro_f1": macro_f1,
                "best_macro_f1": best_macro_f1,
            },
            step=epoch + 1,
        )

    log_mlflow_metrics(
        {
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "best_accuracy": best_epoch_metrics.get("accuracy"),
            "best_train_loss": best_epoch_metrics.get("train_loss"),
            "best_val_loss": best_epoch_metrics.get("val_loss"),
        }
    )
    log_mlflow_params(
        {
            "best_epoch": best_epoch,
            "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(checkpoint_path),
            "metrics_json": to_project_relative_path(metrics_path),
        }
    )
    log_mlflow_artifacts(
        [
            metrics_path,
            None if args.no_save_checkpoint else checkpoint_path,
        ]
    )
    end_mlflow_run()

if __name__ == "__main__":
    main()

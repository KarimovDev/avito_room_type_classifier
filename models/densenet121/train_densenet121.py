from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.densenet121.densenet121 import build_densenet121
from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.mlflow_utils import end_mlflow_run, log_mlflow_artifacts, log_mlflow_metrics, log_mlflow_params, start_mlflow_run
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1
from src.training_helpers import build_checkpoint, set_seed, to_project_relative_path



# Буст-множители весов классов

_CLASS_WEIGHT_BOOSTS: dict[int, float] = {
    2: 3.0, # универсальная комната (низкий recall)
    4: 0.7, # спальня (слишком жадная)
    5: 1.5, # кабинет
    11: 1.5, # гардеробная
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet121 on room type dataset")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "train_df.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "val_df.csv")
    parser.add_argument("--train-images", type=Path, default=ROOT_DIR / "data" / "raw" / "train_images")
    parser.add_argument("--val-images", type=Path, default=ROOT_DIR / "data" / "raw" / "val_images")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "models" / "densenet121")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT_DIR / "reports" / "metrics" / "densenet121")
    parser.add_argument("--no-pretrained", action="store_true", help="Не использовать веса ImageNet")
    parser.add_argument("--no-class-weights", action="store_true", help="Отключить веса классов в loss")
    parser.add_argument(
        "--no-weighted-sampling",
        action="store_true",
        help="Отключить WeightedRandomSampler для train DataLoader",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Не сохранять веса модели, оставить только JSON с F1-метриками",
    )
    # 3 stages, как в последних экспериментах
    parser.add_argument("--epochs-stage1", type=int, default=2, help="Эпох для head-only")
    parser.add_argument("--epochs-stage2", type=int, default=8, help="Эпох для full fine-tuning")
    parser.add_argument("--epochs-stage3", type=int, default=5, help="Эпох для дожига")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="LR для head-only")
    parser.add_argument("--lr-stage2", type=float, default=1e-4, help="LR для full fine-tuning")
    parser.add_argument("--lr-stage3", type=float, default=3e-5, help="LR для дожига")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing для CrossEntropyLoss")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Сколько эпох ждать улучшения macro-F1, 0 = не останавливать",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Минимальный прирост macro-F1, который считается улучшением",
    )
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


def get_class_weights(csv_path: Path, num_classes: int, device: torch.device) -> torch.Tensor:
    targets = pd.read_csv(csv_path)["result"].astype(int)
    counts = torch.bincount(torch.tensor(targets.to_list()), minlength=num_classes).float()

    weights = torch.zeros(num_classes, dtype=torch.float32)
    existing_classes = counts > 0
    weights[existing_classes] = counts.sum() / (existing_classes.sum() * counts[existing_classes])

    for class_id, boost in _CLASS_WEIGHT_BOOSTS.items():
        if class_id < num_classes:
            weights[class_id] *= boost

    return weights.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in loader:
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
) -> tuple[float, float, float, list[dict[str, object]], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    per_class_loss_sum = torch.zeros(num_classes, dtype=torch.float64)
    per_class_loss_count = torch.zeros(num_classes, dtype=torch.float64)
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        per_sample_loss = F.cross_entropy(outputs, targets, weight=criterion.weight, reduction="none")
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
    return total_loss / len(loader.dataset), accuracy, macro_f1, per_class_f1, y_true, y_pred


def add_label_names(per_class_f1: list[dict[str, object]]) -> list[dict[str, object]]:
    label_mapping = load_label_mapping()
    return [
        {**item, "label": label_mapping.get(int(item["class_id"]), str(item["class_id"]))}
        for item in per_class_f1
    ]


def save_metrics_report(metrics: dict[str, object], metrics_dir: Path) -> tuple[Path, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(metrics.get("run_id") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    metrics_path = metrics_dir / "densenet121_metrics.json"
    experiments_path = metrics_dir / "densenet121_experiments.json"
    report_json_path = metrics_dir / "densenet121_classification_report.json"
    report_txt_path = metrics_dir / "densenet121_classification_report.txt"

    best_epoch_metrics = metrics["best_epoch_metrics"]
    if "classification_report" in best_epoch_metrics:
        report_json_path.write_text(
            json.dumps(best_epoch_metrics["classification_report"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if "classification_report_text" in best_epoch_metrics:
        report_txt_path.write_text(str(best_epoch_metrics["classification_report_text"]), encoding="utf-8")

    metrics_with_run = {"run_id": run_id, **metrics}
    metrics_with_run["best_epoch_metrics"] = {
        key: value
        for key, value in best_epoch_metrics.items()
        if key not in {"classification_report", "classification_report_text"}
    }
    metrics_path.write_text(json.dumps(metrics_with_run, indent=2, ensure_ascii=False), encoding="utf-8")

    hyperparameters = metrics["hyperparameters"]
    experiment = {
        "run_id": run_id,
        "model": metrics["model"],
        "best_epoch": metrics["best_epoch"],
        "best_stage_epoch": best_epoch_metrics.get("stage_epoch"),
        "best_macro_f1": metrics["best_macro_f1"],
        "best_accuracy": best_epoch_metrics.get("accuracy"),
        "best_train_loss": best_epoch_metrics.get("train_loss"),
        "best_val_loss": best_epoch_metrics.get("val_loss"),
        "stop_reason": metrics["stop_reason"],
        "checkpoint": metrics["checkpoint"],
        "epochs": hyperparameters["epochs"],
        "epochs_stage1": hyperparameters["epochs_stage1"],
        "epochs_stage2": hyperparameters["epochs_stage2"],
        "epochs_stage3": hyperparameters["epochs_stage3"],
        "lr_stage1": hyperparameters["lr_stage1"],
        "lr_stage2": hyperparameters["lr_stage2"],
        "lr_stage3": hyperparameters["lr_stage3"],
        "batch_size": hyperparameters["batch_size"],
        "image_size": hyperparameters["image_size"],
        "weight_decay": hyperparameters["weight_decay"],
        "label_smoothing": hyperparameters["label_smoothing"],
        "seed": hyperparameters["seed"],
        "pretrained": hyperparameters["pretrained"],
        "class_weights": hyperparameters["class_weights"],
        "weighted_sampling": hyperparameters["weighted_sampling"],
        "early_stopping_patience": hyperparameters["early_stopping_patience"],
        "early_stopping_min_delta": hyperparameters["early_stopping_min_delta"],
        "metrics_json": to_project_relative_path(metrics_path),
    }

    if experiments_path.exists():
        experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    else:
        experiments = []
    experiments.append(experiment)
    experiments_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics_path, experiments_path


def _run_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    num_classes: int,
    checkpoint_path: Path,
    save_checkpoint: bool,
    best_macro_f1: float,
    best_epoch: int,
    best_epoch_metrics: dict[str, object],
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    idx_to_class: dict[str, str],
    epoch_offset: int = 0,
) -> tuple[float, int, dict[str, object], str, list[dict]]:
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    history: list[dict] = []

    print(f"\n{'='*60}")
    print(f"Этап {stage_name}: {num_epochs} эпох, lr={optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*60}")

    for local_epoch in range(1, num_epochs + 1):
        global_epoch = epoch_offset + local_epoch

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, macro_f1, per_class_f1, y_true, y_pred = validate(
            model, val_loader, criterion, device, num_classes
        )
        per_class_f1 = add_label_names(per_class_f1)
        label_mapping = load_label_mapping()
        target_names = [label_mapping.get(i, str(i)) for i in range(num_classes)]

        history.append({
            "epoch": f"s{stage_name}_{local_epoch}",
            "global_epoch": global_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        improved = macro_f1 > best_macro_f1 + early_stopping_min_delta
        if improved:
            best_macro_f1 = macro_f1
            best_epoch = global_epoch
            best_stage_epoch = f"s{stage_name}_{local_epoch}"
            epochs_without_improvement = 0
            best_epoch_metrics = {
                "epoch": best_epoch,
                "stage_epoch": best_stage_epoch,
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
                "classification_report": classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                ),
                "classification_report_text": classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    target_names=target_names,
                    zero_division=0,
                ),
            }
            if save_checkpoint:
                torch.save(
                    build_checkpoint(
                        model=model,
                        model_name="densenet121",
                        epoch=global_epoch,
                        best_metric=best_macro_f1,
                        optimizer=optimizer,
                        checkpoint_path=checkpoint_path,
                        extra={
                            "num_classes": num_classes,
                            "stage_epoch": best_stage_epoch,
                            "idx_to_class": idx_to_class,
                        },
                    ),
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        saved_marker = " saved" if improved else ""
        print(
            f"  epoch=s{stage_name}_{local_epoch} (global={global_epoch})"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  accuracy={accuracy:.4f}"
            f"  macro_f1={macro_f1:.4f}"
            f"  best={best_macro_f1:.4f}{saved_marker}"
        )
        log_mlflow_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "best_macro_f1": best_macro_f1,
            },
            step=global_epoch,
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"  early stopping: macro_f1 не улучшался {early_stopping_patience} эпох, "
                f"best={best_macro_f1:.4f}"
            )
            break

    print(f"Этап {stage_name} завершён. best_macro_f1={best_macro_f1:.4f}")
    return best_macro_f1, best_epoch, best_epoch_metrics, stop_reason, history


def main() -> None:
    args = parse_args()
    if args.epochs_stage1 < 1 or args.epochs_stage2 < 1 or args.epochs_stage3 < 1:
        raise ValueError("Все --epochs-stage* должны быть >= 1")
    total_epochs = args.epochs_stage1 + args.epochs_stage2 + args.epochs_stage3

    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    set_seed(args.seed)

    device = get_default_device()
    print(f"Using device: {device}")

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

    model = build_densenet121(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = get_class_weights(args.train_csv, args.num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    start_mlflow_run(
        "densenet121",
        f"densenet121_{run_id}",
        {
            "model": "densenet121",
            "epochs": total_epochs,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "epochs_stage3": args.epochs_stage3,
            "lr_stage1": args.lr_stage1,
            "lr_stage2": args.lr_stage2,
            "lr_stage3": args.lr_stage3,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
            "class_weights": not args.no_class_weights,
            "weighted_sampling": not args.no_weighted_sampling,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
    )

    # after head-only
    head_checkpoint = args.output_dir / "densenet121_head_best.pt"
    finetune_checkpoint = args.output_dir / "densenet121_best.pt"

    best_macro_f1 = -1.0
    best_epoch = 0
    best_epoch_metrics: dict[str, object] = {}
    full_history: list[dict] = []
    idx_to_class = {str(class_id): label for class_id, label in load_label_mapping().items()}
    stop_reason = "max_epochs"

    # stage 1: classifier only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_s1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1,
    )

    best_macro_f1, best_epoch, best_epoch_metrics, _, history_s1 = _run_stage(
        stage_name="1",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s1,
        device=device,
        num_epochs=args.epochs_stage1,
        num_classes=args.num_classes,
        checkpoint_path=head_checkpoint,
        save_checkpoint=not args.no_save_checkpoint,
        best_macro_f1=best_macro_f1,
        best_epoch=best_epoch,
        best_epoch_metrics=best_epoch_metrics,
        early_stopping_patience=0,
        early_stopping_min_delta=args.early_stopping_min_delta,
        idx_to_class=idx_to_class,
        epoch_offset=0,
    )
    full_history.extend(history_s1)

    # stage 2: full fine-tuning
    if not args.no_save_checkpoint and head_checkpoint.exists():
        ckpt = torch.load(head_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = True

    optimizer_s2 = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_stage2,
        weight_decay=args.weight_decay,
    )

    best_macro_f1, best_epoch, best_epoch_metrics, stop_reason_s2, history_s2 = _run_stage(
        stage_name="2",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s2,
        device=device,
        num_epochs=args.epochs_stage2,
        num_classes=args.num_classes,
        checkpoint_path=finetune_checkpoint,
        save_checkpoint=not args.no_save_checkpoint,
        best_macro_f1=best_macro_f1,
        best_epoch=best_epoch,
        best_epoch_metrics=best_epoch_metrics,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        idx_to_class=idx_to_class,
        epoch_offset=args.epochs_stage1,
    )
    full_history.extend(history_s2)
    stop_reason = stop_reason_s2

    # stage 3: smaller lr from best checkpoint
    if not args.no_save_checkpoint and finetune_checkpoint.exists():
        ckpt = torch.load(finetune_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    optimizer_s3 = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_stage3,
        weight_decay=args.weight_decay,
    )

    best_macro_f1, best_epoch, best_epoch_metrics, stop_reason_s3, history_s3 = _run_stage(
        stage_name="3",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_s3,
        device=device,
        num_epochs=args.epochs_stage3,
        num_classes=args.num_classes,
        checkpoint_path=finetune_checkpoint,
        save_checkpoint=not args.no_save_checkpoint,
        best_macro_f1=best_macro_f1,
        best_epoch=best_epoch,
        best_epoch_metrics=best_epoch_metrics,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        idx_to_class=idx_to_class,
        epoch_offset=args.epochs_stage1 + args.epochs_stage2,
    )
    full_history.extend(history_s3)
    if stop_reason_s3 == "early_stopping":
        stop_reason = "early_stopping"

    metrics = {
        "run_id": run_id,
        "model": "densenet121",
        "hyperparameters": {
            "epochs": total_epochs,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "epochs_stage3": args.epochs_stage3,
            "lr_stage1": args.lr_stage1,
            "lr_stage2": args.lr_stage2,
            "lr_stage3": args.lr_stage3,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
            "class_weights": not args.no_class_weights,
            "weighted_sampling": not args.no_weighted_sampling,
            "save_checkpoint": not args.no_save_checkpoint,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_accuracy": best_epoch_metrics.get("accuracy"),
        "best_train_loss": best_epoch_metrics.get("train_loss"),
        "best_val_loss": best_epoch_metrics.get("val_loss"),
        "best_epoch_metrics": best_epoch_metrics,
        "history": full_history,
        "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(finetune_checkpoint),
        "stop_reason": stop_reason,
    }

    metrics_path, experiments_path = save_metrics_report(metrics, args.metrics_dir)
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
            "best_stage_epoch": best_epoch_metrics.get("stage_epoch"),
            "checkpoint": None if args.no_save_checkpoint else to_project_relative_path(finetune_checkpoint),
            "metrics_json": to_project_relative_path(metrics_path),
        }
    )
    log_mlflow_artifacts(
        [
            metrics_path,
            experiments_path,
            args.metrics_dir / "densenet121_classification_report.json",
            args.metrics_dir / "densenet121_classification_report.txt",
            None if args.no_save_checkpoint else finetune_checkpoint,
        ]
    )
    end_mlflow_run()

    print(f"\nbest_macro_f1={best_macro_f1:.4f}")
    if args.no_save_checkpoint:
        print("checkpoint=не сохранялся")
    else:
        print(f"checkpoint={finetune_checkpoint}")
    print(f"metrics={metrics_path}")
    print(f"experiments={experiments_path}")


if __name__ == "__main__":
    main()

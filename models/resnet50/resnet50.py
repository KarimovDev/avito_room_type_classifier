import os
import copy
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import default_collate
from torchvision.transforms import v2

from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import create_dataloaders
from src.metrics import calculate_macro_f1
from src.device import get_default_device

# настройки
# - пути к данным
# - batch size
# - число эпох
# - learning rate
# - долю validation
# - путь сохранения модели
DATA_DIR_PROCESSED = PROJECT_ROOT/Path("./data/processed").resolve()
DATA_DIR_RAW = PROJECT_ROOT/Path("./data/raw").resolve()
CSV_PATH = os.path.join(DATA_DIR_PROCESSED, 'train_df.csv')
CSV_PATH_VAL = os.path.join(DATA_DIR_PROCESSED, 'val_df.csv')
IMAGES_DIR = os.path.join(DATA_DIR_RAW, 'train_images')
VAL_DIR = os.path.join(DATA_DIR_RAW, 'val_images')

BATCH_SIZE = 32           # Можно увеличить, если хватает GPU памяти
NUM_WORKERS = 0           # В Colab обычно 2-4 достаточно
NUM_EPOCHS = 15           # Для старта 10, потом можно 15-30
LEARNING_RATE = 1e-4      # Хороший старт для fine-tuning ResNet50
RANDOM_SEED = 42
MODEL_SAVE_PATH = PROJECT_ROOT/Path("./outputs/models/best_resnet50_avito.pth").resolve()

DEVICE = get_default_device()
print(DEVICE)


# Нужно для воспроизводимости разбиения и обучения.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# загрузка и разбиение
def load_dataset():
    train_loader, val_loader = create_dataloaders(
        train_csv_path=None,
        val_csv_path=None,
        train_image_root=None,
        val_image_root=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=224,
        use_weighted_sampling=True,
    )

    classes_df = pd.read_csv(CSV_PATH, usecols=["result", "label"]).dropna()
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1_macro, all_targets, all_preds = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}] -- '
            f'train_loss: {train_loss:.4f} -- train_acc: {train_acc:.4f} -- '
            f'val_loss: {val_loss:.4f} -- val_acc: {val_acc:.4f} -- '
            f'val_f1_macro: {val_f1_macro:.4f}'
        )

        # Сохраняем лучшую модель по macro F1
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Best model saved to: {MODEL_SAVE_PATH}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    set_seed(RANDOM_SEED)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(CSV_PATH)
    print(IMAGES_DIR)
    print(f'DEVICE: {DEVICE}')

    train_loader, val_loader, classes = load_dataset()

    print('Classes:')
    for i, cls_name in enumerate(classes):
        print(f'  {i}: {cls_name}')

    # Количество классов определяется автоматически
    model = build_model(num_classes=len(classes))
    model = model.to(DEVICE)

    # Если классы сильно несбалансированы, можно добавить class weights.
    criterion = nn.CrossEntropyLoss()

    # Adam — хороший старт.
    # Потом можно попробовать AdamW.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Уменьшает learning rate, если val_loss не улучшается.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS
    )

    print('Training finished.')

    print('\nBest model evaluation on validation set:')
    evaluate_and_print_report(model, val_loader, criterion, DEVICE, classes)

if __name__ == '__main__':
    main()
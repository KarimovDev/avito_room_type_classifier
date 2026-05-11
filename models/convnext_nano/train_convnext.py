import torch
import torch.nn as nn
import torch.optim as optim
import timm
from pathlib import Path
import sys
import argparse  # Добавлено
from src.device import get_default_device 
from sklearn.metrics import f1_score
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataloaders import create_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    CURRENT_SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent 

    DEVICE = get_default_device()
    print(f"Используемое устройство: {DEVICE}")
    
    BATCH_SIZE = 32
    EPOCHS = args.epochs
  
    DATA_DIR = PROJECT_ROOT / "data"
    train_csv = DATA_DIR / "processed" / "train_df.csv"
    train_imgs = DATA_DIR / "raw" / "train_images"
    val_imgs = DATA_DIR / "raw" / "val_images"

    save_dir = PROJECT_ROOT / "outputs"
    save_dir.mkdir(exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=224,
        num_workers=0,
        train_csv_path=str(train_csv),
        train_image_root=str(train_imgs),
        val_image_root=str(val_imgs)
    )

    print(f"ИТОГО: Объектов в Train: {len(train_loader.dataset)}")

    print(f"Инициализация ConvNeXt Nano...")
    model = timm.create_model(
        'convnext_nano', 
        pretrained=True, 
        num_classes=19, 
        drop_rate=0.5, 
        drop_path_rate=0.3
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

    print(f"Начало обучения на {EPOCHS} эпох...")
    
    best_macro_f1 = 0.0

    for epoch in range(EPOCHS):
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

        print(f"Эпоха {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Acc: {val_acc:.2f}% | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), save_dir / "best_model_ConvNeXt_Nano.pth")
            
            metrics = {
                "epoch": int(epoch + 1),
                "train_loss": float(epoch_train_loss),
                "val_loss": float(epoch_val_loss),
                "accuracy": float(val_acc),
                "macro_f1": float(macro_f1)
            }
            
            with open(save_dir / "best_metrics_convnext.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
                print(f"Найдена лучшая модель (F1: {best_macro_f1:.4f})")

if __name__ == "__main__":
    main()

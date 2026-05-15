## Установка и запуск проекта

### 0. Клонировать репозиторий

```bash
git clone https://github.com/your_username/room_type_classifier.git
cd room_type_classifier
```

### Требования

- Python: **3.12.x** (фиксируется в `.python-version`)
- Менеджер окружений: **uv**
- Утилита команд: **just** (используется `justfile` в корне проекта)

### 1. Установить `just` (если не установлен)

**macOS (Homebrew):**

```bash
brew install just
```

**Windows (Scoop):**

```powershell
scoop install just
```

**Windows (Chocolatey):**

```powershell
choco install just
```

### 2. Установить `uv` (если не установлен)

**macOS / Linux:**

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Если после установки `uv` не находится в терминале, добавьте в `PATH` (один раз на сессию):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Установка зависимостей через `just`

Проект использует единый `pyproject.toml`. Зависимости разбиты на группы (могут пересекаться):

- `data` — датасеты, dataloader, transforms, метрики, `torch`/`torchvision`
- `efficientnet` — обучение EfficientNet baseline на общем pipeline
- `resnet18` — обучение ResNet18
- `resnet50` — обучение ResNet50
- `densenet121` — обучение DenseNet121
- `convnext_nano` — обучение ConvNeXt Nano
- `convnext_tiny` — обучение ConvNeXt Tiny
- `streamlit` — UI-сервис проекта
- `yolo` — YOLO demo/inference зависимости

```bash
just install
```

Для установки всех групп:

```bash
just install-all
```

### 3.1. PyTorch (`torch` / `torchvision`): опционально переустановить (CPU / CUDA / PyPI)

`torch` и `torchvision` **закреплены** в группе `data` в `pyproject.toml` / `uv.lock` — обычно достаточно `just install`.

Если вам нужен CPU-репозиторий PyTorch или конкретная CUDA-ветка, после `just install` можно переустановить `torch`/`torchvision` одной из команд ниже.

- **По умолчанию (PyPI):**

```bash
just pytorch-pypi
```

- **CPU wheels из репозитория PyTorch:**

```bash
just pytorch-cpu
```

- **CUDA 13.0 wheels из репозитория PyTorch:**

```bash
just pytorch-cu130
```

Если команда с CUDA-репозиторием не находит wheel для вашей связки **OS + Python + CUDA**, значит **для вас вариант `cu130` несовместим** — в этом случае ориентируйтесь на “pip install” комбинацию с [официального конфигуратора PyTorch](https://pytorch.org/get-started/locally/) и/или переключитесь на `just pytorch-pypi` / `just pytorch-cpu`.

Проверка:

```bash
just run "python -c \"import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())\""
```

### 4. Запуск примера (YOLO)

```bash
just run-yolo
```

---

## Работа с Jupyter (если не виден kernel)

Если после создания нового окружения (.venv) Jupyter / PyCharm не видит kernel проекта, выполните:

```bash
uv run python -m pip install ipykernel

uv run python -m ipykernel install --user \
  --name room_type_classifier \
  --display-name "room_type_classifier"
```

После этого выберите kernel:

room_type_classifier

---

## Структура проекта

```
data/
  raw/           # сырые данные
  processed/     # обработанные данные

notebooks/       # EDA и исследование данных

models/          # эксперименты и код отдельных моделей

src/             # общий код проекта: dataset, dataloaders, transforms, metrics, preprocessing

streamlit/       # UI-сервис проекта
```

## Data pipeline

Для работы pipeline в папке `data/` должны лежать raw-данные:

```
data/
  raw/
    train_df.csv
    val_df.csv
    test_df.csv
    heuristics_cabinet.csv
    heuristics_detskaya.csv
    heuristics_dressing_room.csv

    train_images/
    val_images/
    test_images/
    heuristics_images/
```

`train_df.csv` и `val_df.csv` содержат разметку классов в признаке `result`.
В `test_df.csv` колонки `result` нет.

Изображения берутся не по URL из признака `image`, а из локальных папок:

```
train_images/
val_images/
test_images/
heuristics_images/
```

Связь между CSV и файлом изображения идёт через признак:

```
image_id_ext -> {image_id_ext}.jpg
```

После preprocessing в CSV также появляется `image_path`. Для обычного train
он указывает на `train_images/{image_id_ext}.jpg`, а для heuristics — на
`heuristics_images/{image_id_ext}.jpg`.

Перед обучением нужно подготовить CSV:

```bash
just prepare-data
```

Команда читает `data/raw/*.csv`, очищает train/val, добавляет флаги для test
и сохраняет результат в `data/processed/`.

Вспомогательные heuristics-датасеты можно добавить в train опционально:

```bash
just prepare-data-with-heuristics
```

Эта команда добавляет рекомендуемый набор: `cabinet` и `dressing_room`.
`detskaya` остаётся выключенной, потому что в базовом train для неё уже достаточно
примеров, а качество heuristics-разметки может быть шумным. Heuristics не
добавляются целиком: preprocessing считает средний размер класса в train и
добирает только недостающее количество. Дубли удаляются по `image_id_ext` и
`title`.

Для выборочного добавления:

```bash
just prepare-data-heuristics cabinet,dressing_room
```

Можно дополнительно ограничить количество строк из каждого heuristics-датасета:

```bash
just prepare-data-heuristics-limited cabinet,dressing_room 500
```

После подготовки используются уже не raw CSV, а обработанные файлы:

```
data/processed/train_df.csv
data/processed/val_df.csv
data/processed/test_df.csv
```

Для train/val после preprocessing остаются признаки:

```
image_id_ext
image
result
label
title
source
is_auxiliary
image_path
```

Для test остаются признаки:

```
image_id_ext
image
item_id
image_exists
image_is_valid
can_predict
source
is_auxiliary
image_path
```

В test строки не удаляются, чтобы не менять порядок и количество объектов.
Для предсказаний в `DataLoader` используются только строки `can_predict=True`.

Для train/val применяется единая схема классов:

```text
старый класс 18 удаляется
старый класс 19 -> новый класс 18
```

Поэтому после подготовки модель обучается на 19 классах `0..18`. Для test
классы не меняются, потому что там нет `result`: строки не удаляются,
а изображения помечаются флагом `can_predict`. Обратное соответствие классов сохраняется в:

```text
data/processed/class_mapping.json
```

Параметры последнего preprocessing сохраняются в:

```text
data/processed/preprocessing_manifest.json
```

---

## Обучение моделей

Перед обучением подготовьте данные:

```bash
just prepare-data
```

Запуски моделей идут через `just`:

```bash
just train-resnet18
just train-resnet50
just train-densenet121
just train-efficientnet
just train-convnext-nano
just train-convnext-tiny
```

MLflow логируется в DagsHub remote tracking. Перед первым обучением нужно
установить зависимости и авторизоваться:

```bash
just install-tracking
just dagshub-login
```

После этого запуски обучения автоматически попадают в DagsHub:

```text
https://dagshub.com/YashinSergey/room_type_classifier/experiments
```

Если нужно временно использовать локальный MLflow без DagsHub, можно запустить
обучение с переменной:

```bash
RTC_MLFLOW_LOCAL=1 just train-resnet18
```

Таблица сравнения моделей строится из MLflow:

```bash
just compare-models
```

Результат сохраняется в:

```text
reports/model_comparison.csv
```

Главная метрика для выбора модели:

```text
best_macro_f1
```

Для YOLO обучение не запускается: используется внешний pretrained checkpoint. Для проверки inference:

```bash
just run-yolo
```

YOLO тоже попадает в MLflow, но у него другая метрика:

```text
avg_top1_confidence
```

Ее нельзя напрямую сравнивать с `best_macro_f1` обученных классификаторов,
поэтому в итоговой таблице для YOLO нужно смотреть поля `metric_name` и
`best_metric`.

После обучения проверьте метрики и checkpoint-и:

```bash
just check-training-outputs
```

Все новые checkpoint-и должны сохранять относительные пути и общий набор полей:

```text
model_name
model_state_dict
epoch
best_metric
metric_name
checkpoint_path
```

---

## Dataset

Для чтения данных используется класс:

```python
RoomTypeDataset
```

Он делает следующее:

1. читает CSV-файл
2. берёт `image_path`, если он есть, или `image_id_ext`
3. формирует путь к локальному изображению
4. открывает изображение
5. приводит изображение к RGB
6. берёт числовой класс из колонки `result`
7. применяет transforms
8. возвращает:

```python
image, target
```

где:

```
image  — tensor изображения
target — номер класса от 0 до 18 после подготовки данных
```

Для test в CSV нет `result`, поэтому `RoomTypeDataset` возвращает:

```python
image, image_id, item_id
```

---

## Transforms

Для train и validation используются разные transforms

### Train transforms

```
Resize(224, 224)
RandomHorizontalFlip
RandomRotation(10)
ColorJitter(brightness=0.2, contrast=0.2)
ToTensor
Normalize(ImageNet mean/std)
```

### Validation transforms

```
Resize(224, 224)
ToTensor
Normalize(ImageNet mean/std)
```

---

## DataLoaders

Для создания загрузчиков используется функция:

```python
create_dataloaders(...)
```

Она создаёт:

```python
train_loader, val_loader
```

`train_loader` использует:

```
shuffle=True или WeightedRandomSampler
```

`val_loader` использует:

```
shuffle=False
```

### Пример использования:

```python
train_loader, val_loader = create_dataloaders(
    batch_size=32,
    num_workers=2,
    image_size=224
)
```

По умолчанию функция берёт:

```
data/processed/train_df.csv
data/processed/val_df.csv
data/raw/train_images/
data/raw/val_images/
```

Если в processed CSV есть колонка `image_path`, `RoomTypeDataset` использует её
и читает путь относительно `data/raw/`. Поэтому один `train_df.csv` может
содержать изображения и из `train_images/`, и из `heuristics_images/`.

Если нужно включить балансировку классов для train:

```python
train_loader, val_loader = create_dataloaders(
    batch_size=32,
    num_workers=2,
    image_size=224,
    use_weighted_sampling=True
)
```

В этом случае для train используется `WeightedRandomSampler`.
Для validation балансировка не применяется.
После добавления recommended heuristics классы `кабинет` и `гардеробная`
добираются до среднего размера, поэтому weighted sampling остаётся опциональным
экспериментом, а не обязательной частью pipeline.

Для test используется отдельная функция:

```python
test_loader = create_test_dataloader(
    batch_size=32,
    num_workers=2,
    image_size=224
)
```

`test_loader` возвращает:

```python
images, image_ids, item_ids
```

---

```python
# В реальном обучении используется цикл по DataLoader:
# for images, targets in train_loader:
#     ...

# Здесь берём один batch для проверки
images, targets = next(iter(train_loader))

print(images.shape)
print(targets.shape)
```

Ожидаемый результат:

```
torch.Size([32, 3, 224, 224])
torch.Size([32])
```

---

## Контракт для моделей

Любая модель при обучении и валидации должна работать с:

```python
images, targets = batch
```

Формат:

```
images  — tensor [batch_size, 3, 224, 224]
targets — tensor [batch_size]
```

Модель должна возвращать:

```
outputs — tensor [batch_size, 19]
```

Для test используется другой формат batch:

```python
images, image_ids, item_ids = batch
```

В test нет `targets`, потому что в `test_df.csv` нет колонки `result`.

---

## Пример проверки на ResNet18

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

num_classes = 19
in_features = model.fc.in_features

model.fc = nn.Linear(in_features, num_classes)

outputs = model(images)

print(outputs.shape)
```

Ожидаемый результат:

```
torch.Size([32, 19])
```

---

## Metric

Используется метрика:

```
Macro F1
```

Функция:

```python
calculate_macro_f1(y_true, y_pred)
```

---

## Хранение метрик обучения

Метрики обучения нужно сохранять в git, чтобы сравнивать эксперименты
и видеть, какая модель лучше. Для отчётов используется папка:

```text
reports/metrics/
```

Каждая модель может хранить отчёт в удобном для неё формате: JSON, CSV или
несколько файлов. Главное — складывать эти отчёты в подпапку своей модели.

Например:

```text
reports/metrics/resnet18/resnet18_metrics.json
reports/metrics/efficientnet/model_comparison.csv
```

Хранение весов моделей и других тяжёлых артефактов (`*.pt`, `*.pth`,
Grad-CAM изображения, промежуточные outputs) настраивается отдельно.

---

## Важно

- Не менять train/val split  
- Не использовать test_df.csv в обучении  
- Все модели используют общий data pipeline  
- Для YOLOv8 будет отдельная адаптация  

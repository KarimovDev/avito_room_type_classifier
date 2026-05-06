set dotenv-load := true

export UV_CACHE_DIR := ".uv-cache"
PYTHON_VERSION := "3.12.8"
PYTORCH_PIP := "uv pip"

# Показать список доступных команд
default:
    @just --list

# Установить нужную версию Python через uv
setup:
    uv python install {{PYTHON_VERSION}}

# Пересоздать виртуальное окружение проекта
recreate-venv: setup
    uv venv --python {{PYTHON_VERSION}} --clear

# Установить базовые зависимости для работы с data pipeline
install: install-data

# Установить зависимости для Dataset, DataLoader, transforms, metrics и preprocessing
install-data: setup
    uv sync --group data

# Установить зависимости для Streamlit-приложения
install-streamlit: setup
    uv sync --only-group streamlit

# Установить зависимости для обучения EfficientNet
install-efficientnet: setup
    uv sync --group efficientnet

# Установить зависимости для обучения ResNet18
install-resnet18: setup
    uv sync --group resnet18

# Установить зависимости для EfficientNet и интерпретации результатов
install-interpretability: setup
    uv sync --group efficientnet --group interpretability

# Установить зависимости для YOLO
install-yolo: setup
    uv sync --group yolo

# Установить все группы зависимостей проекта
install-all: setup
    uv sync --all-groups

# Запустить preprocessing: raw CSV -> processed CSV
prepare-data:
    uv run --group data python -m src.preprocess_data

# Запустить preprocessing с рекомендуемыми heuristics: кабинет и гардеробная
prepare-data-with-heuristics:
    uv run --group data python -m src.preprocess_data --include-heuristics recommended

# Запустить preprocessing с выбранными heuristics
# Пример: just prepare-data-heuristics cabinet,dressing_room
prepare-data-heuristics HEURISTICS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}}

# Запустить preprocessing с выбранными heuristics и лимитом на каждый источник
# Пример: just prepare-data-heuristics-limited cabinet,dressing_room 500
prepare-data-heuristics-limited HEURISTICS MAX_ROWS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}} --max-heuristics-per-source {{MAX_ROWS}}

# Обновить lockfile при необходимости
lock:
    uv lock

#
# - cpu: https://download.pytorch.org/whl/cpu
# - cu130: https://download.pytorch.org/whl/cu130
# Переустановить torch/torchvision из обычного PyPI
pytorch-pypi:
    {{PYTORCH_PIP}} install --upgrade --reinstall torch torchvision

# Переустановить CPU-версию torch/torchvision
pytorch-cpu:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cpu" torch torchvision

# Переустановить CUDA 13.0 версию torch/torchvision
pytorch-cu130:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cu130" torch torchvision

# Запустить YOLO demo/inference
run-yolo:
    uv run --group yolo python models/yolo/main_yolo.py

# Запустить обучение EfficientNet
train-efficientnet:
    uv run --group efficientnet python models/efficientNet/train_efficientnet.py

# Запустить обучение ResNet18
train-resnet18 EPOCHS="30":
    uv run --group resnet18 python models/resnet18/train_resnet18.py --epochs {{EPOCHS}}

# Повторить лучший зафиксированный запуск ResNet18: class weights + без weighted sampler
train-resnet18-best EPOCHS="30" SEED="42":
    uv run --group resnet18 python models/resnet18/train_resnet18.py --epochs {{EPOCHS}} --seed {{SEED}} --no-weighted-sampling

# Построить Grad-CAM для EfficientNet
grad-cam-efficientnet:
    uv run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py

# Запустить Streamlit-приложение
run-streamlit:
    uv run --group streamlit --group efficientnet --group yolo streamlit run streamlit/app.py

# Выполнить произвольную команду через uv
# Пример: just run "python -V"
run *ARGS:
    uv run {{ARGS}}

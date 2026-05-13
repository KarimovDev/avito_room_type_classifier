set dotenv-load := true

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

PYTHON_VERSION := "3.12.8"
PYTORCH_PIP := "uv pip"

# Показать список доступных команд
default:
    @just --list

# Установить нужную версию Python через uv
setup:
    uv python install {{PYTHON_VERSION}}

# Пересоздать локальное виртуальное окружение
recreate-venv: setup
    uv venv --python {{PYTHON_VERSION}} --clear

# Базовая установка проекта
install: install-data

# Зависимости для preprocessing и общих даталоадеров
install-data: setup
    uv sync --group data

# Зависимости для Streamlit-приложения
install-streamlit: setup
    uv sync --only-group streamlit

# Зависимости для обучения EfficientNet
install-efficientnet: setup
    uv sync --group efficientnet

# Зависимости для обучения ResNet18
install-resnet18: setup
    uv sync --group resnet18

# Зависимости для обучения ResNet50
install-resnet50: setup
    uv sync --group resnet50

# Зависимости для обучения DenseNet121
install-densenet121: setup
    uv sync --group densenet121

# EfficientNet плюс библиотеки для Grad-CAM
install-interpretability: setup
    uv sync --group efficientnet --group interpretability

# Зависимости для YOLO-скрипта
install-yolo: setup
    uv sync --group yolo

# Зависимости для ConvNeXt Nano
install-convnext-nano: setup
    uv sync --group convnext_nano

# Старое имя команды для ConvNeXt Nano
install-convnext_nano: install-convnext-nano

# Зависимости для ConvNeXt Tiny
install-convnext-tiny: setup
    uv sync --group convnext_tiny

# Установить все группы зависимостей
install-all: setup
    uv sync --all-groups

# Подготовить processed CSV из raw-данных
prepare-data:
    uv run --group data python -m src.preprocess_data

# Подготовить данные с рекомендованными эвристиками
prepare-data-with-heuristics:
    uv run --group data python -m src.preprocess_data --include-heuristics recommended

# Подготовить данные с выбранными эвристиками
prepare-data-heuristics HEURISTICS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}}

# Подготовить данные с ограничением строк на эвристику
prepare-data-heuristics-limited HEURISTICS MAX_ROWS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}} --max-heuristics-per-source {{MAX_ROWS}}

# Обновить uv.lock после правок pyproject.toml
lock:
    uv lock

# Переустановить torch/torchvision из обычного PyPI
pytorch-pypi:
    {{PYTORCH_PIP}} install --upgrade --reinstall torch torchvision

# Переустановить CPU-версию torch/torchvision
pytorch-cpu:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cpu" torch torchvision

# Переустановить CUDA 13.0-версию torch/torchvision
pytorch-cu130:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cu130" torch torchvision

# Запустить YOLO demo/inference
run-yolo:
    uv run --group yolo python models/yolo/main_yolo.py

# Локальное обучение

# Обучить EfficientNet
train-efficientnet EPOCHS="30" BATCH="32":
    uv run --group efficientnet python -m models.efficientNet.train_efficientnet \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Обучить ResNet50
train-resnet50 EPOCHS="15" BATCH="32":
    uv run --group resnet50 python -m models.resnet50.resnet50 \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Обучить ResNet18
train-resnet18 EPOCHS="30":
    uv run --group resnet18 python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}}

# Обучить ResNet18 без weighted sampler
train-resnet18-best EPOCHS="30" SEED="42":
    uv run --group resnet18 python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}} --seed {{SEED}} --no-weighted-sampling

# Обучить DenseNet121 по трем этапам
train-densenet121 STAGE1="2" STAGE2="8" STAGE3="5" BATCH="32":
    uv run --group densenet121 python -m models.densenet121.train_densenet121 \
      --epochs-stage1 {{STAGE1}} --epochs-stage2 {{STAGE2}} --epochs-stage3 {{STAGE3}} --batch-size {{BATCH}}

# Обучить ConvNeXt Nano
train-convnext-nano EPOCHS="30" BATCH="32":
    uv run --group convnext_nano python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Старое короткое имя для обучения ConvNeXt Nano
train-convnext EPOCHS="30" BATCH="32":
    uv run --group convnext_nano python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Старое имя с подчеркиванием для ConvNeXt Nano
train-convnext_nano EPOCHS="30" BATCH="32":
    uv run --group convnext_nano python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Обучить ConvNeXt Tiny по JSON-конфигу
train-convnext-tiny CONFIG="models/convnext_tiny/train_config.json":
    uv run --group convnext_tiny python -m models.convnext_tiny.train_convnext_tiny --config {{CONFIG}}

# Построить Grad-CAM для EfficientNet
grad-cam-efficientnet:
    uv run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py

# Запустить Streamlit-приложение
run-streamlit:
    uv run --group streamlit --group efficientnet --group yolo streamlit run streamlit/app.py

# Запустить произвольную команду через uv
run *ARGS:
    uv run {{ARGS}}

# Docker

# Собрать Docker-образы
docker-build:
    docker compose build

# Проверить доступность GPU внутри Docker
docker-check-gpu:
    docker compose run --rm base python -c "exec('import torch\nprint(\"torch:\", torch.__version__)\nprint(\"CUDA:\", torch.cuda.is_available())\nprint(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')"

# Обучить DenseNet121 в Docker
docker-train-densenet121 EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-densenet121

# Обучить ResNet18 в Docker
docker-train-resnet18 EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-resnet18

# Обучить ResNet50 в Docker
docker-train-resnet50 EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-resnet50

# Обучить EfficientNet в Docker
docker-train-efficientnet EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-efficientnet

# Обучить ConvNeXt Nano в Docker
docker-train-convnext EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-convnext

# Запустить YOLO в Docker
docker-run-yolo:
    docker compose run --rm yolo

# Построить Grad-CAM в Docker
docker-gradcam:
    docker compose run --rm gradcam

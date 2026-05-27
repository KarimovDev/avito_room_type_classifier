set dotenv-load := true

set windows-shell := ["bash.exe", "-uc"]

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

# Зависимости для MLflow и DagsHub
install-tracking: setup
    uv sync --group tracking

# Авторизоваться в DagsHub для remote MLflow
dagshub-login:
    uv run --group tracking dagshub login

# Показать страницу общих экспериментов DagsHub
dagshub-experiments:
    @echo "https://dagshub.com/YashinSergey/room_type_classifier/experiments"

# Зависимости для Streamlit
install-streamlit: setup
    uv sync --group streamlit --group yolo --group efficientnet --group resnet18 --group resnet50 --group densenet121 --group convnext_nano --group convnext_tiny

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

# Проверить формат метрик и чекпоинтов
check-training-outputs:
    uv run --group data python -m src.validate_training_outputs --allow-empty-checkpoints

# Построить таблицу сравнения моделей из MLflow
compare-models:
    uv run --group tracking python -m src.compare_mlflow_models

# Скачать лучшие чекпоинты из MLflow (DagsHub) в outputs/models для Streamlit
pull-checkpoints *ARGS:
    uv run --group tracking python -m src.pull_mlflow_checkpoints {{ARGS}}

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
    uv run --group yolo --group tracking python -m models.yolo.main_yolo

# Обучить EfficientNet B0
train-efficientnet-b0 EPOCHS="80" BATCH="32" EARLY_STOPPING_PATIENCE="8" LR_SCHEDULER="plateau":
    uv run --group efficientnet --group tracking python -m models.efficientNet.train_efficientnet \
      --variant b0 --epochs {{EPOCHS}} --batch-size {{BATCH}} --early-stopping-patience {{EARLY_STOPPING_PATIENCE}} --lr-scheduler {{LR_SCHEDULER}}

# Обучить EfficientNet B1
train-efficientnet-b1 EPOCHS="80" BATCH="32" IMAGE_SIZE="240" EARLY_STOPPING_PATIENCE="8" LR_SCHEDULER="plateau":
    uv run --group efficientnet --group tracking python -m models.efficientNet.train_efficientnet \
      --variant b1 --epochs {{EPOCHS}} --batch-size {{BATCH}} --image-size {{IMAGE_SIZE}} --early-stopping-patience {{EARLY_STOPPING_PATIENCE}} --lr-scheduler {{LR_SCHEDULER}}

# Обучить EfficientNet B0 с замороженным backbone (только classifier)
train-efficientnet-b0-freeze-backbone EPOCHS="80" BATCH="32" EARLY_STOPPING_PATIENCE="8" LR_SCHEDULER="plateau":
    uv run --group efficientnet --group tracking python -m models.efficientNet.train_efficientnet \
      --variant b0 --epochs {{EPOCHS}} --batch-size {{BATCH}} --early-stopping-patience {{EARLY_STOPPING_PATIENCE}} --lr-scheduler {{LR_SCHEDULER}} --freeze-backbone

# Обучить EfficientNet B1 с замороженным backbone (только classifier)
train-efficientnet-b1-freeze-backbone EPOCHS="80" BATCH="32" IMAGE_SIZE="240" EARLY_STOPPING_PATIENCE="8" LR_SCHEDULER="plateau":
    uv run --group efficientnet --group tracking python -m models.efficientNet.train_efficientnet \
      --variant b1 --epochs {{EPOCHS}} --batch-size {{BATCH}} --image-size {{IMAGE_SIZE}} --early-stopping-patience {{EARLY_STOPPING_PATIENCE}} --lr-scheduler {{LR_SCHEDULER}} --freeze-backbone

# Старое короткое имя для EfficientNet B0
train-efficientnet EPOCHS="30" BATCH="32":
    just train-efficientnet-b0 {{EPOCHS}} {{BATCH}}

# Обучить ResNet50
train-resnet50 EPOCHS="15" BATCH="32":
    uv run --group resnet50 --group tracking python -m models.resnet50.resnet50 \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Обучить ResNet18
train-resnet18 EPOCHS="30":
    uv run --group resnet18 --group tracking python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}}

# Обучить ResNet18 без weighted sampler
train-resnet18-best EPOCHS="30" SEED="42":
    uv run --group resnet18 --group tracking python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}} --seed {{SEED}} --no-weighted-sampling

# Обучить DenseNet121 по трем этапам
train-densenet121 STAGE1="2" STAGE2="8" STAGE3="5" BATCH="32":
    uv run --group densenet121 --group tracking python -m models.densenet121.train_densenet121 \
      --epochs-stage1 {{STAGE1}} --epochs-stage2 {{STAGE2}} --epochs-stage3 {{STAGE3}} --batch-size {{BATCH}}

# Обучить ConvNeXt Nano
train-convnext-nano EPOCHS="30" BATCH="32":
    uv run --group convnext_nano --group tracking python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Старое короткое имя для обучения ConvNeXt Nano
train-convnext EPOCHS="30" BATCH="32":
    uv run --group convnext_nano --group tracking python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Старое имя с подчеркиванием для ConvNeXt Nano
train-convnext_nano EPOCHS="30" BATCH="32":
    uv run --group convnext_nano --group tracking python -m models.convnext_nano.train_convnext \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

# Обучить ConvNeXt Tiny по JSON-конфигу
train-convnext-tiny CONFIG="models/convnext_tiny/train_config.json":
    uv run --group convnext_tiny --group tracking python -m models.convnext_tiny.train_convnext_tiny --config {{CONFIG}}

# Проверить компактный ансамбль на validation и отправить эксперимент в DagsHub MLflow
eval-ensemble:
    uv run --group convnext_nano --group resnet50 --group resnet18 --group tracking python -m src.evaluate_ensemble \
      --weighting val_f1 --run-name ensemble_convnext_nano_resnet50_resnet18 --log-mlflow --no-mlflow-local

# Сделать submission ансамблем ConvNeXt Nano + ResNet50 + ResNet18 в data/submissions
make-submission:
    uv run --group convnext_nano --group resnet50 --group resnet18 python -m src.infer_ensemble

# Открыть локальный MLflow UI в fallback-режиме
mlflow-ui:
    uv run --group tracking mlflow ui --backend-store-uri sqlite:///mlflow.db

# Построить Grad-CAM для EfficientNet
grad-cam-efficientnet:
    uv run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py --sample-index 0

# Запустить Streamlit-приложение
run-streamlit:
    uv run --group streamlit --group yolo --group efficientnet --group resnet18 --group resnet50 --group densenet121 --group convnext_nano --group convnext_tiny streamlit run streamlit/app.py

# Запустить произвольную команду через uv
run *ARGS:
    @if [ "{{ARGS}}" = "yolo" ]; then \
        just run-yolo; \
    else \
        uv run {{ARGS}}; \
    fi

# Docker

# Собрать Docker-образы
docker-build:
    docker compose build

# Собрать Docker-образ Streamlit
docker-build-streamlit:
    docker build -f streamlit/Dockerfile -t room-type-classifier-streamlit .

# Запустить Streamlit в Docker
docker-run-streamlit:
    docker run --rm -p 8501:8501 room-type-classifier-streamlit

# Проверить доступность GPU внутри Docker
docker-check-gpu:
    @if docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm base python -c "exec('import torch\nprint(\"torch:\", torch.__version__)\nprint(\"CUDA:\", torch.cuda.is_available())\nprint(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')" 2>/dev/null; then \
        true; \
    else \
        docker compose run --rm base python -c "exec('import torch\nprint(\"torch:\", torch.__version__)\nprint(\"CUDA:\", torch.cuda.is_available())\nprint(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')"; \
    fi

# Запустить Docker-сервис с GPU, если CUDA доступна внутри контейнера
_docker-compose-run SERVICE *ARGS:
    @if docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm base python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then \
        echo "Docker CUDA доступна, запускаем с GPU"; \
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm {{ARGS}} {{SERVICE}}; \
    else \
        echo "Docker CUDA недоступна, запускаем на CPU"; \
        docker compose run --rm {{ARGS}} {{SERVICE}}; \
    fi

# Обучить DenseNet121 в Docker
docker-train-densenet121 STAGE1="2" STAGE2="8" STAGE3="5" BATCH="32":
    just _docker-compose-run train-densenet121 -e STAGE1={{STAGE1}} -e STAGE2={{STAGE2}} -e STAGE3={{STAGE3}} -e BATCH={{BATCH}}

# Обучить ResNet18 в Docker с лучшими параметрами
docker-train-resnet18 EPOCHS="30" BATCH="32" SEED="42":
    just _docker-compose-run train-resnet18 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} -e SEED={{SEED}}

# Обучить ResNet50 в Docker
docker-train-resnet50 EPOCHS="15" BATCH="32":
    just _docker-compose-run train-resnet50 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Обучить EfficientNet в Docker
docker-train-efficientnet EPOCHS="30" BATCH="32":
    just _docker-compose-run train-efficientnet -e VARIANT=b0 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Обучить EfficientNet B0 в Docker
docker-train-efficientnet-b0 EPOCHS="30" BATCH="32":
    just _docker-compose-run train-efficientnet -e VARIANT=b0 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Обучить EfficientNet B1 в Docker
docker-train-efficientnet-b1 EPOCHS="30" BATCH="32":
    just _docker-compose-run train-efficientnet -e VARIANT=b1 -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Обучить ConvNeXt Nano в Docker
docker-train-convnext EPOCHS="30" BATCH="32":
    just _docker-compose-run train-convnext -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}}

# Обучить ConvNeXt Nano в Docker
docker-train-convnext-nano EPOCHS="30" BATCH="32":
    just docker-train-convnext {{EPOCHS}} {{BATCH}}

# Обучить ConvNeXt Tiny в Docker
docker-train-convnext-tiny CONFIG="models/convnext_tiny/train_config.json":
    just _docker-compose-run train-convnext-tiny -e CONFIG={{CONFIG}}

# Запустить YOLO в Docker
docker-run-yolo:
    just _docker-compose-run yolo

# Запустить YOLO в Docker и залогировать inference-run
docker-train-yolo:
    just docker-run-yolo

# Построить Grad-CAM в Docker
docker-gradcam:
    docker compose run --rm gradcam

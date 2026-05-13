set dotenv-load := true

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

PYTHON_VERSION := "3.12.8"
PYTORCH_PIP := "uv pip"

default:
    @just --list

setup:
    uv python install {{PYTHON_VERSION}}

recreate-venv: setup
    uv venv --python {{PYTHON_VERSION}} --clear

install: install-data

install-data: setup
    uv sync --group data

install-streamlit: setup
    uv sync --only-group streamlit

install-efficientnet: setup
    uv sync --group efficientnet

install-resnet18: setup
    uv sync --group resnet18

install-resnet50: setup
    uv sync --group resnet50

install-interpretability: setup
    uv sync --group efficientnet --group interpretability

install-yolo: setup
    uv sync --group yolo

install-convnext_nano: setup
    uv sync --group convnext_nano

install-all: setup
    uv sync --all-groups

prepare-data:
    uv run --group data python -m src.preprocess_data

prepare-data-with-heuristics:
    uv run --group data python -m src.preprocess_data --include-heuristics recommended

prepare-data-heuristics HEURISTICS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}}

prepare-data-heuristics-limited HEURISTICS MAX_ROWS:
    uv run --group data python -m src.preprocess_data --include-heuristics {{HEURISTICS}} --max-heuristics-per-source {{MAX_ROWS}}

lock:
    uv lock

pytorch-pypi:
    {{PYTORCH_PIP}} install --upgrade --reinstall torch torchvision

pytorch-cpu:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cpu" torch torchvision

pytorch-cu130:
    {{PYTORCH_PIP}} install --upgrade --reinstall --index-url "https://download.pytorch.org/whl/cu130" torch torchvision

run-yolo:
    uv run --group yolo python models/yolo/main_yolo.py

# ── Локальное обучение ──────────────────────────────────────────────

train-efficientnet EPOCHS="30" BATCH="32":
    uv run --group efficientnet python -m models.efficientNet.train_efficientnet \
      --epochs {{EPOCHS}} --batch-size {{BATCH}}

train-resnet50:
    uv run --group resnet50 python models/resnet50/resnet50.py

train-resnet18 EPOCHS="30":
    uv run --group resnet18 python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}}

train-resnet18-best EPOCHS="30" SEED="42":
    uv run --group resnet18 python -m models.resnet18.train_resnet18 --epochs {{EPOCHS}} --seed {{SEED}} --no-weighted-sampling

train-densenet121 EPOCHS="30":
    uv run --group data python -m models.densenet121.train_densenet121 --epochs {{EPOCHS}}

train-convnext EPOCHS="30":
    uv run --group data python -m models.convnext_nano.train_convnext --epochs {{EPOCHS}}

train-convnext_nano EPOCHS="30":
    uv run --group convnext_nano python models/convnext_nano/train_convnext.py --epochs {{EPOCHS}}

grad-cam-efficientnet:
    uv run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py

run-streamlit:
    uv run --group streamlit --group efficientnet --group yolo streamlit run streamlit/app.py

run *ARGS:
    uv run {{ARGS}}

# ── Docker ──────────────────────────────────────────────────────────

docker-build:
    docker compose build

docker-check-gpu:
    docker compose run --rm base python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

docker-train-densenet121 EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-densenet121

docker-train-resnet18 EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-resnet18

docker-train-resnet50 EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-resnet50

docker-train-efficientnet EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-efficientnet

docker-train-convnext EPOCHS="30" BATCH="32":
    docker compose run --rm -e EPOCHS={{EPOCHS}} -e BATCH={{BATCH}} train-convnext

docker-run-yolo:
    docker compose run --rm yolo

docker-gradcam:
    docker compose run --rm gradcam

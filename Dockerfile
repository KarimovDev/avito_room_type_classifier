FROM python:3.12-slim

# Устанавливаем системные зависимости для OpenCV, PIL и ML-библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Копируем зависимости первыми для кэша Docker
COPY pyproject.toml uv.lock ./

# Ставим группы, которые нужны docker-compose сервисам
RUN uv sync \
    --group data \
    --group densenet121 \
    --group efficientnet \
    --group resnet18 \
    --group resnet50 \
    --group interpretability \
    --group convnext_nano \
    --group convnext_tiny \
    --group yolo \
    --no-install-project \
    --frozen

# Добавляем .venv/bin в PATH, python и все пакеты берутся из виртуального окружения
ENV PATH="/app/.venv/bin:$PATH"

# Копируем остальной исходный код
COPY . .

# Команда по умолчанию переопределяется аргументами docker run
CMD ["python", "-m", "models.densenet121.train_densenet121"]

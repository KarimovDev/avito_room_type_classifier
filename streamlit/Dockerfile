FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

COPY pyproject.toml uv.lock ./

RUN uv venv --python 3.12 \
    && uv sync --frozen \
      --group streamlit \
      --group yolo \
      --group efficientnet \
      --group resnet18 \
      --group resnet50 \
      --group densenet121 \
      --group convnext_nano \
      --group convnext_tiny

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py"]

# Streamlit service

Сервис загружает одно или несколько изображений, позволяет выбрать модели и выводит предсказанный тип комнаты с вероятностью для каждой выбранной модели.

## Локальный запуск

```bash
just install-streamlit
just run-streamlit
```

## Docker

Dockerfile рассчитан на сборку из корня репозитория:

```bash
just docker-build-streamlit
just docker-run-streamlit
```

Если нужно добавить Streamlit в `docker-compose.yml`, можно использовать такую секцию:

```yaml
services:
  streamlit:
    build:
      context: .
      dockerfile: streamlit/Dockerfile
    ports:
      - "8501:8501"
    environment:
      STREAMLIT_ALLOW_MODEL_DOWNLOAD: "1"
```

Docker-образ ставит `streamlit` и группы всех моделей, которые нужны приложению для inference.

Проектная конфигурация Streamlit лежит в `.streamlit/config.toml`: отключены file watcher, run-on-save, prompt email и usage stats. Порт можно переопределить через переменную окружения `STREAMLIT_SERVER_PORT`.

## Модели

Сервис показывает только реальные доступные модели. Если checkpoint или вес не найден, соответствующая модель будет отключена в сайдбаре.

В сравнении участвуют YOLO, EfficientNet B0/B1, ResNet18, ResNet50, DenseNet121, ConvNeXt Nano и ConvNeXt Tiny.

Для `YOLO scene classifier` используется внешний pretrained вес:

```text
models/yolo/downloads/keremberke/yolov8m-scene-classification/best.pt
```

Чтобы разрешить автозагрузку YOLO из Hugging Face при старте inference:

```bash
STREAMLIT_ALLOW_MODEL_DOWNLOAD=1 just run-streamlit
```

Для EfficientNet по умолчанию используются checkpoints:

```text
outputs/models/efficientnet/efficientnet_b0_best.pt
outputs/models/efficientnet/efficientnet_b1_best.pt
```

Пути можно переопределить через `EFFICIENTNET_B0_CHECKPOINT_PATH` и
`EFFICIENTNET_B1_CHECKPOINT_PATH`.

Для ResNet по умолчанию используются checkpoints:

```text
outputs/models/resnet18/resnet18_best.pt
outputs/models/resnet50/resnet50_best.pt
```

Для ConvNeXt Nano по умолчанию используется checkpoint:

```text
outputs/models/convnext_nano/convnext_nano_best.pt
```

Для ConvNeXt Tiny по умолчанию используется checkpoint:

```text
outputs/models/convnext_tiny/convnext_tiny_best.pt
```

Для DenseNet121 по умолчанию используется checkpoint:

```text
outputs/models/densenet121/densenet121_best.pt
```

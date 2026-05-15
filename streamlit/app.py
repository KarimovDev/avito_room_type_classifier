from __future__ import annotations

import io
import os
import sys
import timm
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch.nn as nn
from torchvision import models

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.convnext_tiny.model import build_convnext_tiny
from models.densenet121.densenet121 import build_densenet121
from src.device import get_default_device
from src.labels import load_label_mapping
from src.transforms import get_val_transforms
from models.resnet18.resnet18 import build_resnet18


# Пути/настройки моделей. Их можно переопределять через переменные окружения
YOLO_MODEL_PATH = ROOT_DIR / "models" / "yolo" / "downloads" / "keremberke" / "yolov8m-scene-classification" / "best.pt"
YOLO_REPO_ID = "keremberke/yolov8m-scene-classification"
YOLO_FILENAME = "best.pt"
EFFICIENTNET_B0_CHECKPOINT_PATH = Path(
    os.getenv(
        "EFFICIENTNET_B0_CHECKPOINT_PATH",
        ROOT_DIR / "models" / "efficientNet" / "artifacts" / "efficientnet_b0_best.pt",
    )
)
EFFICIENTNET_B1_CHECKPOINT_PATH = Path(
    os.getenv(
        "EFFICIENTNET_B1_CHECKPOINT_PATH",
        ROOT_DIR / "models" / "efficientNet" / "artifacts" / "efficientnet_b1_best.pt",
    )
)
PREVIEW_WIDTH_PX = 300
RESNET50_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "resnet50" / "resnet50_best.pt"
RESNET18_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "resnet18" / "resnet18_best.pt"
DENSENET121_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "densenet121" / "densenet121_best.pt"
CONVNEXT_NANO_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "convnext_nano" / "convnext_nano_best.pt"
CONVNEXT_TINY_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "convnext_tiny" / "convnext_tiny_best.pt"


@dataclass(frozen=True)
class ModelConfig:
    """Описание одной модели для сайдбара и запуска предсказания."""

    key: str
    title: str
    description: str
    predictor: Callable[[bytes], tuple[str, float]]
    is_available: Callable[[], bool]


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    """Открываем картинку из байтов и приводим к RGB.

    `exif_transpose` нужен, чтобы фото с телефона не было боком.
    """
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


@st.cache_data(show_spinner=False)
def load_room_type_labels() -> dict[int, str]:
    """Словарь: id класса -> лейбл из csv."""
    return load_label_mapping()


@st.cache_resource(show_spinner="Загружаем YOLO модель...") # Кэшируем модель YOLO
def load_yolo_model() -> object | None:
    """Загружаем YOLO модель (локально или с HuggingFace, если разрешено).
    """
    if not YOLO_MODEL_PATH.exists():
        # Без явного флага мы не скачиваем веса автоматически
        if os.getenv("STREAMLIT_ALLOW_MODEL_DOWNLOAD") != "1":
            return None

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        # Скачиваем best.pt в папку models/
        hf_hub_download(
            repo_id=YOLO_REPO_ID,
            filename=YOLO_FILENAME,
            local_dir=YOLO_MODEL_PATH.parent,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

    try:
        from ultralyticsplus import YOLO
    except ImportError:
        return None

    # Создаём объект модели и выставляем порог уверенности
    model = YOLO(YOLO_MODEL_PATH)
    model.overrides["conf"] = 0.25
    return model


@st.cache_data(show_spinner=False)
def _predict_yolo_cached(image_bytes: bytes) -> tuple[str, float] | None:
    """Предсказание YOLO с кэшированием (чтобы не считать одинаковое несколько раз)."""
    model = load_yolo_model()
    if model is None:
        return None

    try:
        from ultralyticsplus import postprocess_classify_output
    except ImportError:
        return None

    # YOLO принимает PIL.Image, поэтому сначала декодируем байты
    image = load_rgb_image(image_bytes)
    results = model.predict(image)
    processed_result = postprocess_classify_output(model, result=results[0])
    # Берём класс с максимальной вероятностью
    prediction, probability = max(processed_result.items(), key=lambda item: item[1])
    return prediction, float(probability)


def yolo_predict(image_bytes: bytes) -> tuple[str, float]:
    """Враппер над кэшированным предсказанием: либо возвращаем результат, либо падаем с ошибкой."""
    prediction = _predict_yolo_cached(image_bytes)
    if prediction is not None:
        return prediction
    raise RuntimeError("YOLO model is unavailable. Install yolo dependencies and provide best.pt.")


def build_efficientnet_model(variant: str, num_classes: int) -> object:
    """Собираем EfficientNet (без предобученных весов) и меняем последний слой под число классов."""
    from torch import nn
    from torchvision.models import efficientnet_b0, efficientnet_b1

    builders = {
        "b0": efficientnet_b0,
        "b1": efficientnet_b1,
    }
    model = builders.get(variant, efficientnet_b0)(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_default_image_size(variant: str) -> int:
    """Размер картинки по умолчанию для варианта модели. Если не передано в чекпоинте."""
    return 240 if variant == "b1" else 224


@st.cache_resource(show_spinner="Загружаем EfficientNet...") # Кэшируем модель EfficientNet
def load_efficientnet_model(checkpoint_path: str) -> tuple[object, object, int] | None:
    """Загружаем EfficientNet из чекпоинта (веса + параметры) и переводим в режим eval()."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    # Автовыбор устройства: CUDA -> MPS -> CPU
    device = get_default_device()
    # Чекпоинт хранит: веса модели и несколько параметров (variant/num_classes/image_size)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    variant = checkpoint.get("variant", "b0")
    num_classes = int(checkpoint.get("num_classes", 20))
    image_size = int(checkpoint.get("image_size", get_default_image_size(variant)))
    model = build_efficientnet_model(variant, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, image_size


@st.cache_data(show_spinner=False)
def _predict_efficientnet_cached(image_bytes: bytes, checkpoint_path: str) -> tuple[str, float] | None:
    """Предсказание EfficientNet с кэшированием.

    Ключ кэша включает путь к чекпоинту, чтобы разные модели не смешивались.
    """
    loaded_model = load_efficientnet_model(checkpoint_path)
    if loaded_model is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    model, device, image_size = loaded_model
    # Трансформации должны совпадать с теми, что были при обучении/валидации
    # Берем из общего модуля src/transforms.py
    preprocess = get_val_transforms(image_size=image_size)
    image = load_rgb_image(image_bytes)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        # Softmax превращает logits в вероятности по классам
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability, class_index = torch.max(probabilities, dim=0)

    labels = load_room_type_labels()
    class_id = int(class_index.item())
    prediction = labels.get(class_id, f"class_{class_id}")
    return prediction, float(probability.item())


def efficientnet_predict(image_bytes: bytes, checkpoint_path: Path) -> tuple[str, float]:
    """Враппер: либо возвращаем предсказание, либо сообщаем, что чекпоинт недоступен."""
    prediction = _predict_efficientnet_cached(image_bytes, str(checkpoint_path))
    if prediction is not None:
        return prediction
    raise RuntimeError(f"EfficientNet checkpoint is unavailable: {checkpoint_path}")


def build_resnet50_model(num_classes):
    model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

@st.cache_resource(show_spinner="Загружаем ResNet50...")
def load_resnet50_model(checkpoint_path: str) -> tuple[object, object, int] | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    device = get_default_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", 20))
        image_size = int(checkpoint.get("image_size", 224))
    else:
        state_dict = checkpoint
        num_classes = state_dict["fc.weight"].shape[0]
        image_size = 224

    model = build_resnet50_model(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, image_size


@st.cache_data(show_spinner=False)
def _predict_resnet50_cached(image_bytes: bytes, checkpoint_path: str) -> tuple[str, float] | None:
    """Предсказание resnet50 с кэшированием.

    Ключ кэша включает путь к чекпоинту, чтобы разные модели не смешивались.
    """
    loaded_model = load_resnet50_model(checkpoint_path)
    if loaded_model is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    model, device, image_size = loaded_model
    # Трансформации должны совпадать с теми, что были при обучении/валидации
    # Берем из общего модуля src/transforms.py
    preprocess = get_val_transforms(image_size=image_size)
    image = load_rgb_image(image_bytes)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        # Softmax превращает logits в вероятности по классам
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability, class_index = torch.max(probabilities, dim=0)

    labels = load_room_type_labels()
    class_id = int(class_index.item())
    prediction = labels.get(class_id, f"class_{class_id}")
    return prediction, float(probability.item())


def resnet50_predict(image_bytes: bytes, checkpoint_path: Path) -> tuple[str, float]:
    """Враппер: либо возвращаем предсказание, либо сообщаем, что чекпоинт недоступен."""
    prediction = _predict_resnet50_cached(image_bytes, str(checkpoint_path))
    if prediction is not None:
        return prediction
    raise RuntimeError(f"ResNet50 checkpoint is unavailable: {checkpoint_path}")


@st.cache_resource(show_spinner="Загружаем ResNet18...")
def load_resnet18_model(checkpoint_path: str) -> tuple[object, object, int] | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    device = get_default_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", 20))
        image_size = int(checkpoint.get("image_size", 224))
    else:
        state_dict = checkpoint
        num_classes = state_dict["fc.weight"].shape[0]
        image_size = 224

    model = build_resnet18(num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, image_size

@st.cache_data(show_spinner=False)
def _predict_resnet18_cached(image_bytes: bytes, checkpoint_path: str) -> tuple[str, float] | None:
    """Предсказание resnet18 с кэшированием.

    Ключ кэша включает путь к чекпоинту, чтобы разные модели не смешивались.
    """
    loaded_model = load_resnet18_model(checkpoint_path)
    if loaded_model is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    model, device, image_size = loaded_model
    # Трансформации должны совпадать с теми, что были при обучении/валидации
    # Берем из общего модуля src/transforms.py
    preprocess = get_val_transforms(image_size=image_size)
    image = load_rgb_image(image_bytes)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        # Softmax превращает logits в вероятности по классам
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability, class_index = torch.max(probabilities, dim=0)

    labels = load_room_type_labels()
    class_id = int(class_index.item())
    prediction = labels.get(class_id, f"class_{class_id}")
    return prediction, float(probability.item())

def resnet18_predict(image_bytes: bytes, checkpoint_path: Path) -> tuple[str, float]:
    """Враппер: либо возвращаем предсказание, либо сообщаем, что чекпоинт недоступен."""
    prediction = _predict_resnet18_cached(image_bytes, str(checkpoint_path))
    if prediction is not None:
        return prediction
    raise RuntimeError(f"ResNet18 checkpoint is unavailable: {checkpoint_path}")


@st.cache_resource(show_spinner="Загружаем DenseNet121...")
def load_densenet121_model(checkpoint_path: str) -> tuple[object, object, int] | None:
    """Загружаем DenseNet121 из checkpoint"""
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    device = get_default_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return None

    state_dict = checkpoint["model_state_dict"]
    num_classes = int(checkpoint.get("num_classes", len(load_room_type_labels())))
    image_size = int(checkpoint.get("image_size", 224))

    model = build_densenet121(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, image_size


@st.cache_data(show_spinner=False)
def _predict_densenet121_cached(image_bytes: bytes, checkpoint_path: str) -> tuple[str, float] | None:
    """Предсказание DenseNet121 с кэшированием"""
    loaded_model = load_densenet121_model(checkpoint_path)
    if loaded_model is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    model, device, image_size = loaded_model
    preprocess = get_val_transforms(image_size=image_size)
    image = load_rgb_image(image_bytes)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability, class_index = torch.max(probabilities, dim=0)

    labels = load_room_type_labels()
    class_id = int(class_index.item())
    prediction = labels.get(class_id, f"class_{class_id}")
    return prediction, float(probability.item())


def densenet121_predict(image_bytes: bytes, checkpoint_path: Path) -> tuple[str, float]:
    """Враппер для DenseNet121"""
    prediction = _predict_densenet121_cached(image_bytes, str(checkpoint_path))
    if prediction is not None:
        return prediction
    raise RuntimeError(f"DenseNet121 checkpoint is unavailable: {checkpoint_path}")


def num_classes_convnext_nano(state_dict: dict) -> int:
    """Число классов из весов головы timm ConvNeXt (plain state_dict без метаданных)."""
    if "head.fc.weight" in state_dict:
        return int(state_dict["head.fc.weight"].shape[0])
    for key in ("head.weight", "classifier.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError(
        "Не удалось определить num_classes: нет ключей head.fc.weight / head.weight / classifier.weight"
    )


@st.cache_resource(show_spinner="Загружаем convnext nano...")
def load_convnext_nano_model(checkpoint_path: str) -> tuple[object, object, int] | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    device = get_default_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        image_size = int(checkpoint.get("image_size", 224))
        if "num_classes" in checkpoint:
            num_classes = int(checkpoint["num_classes"])
        else:
            num_classes = num_classes_convnext_nano(state_dict)
    else:
        state_dict = checkpoint
        image_size = 224
        num_classes = num_classes_convnext_nano(state_dict)

    model = timm.create_model(
        'convnext_nano',
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.5,
        drop_path_rate=0.3)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, image_size

@st.cache_data(show_spinner=False)
def _predict_convnext_nano_cached(image_bytes: bytes, checkpoint_path: str) -> tuple[str, float] | None:
    """Предсказание convnext nano с кэшированием.

    Ключ кэша включает путь к чекпоинту, чтобы разные модели не смешивались.
    """
    loaded_model = load_convnext_nano_model(checkpoint_path)
    if loaded_model is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    model, device, image_size = loaded_model
    # Трансформации должны совпадать с теми, что были при обучении/валидации
    # Берем из общего модуля src/transforms.py
    preprocess = get_val_transforms(image_size=image_size)
    image = load_rgb_image(image_bytes)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        # Softmax превращает logits в вероятности по классам
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability, class_index = torch.max(probabilities, dim=0)

    labels = load_room_type_labels()
    class_id = int(class_index.item())
    prediction = labels.get(class_id, f"class_{class_id}")
    return prediction, float(probability.item())

def convnext_nano_predict(image_bytes: bytes, checkpoint_path: Path) -> tuple[str, float]:
    """Враппер: либо возвращаем предсказание, либо сообщаем, что чекпоинт недоступен."""
    prediction = _predict_convnext_nano_cached(image_bytes, str(checkpoint_path))
    if prediction is not None:
        return prediction
    raise RuntimeError(f"Convnext Nano checkpoint is unavailable: {checkpoint_path}")


@st.cache_resource(show_spinner="Загружаем convnext tiny...")
def load_convnext_tiny_model(checkpoint_path: str) -> tuple[object, object, int] | None:
    """Загружаем ConvNeXt Tiny из checkpoint"""
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    device = get_default_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return None

    state_dict = checkpoint["model_state_dict"]
    num_classes = int(checkpoint.get("num_classes", len(load_room_type_labels())))
    image_size = int(checkpoint.get("image_size", 224))

    model = build_convnext_tiny(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, image_size


@st.cache_data(show_spinner=False)
def _predict_convnext_tiny_cached(image_bytes: bytes, checkpoint_path: str) -> tuple[str, float] | None:
    """Предсказание convnext tiny с кэшированием"""
    loaded_model = load_convnext_tiny_model(checkpoint_path)
    if loaded_model is None:
        return None

    try:
        import torch
    except ImportError:
        return None

    model, device, image_size = loaded_model
    preprocess = get_val_transforms(image_size=image_size)
    image = load_rgb_image(image_bytes)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        probability, class_index = torch.max(probabilities, dim=0)

    labels = load_room_type_labels()
    class_id = int(class_index.item())
    prediction = labels.get(class_id, f"class_{class_id}")
    return prediction, float(probability.item())


def convnext_tiny_predict(image_bytes: bytes, checkpoint_path: Path) -> tuple[str, float]:
    """Враппер для ConvNeXt Tiny"""
    prediction = _predict_convnext_tiny_cached(image_bytes, str(checkpoint_path))
    if prediction is not None:
        return prediction
    raise RuntimeError(f"Convnext Tiny checkpoint is unavailable: {checkpoint_path}")


MODELS = [
    # Список моделей, которые можно включать/выключать в сайдбаре
    ModelConfig(
        key="yolo_scene_classifier",
        title="YOLO scene classifier",
        description="Внешний pretrained YOLO scene classifier.",
        predictor=yolo_predict,
        is_available=YOLO_MODEL_PATH.exists,
    ),
    ModelConfig(
        key="efficientnet_b0",
        title="EfficientNet B0",
        description="Обученный EfficientNet-B0 checkpoint на локальном датасете.",
        predictor=lambda image_bytes: efficientnet_predict(image_bytes, EFFICIENTNET_B0_CHECKPOINT_PATH),
        is_available=EFFICIENTNET_B0_CHECKPOINT_PATH.exists,
    ),
    ModelConfig(
        key="efficientnet_b1",
        title="EfficientNet B1",
        description="Обученный EfficientNet-B1 checkpoint на локальном датасете.",
        predictor=lambda image_bytes: efficientnet_predict(image_bytes, EFFICIENTNET_B1_CHECKPOINT_PATH),
        is_available=EFFICIENTNET_B1_CHECKPOINT_PATH.exists,
    ),
    ModelConfig(
        key="resnet50",
        title="ResNet50",
        description="Обученный ResNet50 checkpoint на локальном датасете.",
        predictor=lambda image_bytes: resnet50_predict(image_bytes, RESNET50_MODEL_PATH),
        is_available=RESNET50_MODEL_PATH.exists,
    ),
    ModelConfig(
        key="resnet18",
        title="ResNet18",
        description="Обученный ResNet18 checkpoint на локальном датасете.",
        predictor=lambda image_bytes: resnet18_predict(image_bytes, RESNET18_MODEL_PATH),
        is_available=RESNET18_MODEL_PATH.exists,
    ),
    ModelConfig(
        key="densenet121",
        title="DenseNet121",
        description="Обученный DenseNet121 checkpoint на локальном датасете.",
        predictor=lambda image_bytes: densenet121_predict(image_bytes, DENSENET121_MODEL_PATH),
        is_available=DENSENET121_MODEL_PATH.exists,
    ),
    ModelConfig(
        key="convnext_nano",
        title="ConvNext Nano",
        description="Обученный ConvNext Nano checkpoint на локальном датасете.",
        predictor=lambda image_bytes: convnext_nano_predict(image_bytes, CONVNEXT_NANO_MODEL_PATH),
        is_available=CONVNEXT_NANO_MODEL_PATH.exists,
    ),
    ModelConfig(
        key="convnext_tiny",
        title="ConvNext Tiny",
        description="Обученный ConvNext Tiny checkpoint на локальном датасете.",
        predictor=lambda image_bytes: convnext_tiny_predict(image_bytes, CONVNEXT_TINY_MODEL_PATH),
        is_available=CONVNEXT_TINY_MODEL_PATH.exists,
    ),
]


def configure_page() -> None:
    """Базовые настройки страницы Streamlit."""
    st.set_page_config(
        page_title="Room Type Classifier",
        layout="wide",
    )


def render_sidebar() -> list[ModelConfig]:
    """Отображаем сайдбар и возвращаем список выбранных моделей."""
    st.sidebar.header("Модели")
    selected_models = []
    for model in MODELS:
        is_available = model.is_available()
        if st.sidebar.checkbox(model.title, value=is_available, disabled=not is_available, help=model.description):
            selected_models.append(model)

    st.sidebar.divider()
    if YOLO_MODEL_PATH.exists():
        st.sidebar.success("YOLO best.pt найден локально")
    else:
        st.sidebar.info(
            "YOLO best.pt не найден. Для автозагрузки задайте STREAMLIT_ALLOW_MODEL_DOWNLOAD=1"
        )
    if EFFICIENTNET_B0_CHECKPOINT_PATH.exists():
        st.sidebar.success("EfficientNet B0 checkpoint найден")
    else:
        st.sidebar.info("EfficientNet B0 checkpoint не найден")
    if EFFICIENTNET_B1_CHECKPOINT_PATH.exists():
        st.sidebar.success("EfficientNet B1 checkpoint найден")
    else:
        st.sidebar.info("EfficientNet B1 checkpoint не найден")
    if RESNET50_MODEL_PATH.exists():
        st.sidebar.success("ResNet50 checkpoint найден")
    else:
        st.sidebar.info("ResNet50 checkpoint не найден")
    if RESNET18_MODEL_PATH.exists():
        st.sidebar.success("ResNet18 checkpoint найден")
    else:
        st.sidebar.info("ResNet18 checkpoint не найден")
    if DENSENET121_MODEL_PATH.exists():
        st.sidebar.success("DenseNet121 checkpoint найден")
    else:
        st.sidebar.info("DenseNet121 checkpoint не найден")
    if CONVNEXT_NANO_MODEL_PATH.exists():
        st.sidebar.success("ConvNext Nano checkpoint найден")
    else:
        st.sidebar.info("ConvNext Nano checkpoint не найден")
    if CONVNEXT_TINY_MODEL_PATH.exists():
        st.sidebar.success("ConvNext Tiny checkpoint найден")
    else:
        st.sidebar.info("ConvNext Tiny checkpoint не найден")

    return selected_models


def render_results(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile], selected_models: list[ModelConfig]) -> None:
    """Запускаем распознавание по всем картинкам и отображаем таблицу результатов."""
    rows = []
    progress = st.progress(0, text="Подготавливаем изображения...")
    total_steps = len(uploaded_files) * len(selected_models)
    completed_steps = 0

    for image_index, uploaded_file in enumerate(uploaded_files, start=1):
        image_bytes = uploaded_file.getvalue()
        for model in selected_models:
            prediction, probability = model.predictor(image_bytes)
            rows.append(
                {
                    "Изображение": uploaded_file.name or f"image_{image_index}",
                    "Номер": image_index,
                    "Модель": model.title,
                    "Предсказание": prediction,
                    "Вероятность": round(probability, 3),
                }
            )
            completed_steps += 1
            progress.progress(
                completed_steps / total_steps,
                text=f"Распознаем: {uploaded_file.name or image_index}",
            )

    progress.empty()

    results = pd.DataFrame(rows)
    st.subheader("Результаты")
    # Таблица с прогресс-баром по вероятности
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Вероятность": st.column_config.ProgressColumn(
                "Вероятность",
                min_value=0,
                max_value=1,
                format="%.3f",
            )
        },
    )

    with st.expander("Загруженные изображения", expanded=False):
        # Показываем превью загруженных картинок в несколько колонок
        columns = st.columns(min(3, len(uploaded_files)))
        for index, uploaded_file in enumerate(uploaded_files):
            with columns[index % len(columns)]:
                st.image(
                    uploaded_file.getvalue(), 
                    caption=uploaded_file.name, 
                    width=PREVIEW_WIDTH_PX,
                    #use_container_width=True
                )


def main() -> None:
    """Точка входа Streamlit-приложения."""
    configure_page()

    st.title("Room Type Classifier")
    st.caption("Загрузка изображений и сравнение результатов выбранных моделей.")

    # Пользователь выбирает модели в сайдбаре
    selected_models = render_sidebar()
    # Пользователь загружает изображения
    uploaded_files = st.file_uploader(
        "Изображения",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    # Кнопка запуска распознавания
    recognize = st.button(
        "Распознать",
        type="primary",
        disabled=not uploaded_files or not selected_models,
        use_container_width=False,
    )

    if not uploaded_files:
        st.info("Загрузите одно или несколько изображений")
        return

    if not selected_models:
        st.warning("Выберите хотя бы одну модель")
        return

    if recognize:
        render_results(uploaded_files, selected_models)


if __name__ == "__main__":
    main()

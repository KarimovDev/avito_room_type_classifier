from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Создает ResNet18 и заменяет последний слой

    Args:
        num_classes: Количество классов
        pretrained: Использовать ли веса ImageNet для дообучения
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    # Предобученная ResNet18 уже умеет извлекать общие признаки изображений
    model = resnet18(weights=weights)

    # Меняем последний слой
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


__all__ = ["build_resnet18"]

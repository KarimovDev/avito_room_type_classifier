from __future__ import annotations

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Создает DenseNet121 и заменяет classifier под нужное количество классов

    Args:
        num_classes: Количество классов
        pretrained: Использовать ли веса ImageNet для дообучения
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    # classifier
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


__all__ = ["build_densenet121"]

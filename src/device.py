from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Выбираем устройство для PyTorch: CUDA -> MPS -> CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

import torch.nn as nn
import json
from typing import List, Dict, Any


def create_fc_model(
        input_size: int,
        num_classes: int,
        hidden_sizes: List[int],
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
        activation: str = 'relu'
) -> nn.Module:
    """
    Создает полносвязную модель с заданными параметрами

    Args:
        input_size: размер входного слоя
        num_classes: количество классов
        hidden_sizes: список размеров скрытых слоев
        use_dropout: использовать ли Dropout
        dropout_rate: коэффициент Dropout
        use_batch_norm: использовать ли BatchNorm
        activation: функция активации ('relu', 'tanh', 'sigmoid')

    Returns:
        nn.Module: созданная модель
    """
    layers = []
    prev_size = input_size

    # Словарь функций активации
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }

    # Создаем скрытые слои
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))

        layers.append(activations.get(activation, nn.ReLU()))

        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        prev_size = hidden_size

    # Выходной слой
    layers.append(nn.Linear(prev_size, num_classes))

    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    """Подсчитывает количество обучаемых параметров в модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_config(
        hidden_sizes: List[int],
        use_dropout: bool,
        dropout_rate: float,
        use_batch_norm: bool,
        activation: str,
        filepath: str
):
    """Сохраняет конфигурацию модели в JSON"""
    config = {
        'hidden_sizes': hidden_sizes,
        'use_dropout': use_dropout,
        'dropout_rate': dropout_rate,
        'use_batch_norm': use_batch_norm,
        'activation': activation
    }
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def load_model_config(filepath: str) -> Dict[str, Any]:
    """Загружает конфигурацию модели из JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


class ModelWrapper(nn.Module):
    """Обертка для модели с методом flatten входа"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

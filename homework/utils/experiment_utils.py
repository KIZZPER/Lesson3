import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import json
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:
    """
    Обучение модели на одной эпохе

    Args:
        model: модель для обучения
        train_loader: DataLoader с обучающими данными
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство (cpu/cuda)

    Returns:
        tuple: (средние потери, точность)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Статистика
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """
    Оценка модели на тестовых данных

    Args:
        model: модель для оценки
        test_loader: DataLoader с тестовыми данными
        criterion: функция потерь
        device: устройство (cpu/cuda)

    Returns:
        tuple: (средние потери, точность)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
        weight_decay: float = 0.0,
        verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Полное обучение модели с отслеживанием метрик

    Args:
        model: модель для обучения
        train_loader: DataLoader с обучающими данными
        test_loader: DataLoader с тестовыми данными
        epochs: количество эпох
        learning_rate: скорость обучения
        device: устройство (если None, автоматически определяется)
        weight_decay: коэффициент L2 регуляризации
        verbose: выводить ли информацию о процессе

    Returns:
        dict: словарь с историей обучения (потери и точность)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }

    for epoch in range(epochs):
        start_time = time.time()

        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Оценка
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)

        if verbose:
            logger.info(
                f'Epoch {epoch + 1}/{epochs} | '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
                f'Time: {epoch_time:.2f}s'
            )

    return history


def run_experiment(
        model_fn: callable,
        train_loader: DataLoader,
        test_loader: DataLoader,
        experiment_name: str,
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        save_path: Optional[str] = None,
        device: Optional[torch.device] = None
) -> Dict[str, any]:
    """
    Запуск полного эксперимента с моделью

    Args:
        model_fn: функция, возвращающая модель
        train_loader: DataLoader с обучающими данными
        test_loader: DataLoader с тестовыми данными
        experiment_name: название эксперимента
        epochs: количество эпох
        learning_rate: скорость обучения
        weight_decay: коэффициент L2 регуляризации
        save_path: путь для сохранения результатов
        device: устройство для обучения

    Returns:
        dict: результаты эксперимента
    """
    logger.info(f'Starting experiment: {experiment_name}')

    # Создание модели
    model = model_fn()

    # Подсчет параметров
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params:,}')

    # Обучение
    start_time = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        weight_decay=weight_decay,
        verbose=True
    )
    total_time = time.time() - start_time

    # Финальные метрики
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    best_test_acc = max(history['test_acc'])

    results = {
        'experiment_name': experiment_name,
        'num_parameters': num_params,
        'history': history,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'best_test_acc': best_test_acc,
        'total_time': total_time,
        'avg_epoch_time': np.mean(history['epoch_times'])
    }

    logger.info(f'Experiment completed: {experiment_name}')
    logger.info(f'Final Train Accuracy: {final_train_acc:.2f}%')
    logger.info(f'Final Test Accuracy: {final_test_acc:.2f}%')
    logger.info(f'Best Test Accuracy: {best_test_acc:.2f}%')
    logger.info(f'Total Time: {total_time:.2f}s')

    # Сохранение результатов
    if save_path:
        save_results(results, save_path)

    return results


def compare_experiments(
        results_list: List[Dict],
        metric: str = 'final_test_acc'
) -> None:
    """
    Сравнение результатов нескольких экспериментов

    Args:
        results_list: список словарей с результатами экспериментов
        metric: метрика для сравнения
    """
    logger.info('\n' + '=' * 80)
    logger.info('COMPARISON OF EXPERIMENTS')
    logger.info('=' * 80)

    # Сортировка по метрике
    sorted_results = sorted(results_list, key=lambda x: x.get(metric, 0), reverse=True)

    for i, result in enumerate(sorted_results, 1):
        logger.info(f'\n{i}. {result["experiment_name"]}')
        logger.info(f'   Parameters: {result["num_parameters"]:,}')
        logger.info(f'   Final Test Acc: {result["final_test_acc"]:.2f}%')
        logger.info(f'   Best Test Acc: {result["best_test_acc"]:.2f}%')
        logger.info(f'   Training Time: {result["total_time"]:.2f}s')


def save_results(results: Dict, filepath: str) -> None:
    """Сохранение результатов эксперимента"""

    # Преобразование numpy типов в Python типы для JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    # Создание директории если не существует
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    logger.info(f'Results saved to {filepath}')


def load_results(filepath: str) -> Dict:
    """Загрузка результатов эксперимента"""

    with open(filepath, 'r') as f:
        return json.load(f)




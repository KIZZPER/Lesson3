import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch.nn as nn

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Визуализация истории обучения (потери и точность)

    Args:
        history: словарь с историей обучения
        title: заголовок графика
        save_path: путь для сохранения графика
        figsize: размер фигуры
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # График потерь
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # График точности
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_multiple_experiments(
        results_list: List[Dict],
        metric: str = 'test_acc',
        title: str = "Comparison of Experiments",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Сравнение нескольких экспериментов на одном графике

    Args:
        results_list: список словарей с результатами
        metric: метрика для отображения ('test_acc', 'train_acc', 'test_loss', 'train_loss')
        title: заголовок графика
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    plt.figure(figsize=figsize)

    for result in results_list:
        history = result['history']
        epochs = range(1, len(history[metric]) + 1)
        label = result['experiment_name']

        plt.plot(epochs, history[metric], marker='o', linewidth=2,
                 markersize=4, label=label, alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)

    if 'acc' in metric:
        plt.ylabel('Accuracy (%)', fontsize=12)
    else:
        plt.ylabel('Loss', fontsize=12)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_overfitting_analysis(
        history: Dict[str, List[float]],
        title: str = "Overfitting Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Анализ переобучения: разница между train и test метриками

    Args:
        history: история обучения
        title: заголовок графика
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    epochs = range(1, len(history['train_acc']) + 1)

    # Разница между train и test accuracy
    gap = np.array(history['train_acc']) - np.array(history['test_acc'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # График точностей
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax1.fill_between(epochs, history['train_acc'], history['test_acc'],
                     alpha=0.2, color='orange')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Train vs Test Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # График разницы (overfitting gap)
    ax2.plot(epochs, gap, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red',
                     label='Overfitting')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Gap (%)', fontsize=12)
    ax2.set_title('Overfitting Gap (Train - Test)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_comparison_bar(
        results_list: List[Dict],
        metrics: List[str] = ['final_test_acc', 'best_test_acc'],
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Столбчатая диаграмма для сравнения моделей

    Args:
        results_list: список результатов экспериментов
        metrics: список метрик для отображения
        title: заголовок
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    names = [r['experiment_name'] for r in results_list]
    x = np.arange(len(names))
    width = 0.35 if len(metrics) == 2 else 0.25

    fig, ax = plt.subplots(figsize=figsize)

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        offset = width * (i - len(metrics) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(),
                      alpha=0.8)

        # Добавление значений на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_parameter_vs_accuracy(
        results_list: List[Dict],
        title: str = "Parameters vs Accuracy",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Зависимость точности от количества параметров

    Args:
        results_list: список результатов
        title: заголовок
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    params = [r['num_parameters'] for r in results_list]
    accuracies = [r['final_test_acc'] for r in results_list]
    names = [r['experiment_name'] for r in results_list]

    plt.figure(figsize=figsize)

    scatter = plt.scatter(params, accuracies, s=200, alpha=0.6,
                          c=range(len(results_list)), cmap='viridis')

    # Добавление подписей
    for i, name in enumerate(names):
        plt.annotate(name, (params[i], accuracies[i]),
                     fontsize=9, ha='right', va='bottom')

    plt.xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Experiment Index')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_heatmap(
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str = "Heatmap",
        xlabel: str = "X",
        ylabel: str = "Y",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'YlOrRd',
        annot: bool = True
) -> None:
    """
    Тепловая карта для визуализации результатов grid search

    Args:
        data: 2D массив с данными
        x_labels: подписи для оси X
        y_labels: подписи для оси Y
        title: заголовок
        xlabel: название оси X
        ylabel: название оси Y
        save_path: путь для сохранения
        figsize: размер фигуры
        cmap: цветовая схема
        annot: показывать ли значения в ячейках
    """
    plt.figure(figsize=figsize)

    sns.heatmap(data, annot=annot, fmt='.2f', cmap=cmap,
                xticklabels=x_labels, yticklabels=y_labels,
                cbar_kws={'label': 'Accuracy (%)'}, linewidths=0.5)

    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_training_time_comparison(
        results_list: List[Dict],
        title: str = "Training Time Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Сравнение времени обучения моделей

    Args:
        results_list: список результатов
        title: заголовок
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    names = [r['experiment_name'] for r in results_list]
    times = [r['total_time'] for r in results_list]
    avg_epoch_times = [r['avg_epoch_time'] for r in results_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width / 2, times, width, label='Total Time', alpha=0.8)
    bars2 = ax.bar(x + width / 2, avg_epoch_times, width,
                   label='Avg Epoch Time', alpha=0.8)

    # Добавление значений
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_weight_distribution(
        model: nn.Module,
        title: str = "Weight Distribution",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Визуализация распределения весов модели

    Args:
        model: модель PyTorch
        title: заголовок
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    weights = []
    layer_names = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weights.append(param.detach().cpu().numpy().flatten())
            layer_names.append(name)

    num_layers = len(weights)
    cols = 3
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_layers > 1 else [axes]

    for i, (weight, name) in enumerate(zip(weights, layer_names)):
        ax = axes[i]
        ax.hist(weight, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}\nMean: {weight.mean():.4f}, Std: {weight.std():.4f}',
                     fontsize=10)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Скрытие пустых subplot
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


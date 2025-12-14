import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
import numpy as np

# Добавляем пути для импорта
sys.path.append(str(Path(__file__).parent))

from utils.model_utils import create_fc_model, count_parameters
from utils.experiment_utils import run_experiment, compare_experiments
from utils.visualization_utils import (
    plot_training_history,
    plot_multiple_experiments,
    plot_comparison_bar,
    plot_parameter_vs_accuracy,
    plot_training_time_comparison,
    plot_heatmap
)


def prepare_mnist_data(batch_size=64):
    """
    Подготовка датасета MNIST

    Args:
        batch_size: размер батча

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def prepare_cifar10_data(batch_size=64):
    """
    Подготовка датасета CIFAR-10

    Args:
        batch_size: размер батча

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def experiment_2_1_width_comparison():
    """
    Задание 2.1: Сравнение моделей разной ширины
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2.1: Width Comparison")
    print("=" * 80 + "\n")

    # Подготовка данных
    train_loader, test_loader = prepare_mnist_data(batch_size=128)

    # Конфигурации моделей с разной шириной
    model_configs = [
        {
            'name': 'narrow_network',
            'hidden_sizes': [64, 32, 16],
            'description': 'Narrow network [64, 32, 16]'
        },
        {
            'name': 'medium_network',
            'hidden_sizes': [256, 128, 64],
            'description': 'Medium network [256, 128, 64]'
        },
        {
            'name': 'wide_network',
            'hidden_sizes': [1024, 512, 256],
            'description': 'Wide network [1024, 512, 256]'
        },
        {
            'name': 'very_wide_network',
            'hidden_sizes': [2048, 1024, 512],
            'description': 'Very wide network [2048, 1024, 512]'
        }
    ]

    results_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Запуск экспериментов для каждой конфигурации
    for config in model_configs:
        print(f"\nTraining: {config['description']}")
        print("-" * 80)

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=config['hidden_sizes'],
                use_dropout=False,
                use_batch_norm=False,
                activation='relu'
            )
            return nn.Sequential(nn.Flatten(), model)

        results = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=config['name'],
            epochs=15,
            learning_rate=0.001,
            device=device,
            save_path=f'results/width_experiments/{config["name"]}.json'
        )

        results_list.append(results)

        # Визуализация для каждой модели
        plot_training_history(
            results['history'],
            title=f"Training History: {config['description']}",
            save_path=f'plots/width_experiments/{config["name"]}_history.png'
        )

    # Сравнение всех моделей
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL WIDTH CONFIGURATIONS")
    print("=" * 80)
    compare_experiments(results_list, metric='final_test_acc')

    # Визуализация сравнений
    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Test Accuracy Comparison: Different Widths',
        save_path='plots/width_experiments/all_widths_comparison.png'
    )

    plot_comparison_bar(
        results_list,
        metrics=['final_test_acc', 'best_test_acc'],
        title='Model Width Comparison',
        save_path='plots/width_experiments/width_bar_comparison.png'
    )

    plot_parameter_vs_accuracy(
        results_list,
        title='Parameters vs Accuracy: Width Experiments',
        save_path='plots/width_experiments/params_vs_accuracy.png'
    )

    plot_training_time_comparison(
        results_list,
        title='Training Time: Different Widths',
        save_path='plots/width_experiments/training_time_comparison.png'
    )

    return results_list


def experiment_2_2_architecture_optimization():
    """
    Задание 2.2: Оптимизация архитектуры
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2.2: Architecture Optimization")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Различные схемы архитектур
    architecture_patterns = [
        {
            'name': 'expanding_pattern',
            'hidden_sizes': [64, 128, 256],
            'description': 'Expanding [64 → 128 → 256]'
        },
        {
            'name': 'narrowing_pattern',
            'hidden_sizes': [256, 128, 64],
            'description': 'Narrowing [256 → 128 → 64]'
        },
        {
            'name': 'constant_pattern',
            'hidden_sizes': [128, 128, 128],
            'description': 'Constant [128 → 128 → 128]'
        },
        {
            'name': 'bottleneck_pattern',
            'hidden_sizes': [256, 64, 256],
            'description': 'Bottleneck [256 → 64 → 256]'
        },
        {
            'name': 'pyramid_pattern',
            'hidden_sizes': [512, 256, 128, 64],
            'description': 'Pyramid [512 → 256 → 128 → 64]'
        },
        {
            'name': 'inverted_pyramid',
            'hidden_sizes': [64, 128, 256, 512],
            'description': 'Inverted Pyramid [64 → 128 → 256 → 512]'
        }
    ]

    results_list = []

    for config in architecture_patterns:
        print(f"\nTraining: {config['description']}")
        print("-" * 80)

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=config['hidden_sizes'],
                use_dropout=False,
                use_batch_norm=False,
                activation='relu'
            )
            return nn.Sequential(nn.Flatten(), model)

        results = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=config['name'],
            epochs=15,
            learning_rate=0.001,
            device=device,
            save_path=f'results/width_experiments/{config["name"]}.json'
        )

        results_list.append(results)

        # Визуализация
        plot_training_history(
            results['history'],
            title=f"Training: {config['description']}",
            save_path=f'plots/width_experiments/{config["name"]}_history.png'
        )

    # Сравнение паттернов
    print("\n" + "=" * 80)
    print("COMPARISON OF ARCHITECTURE PATTERNS")
    print("=" * 80)
    compare_experiments(results_list, metric='final_test_acc')

    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Architecture Patterns Comparison',
        save_path='plots/width_experiments/patterns_comparison.png'
    )

    plot_comparison_bar(
        results_list,
        metrics=['final_test_acc', 'best_test_acc'],
        title='Architecture Patterns Performance',
        save_path='plots/width_experiments/patterns_bar_comparison.png'
    )

    return results_list


def experiment_grid_search():
    """
    Дополнительный эксперимент: Grid Search для первого и второго слоя

    Проводим поиск оптимальной комбинации ширины для двух слоев
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL EXPERIMENT: Grid Search for Layer Widths")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Диапазоны для grid search
    first_layer_sizes = [64, 128, 256, 512]
    second_layer_sizes = [32, 64, 128, 256]

    # Матрица для результатов
    accuracy_matrix = np.zeros((len(first_layer_sizes), len(second_layer_sizes)))

    results_grid = []

    for i, first_size in enumerate(first_layer_sizes):
        for j, second_size in enumerate(second_layer_sizes):
            print(f"\nTesting: [{first_size}, {second_size}]")
            print("-" * 40)

            def model_fn():
                model = create_fc_model(
                    input_size=784,
                    num_classes=10,
                    hidden_sizes=[first_size, second_size],
                    use_dropout=False,
                    use_batch_norm=False,
                    activation='relu'
                )
                return nn.Sequential(nn.Flatten(), model)

            results = run_experiment(
                model_fn=model_fn,
                train_loader=train_loader,
                test_loader=test_loader,
                experiment_name=f'grid_{first_size}_{second_size}',
                epochs=10,
                learning_rate=0.001,
                device=device,
                save_path=f'results/width_experiments/grid_{first_size}_{second_size}.json'
            )

            accuracy_matrix[i, j] = results['final_test_acc']
            results_grid.append(results)

    # Визуализация heatmap
    plot_heatmap(
        data=accuracy_matrix,
        x_labels=[str(s) for s in second_layer_sizes],
        y_labels=[str(s) for s in first_layer_sizes],
        title='Grid Search: Layer Width Combinations',
        xlabel='Second Layer Size',
        ylabel='First Layer Size',
        save_path='plots/width_experiments/grid_search_heatmap.png',
        figsize=(10, 8)
    )

    # Найти лучшую комбинацию
    best_idx = np.unravel_index(accuracy_matrix.argmax(), accuracy_matrix.shape)
    best_first = first_layer_sizes[best_idx[0]]
    best_second = second_layer_sizes[best_idx[1]]
    best_accuracy = accuracy_matrix[best_idx]

    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)
    print(f"Best combination: [{best_first}, {best_second}]")
    print(f"Best accuracy: {best_accuracy:.2f}%")

    return results_grid, accuracy_matrix


def experiment_cifar10_width():
    """
    Дополнительный эксперимент: тестирование на CIFAR-10

    Проверяем как ширина влияет на более сложном датасете
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL EXPERIMENT: Width on CIFAR-10")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_cifar10_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Конфигурации для CIFAR-10 (нужны более широкие сети)
    cifar_configs = [
        {
            'name': 'cifar_medium',
            'hidden_sizes': [512, 256, 128],
            'description': 'Medium [512, 256, 128]'
        },
        {
            'name': 'cifar_wide',
            'hidden_sizes': [1024, 512, 256],
            'description': 'Wide [1024, 512, 256]'
        },
        {
            'name': 'cifar_very_wide',
            'hidden_sizes': [2048, 1024, 512],
            'description': 'Very Wide [2048, 1024, 512]'
        }
    ]

    results_list = []

    for config in cifar_configs:
        print(f"\nTraining on CIFAR-10: {config['description']}")
        print("-" * 80)

        def model_fn():
            model = create_fc_model(
                input_size=3072,  # 32x32x3 для CIFAR-10
                num_classes=10,
                hidden_sizes=config['hidden_sizes'],
                use_dropout=True,  # Dropout для более сложного датасета
                dropout_rate=0.3,
                use_batch_norm=True,  # BatchNorm помогает на CIFAR
                activation='relu'
            )
            return nn.Sequential(nn.Flatten(), model)

        results = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=config['name'],
            epochs=20,
            learning_rate=0.001,
            device=device,
            save_path=f'results/width_experiments/{config["name"]}_cifar.json'
        )

        results_list.append(results)

        plot_training_history(
            results['history'],
            title=f"CIFAR-10: {config['description']}",
            save_path=f'plots/width_experiments/{config["name"]}_cifar_history.png'
        )

    # Сравнение на CIFAR-10
    print("\n" + "=" * 80)
    print("CIFAR-10 WIDTH COMPARISON")
    print("=" * 80)
    compare_experiments(results_list, metric='final_test_acc')

    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='CIFAR-10: Width Comparison',
        save_path='plots/width_experiments/cifar_width_comparison.png'
    )

    return results_list


def main():
    """Главная функция для запуска всех экспериментов по ширине"""

    print("\n" + "#" * 80)
    print("# HOMEWORK: WIDTH EXPERIMENTS")
    print("#" * 80 + "\n")

    # Создание директорий для результатов
    Path('results/width_experiments').mkdir(parents=True, exist_ok=True)
    Path('plots/width_experiments').mkdir(parents=True, exist_ok=True)

    # Эксперимент 2.1: Сравнение разных ширин
    results_2_1 = experiment_2_1_width_comparison()

    # Эксперимент 2.2: Оптимизация архитектуры
    results_2_2 = experiment_2_2_architecture_optimization()

    # Дополнительный эксперимент: Grid Search
    results_grid, accuracy_matrix = experiment_grid_search()

    # Дополнительный эксперимент: CIFAR-10
    results_cifar = experiment_cifar10_width()

    print("\n" + "#" * 80)
    print("# ALL WIDTH EXPERIMENTS COMPLETED!")
    print("#" * 80 + "\n")

    # Финальный анализ
    print("Summary of findings:")
    print("-" * 80)

    # Лучшая модель по ширине (MNIST)
    best_width = max(results_2_1, key=lambda x: x['final_test_acc'])
    print(f"Best width configuration (MNIST): {best_width['experiment_name']}")
    print(f"  Test Accuracy: {best_width['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_width['num_parameters']:,}")

    # Лучший паттерн архитектуры
    best_pattern = max(results_2_2, key=lambda x: x['final_test_acc'])
    print(f"\nBest architecture pattern: {best_pattern['experiment_name']}")
    print(f"  Test Accuracy: {best_pattern['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_pattern['num_parameters']:,}")

    # Результаты на CIFAR-10
    best_cifar = max(results_cifar, key=lambda x: x['final_test_acc'])
    print(f"\nBest configuration on CIFAR-10: {best_cifar['experiment_name']}")
    print(f"  Test Accuracy: {best_cifar['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_cifar['num_parameters']:,}")

    # Grid Search результаты
    print(f"\nGrid Search optimal combination found")
    print(f"  See heatmap in plots/width_experiments/grid_search_heatmap.png")


if __name__ == "__main__":
    main()

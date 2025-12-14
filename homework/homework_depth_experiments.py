import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path

# Пути для импорта
sys.path.append(str(Path(__file__).parent))

from utils.model_utils import create_fc_model
from utils.experiment_utils import run_experiment, compare_experiments
from utils.visualization_utils import (
    plot_training_history,
    plot_multiple_experiments,
    plot_overfitting_analysis,
    plot_comparison_bar,
    plot_parameter_vs_accuracy,
    plot_training_time_comparison
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


def experiment_1_1_depth_comparison():
    """
    Задание 1.1: Сравнение моделей разной глубины
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.1: Depth Comparison")
    print("=" * 80 + "\n")

    # Подготовка данных
    train_loader, test_loader = prepare_mnist_data(batch_size=128)

    # Конфигурации моделей с разной глубиной
    model_configs = [
        {
            'name': '1_layer_linear',
            'hidden_sizes': [],  # Только выходной слой
            'description': 'Linear classifier (no hidden layers)'
        },
        {
            'name': '2_layers_1_hidden',
            'hidden_sizes': [128],
            'description': '1 hidden layer (128 units)'
        },
        {
            'name': '3_layers_2_hidden',
            'hidden_sizes': [256, 128],
            'description': '2 hidden layers (256, 128)'
        },
        {
            'name': '5_layers_4_hidden',
            'hidden_sizes': [512, 256, 128, 64],
            'description': '4 hidden layers (512, 256, 128, 64)'
        },
        {
            'name': '7_layers_6_hidden',
            'hidden_sizes': [512, 512, 256, 256, 128, 64],
            'description': '6 hidden layers (512, 512, 256, 256, 128, 64)'
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
            save_path=f'results/depth_experiments/{config["name"]}.json'
        )

        results_list.append(results)

        # Визуализация для каждой модели
        plot_training_history(
            results['history'],
            title=f"Training History: {config['description']}",
            save_path=f'plots/depth_experiments/{config["name"]}_history.png'
        )

    # Сравнение всех моделей
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL DEPTH CONFIGURATIONS")
    print("=" * 80)
    compare_experiments(results_list, metric='final_test_acc')

    # Визуализация сравнений
    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Test Accuracy Comparison: Different Depths',
        save_path='plots/depth_experiments/all_depths_comparison.png'
    )

    plot_comparison_bar(
        results_list,
        metrics=['final_test_acc', 'best_test_acc'],
        title='Model Depth Comparison',
        save_path='plots/depth_experiments/depth_bar_comparison.png'
    )

    plot_parameter_vs_accuracy(
        results_list,
        title='Parameters vs Accuracy: Depth Experiments',
        save_path='plots/depth_experiments/params_vs_accuracy.png'
    )

    plot_training_time_comparison(
        results_list,
        title='Training Time: Different Depths',
        save_path='plots/depth_experiments/training_time_comparison.png'
    )

    return results_list


def experiment_1_2_overfitting_analysis():
    """
    Задание 1.2: Анализ переобучения
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.2: Overfitting Analysis")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Глубокая модель для анализа переобучения
    deep_configs = [
        {
            'name': 'deep_no_regularization',
            'hidden_sizes': [512, 512, 256, 256, 128, 64],
            'use_dropout': False,
            'use_batch_norm': False,
            'description': 'Deep network without regularization'
        },
        {
            'name': 'deep_with_dropout',
            'hidden_sizes': [512, 512, 256, 256, 128, 64],
            'use_dropout': True,
            'dropout_rate': 0.3,
            'use_batch_norm': False,
            'description': 'Deep network with Dropout (0.3)'
        },
        {
            'name': 'deep_with_batchnorm',
            'hidden_sizes': [512, 512, 256, 256, 128, 64],
            'use_dropout': False,
            'use_batch_norm': True,
            'description': 'Deep network with BatchNorm'
        },
        {
            'name': 'deep_with_both',
            'hidden_sizes': [512, 512, 256, 256, 128, 64],
            'use_dropout': True,
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'description': 'Deep network with Dropout + BatchNorm'
        }
    ]

    results_list = []

    for config in deep_configs:
        print(f"\nTraining: {config['description']}")
        print("-" * 80)

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=config['hidden_sizes'],
                use_dropout=config['use_dropout'],
                dropout_rate=config.get('dropout_rate', 0.5),
                use_batch_norm=config['use_batch_norm'],
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
            save_path=f'results/depth_experiments/{config["name"]}.json'
        )

        results_list.append(results)

        # Анализ переобучения для каждой модели
        plot_overfitting_analysis(
            results['history'],
            title=f"Overfitting Analysis: {config['description']}",
            save_path=f'plots/depth_experiments/{config["name"]}_overfitting.png'
        )

    # Сравнение эффектов регуляризации
    print("\n" + "=" * 80)
    print("REGULARIZATION EFFECTS COMPARISON")
    print("=" * 80)
    compare_experiments(results_list, metric='final_test_acc')

    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Regularization Effects on Deep Networks',
        save_path='plots/depth_experiments/regularization_comparison.png'
    )

    # График Train vs Test для всех конфигураций
    plot_multiple_experiments(
        results_list,
        metric='train_acc',
        title='Train Accuracy: Regularization Effects',
        save_path='plots/depth_experiments/train_acc_regularization.png'
    )

    return results_list


def analyze_optimal_depth():
    """
    Дополнительный анализ: поиск оптимальной глубины
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS: Finding Optimal Depth")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Тестируем различные глубины с фиксированной шириной
    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    fixed_width = 256

    results_list = []

    for depth in depths:
        hidden_sizes = [fixed_width] * (depth - 1) if depth > 1 else []

        print(f"\nTesting depth: {depth} layers")
        print(f"Architecture: {hidden_sizes}")

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=hidden_sizes,
                use_dropout=False,
                use_batch_norm=False,
                activation='relu'
            )
            return nn.Sequential(nn.Flatten(), model)

        results = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=f'depth_{depth}_layers',
            epochs=15,
            learning_rate=0.001,
            device=device,
            save_path=f'results/depth_experiments/optimal_depth_{depth}.json'
        )

        results_list.append(results)

    # Визуализация зависимости точности от глубины
    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Test Accuracy vs Network Depth',
        save_path='plots/depth_experiments/optimal_depth_search.png'
    )

    plot_comparison_bar(
        results_list,
        metrics=['final_test_acc'],
        title='Optimal Depth Analysis',
        save_path='plots/depth_experiments/optimal_depth_bar.png'
    )

    return results_list


def main():
    """Главная функция для запуска всех экспериментов по глубине"""
    print("\n" + "#" * 80)
    print("# HOMEWORK: DEPTH EXPERIMENTS")
    print("#" * 80 + "\n")

    # Создание директорий для результатов
    Path('results/depth_experiments').mkdir(parents=True, exist_ok=True)
    Path('plots/depth_experiments').mkdir(parents=True, exist_ok=True)

    # Эксперимент 1.1: Сравнение разных глубин
    results_1_1 = experiment_1_1_depth_comparison()

    # Эксперимент 1.2: Анализ переобучения
    results_1_2 = experiment_1_2_overfitting_analysis()

    # Дополнительный анализ: оптимальная глубина
    results_optimal = analyze_optimal_depth()

    print("\n" + "#" * 80)
    print("# ALL DEPTH EXPERIMENTS COMPLETED!")
    print("#" * 80 + "\n")

    # Финальный анализ
    print("Summary of findings:")
    print("-" * 80)

    # Лучшая модель без регуляризации
    best_no_reg = max(results_1_1, key=lambda x: x['final_test_acc'])
    print(f"Best model without regularization: {best_no_reg['experiment_name']}")
    print(f"  Test Accuracy: {best_no_reg['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_no_reg['num_parameters']:,}")

    # Лучшая модель с регуляризацией
    best_with_reg = max(results_1_2, key=lambda x: x['final_test_acc'])
    print(f"\nBest model with regularization: {best_with_reg['experiment_name']}")
    print(f"  Test Accuracy: {best_with_reg['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_with_reg['num_parameters']:,}")

    # Оптимальная глубина
    best_depth = max(results_optimal, key=lambda x: x['final_test_acc'])
    print(f"\nOptimal depth configuration: {best_depth['experiment_name']}")
    print(f"  Test Accuracy: {best_depth['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_depth['num_parameters']:,}")


if __name__ == "__main__":
    main()

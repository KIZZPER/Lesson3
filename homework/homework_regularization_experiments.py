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
    plot_overfitting_analysis,
    plot_comparison_bar,
    plot_weight_distribution,
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


def experiment_3_1_regularization_comparison():
    """
    Задание 3.1: Сравнение техник регуляризации
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3.1: Regularization Techniques Comparison")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Базовая архитектура для всех экспериментов
    base_architecture = [512, 256, 128, 64]

    # Конфигурации для сравнения
    regularization_configs = [
        {
            'name': 'no_regularization',
            'use_dropout': False,
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'description': 'No regularization'
        },
        {
            'name': 'dropout_01',
            'use_dropout': True,
            'dropout_rate': 0.1,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'description': 'Dropout (0.1)'
        },
        {
            'name': 'dropout_03',
            'use_dropout': True,
            'dropout_rate': 0.3,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'description': 'Dropout (0.3)'
        },
        {
            'name': 'dropout_05',
            'use_dropout': True,
            'dropout_rate': 0.5,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'description': 'Dropout (0.5)'
        },
        {
            'name': 'batch_norm',
            'use_dropout': False,
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'weight_decay': 0.0,
            'description': 'BatchNorm only'
        },
        {
            'name': 'dropout_batchnorm',
            'use_dropout': True,
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'weight_decay': 0.0,
            'description': 'Dropout (0.3) + BatchNorm'
        },
        {
            'name': 'l2_regularization',
            'use_dropout': False,
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'weight_decay': 0.0001,
            'description': 'L2 Regularization (weight_decay=0.0001)'
        },
        {
            'name': 'l2_strong',
            'use_dropout': False,
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'weight_decay': 0.001,
            'description': 'L2 Strong (weight_decay=0.001)'
        }
    ]

    results_list = []

    for config in regularization_configs:
        print(f"\nTraining: {config['description']}")
        print("-" * 80)

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=base_architecture,
                use_dropout=config['use_dropout'],
                dropout_rate=config['dropout_rate'],
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
            weight_decay=config['weight_decay'],
            device=device,
            save_path=f'results/regularization_experiments/{config["name"]}.json'
        )

        results_list.append(results)

        # Визуализация для каждой конфигурации
        plot_training_history(
            results['history'],
            title=f"Training: {config['description']}",
            save_path=f'plots/regularization_experiments/{config["name"]}_history.png'
        )

        plot_overfitting_analysis(
            results['history'],
            title=f"Overfitting Analysis: {config['description']}",
            save_path=f'plots/regularization_experiments/{config["name"]}_overfitting.png'
        )

    # Сравнение всех техник регуляризации
    print("\n" + "=" * 80)
    print("COMPARISON OF REGULARIZATION TECHNIQUES")
    print("=" * 80)
    compare_experiments(results_list, metric='final_test_acc')

    # Визуализации сравнений
    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Test Accuracy: Regularization Techniques',
        save_path='plots/regularization_experiments/all_techniques_test_acc.png'
    )

    plot_multiple_experiments(
        results_list,
        metric='train_acc',
        title='Train Accuracy: Regularization Techniques',
        save_path='plots/regularization_experiments/all_techniques_train_acc.png'
    )

    plot_comparison_bar(
        results_list,
        metrics=['final_test_acc', 'best_test_acc'],
        title='Regularization Techniques Performance',
        save_path='plots/regularization_experiments/techniques_bar_comparison.png'
    )

    return results_list


def experiment_3_2_adaptive_regularization():
    """
    Задание 3.2: Адаптивная регуляризация
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3.2: Adaptive Regularization")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_list = []

    # Эксперимент 1: Различные dropout rates для разных глубин
    print("\n--- Testing different dropout rates ---")
    dropout_rates = [0.2, 0.3, 0.4, 0.5]

    for dropout_rate in dropout_rates:
        print(f"\nDropout rate: {dropout_rate}")

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=[512, 256, 128, 64],
                use_dropout=True,
                dropout_rate=dropout_rate,
                use_batch_norm=False,
                activation='relu'
            )
            return nn.Sequential(nn.Flatten(), model)

        results = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=f'adaptive_dropout_{int(dropout_rate * 10)}',
            epochs=20,
            learning_rate=0.001,
            device=device,
            save_path=f'results/regularization_experiments/adaptive_dropout_{int(dropout_rate * 10)}.json'
        )

        results_list.append(results)

    # Визуализация dropout rates
    plot_multiple_experiments(
        results_list,
        metric='test_acc',
        title='Adaptive Dropout: Different Rates',
        save_path='plots/regularization_experiments/adaptive_dropout_comparison.png'
    )

    # Эксперимент 2: Комбинированные подходы
    print("\n--- Testing combined approaches ---")

    combined_configs = [
        {
            'name': 'light_regularization',
            'hidden_sizes': [256, 128, 64],
            'use_dropout': True,
            'dropout_rate': 0.2,
            'use_batch_norm': True,
            'weight_decay': 0.00001,
            'description': 'Light regularization (Dropout 0.2 + BN + L2 weak)'
        },
        {
            'name': 'medium_regularization',
            'hidden_sizes': [512, 256, 128, 64],
            'use_dropout': True,
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'weight_decay': 0.0001,
            'description': 'Medium regularization (Dropout 0.3 + BN + L2 medium)'
        },
        {
            'name': 'heavy_regularization',
            'hidden_sizes': [512, 256, 128, 64],
            'use_dropout': True,
            'dropout_rate': 0.5,
            'use_batch_norm': True,
            'weight_decay': 0.001,
            'description': 'Heavy regularization (Dropout 0.5 + BN + L2 strong)'
        }
    ]

    combined_results = []

    for config in combined_configs:
        print(f"\n{config['description']}")

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=config['hidden_sizes'],
                use_dropout=config['use_dropout'],
                dropout_rate=config['dropout_rate'],
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
            weight_decay=config['weight_decay'],
            device=device,
            save_path=f'results/regularization_experiments/{config["name"]}.json'
        )

        combined_results.append(results)

        plot_overfitting_analysis(
            results['history'],
            title=f"Overfitting: {config['description']}",
            save_path=f'plots/regularization_experiments/{config["name"]}_overfitting.png'
        )

    # Сравнение комбинированных подходов
    print("\n" + "=" * 80)
    print("COMPARISON OF COMBINED APPROACHES")
    print("=" * 80)
    compare_experiments(combined_results, metric='final_test_acc')

    plot_multiple_experiments(
        combined_results,
        metric='test_acc',
        title='Combined Regularization Approaches',
        save_path='plots/regularization_experiments/combined_approaches_comparison.png'
    )

    plot_comparison_bar(
        combined_results,
        metrics=['final_test_acc', 'best_test_acc'],
        title='Combined Approaches Performance',
        save_path='plots/regularization_experiments/combined_bar_comparison.png'
    )

    return results_list + combined_results


def experiment_weight_distribution_analysis():
    """
    Дополнительный эксперимент: Анализ распределения весов

    Визуализируем, как регуляризация влияет на распределение весов
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL EXPERIMENT: Weight Distribution Analysis")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        {
            'name': 'no_reg_weights',
            'use_dropout': False,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'description': 'No Regularization'
        },
        {
            'name': 'dropout_weights',
            'use_dropout': True,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'description': 'With Dropout'
        },
        {
            'name': 'l2_weights',
            'use_dropout': False,
            'use_batch_norm': False,
            'weight_decay': 0.001,
            'description': 'With L2'
        }
    ]

    for config in configs:
        print(f"\nAnalyzing weights: {config['description']}")

        def model_fn():
            model = create_fc_model(
                input_size=784,
                num_classes=10,
                hidden_sizes=[256, 128, 64],
                use_dropout=config['use_dropout'],
                dropout_rate=0.3,
                use_batch_norm=config['use_batch_norm'],
                activation='relu'
            )
            return nn.Sequential(nn.Flatten(), model)

        # Обучаем модель
        results = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=config['name'],
            epochs=15,
            learning_rate=0.001,
            weight_decay=config['weight_decay'],
            device=device
        )

        # Создаем модель заново для анализа весов
        model = model_fn().to(device)

        # Загружаем лучшие веса (или просто анализируем текущие)
        plot_weight_distribution(
            model,
            title=f"Weight Distribution: {config['description']}",
            save_path=f'plots/regularization_experiments/weights_{config["name"]}.png'
        )


def experiment_stability_analysis():
    """
    Дополнительный эксперимент: Анализ стабильности обучения

    Проверяем стабильность различных техник регуляризации
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL EXPERIMENT: Training Stability Analysis")
    print("=" * 80 + "\n")

    train_loader, test_loader = prepare_mnist_data(batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Запускаем несколько экспериментов с разными random seeds
    num_runs = 3

    configs = [
        {
            'name': 'stable_no_reg',
            'use_dropout': False,
            'use_batch_norm': False,
            'description': 'No Regularization'
        },
        {
            'name': 'stable_with_bn',
            'use_dropout': False,
            'use_batch_norm': True,
            'description': 'With BatchNorm'
        }
    ]

    stability_results = {}

    for config in configs:
        print(f"\n--- Testing stability: {config['description']} ---")
        run_results = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")

            # Устанавливаем seed для воспроизводимости
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            def model_fn():
                model = create_fc_model(
                    input_size=784,
                    num_classes=10,
                    hidden_sizes=[256, 128, 64],
                    use_dropout=config['use_dropout'],
                    dropout_rate=0.3,
                    use_batch_norm=config['use_batch_norm'],
                    activation='relu'
                )
                return nn.Sequential(nn.Flatten(), model)

            results = run_experiment(
                model_fn=model_fn,
                train_loader=train_loader,
                test_loader=test_loader,
                experiment_name=f"{config['name']}_run{run}",
                epochs=15,
                learning_rate=0.001,
                device=device,
            )

            run_results.append(results)

        stability_results[config['name']] = run_results

        # Анализ стабильности
        final_accs = [r['final_test_acc'] for r in run_results]
        mean_acc = np.mean(final_accs)
        std_acc = np.std(final_accs)

        print(f"\nStability for {config['description']}:")
        print(f"  Mean accuracy: {mean_acc:.2f}%")
        print(f"  Std deviation: {std_acc:.3f}%")
        print(f"  All runs: {[f'{acc:.2f}%' for acc in final_accs]}")

    return stability_results


def main():
    """Главная функция для запуска всех экспериментов по регуляризации"""
    print("\n" + "#" * 80)
    print("# HOMEWORK: REGULARIZATION EXPERIMENTS")
    print("#" * 80 + "\n")

    # Создание директорий для результатов
    Path('results/regularization_experiments').mkdir(parents=True, exist_ok=True)
    Path('plots/regularization_experiments').mkdir(parents=True, exist_ok=True)

    # Эксперимент 3.1: Сравнение техник регуляризации
    results_3_1 = experiment_3_1_regularization_comparison()

    # Эксперимент 3.2: Адаптивная регуляризация
    results_3_2 = experiment_3_2_adaptive_regularization()

    # Дополнительный эксперимент: Анализ распределения весов
    experiment_weight_distribution_analysis()

    # Дополнительный эксперимент: Анализ стабильности
    stability_results = experiment_stability_analysis()

    print("\n" + "#" * 80)
    print("# ALL REGULARIZATION EXPERIMENTS COMPLETED!")
    print("#" * 80 + "\n")

    # Финальный анализ
    print("Summary of findings:")
    print("-" * 80)

    # Лучшая техника регуляризации
    best_technique = max(results_3_1, key=lambda x: x['final_test_acc'])
    print(f"Best regularization technique: {best_technique['experiment_name']}")
    print(f"  Test Accuracy: {best_technique['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_technique['num_parameters']:,}")

    # Лучший комбинированный подход
    best_combined = max(results_3_2, key=lambda x: x['final_test_acc'])
    print(f"\nBest combined approach: {best_combined['experiment_name']}")
    print(f"  Test Accuracy: {best_combined['final_test_acc']:.2f}%")
    print(f"  Parameters: {best_combined['num_parameters']:,}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle

from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SelfCGP, SHADE
from thefittest.regressors._gpnneregression_one_tree_mo import (
    GeneticProgrammingNeuralNetStackingRegressorMO,
)
from thefittest.tools.print import print_tree, print_trees, print_net, print_nets, print_ens
from thefittest.benchmarks import SolarBatteryDegradationDataset
import warnings

# Игнорируем все предупреждения
warnings.filterwarnings("ignore")


import torch
import numpy as np


def calculate_fitness_for_ensemble(y_true, y_pred) -> float:
    # Преобразуем данные в тензоры PyTorch
    targets_tensor = torch.tensor(y_true, dtype=torch.float32)
    output_tensor = torch.tensor(y_pred, dtype=torch.float32)

    # Находим минимальные и максимальные значения для целевых значений
    y_min, _ = torch.min(targets_tensor, dim=0)
    y_max, _ = torch.max(targets_tensor, dim=0)

    s = len(targets_tensor)  # Размер тестовой выборки
    m = targets_tensor.shape[1]  # Количество выходных значений (например, 4)

    # Инициализируем переменную для суммы ошибок
    error_sum = 0

    # Вычисляем ошибку для каждого выхода
    for j in range(m):  # Для каждого выхода (столбца)
        y_diff = y_max[j] - y_min[j]  # Нормализация по максимуму и минимуму
        error_sum += torch.sum(torch.abs(targets_tensor[:, j] - output_tensor[:, j])) / y_diff

    # Рассчитываем итоговую ошибку
    error = (100 / s) * error_sum
    return error.item()  # Возвращаем ошибку как скалярное значение


def run_experiment(run_id, output_dir):
    # === Подготовка данных ===
    dataset = SolarBatteryDegradationDataset()
    X = minmax_scale(dataset.get_X())  # Нормализация входов от 0 до 1
    y = dataset.get_y()  # Только первый выход

    X_train, X_test = X[:169], X[169:]
    y_train, y_test = y[:169], y[169:]

    # === Обучение модели ===
    model = GeneticProgrammingNeuralNetStackingRegressorMO(
        iters=30,
        pop_size=10,
        input_block_size=7,
        optimizer=PDPSHAGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 1},
        weights_optimizer=SHADE,
        weights_optimizer_args={
            "iters": 20000,
            "pop_size": 20000,
            "no_increase_num": 100,
            "fitness_update_eps": 0.01,
            "show_progress_each": 1,
        },
        test_sample_ratio=0.33,
    )
    model.fit(X_train, y_train)

    # === Предсказания ===
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_true = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])
    split_index = len(y_train)

    r2 = r2_score(y_test, y_test_pred)

    # === Вычисление GNE по формуле для каждого выхода ===
    gne_train = calculate_fitness_for_ensemble(y_train, y_train_pred)
    gne_test = calculate_fitness_for_ensemble(y_test, y_test_pred)
    gne_full = calculate_fitness_for_ensemble(y_all_true, y_all_pred)

    # === Сохранение результатов ===
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "r2_score.txt"), "w") as f:
        f.write(f"R2 Score: {r2:.6f}\n")
        f.write(f"GNE Train: {gne_train:.6f}\n")
        f.write(f"GNE Test: {gne_test:.6f}\n")
        f.write(f"GNE Full: {gne_full:.6f}\n")

    np.savetxt(os.path.join(run_dir, "X_train.txt"), X_train)
    np.savetxt(os.path.join(run_dir, "X_test.txt"), X_test)
    np.savetxt(os.path.join(run_dir, "y_train.txt"), y_train)
    np.savetxt(os.path.join(run_dir, "y_test.txt"), y_test)
    np.savetxt(os.path.join(run_dir, "train_pred.txt"), y_train_pred)
    np.savetxt(os.path.join(run_dir, "test_pred.txt"), y_test_pred)
    np.savetxt(os.path.join(run_dir, "full_true.txt"), y_all_true)
    np.savetxt(os.path.join(run_dir, "full_pred.txt"), y_all_pred)

    # === Графики для каждого выхода ===
    for i in range(y_all_true.shape[1]):  # Для каждого выхода создадим график
        plt.figure(figsize=(14, 5))
        plt.plot(y_all_true[:, i], label="True", linewidth=2)
        plt.plot(y_all_pred[:, i], label="Predicted", linestyle="--")
        plt.axvline(split_index, color="gray", linestyle=":", label="Train/Test Split")
        plt.title(f"Prediction for Output {i+1} — Train/Test")
        plt.xlabel("Sample Index")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"prediction_curve_output_{i+1}.png"))
        plt.close()

    # === Структуры модели ===
    optimizer = model.get_optimizer()
    stat = optimizer.get_stats()
    stat["population_ph"] = None

    common_tree = optimizer.get_fittest()["genotype"]
    ens = optimizer.get_fittest()["phenotype"]
    trees = ens._trees

    print_tree(common_tree)
    plt.savefig(os.path.join(run_dir, "1_common_tree.png"))
    plt.close()

    print_trees(trees)
    plt.savefig(os.path.join(run_dir, "2_trees.png"))
    plt.close()

    print_nets(ens._nets)
    plt.savefig(os.path.join(run_dir, "3_nets.png"))
    plt.close()

    print_net(ens._meta_algorithm)
    plt.savefig(os.path.join(run_dir, "4_meta_net.png"))
    plt.close()

    print_ens(ens)
    plt.savefig(os.path.join(run_dir, "5_ens.png"))
    plt.close()

    # === Сериализация объектов ===
    ens.save_to_file(os.path.join(run_dir, "ens.pkl"))
    common_tree.save_to_file(os.path.join(run_dir, "common_tree.pkl"))
    with open(os.path.join(run_dir, "stat.pkl"), "wb") as file:
        cloudpickle.dump(stat, file)

    print(f"✅ Run {run_id} done — R2: {r2:.4f} | GNE test: {gne_test:.6f}\n")
    return r2


def run_multiple_experiments(n_runs, output_dir, start_run=0):
    r2_scores = []
    for i in range(start_run, n_runs):
        r2 = run_experiment(i, output_dir)
        r2_scores.append(r2)

    avg_r2 = np.mean(r2_scores)
    with open(os.path.join(output_dir, "average_r2_score.txt"), "w") as f:
        f.write(f"Average R2 Score: {avg_r2:.6f}\n")

    print(f"\n✅ All runs complete. Average R2: {avg_r2:.4f}")


if __name__ == "__main__":
    output_dir = r"C:\Users\pasha\OneDrive\Рабочий стол\results_regression_combined"
    n_runs = 1
    run_multiple_experiments(n_runs, output_dir)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle

from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale, MinMaxScaler

from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SHADE
from thefittest.regressors._gpnneregression_one_tree_mo import \
    GeneticProgrammingNeuralNetStackingRegressorMO
from thefittest.benchmarks import SolarBatteryDegradationDataset
from thefittest.tools.print import print_tree, print_ens

import warnings
warnings.filterwarnings("ignore")


def calculate_global_normalized_error(y_true, y_pred, y_min_global, y_max_global):
    """
    global normalized error with fixed global ranges:
    (100/s)*(1/m) * sum_i sum_j |y_true[i,j] - y_pred[i,j]| / (y_max_global[j] - y_min_global[j])
    """
    s, m = y_true.shape
    total_error = 0.0
    for j in range(m):
        y_j = y_true[:, j]
        y_pj = y_pred[:, j]
        denom = y_max_global[j] - y_min_global[j] if y_max_global[j] != y_min_global[j] else 1.0
        total_error += np.sum(np.abs(y_j - y_pj)) / denom
    return (100.0 / s) * (1.0 / m) * total_error


def run_experiment(run_id, output_dir):
    # === Данные ===
    dataset = SolarBatteryDegradationDataset()
    X_raw = dataset.get_X()
    X = minmax_scale(X_raw)
    y_raw = dataset.get_y()

    # масштабирование y
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y_raw)

    # глобальные min/max для GNE по оригинальным y
    y_min_global = np.min(y_raw, axis=0)
    y_max_global = np.max(y_raw, axis=0)

    # train/test split
    n_train = 169
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    y_train_raw, y_test_raw = y_raw[:n_train], y_raw[n_train:]

    # === Модель ===
    model = GeneticProgrammingNeuralNetStackingRegressorMO(
        iters=10,
        pop_size=10,
        input_block_size=1,
        optimizer=PDPSHAGP,
        optimizer_args={"show_progress_each": 1, "keep_history": True, "n_jobs": 10},
        weights_optimizer=SHADE,
        weights_optimizer_args={
            "iters": 50,
            "pop_size": 50,
            "no_increase_num": 100,
            "fitness_update_eps": 0.01,
        },
        test_sample_ratio=0.33,
    )

        # обучение на масштабированных y
    model.fit(X, y)

    # предсказание масштабированных значений
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled  = model.predict(X_test)

    # обратное масштабирование предсказаний
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred  = scaler_y.inverse_transform(y_test_pred_scaled)

    # собираем все значения
    y_all_true = np.vstack([y_train_raw, y_test_raw])
    y_all_pred = np.vstack([y_train_pred, y_test_pred])

    # метрики
    r2 = r2_score(y_test_raw, y_test_pred)
    gne_train = calculate_global_normalized_error(y_train_raw, y_train_pred, y_min_global, y_max_global)
    gne_test  = calculate_global_normalized_error(y_test_raw, y_test_pred,  y_min_global, y_max_global)
    gne_full  = calculate_global_normalized_error(y_all_true,  y_all_pred,  y_min_global, y_max_global)

    # директория для run
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # сохраняем метрики
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"GNE_train: {gne_train:.6f}\n")
        f.write(f"GNE_test: {gne_test:.6f}\n")
        f.write(f"GNE_full: {gne_full:.6f}\n")

    # сохраняем предсказания
    np.savetxt(os.path.join(run_dir, "train_pred.txt"), y_train_pred)
    np.savetxt(os.path.join(run_dir, "test_pred.txt"),  y_test_pred)
    np.savetxt(os.path.join(run_dir, "full_true.txt"), y_all_true)
    np.savetxt(os.path.join(run_dir, "full_pred.txt"), y_all_pred)

    # графики
    split_idx = len(y_train_pred)
    for j in range(y_all_true.shape[1]):
        plt.figure(figsize=(14,5))
        plt.plot(y_all_true[:, j], label="True", linewidth=2)
        plt.plot(y_all_pred[:, j], label="Pred", linestyle='--')
        plt.axvline(split_idx, color='gray', linestyle=':')
        plt.title(f"Output {j+1} — Predictions")
        plt.xlabel("Index")
        plt.ylabel(f"Output_{j+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"curve_output_{j+1}.png"))
        plt.close()

    # сохраняем структуру
    optimizer = model.get_optimizer()
    fittest = optimizer.get_fittest()
    ensembling = fittest['phenotype']

    # сохраняем дерево
    print_tree(fittest['genotype'])
    plt.savefig(os.path.join(run_dir, "common_tree.png"))
    plt.close()

    # сохраняем ансамбль
    print_ens(ensembling)
    plt.savefig(os.path.join(run_dir, "ensemble.png"))
    plt.close()

    # сериализация
    cloudpickle.dump(ensembling, open(os.path.join(run_dir, "ens.pkl"), "wb"))

    # === Сохраняем CSV входов, используемых ансамблем, и предсказаний ===
        # определяем индексы входов в ансамбле
    used_inputs = set()
    for net in ensembling._nets:
        used_inputs.update(net._inputs)
    # убрать смещение: индекс равный числу фич
    bias_idx = X_raw.shape[1]
    if bias_idx in used_inputs:
        used_inputs.discard(bias_idx)
    used_indices = sorted(used_inputs)

    # получаем имена признаков
    x_names = dataset.get_X_names()  # должно возвращать dict {idx: name}
    selected_names = [x_names[i] for i in used_indices]

    # создаём DataFrame
    df = pd.DataFrame(X_raw[:, used_indices], columns=selected_names)
    # добавляем все выходы предсказания полного набора
    y_names = dataset.get_y_names()
    for j, name in enumerate(y_names):
        df[name] = y_all_pred[:, j]

    df.to_csv(os.path.join(run_dir, "selected_inputs_and_preds.csv"), index=False)

    print(f"✅ Run {run_id} done | R2={r2:.4f} | GNE_test={gne_test:.4f}")
    return r2


def run_multiple_experiments(n_runs, output_dir):
    scores = []
    for i in range(n_runs):
        r2 = run_experiment(i, output_dir)
        scores.append(r2)
    avg = np.mean(scores)
    with open(os.path.join(output_dir, "avg_r2.txt"), "w") as f:
        f.write(f"Average R2: {avg:.6f}\n")
    print(f"✅ All runs done | avg R2={avg:.4f}")


if __name__ == "__main__":
    run_multiple_experiments(n_runs=1, output_dir="results_regression_combined")

import os
import time
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from thefittest.benchmarks import SolarBatteryDegradationDataset
from thefittest.fuzzy import FuzzyRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


def calculate_global_normalized_error(y_true, y_pred, y_min_global, y_max_global):
    s, m = y_true.shape
    total_error = 0.0
    for j in range(m):
        denom = y_max_global[j] - y_min_global[j] or 1.0
        total_error += np.sum(np.abs(y_true[:, j] - y_pred[:, j])) / denom
    return (100.0 / s) * (1.0 / m) * total_error


def run_fuzzy_solar_single(run_id: int, base_dir="results_fuzzy_solar"):
    """
    Один запуск: создаёт папку base_dir/run_<run_id>,
    тренирует модель, сохраняет результаты и кривые.
    """
    output_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # === Загрузка и масштабирование ===
    ds = SolarBatteryDegradationDataset()
    X_raw, y_raw = ds.get_X(), ds.get_y()
    target_names = ds.get_y_names()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X_raw)
    y = scaler_y.fit_transform(y_raw)
    y_min_g, y_max_g = y_raw.min(axis=0), y_raw.max(axis=0)

    # split
    n_train = 169
    X_train, X_test = X[:n_train], X[n_train:]
    y_train_raw, y_test_raw = y_raw[:n_train], y_raw[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # === модель ===
    n_f, n_o = X.shape[1], y.shape[1]
    model = FuzzyRegressor(
        iters=3000,
        pop_size=200,
        n_features_fuzzy_sets=[5] * n_f,
        n_target_fuzzy_sets=[7] * n_o,
        max_rules_in_base=20,
        target_grid_volume=100,
    )
    # задаём имена множеств
    labels5 = ["очень низкое", "низкое", "среднее", "высокое", "очень высокое"]
    labels3 = ["низкое", "среднее", "высокое"]
    Xnames = [ds.get_X_names()[i] for i in range(n_f)]
    set_names = {nm: labels5 for nm in Xnames}
    target_set_names = {nm: labels5 for nm in target_names}

    model.define_sets(
        X,
        y,
        # feature_names=Xnames,
        # set_names=set_names,
        # target_names=target_names,
        # target_set_names=target_set_names,
    )

    # тренировка
    t0 = time.time()
    model.fit(X, y)
    train_time = time.time() - t0

    # предикт + inverse
    y_pred = scaler_y.inverse_transform(model.predict(X_test))

    # метрика
    gne = calculate_global_normalized_error(y_test_raw, y_pred, y_min_g, y_max_g)

    # --- сохраняем модель и правила ---
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, "rules.txt"), "w", encoding="utf-8") as f:
        f.write(model.get_text_rules(print_intervals=False))

    # --- сохраняем таблицу предсказаний ---
    cols = [str(n) for n in target_names]
    df_true = pd.DataFrame(y_test_raw, columns=cols)
    df_pred = pd.DataFrame(y_pred, columns=[c + "_pred" for c in cols])
    pd.concat([df_true, df_pred], axis=1).to_csv(
        os.path.join(output_dir, "predictions.csv"), index=False
    )

    # --- строим графики True vs Pred ---
    y_all_true = np.vstack([y_train_raw, y_test_raw])
    y_all_pred = np.vstack([scaler_y.inverse_transform(model.predict(X_train)), y_pred])
    for j, name in enumerate(cols):
        plt.figure(figsize=(10, 4))
        plt.plot(y_all_true[:, j], label="True", linewidth=2)
        plt.plot(y_all_pred[:, j], label="Pred", linestyle="--")
        plt.axvline(len(y_train_raw), color="gray", linestyle=":")
        plt.title(f"{name}: True vs Pred")
        plt.xlabel("Index")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"curve_{j+1}.png"))
        plt.close()

    # --- метрики в файл ---
    metrics = {"GNE": float(gne), "Train_time_sec": train_time}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"[run_{run_id}] Done GNE={gne:.4f}, time={train_time:.1f}s")


def worker(run_id, base_dir):
    import traceback

    while True:
        try:
            run_fuzzy_solar_single(run_id, base_dir)
            break
        except Exception as e:
            print(f"[run_{run_id}] Error: {e}. Retrying in 5s...")
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    BASE_DIR = "results_fuzzy_solar"
    N_RUNS = 100  # сколько параллельных экспериментов

    os.makedirs(BASE_DIR, exist_ok=True)
    with mp.Pool(processes=min(mp.cpu_count(), N_RUNS)) as pool:
        # передаём в каждую задачу свой run_id
        pool.starmap(worker, [(i, BASE_DIR) for i in range(N_RUNS)])

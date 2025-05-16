import os
import time
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thefittest.benchmarks import SolarBatteryDegradationDataset
from thefittest.fuzzy import FuzzyRegressor
from sklearn.preprocessing import MinMaxScaler
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


def run_fuzzy_solar(output_dir="results_fuzzy_solar"):
    # === Загрузка и масштабирование данных ===
    dataset = SolarBatteryDegradationDataset()
    X_raw = dataset.get_X()       # (n_samples, n_features)
    y_raw = dataset.get_y()       # (n_samples, n_outputs)
    target_names = dataset.get_y_names()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X_raw)
    y = scaler_y.fit_transform(y_raw)

    # глобальные min/max для GNE по оригинальным y
    y_min_global = np.min(y_raw, axis=0)
    y_max_global = np.max(y_raw, axis=0)

    # === Разбиение на train/test ===
    n_train = 169
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    y_train_raw, y_test_raw = y_raw[:n_train], y_raw[n_train:]

    # === Определение и обучение модели ===
    n_features = X.shape[1]
    n_outputs = y.shape[1]
    model = FuzzyRegressor(
        iters=1000,
        pop_size=1000,
        n_features_fuzzy_sets=[5]*n_features,
        n_target_fuzzy_sets=[3]*n_outputs,
        max_rules_in_base=5,
        target_grid_volume=100,
    )
    # Названия и множества
    labels5 = ["очень низкое","низкое","среднее","высокое","очень высокое"]
    labels3 = ["низкое","среднее","высокое",]
    x_names_dict = dataset.get_X_names()  # {idx: name}
    Xnames = [x_names_dict[i] for i in range(n_features)]
    set_names = {name: labels5 for name in Xnames}
    target_set_names = {name: labels3 for name in target_names}
    model.define_sets(
        X, y,
        feature_names=Xnames,
        set_names=set_names,
        target_names=target_names,
        target_set_names=target_set_names,
    )

    # fit
    t0 = time.time()
    model.fit(X, y)
    train_time = time.time() - t0

    # predict on scaled data
    y_pred_scaled = model.predict(X_test)
    # inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # вычисление метрик
    gne = calculate_global_normalized_error(y_test_raw, y_pred, y_min_global, y_max_global)

    # сохраняем результаты
    os.makedirs(output_dir, exist_ok=True)

    # 1) модель
    with open(os.path.join(output_dir, "fuzzy_solar.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 2) база правил в текстовом виде
    rules_text = model.get_text_rules(print_intervals=False)
    with open(os.path.join(output_dir, "rules.txt"), "w", encoding="utf-8") as f:
        f.write(rules_text)

    # 3) предсказания и истинные
    str_target_names = [str(f) for f in target_names]
    df_true = pd.DataFrame(y_test_raw, columns=str_target_names)
    df_pred = pd.DataFrame(
        y_pred,
        columns=[f"{name}_pred" for name in str_target_names]
    )
    df = pd.concat([df_true, df_pred], axis=1)
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # === 4) построение графиков True vs Pred ===
    split_idx = len(y_test_raw)  # это начало тестовой части
    # объединяем для удобства: сначала train (на котором не рисуем), потом тест
    y_all_true = np.vstack([y_train_raw, y_test_raw])
    y_all_pred = np.vstack([scaler_y.inverse_transform(model.predict(X_train)), y_pred])
    for j, name in enumerate(str_target_names):
        plt.figure(figsize=(10, 4))
        plt.plot(y_all_true[:, j], label="True", linewidth=2)
        plt.plot(y_all_pred[:, j], label="Pred", linestyle="--")
        # граница train/test
        plt.axvline(len(y_train_raw), color='gray', linestyle=':')
        plt.title(f"{name}: True vs Pred")
        plt.xlabel("Index")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"curve_output_{j+1}.png"))
        plt.close()

    # 5) метрики и вывод
    metrics = {"GNE": float(gne), "Train_time_sec": train_time}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Done GNE={gne:.4f}, train_time={train_time:.1f}s")

if __name__ == "__main__":
    run_fuzzy_solar()

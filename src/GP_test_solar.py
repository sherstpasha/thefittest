import os
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from thefittest.benchmarks import SolarBatteryDegradationDataset
from thefittest.regressors import SymbolicRegressionGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP


def calculate_global_normalized_error(y_true, y_pred, y_min_global, y_max_global):
    """
    Вычисляет глобальную нормализованную ошибку с диапазонами, рассчитанными на всей выборке:
    error = 100/(s*m) * sum_i sum_j |y_true_ij - y_pred_ij| / (y_max_global_j - y_min_global_j)
    """
    s, m = y_true.shape
    denom = y_max_global - y_min_global
    denom[denom == 0] = 1.0
    return 100.0 / (s * m) * np.sum(np.abs((y_true - y_pred) / denom))


def main():
    # === Параметры эксперимента ===
    number_of_iterations = 500
    population_size = 1000

    # === Загрузка исходных данных ===
    dataset = SolarBatteryDegradationDataset()
    X_raw = dataset.get_X()        # (n_samples, n_features)
    y_raw = dataset.get_y()        # (n_samples, n_outputs)
    target_names = dataset.get_y_names()

    # === Вычисляем глобальные мини и макси по всем выходам once ===
    y_max_global = np.max(y_raw, axis=0)
    y_min_global = np.min(y_raw, axis=0)

    # === Нормализация X и y по всему датасету ===
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    scaler_Y = StandardScaler()
    y_scaled = scaler_Y.fit_transform(y_raw)

    # === Разбиение на train/test: первые 169 и остальные ===
    n_train = 169
    X_train = X_scaled[:n_train]
    X_test = X_scaled[n_train:]
    y_train_scaled = y_scaled[:n_train]
    y_test_scaled = y_scaled[n_train:]
    y_train_raw = y_raw[:n_train]
    y_test_raw = y_raw[n_train:]

    # === Подготовка результатов ===
    os.makedirs("results", exist_ok=True)
    all_metrics = []
    y_pred_all = np.zeros_like(y_test_raw)

    # === Основной цикл по выходам ===
    for idx, target_name in enumerate(target_names):
        y_train_i_scaled = y_train_scaled[:, idx]
        y_test_i_scaled  = y_test_scaled[:, idx]
        y_train_i_raw    = y_train_raw[:, idx]
        y_test_i_raw     = y_test_raw[:, idx]

        # инициализация GP-регрессора
        model = SymbolicRegressionGP(
            iters=number_of_iterations,
            pop_size=population_size,
            optimizer=SelfCSHAGP,
            optimizer_args={
                "elitism": True,
                "keep_history": True,
                "show_progress_each": 1,
            },
        )

        # обучение на масштабированных данных
        start_time = time.time()
        model.fit(X_scaled, y_scaled[:, idx])
        train_time = time.time() - start_time

        # предсказания (scaled)
        y_train_pred_scaled = model.predict(X_train)
        y_test_pred_scaled  = model.predict(X_test)

        # обратное масштабирование в оригинальный масштаб
        y_train_pred = y_train_pred_scaled * scaler_Y.scale_[idx] + scaler_Y.mean_[idx]
        y_test_pred  = y_test_pred_scaled  * scaler_Y.scale_[idx] + scaler_Y.mean_[idx]
        y_pred_all[:, idx] = y_test_pred

        # расчёт метрик на тесте
        mse  = mean_squared_error(y_test_i_raw, y_test_pred)
        r2   = r2_score(y_test_i_raw, y_test_pred)
        mape = np.mean(
            np.abs((y_test_i_raw - y_test_pred) /
                   np.where(y_test_i_raw == 0, 1, y_test_i_raw))
        ) * 100
        all_metrics.append({
            "Target": target_name,
            "MSE": mse,
            "R2": r2,
            "MAPE(%)": mape,
            "Train_time_sec": train_time
        })
        print(f"[{target_name}] time={train_time:.1f}s, MSE={mse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")

        # сохраняем модель и скейлеры
        with open(f"results/solar_{target_name}_srgp.pkl", "wb") as f:
            pickle.dump({
                'model': model,
                'scaler_X': scaler_X,
                'scaler_Y': scaler_Y
            }, f)

        # сохраняем предсказания теста
        preds = np.vstack([y_test_i_raw, y_test_pred]).T
        np.savetxt(
            f"results/solar_{target_name}_predictions.csv",
            preds,
            delimiter=",",
            header="true,pred",
            comments="",
            fmt="%.6f"
        )

        # строим график по всем данным
        plt.figure(figsize=(12, 5))
        idxs_train = np.arange(n_train)
        idxs_test  = np.arange(n_train, n_train + len(y_test_i_raw))
        plt.scatter(idxs_train, y_train_i_raw, label="Train True", marker="o", s=30, alpha=0.6)
        plt.scatter(idxs_test,  y_test_i_raw,  label="Test True",  marker="o", s=30, alpha=0.6)
        plt.plot(idxs_train, y_train_pred, label="Train Pred", linestyle='-', linewidth=2)
        plt.plot(idxs_test,  y_test_pred,  label="Test Pred",  linestyle='--', linewidth=2)
        plt.axvline(n_train, color='gray', linestyle=':')
        plt.title(f"{target_name} — SRGP Full Data Predictions")
        plt.xlabel("Sample Index")
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/solar_{target_name}_srgp_fullplot.png")
        plt.close()

    # сохраняем метрики по всем выходам
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv("results/solar_srgp_metrics.csv", index=False)

    # вычисляем глобальную нормализованную ошибку на тесте,
    # используя глобальные диапазоны
    gne_test = calculate_global_normalized_error(
        y_test_raw, y_pred_all, y_min_global, y_max_global
    )
    print(f"Global Normalized Error (test, all outputs): {gne_test:.2f}%")

    overall_mape = df_metrics['MAPE(%)'].mean()
    print(f"Overall average MAPE: {overall_mape:.2f}%")
    print("\n✅ Все модели обучены. Результаты сохранены в 'results/'")


if __name__ == "__main__":
    main()

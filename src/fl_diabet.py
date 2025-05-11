import os
import time
import pickle
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from thefittest.fuzzy import FuzzyRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# === Функция для вычисления метрик ===
def calculate_metrics(y_true, y_pred, y_train):
    # наивный прогноз «сдвиг на 1»
    naive = np.roll(y_train, 1)[1:]
    mae_naive = np.mean(np.abs(y_train[1:] - naive))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    mase = np.mean(np.abs(y_true - y_pred)) / mae_naive

    return {
        "MASE": mase,
        "sMAPE": smape,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MSE": mse,
        "MAPE": mape,
    }


def run_experiment(run_id: int, base_output_dir="results_regressor_diabetes"):
    """
    Запускает один эксперимент (fit + predict + save) на датасете diabetes и сохраняет результаты в папке run_{run_id}.
    """
    np.random.seed(run_id)

    # Папка для этого прогона
    output_dir = os.path.join(base_output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Загрузка и подготовка данных ---
    data = load_diabetes()
    X = data.data
    y = data.target

    # масштабирование и сплит
    X_scaled = minmax_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=run_id
    )

    # Приводим таргеты к двумерному виду для соответствия API
    Y_train = y_train.reshape(-1, 1)
    Y_test = y_test.reshape(-1, 1)

    feature_names = data.feature_names
    target_names = ["target"]

    n_features = X_train.shape[1]
    n_targets = Y_train.shape[1]

    # --- 2. Настройка модели ---
    labels7 = [
        "низкое",
        "среднее",
        "высокое",
        "очень высокое",
        "экстремально высокое",
    ]

    set_names = {name: labels7 for name in feature_names}
    target_set_names = {name: labels7 for name in target_names}

    model = FuzzyRegressor(
        iters=100,
        pop_size=1000,
        n_features_fuzzy_sets=[3] * n_features,
        n_target_fuzzy_sets=[2] * n_targets,
        max_rules_in_base=10,
        target_grid_volume=50,
    )
    model.define_sets(
        X_train,
        Y_train,
        feature_names=feature_names,
        set_names=set_names,
        target_names=target_names,
        target_set_names=target_set_names,
    )

    # --- 3. Обучение ---
    t0 = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - t0

    # --- 4. Предсказание и сбор метрик ---
    Y_pred = model.predict(X_test)

    # R2 по каждой цели
    r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(n_targets)]
    avg_r2 = float(np.mean(r2_scores))

    # Метрики по целям
    all_metrics = {}
    for i, name in enumerate(target_names):
        all_metrics[name] = calculate_metrics(Y_test[:, i], Y_pred[:, i], Y_train[:, i])

    # Средние метрики по всем целям
    metric_names = all_metrics[target_names[0]].keys()
    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics.values()])) for metric in metric_names
    }

    # --- 5. Сохранение результатов ---
    # 5.1. Правила
    with open(os.path.join(output_dir, "rule_base.txt"), "w", encoding="utf-8") as f:
        f.write(model.get_text_rules(print_intervals=True))
    # 5.2. Модель
    with open(os.path.join(output_dir, "fuzzy_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    # 5.3. Предсказания
    preds_df = pd.DataFrame(
        {
            "true": Y_test.flatten(),
            "pred": Y_pred.flatten(),
        }
    )
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    # 5.4. Метрики по целям
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    # 5.5. Средние метрики
    with open(os.path.join(output_dir, "average_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
    # 5.6. Лог
    with open(os.path.join(output_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Training time: {train_time:.2f} sec\n")
        f.write(f"Average R2:    {avg_r2:.4f}\n")
        f.write("R2 by target:\n")
        for name, r2v in zip(target_names, r2_scores):
            f.write(f"  {name}: {r2v:.4f}\n")

    print(f"[run_{run_id}] done: train {train_time:.1f}s, avg R2 {avg_r2:.4f}")


if __name__ == "__main__":
    # Число прогонов и процессов
    N_RUNS = 1
    N_PROCESSES = 1

    os.makedirs("results_regressor_diabetes", exist_ok=True)
    with mp.Pool(processes=N_PROCESSES) as pool:
        pool.map(run_experiment, range(N_RUNS))

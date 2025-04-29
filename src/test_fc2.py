import os
import time
import pickle
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from thefittest.fuzzy_gpu import FuzzyRegressorTorch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# === Функция для вычисления метрик ===
def calculate_metrics(y_true, y_pred, y_train):
    # наивный прогноз «сдвиг на 1»
    naive = np.roll(y_train, 1)[1:]
    mae_naive = np.mean(np.abs(y_train[1:] - naive))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    mase  = np.mean(np.abs(y_true - y_pred)) / mae_naive

    return {
        "MASE":  mase,
        "sMAPE": smape,
        "RMSE":  rmse,
        "MAE":   mae,
        "R2":    r2,
        "MSE":   mse,
        "MAPE":  mape,
    }

def run_experiment(run_id: int, base_output_dir="results_regressor"):
    """
    Запускает один эксперимент (fit + predict + save) и сохраняет результаты в папке results_regressor/run_{run_id}.
    """
    np.random.seed(run_id)

    # Путь к данным для этого прогона
    gpenn_run_dir = os.path.join(
        r"C:\Users\USER\Desktop\Расчеты по нейросетям\расчеты сетей метео\GPENN", f"run_{run_id}"
    )

    # Папка для сохранения результатов
    output_dir = os.path.join(base_output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Загрузка данных ---
    X_train_df = pd.read_csv(os.path.join(gpenn_run_dir, "X_train_NN.csv"))
    X_test_df  = pd.read_csv(os.path.join(gpenn_run_dir, "X_test_NN.csv"))
    y_train_df = pd.read_csv(os.path.join(gpenn_run_dir, "y_train_NN.csv"))
    y_test_df  = pd.read_csv(os.path.join(gpenn_run_dir, "y_test_NN.csv"))
    y_test_orig_df  = pd.read_csv("src\lookback_1h\y_test.csv")

    # Убираем столбец времени для обучения
    feature_names = [c for c in X_train_df.columns if c != "time_YYMMDD_HHMMSS"]
    target_names  = y_train_df.columns.tolist()

    X_train = X_train_df[feature_names].values
    X_test  = X_test_df[feature_names].values
    Y_train = y_train_df[target_names].values
    Y_test  = y_test_df[target_names].values

    # --- 2. Настройка модели ---
    n_features = X_train.shape[1]
    n_targets  = Y_train.shape[1]

    # Обновленные названия для 7 термов
    labels7 = [
        "крайне низкое", 
        "очень низкое", 
        "низкое", 
        "среднее", 
        "высокое", 
        "очень высокое", 
        "крайне высокое"
    ]
    
    set_names        = {name: labels7 for name in feature_names}
    target_set_names = {name: labels7 for name in target_names}

    model = FuzzyRegressorTorch(
        iters=400,
        pop_size=200,
        n_features_fuzzy_sets=[7] * n_features,  # Изменено с 5 на 7
        n_target_fuzzy_sets=[7] * n_targets,     # Изменено с 5 на 7
        max_rules_in_base=6,
        target_grid_volume=50,
    )
    model.define_sets(
        X_train, Y_train,
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

    # Метрики по целям для Y_test
    all_metrics = {}
    for i, name in enumerate(target_names):
        all_metrics[name] = calculate_metrics(Y_test[:, i], Y_pred[:, i], Y_train[:, i])

    # Метрики по целям для y_test_orig_df
    all_metrics_orig = {}
    for i, name in enumerate(target_names):
        all_metrics_orig[name] = calculate_metrics(y_test_orig_df[target_names[i]].values, Y_pred[:, i], Y_train[:, i])

    # Средние метрики по всем целям для Y_test
    metric_names = all_metrics[target_names[0]].keys()
    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics.values()]))
        for metric in metric_names
    }

    # Средние метрики по всем целям для y_test_orig_df
    avg_metrics_orig = {
        metric: float(np.mean([m[metric] for m in all_metrics_orig.values()]))
        for metric in metric_names
    }

    # --- 5. Сохранение результатов ---
    # Удаляем временные столбцы, если они не нужны
    X_test_df.drop(columns=["time_YYMMDD_HHMMSS"], inplace=True, errors="ignore")

    # Теперь preds_df создается без ошибки
    preds_df = X_test_df.copy()  # Копируем все данные из X_test_df без столбца time_YYMMDD_HHMMSS

    # Заполняем предсказания
    for i, name in enumerate(target_names):
        preds_df[f"{name}_true"] = Y_test[:, i]
        preds_df[f"{name}_pred"] = Y_pred[:, i]

    # Сохраняем предсказания в файл
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # 5.1. Правила
    with open(os.path.join(output_dir, "rule_base.txt"), "w", encoding="utf-8") as f:
        f.write(model.get_text_rules(print_intervals=True))
    # 5.2. Модель
    with open(os.path.join(output_dir, "fuzzy_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    # 5.3. Метрики по целям для Y_test
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    # 5.4. Средние метрики для Y_test
    with open(os.path.join(output_dir, "average_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
    # 5.5. Метрики по целям для y_test_orig_df
    with open(os.path.join(output_dir, "metrics_orig.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics_orig, f, indent=4, ensure_ascii=False)
    # 5.6. Средние метрики для y_test_orig_df
    with open(os.path.join(output_dir, "average_metrics_orig.json"), "w", encoding="utf-8") as f:
        json.dump(avg_metrics_orig, f, indent=4, ensure_ascii=False)
    # 5.7. Лог
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
    N_RUNS = 20
    N_PROCESSES = 8  # None → mp.cpu_count()

    os.makedirs("results_regressor", exist_ok=True)
    with mp.Pool(processes=N_PROCESSES) as pool:
        pool.map(run_experiment, [2, 3, 4, 5, 6, 10, 11, 13, 15, 16, 17, 18])
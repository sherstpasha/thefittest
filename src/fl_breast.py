import os
import time
import pickle
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from thefittest.tools.transformations import SamplingGrid
from thefittest.optimizers._selfcshaga import SelfCSHAGA
from thefittest.fuzzy import FuzzyClassifier
from thefittest.benchmarks import BreastCancerDataset, IrisDataset, BanknoteDataset
from sklearn.metrics import f1_score, accuracy_score
import warnings

from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore")


# === Функция для вычисления метрик классификации ===
def calculate_classification_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return {"F1_macro": float(f1), "Accuracy": float(acc)}


def run_experiment(run_id: int, base_output_dir="results_classifier"):
    """
    Запускает один эксперимент классификации (fit + predict + save) на TwoNormDataset
    и сохраняет результаты в папке run_{run_id}.
    """
    np.random.seed(run_id)

    # Папка для этого прогона
    output_dir = os.path.join(base_output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Загрузка данных ---
    data = BanknoteDataset()
    X = data.get_X()
    y = data.get_y()

    # масштабирование и сплит
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=run_id)

    # Имена признаков
    n_features = X_train.shape[1]
    feature_names = [f"X_{i}" for i in range(n_features)]
    target_name = "class"

    # --- 2. Настройка модели классификации ---
    # количество термов на признак фиксированное
    n_sets = 3
    n_features_sets = [n_sets] * n_features
    max_rules = 10

    labels = [f"set_{i}" for i in range(n_sets)]
    set_names = {name: labels for name in feature_names}

    model = FuzzyClassifier(
        iters=150, pop_size=500, n_features_fuzzy_sets=n_features_sets, max_rules_in_base=max_rules
    )
    model.define_sets(
        X_train,
        y_train.astype(int),
        set_names=set_names,
        feature_names=feature_names,
        target_names=[str(c) for c in np.unique(y_train)],
    )

    # --- 3. Обучение ---
    t0 = time.time()
    model.fit(X_train, y_train.astype(int))
    train_time = time.time() - t0

    # --- 4. Предсказание и сбор метрик ---
    y_pred = model.predict(X_test)
    metrics = calculate_classification_metrics(y_test.astype(int), y_pred)

    # --- 5. Сохранение результатов ---
    # 5.1. Правила
    with open(os.path.join(output_dir, "rule_base.txt"), "w", encoding="utf-8") as f:
        f.write(model.get_text_rules())
    # 5.2. Модель
    with open(os.path.join(output_dir, "fuzzy_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    # 5.3. Предсказания
    preds_df = pd.DataFrame({"true": y_test, "pred": y_pred})
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    # 5.4. Метрики
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    # 5.5. Лог
    with open(os.path.join(output_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Training time: {train_time:.2f} sec\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"[run_{run_id}] done: train {train_time:.1f}s, F1 {metrics['F1_macro']:.4f}")


if __name__ == "__main__":
    N_RUNS = 1
    N_PROCESSES = 1
    os.makedirs("results_classifier", exist_ok=True)
    with mp.Pool(processes=N_PROCESSES) as pool:
        pool.map(run_experiment, range(N_RUNS))

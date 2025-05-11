import os
import time
import pickle
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers._symbolicclassificationgp import SymbolicClassificationGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP


def main():
    # === Параметры эксперимента ===
    number_of_iterations = 300
    population_size = 300
    test_size = 0.25
    random_state = 42

    # === Загрузка и подготовка данных ===
    data = BanknoteDataset()
    X = data.get_X()
    y = data.get_y()

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # === Нормализация признаков ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === Определение модели ===
    model = SymbolicClassificationGP(
        iters=number_of_iterations,
        pop_size=population_size,
        optimizer=SelfCSHAGP,
        optimizer_args={
            "elitism": False,
            "keep_history": True,
            "show_progress_each": 1,
            "max_level": 10,
        },
    )

    # === Обучение ===
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # === Предсказание и оценка ===
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    # Вывод результатов
    print(f"Training time: {train_time:.2f} sec")
    print(f"Test F1-macro: {f1:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # === Сохранение результатов ===
    os.makedirs("results", exist_ok=True)
    # Сохраняем модель
    with open("results/banknote_gp_model.pkl", "wb") as f:
        pickle.dump(model, f)
    # Предсказания
    preds = np.vstack([y_test, y_pred]).T
    np.savetxt(
        "results/banknote_gp_predictions.csv",
        preds,
        delimiter=",",
        header="true,pred",
        comments="",
        fmt="%d",
    )
    # Метрики
    metrics = {"F1_macro": float(f1), "Accuracy": float(acc), "Train_time_sec": train_time}
    print(metrics)
    # При желании можно сохранить метрики в JSON:
    # with open("results/banknote_gp_metrics.json", "w", encoding="utf-8") as f:
    #     json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()

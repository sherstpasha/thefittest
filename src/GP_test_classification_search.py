import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from thefittest.optimizers import SelfCGA
from thefittest.tools.transformations import GrayCode

# Создаём папку для результатов
os.makedirs("results", exist_ok=True)

# --------- Датасет для отладки ---------
datasets = {
    "BreastCancer": load_breast_cancer(return_X_y=True),
}

# --------- Пространство гиперпараметров SVC ---------
# Формат: name: (type, ...), где type in {'float','int','categorical'}.
param_space = {
    "C":          ("float",       1e-3,       1e3),
    "degree":     ("int",         2,          5),
    "kernel":     ("categorical", ["linear","poly","rbf","sigmoid"]),
    "gamma":      ("categorical", ["scale","auto"]),
}

# Каждый параметр (float/int/categorical) занимает ровно 1 позицию
idx = {name: (i, i+1) for i, name in enumerate(param_space)}
vector_length = len(param_space)
print("vector_length =", vector_length)

def vector_to_params(x: np.ndarray) -> dict:
    """
    x: 1-d массив длины vector_length,
       возвращает словарь params для SVC(**params).
    """
    params = {}
    for name, spec in param_space.items():
        lo, _ = idx[name]
        raw = x[lo]

        if spec[0] == "float":
            mn, mx = spec[1], spec[2]
            val = float(np.clip(raw, mn, mx))
            params[name] = val

        elif spec[0] == "int":
            mn, mx = spec[1], spec[2]
            vi = int(round(raw))
            params[name] = int(np.clip(vi, mn, mx))

        else:  # categorical
            choices = spec[1]
            ci = int(round(raw))
            ci = int(np.clip(ci, 0, len(choices) - 1))
            params[name] = choices[ci]

    return params

def run_single_run_svc_hp(dataset_name: str,
                          iteration: int,
                          X_params: np.ndarray,
                          mode: str = 'train') -> np.ndarray:
    """
    X_params: np.ndarray shape (n_candidates, vector_length)
    возвращает np.ndarray из F1-метрик для каждого кандидата.
    mode: 'train' или 'test'
    """
    X, y = datasets[dataset_name]
    X = X.astype(np.float32)
    y = y.astype(int)

    # единый разовый split для всех кандидатов
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=iteration
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = []
    for x in X_params:
        params = vector_to_params(x)
        clf = SVC(random_state=42, **params)
        clf.fit(X_train, y_train)

        if mode == 'train':
            score = f1_score(y_train, clf.predict(X_train), average="macro")
        else:
            score = f1_score(y_test,  clf.predict(X_test),  average="macro")

        results.append(score)

    return np.array(results)


if __name__ == "__main__":
    # Конфигурация для оптимизатора
    conf = {
        'SVC': {
            "fitness": lambda pop: run_single_run_svc_hp("BreastCancer", 1, pop, 'train'),
            "g_to_ph": GrayCode(fit_by="h").fit(
                left  = np.array([1e-3, 2, 0, 0]),
                right = np.array([1e3, 5, 3, 1]),
                arg   = np.array([0.001, 1, 1, 1]),
            ),
        }
    }

    opt = SelfCGA(
        fitness_function      = conf['SVC']['fitness'],
        genotype_to_phenotype = conf['SVC']['g_to_ph'].transform,
        iters                 = 10,
        pop_size              = 10,
        str_len               = sum(conf['SVC']['g_to_ph'].parts),
        show_progress_each    = 1,
    )

    opt.fit()

    # Получаем лучший «сырое» представление и декодируем его
    best_raw = opt.get_fittest()["phenotype"]
    best_params = vector_to_params(best_raw)
    print("Best hyperparameters:", best_params)

    # Оцениваем на train и test с тем же random_state=1
    best_raw_batch = np.array([best_raw])
    train_f1 = run_single_run_svc_hp("BreastCancer", 1, best_raw_batch, 'train')[0]
    test_f1  = run_single_run_svc_hp("BreastCancer", 1, best_raw_batch, 'test')[0]

    print(f"Train  F1-score (should match fitness): {train_f1:.4f}")
    print(f"Test   F1-score:                     {test_f1:.4f}")

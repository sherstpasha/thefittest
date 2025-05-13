import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from thefittest.benchmarks import (
    BanknoteDataset,
    BreastCancerDataset,
    CreditRiskDataset,
    TwoNormDataset,
    RingNormDataset,
)
from thefittest.optimizers import SelfCGA
from thefittest.tools.transformations import GrayCode

# Создаём папку для результатов
os.makedirs("results", exist_ok=True)

# --------- Расширённый список датасетов ---------
datasets = {
    "Banknote":      (BanknoteDataset().get_X(),      BanknoteDataset().get_y()),
    "BreastCancer":  (BreastCancerDataset().get_X(),  BreastCancerDataset().get_y()),
    "CreditRisk":    (CreditRiskDataset().get_X(),    CreditRiskDataset().get_y()),
    "TwoNorm":       (TwoNormDataset().get_X(),       TwoNormDataset().get_y()),
    "RingNorm":      (RingNormDataset().get_X(),      RingNormDataset().get_y()),
}

# --------- Пространство гиперпараметров SVC ---------
param_space = {
    "C":      ("float",       1e-3,       1e3),
    "degree": ("int",         2,          5),
    "kernel": ("categorical", ["linear","poly","rbf","sigmoid"]),
    "gamma":  ("categorical", ["scale","auto"]),
}

# Каждый параметр занимает ровно одну позицию в векторе
idx = {name: (i, i+1) for i, name in enumerate(param_space)}
vector_length = len(param_space)
print("vector_length =", vector_length)

def vector_to_params(x: np.ndarray) -> dict:
    """Декодирует одномерный вектор длины vector_length в словарь параметров."""
    params = {}
    for name, spec in param_space.items():
        lo, _ = idx[name]
        raw = x[lo]
        if spec[0] == "float":
            mn, mx = spec[1], spec[2]
            params[name] = float(np.clip(raw, mn, mx))
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
    X_params: array of shape (n_candidates, vector_length)
    Возвращает array из F1-метрик для каждого кандидата.
    Для mode='train' выполняется 3-fold CV на тренировочной части,
    для mode='test' — оцениваем на отложенной тестовой выборке.
    """
    X, y = datasets[dataset_name]
    X = X.astype(np.float32)
    y = y.astype(int)

    # единый split
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

        if mode == 'train':
            # 3-fold CV на тренировочной части
            cv = KFold(n_splits=3, shuffle=True, random_state=iteration)
            scores = cross_val_score(clf, X_train, y_train,
                                     cv=cv, scoring='f1_macro')
            score = scores.mean()
        else:
            clf.fit(X_train, y_train)
            score = f1_score(y_test, clf.predict(X_test), average="macro")

        results.append(score)
    return np.array(results)

if __name__ == "__main__":
    # Заранее создаём GrayCode одним разом
    g2p = GrayCode(fit_by="h").fit(
        left  = np.array([1e-3, 2, 0, 0]),
        right = np.array([1e3, 5, 3, 1]),
        arg   = np.array([0.001, 1, 1, 1]),
    )

    # Будем хранить результаты для всех датасетов
    summary = []

    for ds_name in datasets:
        print(f"\n=== Optimizing SVC on {ds_name} ===")

        # Лямбда с дефолтным аргументом, чтобы DS фиксировался правильно
        fitness_fn = lambda pop, ds=ds_name: run_single_run_svc_hp(ds, 1, pop, 'train')

        optimizer = SelfCGA(
            fitness_function      = fitness_fn,
            genotype_to_phenotype = g2p.transform,
            iters                 = 10,
            pop_size              = 10,
            str_len               = sum(g2p.parts),
            show_progress_each    = 1,
            optimal_value         = 1.0,
        )
        optimizer.fit()

        # Достаём лучшее решение и декодируем
        best_raw    = optimizer.get_fittest()["phenotype"]
        best_params = vector_to_params(best_raw)

        # Оцениваем: train (CV) и test
        raw_batch = np.array([best_raw])
        train_f1 = run_single_run_svc_hp(ds_name, 1, raw_batch, 'train')[0]
        test_f1  = run_single_run_svc_hp(ds_name, 1, raw_batch, 'test')[0]

        print(f"Dataset: {ds_name}")
        print(" Best params:", best_params)
        print(f" Train F1 (3-fold CV): {train_f1:.4f}, Test F1: {test_f1:.4f}")

        summary.append({
            "dataset": ds_name,
            **best_params,
            "train_f1_cv": train_f1,
            "test_f1":    test_f1
        })

    # Сохраняем итоги в DataFrame и Excel
    df_summary = pd.DataFrame(summary)
    df_summary.to_excel("results/svc_hpo_summary.xlsx", index=False)
    print("\nAll results saved to 'results/svc_hpo_summary.xlsx'")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import traceback
from sklearn.datasets import load_breast_cancer, make_moons, make_circles, fetch_lfw_pairs
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from thefittest.benchmarks import (
    BanknoteDataset,
    BreastCancerDataset,
    CreditRiskDataset,
    TwoNormDataset,
    RingNormDataset,
)

# ---------------------------
# Методы из sklearn
# ---------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ---------------------------
# SymbolicClassificationGP и оптимизаторы из thefittest
# ---------------------------
from thefittest.classifiers._symbolicclassificationgp import SymbolicClassificationGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SelfCGP, PDPGP

import warnings

warnings.filterwarnings("ignore")

# Создаем папку для результатов, если её еще нет
os.makedirs("results", exist_ok=True)

# ---------------------------
# Параметры эксперимента
# ---------------------------
number_of_iterations = 1000  # для SymbolicClassificationGP
population_size = 1000  # для SymbolicClassificationGP
num_runs = 1  # число запусков (итераций) для каждого датасета и каждого метода

# ---------------------------
# Глобальное определение датасетов
# ---------------------------
datasets = {
    "Banknote": (BanknoteDataset().get_X(), BanknoteDataset().get_y()),
    "BreastCancer": (BreastCancerDataset().get_X(), BreastCancerDataset().get_y()),
    "BreastCancer2": (load_breast_cancer().data, load_breast_cancer().target),
    "CreditRisk": (CreditRiskDataset().get_X(), CreditRiskDataset().get_y()),
    "TwoNorm": (TwoNormDataset().get_X(), TwoNormDataset().get_y()),
    "RingNorm": (RingNormDataset().get_X(), RingNormDataset().get_y()),
}

# ---------------------------
# Методы из sklearn: словарь (название: класс)
# ---------------------------
sklearn_methods = {
    # "KNN": KNeighborsClassifier,
    # "DecisionTree": DecisionTreeClassifier,
    # "RandomForest": RandomForestClassifier,
    # "MLP": MLPClassifier,
    # "LogisticRegression": LogisticRegression,
    # "SVC": SVC,
    # "GradientBoosting": GradientBoostingClassifier,
    # "AdaBoost": AdaBoostClassifier,
    # "ExtraTrees": ExtraTreesClassifier,
    # "GaussianNB": GaussianNB,
}

# ---------------------------
# Методы на базе SymbolicClassificationGP
# ---------------------------
symbolic_methods = {
    "SelfCSHAGP": "SelfCSHAGP",
    "PDPSHAGP": "PDPSHAGP",
    "SelfCGP": "SelfCGP",
    "PDPGP": "PDPGP",
}

# Сокращения для листов Excel
abbr = {
    **{m: m[:4].upper() for m in symbolic_methods},
    "KNN": "KNN",
    "DecisionTree": "DT",
    "RandomForest": "RF",
    "MLP": "MLP",
    "LogisticRegression": "LR",
    "SVC": "SVC",
    "GradientBoosting": "GB",
    "AdaBoost": "AB",
    "ExtraTrees": "ET",
    "GaussianNB": "GNB",
}

combined_methods = list(sklearn_methods.keys()) + list(symbolic_methods.keys())


# ---------------------------
# Функция для одного запуска эксперимента для sklearn
# ---------------------------
def run_single_run_sklearn(dataset_name, iteration, method_name):
    try:
        X, y = datasets[dataset_name]
        X = X.astype(np.float32)
        y = y.astype(int)

        # Сплит
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
        # Нормализация для всех sklearn
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = sklearn_methods[method_name]()
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train_f1 = f1_score(y_train, train_pred, average="macro")
        test_f1 = f1_score(y_test, test_pred, average="macro")
        print(
            f"Sklearn: {dataset_name}, iter {iteration}, {method_name} -> train_f1: {train_f1:.4f}, test_f1: {test_f1:.4f}"
        )
        return {
            "dataset": dataset_name,
            "iteration": iteration,
            "method": method_name,
            "train_f1": train_f1,
            "test_f1": test_f1,
        }
    except Exception as e:
        print(f"Error in sklearn {dataset_name} {method_name}: {e}")
        traceback.print_exc()
        return None


# ---------------------------
# Функция для одного запуска эксперимента для GP
# ---------------------------
def run_single_run_symbolic(dataset_name, iteration, method):
    try:
        X, y = datasets[dataset_name]
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
        # Нормализация для GP
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if method == "SelfCSHAGP":
            optimizer_class = SelfCSHAGP
        elif method == "PDPSHAGP":
            optimizer_class = PDPSHAGP
        elif method == "SelfCGP":
            optimizer_class = SelfCGP
        elif method == "PDPGP":
            optimizer_class = PDPGP
        else:
            raise ValueError(f"Unknown symbolic method: {method}")

        model = SymbolicClassificationGP(
            iters=number_of_iterations,
            pop_size=population_size,
            optimizer=optimizer_class,
            optimizer_args={"elitism": False, "keep_history": True, "show_progress_each": 1},
        )
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_f1 = f1_score(y_train, train_pred, average="macro")
        test_f1 = f1_score(y_test, test_pred, average="macro")
        print(
            f"Symbolic: {dataset_name}, iter {iteration}, {method} -> train_f1: {train_f1:.4f}, test_f1: {test_f1:.4f}"
        )
        return {
            "dataset": dataset_name,
            "iteration": iteration,
            "method": method,
            "train_f1": train_f1,
            "test_f1": test_f1,
        }
    except Exception as e:
        print(f"Error in symbolic {dataset_name} {method}: {e}")
        traceback.print_exc()
        return None


# ---------------------------
# Унифицированный запуск
# ---------------------------
def run_single_run(task):
    dataset_name, iteration, method = task
    if method in sklearn_methods:
        return run_single_run_sklearn(dataset_name, iteration, method)
    elif method in symbolic_methods:
        return run_single_run_symbolic(dataset_name, iteration, method)
    else:
        print(f"Unknown method: {method}")
        return None


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    tasks = [
        (ds, it, m) for ds in datasets for it in range(1, num_runs + 1) for m in combined_methods
    ]
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_run, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                print("Executor error:", e)
                traceback.print_exc()

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No results, terminating.")
        exit()
    # ... (остальной код без изменений) ...

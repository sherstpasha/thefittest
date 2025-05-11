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
# Для нормализации данных
# ---------------------------
from sklearn.preprocessing import StandardScaler

# ---------------------------
# SymbolicClassificationGP и оптимизаторы из thefittest
# ---------------------------
from thefittest.classifiers._symbolicclassificationgp import SymbolicClassificationGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import PDPGP

import warnings

warnings.filterwarnings("ignore")

# Создаем папку для результатов, если её еще нет
os.makedirs("results", exist_ok=True)

# ---------------------------
# Параметры эксперимента
# ---------------------------
number_of_iterations = 100  # для SymbolicClassificationGP
population_size = 100  # для SymbolicClassificationGP
num_runs = 30  # число запусков (итераций) для каждого датасета и каждого метода

# ---------------------------
# Глобальное определение датасетов (они будут доступны дочерним процессам)
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
# Методы на базе SymbolicClassificationGP: словарь (название: идентификатор)
# ---------------------------
symbolic_methods = {
    "SelfCSHAGP": "SelfCSHAGP",
    "PDPSHAGP": "PDPSHAGP",
    "SelfCGP": "SelfCGP",
    "PDPGP": "PDPGP",
}

# Для формирования названий листов Excel зададим сокращения для методов
abbr = {}
# Если методов из sklearn нет, то здесь работают только символические:
abbr["SelfCSHAGP"] = "SCSH"
abbr["PDPSHAGP"] = "PDPSH"
abbr["SelfCGP"] = "SCGP"
abbr["PDPGP"] = "PDPGP"
abbr["KNN"] = "KNN"
abbr["DecisionTree"] = "DT"
abbr["RandomForest"] = "RF"
abbr["MLP"] = "MLP"
abbr["LogisticRegression"] = "LR"
abbr["SVC"] = "SVC"
abbr["GradientBoosting"] = "GB"
abbr["AdaBoost"] = "AB"
abbr["ExtraTrees"] = "ET"
abbr["GaussianNB"] = "GNB"

# Объединяем имена методов (в данном примере используются только символические)
combined_methods = list(sklearn_methods.keys()) + list(symbolic_methods.keys())


# ---------------------------
# Функция для одного запуска эксперимента для метода из sklearn
# (если понадобится, можно добавить реализацию)
# ---------------------------
def run_single_run_sklearn(dataset_name, iteration, method_name):
    try:
        X, y = datasets[dataset_name]
        X = X.astype(np.float32)
        y = y.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
        # Нормализуем данные для методов, чувствительных к масштабу
        methods_needing_norm = {"KNN", "SVC", "MLP", "LogisticRegression"}
        if method_name in methods_needing_norm:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        clf = sklearn_methods[method_name]()  # параметры по умолчанию
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
        print(
            f"Error in run_single_run_sklearn for {dataset_name}, iter {iteration}, method {method_name}: {e}"
        )
        traceback.print_exc()
        return None


# ---------------------------
# Функция для одного запуска эксперимента для метода на базе SymbolicClassificationGP
# ---------------------------
def run_single_run_symbolic(dataset_name, iteration, method):
    try:
        X, y = datasets[dataset_name]
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
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
            optimizer_args={"elitism": False, "keep_history": True},
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
        print(
            f"Error in run_single_run_symbolic for {dataset_name}, iter {iteration}, method {method}: {e}"
        )
        traceback.print_exc()
        return None


# ---------------------------
# Унифицированная функция запуска эксперимента
# ---------------------------
def run_single_run(task):
    # task – кортеж: (dataset_name, iteration, method)
    dataset_name, iteration, method = task
    if method in sklearn_methods:
        return run_single_run_sklearn(dataset_name, iteration, method)
    elif method in symbolic_methods:
        return run_single_run_symbolic(dataset_name, iteration, method)
    else:
        print(f"Unknown method: {method}")
        return None


# ---------------------------
# Основной блок (не забудьте, что все должно быть внутри if __name__ == '__main__')
# ---------------------------
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    # Формируем общий список задач для всех методов (sklearn и symbolic)
    tasks = []
    for dataset_name in datasets.keys():
        for iteration in range(1, num_runs + 1):
            for method in combined_methods:
                tasks.append((dataset_name, iteration, method))

    results = []
    # Запускаем все задачи параллельно с помощью ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_run, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print("Error in executor:", e)
                traceback.print_exc()

    # Преобразуем результаты в DataFrame
    df_results = pd.DataFrame(results)
    print("Results DataFrame:")
    print(df_results.head())

    if df_results.empty:
        print("No results collected, terminating.")
    else:
        # ---------------------------
        # Агрегация результатов
        # ---------------------------
        agg_by_dataset_method_test = (
            df_results.groupby(["dataset", "method"])["test_f1"].mean().unstack("method")
        )
        agg_by_dataset_method_train = (
            df_results.groupby(["dataset", "method"])["train_f1"].mean().unstack("method")
        )
        agg_by_iterations_test = df_results.groupby("method")["test_f1"].mean().to_frame().T
        agg_by_iterations_test.index = ["Average"]
        agg_by_iterations_train = df_results.groupby("method")["train_f1"].mean().to_frame().T
        agg_by_iterations_train.index = ["Average"]

        # ---------------------------
        # Сохранение результатов в Excel с подробной статистикой
        # ---------------------------
        excel_path = os.path.join("results", "results.xlsx")
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            # Лист с сырыми результатами
            df_results.to_excel(writer, sheet_name="Raw Results", index=False)
            # Агрегированные результаты по датасетам
            agg_by_dataset_method_test.to_excel(writer, sheet_name="Agg_By_Dataset_Method_Test_F1")
            agg_by_dataset_method_train.to_excel(
                writer, sheet_name="Agg_By_Dataset_Method_Train_F1"
            )
            # Агрегированные результаты по итерациям (усреднение по всем датасетам)
            agg_by_iterations_test.to_excel(writer, sheet_name="Agg_By_Iterations_Test_F1")
            agg_by_iterations_train.to_excel(writer, sheet_name="Agg_By_Iterations_Train_F1")

            # Добавляем листы со статистической проверкой (p-value) для каждой пары алгоритмов.
            for i in range(len(combined_methods)):
                for j in range(i + 1, len(combined_methods)):
                    method1 = combined_methods[i]
                    method2 = combined_methods[j]
                    diff_dict = {}
                    for dataset_name in datasets.keys():
                        df_ds = df_results[df_results["dataset"] == dataset_name]
                        df_pair = df_ds[df_ds["method"].isin([method1, method2])]
                        pivot_df = df_pair.pivot(
                            index="iteration", columns="method", values="test_f1"
                        )
                        if method1 in pivot_df.columns and method2 in pivot_df.columns:
                            d = pivot_df[method1] - pivot_df[method2]
                            diff_dict[dataset_name] = d.values
                        else:
                            diff_dict[dataset_name] = np.array([])
                    pval_matrix = pd.DataFrame(
                        index=list(datasets.keys()), columns=list(datasets.keys()), dtype=float
                    )
                    for ds1 in datasets.keys():
                        for ds2 in datasets.keys():
                            if diff_dict[ds1].size > 0 and diff_dict[ds2].size > 0:
                                stat, pval = ttest_ind(
                                    diff_dict[ds1], diff_dict[ds2], equal_var=False
                                )
                                pval_matrix.loc[ds1, ds2] = pval
                            else:
                                pval_matrix.loc[ds1, ds2] = np.nan
                    sheet_name = f"pv_{abbr[method1]}_{abbr[method2]}"
                    sheet_name = sheet_name[:31]  # ограничение Excel
                    pval_matrix.to_excel(writer, sheet_name=sheet_name)

        print(f"Эксперимент завершен. Результаты сохранены в '{excel_path}'.")

        # ---------------------------
        # Построение графиков распределения F1-score для каждого датасета
        # ---------------------------
        for dataset_name in datasets.keys():
            df_ds = df_results[df_results["dataset"] == dataset_name]
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            df_ds.boxplot(column="train_f1", by="method", ax=ax[0])
            ax[0].set_title("Train F1-score")
            ax[0].set_xlabel("Method")
            ax[0].set_ylabel("F1-score")
            df_ds.boxplot(column="test_f1", by="method", ax=ax[1])
            ax[1].set_title("Test F1-score")
            ax[1].set_xlabel("Method")
            ax[1].set_ylabel("F1-score")
            fig.suptitle(f"Distribution of F1-score for {dataset_name}")
            plt.suptitle("")
            plt.tight_layout()
            fig_path = os.path.join("results", f"{dataset_name}_f1_distribution.png")
            fig.savefig(fig_path)
            plt.close(fig)

        print("Графики распределения F1-score сохранены в папке 'results'.")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from thefittest.benchmarks import (
    BanknoteDataset,
    BreastCancerDataset,
    CreditRiskDataset,
    TwoNormDataset,
    RingNormDataset,
)

# Импорт оптимизаторов
from thefittest.optimizers._selfcshaga import SelfCSHAGA
from thefittest.optimizers import SelfCGA
from thefittest.optimizers._pdpshaga import PDPSHAGA
from thefittest.optimizers import PDPGA

from thefittest.fl2 import FCSelfCGA  # замените на путь к вашему коду

# ---------------------------
# Параметры эксперимента
# ---------------------------
iters = 300
pop_size = 300
n_features_fuzzy_sets = [7]  # пример: 3 терма на каждый признак
max_rules_in_base = 20
num_runs = 5  # число повторов

# Словарь оптимизаторов для эксперимента
optimizers = {
    "SelfCSHAGA": SelfCSHAGA,
    "SelfCGA": SelfCGA,
    "PDPSHAGA": PDPSHAGA,
    "PDPGA": PDPGA,
}

# ---------------------------
# Датасеты для классификации
# ---------------------------
datasets = {
    "Banknote": (BanknoteDataset().get_X(), BanknoteDataset().get_y()),
    "BreastCancer": (BreastCancerDataset().get_X(), BreastCancerDataset().get_y()),
    "CreditRisk": (CreditRiskDataset().get_X(), CreditRiskDataset().get_y()),
    "TwoNorm": (TwoNormDataset().get_X(), TwoNormDataset().get_y()),
    "RingNorm": (RingNormDataset().get_X(), RingNormDataset().get_y()),
}

# ---------------------------
# Функция для одного запуска с FuzzyClassifier
# ---------------------------
def run_single_run_fuzzy(task):
    dataset_name, iteration, opt_name = task
    try:
        X, y = datasets[dataset_name]
        X = X.astype(np.float32)
        y = y.astype(int)
        # Разбиение на train и test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
        # Минимакс-нормализация на основе train
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Создаем классификатор с выбранным оптимизатором
        clf = FCSelfCGA(
            iters=iters,
            pop_size=pop_size,
            #n_features_fuzzy_sets=[n_features_fuzzy_sets[0]] * X.shape[1],
            #max_rules_in_base=max_rules_in_base,
            n_fsets = 7, n_rules = 20,
            optimizer=optimizers[opt_name],
        )
#        clf.define_sets(
#            X,
#            y,
#            feature_names=[f"X_{i}" for i in range(X.shape[1])],
#            target_names=[f"Y_{i}" for i in range(len(np.unique(y_train)))],
#        )
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_f1 = f1_score(y_train, y_pred_train, average="macro")
        test_f1 = f1_score(y_test, y_pred_test, average="macro")
        print(
            f"Fuzzy/{opt_name}: {dataset_name}, run {iteration} -> train_f1: {train_f1:.4f}, test_f1: {test_f1:.4f}"
        )
        return {
            "dataset": dataset_name,
            "optimizer": opt_name,
            "iteration": iteration,
            "train_f1": train_f1,
            "test_f1": test_f1,
        }
    except Exception as e:
        print(f"Error in Fuzzy/{opt_name} on {dataset_name}, run {iteration}: {e}")
        traceback.print_exc()
        return None


# ---------------------------
# Main: сборка задач и параллельный запуск
# ---------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # формируем список задач: все датасеты, все оптимизаторы, все запуски
    tasks = []
    for name in datasets:
        for opt_name in optimizers:
            for i in range(1, num_runs + 1):
                tasks.append((name, i, opt_name))

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_run_fuzzy, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    # Преобразуем в DataFrame
    df_results = pd.DataFrame(results)
    print("Results DataFrame:")
    print(df_results.head())

    # Агрегация по датасетам и оптимизаторам
    agg = (
        df_results.groupby(["dataset", "optimizer"])['test_f1']
        .agg(["mean", "std", "count"])  #
        .unstack("optimizer")
    )
    print("Aggregated results:")
    print(agg)

    # Сохраняем
    excel_path = os.path.join("results", "fuzzy_results.xlsx")
    df_results.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    # Строим боксплоты для каждого датасета
    for dataset_name in datasets:
        df_ds = df_results[df_results["dataset"] == dataset_name]
        plt.figure(figsize=(8, 5))
        df_ds.boxplot(column="test_f1", by="optimizer")
        plt.title(f"F1 Distribution for {dataset_name}")
        plt.suptitle("")
        plt.ylabel("F1-score")
        plt.savefig(os.path.join("results", f"{dataset_name}_fuzzy_f1.png"))
        plt.close()
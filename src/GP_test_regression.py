import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import traceback
from sklearn.datasets import load_diabetes
from thefittest.benchmarks import CombinedCycleDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Методы регрессии из sklearn
# ---------------------------
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# ---------------------------
# SymbolicRegressionGP и оптимизаторы из thefittest
# ---------------------------
from thefittest.regressors import SymbolicRegressionGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.optimizers import SelfCGP, PDPGP

import warnings
warnings.filterwarnings("ignore")

# Создаем папку для результатов
os.makedirs("results", exist_ok=True)

# ---------------------------
# Параметры эксперимента
# ---------------------------
number_of_iterations = 1000
population_size = 300
num_runs = 30

# ---------------------------
# Датасеты для регрессии
# ---------------------------
datasets = {
    "CombinedCycle": (
        CombinedCycleDataset().get_X().astype(np.float32),
        CombinedCycleDataset().get_y().astype(np.float32),
    ),
    "Diabetes": (load_diabetes().data.astype(np.float32), load_diabetes().target.astype(np.float32))
}

# ---------------------------
# Регрессионные методы из sklearn
# ---------------------------
sklearn_methods = {
}

# ---------------------------
# Методы на базе SymbolicRegressionGP
# ---------------------------
symbolic_methods = {
    "SelfCSHAGP": SelfCSHAGP,
    "PDPSHAGP": PDPSHAGP,
    "SelfCGP": SelfCGP,
    "PDPGP": PDPGP,
}

combined_methods = list(sklearn_methods.keys()) + list(symbolic_methods.keys())

# ---------------------------
# Функция для одного запуска sklearn
# ---------------------------
def run_single_run_sklearn(dataset_name, iteration, method_name):
    try:
        X, y = datasets[dataset_name]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
        # Масштабирование признаков
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        # Масштабирование целевой переменной
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        model = sklearn_methods[method_name]()
        model.fit(X_train, y_train_scaled)
        train_pred_scaled = model.predict(X_train)
        test_pred_scaled = model.predict(X_test)
        # Обратное преобразование предсказаний
        train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
        test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(
            f"Sklearn: {dataset_name}, iter {iteration}, {method_name} "
            f"-> train_mse: {train_mse:.4f}, test_mse: {test_mse:.4f}, "
            f"train_r2: {train_r2:.4f}, test_r2: {test_r2:.4f}"
        )
        return {
            "dataset": dataset_name,
            "iteration": iteration,
            "method": method_name,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
    except Exception as e:
        print(f"Error in sklearn {dataset_name} {method_name}: {e}")
        traceback.print_exc()
        return None

# ---------------------------
# Функция для одного запуска SymbolicRegressionGP
# ---------------------------
def run_single_run_symbolic(dataset_name, iteration, method):
    try:
        X, y = datasets[dataset_name]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, random_state=iteration
        )
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        optimizer = symbolic_methods[method]
        model = SymbolicRegressionGP(
            iters=number_of_iterations,
            pop_size=population_size,
            optimizer=optimizer,
            optimizer_args={"elitism": False, "keep_history": True, "max_level": 14},
        )
        model.fit(X_train, y_train_scaled)
        train_pred_scaled = model.predict(X_train)
        test_pred_scaled = model.predict(X_test)
        train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
        test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(
            f"Symbolic: {dataset_name}, iter {iteration}, {method} "
            f"-> train_mse: {train_mse:.4f}, test_mse: {test_mse:.4f}, "
            f"train_r2: {train_r2:.4f}, test_r2: {test_r2:.4f}"
        )
        return {
            "dataset": dataset_name,
            "iteration": iteration,
            "method": method,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
    except Exception as e:
        print(f"Error in symbolic {dataset_name} {method}: {e}")
        traceback.print_exc()
        return None

# ---------------------------
# Универсальный запуск
# ---------------------------
def run_single_run(task):
    dataset_name, iteration, method = task
    if method in sklearn_methods:
        return run_single_run_sklearn(dataset_name, iteration, method)
    return run_single_run_symbolic(dataset_name, iteration, method)

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
        print("No results collected.")
    else:
        # Агрегация метрик
        agg_test_mse = df_results.groupby(["dataset", "method"])["test_mse"].mean().unstack()
        agg_train_mse = df_results.groupby(["dataset", "method"])["train_mse"].mean().unstack()
        agg_test_r2 = df_results.groupby(["dataset", "method"])["test_r2"].mean().unstack()
        agg_train_r2 = df_results.groupby(["dataset", "method"])["train_r2"].mean().unstack()

        avg_test_mse = df_results.groupby("method")["test_mse"].mean().to_frame().T
        avg_train_mse = df_results.groupby("method")["train_mse"].mean().to_frame().T
        avg_test_r2 = df_results.groupby("method")["test_r2"].mean().to_frame().T
        avg_train_r2 = df_results.groupby("method")["train_r2"].mean().to_frame().T
        for df in [avg_test_mse, avg_train_mse, avg_test_r2, avg_train_r2]:
            df.index = ["Average"]

        # Сохранение в Excel
        excel_path = os.path.join("results", "regression_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_results.to_excel(writer, sheet_name="Raw Results", index=False)
            agg_test_mse.to_excel(writer, sheet_name="Agg_Test_MSE")
            agg_train_mse.to_excel(writer, sheet_name="Agg_Train_MSE")
            agg_test_r2.to_excel(writer, sheet_name="Agg_Test_R2")
            agg_train_r2.to_excel(writer, sheet_name="Agg_Train_R2")
            avg_test_mse.to_excel(writer, sheet_name="Average_Test_MSE")
            avg_train_mse.to_excel(writer, sheet_name="Average_Train_MSE")
            avg_test_r2.to_excel(writer, sheet_name="Average_Test_R2")
            avg_train_r2.to_excel(writer, sheet_name="Average_Train_R2")
        print(f"Results saved to {excel_path}")

        # Построение графиков MSE и R2
        for ds in datasets:
            df_ds = df_results[df_results['dataset'] == ds]
            if df_ds.empty:
                print(f"No results for dataset {ds}, skipping plots.")
                continue
            # MSE
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            df_ds.boxplot(column='train_mse', by='method', ax=axes[0])
            axes[0].set_title('Train MSE')
            axes[0].set_xlabel('Method')
            axes[0].set_ylabel('MSE')
            df_ds.boxplot(column='test_mse', by='method', ax=axes[1])
            axes[1].set_title('Test MSE')
            axes[1].set_xlabel('Method')
            axes[1].set_ylabel('MSE')
            plt.suptitle(f'MSE Distribution for {ds}')
            plt.tight_layout()
            plt.savefig(os.path.join('results', f'{ds}_mse_distribution.png'))
            plt.close(fig)
            # R2
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            df_ds.boxplot(column='train_r2', by='method', ax=axes[0])
            axes[0].set_title('Train R2')
            axes[0].set_xlabel('Method')
            axes[0].set_ylabel('R2')
            df_ds.boxplot(column='test_r2', by='method', ax=axes[1])
            axes[1].set_title('Test R2')
            axes[1].set_xlabel('Method')
            axes[1].set_ylabel('R2')
            plt.suptitle(f'R2 Distribution for {ds}')
            plt.tight_layout()
            plt.savefig(os.path.join('results', f'{ds}_r2_distribution.png'))
            plt.close(fig)
        print("Plots saved in 'results' folder.")

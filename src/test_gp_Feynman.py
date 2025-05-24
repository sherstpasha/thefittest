import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers import SelfCGP, PDPGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.regressors._symbolicregressiongp_dual import SymbolicRegressionGP_DUAL
import concurrent.futures
from concurrent.futures import wait, FIRST_COMPLETED
from collections import deque
from scipy.stats import ttest_ind  # Для статистической проверки
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # Игнорируем предупреждения

# Параметры алгоритма и эксперимента
number_of_iterations = 1000
population_size = 100
num_runs = 100  # число запусков для каждого файла (N)

# Путь к папке с данными (файлы без расширения)
data_folder = r"C:\Users\pasha\OneDrive\Рабочий стол\Feynman120"
# Получаем список файлов в папке
files_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
print("Найденные файлы:", files_list)

# Список методов (оптимизаторы) и их сокращения
methods = ["SelfCGP", "PDPGP", "SelfCSHAGP", "PDPSHAGP"]
abbr = {"SelfCGP": "SCGP", "PDPGP": "PDPG", "SelfCSHAGP": "SCSH", "PDPSHAGP": "PDPSH"}


def run_single_run(file_name, iteration, number_of_iterations, population_size, method):
    """
    Один запуск эксперимента для одного файла, итерации и метода.
    """
    file_path = os.path.join(data_folder, file_name)
    data = np.loadtxt(file_path).astype(np.float32)

    X = data[:1000, :-1]
    y = data[:1000, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=iteration
    )

    # Выбор оптимизатора
    optim_map = {
        "CSHAGP": CSHAGP,
        "SelfCGP": SelfCGP,
        "PDPGP": PDPGP,
        "SelfCSHAGP": SelfCSHAGP,
        "PDPSHAGP": PDPSHAGP,
    }
    optimizer_class = optim_map.get(method)
    if optimizer_class is None:
        raise ValueError(f"Unknown method: {method}")

    model = SymbolicRegressionGP_DUAL(
        iters=number_of_iterations,
        pop_size=population_size,
        optimizer=optimizer_class,
        optimizer_args={"elitism": False, "keep_history": True},
    )
    model.fit(X_train, y_train)

    tree = model.get_optimizer().get_fittest()["genotype"].__str__()
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return {
        "iteration": iteration,
        "file": file_name,
        "method": method,
        "tree": tree,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


if __name__ == "__main__":
    # Готовим очередь задач
    tasks_queue = deque(
        (f, i, m) for f in files_list for i in range(1, num_runs + 1) for m in methods
    )
    results = []

    # Параллельный запуск с поддержанием ровно 10 активных задач
    max_workers = 10
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем первые задачи
        futures = {
            executor.submit(run_single_run, f, i, number_of_iterations, population_size, m): (
                f,
                i,
                m,
            )
            for f, i, m in [
                tasks_queue.popleft() for _ in range(min(max_workers, len(tasks_queue)))
            ]
        }

        with tqdm(
            total=len(files_list) * num_runs * len(methods), desc="Обработка экспериментов"
        ) as pbar:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        results.append(fut.result())
                    except Exception:
                        pass
                    pbar.update(1)
                    del futures[fut]

                    # Добавляем новую задачу, если остались
                    if tasks_queue:
                        f, i, m = tasks_queue.popleft()
                        futures[
                            executor.submit(
                                run_single_run, f, i, number_of_iterations, population_size, m
                            )
                        ] = (f, i, m)

    # ---- Аггрегация результатов ----
    df_results = pd.DataFrame(results)

    # RMSE и R2 агрегаты по файлам и методам
    agg_test_rmse = df_results.groupby(["file", "method"])["test_rmse"].mean().unstack()
    agg_train_rmse = df_results.groupby(["file", "method"])["train_rmse"].mean().unstack()
    agg_test_r2 = df_results.groupby(["file", "method"])["test_r2"].mean().unstack()
    agg_train_r2 = df_results.groupby(["file", "method"])["train_r2"].mean().unstack()

    # Средние по всем итерациям
    avg_test_rmse = df_results.groupby("method")["test_rmse"].mean().to_frame().T
    avg_train_rmse = df_results.groupby("method")["train_rmse"].mean().to_frame().T
    avg_test_r2 = df_results.groupby("method")["test_r2"].mean().to_frame().T
    avg_train_r2 = df_results.groupby("method")["train_r2"].mean().to_frame().T

    # Создаём папки для сохранения
    results_dir = os.path.join("results")
    csv_dir = os.path.join(results_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Сохраняем CSV
    df_results.to_csv(os.path.join(csv_dir, "Raw_Results.csv"), index=False)
    agg_test_rmse.to_csv(os.path.join(csv_dir, "Agg_By_File_Method_Test_RMSE.csv"))
    agg_train_rmse.to_csv(os.path.join(csv_dir, "Agg_By_File_Method_Train_RMSE.csv"))
    avg_test_rmse.to_csv(os.path.join(csv_dir, "Agg_By_Iterations_Test_RMSE.csv"))
    avg_train_rmse.to_csv(os.path.join(csv_dir, "Agg_By_Iterations_Train_RMSE.csv"))
    agg_test_r2.to_csv(os.path.join(csv_dir, "Agg_By_File_Method_Test_R2.csv"))
    agg_train_r2.to_csv(os.path.join(csv_dir, "Agg_By_File_Method_Train_R2.csv"))
    avg_test_r2.to_csv(os.path.join(csv_dir, "Agg_By_Iterations_Test_R2.csv"))
    avg_train_r2.to_csv(os.path.join(csv_dir, "Agg_By_Iterations_Train_R2.csv"))

    # P-value матрицы для каждой пары методов
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            diffs = {}
            for f in files_list:
                df_f = df_results[df_results["file"] == f]
                pivot = df_f.pivot(index="iteration", columns="method", values="test_rmse")
                if m1 in pivot and m2 in pivot:
                    diffs[f] = pivot[m1] - pivot[m2]
                else:
                    diffs[f] = np.array([])

            pval_mat = pd.DataFrame(index=files_list, columns=files_list, dtype=float)
            for f1 in files_list:
                for f2 in files_list:
                    if diffs[f1].size and diffs[f2].size:
                        _, p = ttest_ind(diffs[f1], diffs[f2], equal_var=False)
                        pval_mat.loc[f1, f2] = p
            name = f"pv_{abbr[m1]}_{abbr[m2]}.csv"
            pval_mat.to_csv(os.path.join(csv_dir, name))

    # Запись в Excel
    excel_path = os.path.join(results_dir, "results.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="Raw Results", index=False)
        agg_test_rmse.to_excel(writer, sheet_name="Agg_Test_RMSE")
        agg_train_rmse.to_excel(writer, sheet_name="Agg_Train_RMSE")
        avg_test_rmse.to_excel(writer, sheet_name="Avg_Test_RMSE")
        avg_train_rmse.to_excel(writer, sheet_name="Avg_Train_RMSE")
        agg_test_r2.to_excel(writer, sheet_name="Agg_Test_R2")
        agg_train_r2.to_excel(writer, sheet_name="Agg_Train_R2")
        avg_test_r2.to_excel(writer, sheet_name="Avg_Test_R2")
        avg_train_r2.to_excel(writer, sheet_name="Avg_Train_R2")

    print(f"Эксперимент завершён. Результаты сохранены в '{excel_path}'.")

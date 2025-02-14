import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import PDPGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.regressors import SymbolicRegressionGP
import concurrent.futures
from scipy.stats import ttest_ind  # Для статистической проверки
from tqdm import tqdm  # Импорт прогресс-бара

import warnings

warnings.filterwarnings("ignore")  # Игнорируем предупреждения

# Создаем папку для результатов, если она не существует
os.makedirs("results", exist_ok=True)

# Параметры алгоритма и эксперимента
number_of_iterations = 1000
population_size = 100
num_runs = 100 # число запусков для каждого файла (N)

# Путь к папке с данными (файлы без расширения)
data_folder = r"C:\Users\USER\Desktop\Feynman120"
# Получаем список файлов в папке (файлы не имеют расширения, но их можно прочитать с помощью np.loadtxt)
files_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
print("Найденные файлы:", files_list)

# Список методов (оптимизаторов)
methods = ["SelfCGP", "PDPGP", "SelfCSHAGP", "PDPSHAGP"]

# Словарь сокращений для названий методов (для имен листов/файлов)
abbr = {"SelfCGP": "SCGP", "PDPGP": "PDPG", "SelfCSHAGP": "SCSH", "PDPSHAGP": "PDPSH"}


def run_single_run(file_name, iteration, number_of_iterations, population_size, method):
    """
    Выполняет один запуск эксперимента для данных из файла file_name с заданным номером итерации и методом.
    Данные загружаются из файла, формируется выборка (X, y) (берутся первые 100 строк),
    затем производится разбиение на обучающую и тестовую выборки, обучение модели,
    вычисление метрик и извлечение найденного дерева (генотипа).
    """
    # Полный путь к файлу
    file_path = os.path.join(data_folder, file_name)
    # Загружаем данные из файла (файл читается как текстовый, np.loadtxt способен читать файлы без расширения)
    data = np.loadtxt(file_path).astype(np.float32)

    # Формируем выборку: берем первые 100 строк,
    # X – все столбцы, кроме последнего; y – последний столбец
    X = data[:1000, :-1]
    y = data[:1000, -1]

    # Делим данные на обучающую (75%) и тестовую (25%) выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=iteration
    )

    # Выбираем оптимизатор по переданному методу
    if method == "CSHAGP":
        optimizer_class = CSHAGP
    elif method == "SelfCGP":
        optimizer_class = SelfCGP
    elif method == "PDPGP":
        optimizer_class = PDPGP
    elif method == "SelfCSHAGP":
        optimizer_class = SelfCSHAGP
    elif method == "PDPSHAGP":
        optimizer_class = PDPSHAGP
    else:
        raise ValueError(f"Unknown method: {method}")

    # Инициализируем и обучаем модель символической регрессии
    model = SymbolicRegressionGP(
        iters=number_of_iterations,
        pop_size=population_size,
        optimizer=optimizer_class,
        optimizer_args={"elitism": False, "keep_history": True},
    )
    model.fit(X_train, y_train)

    # Извлекаем строковое представление лучшего найденного дерева (генотипа)
    tree = model.get_optimizer().get_fittest()["genotype"].__str__()

    # Получаем предсказания для обучающей и тестовой выборок
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Возвращаем результаты вместе со строковым представлением дерева
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
    results = []
    tasks = []

    # Формируем список задач: для каждого файла, для каждого запуска (итерация) и для каждого метода
    # Здесь для примера берутся первые 20 файлов (при необходимости можно обработать все)
    for file_name in files_list[:]:
        for iteration in range(1, num_runs + 1):
            for method in methods:
                tasks.append((file_name, iteration, method))

    # Параллельное выполнение экспериментов с отображением прогресс-бара
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_run,
                file_name,
                iteration,
                number_of_iterations,
                population_size,
                method,
            )
            for file_name, iteration, method in tasks
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Обработка экспериментов",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Здесь можно добавить логирование ошибок
                pass

    # Преобразуем результаты в DataFrame (метрики и дерево для каждого запуска)
    df_results = pd.DataFrame(results)

    # Агрегация RMSE по файлам (строки – файлы, столбцы – методы)
    agg_by_file_method_test = (
        df_results.groupby(["file", "method"])["test_rmse"].mean().unstack("method")
    )
    agg_by_file_method_train = (
        df_results.groupby(["file", "method"])["train_rmse"].mean().unstack("method")
    )

    # Агрегация RMSE по итерациям (усреднение по всем файлам и итерациям)
    agg_by_iterations_test = df_results.groupby("method")["test_rmse"].mean().to_frame().T
    agg_by_iterations_test.index = ["Average"]
    agg_by_iterations_train = df_results.groupby("method")["train_rmse"].mean().to_frame().T
    agg_by_iterations_train.index = ["Average"]

    # Агрегация R2 по файлам (строки – файлы, столбцы – методы)
    agg_by_file_method_test_r2 = (
        df_results.groupby(["file", "method"])["test_r2"].mean().unstack("method")
    )
    agg_by_file_method_train_r2 = (
        df_results.groupby(["file", "method"])["train_r2"].mean().unstack("method")
    )

    # Агрегация R2 по итерациям (усреднение по всем файлам и итерациям)
    agg_by_iterations_test_r2 = df_results.groupby("method")["test_r2"].mean().to_frame().T
    agg_by_iterations_test_r2.index = ["Average"]
    agg_by_iterations_train_r2 = df_results.groupby("method")["train_r2"].mean().to_frame().T
    agg_by_iterations_train_r2.index = ["Average"]

    # ==================================================================
    # Сначала сохраняем все результаты в CSV-файлы

    csv_folder = os.path.join("results", "csv")
    os.makedirs(csv_folder, exist_ok=True)

    # Основные DataFrame
    df_results.to_csv(os.path.join(csv_folder, "Raw_Results.csv"), index=False)
    agg_by_file_method_test.to_csv(os.path.join(csv_folder, "Agg_By_File_Method_Test_RMSE.csv"))
    agg_by_file_method_train.to_csv(os.path.join(csv_folder, "Agg_By_File_Method_Train_RMSE.csv"))
    agg_by_iterations_test.to_csv(os.path.join(csv_folder, "Agg_By_Iterations_Test_RMSE.csv"))
    agg_by_iterations_train.to_csv(os.path.join(csv_folder, "Agg_By_Iterations_Train_RMSE.csv"))
    agg_by_file_method_test_r2.to_csv(os.path.join(csv_folder, "Agg_By_File_Method_Test_R2.csv"))
    agg_by_file_method_train_r2.to_csv(os.path.join(csv_folder, "Agg_By_File_Method_Train_R2.csv"))
    agg_by_iterations_test_r2.to_csv(os.path.join(csv_folder, "Agg_By_Iterations_Test_R2.csv"))
    agg_by_iterations_train_r2.to_csv(os.path.join(csv_folder, "Agg_By_Iterations_Train_R2.csv"))

    # CSV для p-value листов для каждой пары методов
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            diff_dict = {}
            for file_name in files_list:
                df_file = df_results[df_results["file"] == file_name]
                df_pair = df_file[df_file["method"].isin([method1, method2])]
                pivot_df = df_pair.pivot(index="iteration", columns="method", values="test_rmse")
                if method1 in pivot_df.columns and method2 in pivot_df.columns:
                    d = pivot_df[method1] - pivot_df[method2]
                    diff_dict[file_name] = d.values
                else:
                    diff_dict[file_name] = np.array([])

            pval_matrix = pd.DataFrame(index=files_list, columns=files_list, dtype=float)
            for f1 in files_list:
                for f2 in files_list:
                    if diff_dict[f1].size > 0 and diff_dict[f2].size > 0:
                        stat, pval = ttest_ind(diff_dict[f1], diff_dict[f2], equal_var=False)
                        pval_matrix.loc[f1, f2] = pval
                    else:
                        pval_matrix.loc[f1, f2] = np.nan

            sheet_name = f"pv_{abbr.get(method1, method1)}_{abbr.get(method2, method2)}"
            csv_path = os.path.join(csv_folder, f"{sheet_name}.csv")
            pval_matrix.to_csv(csv_path)

    print(f"CSV-версии всех листов сохранены в папке '{csv_folder}'.")

    # ==================================================================
    # Теперь сохраняем результаты в один Excel-файл
    excel_path = os.path.join("results", "results.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # Лист с сырыми результатами (включая поле "tree")
        df_results.to_excel(writer, sheet_name="Raw Results", index=False)
        # RMSE агрегаты
        agg_by_file_method_test.to_excel(writer, sheet_name="Agg_By_File_Method_Test_RMSE")
        agg_by_file_method_train.to_excel(writer, sheet_name="Agg_By_File_Method_Train_RMSE")
        agg_by_iterations_test.to_excel(writer, sheet_name="Agg_By_Iterations_Test_RMSE")
        agg_by_iterations_train.to_excel(writer, sheet_name="Agg_By_Iterations_Train_RMSE")
        # R2 агрегаты
        agg_by_file_method_test_r2.to_excel(writer, sheet_name="Agg_By_File_Method_Test_R2")
        agg_by_file_method_train_r2.to_excel(writer, sheet_name="Agg_By_File_Method_Train_R2")
        agg_by_iterations_test_r2.to_excel(writer, sheet_name="Agg_By_Iterations_Test_R2")
        agg_by_iterations_train_r2.to_excel(writer, sheet_name="Agg_By_Iterations_Train_R2")

        # Дополнительный лист: p-value для каждой пары методов
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                diff_dict = {}
                for file_name in files_list:
                    df_file = df_results[df_results["file"] == file_name]
                    df_pair = df_file[df_file["method"].isin([method1, method2])]
                    pivot_df = df_pair.pivot(
                        index="iteration", columns="method", values="test_rmse"
                    )
                    if method1 in pivot_df.columns and method2 in pivot_df.columns:
                        d = pivot_df[method1] - pivot_df[method2]
                        diff_dict[file_name] = d.values
                    else:
                        diff_dict[file_name] = np.array([])

                pval_matrix = pd.DataFrame(index=files_list, columns=files_list, dtype=float)
                for f1 in files_list:
                    for f2 in files_list:
                        if diff_dict[f1].size > 0 and diff_dict[f2].size > 0:
                            stat, pval = ttest_ind(diff_dict[f1], diff_dict[f2], equal_var=False)
                            pval_matrix.loc[f1, f2] = pval
                        else:
                            pval_matrix.loc[f1, f2] = np.nan

                sheet_name = f"pv_{abbr.get(method1, method1)}_{abbr.get(method2, method2)}"
                pval_matrix.to_excel(writer, sheet_name=sheet_name)

    print(f"Эксперимент завершён. Результаты сохранены в '{excel_path}'.")

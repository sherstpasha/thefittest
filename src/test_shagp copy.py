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
from thefittest.benchmarks.symbolicregression17 import problems_dict
import concurrent.futures
from scipy.stats import ttest_ind  # Для статистической проверки

import warnings

warnings.filterwarnings("ignore")

# Создаем папку для результатов, если она не существует
os.makedirs("results", exist_ok=True)

# Параметры алгоритма и эксперимента
number_of_iterations = 1000
population_size = 100
sample_size = 300  # число точек для формирования выборки
num_runs = 100  # число запусков для каждой функции (N)
noise_percentage = 0.1  # добавляем шум 5% (уровень шума)

# Список тестовых функций: F1, F2, ..., F17
functions_list = [f"F{i}" for i in range(1, 18)]
# Список методов (оптимизаторов)
methods = ["SelfCGP", "PDPGP", "SelfCSHAGP", "PDPSHAGP"]

# Словарь сокращений для названий методов, чтобы итоговое имя листа не превышало 31 символа.
abbr = {"SelfCGP": "SCGP", "PDPGP": "PDPG", "SelfCSHAGP": "SCSH", "PDPSHAGP": "PDPSH"}


def run_single_run(F, iteration, sample_size, number_of_iterations, population_size, method):
    """
    Выполняет один запуск эксперимента для функции F с заданным номером итерации и методом.
    Данные делятся на обучающую (75%) и тестовую (25%) выборки, модель обучается,
    затем вычисляются предсказания для обеих выборок и рассчитываются метрики RMSE и R2.
    При этом к целевым значениям (y) добавляется 5%-ный шум.
    """
    # Получаем параметры для функции F
    func_info = problems_dict[F]
    # Функция возвращает значения типа float32
    function = lambda x: func_info["function"](x).astype(np.float32)
    left_border, right_border = func_info["bounds"]
    n_dimension = func_info["n_vars"]

    # Формируем исходную выборку
    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T.astype(np.float32)
    y = function(X)

    # Добавляем шум: генерируем Гауссов шум с масштабом 5% от std(y)
    noise = np.random.normal(loc=0, scale=noise_percentage * np.std(y), size=y.shape).astype(
        np.float32
    )
    y = y + noise

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

    # Инициализируем и обучаем модель
    model = SymbolicRegressionGP(
        iters=number_of_iterations,
        pop_size=population_size,
        optimizer=optimizer_class,
        optimizer_args={"elitism": False, "keep_history": True},
    )
    model.fit(X_train, y_train)

    # Получаем предсказания для обучающей и тестовой выборок
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(
        f"Function {F} | Iteration {iteration} | Method {method} | "
        f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | "
        f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}"
    )

    return {
        "iteration": iteration,
        "F": F,
        "method": method,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


if __name__ == "__main__":
    results = []
    tasks = []

    # Формируем список задач: для каждой функции, для каждого запуска и для каждого метода
    for F in functions_list:
        for iteration in range(1, num_runs + 1):
            for method in methods:
                tasks.append((F, iteration, method))

    # Параллельное выполнение экспериментов
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                run_single_run,
                F,
                iteration,
                sample_size,
                number_of_iterations,
                population_size,
                method,
            )
            for F, iteration, method in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print("An error occurred:", e)

    # Преобразуем результаты в DataFrame (метрики для каждого запуска)
    df_results = pd.DataFrame(results)

    # Агрегация RMSE по функциям (строки – функции, столбцы – методы)
    agg_by_F_method_test = df_results.groupby(["F", "method"])["test_rmse"].mean().unstack("method")
    agg_by_F_method_train = (
        df_results.groupby(["F", "method"])["train_rmse"].mean().unstack("method")
    )

    # Агрегация RMSE по итерациям (усреднение по всем функциям и итерациям)
    agg_by_iterations_test = df_results.groupby("method")["test_rmse"].mean().to_frame().T
    agg_by_iterations_test.index = ["Average"]
    agg_by_iterations_train = df_results.groupby("method")["train_rmse"].mean().to_frame().T
    agg_by_iterations_train.index = ["Average"]

    # Агрегация R2 по функциям (строки – функции, столбцы – методы)
    agg_by_F_method_test_r2 = (
        df_results.groupby(["F", "method"])["test_r2"].mean().unstack("method")
    )
    agg_by_F_method_train_r2 = (
        df_results.groupby(["F", "method"])["train_r2"].mean().unstack("method")
    )

    # Агрегация R2 по итерациям (усреднение по всем функциям и итерациям)
    agg_by_iterations_test_r2 = df_results.groupby("method")["test_r2"].mean().to_frame().T
    agg_by_iterations_test_r2.index = ["Average"]
    agg_by_iterations_train_r2 = df_results.groupby("method")["train_r2"].mean().to_frame().T
    agg_by_iterations_train_r2.index = ["Average"]

    # Сохраняем результаты в Excel-файл (в папке results)
    excel_path = os.path.join("results", "results.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # Лист с сырыми результатами
        df_results.to_excel(writer, sheet_name="Raw Results", index=False)
        # RMSE агрегаты
        agg_by_F_method_test.to_excel(writer, sheet_name="Agg_By_F_Method_Test_RMSE")
        agg_by_F_method_train.to_excel(writer, sheet_name="Agg_By_F_Method_Train_RMSE")
        agg_by_iterations_test.to_excel(writer, sheet_name="Agg_By_Iterations_Test_RMSE")
        agg_by_iterations_train.to_excel(writer, sheet_name="Agg_By_Iterations_Train_RMSE")
        # R2 агрегаты
        agg_by_F_method_test_r2.to_excel(writer, sheet_name="Agg_By_F_Method_Test_R2")
        agg_by_F_method_train_r2.to_excel(writer, sheet_name="Agg_By_F_Method_Train_R2")
        agg_by_iterations_test_r2.to_excel(writer, sheet_name="Agg_By_Iterations_Test_R2")
        agg_by_iterations_train_r2.to_excel(writer, sheet_name="Agg_By_Iterations_Train_R2")

        # Добавляем листы со статистической проверкой (p-value) для каждой пары алгоритмов.
        # Для каждой пары алгоритмов рассчитывается матрица, где строки и столбцы – тестовые функции (F1..F17)
        # Ячейка (F1, F2) содержит p-value теста (Welch's t-test) для проверки равенства распределений разностей
        # test_rmse между алгоритмами на функциях F1 и F2.
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                # Для каждой функции вычисляем вектор разностей (по num_runs запускам) между test_rmse двух алгоритмов
                diff_dict = {}
                for F in functions_list:
                    df_func = df_results[df_results["F"] == F]
                    df_pair = df_func[df_func["method"].isin([method1, method2])]
                    # Пивотируем по iteration, чтобы сопоставить результаты одного запуска
                    pivot_df = df_pair.pivot(
                        index="iteration", columns="method", values="test_rmse"
                    )
                    if method1 in pivot_df.columns and method2 in pivot_df.columns:
                        d = pivot_df[method1] - pivot_df[method2]
                        diff_dict[F] = d.values  # numpy-массив длины num_runs
                    else:
                        diff_dict[F] = np.array([])

                # Формируем матрицу p-value (индексы и столбцы – названия функций)
                pval_matrix = pd.DataFrame(
                    index=functions_list, columns=functions_list, dtype=float
                )
                for F1 in functions_list:
                    for F2 in functions_list:
                        if diff_dict[F1].size > 0 and diff_dict[F2].size > 0:
                            # Независимый Welch's t-test для проверки равенства средних двух выборок
                            stat, pval = ttest_ind(diff_dict[F1], diff_dict[F2], equal_var=False)
                            pval_matrix.loc[F1, F2] = pval
                        else:
                            pval_matrix.loc[F1, F2] = np.nan

                # Используем сокращенные имена алгоритмов для формирования имени листа
                sheet_name = f"pv_{abbr[method1]}_{abbr[method2]}"
                pval_matrix.to_excel(writer, sheet_name=sheet_name)
    print(f"Эксперимент завершён. Результаты сохранены в '{excel_path}'.")

    # Построение графиков распределения RMSE для каждой функции по всем алгоритмам.
    # Для каждой функции строим один рисунок с двумя subplot'ами:
    # левый: распределение train_rmse, правый: распределение test_rmSE (boxplot по методам)
    for F in functions_list:
        df_func = df_results[df_results["F"] == F]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # Boxplot для обучающей выборки
        df_func.boxplot(column="train_rmse", by="method", ax=ax[0])
        ax[0].set_title("Train RMSE")
        ax[0].set_xlabel("Method")
        ax[0].set_ylabel("RMSE")
        # Boxplot для тестовой выборки
        df_func.boxplot(column="test_rmse", by="method", ax=ax[1])
        ax[1].set_title("Test RMSE")
        ax[1].set_xlabel("Method")
        ax[1].set_ylabel("RMSE")
        # Заголовок всего рисунка
        fig.suptitle(f"Distribution of RMSE for {F}")
        plt.suptitle("")  # отключаем автоматически генерируемый заголовок
        plt.tight_layout()
        # Сохраняем рисунок в папку results
        fig_path = os.path.join("results", f"{F}_rmse_distribution.png")
        fig.savefig(fig_path)
        plt.close(fig)
    print("Графики распределения RMSE сохранены в папке 'results'.")

    # ===============================================
    # Дополнительный блок: Отрисовка графиков функций с шумом.
    # Для каждой функции генерируется выборка (аналогичная run_single_run),
    # к y добавляется шум с фиксированным сидом (42) и строится график зависимости от первой переменной.
    for F in functions_list:
        func_info = problems_dict[F]
        function = lambda x: func_info["function"](x).astype(np.float32)
        left_border, right_border = func_info["bounds"]
        n_dimension = func_info["n_vars"]

        # Формируем выборку по аналогии с run_single_run
        X = np.array(
            [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
        ).T.astype(np.float32)
        y = function(X)

        # Устанавливаем фиксированный сид для шума
        np.random.seed(42)
        noise = np.random.normal(loc=0, scale=noise_percentage * np.std(y), size=y.shape).astype(
            np.float32
        )
        y_noisy = y + noise

        plt.figure(figsize=(8, 6))
        # Если переменных 1 или 2, строим график по первой координате.
        plt.plot(X[:, 0], y_noisy, label=f"{F} with noise")
        plt.title(f"Function {F} with Noise")
        plt.xlabel("x (first coordinate)")
        plt.ylabel("y")
        plt.legend()
        plot_path = os.path.join("results", f"{F}_function_with_noise.png")
        plt.savefig(plot_path)
        plt.close()
    print("Графики функций с шумом сохранены в папке 'results'.")

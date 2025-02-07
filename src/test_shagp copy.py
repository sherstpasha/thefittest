import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import PDPGP
from thefittest.regressors import SymbolicRegressionGP
from thefittest.benchmarks.symbolicregression17 import problems_dict
import concurrent.futures
from thefittest.optimizers._selfcshagp import SelfCSHAGP


# Параметры алгоритма и эксперимента
number_of_iterations = 200
population_size = 200
sample_size = 300  # число точек для формирования выборки
num_runs = 100  # число запусков для каждой функции (N)

# Список функций: F1, F2, ..., F17
functions_list = [f"F{i}" for i in range(1, 18)]
# Список методов (оптимизаторов)
methods = ["CSHAGP", "SelfCGP", "PDPGP", "SelfCSHAGP"]


def run_single_run(F, iteration, sample_size, number_of_iterations, population_size, method):
    """
    Выполняет один запуск эксперимента для функции F с указанным номером итерации и методом.
    """
    # Получаем параметры для функции F
    func_info = problems_dict[F]
    # Функция должна возвращать значения типа float32
    function = lambda x: func_info["function"](x).astype(np.float32)
    left_border, right_border = func_info["bounds"]
    n_dimension = func_info["n_vars"]

    # Создаем выборку: для каждой переменной sample_size равномерно распределённых точек
    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T.astype(np.float32)
    y = function(X)

    # Разбиваем данные на обучающую (75%) и тестовую (25%) выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=iteration
    )

    # Выбираем оптимизатор согласно переданному методу
    if method == "CSHAGP":
        optimizer_class = CSHAGP
    elif method == "SelfCGP":
        optimizer_class = SelfCGP
    elif method == "PDPGP":
        optimizer_class = PDPGP
    elif method == "SelfCSHAGP":
        optimizer_class = SelfCSHAGP
    else:
        raise ValueError(f"Unknown method: {method}")

    # Инициализируем и обучаем модель с выбранным оптимизатором
    model = SymbolicRegressionGP(
        iters=number_of_iterations,
        pop_size=population_size,
        optimizer=optimizer_class,
        optimizer_args={"show_progress_each": 10, "keep_history": True},
    )
    model.fit(X_train, y_train)

    # Получаем прогноз на тестовой выборке и вычисляем RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Function {F} | Iteration {iteration} | Method {method} | RMSE: {rmse:.4f}")
    return {"iteration": iteration, "F": F, "method": method, "rmse": rmse}


if __name__ == "__main__":
    results = []
    tasks = []

    # Формируем список задач: для каждой функции, для каждого запуска и для каждого метода
    for F in functions_list:
        for iteration in range(1, num_runs + 1):
            for method in methods:
                tasks.append((F, iteration, method))

    # Параллельное выполнение экспериментов с использованием ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
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

    # Преобразуем результаты в DataFrame
    df_results = pd.DataFrame(results)

    # Агрегирование по функциям: строки – функции, столбцы – методы
    agg_by_F_method = df_results.groupby(["F", "method"])["rmse"].mean().unstack("method")

    # Агрегация по итерациям: усредняем по всем итерациям (и функциям) для каждого метода
    # Получаем один ряд со средним значением RMSE для каждого метода
    agg_by_iterations = df_results.groupby("method")["rmse"].mean().to_frame().T
    agg_by_iterations.index = ["Average"]

    # Сохраняем результаты в один Excel-файл с тремя листами:
    # "Raw Results", "Agg_By_F_Method" и "Agg_By_Iterations"
    with pd.ExcelWriter("results.xlsx", engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="Raw Results", index=False)
        agg_by_F_method.to_excel(writer, sheet_name="Agg_By_F_Method")
        agg_by_iterations.to_excel(writer, sheet_name="Agg_By_Iterations")

    print("Эксперимент завершён. Результаты сохранены в 'results.xlsx'.")

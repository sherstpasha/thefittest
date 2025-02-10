import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import PDPGP
from thefittest.optimizers._selfcshagp import SelfCSHAGP
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.regressors import SymbolicRegressionGP
from thefittest.benchmarks.symbolicregression17 import problems_dict
import concurrent.futures
import warnings

warnings.filterwarnings("ignore")

# Создаем папку для результатов, если её нет
os.makedirs("results", exist_ok=True)

# Параметры эксперимента
number_of_iterations = 300
population_size = 100
sample_size = 20  # число точек для формирования выборки
num_runs = 10  # число запусков для каждой функции

# Список тестовых функций: F1, F2, ..., F17
functions_list = [f"F{i}" for i in range(1, 18)]
# Список методов (оптимизаторов)
methods = ["SelfCGP", "PDPGP", "SelfCSHAGP", "PDPSHAGP"]


def run_single_run(F, iteration, sample_size, number_of_iterations, population_size, method):
    """
    Выполняет один запуск эксперимента для функции F с заданным номером итерации и методом.
    Используется один набор данных (без разделения на train/test).
    Вычисляются метрики RMSE, MSE, R2 и надёжность – флаг найденного решения (1, если RMSE < 0.01, иначе 0).
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

    # Выбираем оптимизатор по методу
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
        optimizer_args={"keep_history": True},
    )
    model.fit(X, y)

    # Получаем предсказания для всего набора данных
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    # Алгоритм считается нашедшим решение, если RMSE < 0.01
    solution_found = 1 if rmse < 0.01 else 0

    print(
        f"Function {F} | Iteration {iteration} | Method {method} | "
        f"RMSE: {rmse:.4f} | MSE: {mse:.4f} | R2: {r2:.4f} | Solution Found: {solution_found}"
    )

    return {
        "iteration": iteration,
        "F": F,
        "method": method,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "solution_found": solution_found,
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

    # Преобразуем результаты в DataFrame (для каждого запуска)
    df_results = pd.DataFrame(results)

    # Агрегированные метрики – усреднение по всем функциям и итерациям, группировка только по методам
    agg_rmse = df_results.groupby("method")["rmse"].mean().to_frame().T
    agg_mse = df_results.groupby("method")["mse"].mean().to_frame().T
    agg_r2 = df_results.groupby("method")["r2"].mean().to_frame().T
    agg_reliability = df_results.groupby("method")["solution_found"].mean().to_frame().T

    # Сохраняем результаты в Excel-файл (в папке results) с 5 вкладками:
    # 1) Raw Results, 2) Agg_RMSE, 3) Agg_MSE, 4) Agg_R2, 5) Agg_Reliability
    excel_path = os.path.join("results", "results.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="Raw Results", index=False)
        agg_rmse.to_excel(writer, sheet_name="Agg_RMSE")
        agg_mse.to_excel(writer, sheet_name="Agg_MSE")
        agg_r2.to_excel(writer, sheet_name="Agg_R2")
        agg_reliability.to_excel(writer, sheet_name="Agg_Reliability")

    print(f"Эксперимент завершён. Результаты сохранены в '{excel_path}'.")

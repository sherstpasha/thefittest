import pandas as pd
from thefittest.optimizers import SHAGA
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
import numpy as np
from tqdm import tqdm  # Для прогресс-бара
from combproblems_ga import problems_tuple


# Функции для оптимизации и анализа результатов
def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        error = np.abs(solution - true_solution)
        if np.all(error <= precision):
            return i + 1  # Возвращаем только количество итераций
    return None


def run_optimization_selfcga(function, eps, iters, pop_size):
    # dimension = function["dimention"]  # Получаем реальную размерность задачи
    # left = np.array([function["bounds"][0]] * dimension, dtype=np.float64)
    # right = np.array([function["bounds"][1]] * dimension, dtype=np.float64)
    # h = np.array([eps] * dimension, dtype=np.float64)

    if "k" in function.keys():
        problem = function["function"](function["k"]).__call__
    else:
        problem = function["function"]().__call__

    # genotype_to_phenotype = GrayCode().fit(left, right, h)
    # str_len = genotype_to_phenotype.parts.sum()

    optimizer = SHAGA(
        fitness_function=problem,
        # genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=iters,
        pop_size=pop_size,
        str_len=function["str_len"],
        elitism=False,
        # selections=("proportional", "rank", "tournament_3"),
        # crossovers=("empty", "one_point", "uniform_2"),
        # mutations=("weak", "average", "strong"),
        keep_history=True,
        minimization=function["minimization"],
    )
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i = find_solution_with_precision(stat["max_fitness"], function["optimum"], 0)
    return optimizer.get_fittest()["fitness"], speed_i

    # if speed_i is not None:
    #     return 1, speed_i  # Возвращаем 1 и номер поколения
    # return 0, np.nan  # Возвращаем 0 и NaN, если решение не найдено


def process_problem(problem):
    results = []

    # Параметры для цикла

    with mp.Pool(4) as pool:  # Создаем пул один раз
        futures = [
            pool.apply_async(
                run_optimization_selfcga,
                (
                    problem,
                    eps,
                    problem["iters"],
                    problem["pop_size"],
                ),
            )
            for _ in range(n_runs)
        ]

        for future in futures:
            # print(future.get())
            fitness, speed_i = future.get()
            if speed_i is not None:
                fe = speed_i * problem["iters"]
            else:
                fe = None
            results.append(
                [
                    problem["function"].__name__,
                    # problem["dimention"],
                    problem["pop_size"],
                    problem["iters"],
                    fitness,  # Это будет 1 или 0
                    speed_i,  # Это будет номер поколения или NaN,
                    fe,
                ]
            )

    return results


n_runs = 100
eps = 0.01

if __name__ == "__main__":
    results_file = "shaga_combproblem_new.csv"

    # Заголовки для CSV-файла
    columns = [
        "Function",
        # "Dimensions",
        "Pop_Size",
        "Iters",
        "fitness",  # Это будет 1 или 0 для каждого отдельного запуска
        "generation_found",  # Номер поколения, на котором найдено решение, или NaN
        "FE",
    ]

    # Запись заголовков в CSV (только если файл не существует)
    pd.DataFrame(columns=columns).to_csv(results_file, index=False, mode="w")

    for problem in tqdm(problems_tuple, desc="Processing functions", ncols=100):
        results = process_problem(problem)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(results_file, index=False, mode="a", header=False)

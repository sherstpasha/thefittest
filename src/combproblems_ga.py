import numpy as np
import pandas as pd
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from tqdm import tqdm  # Для прогресс-бара
from comb_problems import problems_tuple


# Функции для оптимизации и анализа результатов
def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        error = np.abs(solution - true_solution)
        if np.all(error <= precision):
            return i + 1  # Возвращаем только количество итераций
    return None

def run_optimization(
    function,
    eps,
    iters,
    pop_size,
    selection,
    crossover,
    mutation,
):
    reliability = 0.0
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0
    min_error = np.nan
    max_error = np.nan

    if "k" in function.keys():
        problem = function["function"](function["k"]).__call__
    else:
        problem = function["function"]().__call__

    optimizer = GeneticAlgorithm(
        fitness_function=problem,
        iters=iters,
        pop_size=pop_size,
        str_len=function["str_len"],
        elitism=False,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        keep_history=True,
        minimization=False,
    )
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i = find_solution_with_precision(stat["max_fitness"], function["optimum"], 1)
    return optimizer.get_fittest()["fitness"], speed_i

    # if speed_i is not None:
    #     return 1, speed_i  # Возвращаем 1 и номер поколения
    return 0, np.nan  # Возвращаем 0 и NaN, если решение не найдено


def process_problem(problem):
    results = []

    # Параметры для цикла
    selections = ["proportional", "rank", "tournament_3", "tournament_5", "tournament_7"]
    crossovers = [
        "empty",
        "one_point",
        "two_point",
        "uniform_2",
        "uniform_7",
        "uniform_prop_2",
        "uniform_prop_7",
        "uniform_rank_2",
        "uniform_rank_7",
        "uniform_tour_3",
        "uniform_tour_7",
    ]
    mutations = ["weak", "average", "strong"]

    total_combinations = len(selections) * len(crossovers) * len(mutations)

    with mp.Pool(20) as pool:  # Создаем пул один раз
        with tqdm(
            total=total_combinations, desc="Processing combinations", ncols=100, leave=False
        ) as pbar:
            for selection in selections:
                for crossover in crossovers:
                    for mutation in mutations:
                        futures = [
                            pool.apply_async(
                                run_optimization,
                                (
                                    problem,
                                    eps,
                                    problem["iters"],
                                    problem["pop_size"],
                                    selection,
                                    crossover,
                                    mutation,
                                ),
                            )
                            for _ in range(n_runs)
                        ]

                        for future in futures:
                            # fitness = future.get()
                            fitness, speed_i = future.get()
                            if speed_i is not None:
                                fe = speed_i*problem["iters"]
                            else:
                                fe = None
                            results.append(
                                [
                                    problem["function"].__name__,
                                    selection,
                                    crossover,
                                    mutation,
                                    problem["pop_size"],
                                    problem["iters"],
                                    fitness,  # Это будет 1 или 0
                                    speed_i,  # Это будет номер поколения или NaN,
                                    fe
                                ]
                            )

                        pbar.update(1)  # Обновляем прогресс-бар после каждой комбинации

    return results


n_runs = 5
eps = 0.01

if __name__ == "__main__":
    results_file = "ga_combproblems.csv"

    # Заголовки для CSV-файла
    columns = [
        "Function",
        "Selection",
        "Crossover",
        "Mutation",
        "Pop_Size",
        "Iters",
        "fitness",
        "generation_found",  # Номер поколения, на котором найдено решение, или NaN
        'FE'
    ]

    # Запись заголовков в CSV (только если файл не существует)
    pd.DataFrame(columns=columns).to_csv(results_file, index=False, mode="w")

    for problem in tqdm(problems_tuple, desc="Processing functions", ncols=100):
        results = process_problem(problem)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(results_file, index=False, mode="a", header=False)

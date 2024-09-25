import numpy as np
import pandas as pd

# from opfunu.cec_based import cec2005
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from comb_problems import problems_tuple

def find_solution_with_precision(solution_list, true_solution):

    for i, solution in enumerate(solution_list):
        if np.array_equal(solution, true_solution):
            return i + 1, None  # Возвращаем индекс + 1 и ошибки

    return None, None  # Если ни одно решение не совпало


def run_optimization(function, eps, iters, pop_size, selection, crossover, mutation,):
    reliability = 0.0
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0
    min_error = np.nan
    max_error = np.nan

    optimizer = GeneticAlgorithm(
        fitness_function=function["function"]().__call__,
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
    speed_i, errors = find_solution_with_precision(stat["max_ph"], function["optimum_x"])
    argmin = 0

    if speed_i is not None:
        reliability = 1
        speed_sum = speed_i
        range_left = speed_i
        range_right = speed_i
        find_count = 1
    else:
        if errors:
            argmin = np.argmin(np.array(errors)[:, 1])

    return reliability, speed_sum, range_left, range_right, find_count, None


def main():
    eps = 0.0
    n_runs = 20
    initial_iters_pop = 100
    max_iters = 50000
    max_pop_size = 5000
    target_reliability = 0.5

    iters_values = []
    pop_size_values = []

    iters = initial_iters_pop
    pop_size = 5000

    while pop_size <= max_pop_size and iters <= max_iters:
        iters_values.append(iters)
        pop_size_values.append(pop_size)
        iters = iters + int(iters * .15)
        pop_size = pop_size + int(pop_size * 0)

    while iters <= max_iters:
        iters_values.append(iters)
        pop_size_values.append(max_pop_size)
        iters = iters + int(iters * 0.1)


    function_ = problems_tuple[3]
    functions = [function_]

    results = []

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for function in functions:
            successful = False

            for iters, pop_size in zip(iters_values, pop_size_values):
                for selection in ["tournament_3", "rank", "proportional"]:
                    for crossover in ["uniform_2", "one_point", "two_point"]:
                        for mutation in ["weak", "average", "strong"]:
                            futures = [
                                pool.apply_async(
                                    run_optimization,
                                    args=(
                                        function,
                                        eps,
                                        iters,
                                        pop_size,
                                        selection,
                                        crossover,
                                        mutation,
                                    ),
                                )
                                for _ in range(n_runs)
                            ]

                            reliability_sum = 0
                            speed_sum = 0
                            range_left = np.nan
                            range_right = np.nan
                            find_count = 0
                            all_errors = []

                            for future in futures:
                                rel, speed, left, right, count, errors = future.get()  #

                                all_errors.append(errors)
                                reliability_sum += rel
                                speed_sum += speed
                                if not np.isnan(left):
                                    range_left = (
                                        min(range_left, left) if not np.isnan(range_left) else left
                                    )
                                if not np.isnan(right):
                                    range_right = (
                                        max(range_right, right)
                                        if not np.isnan(range_right)
                                        else right
                                    )
                                find_count += count

                            reliability = reliability_sum / n_runs

                            # argmin = np.argmin(np.array(all_errors)[:, 1])
                            # error = np.array(all_errors)[argmin]

                            print(
                                function["function"],
                                iters,
                                pop_size,
                                reliability,
                                selection,
                                crossover,
                                mutation,
                                0,
                            )

                            if reliability >= target_reliability:
                                successful = True
                                results.append(
                                    [
                                        function["function"].__class__.__name__,
                                        selection,
                                        crossover,
                                        mutation,
                                        pop_size,
                                        iters,
                                        reliability,
                                        speed_sum / find_count,
                                        range_left,
                                        range_right,
                                    ]
                                )
                                print(results[-1])

                if successful:
                    break


if __name__ == "__main__":
    main()

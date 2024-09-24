import numpy as np
import pandas as pd

# from opfunu.cec_based import cec2005
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from problems13 import problems_tuple

def find_solution_with_precision(solution_list, true_solution, precision):
    errors = []
    for i, solution in enumerate(solution_list):
        error = np.abs(solution - true_solution)
        errors.append([error.min(), error.max()])
        if np.all(error <= precision):
            return i + 1, errors
    return None, errors


def run_optimization(function, eps, iters, pop_size, selection, crossover, mutation, dim):
    reliability = 0.0
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0
    min_error = np.nan
    max_error = np.nan

    left = np.array([function["bounds"][0]] * dim, dtype=np.float64)
    right = np.array([function["bounds"][1]] * dim, dtype=np.float64)
    h = np.array([eps] * dim, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(left, right, h)
    str_len = genotype_to_phenotype.parts.sum()

    optimizer = GeneticAlgorithm(
        fitness_function=function["function"]().__call__,
        genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        elitism=False,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        keep_history=True,
        minimization=True,
    )
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i, errors = find_solution_with_precision(stat["max_ph"], function["optimum_x"], h)
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

    return reliability, speed_sum, range_left, range_right, find_count, np.array(errors)[argmin]


def main():
    eps = 0.01
    n_runs = 100
    initial_iters_pop = 15
    max_iters = 50000
    max_pop_size = 5000
    target_reliability = 0.5

    iters_values = []
    pop_size_values = []

    iters = initial_iters_pop
    pop_size = initial_iters_pop

    while pop_size <= max_pop_size and iters <= max_iters:
        iters_values.append(iters)
        pop_size_values.append(pop_size)
        iters = iters + int(iters * 0.2)
        pop_size = pop_size + int(pop_size * 0.2)

    while iters <= max_iters:
        iters_values.append(iters)
        pop_size_values.append(max_pop_size)
        iters = iters + int(iters * 0.2)


    function_ = problems_tuple[2]
    dimensions = function_["dimention"]
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
                                        dimensions
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

                            argmin = np.argmin(np.array(all_errors)[:, 1])
                            error = np.array(all_errors)[argmin]

                            print(
                                function["function"],
                                iters,
                                pop_size,
                                reliability,
                                selection,
                                crossover,
                                mutation,
                                error,
                            )

                            if reliability >= target_reliability:
                                successful = True
                                results.append(
                                    [
                                        function["function"].__class__.__name__,
                                        dimensions,
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
                                # break

                if successful:
                    break


if __name__ == "__main__":
    main()

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

from problems25 import problems_tuple
from thefittest.optimizers._selfcshaga import SelfCSHAGA
from thefittest.utils.transformations import GrayCode

N_RUNS = int(os.environ.get("SELFCSHAGA_N_RUNS", "100"))
MAX_PROBLEMS = int(os.environ.get("SELFCSHAGA_MAX_PROBLEMS", "25"))
PROCESSES = int(os.environ.get("SELFCSHAGA_PROCESSES", "10"))
EXECUTOR = os.environ.get("SELFCSHAGA_EXECUTOR", "thread").lower()
EPS = 0.01


def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        error = np.abs(solution - true_solution)
        if np.all(error <= precision):
            return i + 1
    return None


def run_optimization_selfcshaga(function, eps, iters, pop_size):
    dimension = function["dimention"]
    left = function["bounds"][0]
    right = function["bounds"][1]
    h = np.array([eps] * dimension, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(
        left_border=left,
        right_border=right,
        num_variables=dimension,
        h_per_variable=eps,
    )
    str_len = genotype_to_phenotype.get_str_len()
    max_ph_history = []

    def on_generation(optimizer):
        max_fitness_id = np.argmax(optimizer._fitness_i)
        max_ph_history.append(optimizer._population_ph_i[max_fitness_id].copy())

    optimizer = SelfCSHAGA(
        fitness_function=function["function"]().__call__,
        genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        elitism=False,
        K=2,
        selections=(
            "proportional",
            "rank",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers=(
            "empty",
            "one_point_1",
            "two_point_1",
            "uniform_1",
            "uniform_2",
            "uniform_7",
            "uniform_prop_2",
            "uniform_prop_7",
            "uniform_rank_2",
            "uniform_rank_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        keep_history=False,
        minimization=True,
        n_jobs=1,
        on_generation=on_generation,
    )
    optimizer.fit()
    speed_i = find_solution_with_precision(max_ph_history, function["optimum_x"], h)

    if speed_i is not None:
        return 1, speed_i
    return 0, np.nan


def process_problem(problem):
    results = []
    common_result = [
        problem["function"].__name__,
        problem["dimention"],
        problem["pop_size"],
        problem["iters"],
    ]

    if PROCESSES <= 1:
        for _ in tqdm(range(N_RUNS), desc=problem["function"].__name__, leave=False, ncols=100):
            find_solution, speed_i = run_optimization_selfcshaga(
                problem,
                EPS,
                problem["iters"],
                problem["pop_size"],
            )
            results.append(common_result + [find_solution, speed_i])
        return results

    max_workers = min(PROCESSES, N_RUNS)
    executor_class = ProcessPoolExecutor if EXECUTOR == "process" else ThreadPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_optimization_selfcshaga,
                problem,
                EPS,
                problem["iters"],
                problem["pop_size"],
            )
            for _ in range(N_RUNS)
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=problem["function"].__name__,
            leave=False,
            ncols=100,
        ):
            find_solution, speed_i = future.result()
            results.append(common_result + [find_solution, speed_i])

    return results


if __name__ == "__main__":
    results_file = "selfcshaga_cec2005_new2.csv"
    columns = [
        "Function",
        "Dimensions",
        "Pop_Size",
        "Iters",
        "find_solution",
        "generation_found",
    ]

    pd.DataFrame(columns=columns).to_csv(results_file, index=False, mode="w")

    problems = problems_tuple[:MAX_PROBLEMS] if MAX_PROBLEMS > 0 else problems_tuple
    for problem in tqdm(problems, desc="Processing functions", ncols=100):
        results = process_problem(problem)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(results_file, index=False, mode="a", header=False)

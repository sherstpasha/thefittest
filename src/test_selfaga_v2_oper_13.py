import numpy as np
import pandas as pd
from thefittest.benchmarks.testfunctions14 import problems_dict
from thefittest.optimizers._my_adapt_ga_var2_3oper import MyAdaptGAVar23oper
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from tqdm import tqdm


def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        if np.all(np.abs(solution - true_solution) <= precision):
            return i + 1
    return None


def run_optimization(F, content, n_vars, eps, iters_pop, adapt_oper, mutate_operator_proba):
    reliability = 0.0
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0

    left = np.array([content["bounds"][0]] * n_vars, dtype=np.float64)
    right = np.array([content["bounds"][1]] * n_vars, dtype=np.float64)
    h = np.array([eps] * n_vars, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(left, right, h)
    str_let = genotype_to_phenotype.parts.sum()

    optimizer = MyAdaptGAVar23oper(
        fitness_function=content["function"](),
        genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=iters_pop[F],
        pop_size=iters_pop[F],
        str_len=str_let,
        elitism=True,
        selections=("tournament_k", "rank", "proportional"),
        crossovers=("two_point", "one_point", "uniform_2"),
        adaptation_operator=adapt_oper,
        mutate_operator_proba=mutate_operator_proba,
        keep_history=True,
        minimization=True,
    )
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i = find_solution_with_precision(stat["max_ph"], content["optimum_x"][:n_vars], h)

    if speed_i is not None:
        reliability = 1
        speed_sum = speed_i
        range_left = speed_i
        range_right = speed_i
        find_count = 1

    return reliability, speed_sum, range_left, range_right, find_count


def main():
    eps = 0.01
    n_runs = 1000
    iters_pop = {
        "F1": 15,
        "F2": 22,
        "F3": 25,
        "F4": 20,
        "F5": 30,
        "F6": 70,
        "F7": 80,
        "F8": 120,
        "F9": 70,
        "F10": 400,
        "F11": 45,
        "F12": 16,
        "F13": 18,
    }

    results = []
    adaptation_operator_list = ["proportional", "rank", "tournament_3"]
    mutate_operator_proba_range = np.arange(0, 0.21, 0.01)
    total_combinations = (
        len(problems_dict) * len(adaptation_operator_list) * len(mutate_operator_proba_range)
    )
    progress_bar = tqdm(total=total_combinations, desc="Optimization Progress")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for F, content in problems_dict.items():
            for apart_operator in adaptation_operator_list:
                for mutate_operator_proba_i in mutate_operator_proba_range:
                    for n_vars in content["dimentions"]:
                        futures = [
                            pool.apply_async(
                                run_optimization,
                                args=(
                                    F,
                                    content,
                                    n_vars,
                                    eps,
                                    iters_pop,
                                    apart_operator,
                                    mutate_operator_proba_i,
                                ),
                            )
                            for _ in range(n_runs)
                        ]

                        reliability = 0.0
                        speed_sum = 0
                        range_left = np.nan
                        range_right = np.nan
                        find_count = 0

                        for future in futures:
                            rel, speed, left, right, count = future.get()
                            reliability += rel / n_runs
                            speed_sum += speed
                            range_left = np.nanmin([range_left, left])
                            range_right = np.nanmax([range_right, right])
                            find_count += count

                        reliability = round(reliability, 2)
                        if speed_sum > 0:
                            speed = round(speed_sum / find_count, 2)
                        else:
                            speed = np.nan

                        results.append(
                            [
                                F,
                                n_vars,
                                apart_operator,
                                mutate_operator_proba_i,
                                reliability,
                                speed,
                                range_left,
                                range_right,
                            ]
                        )

                    progress_bar.update(1)

    progress_bar.close()

    combined_df = pd.DataFrame(
        results,
        columns=[
            "Function",
            "Dimensions",
            "Adaptation_Operator",
            "Mutate_Operator_Proba",
            "Reliability",
            "Speed",
            "Range_Left",
            "Range_Right",
        ],
    )
    combined_df.to_csv("combined_results_my_adapt_ga_v2_oper.csv", index=False)

    # iters_pop_df = pd.DataFrame(list(iters_pop.items()), columns=["Function", "Iters_Pop_Size"])
    # iters_pop_df.to_csv("iters_pop_size_my_adapt_ga_v2.csv", index=False)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from thefittest.benchmarks.testfunctions14 import problems_dict
from thefittest.optimizers._my_adapt_ga_var2 import MyAdaptGAVar2
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from tqdm import tqdm

def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        if np.all(np.abs(solution - true_solution) <= precision):
            return i + 1
    return None

def run_optimization(F, content, n_vars, eps, iters_pop):
    reliability = 0.
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0

    left = np.array([content["bounds"][0]] * n_vars, dtype=np.float64)
    right = np.array([content["bounds"][1]] * n_vars, dtype=np.float64)
    h = np.array([eps] * n_vars, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(left, right, h)
    str_let = genotype_to_phenotype.parts.sum()

    optimizer = MyAdaptGAVar2(fitness_function=content["function"](),
                        genotype_to_phenotype=genotype_to_phenotype.transform,
                                 iters=iters_pop[F], 
                                 pop_size=iters_pop[F],
                                 str_len=str_let,
                                 elitism=True,
                                selections=("tournament_k", "rank", "proportional"),
                                crossovers=("two_point", "one_point", "uniform_2"),
                                # mutations=("weak", "average", "strong"),
                                 keep_history=True,
                                 minimization=True)        
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
    iters_pop = {"F1": 15,
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
                 "F13": 18}

    reliability_results = []
    speed_results = []
    range_results = []

    total_combinations = len(problems_dict) # Total combinations of functions, selections, crossovers, and mutations
    progress_bar = tqdm(total=total_combinations, desc="Optimization Progress")

    for F, content in problems_dict.items():
                    content = problems_dict[F]

                    for n_vars in content["dimentions"]:
                        pool = mp.Pool(mp.cpu_count())
                        results = [pool.apply_async(run_optimization, args=(F, content, n_vars, eps, iters_pop)) for _ in range(n_runs)]
                        pool.close()
                        pool.join()

                        reliability = 0.
                        speed_sum = 0
                        range_left = np.nan
                        range_right = np.nan
                        find_count = 0

                        for result in results:
                            rel, speed, left, right, count = result.get()
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

                        reliability_results.append([F, n_vars, reliability])
                        speed_results.append([F, n_vars, speed])
                        range_results.append([F, n_vars, range_left, range_right])

                    progress_bar.update(1)

    progress_bar.close()

    reliability_df = pd.DataFrame(reliability_results, columns=["Function", "Dimensions", "Reliability"])
    speed_df = pd.DataFrame(speed_results, columns=["Function", "Dimensions", "Speed"])
    range_df = pd.DataFrame(range_results, columns=["Function", "Dimensions", "Range_Left", "Range_Right"])

    reliability_df.to_csv("reliability_results_selfaga_v2.csv", index=False)
    speed_df.to_csv("speed_results_selfaga_v2.csv", index=False)
    range_df.to_csv("range_results_selfaga_v2.csv", index=False)

    iters_pop_df = pd.DataFrame(list(iters_pop.items()), columns=["Function", "Iters_Pop_Size"])
    iters_pop_df.to_csv("iters_pop_size_selfaga_v2.csv", index=False)

if __name__ == '__main__':
    main()

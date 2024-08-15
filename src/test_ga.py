import numpy as np
import pandas as pd
from opfunu.cec_based import cec2005
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from tqdm import tqdm

def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        if np.all(np.abs(solution - true_solution) <= precision):
            return i + 1
    return None

def run_optimization(function, eps, iters, pop_size, selection, crossover, mutation):
    reliability = 0.
    speed_sum = 0
    range_left = np.nan
    range_right = np.nan
    find_count = 0

    left = np.array(function.bounds[:, 0], dtype=np.float64)
    right = np.array(function.bounds[:, 1], dtype=np.float64)
    h = np.array([eps] * function.ndim, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(left, right, h)
    str_len = genotype_to_phenotype.parts.sum()

    def fitness_function(population_g):
        fitness = np.array(list(map(function.evaluate, population_g)), dtype=np.float64)
        return fitness

    optimizer = GeneticAlgorithm(fitness_function=fitness_function,
                                 genotype_to_phenotype=genotype_to_phenotype.transform,
                                 iters=iters,
                                 pop_size=pop_size,
                                 str_len=str_len,
                                 elitism=False,
                                 selection=selection,
                                 crossover=crossover,
                                 mutation=mutation,
                                 keep_history=True,
                                 minimization=True)
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i = find_solution_with_precision(stat["max_ph"], function.f_global, h)

    if speed_i is not None:
        reliability = 1
        speed_sum = speed_i
        range_left = speed_i
        range_right = speed_i
        find_count = 1

    return reliability, speed_sum, range_left, range_right, find_count

def main():
    eps = 0.0001
    n_runs = 10
    initial_iters_pop = 50  # Начальное значение итераций и размера популяции
    max_iters_pop = 10000  # Максимальное значение итераций и размера популяции
    target_reliability = 0.5
    increment_step = 50  # Шаг увеличения итераций и размера популяции

    ndim = 10
    functions = [cec2005.F12005(ndim=ndim),
                 cec2005.F22005(ndim=ndim),
                 cec2005.F32005(ndim=ndim),
                 cec2005.F42005(ndim=ndim),
                 cec2005.F52005(ndim=ndim),
                 cec2005.F62005(ndim=ndim),
                 cec2005.F72005(ndim=ndim),
                 cec2005.F82005(ndim=ndim),
                 cec2005.F92005(ndim=ndim),
                 cec2005.F102005(ndim=ndim),
                 cec2005.F112005(ndim=ndim),
                 cec2005.F122005(ndim=ndim),
                 cec2005.F132005(ndim=ndim),
                 cec2005.F142005(ndim=ndim),
                 cec2005.F152005(ndim=ndim),
                 cec2005.F162005(ndim=ndim),
                 cec2005.F172005(ndim=ndim),
                 cec2005.F182005(ndim=ndim),
                 cec2005.F192005(ndim=ndim),
                 cec2005.F202005(ndim=ndim),
                 cec2005.F212005(ndim=ndim),
                 cec2005.F222005(ndim=ndim),
                 cec2005.F232005(ndim=ndim),
                 cec2005.F242005(ndim=ndim),
                 cec2005.F252005(ndim=ndim)]

    results = []

    progress_bar = tqdm(total=len(functions), desc="Optimization Progress")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for function in functions:
            successful = False
            for iters_pop in range(initial_iters_pop, max_iters_pop + increment_step, increment_step):
                for selection in ["proportional", "rank", "tournament_3"]:
                    for crossover in ["one_point", "two_point", "uniform_2"]:
                        for mutation in ["weak", "average", "strong"]:
                            futures = [pool.apply_async(run_optimization, args=(function, eps, iters_pop, iters_pop, selection, crossover, mutation)) for _ in range(n_runs)]
                            
                            reliability_sum = 0
                            speed_sum = 0
                            range_left = np.nan
                            range_right = np.nan
                            find_count = 0

                            for future in futures:
                                rel, speed, left, right, count = future.get()
                                reliability_sum += rel
                                speed_sum += speed
                                if not np.isnan(left):
                                    range_left = min(range_left, left) if not np.isnan(range_left) else left
                                if not np.isnan(right):
                                    range_right = max(range_right, right) if not np.isnan(range_right) else right
                                find_count += count

                            reliability = reliability_sum / n_runs

                            if reliability >= target_reliability:
                                successful = True
                                results.append([function.__class__.__name__, ndim, selection, crossover, mutation, iters_pop, reliability, speed_sum / find_count, range_left, range_right])
                                break

                        if successful:
                            break
                    if successful:
                        break
                if successful:
                    break

            progress_bar.update(1)

    progress_bar.close()

    results_df = pd.DataFrame(results, columns=["Function", "Dimensions", "Selection", "Crossover", "Mutation", "Iters_Pop_Size", "Reliability", "Speed", "Range_Left", "Range_Right"])
    results_df.to_csv("optimal_iters_pop_size_cec2005.csv", index=False)

if __name__ == '__main__':
    main()

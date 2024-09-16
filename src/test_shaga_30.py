import numpy as np

from thefittest.benchmarks._optproblems import ShiftedSphere  # 1 Unimodal (5)
from thefittest.benchmarks._optproblems import ShiftedSchwefe1_2  # 2
from thefittest.benchmarks._optproblems import ShiftedRotatedHighConditionedElliptic  # 3
from thefittest.benchmarks._optproblems import ShiftedSchwefe1_2WithNoise  # 4

from thefittest.benchmarks._optproblems import (
    ShiftedRosenbrock,
)  # 6 Multimodal (20) Basic Functions (7)
from thefittest.benchmarks._optproblems import ShiftedRotatedGriewank  # 7
from thefittest.benchmarks._optproblems import ShiftedRastrigin  # 9
from thefittest.benchmarks._optproblems import ShiftedRotatedRastrigin  # 10
from thefittest.benchmarks._optproblems import Schwefel2_13  # 12

from thefittest.benchmarks._optproblems import (
    ShiftedExpandedGriewankRosenbrock,
)  # 13 Expanded Functions (2)
from thefittest.benchmarks._optproblems import ShiftedRotatedExpandedScaffes_F6  # 14

from thefittest.benchmarks._optproblems import (
    HybridCompositionFunction1,
)  # 15 Hybrid Composition Functions (11)
from thefittest.benchmarks._optproblems import RotatedVersionHybridCompositionFunction1  # 16
from thefittest.benchmarks._optproblems import RotatedVersionHybridCompositionFunction1Noise  # 17
from thefittest.benchmarks._optproblems import RotatedHybridCompositionFunction  # 18
from thefittest.benchmarks._optproblems import RotatedHybridCompositionFunctionNarrowBasin  # 19
from thefittest.benchmarks._optproblems import HybridCompositionFunction3  # 21
from thefittest.benchmarks._optproblems import HybridCompositionFunction3H  # 22
from thefittest.benchmarks._optproblems import NonContinuousHybridCompositionFunction3  # 23
from thefittest.benchmarks._optproblems import HybridCompositionFunction4  # 24
from thefittest.benchmarks._optproblems import HybridCompositionFunction4withoutbounds  # 25

from thefittest.benchmarks import Sphere
from thefittest.benchmarks import Schwefe1_2
from thefittest.benchmarks import Rosenbrock
from thefittest.benchmarks import Rastrigin
from thefittest.benchmarks import Ackley
from thefittest.benchmarks import Weierstrass
from thefittest.benchmarks import HighConditionedElliptic
from thefittest.benchmarks import Griewank
from thefittest.benchmarks._optproblems import ExpandedScaffers_F6

import pandas as pd
from thefittest.optimizers import SHAGA
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp

import numpy as np
import pandas as pd
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from tqdm import tqdm  # Для прогресс-бара
import numpy as np
import pandas as pd
from thefittest.tools.transformations import GrayCode
import multiprocessing as mp
from tqdm import tqdm  # Для прогресс-бара


# Функции для оптимизации и анализа результатов
def find_solution_with_precision(solution_list, true_solution, precision):
    for i, solution in enumerate(solution_list):
        error = np.abs(solution - true_solution)
        print(
            f"Solution: {solution}, True Solution: {true_solution}, Error: {error}"
        )  # Добавьте это для отладки
        if np.all(error <= precision):
            return i + 1  # Возвращаем количество итераций (начиная с 1)
    return None
    # selections = ["proportional", "rank", "tournament_3", "tournament_5", "tournament_7"]
    # crossovers = ["empty", "one_point", "two_point", "uniform_2", "uniform_7", "uniform_prop_2", "uniform_prop_7", "uniform_rank_2", "uniform_rank_7", "uniform_tour_3", "uniform_tour_7"]
    # mutations = ["weak", "average", "strong"]


def run_optimization_pdp(function, eps, iters, pop_size):
    dimension = function["dimention"]  # Получаем реальную размерность задачи
    left = np.array([function["bounds"][0]] * dimension, dtype=np.float64)
    right = np.array([function["bounds"][1]] * dimension, dtype=np.float64)
    h = np.array([eps] * dimension, dtype=np.float64)

    genotype_to_phenotype = GrayCode().fit(left, right, h)
    str_len = genotype_to_phenotype.parts.sum()

    optimizer = SHAGA(
        fitness_function=function["function"]().__call__,
        genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=iters,
        pop_size=pop_size,
        str_len=str_len,
        elitism=False,
        keep_history=True,
        minimization=True,
    )
    optimizer.fit()
    stat = optimizer.get_stats()
    speed_i = find_solution_with_precision(stat["max_ph"], function["optimum_x"], h)

    if speed_i is not None:
        return 1, speed_i  # Возвращаем 1 и номер поколения
    return 0, np.nan  # Возвращаем 0 и NaN, если решение не найдено


def process_problem(problem):
    results = []

    # Параметры для цикла

    with mp.Pool(processes=mp.cpu_count()) as pool:  # Создаем пул один раз
        futures = [
            pool.apply_async(
                run_optimization_pdp,
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
            find_solution, speed_i = future.get()
            results.append(
                [
                    problem["function"].__name__,
                    problem["dimention"],
                    problem["pop_size"],
                    problem["iters"],
                    find_solution,  # Это будет 1 или 0
                    speed_i,  # Это будет номер поколения или NaN
                ]
            )

    return results


problems_tuple = (
    {  # 1
        "function": ShiftedRosenbrock,
        "bounds": (-100, 100),
        "optimum": 390,
        "optimum_x": ShiftedRosenbrock().x_shift[:2],
        "dimention": 2,
        "iters": 270,
        "pop_size": 270,
    },
    {  # 2
        "function": ShiftedRotatedGriewank,
        "bounds": (-1000, 1000),
        "optimum": -180,
        "optimum_x": ShiftedRotatedGriewank().x_shift[:2],
        "dimention": 2,
        "iters": 650,
        "pop_size": 650,
    },
    {  # 3
        "function": ShiftedExpandedGriewankRosenbrock,
        "bounds": (-3, 1),
        "optimum": -130,
        "optimum_x": ShiftedExpandedGriewankRosenbrock().x_shift[:2],
        "dimention": 2,
        "iters": 110,
        "pop_size": 110,
    },
    {  # 4
        "function": RotatedVersionHybridCompositionFunction1,
        "bounds": (-5, 5),
        "optimum": 120,
        "optimum_x": RotatedVersionHybridCompositionFunction1().x_shift[0][:2],
        "dimention": 2,
        "iters": 130,
        "pop_size": 130,
    },
    {  # 5
        "function": RotatedVersionHybridCompositionFunction1Noise,
        "bounds": (-5, 5),
        "optimum": 120,
        "optimum_x": RotatedVersionHybridCompositionFunction1Noise().x_shift[0][:2],
        "dimention": 2,
        "iters": 130,
        "pop_size": 130,
    },
    {  # 6
        "function": RotatedHybridCompositionFunction,
        "bounds": (-5, 5),
        "optimum": 10,
        "optimum_x": RotatedHybridCompositionFunction().x_shift[0][:2],
        "dimention": 2,
        "iters": 250,
        "pop_size": 250,
    },
    {  # 7
        "function": HybridCompositionFunction3,
        "bounds": (-5, 5),
        "optimum": 360,
        "optimum_x": HybridCompositionFunction3().x_shift[0][:2],
        "dimention": 2,
        "iters": 477,
        "pop_size": 477,
    },
    {  # 8
        "function": HybridCompositionFunction3H,
        "bounds": (-5, 5),
        "optimum": 360,
        "optimum_x": HybridCompositionFunction3H().x_shift[0][:2],
        "dimention": 2,
        "iters": 477,
        "pop_size": 477,
    },
    {  # 9
        "function": NonContinuousHybridCompositionFunction3,
        "bounds": (-5, 5),
        "optimum": 360,
        "optimum_x": NonContinuousHybridCompositionFunction3().x_shift[0][:2],
        "dimention": 2,
        "iters": 477,
        "pop_size": 477,
    },
    {  # 10
        "function": HybridCompositionFunction4,
        "bounds": (-5, 5),
        "optimum": 260,
        "optimum_x": HybridCompositionFunction4().x_shift[0][:2],
        "dimention": 2,
        "iters": 799,
        "pop_size": 799,
    },
    {  # 11
        "function": HybridCompositionFunction4withoutbounds,
        "bounds": (-10, 10),
        "optimum": 260,
        "optimum_x": HybridCompositionFunction4withoutbounds().x_shift[0][:2],
        "dimention": 2,
        "iters": 1000,
        "pop_size": 1000,
    },
    {  # 12
        "function": Rosenbrock,
        "bounds": (-2.048, 2.048),
        "optimum": 0,
        "optimum_x": np.ones(shape=2, dtype=np.float64),
        "dimention": 2,
        "iters": 100,
        "pop_size": 100,
    },
    {  # 13
        "function": ExpandedScaffers_F6,
        "bounds": (-100, 100),
        "optimum": 0,
        "optimum_x": np.zeros(shape=2, dtype=np.float64),
        "dimention": 2,
        "iters": 100,
        "pop_size": 100,
    },
    {  # 14
        "function": Weierstrass,
        "bounds": (-1, 1),
        "optimum": 0,
        "optimum_x": np.zeros(shape=5, dtype=np.float64),
        "dimention": 5,
        "iters": 1157,
        "pop_size": 1157,
    },
    {  # 15
        "function": ShiftedSphere,
        "bounds": (-100, 100),
        "optimum": -450,
        "optimum_x": ShiftedSphere().x_shift[:10],
        "dimention": 10,
        "iters": 125,
        "pop_size": 125,
    },
    {  # 16
        "function": ShiftedSchwefe1_2,
        "bounds": (-100, 100),
        "optimum": -450,
        "optimum_x": ShiftedSchwefe1_2().x_shift[:10],
        "dimention": 10,
        "iters": 728,
        "pop_size": 728,
    },
    {  # 17
        "function": ShiftedSchwefe1_2WithNoise,
        "bounds": (-100, 100),
        "optimum": -450,
        "optimum_x": ShiftedSchwefe1_2WithNoise().x_shift[:10],
        "dimention": 10,
        "iters": 757,
        "pop_size": 757,
    },
    {  # 18
        "function": ShiftedRastrigin,
        "bounds": (-5, 5),
        "optimum": -330,
        "optimum_x": ShiftedRastrigin().x_shift[:10],
        "dimention": 10,
        "iters": 470,
        "pop_size": 470,
    },
    {  # 19
        "function": ShiftedRotatedRastrigin,
        "bounds": (-5, 5),
        "optimum": -330,
        "optimum_x": ShiftedRotatedRastrigin().x_shift[:10],
        "dimention": 10,
        "iters": 463,
        "pop_size": 463,
    },
    {  # 20
        "function": HybridCompositionFunction1,
        "bounds": (-5, 5),
        "optimum": 120,
        "optimum_x": HybridCompositionFunction1().x_shift[0][:10],
        "dimention": 10,
        "iters": 799,
        "pop_size": 799,
    },
    {  # 21
        "function": Sphere,
        "bounds": (-5.12, 5.12),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 210,
        "pop_size": 210,
    },
    {  # 22
        "function": HighConditionedElliptic,
        "bounds": (-100, 100),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 355,
        "pop_size": 355,
    },
    {  # 23
        "function": Griewank,
        "bounds": (-600, 600),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 600,
        "pop_size": 600,
    },
    {  # 24
        "function": Ackley,
        "bounds": (-32.768, 32.768),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 247,
        "pop_size": 247,
    },
    {  # 25
        "function": Rastrigin,
        "bounds": (-5.12, 5.12),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 799,
        "pop_size": 799,
    },
)
n_runs = 100
eps = 0.01

if __name__ == "__main__":
    results_file = "shaga_cec2005.csv"

    # Заголовки для CSV-файла
    columns = [
        "Function",
        "Dimensions",
        "Pop_Size",
        "Iters",
        "find_solution",  # Это будет 1 или 0 для каждого отдельного запуска
        "generation_found",  # Номер поколения, на котором найдено решение, или NaN
    ]

    # Запись заголовков в CSV (только если файл не существует)
    pd.DataFrame(columns=columns).to_csv(results_file, index=False, mode="w")

    for problem in tqdm(problems_tuple, desc="Processing functions", ncols=100):
        results = process_problem(problem)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(results_file, index=False, mode="a", header=False)

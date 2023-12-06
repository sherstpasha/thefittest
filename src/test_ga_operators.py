import itertools

import pandas as pd
import numpy as np

from thefittest.benchmarks.CEC2005 import problems_dict
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.random import float_population
from thefittest.tools.transformations import SamplingGrid


def check_solution(x, optimum_x, h):
    shape = x.shape
    axis = [1] * (len(shape) - 1) + [-1]
    reliability = np.all(np.abs(x - optimum_x[: shape[-1]].reshape(axis)) <= h)
    print(np.abs(x - optimum_x[: shape[-1]].reshape(axis)), h, reliability)
    return reliability.astype(np.int64)


iters = (400,)
pop_size = (400,)
print(iters)
h_all = 0.1
n_runs = 5

selection_operators = ("proportional", "rank", "tournament_3", "tournament_5", "tournament_7")

crossover_operators = ("one_point", "two_point", "uniform_2")

mutation_operators = ("weak", "average", "strong")

functions = (
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F12",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "F25",
)


combinations = list(itertools.product(selection_operators, crossover_operators, mutation_operators))
combinations_str = ["_".join(combination) for combination in combinations]


result_reliability = pd.DataFrame(columns=combinations_str)
for i in range(n_runs):
    result_fitness = pd.DataFrame(columns=combinations_str)
    result_succes = pd.DataFrame(columns=combinations_str)

    for selection in selection_operators:
        for crossover in crossover_operators:
            for mutation in mutation_operators:
                # print(selection, crossover, mutation)
                for iters_i, pop_size_i, function in zip(iters, pop_size, functions):
                    print(function, iters_i, pop_size_i)
                    problem = problems_dict[function]
                    result_fitness
                    for dimention in [10]:
                        fitness_function = problem["function"]
                        bounds = problem["bounds"]
                        optimum = problem["optimum"]
                        optimum_x = problem["optimum_x"]

                        left = np.full(shape=dimention, fill_value=bounds[0], dtype=np.float64)
                        right = np.full(shape=dimention, fill_value=bounds[1], dtype=np.float64)

                        h = np.full(shape=dimention, fill_value=h_all, dtype=np.float64)

                        genotype_to_phenotype = SamplingGrid(fit_by="h").fit(
                            left=left, right=right, arg=h
                        )

                        if "init_bounds" in problem.keys():
                            float_init_population = float_population(
                                pop_size=pop_size_i, left=left, right=right
                            )

                            bit_init_population = genotype_to_phenotype.inverse_transform(
                                float_init_population
                            )
                        else:
                            bit_init_population = None

                        optimizer = GeneticAlgorithm(
                            fitness_function=fitness_function(),
                            genotype_to_phenotype=genotype_to_phenotype.transform,
                            iters=iters_i,
                            pop_size=pop_size_i,
                            str_len=np.sum(genotype_to_phenotype.parts),
                            elitism=True,
                            selection=selection,
                            crossover=crossover,
                            mutation=mutation,
                            init_population=bit_init_population,
                            minimization=True,
                        )

                        optimizer.fit()

                        fittest = optimizer.get_fittest()

                        row_name = f"{function}_{dimention}"
                        col_name = f"{selection}_{crossover}_{mutation}"

                        result_fitness.loc[row_name, col_name] = (-1) * fittest["fitness"]

                        reliability = check_solution(
                            x=fittest["phenotype"], optimum_x=optimum_x, h=genotype_to_phenotype.h
                        )

                        result_succes.loc[row_name, col_name] = reliability

    if i == 0:
        result_fitness.to_excel("result_fitness.xlsx", sheet_name=str(i))

        result_succes.to_excel("result_succes.xlsx", sheet_name=str(i))

        result_reliability = result_succes.copy()

    else:
        with pd.ExcelWriter("result_fitness.xlsx", engine="openpyxl", mode="a") as writer:
            result_fitness.to_excel(writer, sheet_name=str(i))

        with pd.ExcelWriter("result_succes.xlsx", engine="openpyxl", mode="a") as writer:
            result_succes.to_excel(writer, sheet_name=str(i))

        result_reliability += result_succes.copy()

result_reliability = result_reliability / n_runs

result_reliability.loc[:, "mean"] = result_reliability.mean(axis=1)
result_reliability.to_excel("result_reliability.xlsx")

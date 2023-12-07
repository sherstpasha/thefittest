from collections import defaultdict
from thefittest.optimizers._my_adapt_ga import MyAdaptGA
from thefittest.benchmarks import Sphere
import matplotlib.pyplot as plt

import numpy as np
from thefittest.tools.transformations import GrayCode
from thefittest.benchmarks import Rastrigin


n_dimension = 10
left_border = -5.0
right_border = 5.0
n_bits_per_variable = 32

number_of_iterations = 300
population_size = 20

left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)
parts = np.full(shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

genotype_to_phenotype = GrayCode(fit_by="parts").fit(
    left=left_border_array, right=right_border_array, arg=parts
)
optimizer = MyAdaptGA(
    fitness_function=Sphere(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=sum(parts),
    show_progress_each=1,
    minimization=True,
    selections=("tournament_k", "rank", "proportional"),
    crossovers=("two_point", "one_point", "uniform_2", "uniform_rank_2"),
    mutations=("weak", "average", "strong"),
    tour_size=5,
    keep_history=True,
)


optimizer.fit()
# print(optimizer._adaptation_operator)
# print(optimizer._crossover_operators)

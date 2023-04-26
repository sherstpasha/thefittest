import numpy as np
from thefittest.optimizers import DifferentialEvolution
from thefittest.benchmarks import Griewank
from thefittest.tools.transformations import donothing
from line_profiler import LineProfiler
from thefittest.tools.generators import float_population
from thefittest.tools.operators import binomial
from thefittest.tools.operators import rand_1
from thefittest.tools.operators import rand_2
from functools import partial
import random
from thefittest.tools.numba_funcs import rand_1_

n_dimension = 1000
left_border = -100.
right_border = 100.


number_of_iterations = 500
population_size = 5000


left_border_array = np.full(
    shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(
    shape=n_dimension, fill_value=right_border, dtype=np.float64)


population_g = float_population(
    population_size, left_border_array, right_border_array)
population_ph = donothing(population_g)
fitness = Griewank()(population_ph)


def _bounds_control(
        individ_g: np.ndarray) -> np.ndarray:
    individ_g = individ_g.copy()
    low_mask = individ_g < left_border_array
    high_mask = individ_g > right_border_array

    individ_g[low_mask] = left_border_array[low_mask]
    individ_g[high_mask] = right_border_array[high_mask]
    return individ_g


def _mutation_and_crossover(
        popuation_g: np.ndarray,
        individ_g: np.ndarray,
        F: float,
        CR: float) -> np.ndarray:
    mutant = rand_1(individ_g, popuation_g, F)
    mutant_cr_g = binomial(individ_g, mutant, CR)
    # mutant_cr_g = _bounds_control(mutant_cr_g)
    return mutant_cr_g




F_i = [0.5]*population_size
CR_i = [0.5]*population_size

# _mutation_and_crossover(population_g, population_g[0], F_i[0], CR_i[0])
# mutation_and_crossover = partial(_mutation_and_crossover,
#                                  population_g)
# mutant_cr_g = np.array(list(map(mutation_and_crossover,
#                                 population_g, F_i, CR_i)))
rand_1(population_g[0], population_g, F_i[0])
rand_1_(population_g[0], population_g, F_i[0])
import time
n = 5000
begin = time.time()
for _ in range(n):
    rand_1(population_g[0], population_g, F_i[0])
print(time.time() - begin)

begin = time.time()
for _ in range(n):
    rand_1_(population_g[0], population_g, F_i[0])
print(time.time() - begin)

# begin = time.time()
# for _ in range(n):
#     rand_222(population_g[0], population_g, F_i[0])
# print(time.time() - begin)

# lp = LineProfiler()
# lp_wrapper = lp(_mutation_and_crossover)
# lp_wrapper(population_g, population_g[0], F_i[0], CR_i[0])
# lp.print_stats()

# lp = LineProfiler()
# lp_wrapper = lp(rand_12)
# lp_wrapper(population_g[0], population_g, F_i[0])
# lp.print_stats()

# model = DifferentialEvolution(fitness_function=Griewank(),
#                               genotype_to_phenotype=donothing,
#                               iters=number_of_iterations,
#                               pop_size=population_size,
#                               left=left_border_array,
#                               right=right_border_array,
#                               minimization=True)

# model.set_strategy(mutation_oper='rand_1', F_param=0.1, CR_param=0.5)

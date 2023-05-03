import time
import numpy as np
from thefittest.tools.random import float_population
from thefittest.benchmarks import Sphere
from thefittest.tools.random import random_sample
from numba import njit
from numba import float64
from numpy.typing import NDArray
import random








n_vars = 500
left = np.full(n_vars, -5, dtype=np.float64)
right = np.full(n_vars, 5, dtype=np.float64)

population = float_population(5000, left, right)
fitness = Sphere()(population)

argsort = np.argsort(fitness)
population = population[argsort]
fitness = fitness[argsort]


n = 10000

begin = time.time()
for i in range(n):
    binomial(population[0], population[1], 1)
print(time.time() - begin)

begin = time.time()
for i in range(n):
    binomial_(population[0], population[1], np.float64(1))
print(time.time() - begin)

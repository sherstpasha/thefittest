import time
import numpy as np
import numba
import random
from numba import float64
from numba import njit
from numpy.typing import NDArray
from thefittest.tools.random import random_sample
from thefittest.tools.random import float_population


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def best_1(current: NDArray[np.float64],
           best: NDArray[np.float64],
           population: NDArray[np.float64],
           F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2 = random_sample(
        range_size=size, quantity=np.int64(2), replace=False)
        
    return best + F*(population[r1] - population[r2])

@njit(float64[:](float64[:], float64[:], float64))
def binomial(individ: NDArray[np.float64],
             mutant: NDArray[np.float64],
             CR: np.float64):
    size = len(individ)
    offspring = individ.copy()
    j = random.randrange(size)

    for i in range(size):
        if np.random.rand() <= CR or i == j:
            offspring[i] = mutant[i]
    return offspring


@njit(float64[:](float64[:], float64[:], float64[:]))
def bounds_control(array: NDArray[np.float64],
                   left: NDArray[np.float64],
                   right: NDArray[np.float64]) -> NDArray[np.float64]:
    to_return = array.copy()

    size = len(array)
    for i in range(size):
        if array[i] < left[i]:
            to_return[i] = left[i]
        elif array[i] > right[i]:
            to_return[i] = right[i]
    return to_return

def best_1_(current, population, F_value):
    best = population[-1]
    r1, r2 = np.random.choice(
        np.arange(len(population)), size=2, replace=False)
    return best + F_value*(population[r1] - population[r2])

def binomial_(individ, mutant, CR):
    individ = individ.copy()
    j = random.randrange(len(individ))
    mask_random = np.random.random(len(individ)) <= CR
    mask_j = np.arange(len(individ)) == j
    mask_union = mask_random | mask_j
    individ[mask_union] = mutant[mask_union].copy()
    return individ

def bounds_control_(individ_g, left, right):
    individ_g = individ_g.copy()
    low_mask = individ_g < left
    high_mask = individ_g > right

    individ_g[low_mask] = left[low_mask]
    individ_g[high_mask] = right[high_mask]
    return individ_g

def custom_problem(x):
    return np.sum(x**2, axis=-1)


n_dimension = 1000
left_border = -100.
right_border = 100.


number_of_iterations = 500
population_size = 500


left_border_array = np.full(
    shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(
    shape=n_dimension, fill_value=right_border, dtype=np.float64)

population = float_population(population_size, left_border_array, right_border_array)

fitness = custom_problem(population)

n = 1000
begin = time.time()
for i in range(n):
    mutant = best_1(population[0], population[1], population, 0.5)
    mutant_cr_g = binomial(population[0], mutant, 0.5)
    mutant_cr_g = bounds_control(mutant_cr_g, left_border_array, right_border_array)
print(time.time() - begin)

begin = time.time()
for i in range(n):
    mutant = best_1_(population[0], population, 0.5)
    mutant_cr_g_ = binomial_(population[0], mutant, 0.5)
    mutant_cr_g = bounds_control_(mutant_cr_g, left_border_array, right_border_array)
print(time.time() - begin)
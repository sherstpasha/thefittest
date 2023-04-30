import time
import numpy as np
from numba import njit
from numba import int32
from numba import float32
from numba import float64
from numba import int8
import numba
from thefittest.tools.generators import binary_string_population
import random
from thefittest.benchmarks import OneMax
from thefittest.tools.transformations import rank_data
from thefittest.tools.transformations import protect_norm
from thefittest.tools.numba_funcs import nb_choice, select_quantity_id_by_tournament
from thefittest.tools.operators import tournament_selection

def protect_norm_old(x: np.ndarray) -> np.ndarray:
    sum_ = x.sum()
    if sum_ > 0:
        to_return = x/sum_
    else:
        len_ = len(x)
        to_return = np.full(len_, 1/len_)
    return to_return


@njit
def one_point_crossover(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray) -> np.ndarray:
    cross_point = random.randrange(1, len(individs[0]))
    if random.random() > 0.5:
        offspring = individs[0].copy()
        for i in range(individs.shape[1]):
            if i > cross_point:
                offspring[i] = individs[1][i]
    else:
        offspring = individs[1].copy()
        for i in range(individs.shape[1]):
            if i > cross_point:
                offspring[i] = individs[0][i]
    return offspring

@njit(int8[:](int8[:,:], float64[:], float64[:]))
def one_point_crossover2(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray) -> np.ndarray:
    print(fitness.dtype)
    print(rank.dtype)
    cross_point = random.randrange(1, len(individs[0]))
    if random.random() > 0.5:
        offspring = individs[0].copy()
        for i in range(individs.shape[1]):
            if i > cross_point:
                offspring[i] = individs[1][i]
    else:
        offspring = individs[1].copy()
        for i in range(individs.shape[1]):
            if i > cross_point:
                offspring[i] = individs[0][i]
    return offspring
    # range_ = np.arange(len(individs))
    # diag = np.arange(len(individs[0]))

    # tournament = np.random.choice(range_, 2*len(individs[0]))
    # tournament = tournament.reshape(-1, 2)
    # choosen = np.argmin(fitness[tournament], axis=1)
    # offspring = individs[choosen, diag].copy()
    # return offspring



pop_size = 100
str_len = 1000

population = binary_string_population(pop_size=pop_size, str_len=str_len)
fitness = OneMax()(population).astype(np.float64)
ranks = rank_data(fitness).astype(np.float64)


# tournament_selection_n(fitness, fitness, 4, 7)1

# print(population[[0, 1]])
# offs = flip_mutation(population[0], 1)
# offs = flip_mutation2(population[0], 1)



n = 1
print('start')
begin = time.time()
for i in range(n):
    arr1 = one_point_crossover(individs=population[[1, 2]],
                               fitness=fitness[[1, 2]],
                               rank=ranks[[1,2]])
print(time.time() - begin)

begin = time.time()
for i in range(n):
    arr1 = one_point_crossover2(individs=population[[1, 2]],
                               fitness=fitness[[1, 2]],
                               rank=ranks[[1,2]])
    # print(arr2)
print(time.time() - begin)

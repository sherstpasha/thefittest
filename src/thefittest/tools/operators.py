import random
from typing import Any
from typing import Union

from numba import float64
from numba import int64
from numba import int8
from numba import njit
from numba.types import List as numbaListType

import numpy as np
from numpy.typing import NDArray

from ._numba_funcs import max_axis
from .random import growing_method
from .random import random_sample
from .random import random_weighted_sample
from .random import sattolo_shuffle
from .transformations import common_region
from ..base import Tree
from ..base import UniversalSet


min_value = np.finfo(np.float64).min
max_value = np.finfo(np.float64).max


# MUTATUIONS
# genetic algorithm
@njit(int8[:](int8[:], float64))
def flip_mutation(individ: np.ndarray,
                  proba: float):
    offspring = individ.copy()
    for i in range(offspring.size):
        if random.random() < proba:
            offspring[i] = 1 - offspring[i]
    return offspring


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def best_1(current: NDArray[np.float64],
           best: NDArray[np.float64],
           population: NDArray[np.float64],
           F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2 = random_sample(
        range_size=size, quantity=np.int64(2), replace=False)
    return best + F * (population[r1] - population[r2])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_1(current: NDArray[np.float64],
           best: NDArray[np.float64],
           population: NDArray[np.float64],
           F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(
        range_size=size, quantity=np.int64(3), replace=False)
    return population[r3] + F * (population[r1] - population[r2])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_to_best1(current: NDArray[np.float64],
                  best: NDArray[np.float64],
                  population: NDArray[np.float64],
                  F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(
        range_size=size, quantity=np.int64(3), replace=False)
    return population[r1] + F * (
        population[r1]) + F * (population[r2] - population[r3])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def current_to_best_1(current: NDArray[np.float64],
                      best: NDArray[np.float64],
                      population: NDArray[np.float64],
                      F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2 = random_sample(
        range_size=size, quantity=np.int64(2), replace=False)
    return current + F * (best - current) + F * (population[r1] - population[r2])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def best_2(current: NDArray[np.float64],
           best: NDArray[np.float64],
           population: NDArray[np.float64],
           F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2, r3, r4 = random_sample(
        range_size=size, quantity=np.int64(4), replace=False)
    return best + F * (population[r1] - population[r2]) +\
        F * (population[r3] - population[r4])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_2(current: NDArray[np.float64],
           best: NDArray[np.float64],
           population: NDArray[np.float64],
           F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2, r3, r4, r5 = random_sample(
        range_size=size, quantity=np.int64(5), replace=False)
    return population[r5] + F * (population[r1] - population[r2]) +\
        F * (population[r3] - population[r4])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def current_to_rand_1_(current: NDArray[np.float64],
                       best: NDArray[np.float64],
                       population: NDArray[np.float64],
                       F: np.float64) -> NDArray[np.float64]:
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(
        range_size=size, quantity=np.int64(3), replace=False)
    return population[r1] + F * (population[r3] - current) +\
        F * (population[r1] - population[r2])


@njit(float64[:](float64[:], float64[:, :], int64[:], float64, float64[:, :]))
def current_to_pbest_1_archive(
        current: NDArray[np.float64],
        population: NDArray[np.float64],
        pbest: NDArray[np.int64],
        F: np.float64,
        pop_archive: NDArray[np.float64]) -> NDArray[np.float64]:
    p_best_ind = random.randrange(len(pbest))
    best = population[pbest[p_best_ind]]
    r1 = random_sample(range_size=len(population),
                       quantity=np.int64(1), replace=True)[0]
    r2 = random_sample(range_size=len(pop_archive),
                       quantity=np.int64(1), replace=True)[0]
    return current + F * (best - current) + F * (
        population[r1] - pop_archive[r2])


@njit(float64[:](float64[:], float64[:, :], int64[:], float64, float64[:, :]))
def current_to_pbest_1_archive_p_min(
        current: NDArray[np.float64],
        population: NDArray[np.float64],
        pbest: NDArray[np.int64],
        F: np.float64,
        pop_archive: NDArray[np.float64]) -> NDArray[np.float64]:
    size = len(population)
    p_min = 2 / size
    p_i = np.random.uniform(p_min, 0.2)
    value = np.int64(p_i * size)
    pbest_cut = pbest[:value]

    p_best_ind = random.randrange(len(pbest_cut))
    best = population[pbest_cut[p_best_ind]]
    r1 = random_sample(range_size=size,
                       quantity=np.int64(1), replace=True)[0]
    r2 = random_sample(range_size=len(pop_archive),
                       quantity=np.int64(1), replace=True)[0]
    return current + F * (best - current) + F * (
        population[r1] - pop_archive[r2])


# genetic propramming
def point_mutation(tree: Tree,
                   uniset: UniversalSet,
                   proba: float,
                   max_level) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        if to_return._nodes[i].is_functional():
            n_args = to_return._nodes[i]._n_args
            new_node = uniset._random_functional(n_args)
        else:
            new_node = uniset._random_terminal_or_ephemeral()
        to_return._nodes[i] = new_node
    return to_return


def growing_mutation(tree: Tree,
                     uniset: UniversalSet,
                     proba: float,
                     max_level) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        grown_tree = growing_method(uniset, max(to_return.get_levels(i)))
        to_return = to_return.concat(i, grown_tree)
    return to_return


def swap_mutation(tree: Tree,
                  uniset: UniversalSet,
                  proba: float,
                  max_level) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        more_one_args_cond = to_return._n_args > 1
        indexes = np.arange(len(tree), dtype=int)[more_one_args_cond]
        if len(indexes) > 0:
            i = random.choice(indexes)
            args_id = to_return.get_args_id(i)
            new_arg_id = args_id.copy()
            sattolo_shuffle(new_arg_id)
            for old_j, new_j in zip(args_id, new_arg_id):
                subtree = tree.subtree(old_j, return_class=True)
                to_return = to_return.concat(new_j, subtree)
    return to_return


def shrink_mutation(tree: Tree,
                    uniset: UniversalSet,
                    proba: float,
                    max_level) -> Tree:
    to_return = tree.copy()
    if len(to_return) > 2:
        if random.random() < proba:
            no_terminal_cond = to_return._n_args > 0
            indexes = np.arange(len(tree), dtype=int)[no_terminal_cond]
            if len(indexes) > 0:
                i = random.choice(indexes)
                args_id = to_return.get_args_id(i)
                choosen_id = random.choice(args_id)
                to_return = to_return.concat(i, tree.subtree(
                    choosen_id, return_class=True))
    return to_return


# CROSSOVERS
# genetic algorithm
def empty_crossover(individs: np.ndarray,
                    fitness: np.ndarray,
                    rank: np.ndarray,
                    *args) -> np.ndarray:
    offspring = individs[0].copy()
    return offspring


@njit(int8[:](int8[:], int8[:], float64))
def binomialGA(individ: NDArray[np.int8],
               mutant: NDArray[np.int8],
               CR: np.float64):
    size = len(individ)
    offspring = individ.copy()
    j = random.randrange(size)

    for i in range(size):
        if np.random.rand() <= CR or i == j:
            offspring[i] = mutant[i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def one_point_crossover(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray) -> np.ndarray:
    cross_point = random_sample(range_size=len(individs[0]),
                                quantity=1, replace=True)[0]
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


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def two_point_crossover(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray) -> np.ndarray:
    size = len(individs[0])
    c_points = random_sample(range_size=len(individs[0]),
                             quantity=2, replace=False)
    c_points = sorted(c_points)

    if random.random() > 0.5:
        offspring = individs[0].copy()
        other_individ = individs[1]
    else:
        offspring = individs[1].copy()
        other_individ = individs[0]
    for i in range(size):
        if c_points[0] <= i <= c_points[1]:
            offspring[i] = other_individ[i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_crossover(individs: np.ndarray,
                      fitness: np.ndarray,
                      rank: np.ndarray) -> np.ndarray:
    choosen = random_sample(range_size=len(fitness),
                            quantity=len(individs[0]),
                            replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_prop_crossover(individs: np.ndarray,
                           fitness: np.ndarray,
                           rank: np.ndarray) -> np.ndarray:
    choosen = random_weighted_sample(weights=fitness,
                                     quantity=len(individs[0]),
                                     replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_rank_crossover(individs: np.ndarray,
                           fitness: np.ndarray,
                           rank: np.ndarray) -> np.ndarray:
    choosen = random_weighted_sample(weights=rank,
                                     quantity=len(individs[0]),
                                     replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


def uniform_tour_crossover(individs: np.ndarray,
                           fitness: np.ndarray,
                           rank: np.ndarray) -> np.ndarray:
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    tournament = np.random.choice(range_, 2 * len(individs[0]))
    tournament = tournament.reshape(-1, 2)
    choosen = np.argmax(fitness[tournament], axis=1)
    offspring = individs[choosen, diag].copy()
    return offspring


# differential evolution
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


# genetic propramming
def standart_crossover(individs: np.ndarray,
                       fitness: np.ndarray,
                       rank: np.ndarray,
                       max_level: int) -> Tree:
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
    first_point = random.randrange(len(individ_1))
    second_point = random.randrange(len(individ_2))

    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat(second_point, first_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_2
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat(first_point, second_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_1
    return offspring


def one_point_crossoverGP(individs: np.ndarray,
                          fitness: np.ndarray,
                          rank: np.ndarray,
                          max_level: int) -> Tree:
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = common_region([individ_1, individ_2])

    point = random.randrange(len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring


def uniform_crossoverGP(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray,
                        max_level: int) -> Tree:
    '''Poli, Riccardo & Langdon, W.. (2001). On the Search
    Properties of Different Crossover Operators in Genetic Programming. '''
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    pool = random_sample(range_size=len(fitness),
                         quantity=len(common[0]),
                         replace=True)
    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


def uniform_crossoverGP_prop(individs: np.ndarray,
                             fitness: np.ndarray,
                             rank: np.ndarray,
                             max_level: int) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    pool = random_weighted_sample(weights=fitness,
                                  quantity=len(common[0]),
                                  replace=True)
    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


def uniform_crossoverGP_rank(individs: np.ndarray,
                             fitness: np.ndarray,
                             rank: np.ndarray,
                             max_level: int) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    pool = random_weighted_sample(weights=rank,
                                  quantity=len(common[0]),
                                  replace=True)

    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


def uniform_crossoverGP_tour(individs: np.ndarray,
                             fitness: np.ndarray,
                             rank: np.ndarray,
                             max_level: int) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    pool = tournament_selection(fitness, rank, 2, len(common[0]))

    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


# SELECTIONS
# genetic algorithm
@njit(int64[:](float64[:], float64[:], int64, int64))
def proportional_selection(fitness: np.ndarray,
                           rank: np.ndarray,
                           tour_size: int,
                           quantity: int) -> np.ndarray:
    choosen = random_weighted_sample(weights=fitness,
                                     quantity=quantity,
                                     replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def rank_selection(fitness: np.ndarray,
                   rank: np.ndarray,
                   tour_size: int,
                   quantity: int) -> np.ndarray:
    choosen = random_weighted_sample(weights=rank,
                                     quantity=quantity,
                                     replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def tournament_selection(fitness: NDArray[np.float64],
                         rank: NDArray[np.float64],
                         tour_size: np.int64,
                         quantity: np.int64) -> NDArray[np.int64]:
    to_return = np.empty(quantity, dtype=np.int64)
    for i in range(quantity):
        tournament = random_sample(range_size=len(fitness),
                                   quantity=tour_size, replace=False)
        argmax = np.argmax(fitness[tournament])
        to_return[i] = tournament[argmax]
    return to_return


# GP MATH
class Operator:
    def __init__(self,
                 formula: str,
                 name: str,
                 sign: str) -> None:
        self._formula = formula
        self.__name__ = name
        self._sign = sign

    def _write(self,
               *args) -> str:
        formula = self._formula.format(*args)
        return formula

    def __call__(self, *args: Any) -> None:
        pass


class Cos(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='cos({})',
                          name='cos',
                          sign='cos')

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.cos(x)
        return result


class Sin(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='sin({})',
                          name='sin',
                          sign='sin')

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.sin(x)
        return result


class Add(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='({} + {})',
                          name='add',
                          sign='+')

    def __call__(self,
                 x: Union[float, NDArray[np.float64]],
                 y: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        result = x + y
        return result


class Sub(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='({} - {})',
                          name='sub',
                          sign='-')

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = x - y
        return result


class Neg(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='-{}',
                          name='neg',
                          sign='-')

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = -x
        return result


class Mul(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='({} * {})',
                          name='mul',
                          sign='*')

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = x * y
        return result


class Pow2(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='({}**2)',
                          name='pow2',
                          sign='**2')

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.clip(x**2, min_value, max_value)
        return result


class Div(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='({}/{})',
                          name='div',
                          sign='/')

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(y, np.ndarray):
            result = np.divide(x, y, out=np.ones_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                result = 0
            else:
                result = x / y
        result = np.clip(result, min_value, max_value)
        return result


class Inv(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='(1/{})',
                          name='Inv',
                          sign='1/')

    def __call__(self,
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if type(y) == np.ndarray:
            result = np.divide(1, y, out=np.ones_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                result = 1
            else:
                result = 1 / y
        result = np.clip(result, min_value, max_value)
        return result


class LogAbs(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='log(abs({}))',
                          name='log(abs)',
                          sign='log(abs)')

    def __call__(self,
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        y_ = np.abs(y)
        if isinstance(y_, np.ndarray):
            result = np.log(y_, out=np.ones_like(
                y_, dtype=np.float64), where=y_ != 0)
        else:
            if y_ == 0:
                result = 1
            else:
                result = np.log(y_)
        return result


class Exp(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='exp({})',
                          name='exp',
                          sign='exp')

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.clip(np.exp(x), min_value, max_value)
        return result


class SqrtAbs(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='sqrt(abs({}))',
                          name='sqrt(abs)',
                          sign='sqrt(abs)')

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.sqrt(np.abs(x))
        return result


class Abs(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='abs({})',
                          name='abs()',
                          sign='abs()')

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.abs(x)
        return result


class More(Operator):
    def __init__(self) -> None:
        Operator.__init__(self,
                          formula='({} > {})',
                          name='more',
                          sign='>')

    def __call__(self,
                 x: Union[float, NDArray[np.float64]],
                 y: Union[float, NDArray[np.float64]]) -> Union[bool, NDArray[np.bool_]]:
        result = x > y
        return result


# Activation functions
@njit(float64[:, :](float64[:, :]))
def softmax_numba(X: NDArray[np.float64]) -> NDArray[np.float64]:
    exps = np.exp(X - max_axis(X))
    sum_ = np.sum(exps, axis=1)
    for j in range(sum_.shape[0]):
        if sum_[j] == 0:
            sum_[j] = 1
    result = ((exps).T / sum_).T
    return result


@njit(float64[:, :](float64[:, :], int64))
def multiactivation2d(X: NDArray[np.float64],
                      activ_id: np.int64) -> NDArray[np.float64]:
    if activ_id == 0:
        result = 1 / (1 + np.exp(-X))
    elif activ_id == 1:
        result = X * (X > 0)
    elif activ_id == 2:
        result = np.exp(-(X**2))
    elif activ_id == 3:
        result = np.tanh(X)
    elif activ_id == 4:
        result = softmax_numba(X)
    return result


@njit(float64[:, ::1](float64[:], int64[:, :]))
def mask2d(arr, mask):
    arr_mask = arr[mask.flatten()]
    arr_mask_reshape = arr_mask.reshape(mask.shape)
    return arr_mask_reshape


@njit
def forward(X: NDArray[np.float64],
            weights: NDArray[np.float64],
            nodes: NDArray[np.float64],
            from_: numbaListType(NDArray[np.int64]),
            to_: numbaListType(NDArray[np.int64]),
            weights_id: numbaListType(NDArray[np.int64]),
            activs_code: numbaListType(NDArray[np.int64]),
            activs_nodes: numbaListType(numbaListType(NDArray[np.int64]))) -> NDArray[np.float64]:

    for from_i, to_i, weights_id_i, a_code_i, a_nodes_i in zip(from_,
                                                               to_,
                                                               weights_id,
                                                               activs_code,
                                                               activs_nodes):
        weights_i = mask2d(weights, weights_id_i)
        out = np.dot(nodes[from_i].T, weights_i.T)
        nodes[to_i] = out.T

        for a_code_i_i, a_nodes_i_i in zip(a_code_i, a_nodes_i):
            nodes[a_nodes_i_i] = multiactivation2d(
                nodes[a_nodes_i_i].T, a_code_i_i).T

    return nodes


@njit
def forward2d(X: NDArray[np.float64],
              inputs: NDArray[np.int64],
              n_hiddens: np.int64,
              outputs: NDArray[np.int64],
              from_: numbaListType(NDArray[np.int64]),
              to_: numbaListType(NDArray[np.int64]),
              weights_id: numbaListType(NDArray[np.int64]),
              activs_code: numbaListType(NDArray[np.int64]),
              activs_nodes: numbaListType(numbaListType(NDArray[np.int64])),
              weights: NDArray[np.float64]) -> NDArray[np.float64]:

    outs = np.empty(shape=(len(weights), X.shape[0], len(outputs)))
    num_nodes = X.shape[1] + n_hiddens + len(outputs)
    shape = (num_nodes, len(X))
    nodes = np.empty(shape, dtype=np.float64)
    nodes[inputs] = X.T[inputs]

    for n in range(outs.shape[0]):
        forward(X,
                weights[n],
                nodes,
                from_,
                to_,
                weights_id,
                activs_code,
                activs_nodes)

        outs[n] = nodes[outputs].T
    return outs

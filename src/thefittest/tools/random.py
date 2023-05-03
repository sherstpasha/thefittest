import random
from typing import List
from numba import njit
from numba import int64
from numba import float64
from numba import boolean
import numpy as np
from numpy.typing import NDArray
from ..base import Tree
from ..base import UniversalSet
from ..tools import binary_search_interval
from ..tools import check_for_value


def sattolo_shuffle(items: List) -> None:
    i = len(items)
    while i > 1:
        i = i - 1
        j = random.randrange(i)
        items[j], items[i] = items[i], items[j]


def binary_string_population(pop_size: int,
                             str_len: int) -> NDArray[np.byte]:
    size = (pop_size, str_len)
    return np.random.randint(low=2, size=size, dtype=np.byte)


def float_population(pop_size: int,
                     left: NDArray[np.float64],
                     right: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([np.random.uniform(left_i, right_i, pop_size)
                     for left_i, right_i in zip(left, right)]).T


def cauchy_distribution(loc: int = 0,
                        scale: int = 1,
                        size: int = 1) -> NDArray[np.float64]:
    x_ = np.random.standard_cauchy(size=size)
    return loc + scale*x_


def full_growing_method(uniset: UniversalSet,
                        max_level: int) -> Tree:
    nodes = []
    levels = []
    n_args = []
    possible_steps = [1]
    previous_levels = [-1]
    level_i = -1
    while len(possible_steps):
        possible_steps[-1] = possible_steps[-1] - 1
        if possible_steps[-1] == 0:
            possible_steps.pop()
            level_i = previous_levels.pop() + 1
        else:
            level_i = previous_levels[-1] + 1
        levels.append(level_i)
        if level_i == max_level:
            nodes.append(uniset._random_terminal_or_ephemeral())
            n_args.append(0)
        else:
            nodes.append(uniset._random_functional())
            n_i = nodes[-1]._n_args
            n_args.append(n_i)
            possible_steps.append(n_i)
            previous_levels.append(level_i)
    to_return = Tree(nodes, n_args)
    return to_return


def growing_method(uniset: UniversalSet,
                   max_level: int) -> Tree:

    nodes = []
    levels = []
    n_args = []
    possible_steps = [1]
    previous_levels = [-1]
    level_i = -1
    while len(possible_steps):
        possible_steps[-1] = possible_steps[-1] - 1
        if possible_steps[-1] == 0:
            possible_steps.pop()
            level_i = previous_levels.pop() + 1
        else:
            level_i = previous_levels[-1] + 1
        levels.append(level_i)

        if level_i == max_level:
            nodes.append(uniset._random_terminal_or_ephemeral())
            n_args.append(0)
        elif level_i == 0:
            nodes.append(uniset._random_functional())
            n_i = nodes[-1]._n_args
            n_args.append(n_i)
            possible_steps.append(n_i)
            previous_levels.append(level_i)
        else:
            if np.random.random() < 0.5:
                nodes.append(uniset._random_terminal_or_ephemeral())
            else:
                nodes.append(uniset._random_functional())
            n_i = nodes[-1]._n_args
            n_args.append(n_i)

            if n_i > 0:
                possible_steps.append(n_i)
                previous_levels.append(level_i)
    to_return = Tree(nodes, n_args)
    return to_return


def random_tree(uniset: UniversalSet,
                max_level: int) -> Tree:
    if random.random() < 0.5:
        to_return = full_growing_method(uniset, max_level)
    else:
        to_return = growing_method(uniset, max_level)
    return to_return


def half_and_half(pop_size: int,
                  uniset: UniversalSet,
                  max_level: int) -> NDArray:
    population = [random_tree(uniset, random.randrange(2, max_level))
                  for _ in range(pop_size)]
    return np.array(population, dtype=object)


@njit(int64[:](float64[:], int64, boolean))
def random_weighted_sample(weights: NDArray[np.float64],
                           quantity: np.int64 = 1,
                           replace: bool = True) -> NDArray[np.int64]:
    if not replace:
        assert len(weights) >= quantity
    to_return = np.empty(quantity, dtype=np.int64)

    cumsumweights = np.cumsum(weights)
    sumweights = cumsumweights[-1]

    i = 0
    while i < quantity:
        roll = sumweights*np.random.rand()
        ind = binary_search_interval(roll, cumsumweights)
        if not replace:
            if check_for_value(ind, to_return, i):
                continue

        to_return[i] = ind
        i += 1
    return to_return


@njit(int64[:](int64, int64, boolean))
def random_sample(range_size: np.int64,
                  quantity: np.int64,
                  replace: bool = True) -> NDArray[np.int64]:
    if not replace:
        assert range_size >= quantity
    to_return = np.empty(quantity, dtype=np.int64)
    i = 0
    while i < quantity:
        ind = random.randrange(range_size)

        if not replace:
            if check_for_value(ind, to_return, i):
                continue

        to_return[i] = ind
        i += 1
    return to_return

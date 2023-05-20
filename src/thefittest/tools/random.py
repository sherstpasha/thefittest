import random
from typing import List
from typing import Tuple
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
from ..tools.transformations import numpy_group_by


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


@njit(float64[:](float64, float64, int64))
def cauchy_distribution(loc: np.float64,
                        scale: np.float64,
                        size: np.int64) -> NDArray[np.float64]:
    x_ = np.random.standard_cauchy(size=size).astype(np.float64)
    return loc + scale*x_


@njit(float64(float64))
def randc01(u):
    value = cauchy_distribution(
        loc=u, scale=np.float64(0.1), size=np.int64(1))[0]
    while value <= 0:
        value = cauchy_distribution(
            loc=u, scale=np.float64(0.1), size=np.int64(1))[0]
    if value > 1:
        value = 1
    return value


@njit(float64(float64))
def randn01(u):
    value = np.random.normal(u, 0.1)
    if value < 0:
        value = 0
    elif value > 1:
        value = 1
    return value


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


def stratified_sample(data: NDArray[np.int64],
                      sample_ratio: float) -> NDArray[np.int64]:
    to_return = []
    data_size = len(data)
    sample_size = int(sample_ratio*data_size)
    indexes = np.arange(len(data), dtype=np.int64)
    keys, groups = numpy_group_by(indexes, data)
    assert sample_size >= len(keys)

    for group in groups:
        group_size = len(group)
        sample_size_i = int((group_size/data_size)*sample_size)
        sample_i_id = random_sample(group_size, sample_size_i, False)
        sample_i = group[sample_i_id]
        to_return.extend(sample_i)

    return np.array(to_return, dtype=np.int64)


def train_test_split_stratified(X: NDArray[np.float64],
                                y: NDArray[np.int64],
                                tests_size: float) -> Tuple:
    indexes = np.arange(len(y), dtype=np.int64)
    sample_id = stratified_sample(y,  tests_size)
    test_id = sample_id
    train_id = np.setdiff1d(indexes, test_id)
    return (X[train_id].astype(np.float64),
            X[test_id].astype(np.float64),
            y[train_id].astype(np.int64),
            y[test_id].astype(np.int64))

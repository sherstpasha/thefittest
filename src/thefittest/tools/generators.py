import numpy as np
from ..base import Tree
from ..base import UniversalSet
import random
from typing import List
from typing import Iterable


def sattolo_shuffle(items: List) -> None:
    i = len(items)
    while i > 1:
        i = i - 1
        j = random.randrange(i)
        items[j], items[i] = items[i], items[j]


def cauchy_distribution(loc: int = 0,
                        scale: int = 1,
                        size: int = 1) -> np.ndarray:
    x_ = np.random.standard_cauchy(size=size)
    return loc + scale*x_


def binary_string_population(pop_size: int,
                             str_len: int) -> np.ndarray:
    size = (pop_size, str_len)
    return np.random.randint(low=2, size=size, dtype=np.byte)


def float_population(pop_size: int,
                     left: Iterable,
                     right: Iterable) -> np.ndarray:
    return np.array([np.random.uniform(left_i, right_i, pop_size)
                     for left_i, right_i in zip(left, right)]).T


def full_growing_method(uniset: UniversalSet,
                        max_level: int):
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
            nodes.append(uniset.random_terminal_or_ephemeral())
            n_args.append(0)
        else:
            nodes.append(uniset.random_functional())
            n_i = nodes[-1].n_args
            n_args.append(n_i)
            possible_steps.append(n_i)
            previous_levels.append(level_i)
    to_return = Tree(nodes, np.array(n_args, dtype=np.int32))
    return to_return


def growing_method(uniset: UniversalSet,
                   max_level: int):

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
            nodes.append(uniset.random_terminal_or_ephemeral())
            n_args.append(0)
        elif level_i == 0:
            nodes.append(uniset.random_functional())
            n_i = nodes[-1].n_args
            n_args.append(n_i)
            possible_steps.append(n_i)
            previous_levels.append(level_i)
        else:
            if np.random.random() < 0.5:
                nodes.append(uniset.random_terminal_or_ephemeral())
            else:
                nodes.append(uniset.random_functional())
            n_i = nodes[-1].n_args
            n_args.append(n_i)

            if n_i > 0:
                possible_steps.append(n_i)
                previous_levels.append(level_i)
    to_return = Tree(nodes, np.array(n_args, dtype=np.int32))
    return to_return


def random_method(uniset: UniversalSet,
                  max_level: int) -> Tree:
    if random.random() < 0.5:
        to_return = full_growing_method(uniset, max_level)
    else:
        to_return = growing_method(uniset, max_level)
    return to_return


def half_and_half(pop_size: int,
                  uniset: UniversalSet,
                  max_level: int) -> List:
    population = [random_method(uniset, random.randrange(2, max_level))
                  for _ in range(pop_size)]
    return np.array(population, dtype=object)

import numpy as np
from ..optimizers._base import Tree
import random


def sattolo_shuffle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = random.randrange(i)
        items[j], items[i] = items[i], items[j]
    return


def cauchy_distribution(loc=0, scale=1, size=1):
    x_ = np.random.standard_cauchy(size=size)
    return loc + scale*x_


def binary_string_population(pop_size, str_len):
    return np.random.randint(low=2,
                             size=(pop_size, str_len),
                             dtype=np.byte)


def float_population(pop_size,
                     left,
                     right):
    return np.array([np.random.uniform(left_i, right_i, pop_size)
                     for left_i, right_i in zip(left, right)]).T


def full_growing_method(uniset, level_max):
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
        if level_i == level_max:
            nodes.append(uniset.random_terminal_or_ephemeral())
            n_args.append(0)
        else:
            nodes.append(uniset.random_functional())
            n_i = nodes[-1].n_args
            n_args.append(n_i)
            possible_steps.append(n_i)
            previous_levels.append(level_i)
    to_return = Tree(nodes, np.array(n_args, dtype = np.int32))
    return to_return


def growing_method(uniset, level_max):

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

        if level_i == level_max:
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
    to_return = Tree(nodes, np.array(n_args, dtype = np.int32))
    return to_return


def half_and_half(pop_size, uniset, level_max):
    population = []
    first_part = int(pop_size/2)
    second_part = pop_size - first_part
    for _ in range(first_part):
        level = np.random.randint(2, level_max)
        new_tree = full_growing_method(uniset, level)
        population.append(new_tree)

    for _ in range(second_part):
        level = np.random.randint(2, level_max)
        new_tree = growing_method(uniset, level)
        population.append(new_tree)

    return np.array(population, dtype=object)

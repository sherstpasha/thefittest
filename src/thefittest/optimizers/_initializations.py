from ._base import Tree
import numpy as np


def binary_string_population(pop_size: int, str_len: int) -> np.ndarray[np.byte]:
    return np.random.randint(low=2,
                             size=(pop_size, str_len),
                             dtype=np.byte)


def float_population(pop_size: int,
                     left: np.ndarray[float],
                     right: np.ndarray[float]) -> np.ndarray[float]:
    return np.array([np.random.uniform(left_i, right_i, pop_size)
                     for left_i, right_i in zip(left, right)]).T


def full_growing_method(uniset, level_max):
    nodes = []
    levels = []
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
            nodes.append(uniset.choice_terminal())
        else:
            nodes.append(uniset.choice_functional())
            n_i = nodes[-1].n_args

            possible_steps.append(n_i)
            previous_levels.append(level_i)
    to_return = Tree(nodes, levels)
    return to_return


def growing_method(uniset, level_max):

    nodes = []
    levels = []
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
            nodes.append(uniset.choice_terminal())
        elif level_i == 0:
            nodes.append(uniset.choice_functional())
            n_i = nodes[-1].n_args
            possible_steps.append(n_i)
            previous_levels.append(level_i)
        else:
            if np.random.random() < 0.5:
                nodes.append(uniset.choice_terminal())
            else:
                nodes.append(uniset.choice_functional())
            n_i = nodes[-1].n_args

            if n_i > 0:
                possible_steps.append(n_i)
                previous_levels.append(level_i)
    to_return = Tree(nodes, levels)
    return to_return


def half_and_half(pop_size, uniset, level_max):
    population = []
    first_part = int(pop_size/2)
    second_part = pop_size - first_part
    for _ in range(first_part):
        population.append(full_growing_method(uniset, level_max))
    for _ in range(second_part):
        population.append(growing_method(uniset, level_max))
    return np.array(population, dtype=object)
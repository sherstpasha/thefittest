from __future__ import annotations

import random
from typing import Any
from typing import Union

from numba import float64
from numba import int64
from numba import int8
from numba import njit

import numpy as np
from numpy.typing import NDArray

from .random import random_sample
from .random import random_weighted_sample
from .random import sattolo_shuffle
from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import Tree
from ..base import UniversalSet


# MUTATUIONS
# genetic algorithm
@njit(int8[:](int8[:], float64))
def flip_mutation(individual: NDArray[np.byte], proba: float) -> NDArray[np.byte]:
    """
    Perform a flip mutation on an individual.

    This function takes an individual (a binary array) and a probability. It creates a copy of the individual,
    then for each bit in the individual, it flips the bit with a certain probability.

    Parameters:
        individ (NDArray[np.byte]): The individual to be mutated.
        proba (float): The probability of flipping each bit.

    Returns:
        NDArray[np.byte]: The mutated individual.
    """
    offspring = individual.copy()
    for i in range(offspring.size):
        if random.random() < proba:
            offspring[i] = 1 - offspring[i]
    return offspring


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def best_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the best-1 mutation operator for differential evolution.

    Parameters:
        current (NDArray[np.float64]): The current individual.
        best (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2 = random_sample(range_size=size, quantity=np.int64(2), replace=False)
    return best_individual + F * (population[r1] - population[r2])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the rand-1 mutation operator for differential evolution.

    Parameters:
        current_individual (NDArray[np.float64]): The current individual.
        best_individual (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
    return population[r3] + F * (population[r1] - population[r2])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_to_best1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the rand-to-best-1 mutation operator for differential evolution.

    Parameters:
        current_individual (NDArray[np.float64]): The current individual.
        best_individual (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
    return population[r1] + F * (population[r1]) + F * (population[r2] - population[r3])


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def current_to_best_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the rand-to-best-1 mutation operator for differential evolution.

    Parameters:
        current_individual (NDArray[np.float64]): The current individual.
        best_individual (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2 = random_sample(range_size=size, quantity=np.int64(2), replace=False)
    return (
        current_individual
        + F * (best_individual - current_individual)
        + F * (population[r1] - population[r2])
    )


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def best_2(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the best-2 mutation operator for differential evolution.

    Parameters:
        current_individual (NDArray[np.float64]): The current individual.
        best_individual (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2, r3, r4 = random_sample(range_size=size, quantity=np.int64(4), replace=False)
    return (
        best_individual
        + F * (population[r1] - population[r2])
        + F * (population[r3] - population[r4])
    )


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_2(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the rand-2 mutation operator for differential evolution.

    Parameters:
        current_individual (NDArray[np.float64]): The current individual.
        best_individual (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2, r3, r4, r5 = random_sample(range_size=size, quantity=np.int64(5), replace=False)
    return (
        population[r5]
        + F * (population[r1] - population[r2])
        + F * (population[r3] - population[r4])
    )


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def current_to_rand_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    This function is the current-to-rand-1 mutation operator for differential evolution.

    Parameters:
        current_individual (NDArray[np.float64]): The current individual.
        best_individual (NDArray[np.float64]): The best individual found so far.
        population (NDArray[np.float64]): The population of individuals.
        F (np.float64): The mutation factor.

    Returns:
        NDArray[np.float64]: The mutated individual.
    """
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
    return (
        population[r1]
        + F * (population[r3] - current_individual)
        + F * (population[r1] - population[r2])
    )


@njit(float64[:](float64[:], float64[:, :], int64[:], float64, float64[:, :]))
def current_to_pbest_1_archive(
    current: NDArray[np.float64],
    population: NDArray[np.float64],
    pbest: NDArray[np.int64],
    F: np.float64,
    pop_archive: NDArray[np.float64],
) -> NDArray[np.float64]:
    p_best_ind = random.randrange(len(pbest))
    best = population[pbest[p_best_ind]]
    r1 = random_sample(range_size=len(population), quantity=np.int64(1), replace=True)[0]
    r2 = random_sample(range_size=len(pop_archive), quantity=np.int64(1), replace=True)[0]
    return current + F * (best - current) + F * (population[r1] - pop_archive[r2])


@njit(float64[:](float64[:], float64[:, :], int64[:], float64, float64[:, :]))
def current_to_pbest_1_archive_p_min(
    current: NDArray[np.float64],
    population: NDArray[np.float64],
    pbest: NDArray[np.int64],
    F: np.float64,
    pop_archive: NDArray[np.float64],
) -> NDArray[np.float64]:
    size = len(population)
    p_min = 2 / size
    p_i = np.random.uniform(p_min, 0.2)
    value = np.int64(p_i * size)
    pbest_cut = pbest[:value]

    p_best_ind = random.randrange(len(pbest_cut))
    best = population[pbest_cut[p_best_ind]]
    r1 = random_sample(range_size=size, quantity=np.int64(1), replace=True)[0]
    r2 = random_sample(range_size=len(pop_archive), quantity=np.int64(1), replace=True)[0]
    return current + F * (best - current) + F * (population[r1] - pop_archive[r2])


# genetic propramming
def point_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_level: int) -> Tree:
    new_node: Union[FunctionalNode, TerminalNode, EphemeralNode]

    to_return = tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        if isinstance(to_return._nodes[i], FunctionalNode):
            n_args = to_return._nodes[i]._n_args
            new_node = uniset._random_functional(n_args)
        else:
            new_node = uniset._random_terminal_or_ephemeral()
        to_return._nodes[i] = new_node
    return to_return


def growing_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_level: int) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        grown_tree = Tree.growing_method(uniset, max(to_return.get_levels(i)))
        to_return = to_return.concat(i, grown_tree)
    return to_return


def swap_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_leve: int) -> Tree:
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
                subtree = tree.subtree(old_j)
                to_return = to_return.concat(new_j, subtree)
    return to_return


def shrink_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_level: int) -> Tree:
    to_return = tree.copy()
    if len(to_return) > 2:
        if random.random() < proba:
            no_terminal_cond = to_return._n_args > 0
            indexes = np.arange(len(tree), dtype=int)[no_terminal_cond]
            if len(indexes) > 0:
                i = random.choice(indexes)
                args_id = to_return.get_args_id(i)
                choosen_id = random.choice(args_id)
                to_return = to_return.concat(i, tree.subtree(choosen_id))
    return to_return


# CROSSOVERS
# genetic algorithm
def empty_crossover(
    individs: np.ndarray, fitness: np.ndarray, rank: np.ndarray, *args: Any
) -> np.ndarray:
    offspring = individs[0].copy()
    return offspring


@njit(int8[:](int8[:], int8[:], float64))
def binomialGA(
    individ: NDArray[np.int8], mutant: NDArray[np.int8], CR: np.float64
) -> NDArray[np.byte]:
    size = len(individ)
    offspring = individ.copy()
    j = random.randrange(size)

    for i in range(size):
        if np.random.rand() <= CR or i == j:
            offspring[i] = mutant[i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def one_point_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    cross_point = random_sample(range_size=len(individs[0]), quantity=1, replace=True)[0]
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
def two_point_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    size = len(individs[0])
    c_points = random_sample(range_size=len(individs[0]), quantity=2, replace=False)
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
def uniform_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    choosen = random_sample(range_size=len(fitness), quantity=len(individs[0]), replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_prop_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    choosen = random_weighted_sample(weights=fitness, quantity=len(individs[0]), replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_rank_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    choosen = random_weighted_sample(weights=rank, quantity=len(individs[0]), replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


def uniform_tour_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    tournament = np.random.choice(range_, 2 * len(individs[0]))
    tournament = tournament.reshape(-1, 2)
    choosen = np.argmax(fitness[tournament], axis=1)
    offspring = individs[choosen, diag].copy()
    return offspring


# differential evolution
@njit(float64[:](float64[:], float64[:], float64))
def binomial(
    individ: NDArray[np.float64], mutant: NDArray[np.float64], CR: np.float64
) -> NDArray[np.float64]:
    size = len(individ)
    offspring = individ.copy()
    j = random.randrange(size)

    for i in range(size):
        if np.random.rand() <= CR or i == j:
            offspring[i] = mutant[i]
    return offspring


# genetic propramming
def standart_crossover(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
    first_point = random.randrange(len(individ_1))
    second_point = random.randrange(len(individ_2))

    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point)
        offspring = individ_2.concat(second_point, first_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_2
    else:
        second_subtree = individ_2.subtree(second_point)
        offspring = individ_1.concat(first_point, second_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_1
    return offspring


def one_point_crossoverGP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = individ_1.get_common_region([individ_2])

    point = random.randrange(len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point)
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point)
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring


def uniform_crossoverGP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """Poli, Riccardo & Langdon, W.. (2001). On the Search
    Properties of Different Crossover Operators in Genetic Programming."""
    to_return = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = random_sample(range_size=len(fitness), quantity=len(common[0]), replace=True)
    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return to_return


def uniform_prop_crossover_GP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = random_weighted_sample(weights=fitness, quantity=len(common[0]), replace=True)
    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return to_return


def uniform_rank_crossover_GP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = random_weighted_sample(weights=rank, quantity=len(common[0]), replace=True)

    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return to_return


def uniform_tour_crossover_GP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = tournament_selection(fitness, rank, 2, len(common[0]))

    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return to_return


# SELECTIONS
# genetic algorithm
@njit(int64[:](float64[:], float64[:], int64, int64))
def proportional_selection(
    fitness: np.ndarray, rank: np.ndarray, tour_size: int, quantity: int
) -> np.ndarray:
    choosen = random_weighted_sample(weights=fitness, quantity=quantity, replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def rank_selection(
    fitness: np.ndarray, rank: np.ndarray, tour_size: int, quantity: int
) -> np.ndarray:
    choosen = random_weighted_sample(weights=rank, quantity=quantity, replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def tournament_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: np.int64, quantity: np.int64
) -> NDArray[np.int64]:
    to_return = np.empty(quantity, dtype=np.int64)
    for i in range(quantity):
        tournament = random_sample(range_size=len(fitness), quantity=tour_size, replace=False)
        argmax = np.argmax(fitness[tournament])
        to_return[i] = tournament[argmax]
    return to_return

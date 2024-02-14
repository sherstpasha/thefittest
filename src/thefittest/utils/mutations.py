from __future__ import annotations

from typing import Union

from numba import float64
from numba import int64
from numba import int8
from numba import njit

import numpy as np
from numpy.typing import NDArray

from .random import random_sample
from .random import sattolo_shuffle
from .random import flip_coin
from .random import randint
from .random import uniform
from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import Tree
from ..base import UniversalSet


@njit(int8[:](int8[:], float64))
def flip_mutation(individual: NDArray[np.byte], proba: float) -> NDArray[np.byte]:
    """
    Perform flip mutation on an individual's binary representation for a genetic algorithm.

    Parameters
    ----------
    individual : NDArray[np.byte]
        A 1D array representing the binary values of an individual.
    proba : float
        The probability of flipping each binary value in the individual.

    Returns
    -------
    NDArray[np.byte]
        A 1D array representing the mutated individual after applying flip mutation.

    Notes
    -----
    Flip mutation randomly flips binary values in the individual with the given probability.

    Examples
    --------
    >>> from thefittest.utils.mutations import flip_mutation
    >>> import numpy as np
    >>>
    >>> # Example
    >>> original_individual = np.array([0, 1, 1, 0, 1], dtype=np.byte)
    >>> mutation_probability = 0.1
    >>> mutated_individual = flip_mutation(original_individual, mutation_probability)
    >>> print("Original Individual:", original_individual)
    Original Individual: [0 1 1 0 1]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    offspring = individual.copy()
    for i in range(offspring.size):
        if flip_coin(proba):
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
    Perform Best-1 mutation on an individual in a population.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation. (Not used in this mutation)
    best_individual : NDArray[np.float64]
        The best individual in the population.
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the Best-1 mutation strategy.

    Notes
    -----
    Best-1 mutation combines the best individual in the population with the difference between two randomly chosen
    individuals, scaled by the mutation factor F.

    Examples
    --------
    >>> from thefittest.utils.mutations import best_1
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = best_1(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2 = random_sample(range_size=size, quantity=np.int64(2), replace=False)
    mutated_individual = best_individual + F * (population[r1] - population[r2])
    return mutated_individual


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    Perform Best-1 mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation. (Not used in this mutation)
    best_individual : NDArray[np.float64]
        The best individual in the population. (Not used in this mutation)
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the Rand-1 mutation strategy.

    Notes
    -----
    Rand-1 mutation combines three randomly chosen individuals from the population, and the mutation is applied by
    adding the scaled difference between two of them to the third.

    Examples
    --------
    >>> from thefittest.utils.mutations import rand_1
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = rand_1(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
    mutated_individual = population[r3] + F * (population[r1] - population[r2])
    return mutated_individual


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_to_best1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    Perform rand-to-best-1 mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation. (Not used in this mutation)
    best_individual : NDArray[np.float64]
        The best individual in the population.
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the rand-to-best-1 mutation strategy.

    Notes
    -----
    Rand-to-best-1 mutation combines one randomly chosen individual with a scaled difference between the best individual
    and two other randomly chosen individuals from the population.

    Examples
    --------
    >>> from thefittest.utils.mutations import rand_to_best1
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = rand_to_best1(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
    mutated_individual = (
        population[r1]
        + F * (population[r1] - best_individual)
        + F * (population[r2] - population[r3])
    )
    return mutated_individual


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def current_to_best_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    Perform current-to-best-1 mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation.
    best_individual : NDArray[np.float64]
        The best individual in the population.
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the current-to-best-1 mutation strategy.

    Notes
    -----
    Current-to-best-1 mutation combines the current individual with a scaled difference between the best individual and
    the difference of two other randomly chosen individuals from the population.

    Examples
    --------
    >>> from thefittest.utils.mutations import current_to_best_1
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = current_to_best_1(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2 = random_sample(range_size=size, quantity=np.int64(2), replace=False)
    mutated_individual = (
        current_individual
        + F * (best_individual - current_individual)
        + F * (population[r1] - population[r2])
    )
    return mutated_individual


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def best_2(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    Perform Best-2 mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation. (Not used in this mutation)
    best_individual : NDArray[np.float64]
        The best individual in the population.
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the Best-2 mutation strategy.

    Notes
    -----
    Best-2 mutation combines the best individual in the population with the difference between two pairs of randomly
    chosen individuals, each pair contributing to the mutation with a scaled difference, scaled by the mutation factor F.

    Examples
    --------
    >>> from thefittest.utils.mutations import best_2
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4], [0.9, 0.5, 0.2]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = best_2(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2, r3, r4 = random_sample(range_size=size, quantity=np.int64(4), replace=False)
    mutated_individual = (
        best_individual
        + F * (population[r1] - population[r2])
        + F * (population[r3] - population[r4])
    )
    return mutated_individual


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def rand_2(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    Perform Rand-2 mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation. (Not used in this mutation)
    best_individual : NDArray[np.float64]
        The best individual in the population. (Not used in this mutation)
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the Rand-2 mutation strategy.

    Notes
    -----
    Rand-2 mutation combines one randomly chosen individual with the difference between two pairs of randomly
    chosen individuals, each pair contributing to the mutation with a scaled difference, scaled by the mutation factor F.

    Examples
    --------
    >>> from thefittest.utils.mutations import rand_2
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4], [0.9, 0.5, 0.2], [0.4, 0.2, 0.6]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = rand_2(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2, r3, r4, r5 = random_sample(range_size=size, quantity=np.int64(5), replace=False)
    mutated_individual = (
        population[r5]
        + F * (population[r1] - population[r2])
        + F * (population[r3] - population[r4])
    )
    return mutated_individual


@njit(float64[:](float64[:], float64[:], float64[:, :], float64))
def current_to_rand_1(
    current_individual: NDArray[np.float64],
    best_individual: NDArray[np.float64],
    population: NDArray[np.float64],
    F: np.float64,
) -> NDArray[np.float64]:
    """
    Perform current-to-rand-1 mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current_individual : NDArray[np.float64]
        The current individual undergoing mutation.
    best_individual : NDArray[np.float64]
        The best individual in the population. (Not used in this mutation)
    population : NDArray[np.float64]
        The entire population of individuals.
    F : float
        The mutation scale factor.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the current-to-rand-1 mutation strategy.

    Notes
    -----
    Current-to-rand-1 mutation combines the current individual with a scaled difference between one randomly chosen
    individual and the difference of two other randomly chosen individuals from the population.

    Examples
    --------
    >>> from thefittest.utils.mutations import current_to_rand_1
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> best_individual = np.array([0.6, 0.4, 0.7], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4]], dtype=np.float64)
    >>> mutation_scale_factor = 0.7
    >>> mutated_individual = current_to_rand_1(current_individual, best_individual, population, mutation_scale_factor)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = np.int64(len(population))
    r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
    mutated_individual = (
        population[r1]
        + F * (population[r3] - current_individual)
        + F * (population[r1] - population[r2])
    )
    return mutated_individual


@njit(float64[:](float64[:], float64[:, :], int64[:], float64, float64[:, :]))
def current_to_pbest_1_archive(
    current: NDArray[np.float64],
    population: NDArray[np.float64],
    pbest: NDArray[np.int64],
    F: np.float64,
    pop_archive: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Perform current-to-pbest-1-archive mutation on an individual in a population for differential evolution.

    Parameters
    ----------
    current : NDArray[np.float64]
        The current individual undergoing mutation.
    population : NDArray[np.float64]
        The entire population of individuals.
    pbest : NDArray[np.int64]
        Indices of the best individuals in the population.
    F : float
        The mutation scale factor.
    pop_archive : NDArray[np.float64]
        The archive of individuals from previous generations.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the current-to-pbest-1-archive mutation strategy.

    Notes
    -----
    Current-to-pbest-1-archive mutation combines the current individual with a scaled difference between one of the best
    individuals and the difference of two other randomly chosen individuals, one from the current population and one
    from the population archive.

    Examples
    --------
    >>> from thefittest.utils.mutations import current_to_pbest_1_archive
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4], [0.9, 0.5, 0.2]], dtype=np.float64)
    >>> pbest_indices = np.array([1, 3], dtype=np.int64)
    >>> mutation_scale_factor = 0.7
    >>> pop_archive = np.array([[0.1, 0.3, 0.6], [0.7, 0.4, 0.8], [0.4, 0.2, 0.5]], dtype=np.float64)
    >>> mutated_individual = current_to_pbest_1_archive(current_individual, population, pbest_indices, mutation_scale_factor, pop_archive)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    p_best_ind = randint(0, len(pbest), 1)[0]
    best = population[pbest[p_best_ind]]
    r1 = random_sample(range_size=len(population), quantity=np.int64(1), replace=True)[0]
    r2 = random_sample(range_size=len(pop_archive), quantity=np.int64(1), replace=True)[0]

    mutated_individual = current + F * (best - current) + F * (population[r1] - pop_archive[r2])
    return mutated_individual


@njit(float64[:](float64[:], float64[:, :], int64[:], float64, float64[:, :]))
def current_to_pbest_1_archive_p_min(
    current: NDArray[np.float64],
    population: NDArray[np.float64],
    pbest: NDArray[np.int64],
    F: np.float64,
    pop_archive: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Perform current-to-pbest-1-archive mutation with a dynamically adjusted p_min on an individual in a population for differential evolution.

    Parameters
    ----------
    current : NDArray[np.float64]
        The current individual undergoing mutation.
    population : NDArray[np.float64]
        The entire population of individuals.
    pbest : NDArray[np.int64]
        Indices of the best individuals in the population.
    F : float
        The mutation scale factor.
    pop_archive : NDArray[np.float64]
        The archive of individuals from previous generations.

    Returns
    -------
    NDArray[np.float64]
        A mutated individual based on the current-to-pbest-1-archive mutation strategy with dynamically adjusted p_min.

    Notes
    -----
    Current-to-pbest-1-archive mutation with dynamically adjusted p_min combines the current individual with a scaled
    difference between one of the best individuals and the difference of two other randomly chosen individuals, one from
    the current population and one from the population archive. The parameter p_min is dynamically adjusted based on
    the size of the population.

    Examples
    --------
    >>> from thefittest.utils.mutations import current_to_pbest_1_archive_p_min
    >>> import numpy as np
    >>>
    >>> # Example
    >>> current_individual = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    >>> population = np.array([[0.2, 0.1, 0.5], [0.8, 0.6, 0.9], [0.3, 0.7, 0.4], [0.9, 0.5, 0.2]], dtype=np.float64)
    >>> pbest_indices = np.array([1, 3], dtype=np.int64)
    >>> mutation_scale_factor = 0.7
    >>> pop_archive = np.array([[0.1, 0.3, 0.6], [0.7, 0.4, 0.8], [0.4, 0.2, 0.5]], dtype=np.float64)
    >>> mutated_individual = current_to_pbest_1_archive_p_min(current_individual, population, pbest_indices, mutation_scale_factor, pop_archive)
    >>> print("Current Individual:", current_individual)
    Current Individual: [0.5 0.3 0.8]
    >>> print("Mutated Individual:", mutated_individual)
    Mutated Individual: ...
    """
    size = len(population)
    p_min = 2 / size
    p_i = uniform(p_min, 0.2, 1)[0]
    value = np.int64(max(1, p_i * size))
    pbest_cut = pbest[:value]

    p_best_ind = randint(0, len(pbest_cut), 1)[0]
    best = population[pbest_cut[p_best_ind]]
    r1 = random_sample(range_size=size, quantity=np.int64(1), replace=True)[0]
    r2 = random_sample(range_size=len(pop_archive), quantity=np.int64(1), replace=True)[0]

    mutated_individual = current + F * (best - current) + F * (population[r1] - pop_archive[r2])
    return mutated_individual


def point_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_level: int) -> Tree:
    """
    Perform point mutation for a tree in genetic programming.

    Parameters
    ----------
    tree : Tree
        The input tree to undergo mutation.
    uniset : UniversalSet
        The universal set containing functional and terminal nodes for mutation.
    proba : float
        The probability of performing a point mutation at each node.
    max_level : int
        The maximum depth/level of the tree. (Unused in this mutation)

    Returns
    -------
    Tree
        A mutated tree after applying point mutation.

    Notes
    -----
    Point mutation randomly selects a node in the tree, and with the given probability, replaces it with a new node
    randomly chosen from the universal set.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>> from thefittest.utils.mutations import point_mutation
    >>>
    >>> # Example Parameters
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul", "neg", "inv")
    >>> max_tree_level = 4
    >>> mutation_probability = 0.5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> # Generate a Random Tree
    >>> tree = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> # Perform Point Mutation
    >>> mutated_tree = point_mutation(tree, universal_set, mutation_probability, max_tree_level)
    >>> print("Original Tree:", tree)
    Original Tree: ...
    >>> print("Mutated Tree:", mutated_tree)
    Mutated Tree: ...
    """
    new_node: Union[FunctionalNode, TerminalNode, EphemeralNode]

    mutated_tree = tree.copy()
    if flip_coin(proba):
        i = randint(0, len(mutated_tree), 1)[0]
        if isinstance(mutated_tree._nodes[i], FunctionalNode):
            n_args = mutated_tree._nodes[i]._n_args
            new_node = uniset._random_functional(n_args)
        else:
            new_node = uniset._random_terminal_or_ephemeral()
        mutated_tree._nodes[i] = new_node
    return mutated_tree


def growing_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_level: int) -> Tree:
    """
    Perform growing mutation for a tree in genetic programming.

    Parameters
    ----------
    tree : Tree
        The input tree to undergo mutation.
    uniset : UniversalSet
        The universal set containing functional and terminal nodes for mutation.
    proba : float
        The probability of performing a growing mutation at each node.
    max_level : int
        The maximum depth/level of the tree. (Unused in this mutation)

    Returns
    -------
    Tree
        A mutated tree after applying growing mutation.

    Notes
    -----
    Growing mutation randomly selects a node in the tree, and with the given probability, replaces it with a new subtree
    generated by the growing method from the universal set. The growing method creates a tree with a random structure.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>> from thefittest.utils.mutations import growing_mutation
    >>>
    >>> # Example Parameters
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul", "neg", "inv")
    >>> max_tree_level = 4
    >>> mutation_probability = 0.5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> # Generate a Random Tree
    >>> tree = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> # Perform Growing Mutation
    >>> mutated_tree = growing_mutation(tree, universal_set, mutation_probability, max_tree_level)
    >>> print("Original Tree:", tree)
    Original Tree: ...
    >>> print("Mutated Tree:", mutated_tree)
    Mutated Tree: ...
    """
    mutated_tree = tree.copy()
    if flip_coin(proba):
        i = randint(0, len(mutated_tree), 1)[0]
        grown_tree = Tree.growing_method(uniset, max(mutated_tree.get_levels(i)))
        mutated_tree = mutated_tree.concat(i, grown_tree)
    return mutated_tree


def swap_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_leve: int) -> Tree:
    """
    Perform swap mutation for a tree in genetic programming.

    Parameters
    ----------
    tree : Tree
        The input tree to undergo mutation.
    uniset : UniversalSet
        The universal set containing functional and terminal nodes for mutation. (Unused in this mutation)
    proba : float
        The probability of performing a swap mutation at each node.
    max_level : int
        The maximum depth/level of the tree. (Unused in this mutation)

    Returns
    -------
    Tree
        A mutated tree after applying swap mutation.

    Notes
    -----
    Swap mutation randomly selects a node in the tree, and with the given probability, swaps the positions of its
    subtrees, effectively changing the structure of the tree.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>> from thefittest.utils.mutations import swap_mutation
    >>>
    >>> # Example Parameters
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul", "neg", "inv")
    >>> max_tree_level = 4
    >>> mutation_probability = 0.5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> # Generate a Random Tree
    >>> tree = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> # Perform Swap Mutation
    >>> mutated_tree = swap_mutation(tree, universal_set, mutation_probability, max_tree_level)
    >>> print("Original Tree:", tree)
    Original Tree: ...
    >>> print("Mutated Tree:", mutated_tree)
    Mutated Tree: ...
    """
    mutated_tree = tree.copy()
    if flip_coin(proba):
        more_one_args_cond = mutated_tree._n_args > 1
        indexes = np.arange(len(tree), dtype=int)[more_one_args_cond]
        if len(indexes) > 0:
            index = random_sample(range_size=len(indexes), quantity=1, replace=True)[0]
            i = indexes[index]
            args_id = mutated_tree.get_args_id(i)
            new_arg_id = args_id.copy()
            new_arg_id = sattolo_shuffle(new_arg_id)
            for old_j, new_j in zip(args_id, new_arg_id):
                subtree = tree.subtree(old_j)
                mutated_tree = mutated_tree.concat(new_j, subtree)
    return mutated_tree


def shrink_mutation(tree: Tree, uniset: UniversalSet, proba: float, max_level: int) -> Tree:
    """
    Perform shrink mutation for a tree in genetic programming.

    Parameters
    ----------
    tree : Tree
        The input tree to undergo mutation.
    uniset : UniversalSet
        The universal set containing functional and terminal nodes for mutation. (Unused in this mutation)
    proba : float
        The probability of performing a shrink mutation at each node.
    max_level : int
        The maximum depth/level of the tree. (Unused in this mutation)

    Returns
    -------
    Tree
        A mutated tree after applying shrink mutation.

    Notes
    -----
    Shrink mutation randomly selects a node in the tree, and with the given probability, replaces it with one of its
    terminal nodes, effectively pruning the subtree rooted at the selected node.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>> from thefittest.utils.mutations import shrink_mutation
    >>>
    >>> # Example Parameters
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul", "neg", "inv")
    >>> max_tree_level = 4
    >>> mutation_probability = 0.5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> # Generate a Random Tree
    >>> tree = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> # Perform Shrink Mutation
    >>> mutated_tree = shrink_mutation(tree, universal_set, mutation_probability, max_tree_level)
    >>> print("Original Tree:", tree)
    Original Tree: ...
    >>> print("Mutated Tree:", mutated_tree)
    Mutated Tree: ...
    """
    mutated_tree = tree.copy()
    if len(mutated_tree) > 2:
        if flip_coin(proba):
            no_terminal_cond = mutated_tree._n_args > 0
            indexes = np.arange(len(tree), dtype=np.int64)[no_terminal_cond]
            if len(indexes) > 0:
                index = random_sample(range_size=len(indexes), quantity=1, replace=True)[0]
                i = indexes[index]

                args_id = mutated_tree.get_args_id(i)

                index = random_sample(range_size=len(indexes), quantity=1, replace=True)[0]
                choosen_id = args_id[index]

                mutated_tree = mutated_tree.concat(i, tree.subtree(choosen_id))
    return mutated_tree

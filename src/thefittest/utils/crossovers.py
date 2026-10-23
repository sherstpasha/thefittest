"""Crossover operators used by evolutionary algorithms.

The module contains operators that build offspring from binary, real-valued,
and tree-based individuals used by evolutionary optimizers in ``thefittest``.
"""

from __future__ import annotations

from numba import float64
from numba import int64
from numba import int8
from numba import njit

import numpy as np
from numpy.typing import NDArray

from .random import random_sample
from .random import random_weighted_sample
from .random import randint
from .random import flip_coin
from ..base import Tree

from .selections import tournament_selection


def empty_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    """
    Perform an empty crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the empty crossover operation.

    Notes
    -----
    This empty crossover function simply returns a copy of the first individual without any crossover.
    It is used as part of a genetic algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import empty_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5], dtype=np.float64)
    >>> ranks = np.array([1.0], dtype=np.float64)
    >>>
    >>> # Perform empty crossover
    >>> offspring = empty_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Empty Crossover:", offspring)
    Offspring After Empty Crossover: ...
    """
    offspring = individs[0].copy()
    return offspring


@njit(int8[:](int8[:], int8[:], float64))
def binomialGA(
    individ: NDArray[np.int8], mutant: NDArray[np.int8], CR: np.float64
) -> NDArray[np.byte]:
    """
    Perform binomial crossover operation for the Genetic Algorithm with Success History based Parameter Adaptation (SHAGA).

    Parameters
    ----------
    individ : NDArray[np.int8]
        A 1D array containing the genetic material of an individual in binary form.
    mutant : NDArray[np.int8]
        A 1D array containing the genetic material of a mutant individual in binary form.
    CR : np.float64
        Crossover probability, determines the likelihood of crossover.

    Returns
    -------
    NDArray[np.byte]
        A 1D array representing offspring resulting from the binomial crossover operation in binary form.

    Notes
    -----
    Binomial crossover is used in the Genetic Algorithm with Success History based Parameter Adaptation (SHAGA).
    It operates on binary chromosomes. For each gene in the chromosome, the gene from the mutant individual is selected with probability CR,
    and the gene from the original individual is selected with probability 1 - CR. The function returns a new chromosome representing
    potential offspring in binary form.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import binomialGA
    >>>
    >>> # Example
    >>> # Define a binary individual, a binary mutant, and the crossover probability
    >>> individ = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    >>> mutant = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    >>> CR = 0.8
    >>>
    >>> # Perform binomial crossover for SHAGA
    >>> offspring = binomialGA(individ, mutant, CR)
    >>>
    >>> # Display results
    >>> print("Original Binary Individual:", individ)
    Original Binary Individual: ...
    >>> print("Mutant Binary Individual:", mutant)
    Mutant Binary Individual: ...
    >>> print("Offspring After Binomial Crossover (SHAGA):", offspring)
    Offspring After Binomial Crossover (SHAGA): ...
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    for i in range(size):
        if flip_coin(CR) or i == j:
            offspring[i] = mutant[i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def one_point_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    """
    Perform one-point crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the one-point crossover operation.

    Notes
    -----
    One-point crossover randomly selects a crossover point along the chromosome. The genetic material beyond this
    point is swapped between two parents, creating two new chromosomes for potential offspring. The function returns
    one of the two resulting offspring randomly.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import one_point_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform one_point crossover
    >>> offspring = one_point_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After One-Point Crossover:", offspring)
    Offspring After One-Point Crossover: ...
    """
    cross_point = random_sample(range_size=len(individs[0]), quantity=1, replace=True)[0]
    if flip_coin(0.5):
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
    """
    Perform two-point crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the two-point crossover operation.

    Notes
    -----
    Two-point crossover randomly selects two crossover points along the chromosome. The genetic material between
    these two points is swapped between two parents, creating two new chromosomes for potential offspring. The function
    returns one of the two resulting offspring randomly.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import two_point_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform two-point crossover
    >>> offspring = two_point_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Two-Point Crossover:", offspring)
    Offspring After Two-Point Crossover: ...
    """
    size = len(individs[0])
    c_points = random_sample(range_size=len(individs[0]), quantity=2, replace=False)
    c_points = sorted(c_points)

    if flip_coin(0.5):
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
    """
    Perform uniform crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the uniform crossover operation.

    Notes
    -----
    Uniform crossover randomly selects genetic material from the parents for each gene in the chromosome.
    Each gene of the offspring is randomly chosen from one of the parents. The function returns a new chromosome
    representing potential offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform uniform crossover
    >>> offspring = uniform_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Uniform Crossover:", offspring)
    Offspring After Uniform Crossover: ...
    """
    choosen = random_sample(range_size=len(fitness), quantity=len(individs[0]), replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_proportional_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    """
    Perform uniform proportional crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the uniform proportional crossover operation.

    Notes
    -----
    Uniform proportional crossover randomly selects genetic material from the parents for each gene in the chromosome.
    The probability of choosing genetic material from a parent is proportional to its fitness value. The function returns
    a new chromosome representing potential offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_proportional_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform uniform proportional crossover
    >>> offspring = uniform_proportional_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Uniform Proportional Crossover:", offspring)
    Offspring After Uniform Proportional Crossover: ...
    """
    choosen = random_weighted_sample(weights=fitness, quantity=len(individs[0]), replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:, :], float64[:], float64[:]))
def uniform_rank_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    """
    Perform uniform rank crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual.

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the uniform rank crossover operation.

    Notes
    -----
    Uniform rank crossover randomly selects genetic material from the parents for each gene in the chromosome.
    The probability of choosing genetic material from a parent is proportional to its rank value. The function returns
    a new chromosome representing potential offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_rank_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform uniform rank crossover
    >>> offspring = uniform_rank_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Uniform Rank Crossover:", offspring)
    Offspring After Uniform Rank Crossover: ...
    """
    choosen = random_weighted_sample(weights=rank, quantity=len(individs[0]), replace=True)
    offspring = np.empty_like(individs[0])
    for i in range(individs.shape[1]):
        offspring[i] = individs[choosen[i]][i]
    return offspring


def uniform_tournament_crossover(
    individs: NDArray[np.byte], fitness: NDArray[np.float64], rank: NDArray[np.float64]
) -> NDArray[np.byte]:
    """
    Perform uniform tournament crossover operation for a genetic algorithm.

    Parameters
    ----------
    individs : NDArray[np.byte]
        A 2D array containing individuals where each row represents an individual.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)

    Returns
    -------
    NDArray[np.byte]
        Offspring resulting from the uniform tournament crossover operation.

    Notes
    -----
    Uniform tournament crossover randomly selects genetic material from the parents for each gene in the chromosome.
    For every gene, two candidate parents are sampled from ``individs`` and the candidate with the higher fitness is
    copied to the offspring. The function returns a new chromosome representing potential offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_tournament_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform uniform tournament crossover
    >>> offspring = uniform_tournament_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Uniform Tournament Crossover:", offspring)
    Offspring After Uniform Tournament Crossover: ...
    """
    diag = np.arange(len(individs[0]))

    tournament = random_sample(len(individs), 2 * len(individs[0]), True)
    tournament = tournament.reshape(-1, 2)
    winners = np.argmax(fitness[tournament], axis=1)
    choosen = tournament[diag, winners]
    offspring = individs[choosen, diag].copy()
    return offspring


@njit(float64[:](float64[:], float64[:], float64))
def binomial(
    individ: NDArray[np.float64], mutant: NDArray[np.float64], CR: np.float64
) -> NDArray[np.float64]:
    """
    Perform binomial crossover operation for differential evolution.

    Parameters
    ----------
    individ : NDArray[np.float64]
        A 1D array containing the genetic material of an individual.
    mutant : NDArray[np.float64]
        A 1D array containing the genetic material of a mutant individual.
    CR : np.float64
        Crossover probability, determines the likelihood of crossover.

    Returns
    -------
    NDArray[np.float64]
        A 1D array representing offspring resulting from the binomial crossover operation.

    Notes
    -----
    Binomial crossover is used in differential evolution. It operates on real-valued chromosomes.
    For each gene in the chromosome, the gene from the mutant individual is selected with probability CR,
    and the gene from the original individual is selected with probability 1 - CR.
    The function returns a new chromosome representing potential offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import binomial
    >>>
    >>> # Example
    >>> # Define an individual, a mutant, and the crossover probability
    >>> individ = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    >>> mutant = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    >>> CR = 0.8
    >>>
    >>> # Perform binomial crossover
    >>> offspring = binomial(individ, mutant, CR)
    >>>
    >>> # Display results
    >>> print("Original Individual:", individ)
    Original Individual: ...
    >>> print("Mutant Individual:", mutant)
    Mutant Individual: ...
    >>> print("Offspring After Binomial Crossover:", offspring)
    Offspring After Binomial Crossover: ...
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    for i in range(size):
        if flip_coin(CR) or i == j:
            offspring[i] = mutant[i]
    return offspring


def empty_crossoverGP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    Return a copy of the first genetic programming parent.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing at least one tree individual.
    fitness : NDArray[np.float64]
        Fitness values of individuals. Not used.
    rank : NDArray[np.float64]
        Rank values of individuals. Not used.
    max_level : int
        Maximum allowed tree level. Not used.

    Returns
    -------
    Tree
        Copy of the first tree in ``individs``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import empty_crossoverGP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>>
    >>> # Example
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> # Define the parents, fitness values, ranks, and maximum allowed depth
    >>> parents = np.array([Tree.random_tree(universal_set, max_tree_level)], dtype=object)
    >>> fitness_values = np.array([0.8], dtype=np.float64)
    >>> ranks = np.array([1.0], dtype=np.float64)
    >>> max_depth = 7
    >>>
    >>> # Perform empty crossover for genetic programming
    >>> offspring = empty_crossoverGP(parents, fitness_values, ranks, max_depth)
    >>>
    >>> print("Original Individual:", parents[0])
    Original Individual: ...
    >>> print("Offspring After Empty Crossover (GP):", offspring)
    Offspring After Empty Crossover (GP): ...
    """
    offspring = individs[0].copy()
    return offspring


def standard_crossover(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    Perform standard crossover operation for genetic programming.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing two individuals represented as trees.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)
    max_level : int
        Maximum allowed depth/level of the resulting offspring.

    Returns
    -------
    Tree
        Offspring resulting from the standard crossover operation.

    Notes
    -----
    Standard crossover is used in genetic programming. It involves selecting a random subtree from one parent
    and replacing a subtree in the other parent. The resulting offspring represents a combination of genetic material
    from both parents. If the depth/level of the offspring exceeds the specified maximum level, a random parent is chosen
    as the offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import standard_crossover
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>> # Example
    >>>
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> parent1 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent2 = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>> max_depth = 7  # Set the maximum allowed depth
    >>>
    >>> # Perform standard crossover
    >>> offspring = standard_crossover(np.array([parent1, parent2]), fitness_values, ranks, max_depth)
    >>>
    >>> # Display results
    >>> print("Parent 1:", parent1)
    Parent 1: ...
    >>> print("Parent 2:", parent2)
    Parent 2: ...
    >>> print("Offspring After Standard Crossover:", offspring)
    Offspring After Standard Crossover: ...
    """
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
    first_point = randint(0, len(individ_1), 1)[0]
    second_point = randint(0, len(individ_2), 1)[0]

    if flip_coin(0.5):
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
    """
    Perform one-point crossover operation for genetic programming.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing two individuals represented as trees.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)
    max_level : int
        Maximum allowed depth/level of the resulting offspring. (not used)

    Returns
    -------
    Tree
        Offspring resulting from the one-point crossover operation.

    Notes
    -----
    One-point crossover is used in genetic programming. It involves selecting a common region between two parents
    and exchanging genetic material at a randomly chosen point within this region. The resulting offspring represents
    a combination of genetic material from both parents.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import one_point_crossoverGP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>> # Example
    >>>
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> parent1 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent2 = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>> max_depth = 7  # Set the maximum allowed depth
    >>>
    >>> # Perform one-point crossover
    >>> offspring = one_point_crossoverGP(np.array([parent1, parent2]), fitness_values, ranks, max_depth)
    >>>
    >>> # Display results
    >>> print("Parent 1:", parent1)
    Parent 1: ...
    >>> print("Parent 2:", parent2)
    Parent 2: ...
    >>> print("Offspring After One-Point Crossover:", offspring)
    Offspring After One-Point Crossover: ...
    """
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = individ_1.get_common_region([individ_2])

    point = randint(0, len(common_indexes[0]), 1)[0]
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if flip_coin(0.5):
        first_subtree = individ_1.subtree(first_point)
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point)
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring


def uniform_crossoverGP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    Perform uniform crossover operation for genetic programming.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing two or more individuals represented as trees.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)
    max_level : int
        Maximum allowed depth/level of the resulting offspring. (not used)

    Returns
    -------
    Tree
        Offspring resulting from the uniform crossover operation.

    Notes
    -----
    Uniform crossover is used in genetic programming. It involves selecting a common region between two or more parents
    and exchanging genetic material at randomly chosen points within this region. The resulting offspring represents
    a combination of genetic material from all parents.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_crossoverGP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>> # Example
    >>>
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> parent1 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent2 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent3 = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> fitness_values = np.array([0.5, 0.8, 0.6], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    >>> max_depth = 7  # Set the maximum allowed depth
    >>>
    >>> # Perform uniform crossover
    >>> offspring = uniform_crossoverGP(np.array([parent1, parent2, parent3]), fitness_values, ranks, max_depth)
    >>>
    >>> # Display results
    >>> print ("Parent 1:", parent1)
    Parent 1: ...
    >>> print("Parent 2:", parent2)
    Parent 2: ...
    >>> print("Parent 3:", parent3)
    Parent 3: ...
    >>> print("Offspring After Uniform Crossover:", offspring)
    Offspring After Uniform Crossover: ...
    """
    mutant = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = random_sample(range_size=len(individs), quantity=len(common[0]), replace=True)
    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            mutant._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            mutant._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    mutant = mutant.copy()
    mutant._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return mutant


def uniform_proportional_crossover_GP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    Perform uniform proportional crossover operation for genetic programming.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing two or more individuals represented as trees.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual. (not used)
    max_level : int
        Maximum allowed depth/level of the resulting offspring. (not used)

    Returns
    -------
    Tree
        Offspring resulting from the uniform proportional crossover operation.

    Notes
    -----
    Uniform proportional crossover is a genetic programming operator. It involves selecting a common region between two or more parents,
    and the genetic material is exchanged at randomly chosen points within this region. The probability of selecting genetic material
    from a parent is proportional to its fitness value. This means that individuals with higher fitness values have a higher chance
    of contributing genetic material to the offspring. The resulting tree represents a combination of genetic material from all parents.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_proportional_crossover_GP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>> # Example
    >>>
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> parent1 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent2 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent3 = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> fitness_values = np.array([0.5, 0.8, 0.6], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    >>> max_depth = 7  # Set the maximum allowed depth
    >>>
    >>> # Perform uniform proportional crossover
    >>> offspring = uniform_proportional_crossover_GP(np.array([parent1, parent2, parent3]), fitness_values, ranks, max_depth)
    >>>
    >>> # Display results
    >>> print("Parent 1:", parent1)
    Parent 1: ...
    >>> print("Parent 2:", parent2)
    Parent 2: ...
    >>> print("Parent 3:", parent3)
    Parent 3: ...
    >>> print("Offspring After Uniform Proportional Crossover:", offspring)
    Offspring After Uniform Proportional Crossover: ...
    """
    mutant = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = random_weighted_sample(weights=fitness, quantity=len(common[0]), replace=True)
    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            mutant._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            mutant._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    mutant = mutant.copy()
    mutant._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return mutant


def uniform_rank_crossover_GP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    Perform uniform rank crossover operation for genetic programming.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing two or more individuals represented as trees.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals. (not used)
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual.
        The rank values are used to determine the probability of selecting genetic material from each parent.
    max_level : int
        Maximum allowed depth/level of the resulting offspring. (not used)

    Returns
    -------
    Tree
        Offspring resulting from the uniform rank crossover operation.

    Notes
    -----
    Uniform rank crossover is a genetic programming operator. It involves selecting a common region between two or more parents,
    and the genetic material is exchanged at randomly chosen points within this region. The probability of selecting genetic material
    from a parent is proportional to its rank value. This means that individuals with higher rank values have a higher chance
    of contributing genetic material to the offspring. The resulting tree represents a combination of genetic material from all parents.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_rank_crossover_GP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>> # Example
    >>>
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> parent1 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent2 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent3 = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> fitness_values = np.array([0.5, 0.8, 0.6], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    >>> max_depth = 7  # Set the maximum allowed depth
    >>>
    >>> # Perform uniform rank crossover
    >>> offspring = uniform_rank_crossover_GP(np.array([parent1, parent2, parent3]), fitness_values, ranks, max_depth)
    >>>
    >>> # Display results
    >>> print("Parent 1:", parent1)
    Parent 1: ...
    >>> print("Parent 2:", parent2)
    Parent 2: ...
    >>> print("Parent 3:", parent3)
    Parent 3: ...
    >>> print("Offspring After Uniform Rank Crossover:", offspring)
    Offspring After Uniform Rank Crossover: ...
    """
    mutant = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = random_weighted_sample(weights=rank, quantity=len(common[0]), replace=True)

    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            mutant._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            mutant._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    mutant = mutant.copy()
    mutant._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return mutant


def uniform_tournament_crossover_GP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    Perform uniform tournament crossover operation for genetic programming.

    Parameters
    ----------
    individs : NDArray
        A 1D array containing two or more individuals represented as trees.
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual.
        The rank values are used to determine the probability of selecting genetic material from each parent.
    max_level : int
        Maximum allowed depth/level of the resulting offspring. (not used)

    Returns
    -------
    Tree
        Offspring resulting from the uniform tournament crossover operation.

    Notes
    -----
    Uniform tournament crossover is a genetic programming operator. It involves selecting a common region between two or more parents,
    and the genetic material is exchanged at randomly chosen points within this region. The probability of selecting genetic material
    from a parent is determined through a tournament selection process, where individuals with higher fitness values have a higher chance
    of contributing genetic material to the offspring. The resulting tree represents a combination of genetic material from all parents.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_tournament_crossover_GP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>> # Example
    >>>
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul")
    >>> max_tree_level = 5
    >>>
    >>> # Initialize Universal Set for Symbolic Regression
    >>> universal_set = init_symbolic_regression_uniset(X, functional_set_names)
    >>>
    >>> parent1 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent2 = Tree.random_tree(universal_set, max_tree_level)
    >>> parent3 = Tree.random_tree(universal_set, max_tree_level)
    >>>
    >>> fitness_values = np.array([0.5, 0.8, 0.6], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    >>> max_depth = 7  # Set the maximum allowed depth
    >>>
    >>> # Perform uniform tournament crossover
    >>> offspring = uniform_tournament_crossover_GP(np.array([parent1, parent2, parent3]), fitness_values, ranks, max_depth)
    >>>
    >>> # Display results
    >>> print("Parent 1:", parent1)
    Parent 1: ...
    >>> print("Parent 2:", parent2)
    Parent 2: ...
    >>> print("Parent 3:", parent3)
    Parent 3: ...
    >>> print("Offspring After Uniform Tournament Crossover:", offspring)
    Offspring After Uniform Tournament Crossover: ...
    """
    mutant = Tree([], [])
    new_n_args = []
    common, border = individs[0].get_common_region(individs[1:])
    pool = tournament_selection(fitness, rank, 2, len(common[0]))

    for i, common_0_i in enumerate(common[0]):
        j = pool[i]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_)
            mutant._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            mutant._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    mutant = mutant.copy()
    mutant._n_args = np.array(new_n_args.copy(), dtype=np.int64)
    return mutant


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def empty_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Return the target chromosome unchanged for configurable SHAGA.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. Not used.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Crossover probability. Not used.

    Returns
    -------
    NDArray[np.byte]
        Copy of ``individ``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import empty_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = empty_crossover_selfcshaga(individ, parents, fitness, rank, 0.7)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    offspring = individ.copy()
    return offspring


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def one_point_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply one-point configurable SHAGA crossover with a uniformly selected parent.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is sampled
        uniformly from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by one-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, a crossover point and one selected
    parent are sampled. The suffix of one chromosome is exchanged with the
    suffix of the other chromosome.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import one_point_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = one_point_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        cross_point = random_sample(range_size=len(individs[0]), quantity=1, replace=True)[0]
        choosen = random_sample(range_size=len(fitness), quantity=1, replace=True)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individs[choosen][i]
        else:
            offspring = individs[choosen].copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def one_point_prop_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply one-point configurable SHAGA crossover with fitness-proportional parent choice.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is sampled
        from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. They define the proportional
        sampling weights.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by one-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, a crossover point is sampled and the
    second chromosome is chosen with probability proportional to ``fitness``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import one_point_prop_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = one_point_prop_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        cross_point = random_sample(range_size=len(individs[0]), quantity=1, replace=True)[0]
        choosen = random_weighted_sample(weights=fitness, quantity=1, replace=True)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individs[choosen][i]
        else:
            offspring = individs[choosen].copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def one_point_rank_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply one-point configurable SHAGA crossover with rank-proportional parent choice.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is sampled
        from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. They define the proportional sampling
        weights.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by one-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, a crossover point is sampled and the
    second chromosome is chosen with probability proportional to ``rank``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import one_point_rank_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = one_point_rank_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        cross_point = random_sample(range_size=len(individs[0]), quantity=1, replace=True)[0]
        choosen = random_weighted_sample(weights=rank, quantity=1, replace=True)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individs[choosen][i]
        else:
            offspring = individs[choosen].copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def one_point_tour_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply one-point configurable SHAGA crossover with tournament parent choice.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is chosen
        from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Tournament winners are selected by
        higher fitness.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used by tournament selection.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by one-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, a crossover point is sampled and the
    second chromosome is chosen by a binary tournament over ``fitness``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import one_point_tour_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = one_point_tour_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        cross_point = random_sample(range_size=len(individs[0]), quantity=1, replace=True)[0]
        choosen = tournament_selection(fitness, rank, 2, 1)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individs[choosen][i]
        else:
            offspring = individs[choosen].copy()
            for i in range(len(individ)):
                if i > cross_point:
                    offspring[i] = individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def two_point_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    if flip_coin(CR):
        size = len(individ)
        c_points = random_sample(range_size=size, quantity=2, replace=False)
        left = c_points[0]
        right = c_points[1]
        if left > right:
            left, right = right, left

        choosen = random_sample(range_size=len(fitness), quantity=1, replace=True)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            other_individ = individs[choosen]
        else:
            offspring = individs[choosen].copy()
            other_individ = individ

        for i in range(size):
            if left <= i <= right:
                offspring[i] = other_individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def two_point_rank_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply two-point configurable SHAGA crossover with rank-proportional parent choice.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is sampled
        from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. They define the proportional sampling
        weights.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by two-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, two crossover points are sampled and
    the segment between them is exchanged with a parent sampled by ``rank``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import two_point_rank_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = two_point_rank_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        size = len(individs[0])
        c_points = random_sample(range_size=len(individs[0]), quantity=2, replace=False)
        c_points = sorted(c_points)
        choosen = random_weighted_sample(weights=rank, quantity=1, replace=True)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            other_individ = individs[choosen]
        else:
            offspring = individs[choosen].copy()
            other_individ = individ
        for i in range(size):
            if c_points[0] <= i <= c_points[1]:
                offspring[i] = other_individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def two_point_prop_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply two-point configurable SHAGA crossover with fitness-proportional parent choice.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is sampled
        from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. They define the proportional
        sampling weights.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by two-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, two crossover points are sampled and
    the segment between them is exchanged with a parent sampled by ``fitness``.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import two_point_prop_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = two_point_prop_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        size = len(individs[0])
        c_points = random_sample(range_size=len(individs[0]), quantity=2, replace=False)
        c_points = sorted(c_points)
        choosen = random_weighted_sample(weights=fitness, quantity=1, replace=True)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            other_individ = individs[choosen]
        else:
            offspring = individs[choosen].copy()
            other_individ = individ
        for i in range(size):
            if c_points[0] <= i <= c_points[1]:
                offspring[i] = other_individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def two_point_tour_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply two-point configurable SHAGA crossover with tournament parent choice.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The parent used for crossover is chosen
        from this array.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Tournament winners are selected by
        higher fitness.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used by tournament selection.
    CR : np.float64
        Probability of applying crossover. If crossover is not applied,
        ``individ`` is returned.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by two-point crossover, or ``individ`` if crossover
        is not applied.

    Notes
    -----
    If crossover is triggered by ``CR``, two crossover points are sampled and
    the segment between them is exchanged with a tournament-selected parent.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import two_point_tour_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = two_point_tour_crossover_selfcshaga(individ, parents, fitness, rank, 1.0)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    if flip_coin(CR):
        size = len(individs[0])
        c_points = random_sample(range_size=len(individs[0]), quantity=2, replace=False)
        c_points = sorted(c_points)
        choosen = tournament_selection(fitness, rank, 2, 1)[0]

        if flip_coin(0.5):
            offspring = individ.copy()
            other_individ = individs[choosen]
        else:
            offspring = individs[choosen].copy()
            other_individ = individ
        for i in range(size):
            if c_points[0] <= i <= c_points[1]:
                offspring[i] = other_individ[i]
        return offspring
    else:
        return individ


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def uniform_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply uniform configurable SHAGA crossover with uniformly selected parents.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. A parent is sampled uniformly for each
        gene.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Probability of replacing each target gene by the corresponding sampled
        parent gene.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by uniform crossover.

    Notes
    -----
    For each gene, a parent is sampled uniformly from ``individs``. The sampled
    gene replaces the target gene with probability ``CR``; one random gene is
    always replaced to ensure that crossover changes at least one position.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = uniform_crossover_selfcshaga(individ, parents, fitness, rank, 0.7)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    choosen = random_sample(range_size=len(fitness), quantity=len(individs[0]), replace=True)
    for i in range(individs.shape[1]):
        if np.random.rand() <= CR or i == j:
            offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def uniform_prop_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply uniform configurable SHAGA crossover with fitness-proportional parents.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. A parent is sampled for each gene.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. They define the proportional
        sampling weights.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Probability of replacing each target gene by the corresponding sampled
        parent gene.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by uniform crossover.

    Notes
    -----
    For each gene, a parent is sampled from ``individs`` with probability
    proportional to ``fitness``. The sampled gene replaces the target gene with
    probability ``CR``; one random gene is always replaced.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_prop_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = uniform_prop_crossover_selfcshaga(individ, parents, fitness, rank, 0.7)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    choosen = random_weighted_sample(weights=fitness, quantity=len(individs[0]), replace=True)

    for i in range(individs.shape[1]):
        if np.random.rand() <= CR or i == j:
            offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def uniform_rank_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply uniform configurable SHAGA crossover with rank-proportional parents.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. A parent is sampled for each gene.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. They define the proportional sampling
        weights.
    CR : np.float64
        Probability of replacing each target gene by the corresponding sampled
        parent gene.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by uniform crossover.

    Notes
    -----
    For each gene, a parent is sampled from ``individs`` with probability
    proportional to ``rank``. The sampled gene replaces the target gene with
    probability ``CR``; one random gene is always replaced.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_rank_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = uniform_rank_crossover_selfcshaga(individ, parents, fitness, rank, 0.7)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    choosen = random_weighted_sample(weights=rank, quantity=len(individs[0]), replace=True)
    for i in range(individs.shape[1]):
        if np.random.rand() <= CR or i == j:
            offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def uniform_tour_crossover_selfcshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply uniform configurable SHAGA crossover with tournament-selected parents.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. A parent is selected for each gene.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Tournament winners are selected by
        higher fitness.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used by tournament selection.
    CR : np.float64
        Probability of replacing each target gene by the corresponding selected
        parent gene.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by uniform crossover.

    Notes
    -----
    For each gene, a parent is selected by binary tournament over ``fitness``.
    The selected gene replaces the target gene with probability ``CR``; one
    random gene is always replaced.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_tour_crossover_selfcshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = uniform_tour_crossover_selfcshaga(individ, parents, fitness, rank, 0.7)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    choosen = tournament_selection(fitness, rank, 2, size)
    for i in range(size):
        if np.random.rand() <= CR or i == j:
            offspring[i] = individs[choosen[i]][i]
    return offspring


@njit(int8[:](int8[:], int8[:, :], float64[:], float64[:], float64))
def binomial_selfshaga(
    individ: NDArray[np.byte],
    individs: NDArray[np.byte],
    fitness: NDArray[np.float64],
    rank: NDArray[np.float64],
    CR: np.float64,
) -> NDArray[np.byte]:
    """
    Apply binomial configurable SHAGA crossover against the first selected parent.

    Parameters
    ----------
    individ : NDArray[np.byte]
        Target chromosome from the current population.
    individs : NDArray[np.byte]
        Selected parent chromosomes. The first row, ``individs[0]``, is used as
        the donor chromosome.
    fitness : NDArray[np.float64]
        Fitness values of selected parents. Not used.
    rank : NDArray[np.float64]
        Rank values of selected parents. Not used.
    CR : np.float64
        Probability of replacing each target gene by the corresponding donor
        gene.

    Returns
    -------
    NDArray[np.byte]
        Offspring produced by binomial crossover.

    Notes
    -----
    Each target gene is replaced by the corresponding gene from ``individs[0]``
    with probability ``CR``; one random gene is always replaced.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import binomial_selfshaga
    >>>
    >>> individ = np.array([1, 0, 1, 0], dtype=np.byte)
    >>> parents = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.byte)
    >>> fitness = np.array([0.2, 0.8], dtype=np.float64)
    >>> rank = np.array([1.0, 2.0], dtype=np.float64)
    >>> offspring = binomial_selfshaga(individ, parents, fitness, rank, 0.7)
    >>> offspring.shape == individ.shape
    True
    >>> offspring.dtype == individ.dtype
    True
    """
    size = len(individ)
    offspring = individ.copy()
    j = randint(0, size, 1)[0]

    for i in range(size):
        if flip_coin(CR) or i == j:
            offspring[i] = individs[0][i]
    return offspring

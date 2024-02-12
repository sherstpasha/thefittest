from __future__ import annotations

import random
from typing import Any

from numba import float64
from numba import int64
from numba import int8
from numba import njit

import numpy as np
from numpy.typing import NDArray

from .random import random_sample
from .random import random_weighted_sample
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
    >>> parents = np.array([[1, 0, 1]], dtype=np.byte)  # First and second parent
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
    j = random.randrange(size)

    for i in range(size):
        if np.random.rand() <= CR or i == j:
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
    >>> parents = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.byte)  # First and second parent
    >>> fitness_values = np.array([0.5, 0.8], dtype=np.float64)
    >>> ranks = np.array([2.0, 1.0], dtype=np.float64)
    >>>
    >>> # Perform one_point crossover
    >>> offspring = one_point_crossover(parents, fitness_values, ranks)
    >>>
    >>> # Display results
    >>> print("Original Individuals:", parents)
    Original Individuals: ...
    >>> print("Offspring After Empty Crossover:", offspring)
    Offspring After Empty Crossover: ...
    """
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
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)  # First and second parent
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
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)  # First and second parent
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
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)  # First and second parent
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
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)  # First and second parent
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
    The parents for each gene are chosen through tournament selection based on their fitness values.
    The function returns a new chromosome representing potential offspring.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import uniform_tournament_crossover
    >>>
    >>> # Example
    >>> # Define the parents, fitness values, and ranks
    >>> parents = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.byte)  # First and second parent
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
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    tournament = random_sample(len(individs), 2 * len(individs[0]), True)
    tournament = tournament.reshape(-1, 2)
    choosen = np.argmax(fitness[tournament], axis=1)
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
    j = random.randrange(size)

    for i in range(size):
        if np.random.rand() <= CR or i == j:
            offspring[i] = mutant[i]
    return offspring


def empty_crossoverGP(
    individs: NDArray, fitness: NDArray[np.float64], rank: NDArray[np.float64], max_level: int
) -> Tree:
    """
    >>> import numpy as np
    >>> from thefittest.utils.crossovers import empty_crossoverGP
    >>> from thefittest.base import Tree
    >>> from thefittest.base import init_symbolic_regression_uniset
    >>>
    >>>
    >>> # Example
    >>> X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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
    from a parent is determined through a tournament selection process, where individuals with higher fitness or rank values have a higher chance
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
    >>> functional_set_names = ("add", "mul", "neg", "inv")
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

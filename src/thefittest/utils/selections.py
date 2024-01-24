from __future__ import annotations

from numba import float64
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from .random import random_sample
from .random import random_weighted_sample


@njit(int64[:](float64[:], float64[:], int64, int64))
def proportional_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: int, quantity: int
) -> NDArray[np.int64]:
    """
    Perform proportional selection to choose individuals based on their fitness.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals.
    tour_size : int
        The size of the tournament selection.
    quantity : int
        The number of individuals to choose.

    Returns
    -------
    NDArray[np.int64]
        A 1D array of selected individuals.

    Examples
    --------
    >>> from thefittest.utils.selections import proportional_selection
    >>> import numpy as np
    >>>
    >>> # Example
    >>> fitness_values = np.array([0.8, 0.5, 0.9, 0.3], dtype = np.float64)
    >>> rank_values = np.array([3, 2, 4, 1], dtype = np.float64)
    >>> tournament_size = 2
    >>> num_selected = 2
    >>> selected_individuals = proportional_selection(fitness_values, rank_values, tournament_size, num_selected)
    >>> print("Selected Individuals:", selected_individuals)
    Selected Individuals: ...
    """
    choosen = random_weighted_sample(weights=fitness, quantity=quantity, replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def rank_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: int, quantity: int
) -> NDArray[np.int64]:
    """
    Perform rank-based selection to choose individuals based on their rank values.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals. The higher the rank, the better the individual.
    tour_size : int
        The size of the tournament selection (not used in rank-based selection).
    quantity : int
        The number of individuals to choose.

    Returns
    -------
    NDArray[np.int64]
        A 1D array of selected individuals based on rank.

    Notes
    -----
    Rank-based selection assigns a probability of selection to each individual based on their rank.
    Individuals with higher ranks have higher probabilities of being selected.

    Examples
    --------
    >>> from thefittest.utils.selections import rank_selection
    >>> import numpy as np
    >>>
    >>> # Example
    >>> fitness_values = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    >>> rank_values = np.array([3, 2, 4, 1], dtype=np.float64)
    >>> tournament_size = 2
    >>> num_selected = 2
    >>> selected_individuals = rank_selection(fitness_values, rank_values, tournament_size, num_selected)
    >>> print("Selected Individuals:", selected_individuals)
    Selected Individuals: ...
    """
    choosen = random_weighted_sample(weights=rank, quantity=quantity, replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def tournament_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: int, quantity: int
) -> NDArray[np.int64]:
    """
    Perform tournament selection to choose individuals based on their fitness values.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        A 1D array containing the fitness values of individuals.
    rank : NDArray[np.float64]
        A 1D array containing the rank values of individuals (not used in tournament selection).
    tour_size : int
        The size of each tournament. Individuals compete in groups of this size, and the winner is selected.
    quantity : int
        The number of individuals to choose.

    Returns
    -------
    NDArray[np.int64]
        A 1D array of selected individuals based on tournament selection.

    Notes
    -----
    Tournament selection involves randomly selecting individuals in groups (tournaments) and choosing the
    individual with the highest fitness in each tournament. This process is repeated to fill the desired quantity.

    Examples
    --------
    >>> from thefittest.utils.selections import tournament_selection
    >>> import numpy as np
    >>>
    >>> # Example
    >>> fitness_values = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    >>> rank_values = np.array([3, 2, 4, 1], dtype=np.float64)
    >>> tournament_size = 2
    >>> num_selected = 2
    >>> selected_individuals = tournament_selection(fitness_values, rank_values, tournament_size, num_selected)
    >>> print("Selected Individuals:", selected_individuals)
    Selected Individuals: ...
    """
    to_return = np.empty(quantity, dtype=np.int64)
    for i in range(quantity):
        tournament = random_sample(range_size=len(fitness), quantity=tour_size, replace=False)
        argmax = np.argmax(fitness[tournament])
        to_return[i] = tournament[argmax]
    return to_return

"""Selection operators used by evolutionary algorithms.

The functions in this module choose parent indexes from a population. Scalar
fitness operators share the ``(fitness, rank, tour_size, quantity)`` signature.
Multiobjective tournaments accept either nondomination ranks or objective
vectors, plus precomputed crowding distances. They return index arrays too.
"""

from __future__ import annotations

from numba import float64
from numba import int64
from numba import njit

import numpy as np
from numpy.typing import NDArray

from .random import random_sample
from .random import random_weighted_sample
from .random import flip_coin
from .multiobjective import compare_rank_and_crowding
from .multiobjective import compare_dominance_and_crowding


@njit(int64[:](float64[:], float64[:], int64, int64))
def proportional_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: int, quantity: int
) -> NDArray[np.int64]:
    """
    Select parent indexes with probability proportional to fitness.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        Fitness values used as sampling weights.
    rank : NDArray[np.float64]
        Rank values of individuals. Not used.
    tour_size : int
        Tournament size. Not used; accepted for the shared selection
        operator interface.
    quantity : int
        Number of parent indexes to choose.

    Returns
    -------
    NDArray[np.int64]
        Indexes of selected individuals.

    Notes
    -----
    Sampling is performed with replacement. The values in ``fitness`` are
    expected to be valid non-negative weights with a positive total weight.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.selections import proportional_selection
    >>>
    >>> fitness = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    >>> rank = np.array([3.0, 2.0, 4.0, 1.0], dtype=np.float64)
    >>> selected = proportional_selection(fitness, rank, 0, 2)
    >>> selected.shape
    (2,)
    >>> selected.dtype == np.int64
    True
    >>> bool(np.all((0 <= selected) & (selected < fitness.size)))
    True
    """
    choosen = random_weighted_sample(weights=fitness, quantity=quantity, replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def rank_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: int, quantity: int
) -> NDArray[np.int64]:
    """
    Select parent indexes with probability proportional to rank.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        Fitness values of individuals. Not used.
    rank : NDArray[np.float64]
        Rank values used as sampling weights. The higher the rank, the higher
        the selection probability.
    tour_size : int
        Tournament size. Not used; accepted for the shared selection
        operator interface.
    quantity : int
        Number of parent indexes to choose.

    Returns
    -------
    NDArray[np.int64]
        Indexes of selected individuals.

    Notes
    -----
    Sampling is performed with replacement. The values in ``rank`` are expected
    to be valid non-negative weights with a positive total weight.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.selections import rank_selection
    >>>
    >>> fitness = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    >>> rank = np.array([3.0, 2.0, 4.0, 1.0], dtype=np.float64)
    >>> selected = rank_selection(fitness, rank, 0, 2)
    >>> selected.shape
    (2,)
    >>> selected.dtype == np.int64
    True
    >>> bool(np.all((0 <= selected) & (selected < rank.size)))
    True
    """
    choosen = random_weighted_sample(weights=rank, quantity=quantity, replace=True)
    return choosen


@njit(int64[:](float64[:], float64[:], int64, int64))
def tournament_selection(
    fitness: NDArray[np.float64], rank: NDArray[np.float64], tour_size: int, quantity: int
) -> NDArray[np.int64]:
    """
    Select parent indexes by tournament over fitness values.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        Fitness values used to choose tournament winners.
    rank : NDArray[np.float64]
        Rank values of individuals. Not used.
    tour_size : int
        Number of candidates sampled for each tournament.
    quantity : int
        Number of parent indexes to choose.

    Returns
    -------
    NDArray[np.int64]
        Indexes of selected individuals.

    Notes
    -----
    Each tournament samples ``tour_size`` different candidates without
    replacement and returns the candidate with the highest fitness. The process
    is repeated ``quantity`` times.

    Examples
    --------
    >>> import numpy as np
    >>> from thefittest.utils.selections import tournament_selection
    >>>
    >>> fitness = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    >>> rank = np.array([3.0, 2.0, 4.0, 1.0], dtype=np.float64)
    >>> selected = tournament_selection(fitness, rank, 2, 2)
    >>> selected.shape
    (2,)
    >>> selected.dtype == np.int64
    True
    >>> bool(np.all((0 <= selected) & (selected < fitness.size)))
    True
    """
    to_return = np.empty(quantity, dtype=np.int64)
    for i in range(quantity):
        tournament = random_sample(range_size=len(fitness), quantity=tour_size, replace=False)
        argmax = np.argmax(fitness[tournament])
        to_return[i] = tournament[argmax]
    return to_return


@njit
def rank_crowding_tournament_selection(
    rank: NDArray[np.int64],
    crowding_distance: NDArray[np.float64],
    tour_size: int,
    quantity: int,
) -> NDArray[np.int64]:
    """Select parent indexes by nondomination rank and crowding distance.

    Parameters
    ----------
    rank : NDArray[np.int64]
        Precomputed nondomination ranks for the current population. Lower
        ranks are better; rank 1 is the first front.
    crowding_distance : NDArray[np.float64]
        Precomputed distances, calculated separately within each front and
        aligned with ``rank`` by population index. Larger values are better.
        Positive infinity is allowed; NaN is not supported.
    tour_size : int
        Number of distinct candidates per tournament, between 2 and the
        population size. Use 2 for the NSGA-II binary tournament.
    quantity : int
        Non-negative number of parent indexes to return.

    Returns
    -------
    NDArray[np.int64]
        Selected population indexes. An individual may win several tournaments.

    Notes
    -----
    Candidates are sampled without replacement within each tournament;
    different tournaments are independent. Lower rank wins regardless of
    distance. At equal ranks, greater distance wins. Exact ties choose either
    candidate with equal probability, including for two infinite distances.
    This matches pymoo's feasible-solution rank-and-crowding comparison.
    Constraint violations are not handled. Candidate sampling and random
    generators differ from pymoo, so equal seeds do not imply equal samples.

    Compute ranks and distances once per population update, before calling
    this function. No fitness evaluation, sorting, or distance calculation
    occurs during selection. Comparison work is O(quantity * tour_size).
    The input arrays are not modified.

    Examples
    --------
    >>> rank = np.array([1, 1, 2], dtype=np.int64)
    >>> distances = np.array([np.inf, 0.5, np.inf])
    >>> selected = rank_crowding_tournament_selection(rank, distances, 2, 10)
    >>> selected.shape
    (10,)
    >>> bool(np.all(rank[selected] == 1))
    True
    """
    if rank.ndim != 1 or crowding_distance.ndim != 1:
        raise ValueError("rank and crowding_distance must be one-dimensional arrays.")
    if len(rank) != len(crowding_distance):
        raise ValueError("rank and crowding_distance must have the same length.")
    if tour_size < 2 or tour_size > len(rank):
        raise ValueError("tour_size must be between 2 and the population size.")
    if quantity < 0:
        raise ValueError("quantity must be non-negative.")

    selected = np.empty(quantity, dtype=np.int64)
    for i in range(quantity):
        candidates = random_sample(range_size=len(rank), quantity=tour_size, replace=False)
        winner = candidates[0]
        for j in range(1, tour_size):
            candidate = candidates[j]
            comparison = compare_rank_and_crowding(
                rank[candidate],
                crowding_distance[candidate],
                rank[winner],
                crowding_distance[winner],
            )
            if comparison > 0 or (comparison == 0 and flip_coin(0.5)):
                winner = candidate
        selected[i] = winner
    return selected


# Alias for rank-based crowded tournament selection.
crowded_tournament_selection = rank_crowding_tournament_selection


@njit
def dominance_crowding_tournament_selection(
    fitness: NDArray[np.float64],
    crowding_distance: NDArray[np.float64],
    tour_size: int,
    quantity: int,
) -> NDArray[np.int64]:
    """Select indexes by direct dominance, then crowding distance.

    ``fitness`` is a matrix of shape (N, M), with all objectives maximized.
    ``crowding_distance`` contains N precomputed distances, calculated
    separately within each front. Neither input may contain NaN.

    A dominating candidate wins regardless of distance. If neither candidate
    dominates the other, greater distance wins; exact ties choose either
    candidate with equal probability. Use ``tour_size=2`` for pymoo's
    feasible-solution dominance-and-crowding binary tournament. Constraint
    violations are not handled here. Candidate sampling uses this library's
    random_sample, so matching seeds do not imply matching pymoo samples.

    Candidates are distinct within a tournament; winners may repeat across
    the ``quantity`` independent tournaments. Return an int64 index array.
    For sizes above 2, candidates compete sequentially in sampled order;
    this is an extension, not pymoo's binary tournament. Comparison work is
    O(quantity * tour_size * M). Inputs are not modified.
    """
    if fitness.ndim != 2 or crowding_distance.ndim != 1:
        raise ValueError("fitness must be a matrix and crowding_distance a vector.")
    if fitness.shape[1] == 0:
        raise ValueError("fitness must contain at least one objective.")
    if len(fitness) != len(crowding_distance):
        raise ValueError("fitness and crowding_distance must have the same length.")
    if tour_size < 2 or tour_size > len(fitness):
        raise ValueError("tour_size must be between 2 and the population size.")
    if quantity < 0:
        raise ValueError("quantity must be non-negative.")

    selected = np.empty(quantity, dtype=np.int64)
    for i in range(quantity):
        candidates = random_sample(range_size=len(fitness), quantity=tour_size, replace=False)
        winner = candidates[0]
        for j in range(1, tour_size):
            candidate = candidates[j]
            comparison = compare_dominance_and_crowding(
                fitness[candidate],
                crowding_distance[candidate],
                fitness[winner],
                crowding_distance[winner],
            )
            if comparison > 0 or (comparison == 0 and flip_coin(0.5)):
                winner = candidate
        selected[i] = winner
    return selected

"""Selection operators used by evolutionary algorithms.

The functions in this module choose parent indexes from a population. They
share the same ``(fitness, rank, tour_size, quantity)`` signature so optimizers
can switch between selection strategies through a common interface.
"""

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

"""Utilities for optimization with multiple objectives."""

from __future__ import annotations

from numba import njit

import numpy as np
from numpy.typing import NDArray


@njit(inline="always")
def compare_rank_and_crowding(
    rank_i: int, distance_i: float, rank_j: int, distance_j: float
) -> int:
    """Compare two solutions using precomputed NSGA-II metadata.

    Return 1 if solution i is better, -1 if solution j is better, or 0
    for a tie. Lower nondomination ranks are preferred. For equal ranks,
    larger crowding distances are preferred. Distances may include positive
    infinity, but must not contain NaN. No objectives are evaluated here.
    """
    if rank_i < rank_j:
        return 1
    if rank_i > rank_j:
        return -1
    if distance_i > distance_j:
        return 1
    if distance_i < distance_j:
        return -1
    return 0


@njit(inline="always")
def compare_dominance_and_crowding(
    fitness_i: NDArray[np.float64],
    distance_i: float,
    fitness_j: NDArray[np.float64],
    distance_j: float,
) -> int:
    """Compare by direct dominance, then crowding distance.

    Return 1 if i wins, -1 if j wins, and 0 for a tie. All objectives
    are maximized, as in ``fast_non_dominated_sort``. Objective vectors
    must have the same positive length and contain no NaN. If neither
    solution dominates the other (including equal vectors), prefer the
    greater crowding distance, regardless of nondomination rank.
    """
    i_better = False
    j_better = False
    for m in range(len(fitness_i)):
        if fitness_i[m] > fitness_j[m]:
            i_better = True
        elif fitness_i[m] < fitness_j[m]:
            j_better = True
        if i_better and j_better:
            break
    if i_better and not j_better:
        return 1
    if j_better and not i_better:
        return -1
    if distance_i > distance_j:
        return 1
    if distance_i < distance_j:
        return -1
    return 0


@njit
def fast_non_dominated_sort(fitness: NDArray[np.float64]) -> NDArray[np.int64]:
    """Assign nondomination ranks to rows of a fitness matrix.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        Matrix of shape ``(n_individuals, n_objectives)``. All objectives
        are maximized, following the library's internal fitness convention.
        Negate objectives that should be minimized before calling this function.
        At least one objective is required. NaN values are not supported.

    Returns
    -------
    NDArray[np.int64]
        Rank of each individual in the original row order. Rank 1 denotes
        the first nondominated front; lower ranks are better. Equal fitness
        vectors do not dominate one another. An empty population returns an
        empty rank array.

    Notes
    -----
    Implements the fast nondominated sorting procedure of Deb et al. (2002),
    NSGA-II, comparing each unordered pair only once. A solution dominates
    another if it is no worse in every objective and strictly better in at
    least one. The dominated sets are stored as a boolean adjacency matrix.
    Time complexity is O(M * N**2), and auxiliary space is O(N**2), where
    N is the number of individuals and M is the number of objectives.
    The input is not modified. The first call compiles the Numba kernel.

    References
    ----------
    .. [1] Deb, K. et al. (2002). A fast and elitist multiobjective genetic
       algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation,
       6(2), 182-197. doi:10.1109/4235.996017.

    Examples
    --------
    >>> fitness = np.array([[3., 1.], [1., 3.], [1., 1.], [0., 0.]])
    >>> ranks = fast_non_dominated_sort(fitness)
    >>> ranks
    array([1, 1, 2, 3])
    >>> np.flatnonzero(ranks == 1)
    array([0, 1])
    """
    if fitness.ndim != 2:
        raise ValueError("fitness must be a two-dimensional matrix.")
    n_individuals, n_objectives = fitness.shape
    if n_objectives == 0:
        raise ValueError("fitness must contain at least one objective.")
    for p in range(n_individuals):
        for m in range(n_objectives):
            if np.isnan(fitness[p, m]):
                raise ValueError("fitness must not contain NaN values.")

    dominates = np.zeros((n_individuals, n_individuals), dtype=np.bool_)
    domination_count = np.zeros(n_individuals, dtype=np.int64)
    ranks = np.zeros(n_individuals, dtype=np.int64)

    for p in range(n_individuals):
        for q in range(p + 1, n_individuals):
            p_better = False
            q_better = False
            for m in range(n_objectives):
                if fitness[p, m] > fitness[q, m]:
                    p_better = True
                elif fitness[p, m] < fitness[q, m]:
                    q_better = True
                if p_better and q_better:
                    break

            if p_better and not q_better:
                dominates[p, q] = True
                domination_count[q] += 1
            elif q_better and not p_better:
                dominates[q, p] = True
                domination_count[p] += 1

    # A single queue stores fronts consecutively, retaining original row indexes.
    queue = np.empty(n_individuals, dtype=np.int64)
    end = 0
    for p in range(n_individuals):
        if domination_count[p] == 0:
            ranks[p] = 1
            queue[end] = p
            end += 1

    start = 0
    while start < end:
        p = queue[start]
        start += 1
        for q in range(n_individuals):
            if dominates[p, q]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = ranks[p] + 1
                    queue[end] = q
                    end += 1

    return ranks


@njit
def crowding_distance_assignment(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute NSGA-II crowding distances within a single front.

    Parameters
    ----------
    fitness : NDArray[np.float64]
        Finite objective values of one front, with shape
        ``(n_individuals, n_objectives)``. At least one objective is required.
        Values may be supplied in their original minimization or maximization
        convention. Front membership must be determined by the caller.

    Returns
    -------
    NDArray[np.float64]
        Distances in the original row order. Larger distances favor less
        crowded solutions. Boundary solutions receive infinity. Fronts of
        one or two individuals receive infinity for every member; an empty
        front returns an empty array.

    Notes
    -----
    Each objective requires its own ordering. Stable mergesort preserves
    the previous objective's ordering for ties, matching successive Python
    list.sort calls. Neither the input values nor their row order is modified.
    Constant objectives contribute nothing and do not assign boundaries.
    Consequently, an entirely constant front with more than two members has
    zero distances. This explicitly handles the zero denominator left
    undefined by the original NSGA-II pseudocode.

    Time complexity is O(M * L * log(L)), with O(L) auxiliary space for a
    front of L individuals and M objectives. Compilation occurs on first use.

    Examples
    --------
    >>> front = np.array([[0., 3.], [1., 2.], [2., 1.], [3., 0.]])
    >>> crowding_distance_assignment(front)
    array([       inf, 1.33333333, 1.33333333,        inf])
    """
    if fitness.ndim != 2:
        raise ValueError("fitness must be a two-dimensional matrix.")
    n_individuals, n_objectives = fitness.shape
    if n_objectives == 0:
        raise ValueError("fitness must contain at least one objective.")
    for p in range(n_individuals):
        for m in range(n_objectives):
            if not np.isfinite(fitness[p, m]):
                raise ValueError("crowding distance requires finite objective values.")

    distances = np.zeros(n_individuals, dtype=np.float64)
    if n_individuals <= 2:
        distances[:] = np.inf
        return distances

    order = np.arange(n_individuals, dtype=np.int64)
    for m in range(n_objectives):
        positions = np.argsort(fitness[order, m], kind="mergesort")
        order = order[positions]
        minimum = fitness[order[0], m]
        maximum = fitness[order[-1], m]
        if minimum == maximum:
            continue

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = maximum - minimum
        if np.isfinite(span):
            for i in range(1, n_individuals - 1):
                gap = fitness[order[i + 1], m] - fitness[order[i - 1], m]
                distances[order[i]] += gap / span
        else:
            # Halve before subtraction to avoid overflow for finite extremes.
            span = maximum * 0.5 - minimum * 0.5
            for i in range(1, n_individuals - 1):
                gap = fitness[order[i + 1], m] * 0.5 - fitness[order[i - 1], m] * 0.5
                distances[order[i]] += gap / span

    return distances


def convergence_metric(front: NDArray[np.float64], reference_front: NDArray[np.float64]) -> float:
    """Deb et al. (2002) convergence metric gamma in original objective units.

    Average the distance from each obtained solution to its nearest reference
    point. This is the opposite direction from IGD. The paper uses 500
    reference points uniformly spaced in objective space. Empty obtained
    fronts have infinite distance; the reference must be nonempty.
    """
    if len(reference_front) == 0:
        raise ValueError("reference_front must be nonempty.")
    if len(front) == 0:
        return float("inf")
    distances = np.linalg.norm(front[:, None, :] - reference_front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def diversity_metric(front: NDArray[np.float64], reference_front: NDArray[np.float64]) -> float:
    """Two-objective spread delta, equation (1) of Deb et al. (2002).

    Inputs are minimization fronts in original objective units. Consecutive
    distances use ascending first-objective order, retaining duplicate points.
    Endpoint distances are measured to the reference front's two extremes.
    Thus endpoint estimation uses the supplied reference rather than fitting
    a parallel curve to the obtained front. A front with fewer than two
    solutions receives 1.0. No objective normalization is applied.
    """
    if len(reference_front) < 2:
        raise ValueError("reference_front must contain at least two points.")
    if len(front) < 2:
        return 1.0
    ordered = front[np.argsort(front[:, 0], kind="mergesort")]
    reference = reference_front[np.argsort(reference_front[:, 0], kind="mergesort")]
    gaps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    endpoint_distance = np.linalg.norm(ordered[0] - reference[0]) + np.linalg.norm(
        ordered[-1] - reference[-1]
    )
    mean_gap = np.mean(gaps)
    denominator = endpoint_distance + np.sum(gaps)
    if denominator == 0.0:
        return 0.0
    return float((endpoint_distance + np.sum(np.abs(gaps - mean_gap))) / denominator)

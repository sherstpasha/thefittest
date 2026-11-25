import numpy as np
import pytest

from ..utils.multiobjective import fast_non_dominated_sort
from ..utils.multiobjective import crowding_distance_assignment
from ..utils.multiobjective import compare_rank_and_crowding
from ..utils.multiobjective import compare_dominance_and_crowding


class Individual:
    """Minimization individual with strict Pareto dominance as comparison."""

    def __init__(self, objectives, index):
        self.objectives = objectives
        self.index = index
        self.n = 0
        self.rank = 0

    def __lt__(self, other):
        return np.all(self.objectives <= other.objectives) and np.any(
            self.objectives < other.objectives
        )


def original_object_sort(P):
    """Object-based sorting reference that includes a trailing empty front."""
    F = {1: []}
    S = {}
    for i in range(len(P)):
        p = P[i]
        S[i] = []
        p.n = 0
        for j in range(len(P)):
            q = P[j]
            if p < q:
                S[i].append(j)
            elif q < p:
                p.n += 1
        if p.n == 0:
            p.rank = 1
            F[1].append(i)
    i = 1
    while F[i] != []:
        Q = []
        for pi in F[i]:
            for qi in S[pi]:
                P[qi].n -= 1
                if P[qi].n == 0:
                    P[qi].rank = i + 1
                    Q.append(qi)
        i += 1
        F[i] = Q

    F1 = {}
    for f in F:
        F1[f] = [P[i] for i in F[f]]
    return F1


def reference_ranks(fitness):
    """Independent oracle: repeatedly remove the nondominated remaining rows."""
    ranks = np.zeros(len(fitness), dtype=np.int64)
    remaining = np.arange(len(fitness))
    rank = 1
    while len(remaining):
        values = fitness[remaining]
        dominates = np.all(values[:, None, :] >= values[None, :, :], axis=2) & np.any(
            values[:, None, :] > values[None, :, :], axis=2
        )
        front = ~np.any(dominates, axis=0)
        ranks[remaining[front]] = rank
        remaining = remaining[~front]
        rank += 1
    return ranks


@pytest.mark.parametrize(
    "fitness, expected",
    [
        ([[3, 1], [1, 3], [1, 1], [0, 0]], [1, 1, 2, 3]),
        ([[0, 0], [2, 2], [1, 1], [3, 3]], [4, 2, 3, 1]),
        ([[3, 1], [2, 2], [1, 3]], [1, 1, 1]),
        ([[2, 2], [2, 2], [1, 1], [1, 1]], [1, 1, 2, 2]),
        ([[1, 1], [1, 1], [1, 1]], [1, 1, 1]),
        ([[2, 1], [1, 1], [0, 1]], [1, 2, 3]),
        ([[1, 2, 3]], [1]),
        ([[1], [3], [2], [3]], [3, 1, 2, 1]),
        ([[np.inf, 1], [np.inf, 0], [-np.inf, 0]], [1, 2, 3]),
    ],
)
def test_fast_non_dominated_sort_known_fronts(fitness, expected):
    fitness = np.array(fitness, dtype=np.float64)
    original = fitness.copy()

    ranks = fast_non_dominated_sort(fitness)

    assert np.array_equal(ranks, expected)
    assert ranks.dtype == np.int64
    assert np.array_equal(fitness, original)


def test_fast_non_dominated_sort_empty_population():
    ranks = fast_non_dominated_sort(np.empty((0, 2), dtype=np.float64))
    assert ranks.shape == (0,)
    assert ranks.dtype == np.int64


@pytest.mark.parametrize("n_objectives", [1, 2, 3, 5])
def test_fast_non_dominated_sort_matches_reference(n_objectives):
    rng = np.random.RandomState(42)
    for _ in range(10):
        # Integer-valued floats deliberately exercise ties and duplicate rows.
        fitness = rng.randint(-3, 4, size=(30, n_objectives)).astype(np.float64)
        assert np.array_equal(fast_non_dominated_sort(fitness), reference_ranks(fitness))


@pytest.mark.parametrize("n_objectives", [1, 2, 3, 5])
@pytest.mark.parametrize("discrete", [False, True])
def test_fast_non_dominated_sort_matches_original_objects(n_objectives, discrete):
    for seed in range(3):
        rng = np.random.RandomState(seed)
        for size in (0, 1, 2, 10, 40):
            if discrete:
                objectives = rng.randint(-2, 3, size=(size, n_objectives)).astype(np.float64)
            else:
                objectives = rng.normal(size=(size, n_objectives))
            objectives = objectives[rng.permutation(size)]
            individuals = [Individual(row, index) for index, row in enumerate(objectives)]
            original_fronts = original_object_sort(individuals)

            # Original objects minimize; the library kernel maximizes fitness.
            ranks = fast_non_dominated_sort(-objectives)

            assert np.array_equal(ranks, [p.rank for p in individuals])
            actual_fronts = {
                int(rank): set(np.flatnonzero(ranks == rank)) for rank in np.unique(ranks)
            }
            expected_fronts = {
                rank: {p.index for p in front} for rank, front in original_fronts.items() if front
            }
            assert actual_fronts == expected_fronts


def test_fast_non_dominated_sort_minimization_and_noncontiguous_input():
    objectives = np.array([[1.0, 3.0], [3.0, 1.0], [3.0, 3.0], [4.0, 4.0]])
    fitness = (-objectives)[::-1]
    assert not fitness.flags.c_contiguous
    assert np.array_equal(fast_non_dominated_sort(fitness), [3, 2, 1, 1])


@pytest.mark.parametrize(
    "fitness",
    [np.empty((2, 0)), np.array([[np.nan, 1.0]]), np.ones(3), np.ones((1, 1, 1))],
)
def test_fast_non_dominated_sort_rejects_invalid_objectives(fitness):
    with pytest.raises(ValueError):
        fast_non_dominated_sort(fitness)


def object_crowding_distances(individuals):
    """Object-based crowding reference with zero contribution from constant objectives."""
    for p in individuals:
        p.distance = 0.0
    if len(individuals) <= 2:
        for p in individuals:
            p.distance = np.inf
        return
    for m in range(len(individuals[0].objectives)):
        individuals.sort(key=lambda p: p.objectives[m])
        minimum = individuals[0].objectives[m]
        maximum = individuals[-1].objectives[m]
        if minimum == maximum:
            continue
        individuals[0].distance = individuals[-1].distance = np.inf
        for i in range(1, len(individuals) - 1):
            individuals[i].distance += (
                individuals[i + 1].objectives[m] - individuals[i - 1].objectives[m]
            ) / (maximum - minimum)


@pytest.mark.parametrize(
    "fitness, expected",
    [
        ([[0, 3], [1, 2], [2, 1], [3, 0]], [np.inf, 4 / 3, 4 / 3, np.inf]),
        ([[2, 1], [0, 3], [3, 0], [1, 2]], [4 / 3, np.inf, np.inf, 4 / 3]),
        ([[1, 4]], [np.inf]),
        ([[1, 4], [1, 4]], [np.inf, np.inf]),
        ([[1, 4], [1, 4], [1, 4]], [0, 0, 0]),
        ([[0, 1], [1, 1], [2, 1], [3, 1]], [np.inf, 2 / 3, 2 / 3, np.inf]),
        ([[1, 0], [1, 1], [1, 2], [1, 3]], [np.inf, 2 / 3, 2 / 3, np.inf]),
        ([[-1e308], [0], [1e308]], [np.inf, 1, np.inf]),
    ],
)
def test_crowding_distance_known_values(fitness, expected):
    fitness = np.array(fitness, dtype=np.float64)
    original = fitness.copy()
    distances = crowding_distance_assignment(fitness)
    np.testing.assert_allclose(distances, expected)
    assert distances.dtype == np.float64
    assert np.array_equal(fitness, original)


def test_crowding_distance_empty_front():
    distances = crowding_distance_assignment(np.empty((0, 2)))
    assert distances.shape == (0,)
    assert distances.dtype == np.float64


@pytest.mark.parametrize("n_objectives", [1, 2, 3, 5])
@pytest.mark.parametrize("discrete", [False, True])
def test_crowding_distance_matches_object_reference(n_objectives, discrete):
    for seed in range(3):
        rng = np.random.RandomState(seed)
        for size in (0, 1, 2, 10, 40):
            if discrete:
                objectives = rng.randint(-2, 3, size=(size, n_objectives)).astype(np.float64)
            else:
                objectives = rng.normal(size=(size, n_objectives))
            # Calculate each actual nondominated front independently.
            ranks = fast_non_dominated_sort(-objectives)
            fronts = [np.flatnonzero(ranks == r) for r in np.unique(ranks)]
            for indexes in fronts:
                front = objectives[indexes]
                individuals = [Individual(row, i) for i, row in enumerate(front)]
                for p in individuals:
                    p.distance = 123.0  # A previous calculation must not leak in.
                object_crowding_distances(individuals)
                expected = np.empty(len(front))
                for p in individuals:
                    expected[p.index] = p.distance
                np.testing.assert_allclose(crowding_distance_assignment(front), expected)


def test_crowding_distance_noncontiguous_input_and_repeated_calls():
    front = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])[::-1]
    expected = np.array([np.inf, 4 / 3, 4 / 3, np.inf])
    assert not front.flags.c_contiguous
    np.testing.assert_allclose(crowding_distance_assignment(front), expected)
    np.testing.assert_allclose(crowding_distance_assignment(front), expected)
    np.testing.assert_allclose(crowding_distance_assignment(-front), expected)


@pytest.mark.parametrize(
    "fitness",
    [
        np.empty((2, 0)),
        np.ones(3),
        np.ones((1, 1, 1)),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[-np.inf, 1.0]]),
    ],
)
def test_crowding_distance_rejects_invalid_objectives(fitness):
    with pytest.raises(ValueError):
        crowding_distance_assignment(fitness)


def test_compare_rank_and_crowding_matches_object_operator():
    def original_comparison(i, j):
        if i.rank < j.rank:
            return 1
        if i.rank == j.rank and i.distance > j.distance:
            return 1
        if i.rank == j.rank and i.distance == j.distance:
            return 0
        return -1

    individuals = []
    for rank in (1, 2, 3):
        for distance in (0.0, 0.5, 2.0, np.inf):
            p = Individual(np.empty(0), len(individuals))
            p.rank = rank
            p.distance = distance
            individuals.append(p)
    for i in individuals:
        for j in individuals:
            actual = compare_rank_and_crowding(i.rank, i.distance, j.rank, j.distance)
            assert actual == original_comparison(i, j)
            assert actual == -compare_rank_and_crowding(j.rank, j.distance, i.rank, i.distance)


def test_compare_dominance_and_crowding_matches_feasible_pymoo_rule():
    rng = np.random.RandomState(42)
    for m in (1, 2, 3, 5):
        fitness = rng.randint(-2, 3, size=(20, m)).astype(np.float64)
        distances = rng.choice(np.array([0.0, 0.5, 2.0, np.inf]), size=20)
        for i in range(20):
            for j in range(20):
                if np.all(fitness[i] >= fitness[j]) and np.any(fitness[i] > fitness[j]):
                    expected = 1
                elif np.all(fitness[j] >= fitness[i]) and np.any(fitness[j] > fitness[i]):
                    expected = -1
                elif distances[i] > distances[j]:
                    expected = 1
                elif distances[i] < distances[j]:
                    expected = -1
                else:
                    expected = 0
                assert (
                    compare_dominance_and_crowding(
                        fitness[i], distances[i], fitness[j], distances[j]
                    )
                    == expected
                )

import numpy as np
import pytest

from ..utils import selections
from ..utils.random import numba_seed
from ..utils.random import random_sample
from ..utils.random import flip_coin


def test_selection_dispatchers_return_index_arrays():
    fitness = np.array([0.8, 0.5, 0.9, 0.3], dtype=np.float64)
    rank = np.array([3.0, 2.0, 4.0, 1.0], dtype=np.float64)

    cases = (
        (selections.proportional_selection, 0),
        (selections.rank_selection, 0),
        (selections.tournament_selection, 2),
    )

    for selection, tour_size in cases:
        selected = selection(fitness, rank, tour_size, 3)

        assert selected.shape == (3,)
        assert selected.dtype == np.int64
        assert np.all(0 <= selected)
        assert np.all(selected < fitness.size)


def test_proportional_selection_uses_fitness_weights(monkeypatch):
    fitness = np.array([0.8, 0.5, 0.9], dtype=np.float64)
    rank = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    calls = []

    def fixed_random_weighted_sample(weights, quantity, replace):
        calls.append((weights, quantity, replace))
        return np.array([2, 0], dtype=np.int64)

    monkeypatch.setattr(selections, "random_weighted_sample", fixed_random_weighted_sample)

    selected = selections.proportional_selection.py_func(fitness, rank, 7, 2)

    assert np.array_equal(selected, np.array([2, 0], dtype=np.int64))
    assert calls[0][0] is fitness
    assert calls[0][1:] == (2, True)


def test_rank_selection_uses_rank_weights(monkeypatch):
    fitness = np.array([0.8, 0.5, 0.9], dtype=np.float64)
    rank = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    calls = []

    def fixed_random_weighted_sample(weights, quantity, replace):
        calls.append((weights, quantity, replace))
        return np.array([1, 2], dtype=np.int64)

    monkeypatch.setattr(selections, "random_weighted_sample", fixed_random_weighted_sample)

    selected = selections.rank_selection.py_func(fitness, rank, 7, 2)

    assert np.array_equal(selected, np.array([1, 2], dtype=np.int64))
    assert calls[0][0] is rank
    assert calls[0][1:] == (2, True)


def test_tournament_selection_uses_highest_fitness(monkeypatch):
    fitness = np.array([0.1, 0.9, 0.4], dtype=np.float64)
    rank = np.array([100.0, 1.0, 50.0], dtype=np.float64)
    tournaments = [
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]
    calls = []

    def fixed_random_sample(range_size, quantity, replace):
        calls.append((range_size, quantity, replace))
        return tournaments[len(calls) - 1]

    monkeypatch.setattr(selections, "random_sample", fixed_random_sample)

    selected = selections.tournament_selection.py_func(fitness, rank, 2, 2)

    assert np.array_equal(selected, np.array([2, 1], dtype=np.int64))
    assert calls == [(len(fitness), 2, False), (len(fitness), 2, False)]


def test_crowded_tournament_uses_rank_distance_and_random_ties(monkeypatch):
    rank = np.array([1, 2, 1, 1, 1, 1], dtype=np.int64)
    distance = np.array([0.0, np.inf, 2.0, np.inf, np.inf, 2.0])
    tournaments = [
        np.array([1, 0]),  # Lower rank wins even with smaller distance.
        np.array([0, 2]),  # Equal rank: greater distance wins.
        np.array([2, 0]),
        np.array([3, 4]),  # Equal infinities: the mocked coin picks the second.
        np.array([4, 3]),
        np.array([5, 2]),  # Equal finite values: the mocked coin picks the second.
    ]
    calls = []

    def fixed_random_sample(range_size, quantity, replace):
        calls.append((range_size, quantity, replace))
        return tournaments[len(calls) - 1]

    monkeypatch.setattr(selections, "random_sample", fixed_random_sample)
    monkeypatch.setattr(selections, "flip_coin", lambda probability: True)
    selected = selections.crowded_tournament_selection.py_func(rank, distance, 2, len(tournaments))
    assert np.array_equal(selected, [0, 2, 2, 4, 3, 2])
    assert calls == [(len(rank), 2, False)] * len(tournaments)


@pytest.mark.parametrize("tour_size", [2, 3, 7])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_compiled_crowded_tournament_matches_reference_for_same_samples(tour_size, seed):
    rank = np.array([1, 2, 1, 3, 1, 2, 1], dtype=np.int64)
    distance = np.array([0.0, np.inf, 2.0, np.inf, np.inf, 0.0, np.inf])
    original_rank, original_distance = rank.copy(), distance.copy()
    quantity = 50
    # Compile before seeding, then replay exactly the same candidate samples.
    selections.crowded_tournament_selection(rank, distance, tour_size, 0)
    numba_seed(seed)
    expected = []
    for _ in range(quantity):
        candidates = random_sample(len(rank), tour_size, False)
        assert len(np.unique(candidates)) == tour_size
        winner = candidates[0]
        for candidate in candidates[1:]:
            key = (rank[candidate], -distance[candidate])
            best_key = (rank[winner], -distance[winner])
            if key < best_key or (key == best_key and flip_coin(0.5)):
                winner = candidate
        expected.append(winner)

    numba_seed(seed)
    selected = selections.crowded_tournament_selection(rank, distance, tour_size, quantity)
    assert np.array_equal(selected, expected)
    assert selected.dtype == np.int64
    assert np.array_equal(rank, original_rank)
    assert np.array_equal(distance, original_distance)


def test_crowded_tournament_empty_output():
    selected = selections.crowded_tournament_selection(
        np.array([1, 1]), np.array([np.inf, np.inf]), 2, 0
    )
    assert selected.shape == (0,)
    assert selected.dtype == np.int64


@pytest.mark.parametrize(
    "rank, distance, tour_size, quantity",
    [
        (np.array([1, 2]), np.ones(1), 2, 1),
        (np.array([1, 2]), np.ones(2), 1, 1),
        (np.array([1, 2]), np.ones(2), 3, 1),
        (np.array([1, 2]), np.ones(2), 2, -1),
        (np.array([1]), np.ones(1), 2, 1),
        (np.empty(0, dtype=np.int64), np.empty(0), 2, 1),
        (np.ones((2, 1), dtype=np.int64), np.ones(2), 2, 1),
        (np.array([1, 2]), np.ones((2, 1)), 2, 1),
    ],
)
def test_crowded_tournament_rejects_invalid_inputs(rank, distance, tour_size, quantity):
    with pytest.raises(ValueError):
        selections.crowded_tournament_selection(rank, distance, tour_size, quantity)


def test_two_modes_can_select_different_winners(monkeypatch):
    # Candidates 1 and 2 are incomparable but belong to different fronts.
    fitness = np.array([[3.0, 3.0], [2.0, 2.0], [4.0, 1.0]])
    ranks = np.array([1, 2, 1])
    distance = np.array([0.0, 2.0, 1.0])
    monkeypatch.setattr(
        selections, "random_sample", lambda range_size, quantity, replace: np.array([1, 2])
    )
    assert selections.rank_crowding_tournament_selection.py_func(ranks, distance, 2, 1)[0] == 2
    assert (
        selections.dominance_crowding_tournament_selection.py_func(fitness, distance, 2, 1)[0] == 1
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_dominance_tournament_matches_reference_for_same_samples(seed):
    fitness = np.array([[3.0, 3.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0]])
    distance = np.array([np.inf, np.inf, 0.5, np.inf])
    original = fitness.copy()
    selections.dominance_crowding_tournament_selection(fitness, distance, 2, 0)
    numba_seed(seed)
    expected = []
    for _ in range(100):
        a, b = random_sample(len(fitness), 2, False)
        if np.all(fitness[a] >= fitness[b]) and np.any(fitness[a] > fitness[b]):
            expected.append(a)
        elif np.all(fitness[b] >= fitness[a]) and np.any(fitness[b] > fitness[a]):
            expected.append(b)
        elif distance[a] > distance[b]:
            expected.append(a)
        elif distance[b] > distance[a]:
            expected.append(b)
        else:
            expected.append(b if flip_coin(0.5) else a)
    numba_seed(seed)
    actual = selections.dominance_crowding_tournament_selection(fitness, distance, 2, 100)
    assert np.array_equal(actual, expected)
    assert np.array_equal(fitness, original)


@pytest.mark.parametrize("mode", ["rank", "dominance"])
def test_both_modes_randomize_exact_ties(mode):
    operator = (
        selections.rank_crowding_tournament_selection
        if mode == "rank"
        else selections.dominance_crowding_tournament_selection
    )
    metadata = np.ones(2, dtype=np.int64) if mode == "rank" else np.ones((2, 2))
    distances = np.array([np.inf, np.inf])
    numba_seed(42)
    selected = operator(metadata, distances, 2, 100)
    assert set(selected) == {0, 1}
    numba_seed(42)
    assert np.array_equal(operator(metadata, distances, 2, 100), selected)


@pytest.mark.parametrize(
    "fitness, distance, tour_size, quantity",
    [
        (np.ones((2, 2)), np.ones(1), 2, 1),
        (np.ones((2, 0)), np.ones(2), 2, 1),
        (np.ones((2, 2)), np.ones(2), 1, 1),
        (np.ones((2, 2)), np.ones(2), 3, 1),
        (np.ones((2, 2)), np.ones(2), 2, -1),
        (np.ones(2), np.ones(2), 2, 1),
        (np.ones((2, 2)), np.ones((2, 1)), 2, 1),
    ],
)
def test_dominance_tournament_rejects_invalid_inputs(fitness, distance, tour_size, quantity):
    with pytest.raises(ValueError):
        selections.dominance_crowding_tournament_selection(fitness, distance, tour_size, quantity)
